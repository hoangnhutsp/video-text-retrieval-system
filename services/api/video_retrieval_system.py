"""
Video-Text Retrieval System using CLIP embeddings
Processes keyframes from V3C dataset and creates searchable index
"""

import os
import json
import pandas as pd
import numpy as np
import torch
import cv2
from PIL import Image
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import logging
from dataclasses import dataclass

# For embeddings
import clip
import faiss  # For efficient similarity search
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class KeyFrameInfo:
    """Information about a keyframe"""
    video_id: str
    frame_number: int
    image_path: str
    start_time: float
    end_time: float
    description: str
    embedding: Optional[np.ndarray] = None

class VideoTextRetrievalSystem:
    """
    Video-Text Retrieval System using CLIP embeddings
    """
    
    def __init__(self, dataset_path: str, model_name: str = "ViT-B/32"):
        """
        Initialize the retrieval system
        
        Args:
            dataset_path: Path to the V3C dataset
            model_name: CLIP model variant to use
        """
        self.dataset_path = Path(dataset_path)
        self.model_name = model_name
        
        # Load CLIP model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading CLIP model {model_name} on {self.device}")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        
        # Storage for keyframe information and embeddings
        self.keyframes: List[KeyFrameInfo] = []
        self.image_embeddings: Optional[np.ndarray] = None
        self.text_embeddings: Optional[np.ndarray] = None
        self.faiss_index: Optional[faiss.IndexFlatIP] = None
        
    def load_video_metadata(self, video_dir: Path) -> Dict:
        """Load metadata for a single video"""
        video_id = video_dir.name
        
        # Load description
        desc_file = video_dir / f"{video_id}.description"
        description = ""
        if desc_file.exists():
            with open(desc_file, 'r', encoding='utf-8') as f:
                description = f.read().strip()
        
        # Load timing information
        tsv_file = video_dir / f"{video_id}.tsv"
        timing_df = None
        if tsv_file.exists():
            timing_df = pd.read_csv(tsv_file, sep='\t')
        
        # Load JSON metadata
        json_file = video_dir / f"{video_id}.info.json"
        metadata = {}
        if json_file.exists():
            with open(json_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        
        return {
            'video_id': video_id,
            'description': description,
            'timing_df': timing_df,
            'metadata': metadata
        }
    
    def extract_keyframes_from_video(self, video_dir: Path) -> List[KeyFrameInfo]:
        """Extract keyframe information from a single video directory"""
        keyframes = []
        
        # Load video metadata
        video_data = self.load_video_metadata(video_dir)
        video_id = video_data['video_id']
        description = video_data['description']
        timing_df = video_data['timing_df']
        
        # Get keyframes directory
        keyframes_dir = video_dir / "keyframes"
        if not keyframes_dir.exists():
            logger.warning(f"No keyframes directory found for video {video_id} at {keyframes_dir}")
            return keyframes
        
        # Get all PNG files in keyframes directory
        keyframe_files = list(keyframes_dir.glob("*.png"))
        logger.debug(f"Found {len(keyframe_files)} PNG files in {keyframes_dir}")
        
        if not keyframe_files:
            logger.warning(f"No PNG keyframe files found in {keyframes_dir}")
            return keyframes
        
        # Process each keyframe image
        for img_file in keyframe_files:
            try:
                # Extract frame number from filename (e.g., shot00001_42_RKF.png -> 42)
                parts = img_file.stem.split('_')
                if len(parts) >= 2:
                    frame_num = int(parts[1])
                else:
                    logger.warning(f"Could not extract frame number from {img_file.name}")
                    continue
                
                # Find corresponding timing information
                start_time, end_time = 0.0, 0.0
                if timing_df is not None and len(timing_df) > 0:
                    # Find the row where this frame number falls
                    matching_row = timing_df[
                        (timing_df['startframe'] <= frame_num) & 
                        (timing_df['endframe'] >= frame_num)
                    ]
                    if not matching_row.empty:
                        start_time = float(matching_row.iloc[0]['starttime'])
                        end_time = float(matching_row.iloc[0]['endtime'])
                
                # Create KeyFrameInfo
                keyframe_info = KeyFrameInfo(
                    video_id=video_id,
                    frame_number=frame_num,
                    image_path=str(img_file),
                    start_time=start_time,
                    end_time=end_time,
                    description=description
                )
                keyframes.append(keyframe_info)
                
            except Exception as e:
                logger.error(f"Error processing keyframe {img_file}: {e}")
        
        logger.debug(f"Extracted {len(keyframes)} keyframes from video {video_id}")
        return keyframes
    
    def load_dataset(self, dataset_versions: List[str] = ['v3c1', 'v3c2', 'v3c3']) -> None:
        """
        Load all keyframes from the dataset
        
        Args:
            dataset_versions: List of dataset versions to process
        """
        logger.info("Loading dataset...")
        self.keyframes = []
        
        for version in dataset_versions:
            version_path = self.dataset_path / version
            if not version_path.exists():
                logger.warning(f"Dataset version {version} not found at {version_path}")
                continue
            
            logger.info(f"Processing {version}...")
            
            # Get all video directories
            video_dirs = [d for d in version_path.iterdir() if d.is_dir()]
            logger.info(f"Found {len(video_dirs)} video directories in {version}")
            
            # Process each video directory
            for video_dir in tqdm(video_dirs, desc=f"Processing {version}"):
                logger.debug(f"Processing video directory: {video_dir}")
                video_keyframes = self.extract_keyframes_from_video(video_dir)
                self.keyframes.extend(video_keyframes)
                logger.debug(f"Video {video_dir.name}: {len(video_keyframes)} keyframes")
        
        logger.info(f"Loaded {len(self.keyframes)} keyframes from dataset")
    
    def encode_images(self, batch_size: int = 32) -> np.ndarray:
        """
        Create CLIP embeddings for all keyframe images
        
        Args:
            batch_size: Number of images to process in each batch
        
        Returns:
            Array of image embeddings
        """
        logger.info("Encoding keyframe images...")
        
        if not self.keyframes:
            logger.warning("No keyframes found to encode!")
            self.image_embeddings = np.array([]).reshape(0, 512)  # Empty array with correct shape
            return self.image_embeddings
        
        embeddings = []
        successful_encodings = 0
        failed_encodings = 0
        
        # Process images in batches
        for i in tqdm(range(0, len(self.keyframes), batch_size)):
            batch_keyframes = self.keyframes[i:i + batch_size]
            batch_images = []
            
            # Load and preprocess batch of images
            for keyframe in batch_keyframes:
                try:
                    image = Image.open(keyframe.image_path).convert('RGB')
                    image = self.preprocess(image)
                    batch_images.append(image)
                    successful_encodings += 1
                except Exception as e:
                    logger.error(f"Error loading image {keyframe.image_path}: {e}")
                    # Add zeros for failed images
                    batch_images.append(torch.zeros(3, 224, 224))
                    failed_encodings += 1
            
            # Create batch tensor and encode
            if batch_images:
                batch_tensor = torch.stack(batch_images).to(self.device)
                
                with torch.no_grad():
                    batch_embeddings = self.model.encode_image(batch_tensor)
                    batch_embeddings = batch_embeddings.cpu().numpy()
                    embeddings.append(batch_embeddings)
        
        # Concatenate all embeddings
        if embeddings:
            self.image_embeddings = np.vstack(embeddings)
            # Normalize embeddings for cosine similarity
            self.image_embeddings = self.image_embeddings / np.linalg.norm(
                self.image_embeddings, axis=1, keepdims=True
            )
            logger.info(f"Created embeddings with shape: {self.image_embeddings.shape}")
            logger.info(f"Successfully encoded: {successful_encodings} images")
            if failed_encodings > 0:
                logger.warning(f"Failed to encode: {failed_encodings} images")
        else:
            logger.warning("No embeddings created - all images failed to load!")
            self.image_embeddings = np.array([]).reshape(0, 512)  # Empty array with correct shape
            
        return self.image_embeddings
    
    def encode_text_descriptions(self) -> np.ndarray:
        """
        Create CLIP embeddings for video descriptions
        
        Returns:
            Array of text embeddings
        """
        logger.info("Encoding text descriptions...")
        
        # Get unique descriptions
        unique_descriptions = list(set([kf.description for kf in self.keyframes if kf.description]))
        
        if not unique_descriptions:
            logger.warning("No descriptions found!")
            return np.array([])
        
        # Encode descriptions
        descriptions_tokens = clip.tokenize(unique_descriptions).to(self.device)
        
        with torch.no_grad():
            text_embeddings = self.model.encode_text(descriptions_tokens)
            text_embeddings = text_embeddings.cpu().numpy()
        
        # Normalize embeddings
        text_embeddings = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)
        
        # Map back to keyframes (each keyframe gets its description's embedding)
        description_to_embedding = dict(zip(unique_descriptions, text_embeddings))
        
        self.text_embeddings = np.array([
            description_to_embedding.get(kf.description, np.zeros(text_embeddings.shape[1]))
            for kf in self.keyframes
        ])
        
        logger.info(f"Created text embeddings with shape: {self.text_embeddings.shape}")
        return self.text_embeddings
    
    def build_faiss_index(self) -> None:
        """Build FAISS index for efficient similarity search"""
        if self.image_embeddings is None:
            raise ValueError("Image embeddings not created yet. Call encode_images() first.")
        
        if len(self.image_embeddings) == 0:
            logger.warning("No embeddings to index - skipping FAISS index creation")
            self.faiss_index = None
            return
        
        logger.info("Building FAISS index...")
        
        # Create FAISS index (Inner Product for cosine similarity with normalized vectors)
        embedding_dim = self.image_embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(embedding_dim)
        
        # Add embeddings to index
        self.faiss_index.add(self.image_embeddings.astype('float32'))
        
        logger.info(f"FAISS index built with {self.faiss_index.ntotal} vectors")
    
    def search_by_text(self, query: str, top_k: int = 10) -> List[Tuple[KeyFrameInfo, float]]:
        """
        Search for keyframes using text query
        
        Args:
            query: Text query to search for
            top_k: Number of results to return
        
        Returns:
            List of (KeyFrameInfo, score) tuples sorted by relevance
        """
        if self.faiss_index is None:
            raise ValueError("FAISS index not built. Call build_faiss_index() first.")
        
        # Encode query text
        query_tokens = clip.tokenize([query]).to(self.device)
        
        with torch.no_grad():
            query_embedding = self.model.encode_text(query_tokens)
            query_embedding = query_embedding.cpu().numpy()
            # Normalize for cosine similarity
            query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        # Search in FAISS index
        scores, indices = self.faiss_index.search(query_embedding.astype('float32'), top_k)
        
        # Prepare results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.keyframes):
                results.append((self.keyframes[idx], float(score)))
        
        return results
    
    def save_index(self, save_path: str) -> None:
        """Save the complete model and index to disk"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving index to {save_path}")
        
        # Save FAISS index
        if self.faiss_index is not None:
            faiss.write_index(self.faiss_index, str(save_path / "faiss_index.bin"))
        
        # Save embeddings and keyframe info
        data_to_save = {
            'keyframes': self.keyframes,
            'image_embeddings': self.image_embeddings,
            'text_embeddings': self.text_embeddings,
            'model_name': self.model_name,
            'created_at': datetime.now().isoformat()
        }
        
        with open(save_path / "index_data.pkl", 'wb') as f:
            pickle.dump(data_to_save, f)
        
        logger.info("Index saved successfully")
    
    def load_index(self, load_path: str) -> None:
        """Load a previously saved index"""
        load_path = Path(load_path)
        
        logger.info(f"Loading index from {load_path}")
        
        # Load FAISS index
        faiss_path = load_path / "faiss_index.bin"
        if faiss_path.exists():
            self.faiss_index = faiss.read_index(str(faiss_path))
        
        # Load embeddings and keyframe info
        data_path = load_path / "index_data.pkl"
        if data_path.exists():
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
                self.keyframes = data['keyframes']
                self.image_embeddings = data['image_embeddings']
                self.text_embeddings = data['text_embeddings']
        
        logger.info("Index loaded successfully")

def main():
    """Example usage of the Video-Text Retrieval System"""
    
    # Initialize system
    dataset_path = "/home/tranghoangnhut/Documents/paper/dataset"
    retrieval_system = VideoTextRetrievalSystem(dataset_path)
    
    # Load dataset and create embeddings
    retrieval_system.load_dataset(['v3c1'])  # Start with v3c1, add others as needed
    retrieval_system.encode_images()
    retrieval_system.encode_text_descriptions()
    retrieval_system.build_faiss_index()
    
    # Save the index
    retrieval_system.save_index("./video_retrieval_index")
    
    # Example search
    query = "people riding bikes in Paris"
    results = retrieval_system.search_by_text(query, top_k=5)
    
    print(f"\nSearch results for: '{query}'")
    print("-" * 50)
    for i, (keyframe, score) in enumerate(results):
        print(f"{i+1}. Video: {keyframe.video_id}, Frame: {keyframe.frame_number}")
        print(f"   Score: {score:.4f}")
        print(f"   Time: {keyframe.start_time:.2f}s - {keyframe.end_time:.2f}s")
        print(f"   Image: {keyframe.image_path}")
        print(f"   Description: {keyframe.description[:100]}...")
        print()

if __name__ == "__main__":
    main()