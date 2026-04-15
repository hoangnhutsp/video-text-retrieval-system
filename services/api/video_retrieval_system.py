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

# Try to import OCR libraries (optional)
try:
    import easyocr
    OCR_AVAILABLE = True
    OCR_TYPE = "easyocr"
except ImportError:
    try:
        import pytesseract
        OCR_AVAILABLE = True
        OCR_TYPE = "pytesseract"
    except ImportError:
        OCR_AVAILABLE = False
        OCR_TYPE = None
        print("OCR not available. Install with: pip install easyocr or pip install pytesseract")

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
    description: str  # Original video description
    source: str  # Dataset source (v3c1, v3c2, v3c3)
    extract_text: Optional[str] = None  # OCR extracted text from image
    clip_description: Optional[str] = None  # CLIP-generated image description
    embedding: Optional[np.ndarray] = None

class VideoTextRetrievalSystem:
    """
    Video-Text Retrieval System using CLIP embeddings
    """
    
    def __init__(self, dataset_path: str, model_name: str = "ViT-B/32", enable_ocr: bool = True, enable_clip_description: bool = True):
        """
        Initialize the retrieval system
        
        Args:
            dataset_path: Path to the V3C dataset
            model_name: CLIP model variant to use
            enable_ocr: Whether to enable OCR text extraction from images
            enable_clip_description: Whether to enable CLIP-based image description
        """
        self.dataset_path = Path(dataset_path)
        self.model_name = model_name
        self.enable_ocr = enable_ocr and OCR_AVAILABLE
        self.enable_clip_description = enable_clip_description
        
        # Initialize OCR reader if available
        self.ocr_reader = None
        if self.enable_ocr:
            self._initialize_ocr()
        
        # Load CLIP model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading CLIP model {model_name} on {self.device}")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        
        # Prepare CLIP description templates if enabled
        if self.enable_clip_description:
            self.description_templates = self._prepare_clip_description_templates()
        
        # Storage for keyframe information and embeddings
        self.keyframes: List[KeyFrameInfo] = []
        self.image_embeddings: Optional[np.ndarray] = None
        self.text_embeddings: Optional[np.ndarray] = None
        self.faiss_index: Optional[faiss.IndexFlatIP] = None
        
    def _initialize_ocr(self):
        """Initialize OCR reader based on available libraries"""
        try:
            if OCR_TYPE == "easyocr":
                logger.info("Initializing EasyOCR...")
                self.ocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
                logger.info("EasyOCR initialized successfully")
            elif OCR_TYPE == "pytesseract":
                logger.info("Using Pytesseract for OCR...")
                # For pytesseract, we don't need to initialize a reader object
                self.ocr_reader = "pytesseract"
                logger.info("Pytesseract configured successfully")
        except Exception as e:
            logger.error(f"Error initializing OCR: {e}")
            self.enable_ocr = False
            self.ocr_reader = None
    
    def extract_text_from_image(self, image_path: str) -> Optional[str]:
        """
        Extract text from image using OCR
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Extracted text or None if extraction fails
        """
        if not self.enable_ocr or not self.ocr_reader:
            return None
        
        try:
            if OCR_TYPE == "easyocr":
                # Use EasyOCR
                results = self.ocr_reader.readtext(image_path)
                # Combine all detected text
                extracted_text = " ".join([result[1] for result in results if result[2] > 0.5])  # confidence > 0.5
                return extracted_text.strip() if extracted_text else None
            
            elif OCR_TYPE == "pytesseract":
                # Use Pytesseract
                image = Image.open(image_path)
                extracted_text = pytesseract.image_to_string(image).strip()
                return extracted_text if extracted_text else None
                
        except Exception as e:
            logger.debug(f"OCR extraction failed for {image_path}: {e}")
            return None
        
        return None
        
    def _prepare_clip_description_templates(self) -> List[str]:
        """Prepare descriptive templates for CLIP-based image description"""
        return [
            # People and activities
            "people walking", "people running", "people sitting", "people standing",
            "people talking", "people working", "people playing", "people dancing",
            "group of people", "crowd of people", "person alone", "children playing",
            "people eating", "people reading", "people using computers", "people on phones",
            
            # Transportation
            "cars driving", "vehicles moving", "traffic jam", "parking lot",
            "bicycles", "motorcycles", "buses", "trains", "airplanes",
            "boats", "ships", "public transportation", "road scene",
            
            # Environment and locations
            "outdoor scene", "indoor scene", "urban environment", "natural landscape",
            "city street", "rural area", "park", "beach", "mountains", "forest",
            "buildings", "architecture", "modern buildings", "old buildings",
            "shopping mall", "restaurant", "office", "home interior",
            
            # Nature and weather
            "trees and plants", "flowers", "grass and lawn", "water and ocean",
            "rivers and lakes", "sky and clouds", "sunny day", "cloudy day",
            "rainy day", "snowy day", "night scene", "sunset", "sunrise",
            
            # Objects and items
            "food and cooking", "technology and computers", "sports equipment",
            "furniture", "clothing and fashion", "books and documents",
            "signs and text", "art and paintings", "musical instruments",
            
            # Activities and events
            "sports activity", "exercise and fitness", "shopping", "dining",
            "meeting and conference", "celebration and party", "performance",
            "construction work", "cleaning", "maintenance",
            
            # Visual characteristics
            "colorful scene", "black and white", "bright lighting", "dark lighting",
            "close-up view", "wide view", "blurry image", "clear image",
            "indoor lighting", "outdoor lighting", "dramatic lighting"
        ]
    
    def generate_clip_description(self, image_path: str, top_k: int = 3, confidence_threshold: float = 0.2) -> Optional[str]:
        """
        Generate description for image using CLIP similarity with predefined templates
        
        Args:
            image_path: Path to the image file
            top_k: Number of top matching descriptions to combine
            confidence_threshold: Minimum confidence score to include a description
            
        Returns:
            Generated description or None if generation fails
        """
        if not self.enable_clip_description or not hasattr(self, 'description_templates'):
            return None
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image = self.preprocess(image).unsqueeze(0).to(self.device)
            
            # Tokenize description templates
            text_tokens = clip.tokenize(self.description_templates).to(self.device)
            
            # Get embeddings
            with torch.no_grad():
                image_features = self.model.encode_image(image)
                text_features = self.model.encode_text(text_tokens)
                
                # Normalize features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # Calculate similarities
                similarities = (image_features @ text_features.T).cpu().numpy()[0]
            
            # Get top matching templates
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            top_descriptions = []
            
            for idx in top_indices:
                score = similarities[idx]
                if score >= confidence_threshold:
                    top_descriptions.append(self.description_templates[idx])
            
            # Create combined description
            if top_descriptions:
                return ", ".join(top_descriptions)
            else:
                # Return the best match even if below threshold
                best_idx = np.argmax(similarities)
                return self.description_templates[best_idx]
                
        except Exception as e:
            logger.debug(f"CLIP description generation failed for {image_path}: {e}")
            return None
        
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
    
    def extract_keyframes_from_video(self, video_dir: Path, source: str = "unknown") -> List[KeyFrameInfo]:
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
                
                # Extract text from image using OCR
                extracted_text = None
                if self.enable_ocr:
                    extracted_text = self.extract_text_from_image(str(img_file))
                
                # Generate CLIP-based description
                clip_description = None
                if self.enable_clip_description:
                    clip_description = self.generate_clip_description(str(img_file))
                
                # Create KeyFrameInfo
                keyframe_info = KeyFrameInfo(
                    video_id=video_id,
                    frame_number=frame_num,
                    image_path=str(img_file),
                    start_time=start_time,
                    end_time=end_time,
                    description=description,
                    source=source,
                    extract_text=extracted_text,
                    clip_description=clip_description
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
                video_keyframes = self.extract_keyframes_from_video(video_dir, source=version)
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

    def get_dataset_statistics(self) -> Dict[str, int]:
        """
        Get statistics about the loaded dataset
        
        Returns:
            Dictionary with counts per dataset source
        """
        if not self.keyframes:
            return {}
        
        source_counts = {}
        for keyframe in self.keyframes:
            source = keyframe.source
            source_counts[source] = source_counts.get(source, 0) + 1
        
        return source_counts
    
    def get_ocr_statistics(self) -> Dict[str, int]:
        """
        Get statistics about OCR text extraction
        
        Returns:
            Dictionary with OCR extraction statistics
        """
        if not self.keyframes:
            return {}
        
        total_keyframes = len(self.keyframes)
        keyframes_with_text = len([kf for kf in self.keyframes if kf.extract_text and kf.extract_text.strip()])
        keyframes_without_text = total_keyframes - keyframes_with_text
        
        return {
            'total_keyframes': total_keyframes,
            'with_extracted_text': keyframes_with_text,
            'without_extracted_text': keyframes_without_text,
            'extraction_rate': (keyframes_with_text / total_keyframes * 100) if total_keyframes > 0 else 0
        }
    
    def get_clip_description_statistics(self) -> Dict[str, int]:
        """
        Get statistics about CLIP description generation
        
        Returns:
            Dictionary with CLIP description statistics
        """
        if not self.keyframes:
            return {}
        
        total_keyframes = len(self.keyframes)
        keyframes_with_clip_desc = len([kf for kf in self.keyframes if kf.clip_description and kf.clip_description.strip()])
        keyframes_without_clip_desc = total_keyframes - keyframes_with_clip_desc
        
        return {
            'total_keyframes': total_keyframes,
            'with_clip_description': keyframes_with_clip_desc,
            'without_clip_description': keyframes_without_clip_desc,
            'generation_rate': (keyframes_with_clip_desc / total_keyframes * 100) if total_keyframes > 0 else 0
        }

def main():
    """Example usage of the Video-Text Retrieval System"""
    
    # Initialize system with OCR and CLIP descriptions enabled
    dataset_path = "/home/tranghoangnhut/Documents/paper/dataset"
    retrieval_system = VideoTextRetrievalSystem(
        dataset_path, 
        enable_ocr=True, 
        enable_clip_description=True
    )
    
    # Load dataset and create embeddings
    retrieval_system.load_dataset(['v3c1'])  # Start with v3c1, add others as needed
    
    # Show dataset statistics
    stats = retrieval_system.get_dataset_statistics()
    print("Dataset Statistics:")
    for source, count in stats.items():
        print(f"  {source}: {count} keyframes")
    print()
    
    # Show OCR statistics
    ocr_stats = retrieval_system.get_ocr_statistics()
    print("OCR Text Extraction Statistics:")
    print(f"  Total keyframes: {ocr_stats['total_keyframes']}")
    print(f"  With extracted text: {ocr_stats['with_extracted_text']}")
    print(f"  Without text: {ocr_stats['without_extracted_text']}")
    print(f"  Text extraction rate: {ocr_stats['extraction_rate']:.2f}%")
    print()
    
    # Show CLIP description statistics
    clip_stats = retrieval_system.get_clip_description_statistics()
    print("CLIP Description Generation Statistics:")
    print(f"  Total keyframes: {clip_stats['total_keyframes']}")
    print(f"  With CLIP descriptions: {clip_stats['with_clip_description']}")
    print(f"  Without CLIP descriptions: {clip_stats['without_clip_description']}")
    print(f"  Description generation rate: {clip_stats['generation_rate']:.2f}%")
    print()
    
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
        print(f"   Source: {keyframe.source}")
        print(f"   Score: {score:.4f}")
        print(f"   Time: {keyframe.start_time:.2f}s - {keyframe.end_time:.2f}s")
        print(f"   Image: {keyframe.image_path}")
        print(f"   Original Description: {keyframe.description[:100]}...")
        if keyframe.extract_text:
            print(f"   OCR Extracted Text: {keyframe.extract_text[:100]}...")
        if keyframe.clip_description:
            print(f"   CLIP Description: {keyframe.clip_description}")
        print()

if __name__ == "__main__":
    main()