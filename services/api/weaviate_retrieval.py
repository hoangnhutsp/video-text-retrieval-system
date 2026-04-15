"""
Weaviate integration for Video-Text Retrieval System
Replaces FAISS with cloud-native vector database
"""

import weaviate
import numpy as np
from typing import List, Dict, Tuple, Optional
import json
import logging
from video_retrieval_system import KeyFrameInfo, VideoTextRetrievalSystem
import clip
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)

class WeaviateVideoRetrieval:
    """
    Video-Text Retrieval using Weaviate vector database
    """
    
    def __init__(self, weaviate_url: str = "http://localhost:8082", dataset_path: str = None):
        self.weaviate_url = weaviate_url
        self.dataset_path = dataset_path
        
        # Initialize Weaviate client (v4 syntax)
        import weaviate.classes as wvc
        
        # Parse URL to get host and port
        url_parts = weaviate_url.replace('http://', '').replace('https://', '')
        if ':' in url_parts:
            host, port = url_parts.split(':')
            port = int(port)
        else:
            host = url_parts
            port = 8080
        
        # Use connect_to_local for simple HTTP connection
        self.client = weaviate.connect_to_local(
            host=host,
            port=port
        )
        
        # Load CLIP model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading CLIP model on {self.device}")
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        
        # Schema name
        self.class_name = "VideoKeyframe"
        
    def create_schema(self):
        """Create Weaviate schema for video keyframes"""
        
        # Delete existing collection if it exists
        try:
            self.client.collections.delete(self.class_name)
            logger.info("Deleted existing collection")
        except Exception:
            pass
        
        # Import necessary classes for v4
        import weaviate.classes as wvc
        
        # Create collection with properties
        collection = self.client.collections.create(
            name=self.class_name,
            description="Video keyframe with image and text embeddings",
            vectorizer_config=wvc.config.Configure.Vectorizer.none(),  # We provide our own vectors
            properties=[
                wvc.config.Property(
                    name="videoId",
                    data_type=wvc.config.DataType.TEXT,
                    description="ID of the source video"
                ),
                wvc.config.Property(
                    name="frameNumber", 
                    data_type=wvc.config.DataType.INT,
                    description="Frame number within the video"
                ),
                wvc.config.Property(
                    name="imagePath",
                    data_type=wvc.config.DataType.TEXT,
                    description="Path to the keyframe image file"
                ),
                wvc.config.Property(
                    name="startTime",
                    data_type=wvc.config.DataType.NUMBER,
                    description="Start time of the keyframe in seconds"
                ),
                wvc.config.Property(
                    name="endTime", 
                    data_type=wvc.config.DataType.NUMBER,
                    description="End time of the keyframe in seconds"
                ),
                wvc.config.Property(
                    name="description",
                    data_type=wvc.config.DataType.TEXT,
                    description="Text description of the video content"
                ),
                wvc.config.Property(
                    name="source",
                    data_type=wvc.config.DataType.TEXT,
                    description="Dataset source (v3c1, v3c2, v3c3)"
                ),
                wvc.config.Property(
                    name="extractText",
                    data_type=wvc.config.DataType.TEXT,
                    description="OCR extracted text from image"
                ),
                wvc.config.Property(
                    name="clipDescription",
                    data_type=wvc.config.DataType.TEXT,
                    description="CLIP-generated image description"
                )
            ]
        )
        logger.info(f"Created Weaviate collection for {self.class_name}")
    
    def encode_image(self, image_path: str) -> np.ndarray:
        """Encode a single image using CLIP"""
        try:
            from PIL import Image
            image = Image.open(image_path).convert('RGB')
            image = self.preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                embedding = self.model.encode_image(image)
                embedding = embedding.cpu().numpy().flatten()
                # Normalize for cosine similarity
                embedding = embedding / np.linalg.norm(embedding)
                return embedding
                
        except Exception as e:
            logger.error(f"Error encoding image {image_path}: {e}")
            return np.zeros(512)  # Return zero vector on error
    
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text using CLIP"""
        try:
            tokens = clip.tokenize([text]).to(self.device)
            with torch.no_grad():
                embedding = self.model.encode_text(tokens)
                embedding = embedding.cpu().numpy().flatten()
                # Normalize for cosine similarity
                embedding = embedding / np.linalg.norm(embedding)
                return embedding
        except Exception as e:
            logger.error(f"Error encoding text: {e}")
            return np.zeros(512)
    
    def ingest_keyframes(self, keyframes: List[KeyFrameInfo], batch_size: int = 32):
        """Ingest keyframes into Weaviate"""
        logger.info(f"Ingesting {len(keyframes)} keyframes into Weaviate...")
        
        # Get collection
        collection = self.client.collections.get(self.class_name)
        
        # Prepare batch data
        batch_data = []
        for i, keyframe in enumerate(tqdm(keyframes, desc="Encoding and preparing keyframes")):
            try:
                # Encode image
                image_embedding = self.encode_image(keyframe.image_path)
                
                # Prepare data object
                data_object = {
                    "videoId": keyframe.video_id,
                    "frameNumber": keyframe.frame_number,
                    "imagePath": keyframe.image_path,
                    "startTime": keyframe.start_time,
                    "endTime": keyframe.end_time,
                    "description": keyframe.description,
                    "source": getattr(keyframe, 'source', 'unknown'),
                    "extractText": getattr(keyframe, 'extract_text', None),
                    "clipDescription": getattr(keyframe, 'clip_description', None)
                }
                
                # Add to batch
                import weaviate.classes as wvc
                batch_data.append(
                    wvc.data.DataObject(
                        properties=data_object,
                        vector=image_embedding.tolist()  # Convert numpy array to list
                    )
                )
                
            except Exception as e:
                logger.error(f"Error encoding keyframe {keyframe.image_path}: {e}")
        
        # Insert batch
        try:
            response = collection.data.insert_many(batch_data)
            logger.info(f"✅ {len(batch_data)} keyframes ingested successfully into Weaviate")
        except Exception as e:
            logger.error(f"Error inserting batch: {e}")
    
    def search_by_text(self, query: str, top_k: int = 10) -> List[Tuple[Dict, float]]:
        """Search for keyframes using text query"""
        
        # Encode query text
        query_vector = self.encode_text(query)
        
        # Get collection and perform vector search
        collection = self.client.collections.get(self.class_name)
        
        response = collection.query.near_vector(
            near_vector=query_vector.tolist(),  # Convert numpy array to list
            limit=top_k,
            return_metadata=['distance']
        )
        
        # Parse results
        results = []
        for obj in response.objects:
            # Convert to KeyFrameInfo-like dict
            keyframe_data = {
                'video_id': obj.properties['videoId'],
                'frame_number': obj.properties['frameNumber'], 
                'image_path': obj.properties['imagePath'],
                'start_time': obj.properties['startTime'],
                'end_time': obj.properties['endTime'],
                'description': obj.properties['description'],
                'source': obj.properties.get('source', 'unknown'),
                'extract_text': obj.properties.get('extractText', None),
                'clip_description': obj.properties.get('clipDescription', None)
            }
            
            # Get similarity score (convert distance to similarity)
            distance = obj.metadata.distance if obj.metadata.distance else 0.0
            similarity = 1.0 - distance  # Convert distance to similarity
            
            results.append((keyframe_data, similarity))
        
        return results
    
    def get_statistics(self) -> Dict:
        """Get statistics about the indexed data"""
        try:
            # Get collection
            collection = self.client.collections.get(self.class_name)
            
            # Get total count using aggregate query
            response = collection.aggregate.over_all(total_count=True)
            count = response.total_count if response.total_count else 0
            
            # Get unique videos and additional statistics
            try:
                # Query all objects to count unique videos and analyze content
                all_objects = collection.query.fetch_objects(limit=10000)  # Adjust limit as needed
                unique_video_ids = set()
                source_counts = {}
                ocr_count = 0
                clip_desc_count = 0
                
                for obj in all_objects.objects:
                    if 'videoId' in obj.properties:
                        unique_video_ids.add(obj.properties['videoId'])
                    
                    # Count sources
                    source = obj.properties.get('source', 'unknown')
                    source_counts[source] = source_counts.get(source, 0) + 1
                    
                    # Count OCR coverage
                    if obj.properties.get('extractText'):
                        ocr_count += 1
                    
                    # Count CLIP description coverage
                    if obj.properties.get('clipDescription'):
                        clip_desc_count += 1
                
                unique_videos = len(unique_video_ids)
            except Exception as e:
                logger.warning(f"Could not get detailed statistics: {e}")
                unique_videos = 0
                source_counts = {}
                ocr_count = 0
                clip_desc_count = 0
            
            return {
                "total_keyframes": count,
                "unique_videos": unique_videos,
                "ocr_coverage": ocr_count,
                "clip_description_coverage": clip_desc_count,
                "source_distribution": source_counts,
                "database": "Weaviate",
                "url": self.weaviate_url
            }
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {"error": str(e)}

    def browse_objects(self, limit: int = 25, offset: int = 0, video_id: Optional[str] = None) -> Dict:
        """Browse stored objects for lightweight admin/debug inspection."""
        try:
            import weaviate.classes as wvc

            collection = self.client.collections.get(self.class_name)

            filters = None
            if video_id:
                filters = wvc.query.Filter.by_property("videoId").equal(video_id)

            query_kwargs = {"limit": limit}
            if filters is not None:
                query_kwargs["filters"] = filters
            if offset:
                query_kwargs["offset"] = offset

            try:
                response = collection.query.fetch_objects(**query_kwargs)
                objects = response.objects
            except TypeError:
                # Older fetch_objects signatures may not support offset.
                fallback_kwargs = {"limit": limit + offset}
                if filters is not None:
                    fallback_kwargs["filters"] = filters
                response = collection.query.fetch_objects(**fallback_kwargs)
                objects = response.objects[offset:offset + limit]

            records = []
            for obj in objects:
                records.append({
                    "uuid": str(obj.uuid),
                    "video_id": obj.properties.get("videoId"),
                    "frame_number": obj.properties.get("frameNumber"),
                    "image_path": obj.properties.get("imagePath"),
                    "start_time": obj.properties.get("startTime"),
                    "end_time": obj.properties.get("endTime"),
                    "description": obj.properties.get("description", ""),
                    "source": obj.properties.get("source", "unknown"),
                    "extract_text": obj.properties.get("extractText", None),
                    "clip_description": obj.properties.get("clipDescription", None)
                })

            return {
                "collection": self.class_name,
                "limit": limit,
                "offset": offset,
                "video_id_filter": video_id,
                "returned_objects": len(records),
                "objects": records
            }

        except Exception as e:
            logger.error(f"Error browsing Weaviate objects: {e}")
            return {"error": str(e)}
    
    def close(self):
        """Close the Weaviate client connection"""
        try:
            self.client.close()
            logger.info("Weaviate client connection closed")
        except Exception as e:
            logger.error(f"Error closing Weaviate client: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()