# Configuration file for Video-Text Retrieval System

import os
from pathlib import Path

class Config:
    """Configuration settings for the video retrieval system"""
    
    # Dataset settings
    DATASET_PATH = "/home/tranghoangnhut/Documents/paper/dataset"
    DATASET_VERSIONS = ['v3c1']  # Add 'v3c2', 'v3c3' as needed
    
    # Model settings  
    CLIP_MODEL = "ViT-B/32"  # Options: ViT-B/32, ViT-B/16, ViT-L/14, RN50
    DEVICE = "auto"  # "auto", "cpu", "cuda"
    
    # Processing settings
    BATCH_SIZE = 32  # Adjust based on GPU memory
    IMAGE_SIZE = 224  # CLIP default, don't change unless necessary
    
    # Index settings
    INDEX_SAVE_PATH = "./video_retrieval_index"
    FAISS_INDEX_TYPE = "IndexFlatIP"  # Inner product for cosine similarity
    
    # Search settings
    DEFAULT_TOP_K = 10
    SIMILARITY_THRESHOLD = 0.0  # Minimum similarity score to return
    
    # Performance settings
    MAX_KEYFRAMES_PER_VIDEO = None  # None for no limit, or set a number
    ENABLE_PROGRESS_BAR = True
    LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
    
    # Memory optimization
    CLEAR_CACHE_AFTER_BATCH = True  # Clear GPU cache between batches
    USE_HALF_PRECISION = False  # Use FP16 to save memory (may reduce accuracy)
    
    @classmethod
    def get_device(cls):
        """Get the appropriate device for computation"""
        if cls.DEVICE == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return cls.DEVICE
    
    @classmethod
    def get_batch_size(cls):
        """Get appropriate batch size based on device"""
        device = cls.get_device()
        if device == "cpu":
            return min(cls.BATCH_SIZE, 8)  # Smaller batches for CPU
        return cls.BATCH_SIZE
    
    @classmethod
    def validate_paths(cls):
        """Validate that required paths exist"""
        dataset_path = Path(cls.DATASET_PATH)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
        
        for version in cls.DATASET_VERSIONS:
            version_path = dataset_path / version
            if not version_path.exists():
                print(f"Warning: Dataset version {version} not found at {version_path}")
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("🔧 Current Configuration:")
        print("-" * 30)
        print(f"Dataset Path: {cls.DATASET_PATH}")
        print(f"Dataset Versions: {cls.DATASET_VERSIONS}")
        print(f"CLIP Model: {cls.CLIP_MODEL}")
        print(f"Device: {cls.get_device()}")
        print(f"Batch Size: {cls.get_batch_size()}")
        print(f"Index Save Path: {cls.INDEX_SAVE_PATH}")
        print(f"Top-K Results: {cls.DEFAULT_TOP_K}")

# Alternative configurations for different use cases

class FastConfig(Config):
    """Configuration optimized for speed"""
    CLIP_MODEL = "ViT-B/32"
    BATCH_SIZE = 64
    USE_HALF_PRECISION = True

class AccuracyConfig(Config):
    """Configuration optimized for accuracy"""
    CLIP_MODEL = "ViT-L/14"
    BATCH_SIZE = 16  # Smaller batch for larger model

class CPUConfig(Config):
    """Configuration for CPU-only systems"""
    DEVICE = "cpu"
    BATCH_SIZE = 4
    USE_HALF_PRECISION = False

class LowMemoryConfig(Config):
    """Configuration for systems with limited memory"""
    CLIP_MODEL = "ViT-B/32"
    BATCH_SIZE = 8
    USE_HALF_PRECISION = True
    CLEAR_CACHE_AFTER_BATCH = True
    MAX_KEYFRAMES_PER_VIDEO = 50

# Example of custom configuration usage:
# 
# from config import AccuracyConfig
# retrieval_system = VideoTextRetrievalSystem(
#     AccuracyConfig.DATASET_PATH, 
#     model_name=AccuracyConfig.CLIP_MODEL
# )
# retrieval_system.load_dataset(AccuracyConfig.DATASET_VERSIONS)
# retrieval_system.encode_images(batch_size=AccuracyConfig.get_batch_size())