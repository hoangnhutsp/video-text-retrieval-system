# Video-Text Retrieval System

A powerful video retrieval system that uses CLIP (Contrastive Language-Image Pre-training) to create searchable embeddings from video keyframes. This system allows you to search for specific video moments using natural language text queries.

## 🚀 Features

- **Multimodal Search**: Find video keyframes using text descriptions
- **CLIP-powered**: Uses OpenAI's CLIP model for state-of-the-art vision-language understanding
- **Efficient Indexing**: FAISS-based vector search for fast retrieval
- **Scalable**: Designed to handle large video datasets (V3C1, V3C2, V3C3)
- **Pretrained Models**: No training required - uses pretrained CLIP embeddings

## 📋 Requirements

- Python 3.7+
- GPU recommended (but CPU works too)
- ~4GB RAM for small datasets, more for larger ones

## 📁 Dataset Structure

Your dataset should be organized like this:

```
dataset/
├── v3c1/
│   ├── 00001/
│   │   ├── 00001.description    # Text description
│   │   ├── 00001.info.json      # Video metadata
│   │   ├── 00001.tsv            # Keyframe timing
│   │   ├── 00001.mp4            # Video file
│   │   └── keyframes/           # Extracted frames
│   │       ├── shot00001_1_RKF.png
│   │       ├── shot00001_2_RKF.png
│   │       └── ...
│   └── 00002/
├── v3c2/
└── v3c3/
```

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend UI   │────│  Nginx Proxy    │────│   Backend API   │
│  (Streamlit)    │    │                 │    │    (Flask)      │
│   Port: 8501    │    │   Port: 8080    │    │   Port: 5001    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                       ┌─────────────────┐             │
                       │  Core Engine    │◄────────────┘
                       │   (CLIP+FAISS)  │
                       └─────────────────┘
                                 │
                       ┌─────────────────┐
                       │   Weaviate DB   │
                       │   Port: 8082    │
                       └─────────────────┘
```