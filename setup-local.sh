#!/bin/bash
# Local Development Setup Script
# Automates the setup process for local development

set -e  # Exit on error

echo "🚀 Video-Text Retrieval System - Local Setup"
echo "============================================="

# Check Python version
echo "1️⃣ Checking Python version..."
python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
echo "   Python version: $python_version"

if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3,8) else 1)" 2>/dev/null; then
    echo "❌ Python 3.8+ required. Current: $python_version"
    exit 1
fi
echo "✅ Python version OK"

# Create virtual environment
echo ""
echo "2️⃣ Creating virtual environment..."
if [ -d "venv" ]; then
    echo "⚠️  Virtual environment already exists. Remove? (y/n)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        rm -rf venv
        echo "   Removed existing venv"
    else
        echo "   Keeping existing venv"
    fi
fi

if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment  
echo ""
echo "3️⃣ Activating virtual environment..."
source venv/bin/activate
echo "✅ Virtual environment activated"

# Install dependencies
echo ""
echo "4️⃣ Installing dependencies..."
echo "   Installing base requirements..."
pip install -r requirements.txt

echo "   Installing CLIP model..."
pip install git+https://github.com/openai/CLIP.git

echo "   Installing API requirements..." 
pip install -r requirements-docker.txt

echo "   Installing Streamlit requirements..."
pip install -r requirements-streamlit.txt

echo "✅ All dependencies installed"

# Create environment file
echo ""
echo "5️⃣ Creating environment configuration..."
cat > .env << EOF
APP_PORT=8088
APP_CONTAINER_PORT=80
API_PORT=5001
API_CONTAINER_PORT=5000
STREAMLIT_PORT=8501
STREAMLIT_CONTAINER_PORT=8501
WEAVIATE_PORT=8082
WEAVIATE_CONTAINER_PORT=8080
VECTOR_DB_URL=http://localhost:8082
DATASET_PATH=$(pwd)/dataset  
DB_TYPE=weaviate
PYTHONPATH=$(pwd)/services/core:$(pwd)/services/weaviate-service
EOF
echo "✅ Environment file created: .env"

# Test imports
echo ""
echo "6️⃣ Testing core imports..."
python3 -c "
try:
    import torch
    print('✅ PyTorch:', torch.__version__)
except Exception as e:
    print('❌ PyTorch error:', e)

try:
    import clip
    print('✅ CLIP: OK')
except Exception as e:
    print('❌ CLIP error:', e)
    
try:
    import weaviate
    print('✅ Weaviate client: OK') 
except Exception as e:
    print('❌ Weaviate error:', e)
    
try:
    import streamlit
    print('✅ Streamlit: OK')
except Exception as e:
    print('❌ Streamlit error:', e)
"

# Check dataset
echo ""
echo "7️⃣ Checking dataset..."
echo "   Make sure to place your V3C dataset in the dataset/ folder"

if [ -d "dataset/v3c1" ]; then
    video_count=$(find dataset/v3c1 -maxdepth 1 -type d | wc -l)
    echo "✅ Dataset found: $((video_count-1)) video directories in v3c1"
else
    echo "⚠️  Dataset not found at dataset/v3c1"
fi

if [ -d "dataset/v3c2" ]; then
    video_count=$(find dataset/v3c2 -maxdepth 1 -type d | wc -l)
    echo "✅ Dataset found: $((video_count-1)) video directories in v3c2"
else
    echo "⚠️  Dataset not found at dataset/v3c2"
fi

if [ -d "dataset/v3c3" ]; then
    video_count=$(find dataset/v3c3 -maxdepth 1 -type d | wc -l)
    echo "✅ Dataset found: $((video_count-1)) video directories in v3c3"
else
    echo "⚠️  Dataset not found at dataset/v3c3"
fi


# Final instructions
echo ""
echo "🎉 Local development setup complete!"
echo ""
