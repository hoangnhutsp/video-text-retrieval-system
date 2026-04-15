# Multi-stage Docker build for Video-Text Retrieval API
FROM python:3.11-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .
COPY requirements-docker.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements-docker.txt
RUN pip install --no-cache-dir git+https://github.com/openai/CLIP.git

# Copy application code
COPY services/api/video_retrieval_system.py .
COPY services/api/weaviate_retrieval.py .
COPY services/api/config.py .
COPY services/api/api.py .

# Create directories
RUN mkdir -p /app/dataset /app/video_retrieval_index /app/logs

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PORT=5000

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD sh -c 'curl -f http://localhost:${PORT:-5000}/health || exit 1'

# Run the API
CMD ["python", "api.py"]
