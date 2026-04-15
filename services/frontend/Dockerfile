# Dockerfile for Streamlit UI
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements-streamlit.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-streamlit.txt

# Copy application code
COPY services/frontend/streamlit_app.py .

# Create necessary directories
RUN mkdir -p /app/cache

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8501

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD sh -c 'curl -f http://localhost:${STREAMLIT_SERVER_PORT:-8501}/_stcore/health || exit 1'

# Run Streamlit
CMD ["sh", "-c", "streamlit run streamlit_app.py --server.port=${STREAMLIT_SERVER_PORT:-8501} --server.address=0.0.0.0"]
