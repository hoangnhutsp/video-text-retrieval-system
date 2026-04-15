"""
Flask API for Video-Text Retrieval System
Provides REST endpoints for video search functionality
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import logging
from pathlib import Path
import json
from datetime import datetime
from typing import List, Dict

# Import our retrieval systems
from video_retrieval_system import VideoTextRetrievalSystem
from weaviate_retrieval import WeaviateVideoRetrieval

app = Flask(__name__)
CORS(app)  # Enable cross-origin requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
DATASET_PATH = os.getenv('DATASET_PATH', '/app/dataset')
VECTOR_DB_URL = os.getenv('VECTOR_DB_URL', 'http://localhost:8082')
INDEX_PATH = os.getenv('INDEX_PATH', '/app/video_retrieval_index')
DB_TYPE = os.getenv('DB_TYPE', 'weaviate')  # 'weaviate', 'faiss', 'qdrant', 'chroma'

# Global retrieval system instance
retrieval_system = None

def initialize_retrieval_system():
    """Initialize the appropriate retrieval system based on configuration"""
    global retrieval_system
    
    try:
        if DB_TYPE.lower() == 'weaviate':
            retrieval_system = WeaviateVideoRetrieval(
                weaviate_url=VECTOR_DB_URL,
                dataset_path=DATASET_PATH
            )
            logger.info(f"Initialized Weaviate system at {VECTOR_DB_URL}")
            
        else:  # Default to FAISS
            retrieval_system = VideoTextRetrievalSystem(DATASET_PATH)
            # Try to load existing index
            try:
                retrieval_system.load_index(INDEX_PATH)
                logger.info(f"Loaded FAISS index from {INDEX_PATH}")
            except Exception as e:
                logger.warning(f"Could not load FAISS index: {e}")
                logger.info("System ready but index needs to be built")
                
    except Exception as e:
        logger.error(f"Failed to initialize retrieval system: {e}")
        retrieval_system = None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    status = {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'database_type': DB_TYPE,
        'database_url': VECTOR_DB_URL,
        'retrieval_system_ready': retrieval_system is not None
    }
    
    if retrieval_system:
        try:
            # Try to get statistics to verify system is working
            if hasattr(retrieval_system, 'get_statistics'):
                stats = retrieval_system.get_statistics()
                status['statistics'] = stats
            else:
                status['keyframe_count'] = len(getattr(retrieval_system, 'keyframes', []))
                
        except Exception as e:
            status['error'] = str(e)
            status['status'] = 'degraded'
    
    return jsonify(status)

@app.route('/search', methods=['POST'])
def search_videos():
    """Search for videos using text query"""
    if not retrieval_system:
        return jsonify({'error': 'Retrieval system not initialized'}), 500
    
    try:
        data = request.get_json()
        query = data.get('query', '')
        top_k = data.get('top_k', 10)
        
        if not query:
            return jsonify({'error': 'Query parameter is required'}), 400
        
        # Perform search
        results = retrieval_system.search_by_text(query, top_k=top_k)
        
        # Format results
        formatted_results = []
        for item, score in results:
            if isinstance(item, dict):  # Weaviate format
                result = {
                    'video_id': item['video_id'],
                    'frame_number': item['frame_number'],
                    'image_path': item['image_path'],
                    'start_time': item['start_time'],
                    'end_time': item['end_time'],
                    'description': item['description'],
                    'source': item.get('source', 'unknown'),
                    'extract_text': item.get('extract_text', None),
                    'clip_description': item.get('clip_description', None),
                    'similarity_score': float(score)
                }
            else:  # FAISS format (KeyFrameInfo object)
                result = {
                    'video_id': item.video_id,
                    'frame_number': item.frame_number,
                    'image_path': item.image_path,
                    'start_time': item.start_time,
                    'end_time': item.end_time,
                    'description': item.description,
                    'source': getattr(item, 'source', 'unknown'),
                    'extract_text': getattr(item, 'extract_text', None),
                    'clip_description': getattr(item, 'clip_description', None),
                    'similarity_score': float(score)
                }
            formatted_results.append(result)
        
        return jsonify({
            'query': query,
            'results': formatted_results,
            'total_results': len(formatted_results),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/image/<path:image_path>')
def serve_image(image_path):
    """Serve keyframe images"""
    try:
        # Security: ensure path is within dataset directory
        full_path = Path(DATASET_PATH) / image_path
        
        # Check if path exists and is within dataset directory
        if not full_path.exists():
            return jsonify({'error': 'Image not found'}), 404
        
        # Resolve to check it's within dataset directory (prevent path traversal)
        try:
            full_path.resolve().relative_to(Path(DATASET_PATH).resolve())
        except ValueError:
            return jsonify({'error': 'Access denied'}), 403
        
        return send_file(full_path, mimetype='image/png')
        
    except Exception as e:
        logger.error(f"Image serve error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/statistics', methods=['GET'])
def get_statistics():
    """Get system statistics"""
    if not retrieval_system:
        return jsonify({'error': 'Retrieval system not initialized'}), 500
    
    try:
        if hasattr(retrieval_system, 'get_statistics'):
            stats = retrieval_system.get_statistics()
        else:
            # FAISS system statistics
            keyframe_count = len(getattr(retrieval_system, 'keyframes', []))
            unique_videos = len(set(kf.video_id for kf in retrieval_system.keyframes)) if hasattr(retrieval_system, 'keyframes') else 0
            
            # Additional statistics for enhanced features
            ocr_count = 0
            clip_desc_count = 0
            source_counts = {}
            
            if hasattr(retrieval_system, 'keyframes') and retrieval_system.keyframes:
                for kf in retrieval_system.keyframes:
                    if hasattr(kf, 'extract_text') and kf.extract_text:
                        ocr_count += 1
                    if hasattr(kf, 'clip_description') and kf.clip_description:
                        clip_desc_count += 1
                    source = getattr(kf, 'source', 'unknown')
                    source_counts[source] = source_counts.get(source, 0) + 1
            
            stats = {
                'total_keyframes': keyframe_count,
                'unique_videos': unique_videos,
                'ocr_coverage': ocr_count,
                'clip_description_coverage': clip_desc_count,
                'source_distribution': source_counts,
                'database': 'FAISS',
                'index_path': INDEX_PATH
            }
            
            if hasattr(retrieval_system, 'image_embeddings') and retrieval_system.image_embeddings is not None:
                stats['embedding_shape'] = list(retrieval_system.image_embeddings.shape)
        
        stats['database_type'] = DB_TYPE
        stats['timestamp'] = datetime.now().isoformat()
        
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Statistics error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/weaviate/objects', methods=['GET'])
def browse_weaviate_objects():
    """Browse Weaviate objects for lightweight inspection"""
    if not retrieval_system:
        return jsonify({'error': 'Retrieval system not initialized'}), 500

    if DB_TYPE.lower() != 'weaviate':
        return jsonify({'error': 'Database browser is only available for Weaviate'}), 400

    if not hasattr(retrieval_system, 'browse_objects'):
        return jsonify({'error': 'Current retrieval system does not support browsing'}), 400

    try:
        limit = int(request.args.get('limit', 25))
        offset = int(request.args.get('offset', 0))
    except ValueError:
        return jsonify({'error': 'limit and offset must be integers'}), 400

    limit = max(1, min(limit, 100))
    offset = max(0, offset)
    video_id = request.args.get('video_id', '').strip() or None

    try:
        payload = retrieval_system.browse_objects(
            limit=limit,
            offset=offset,
            video_id=video_id
        )

        if 'error' in payload:
            return jsonify(payload), 500

        payload['database_type'] = DB_TYPE
        payload['timestamp'] = datetime.now().isoformat()
        return jsonify(payload)

    except Exception as e:
        logger.error(f"Weaviate browse error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/build-index', methods=['POST'])
def build_index():
    """Build search index (for FAISS system)"""
    if not retrieval_system:
        return jsonify({'error': 'Retrieval system not initialized'}), 500
    
    if DB_TYPE.lower() == 'weaviate':
        return jsonify({'error': 'Index building not needed for Weaviate'}), 400
    
    try:
        data = request.get_json() or {}
        dataset_versions = data.get('dataset_versions', ['v3c1'])
        
        logger.info("Starting index build...")
        
        # Load dataset
        retrieval_system.load_dataset(dataset_versions)
        logger.info(f"Loaded {len(retrieval_system.keyframes)} keyframes")
        
        if len(retrieval_system.keyframes) == 0:
            return jsonify({'error': 'No keyframes found in dataset'}), 400
        
        # Create embeddings
        retrieval_system.encode_images()
        retrieval_system.encode_text_descriptions()
        
        # Build index
        retrieval_system.build_faiss_index()
        
        # Save index
        retrieval_system.save_index(INDEX_PATH)
        
        return jsonify({
            'status': 'success',
            'keyframes_indexed': len(retrieval_system.keyframes),
            'dataset_versions': dataset_versions,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Index build error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/ingest', methods=['POST'])
def ingest_data():
    """Ingest data into vector database (for Weaviate/other vector DBs)"""
    if not retrieval_system:
        return jsonify({'error': 'Retrieval system not initialized'}), 500
    
    if DB_TYPE.lower() != 'weaviate':
        return jsonify({'error': 'Ingestion only available for Weaviate'}), 400
    
    try:
        data = request.get_json() or {}
        dataset_versions = data.get('dataset_versions', ['v3c1'])
        
        logger.info("Starting data ingestion...")
        
        # Create schema
        retrieval_system.create_schema()
        
        # Load dataset using helper
        from video_retrieval_system import VideoTextRetrievalSystem
        loader = VideoTextRetrievalSystem(DATASET_PATH)
        loader.load_dataset(dataset_versions)
        
        logger.info(f"Loaded {len(loader.keyframes)} keyframes")
        
        if len(loader.keyframes) == 0:
            return jsonify({'error': 'No keyframes found in dataset'}), 400
        
        # Ingest into Weaviate
        retrieval_system.ingest_keyframes(loader.keyframes)
        
        return jsonify({
            'status': 'success', 
            'keyframes_ingested': len(loader.keyframes),
            'dataset_versions': dataset_versions,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Ingestion error: {e}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Initialize retrieval system on startup
    initialize_retrieval_system()
    
    # Run the app
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
