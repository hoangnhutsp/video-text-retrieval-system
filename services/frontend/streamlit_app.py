"""
Streamlit Web UI for Video-Text Retrieval System
"""

import streamlit as st
import requests
import json
from PIL import Image
import pandas as pd
import plotly.express as px
from datetime import datetime
import os
from io import BytesIO

# Configuration
API_URL = os.getenv('API_URL', 'http://localhost:5000')

st.set_page_config(
    page_title="Video-Text Retrieval System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

def check_api_connection():
    """Check if API is accessible"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except Exception as e:
        return False, str(e)

def search_videos(query, top_k=10):
    """Search for videos using the API"""
    try:
        response = requests.post(
            f"{API_URL}/search",
            json={"query": query, "top_k": top_k},
            timeout=30
        )
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, response.json().get('error', 'Unknown error')
    except Exception as e:
        return False, str(e)

def get_statistics():
    """Get system statistics"""
    try:
        response = requests.get(f"{API_URL}/statistics", timeout=10)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, response.json().get('error', 'Unknown error')
    except Exception as e:
        return False, str(e)

def browse_database(limit=25, offset=0, video_id=""):
    """Browse stored Weaviate objects"""
    try:
        params = {"limit": limit, "offset": offset}
        if video_id:
            params["video_id"] = video_id

        response = requests.get(
            f"{API_URL}/weaviate/objects",
            params=params,
            timeout=30
        )
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, response.json().get('error', 'Unknown error')
    except Exception as e:
        return False, str(e)

def resolve_relative_image_path(image_path):
    """Convert stored dataset paths into API-friendly relative paths."""
    if not image_path:
        return None
    if image_path.startswith('/app/dataset/'):
        return image_path.replace('/app/dataset/', '')
    if '/dataset/' in image_path:
        return image_path.split('/dataset/', 1)[-1]
    return image_path.lstrip('/')

def load_image_from_api(image_path):
    """Fetch an image preview through the API."""
    relative_path = resolve_relative_image_path(image_path)
    if not relative_path:
        return None

    image_url = f"{API_URL}/image/{relative_path}"
    response = requests.get(image_url, timeout=10)
    if response.status_code != 200:
        return None

    return Image.open(BytesIO(response.content))

def format_time_range(start_time, end_time):
    """Render keyframe times safely, even when values are missing."""
    start_value = 0.0 if start_time is None else float(start_time)
    end_value = 0.0 if end_time is None else float(end_time)
    return f"{start_value:.1f}s - {end_value:.1f}s"

def build_index(dataset_versions):
    """Build search index"""
    try:
        response = requests.post(
            f"{API_URL}/build-index",
            json={"dataset_versions": dataset_versions},
            timeout=300  # 5 minutes timeout for index building
        )
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, response.json().get('error', 'Unknown error')
    except Exception as e:
        return False, str(e)

def ingest_data(dataset_versions):
    """Ingest data into vector database"""
    try:
        response = requests.post(
            f"{API_URL}/ingest",
            json={"dataset_versions": dataset_versions},
            timeout=300  # 5 minutes timeout
        )
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, response.json().get('error', 'Unknown error')
    except Exception as e:
        return False, str(e)

def main():
    st.title("🎬 Video-Text Retrieval System")
    st.markdown("Find video keyframes using natural language queries")
    
    # Sidebar for system status and controls
    with st.sidebar:
        st.header("System Status")
        
        # Check API connection
        is_connected, health_data = check_api_connection()
        
        if is_connected:
            st.success("✅ API Connected")
            if health_data:
                st.json({
                    "Status": health_data.get('status', 'unknown'),
                    "Database": health_data.get('database_type', 'unknown'),
                    "Ready": health_data.get('retrieval_system_ready', False)
                })
        else:
            st.error("❌ API Not Connected")
            st.error(f"Error: {health_data}")
            return
        
        st.divider()
        
        # System controls
        st.header("System Controls")
        
        dataset_options = st.multiselect(
            "Dataset Versions",
            options=['v3c1', 'v3c2', 'v3c3'],
            default=['v3c1']
        )
        
        if st.button("🔧 Build Index (FAISS)"):
            with st.spinner("Building index..."):
                success, result = build_index(dataset_options)
                if success:
                    st.success(f"✅ Index built! {result.get('keyframes_indexed', 0)} keyframes indexed")
                else:
                    st.error(f"❌ Build failed: {result}")
        
        if st.button("📥 Ingest Data (Weaviate)"):
            with st.spinner("Ingesting data..."):
                success, result = ingest_data(dataset_options)
                if success:
                    st.success(f"✅ Data ingested! {result.get('keyframes_ingested', 0)} keyframes")
                else:
                    st.error(f"❌ Ingestion failed: {result}")

    stats_success, stats = get_statistics()
    
    # Main search interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🔍 Search Videos")
        
        # Search form
        with st.form("search_form"):
            query = st.text_input(
                "Enter your search query:",
                placeholder="e.g., people riding bikes, urban street scene, cycling tricks..."
            )
            
            col_search, col_limit = st.columns([3, 1])
            with col_search:
                search_button = st.form_submit_button("Search", use_container_width=True)
            with col_limit:
                top_k = st.selectbox("Results", options=[5, 10, 20, 50], index=1)
        
        # Example queries
        st.markdown("**Example queries:**")
        example_queries = [
            "people riding bicycles",
            "riders in Paris", 
            "cycling tricks",
            "urban street scene",
            "group of friends"
        ]
        
        cols = st.columns(len(example_queries))
        for i, example in enumerate(example_queries):
            with cols[i]:
                if st.button(f"'{example}'", key=f"example_{i}"):
                    query = example
                    search_button = True
        
        # Perform search when button clicked
        if search_button and query:
            with st.spinner(f"Searching for '{query}'..."):
                success, search_results = search_videos(query, top_k)
                
                if success:
                    results = search_results.get('results', [])
                    
                    st.success(f"Found {len(results)} results for '{query}'")
                    
                    if results:
                        # Display results
                        for i, result in enumerate(results):
                            with st.container():
                                st.markdown(f"### Result {i+1}")
                                
                                result_col1, result_col2 = st.columns([1, 2])
                                
                                with result_col1:
                                    # Display keyframe image
                                    try:
                                        try:
                                            img = load_image_from_api(result['image_path'])
                                            if img is not None:
                                                st.image(img, caption=f"Frame {result['frame_number']}", use_container_width=True)
                                            else:
                                                st.write(f"**Image:** {result['image_path'].split('/')[-1]}")
                                        except Exception:
                                            st.write(f"**Image:** {result['image_path'].split('/')[-1]}")
                                        
                                        st.write(f"**Video:** {result['video_id']}")
                                        st.write(f"**Frame:** {result['frame_number']}")
                                        st.write(f"**Time:** {format_time_range(result['start_time'], result['end_time'])}")
                                        st.write(f"**Score:** {result['similarity_score']:.4f}")
                                    except Exception as e:
                                        st.error(f"Could not load image: {e}")
                                        st.write(f"**Image:** {result['image_path'].split('/')[-1]}")
                                
                                with result_col2:
                                    st.write("**Description:**")
                                    st.write(result['description'])
                                
                                st.divider()
                    else:
                        st.info("No results found. Try a different query.")
                        
                else:
                    st.error(f"Search failed: {search_results}")
    
    with col2:
        st.header("📊 Statistics")
        
        # Get and display statistics
        if stats_success:
            # Key metrics
            st.metric("Total Keyframes", stats.get('total_keyframes', 0))
            st.metric("Unique Videos", stats.get('unique_videos', 0))
            st.metric("Database Type", stats.get('database_type', 'Unknown'))
            
            # Additional stats if available
            if 'embedding_shape' in stats:
                st.metric("Embedding Dimensions", stats['embedding_shape'][1])
            
            # Show full stats in expandable section
            with st.expander("Full Statistics"):
                st.json(stats)
                
        else:
            st.error(f"Could not load statistics: {stats}")

    st.markdown("---")
    st.header("🗂 Database Browser")

    if not stats_success:
        st.info("Statistics are unavailable right now, so the database browser is hidden until the API is ready.")
    elif stats.get('database_type', '').lower() != 'weaviate':
        st.info("The database browser is only available when the backend is using Weaviate.")
    else:
        show_browser = st.checkbox("Show Weaviate records", value=True)

        if show_browser:
            browser_col1, browser_col2, browser_col3 = st.columns([1, 1, 2])
            with browser_col1:
                browser_limit = st.selectbox("Page Size", options=[10, 25, 50, 100], index=1)
            with browser_col2:
                browser_offset = st.number_input("Offset", min_value=0, value=0, step=25)
            with browser_col3:
                browser_video_id = st.text_input("Filter by Video ID", placeholder="Optional exact videoId")

            with st.spinner("Loading Weaviate records..."):
                browser_success, browser_data = browse_database(
                    limit=browser_limit,
                    offset=browser_offset,
                    video_id=browser_video_id.strip()
                )

            if browser_success:
                records = browser_data.get('objects', [])
                st.caption(
                    f"Collection `{browser_data.get('collection', 'unknown')}` | "
                    f"showing {len(records)} records starting at offset {browser_data.get('offset', 0)}"
                )

                if records:
                    browser_df = pd.DataFrame([
                        {
                            "uuid": record.get("uuid"),
                            "video_id": record.get("video_id"),
                            "frame_number": record.get("frame_number"),
                            "start_time": record.get("start_time"),
                            "end_time": record.get("end_time"),
                            "image_file": os.path.basename(record.get("image_path", "")),
                            "description": record.get("description", "")
                        }
                        for record in records
                    ])
                    st.dataframe(browser_df, use_container_width=True, hide_index=True)

                    with st.expander("Preview first 3 records"):
                        for index, record in enumerate(records[:3], start=1):
                            preview_col1, preview_col2 = st.columns([1, 2])
                            with preview_col1:
                                try:
                                    preview_image = load_image_from_api(record.get("image_path"))
                                    if preview_image is not None:
                                        st.image(preview_image, caption=f"Record {index}", use_container_width=True)
                                    else:
                                        st.write(os.path.basename(record.get("image_path", "")))
                                except Exception as e:
                                    st.write(f"Preview unavailable: {e}")

                            with preview_col2:
                                st.write(f"**UUID:** {record.get('uuid')}")
                                st.write(f"**Video:** {record.get('video_id')}")
                                st.write(f"**Frame:** {record.get('frame_number')}")
                                st.write(f"**Time:** {format_time_range(record.get('start_time'), record.get('end_time'))}")
                                st.write("**Description:**")
                                st.write(record.get("description", ""))
                            st.divider()

                    with st.expander("Raw browser response"):
                        st.json(browser_data)
                else:
                    st.info("No records found for the current page/filter.")
            else:
                st.error(f"Could not load Weaviate records: {browser_data}")
    
    # Footer
    st.markdown("**Video-Text Retrieval System** | Powered by CLIP and Vector Search")

if __name__ == "__main__":
    main()
