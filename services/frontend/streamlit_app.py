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
                                        st.write(f"**Source:** {result.get('source', 'Unknown')}")
                                        st.write(f"**Time:** {format_time_range(result['start_time'], result['end_time'])}")
                                        st.write(f"**Score:** {result['similarity_score']:.4f}")
                                    except Exception as e:
                                        st.error(f"Could not load image: {e}")
                                        st.write(f"**Image:** {result['image_path'].split('/')[-1]}")
                                
                                with result_col2:
                                    st.write("**Original Description:**")
                                    st.write(result['description'])
                                    
                                    # Show CLIP description if available
                                    if result.get('clip_description'):
                                        st.write("**CLIP Description:**")
                                        st.write(result['clip_description'])
                                    
                                    # Show OCR text if available
                                    if result.get('extract_text'):
                                        st.write("**OCR Extracted Text:**")
                                        st.write(result['extract_text'])
                                
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
            col_stat1, col_stat2 = st.columns(2)
            with col_stat1:
                st.metric("Total Keyframes", stats.get('total_keyframes', 0))
                st.metric("Unique Videos", stats.get('unique_videos', 0))
            
            with col_stat2:
                st.metric("OCR Coverage", stats.get('ocr_coverage', 0))
                st.metric("CLIP Descriptions", stats.get('clip_description_coverage', 0))
            
            # Database info
            st.metric("Database Type", stats.get('database_type', 'Unknown'))
            
            # Source distribution if available
            source_dist = stats.get('source_distribution', {})
            if source_dist:
                st.write("**Dataset Distribution:**")
                for source, count in source_dist.items():
                    st.write(f"• {source}: {count} keyframes")
            
            # Additional stats if available
            if 'embedding_shape' in stats:
                st.metric("Embedding Dimensions", stats['embedding_shape'][1])
            
            # Show full stats in expandable section
            with st.expander("Full Statistics"):
                st.json(stats)
                
        else:
            st.error(f"Could not load statistics: {stats}")

    st.markdown("---")
    st.header("🗂 Enhanced Database Browser")

    if not stats_success:
        st.info("Statistics are unavailable right now, so the database browser is hidden until the API is ready.")
    elif stats.get('database_type', '').lower() != 'weaviate':
        st.info("The database browser is only available when the backend is using Weaviate.")
    else:
        # Enhanced browser controls
        browser_tab1, browser_tab2, browser_tab3, browser_tab4 = st.tabs([
            "📋 Browse Records", 
            "📊 Collection Analytics", 
            "🔍 Advanced Search", 
            "⚙️ Management Tools"
        ])
        
        with browser_tab1:
            st.subheader("Browse Database Records")
            
            # Enhanced filtering controls
            filter_col1, filter_col2, filter_col3, filter_col4 = st.columns([1, 1, 1, 1])
            
            with filter_col1:
                browser_limit = st.selectbox("Page Size", options=[10, 25, 50, 100], index=1)
                browser_offset = st.number_input("Offset", min_value=0, value=0, step=25)
                
            with filter_col2:
                dataset_filter = st.selectbox(
                    "Dataset Source", 
                    options=["All", "v3c1", "v3c2", "v3c3"],
                    index=0
                )
                time_filter = st.selectbox(
                    "Duration Range", 
                    options=["All", "< 5s", "5-30s", "> 30s"],
                    index=0
                )
                
            with filter_col3:
                browser_video_id = st.text_input("Video ID", placeholder="Optional exact videoId")
                frame_min = st.number_input("Min Frame #", min_value=0, value=0)
                
            with filter_col4:
                description_filter = st.text_input("Description Contains", placeholder="Search in descriptions")
                frame_max = st.number_input("Max Frame #", min_value=0, value=999999)

            # Display mode selection
            display_mode = st.radio(
                "Display Mode", 
                options=["Table View", "Card View", "Grid View"], 
                horizontal=True,
                index=0
            )
            
            # Load and display records
            with st.spinner("Loading database records..."):
                browser_success, browser_data = browse_database(
                    limit=browser_limit,
                    offset=browser_offset,
                    video_id=browser_video_id.strip()
                )

            if browser_success:
                records = browser_data.get('objects', [])
                
                # Apply client-side filters
                filtered_records = []
                for record in records:
                    # Dataset filter
                    if dataset_filter != "All" and record.get('source') != dataset_filter:
                        continue
                    # Frame range filter
                    frame_num = record.get('frame_number', 0)
                    if frame_num < frame_min or frame_num > frame_max:
                        continue
                    # Description filter
                    if description_filter and description_filter.lower() not in record.get('description', '').lower():
                        continue
                    # Time duration filter
                    if time_filter != "All":
                        start_time = record.get('start_time', 0) or 0
                        end_time = record.get('end_time', 0) or 0
                        duration = end_time - start_time
                        if time_filter == "< 5s" and duration >= 5:
                            continue
                        elif time_filter == "5-30s" and (duration < 5 or duration > 30):
                            continue
                        elif time_filter == "> 30s" and duration <= 30:
                            continue
                    
                    filtered_records.append(record)
                
                st.caption(
                    f"Collection `{browser_data.get('collection', 'unknown')}` | "
                    f"showing {len(filtered_records)}/{len(records)} records "
                    f"(offset {browser_data.get('offset', 0)})"
                )

                if filtered_records:
                    # Table View
                    if display_mode == "Table View":
                        browser_df = pd.DataFrame([
                            {
                                "UUID": record.get("uuid", "")[:8] + "...",
                                "Video ID": record.get("video_id", ""),
                                "Source": record.get("source", ""),
                                "Frame #": record.get("frame_number", ""),
                                "Start Time": f"{record.get('start_time', 0):.1f}s",
                                "End Time": f"{record.get('end_time', 0):.1f}s",
                                "Duration": f"{(record.get('end_time', 0) or 0) - (record.get('start_time', 0) or 0):.1f}s",
                                "Image File": os.path.basename(record.get("image_path", "")),
                                "Description": record.get("description", "")[:100] + "..." if len(record.get("description", "")) > 100 else record.get("description", ""),
                                "Has OCR": record.get("extract_text"),
                                "Has CLIP Desc": record.get("clip_description")
                            }
                            for record in filtered_records
                        ])
                        
                        # Make table interactive
                        selected_rows = st.dataframe(
                            browser_df, 
                            use_container_width=True, 
                            hide_index=True,
                            on_select="rerun",
                            selection_mode="multi-row"
                        )
                        
                        # Show selected record details
                        if hasattr(selected_rows, 'selection') and selected_rows.selection.rows:
                            st.subheader("Selected Record Details")
                            for row_idx in selected_rows.selection.rows[:3]:  # Limit to 3 selections
                                if row_idx < len(filtered_records):
                                    record = filtered_records[row_idx]
                                    with st.expander(f"Record: {record.get('video_id', '')} - Frame {record.get('frame_number', '')}"):
                                        detail_col1, detail_col2 = st.columns([1, 2])
                                        
                                        with detail_col1:
                                            try:
                                                preview_img = load_image_from_api(record.get("image_path"))
                                                if preview_img is not None:
                                                    st.image(preview_img, use_container_width=True)
                                                else:
                                                    st.write("Image preview unavailable")
                                            except:
                                                st.write("Image preview unavailable")
                                        
                                        with detail_col2:
                                            st.json({
                                                "UUID": record.get("uuid"),
                                                "Video ID": record.get("video_id"),
                                                "Source Dataset": record.get("source"),
                                                "Frame Number": record.get("frame_number"),
                                                "Start Time": record.get("start_time"),
                                                "End Time": record.get("end_time"),
                                                "Image Path": record.get("image_path"),
                                                "Description": record.get("description"),
                                                "OCR Text": record.get("extract_text"),
                                                "CLIP Description": record.get("clip_description")
                                            })
                    
                    # Card View
                    elif display_mode == "Card View":
                        for i, record in enumerate(filtered_records[:20]):  # Limit for performance
                            with st.container():
                                card_col1, card_col2 = st.columns([1, 3])
                                
                                with card_col1:
                                    try:
                                        card_img = load_image_from_api(record.get("image_path"))
                                        if card_img is not None:
                                            st.image(card_img, use_container_width=True)
                                    except:
                                        st.write("🖼️ Image unavailable")
                                
                                with card_col2:
                                    st.markdown(f"**{record.get('video_id', 'Unknown')} - Frame {record.get('frame_number', '?')}**")
                                    st.text(f"Source: {record.get('source', 'Unknown')} | Duration: {format_time_range(record.get('start_time'), record.get('end_time'))}")
                                    
                                    if record.get('description'):
                                        st.text("📝 Description:")
                                        st.text(record.get('description', '')[:200] + ("..." if len(record.get('description', '')) > 200 else ""))
                                    
                                    if record.get('clip_description'):
                                        st.text("🎯 CLIP Description:")
                                        st.text(record.get('clip_description', ''))
                                    
                                    if record.get('extract_text'):
                                        st.text("📖 OCR Text:")
                                        st.text(record.get('extract_text', '')[:100] + ("..." if len(record.get('extract_text', '')) > 100 else ""))
                                
                                st.divider()
                    
                    # Grid View
                    elif display_mode == "Grid View":
                        cols_per_row = 4
                        for i in range(0, min(len(filtered_records), 20), cols_per_row):  # Limit for performance
                            cols = st.columns(cols_per_row)
                            for j in range(cols_per_row):
                                if i + j < len(filtered_records):
                                    record = filtered_records[i + j]
                                    with cols[j]:
                                        try:
                                            grid_img = load_image_from_api(record.get("image_path"))
                                            if grid_img is not None:
                                                st.image(grid_img, use_container_width=True)
                                            else:
                                                st.write("🖼️ No preview")
                                        except:
                                            st.write("🖼️ No preview")
                                        
                                        st.caption(f"{record.get('video_id', '')[:10]}")
                                        st.caption(f"Frame {record.get('frame_number', '')}")
                                        st.caption(f"Source: {record.get('source', '')}")

                else:
                    st.info("No records found matching the current filters.")
            else:
                st.error(f"Could not load database records: {browser_data}")

        with browser_tab2:
            st.subheader("Collection Analytics")
            
            if browser_success and filtered_records:
                # Dataset distribution
                source_counts = {}
                duration_data = []
                frame_data = []
                
                for record in records:  # Use all records for analytics
                    source = record.get('source', 'Unknown')
                    source_counts[source] = source_counts.get(source, 0) + 1
                    
                    start_time = record.get('start_time', 0) or 0
                    end_time = record.get('end_time', 0) or 0
                    duration = end_time - start_time
                    duration_data.append(duration)
                    
                    frame_data.append(record.get('frame_number', 0))
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Source distribution pie chart
                    if source_counts:
                        fig_pie = px.pie(
                            values=list(source_counts.values()),
                            names=list(source_counts.keys()),
                            title="Records by Dataset Source"
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    # Duration histogram
                    if duration_data:
                        duration_df = pd.DataFrame({'Duration (seconds)': duration_data})
                        fig_hist = px.histogram(
                            duration_df,
                            x='Duration (seconds)',
                            title="Keyframe Duration Distribution",
                            nbins=20
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)
                
                # Summary statistics
                st.subheader("Summary Statistics")
                
                stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
                
                with stats_col1:
                    st.metric("Total Records", len(records))
                    st.metric("Unique Videos", len(set(r.get('video_id') for r in records)))
                
                with stats_col2:
                    if duration_data:
                        avg_duration = sum(duration_data) / len(duration_data)
                        st.metric("Avg Duration", f"{avg_duration:.1f}s")
                        st.metric("Max Duration", f"{max(duration_data):.1f}s")
                
                with stats_col3:
                    if frame_data:
                        st.metric("Avg Frame #", f"{sum(frame_data) / len(frame_data):.0f}")
                        st.metric("Max Frame #", f"{max(frame_data)}")
                
                with stats_col4:
                    ocr_count = sum(1 for r in records if r.get('extract_text'))
                    clip_count = sum(1 for r in records if r.get('clip_description'))
                    st.metric("OCR Coverage", f"{ocr_count}/{len(records)}")
                    st.metric("CLIP Coverage", f"{clip_count}/{len(records)}")

        with browser_tab3:
            st.subheader("Advanced Search & Filtering")
            
            st.markdown("""
            **Coming Soon:**
            - Semantic search across descriptions
            - Similar image search
            - Time range queries
            - Complex filtering combinations
            - Saved search queries
            """)

        with browser_tab4:
            st.subheader("Database Management Tools")
            
            st.markdown("""
            **Available Tools:**
            """)
            
            mgmt_col1, mgmt_col2 = st.columns(2)
            
            with mgmt_col1:
                st.markdown("**Export Options:**")
                if st.button("📊 Export Filtered Data as CSV"):
                    if browser_success and filtered_records:
                        df_export = pd.DataFrame([
                            {
                                "uuid": record.get("uuid"),
                                "video_id": record.get("video_id"),
                                "source": record.get("source"),
                                "frame_number": record.get("frame_number"),
                                "start_time": record.get("start_time"),
                                "end_time": record.get("end_time"),
                                "image_path": record.get("image_path"),
                                "description": record.get("description"),
                                "extract_text": record.get("extract_text"),
                                "clip_description": record.get("clip_description")
                            }
                            for record in filtered_records
                        ])
                        
                        csv = df_export.to_csv(index=False)
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name=f"keyframes_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.error("No data available for export")
            
            with mgmt_col2:
                st.markdown("**Database Operations:**")
                st.info("Management operations like delete, update, and batch processing will be available in future updates.")
    
    # Footer
    st.markdown("**Video-Text Retrieval System** | Powered by CLIP and Vector Search")

if __name__ == "__main__":
    main()
