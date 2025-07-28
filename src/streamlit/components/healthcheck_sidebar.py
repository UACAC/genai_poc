import streamlit as st
import requests
from utils import *


FASTAPI_API = os.getenv("FASTAPI_URL", "http://localhost:9020")
HEALTH_ENDPOINT = f"{FASTAPI_API}/health"

# ----------------------------------------------------------------------
# SIDEBAR - SYSTEM STATUS & CONTROLS
# ----------------------------------------------------------------------

def healthcheck_sidebar():
    """Render the sidebar with system health check and collections"""
    
    # Initialize session state for health status
    if "health_status" not in st.session_state:
        st.session_state.health_status = None
    
    # Render sidebar
    st.sidebar.title("System Health")
    
    # Check if collections are loaded
    if "collections" not in st.session_state:
        st.session_state.collections = []

    # Render sidebar
    with st.sidebar:
        # Health check
        if st.button("Check Health"):
            try:
                with st.spinner("Checking health..."):
                    response = requests.get(HEALTH_ENDPOINT, timeout=10)
                    if response.ok:
                        st.session_state.health_status = response.json()
                        st.success("Online")
                    else:
                        st.error("System Issues")
            except Exception as e:
                st.error(f"Cannot connect to API: {e}")
    
        # Display cached health status
        if st.session_state.health_status:
            with st.expander("System Details"):
                st.json(st.session_state.health_status)

        st.header("Collections")
        
        if st.button("Load Collections"):
            try:
                # Load collections from both sources
                chromadb_collections = get_chromadb_collections()
                
                # Combine and deduplicate
                all_collections = list(set(chromadb_collections))
                st.session_state.collections = all_collections
                st.success("Collections loaded!")
            except Exception as e:
                st.error(f"Error: {e}")
        
        # Display collections
        collections = st.session_state.collections
        if collections:
            for collection in collections:
                st.text(f"{collection}")
        else:
            st.info("Click 'Load Collections' to see available databases")

        # Get collections for main interface
        try:
            if not st.session_state.collections:
                chromadb_collections = get_chromadb_collections()
                collections = list(set(chromadb_collections))
                st.session_state.collections = collections
            else:
                collections = st.session_state.collections
        except:
            collections = []