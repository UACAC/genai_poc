import streamlit as st
import requests
from utils import *
from components.upload_documents import render_upload_component

def Direct_Chat():
    # Constants
    CHAT_ENDPOINT = "http://localhost:8000/chat"
    HISTORY_ENDPOINT = "http://localhost:8000/chat/history"
    CHROMADB_API = "http://localhost:8000/chromadb"
    
    # Display collections
    collections = st.session_state.collections
    if collections:
        for collection in collections:
            st.text(f"{collection}")
    else:
        st.info("Click 'Load Collections' to see available databases")
    
    # Create tabs for chat functionality
    chat_tab, upload_tab = st.tabs(["Chat Interface", "Document Upload"])
    
    with chat_tab:
        # Model selection (common for all modes)
        col1, col2 = st.columns([2, 1])
        with col1:
            mode = st.selectbox("Select AI Model:", list(model_key_map.keys()))
            if model_key_map[mode] in model_descriptions:
                st.info(model_descriptions[model_key_map[mode]])
                
        # RAG Configuration
        use_rag = st.checkbox("Use RAG (Retrieval Augmented Generation)")
        if use_rag and collections:
            collection_name = st.selectbox("Document Collection:", collections)
        else:
            collection_name = None
            
        if use_rag and not collections:
            st.warning("No collections available. Upload documents first or check your ChromaDB connection.")
        
        # Chat Input
        user_input = st.text_area(
            "Ask your question:", 
            height=100, 
            placeholder="Example: Analyze this document to determine if it meets our quality standards..."
        )

        if st.button("Get Analysis", type="primary"):
            if not user_input:
                st.warning("Please enter a question.")
            elif use_rag and not collection_name:
                st.error("Please select a collection for RAG mode.")
            else:
                payload = {
                    "query": user_input,
                    "model": model_key_map[mode],
                    "use_rag": use_rag,
                    "collection_name": collection_name if use_rag else None
                }

                with st.spinner(f"{mode} is analyzing your question..."):
                    status_placeholder = st.empty()
                    
                    try:
                        status_placeholder.info("Connecting to AI model...")
                        
                        response = requests.post(CHAT_ENDPOINT, json=payload, timeout=300)  
                        
                        if response.ok:
                            result = response.json().get("response", "")
                            status_placeholder.empty()
                            
                            st.success("Analysis Complete:")
                            st.markdown("### Analysis Results:")
                            st.markdown(result)
                            
                            if "response_time_ms" in response.json():
                                response_time = response.json()["response_time_ms"]
                                st.caption(f"Response time: {response_time/1000:.2f} seconds")
                                
                        else:
                            status_placeholder.empty()
                            error_detail = response.json().get("detail", response.text) if response.headers.get("content-type") == "application/json" else response.text
                            
                            if "model" in error_detail and "not found" in error_detail:
                                st.error("Model is loading for the first time. This may take 1-2 minutes. Please try again.")
                                st.info("Tip: The first request to each model takes longer as it loads into memory.")
                            else:
                                st.error(f"Error {response.status_code}: {error_detail}")
                                
                    except requests.exceptions.Timeout:
                        status_placeholder.empty()
                        st.error("Request timed out. The model might be loading - please try again in a moment.")
                        st.info("Large models can take 1-2 minutes to load on first use.")
                    except requests.exceptions.RequestException as e:
                        status_placeholder.empty()
                        st.error(f"Request failed: {e}")
        
        # Chat History Section
        st.markdown("---")
        st.header("Chat History")
        if st.button("Load Chat History"):
            try:
                with st.spinner("Loading chat history..."):
                    response = requests.get(HISTORY_ENDPOINT, timeout=10)
                    if response.ok:
                        history = response.json()
                        if not history:
                            st.info("No chat history found.")
                        else:
                            for record in reversed(history[-10:]):
                                with st.expander(f"{record['timestamp'][:19]}"):
                                    st.markdown(f"**User:** {record['user_query']}")
                                    st.markdown(f"**Response:** {record['response']}")
                    else:
                        st.error(f"Failed to fetch history: {response.status_code}")
            except requests.exceptions.RequestException as e:
                st.error(f"Request failed: {e}")
    
    with upload_tab:
        collections = get_chromadb_collections()
        render_upload_component(
            available_collections= collections,
            load_collections_func= get_chromadb_collections,
            create_collection_func= create_collection,
            upload_endpoint=f"{CHROMADB_API}/documents/upload-and-process",
            job_status_endpoint=f"{CHROMADB_API}/jobs/{{job_id}}"
        )
        
        # Current Collections Status
        st.markdown("---")
        st.subheader("Current Collections")
        
        if st.button("Refresh Collections"):
            try:
                chromadb_collections = get_chromadb_collections()
                st.session_state.collections = chromadb_collections
                st.success(f"Found {len(chromadb_collections)} collections")
            except Exception as e:
                st.error(f"Error refreshing collections: {e}")
        
        if collections:
            for i, collection in enumerate(collections, 1):
                st.write(f"{i}. **{collection}**")
        else:
            st.info("No collections found. Create your first collection to get started!")