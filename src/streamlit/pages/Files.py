import streamlit as st
import requests
from utils import fetch_collections, store_files_in_chromadb, list_all_chunks_with_scores
import torch
import os

torch.classes.__path__ = [] 

# This will resolve to the service discovery DNS name in AWS
CHROMADB_API = os.getenv("CHROMA_URL", "http://localhost:8020")

# Get OpenAI API key from environment
OPENAI_API_KEY = os.getenv("OPEN_AI_API_KEY")

# Available models
AVAILABLE_MODELS = {
    "gpt-4": "(Most capable)",
    "gpt-3.5-turbo": "(Fast and efficient)",
    "llama3": "(Open source)"
}

# Add connection debugging
def test_chromadb_connection():
    """Test ChromaDB connection and provide debugging info"""
    try:
        response = requests.get(f"{CHROMADB_API}/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            return True, health_data
        return False, None
    except requests.exceptions.ConnectTimeout:
        st.error(f"Connection timeout to ChromaDB at {CHROMADB_API}")
        return False, None
    except requests.exceptions.ConnectionError:
        st.error(f"Connection error to ChromaDB at {CHROMADB_API}")
        return False, None
    except Exception as e:
        st.error(f"Unexpected error connecting to ChromaDB: {str(e)}")
        return False, None

st.set_page_config(page_title="Document Management", layout="wide")
st.title("Document & Collection Management")

# connection status 
with st.sidebar:
    st.subheader("Service Status")
    connected, health_data = test_chromadb_connection()
    
    if connected:
        st.success("ChromaDB Connected")
        
    else:
        st.error("ChromaDB Disconnected")
        st.info(f"Trying to connect to: {CHROMADB_API}")
    
    # Model Selection in Sidebar
    st.subheader("Model Configuration")
    selected_model = st.selectbox(
        "Select a model",
        options=list(AVAILABLE_MODELS.keys()),
        format_func=lambda x: f"{x} - {AVAILABLE_MODELS[x]}"
    )

    # Warn if OpenAI model selected but key missing
    if selected_model in ["gpt-4", "gpt-3.5-turbo"] and not OPENAI_API_KEY:
        st.error("You selected an OpenAI model but did not provide `OPEN_AI_API_KEY` in your environment.")


# ---- COLLECTION MANAGEMENT ----
st.header("Manage Collections")
collections = fetch_collections()
col1, col2 = st.columns(2)


# List existing collections
with col1:
    st.subheader("Existing Collections")
    if collections:
        st.table({"Collections": collections})
    else:
        st.write("No collections available.")

# Create a new collection
with col2:
    new_collection = st.text_input("New Collection Name")
    if st.button("Create Collection"):
        try:
            response = requests.post(
                f"{CHROMADB_API}/collection/create", 
                params={"collection_name": new_collection},
                timeout=30  # Added timeout
            )
            if response.status_code == 200:
                st.success(f"Collection '{new_collection}' created!")
                st.rerun()
            else:
                st.error(response.json()["detail"])
        except requests.exceptions.ConnectTimeout:
            st.error(f"Failed to connect to ChromaDB at {CHROMADB_API}. Service may be starting up.")
        except Exception as e:
            st.error(f"Error creating collection: {str(e)}")

# ---- DOCUMENT UPLOAD ----
st.header("Upload Documents to Vector Database for AI Agent Retrieval")
if collections:
    collection_name = st.selectbox("Select Collection", collections)
    
    # File upload configuration
    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded_files = st.file_uploader(
            "Drop files here", 
            type=["pdf", "docx", "xlsx", "csv", "txt", "pptx", "html"], 
            accept_multiple_files=True
        )
    with col2:
        chunk_size = st.number_input("Chunk Size", min_value=100, max_value=5000, value=1000)
        chunk_overlap = st.number_input("Chunk Overlap", min_value=0, max_value=500, value=200)
        store_images = st.checkbox("Store Images", value=True)
    
    if uploaded_files and st.button("Process & Store Documents"):
        # Check if we can proceed with selected model
        can_proceed = True
        
        if can_proceed:
            try:
                with st.spinner("Processing documents..."):
                    # Pass model configuration to the processing function
                    store_files_in_chromadb(
                        uploaded_files, 
                        collection_name,
                        model_name=selected_model,
                        openai_api_key=OPENAI_API_KEY,  # Pass from environment
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        store_images=store_images
                    )
                st.success(f"Documents stored in collection '{collection_name}'!")
                
                # Show what was used
                with st.expander("Processing Details"):
                    st.write(f"**Model Used**: {selected_model}")
                    st.write(f"**Chunk Size**: {chunk_size}")
                    st.write(f"**Chunk Overlap**: {chunk_overlap}")
                    st.write(f"**Images Stored**: {'Yes' if store_images else 'No'}")
                    if selected_model in ["gpt-4", "gpt-3.5-turbo"]:
                        st.write(f"**OpenAI API**: {'Available' if OPENAI_API_KEY else 'Not Available'}")
                        
            except Exception as e:
                st.error(f"Error storing documents: {str(e)}")
                if "API key" in str(e):
                    st.info("💡 Tip: Make sure OPEN_AI_API_KEY is set in your environment variables")
else:
    st.warning("No collections exist. Please create one first.")

# ---- VIEW DOCUMENTS ----
st.header("View Documents in a Collection")
selected_collection = st.selectbox("Select Collection to View Documents", fetch_collections(), key="view_collection")
if selected_collection and st.button("Fetch Documents"):
    try:
        chunks = list_all_chunks_with_scores(selected_collection)
        if chunks:
            st.table(chunks)
        else:
            st.write("No documents found in this collection.")
    except Exception as e:
        st.error(f"Error fetching documents: {str(e)}")

# ---- RECONSTRUCT DOCUMENTS ----
st.header("Reconstruct Documents")
if collections:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        reconstruct_collection = st.selectbox("Select Collection", collections, key="reconstruct_collection")
        document_id = st.text_input("Document ID", placeholder="Enter the document ID to reconstruct")
    
    with col2:
        if st.button("Reconstruct Document", disabled=not document_id):
            try:
                response = requests.get(
                    f"{CHROMADB_API}/documents/reconstruct/{document_id}",
                    params={"collection_name": reconstruct_collection},
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    st.success(f"Document reconstructed: {result['document_name']}")
                    
                    # Show document info
                    with st.expander("Document Information"):
                        st.write(f"**Document ID**: {result['document_id']}")
                        st.write(f"**Document Name**: {result['document_name']}")
                        st.write(f"**Total Chunks**: {result['total_chunks']}")
                        st.write(f"**File Type**: {result['metadata']['file_type']}")
                        st.write(f"**Total Images**: {result['metadata']['total_images']}")
                    
                    # Show reconstructed content
                    st.text_area(
                        "Reconstructed Content", 
                        result['reconstructed_content'], 
                        height=400
                    )
                    
                    # Show images if any
                    if result.get('images'):
                        st.subheader("Associated Images")
                        for img in result['images']:
                            col1, col2 = st.columns([1, 3])
                            with col1:
                                if img['exists']:
                                    # You could add image display here if needed
                                    st.write(f"✅ {img['filename']}")
                                else:
                                    st.write(f"❌ {img['filename']}")
                            with col2:
                                st.write(img['description'])
                                
                elif response.status_code == 404:
                    st.error("Document not found")
                else:
                    st.error(f"Error: {response.text}")
                    
            except Exception as e:
                st.error(f"Error reconstructing document: {str(e)}")

# Footer with additional info
st.markdown("---")
st.caption("💡 **Tips**: ")
st.caption("- OpenAI models provide better image descriptions and document understanding")
st.caption("- Ollama models run locally and are good for privacy-sensitive content")
st.caption("- Adjust chunk size based on your document types (smaller for detailed analysis, larger for overview)")
st.caption("- Enable image storage for PDFs with important visual content")