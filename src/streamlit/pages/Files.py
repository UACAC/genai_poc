import streamlit as st
import requests
from utils import *
import torch
import os
import base64
from io import BytesIO
from PIL import Image
import pandas as pd
from sentence_transformers import SentenceTransformer
import time
import re

torch.classes.__path__ = [] 

# This will resolve to the service discovery DNS name in AWS
CHROMADB_API = os.getenv("CHROMA_URL", "http://localhost:8020")

# Initialize embedding model for queries
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('multi-qa-mpnet-base-dot-v1')

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

def get_all_documents_in_collection(collection_name):
    """Get all documents in a collection with their metadata"""
    try:
        response = requests.get(
            f"{CHROMADB_API}/documents",
            params={"collection_name": collection_name},
            # timeout=300 # Increased timeout
        )
        if response.status_code == 200:
            data = response.json()
            
            # Group chunks by document_id to get unique documents
            documents = {}
            for i, doc_id in enumerate(data.get("ids", [])):
                metadata = data["metadatas"][i] if i < len(data.get("metadatas", [])) else {}
                document_id = metadata.get("document_id")
                document_name = metadata.get("document_name", "Unknown")
                
                if document_id and document_id not in documents:
                    documents[document_id] = {
                        "document_id": document_id,
                        "document_name": document_name,
                        "file_type": metadata.get("file_type", ""),
                        "total_chunks": metadata.get("total_chunks", 0),
                        "has_images": metadata.get("has_images", False),
                        "image_count": metadata.get("image_count", 0),
                        "processing_timestamp": metadata.get("timestamp", "")
                    }
            
            return list(documents.values())
        return []
    except Exception as e:
        st.error(f"Error fetching documents: {str(e)}")
        return []

def query_documents(collection_name, query_text, n_results=5):
    """Query documents using text search"""
    try:
        # Generate embedding for the query
        embedding_model = load_embedding_model()
        query_embedding = embedding_model.encode([query_text]).tolist()
        
        # Query ChromaDB
        response = requests.post(
            f"{CHROMADB_API}/documents/query",
            json={
                "collection_name": collection_name,
                "query_embeddings": query_embedding,
                "n_results": n_results,
                "include": ["documents", "metadatas", "distances"]
            },
            timeout=300
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Query failed: {response.text}")
            return None
            
    except Exception as e:
        st.error(f"Error querying documents: {str(e)}")
        return None
    
def start_ingest(files, collection_name, chunk_size, chunk_overlap,
                store_images, model_name, debug_mode, vision_models):
    # Build the multipart files payload
    payload = []
    for f in files:
        # f is a Streamlit UploadedFile
        # we need ( fieldname, (filename, bytes, content_type) )
        payload.append((
            "files",
            (f.name, f.read(), f.type)
        ))

    params = {
        "collection_name": collection_name,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "store_images": store_images,
        "model_name": model_name,
        "debug_mode": debug_mode,
        "vision_models": ",".join(vision_models),
    }

    resp = requests.post(
        f"{CHROMADB_API}/documents/upload-and-process",
        params=params,
        files=payload,
        timeout=60
    )
    resp.raise_for_status()
    return resp.json()["job_id"]


def wait_for_job(job_id, poll_interval=2):
    """Poll /jobs until status != pending/running."""
    while True:
        r = requests.get(f"{CHROMADB_API}/jobs/{job_id}", timeout=10)
        r.raise_for_status()
        status = r.json().get("status")
        if status in ("pending", "running"):
            time.sleep(poll_interval)
            continue
        return status

## -------------------------------------------------------- 
# Streamlit app configuration
# --------------------------------------------------------

st.set_page_config(page_title="Document Management", layout="wide")
st.title("Document & Collection Management")

# connection status 
with st.sidebar:
    st.subheader("Service Status")
    connected, health_data = test_chromadb_connection()
    
    if connected:
        st.success("ChromaDB Connected")
        
        # Show vision model status
        if health_data and "vision_models" in health_data:
            st.subheader("Vision Models")
            vision_models = health_data["vision_models"]
            for model, status in vision_models.items():
                icon = "✅" if status else "❌"
                st.write(f"{icon} {model.replace('_', ' ').title()}")
        
    else:
        st.error("ChromaDB Disconnected")
        st.info(f"Trying to connect to: {CHROMADB_API}")

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
    st.subheader("Create New Collection")
    new_collection = st.text_input("New Collection Name", placeholder="Enter collection name (e.g., 'my documents')")
    
    if new_collection:
        # Normalize collection name according to ChromaDB rules
        normalized_name = new_collection.strip().lower()
        # Replace spaces with underscores
        normalized_name = normalized_name.replace(' ', '_')
        # Keep only alphanumeric, underscores, and hyphens
        normalized_name = re.sub(r'[^a-zA-Z0-9_-]', '', normalized_name)
        # Ensure it starts and ends with alphanumeric
        normalized_name = re.sub(r'^[^a-zA-Z0-9]+', '', normalized_name)
        normalized_name = re.sub(r'[^a-zA-Z0-9]+$', '', normalized_name)
        
        # Ensure length is 3-63 characters
        if len(normalized_name) < 3:
            normalized_name = normalized_name + "_collection"
        elif len(normalized_name) > 63:
            normalized_name = normalized_name[:60] + "..."
            # Ensure it still ends with alphanumeric after truncation
            normalized_name = re.sub(r'[^a-zA-Z0-9]+$', '', normalized_name)
        
        # Show normalization if changed
        if normalized_name != new_collection:
            st.info(f"Collection name will be normalized to: `{normalized_name}`")
        
        # Validate final name
        if len(normalized_name) < 3:
            st.error("Collection name must be at least 3 characters after normalization")
            new_collection = None
        else:
            new_collection = normalized_name
        
    if st.button("Create Collection"):
        try:
            response = requests.post(
                f"{CHROMADB_API}/collection/create", 
                params={"collection_name": new_collection},
                timeout=60
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
        
        # Vision Model Selection
        st.subheader("Vision Model Configuration")
        
        # Get available models from health check
        try:
            health_response = requests.get(f"{CHROMADB_API}/health")
            if health_response.status_code == 200:
                health_data = health_response.json()
                vision_status = health_data.get("vision_models", {})
            else:
                vision_status = {}
        except:
            vision_status = {}
        
        # Model selection with status indicators
        col_a, col_b = st.columns(2)
        
        with col_a:
            openai_available = vision_status.get("openai_enabled", False)
            openai_icon = "✅" if openai_available else "❌"
            use_openai = st.checkbox(f"{openai_icon} OpenAI Vision", 
                                    value=openai_available, 
                                    disabled=not openai_available,
                                    help="High-quality semantic descriptions using GPT-4V")
            
            ollama_available = vision_status.get("ollama_enabled", False)
            ollama_icon = "✅" if ollama_available else "❌"
            use_ollama = st.checkbox(f"{ollama_icon} Ollama Vision", 
                                    value=False,
                                    disabled=not ollama_available,
                                    help="Local vision model (llava)")
        
        with col_b:
            hf_available = vision_status.get("huggingface_enabled", False)
            hf_icon = "✅" if hf_available else "❌"
            use_huggingface = st.checkbox(f"{hf_icon} HuggingFace BLIP", 
                                        value=hf_available,
                                        disabled=not hf_available,
                                        help="BLIP image captioning model")
            
            enhanced_available = vision_status.get("enhanced_local_enabled", False)
            enhanced_icon = "✅" if enhanced_available else "❌"
            use_enhanced = st.checkbox(f"{enhanced_icon} Enhanced Local", 
                                        value=True,
                                        help="OpenCV + OCR analysis with color/shape detection")
        

        
        # Build selected models list
        selected_models = []
        if use_openai:
            selected_models.append("openai")
        if use_ollama:
            selected_models.append("ollama")
        if use_huggingface:
            selected_models.append("huggingface")
        if use_enhanced:
            selected_models.append("enhanced_local")
        
        # always include basic analysis
        selected_models.append("basic")
        
        # Show selection summary
        if selected_models:
            st.info(f"**Selected models**: {', '.join(selected_models).title()}")
            
            # Estimate processing time
            time_estimate = len(selected_models) * 5  # Rough estimate: 5 seconds per model
            if "openai" in selected_models:
                time_estimate += 5  # OpenAI might be slower
            st.caption(f"Estimated processing time: ~{time_estimate} seconds per image")
        else:
            st.warning("Please select at least one vision model!")
        
        # Processing options
        store_images = st.checkbox("Store Images", value=True)
    
    with col2:
        chunk_size = st.number_input("Chunk Size", min_value=100, max_value=5000, value=1000)
        chunk_overlap = st.number_input("Chunk Overlap", min_value=0, max_value=500, value=200)
    
    if uploaded_files and st.button("Process & Store Documents", disabled=not selected_models):
        try:
            # 1) kick off ingestion
            job_id = start_ingest(
                uploaded_files,
                collection_name,
                chunk_size,
                chunk_overlap,
                store_images,
                selected_models,
                debug_mode,
                selected_models
            )
            st.info(f"Ingestion started")

            # 2) wait for it to finish
            with st.spinner("Uploading and processing in background…"):
                final_status = wait_for_job(job_id)

            # 3) check result
            if final_status == "success":
                st.success("Ingestion complete!")
                collections = fetch_collections()
            else:
                st.error(f"Ingestion failed: {final_status}")

        except Exception as e:
            st.error(f"Error starting ingestion: {e}")


            st.success(f"Documents stored in collection '{collection_name}'!")
            
else:
    st.warning("No collections exist. Please create one first.")
    
st.markdown("---")

# ---- QUERY DOCUMENTS ----
st.header("Query Documents")
if collections:
    query_collection = st.selectbox("Select Collection to Query", collections, key="query_collection")
    query_text = st.text_input("Enter your search query", placeholder="e.g., 'dog with tongue out' or 'red square with text'")
    n_results = st.slider("Number of results", min_value=1, max_value=20, value=5)
    
    if query_text and st.button("Search Documents"):
        with st.spinner("Searching..."):
            results = query_documents(query_collection, query_text, n_results)
            
            if results:
                st.success(f"Found {len(results['ids'][0])} results")
                
                # Display results
                for i, (doc_id, document, metadata, distance) in enumerate(zip(
                    results['ids'][0], 
                    results['documents'][0], 
                    results['metadatas'][0], 
                    results['distances'][0]
                )):
                    with st.expander(f"Result {i+1} - Score: {1-distance:.3f}"):
                        st.write(f"**Document**: {metadata.get('document_name', 'Unknown')}")
                        st.write(f"**Chunk**: {metadata.get('chunk_index', 0)} of {metadata.get('total_chunks', 0)}")
                        st.write(f"**Has Images**: {metadata.get('has_images', False)}")
                        if metadata.get('has_images'):
                            st.write(f"**Image Count**: {metadata.get('image_count', 0)}")
                        
                        st.text_area("Content", document, height=150, key=f"content_{i}")
                        
                        # Show document ID for easy reconstruction
                        st.code(f"Document ID: {metadata.get('document_id', 'Unknown')}")
st.markdown("---")

# ---- BROWSE DOCUMENTS ----
st.header("Browse Documents in Collection")
if collections:
    browse_collection = st.selectbox("Select Collection to Browse", collections, key="browse_collection")
    
    if st.button("Load Documents"):
        with st.spinner("Loading documents..."):
            documents = get_all_documents_in_collection(browse_collection)
            
            if documents:
                st.success(f"Found {len(documents)} documents")
                
                # Create a nice table
                df = pd.DataFrame(documents)
                
                # Show full table
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No documents found in this collection.")
                
st.markdown("---")
# ---- RECONSTRUCT DOCUMENTS ----
st.header("View Image Processed")
if collections:
    reconstruct_collection = st.selectbox("Select Collection", collections, key="reconstruct_collection")
    
    # Use selected document ID if available, otherwise latest uploaded
    default_doc_id = st.session_state.get('selected_doc_id', st.session_state.get('latest_doc_id', ""))
    
    document_id = st.text_input(
        "Document ID",
        placeholder="Enter the document ID to reconstruct (or select from Browse section above)",
        value=default_doc_id
    )
    
    if st.button("Reconstruct Document", disabled=not document_id):
        try:
            with st.spinner("Reconstructing document..."):
                response = requests.get(
                    f"{CHROMADB_API}/documents/reconstruct/{document_id}",
                    params={"collection_name": reconstruct_collection},
                    # timeout=300 
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

                # Build rich markdown with embedded images
                render_reconstructed_document(result)

                # ---- EXPORT DOCUMENTS ----
                with st.expander("Export Document"):
                    col1, col2 = st.columns(2)

                    with col1:
                        if st.button("Generate DOCX"):
                            with st.spinner("Generating DOCX..."):
                                docx_path = export_to_docx(result)
                                with open(docx_path, "rb") as f:
                                    st.download_button(
                                        label="Download DOCX",
                                        data=f,
                                        file_name=f"{result['document_name']}.docx",
                                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                                    )

                    with col2:
                        if st.button("Generate PDF"):
                            with st.spinner("Generating PDF..."):
                                pdf_path = export_to_pdf(result)
                                with open(pdf_path, "rb") as f:
                                    st.download_button(
                                        label="Download PDF",
                                        data=f,
                                        file_name=f"{result['document_name']}.pdf",
                                        mime="application/pdf"
                                    )

            elif response.status_code == 404:
                st.error("Document not found")
            else:
                st.error(f"Error: {response.text}")

        except requests.exceptions.Timeout:
            st.error("Request timed out. The document might be very large or the server is busy.")
        except Exception as e:
            st.error(f"Error reconstructing document: {str(e)}")
            


# ---- DEBUGGING SECTION ----
with st.sidebar:
    if st.checkbox("Show Debug Info"):
        st.subheader("Debug Information")
        st.write(f"**ChromaDB URL**: {CHROMADB_API}")
        
        if connected and health_data:
            with st.expander("Health Data"):
                st.json(health_data)
        
        if 'latest_doc_id' in st.session_state:
            st.write(f"**Latest Doc ID**: {st.session_state['latest_doc_id']}")
        
        if 'selected_doc_id' in st.session_state:
            st.write(f"**Selected Doc ID**: {st.session_state['selected_doc_id']}")