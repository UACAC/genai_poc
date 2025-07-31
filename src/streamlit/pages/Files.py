import streamlit as st
import requests
import utils 
import torch
import os
from components.upload_documents import render_upload_component
from components.healthcheck_sidebar import Healthcheck_Sidebar
import pandas as pd
from sentence_transformers import SentenceTransformer
import time
import re

torch.classes.__path__ = [] 

# This will resolve to the service discovery DNS name in AWS
CHROMADB_API = os.getenv("CHROMA_URL", "http://localhost:8020")


if "collections" not in st.session_state:
    try:
        st.session_state.collections = utils.get_chromadb_collections()
    except Exception:
        st.session_state.collections = []

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
                store_images, model_name, vision_models):
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
    
# ----------------------------------------------------------------------
# SIDEBAR - SYSTEM STATUS & CONTROLS
# ----------------------------------------------------------------------
Healthcheck_Sidebar()

## -------------------------------------------------------- 
# Streamlit app configuration
# --------------------------------------------------------

st.set_page_config(page_title="Document Management", layout="wide")
st.title("Document & Collection Management")


### --------------------------------------------------------------------------
### Document Upload and Management
### --------------------------------------------------------------------------

collections = utils.get_chromadb_collections()

render_upload_component(
    available_collections= collections,
    load_collections_func= utils.get_chromadb_collections,
    create_collection_func= utils.create_collection,
    upload_endpoint=f"{CHROMADB_API}/documents/upload-and-process",
    job_status_endpoint=f"{CHROMADB_API}/jobs/{{job_id}}", 
    key_prefix="files_upload"
)