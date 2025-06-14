import streamlit as st
import requests
from utils import fetch_collections, store_files_in_chromadb, list_all_chunks_with_scores
import torch
import os

torch.classes.__path__ = [] 

# This will resolve to the service discovery DNS name in AWS
CHROMADB_API = os.getenv("CHROMA_URL", "http://localhost:8020")

# Add connection debugging
def test_chromadb_connection():
    """Test ChromaDB connection and provide debugging info"""
    try:
        response = requests.get(f"{CHROMADB_API}/health", timeout=10)
        return response.status_code == 200
    except requests.exceptions.ConnectTimeout:
        st.error(f"Connection timeout to ChromaDB at {CHROMADB_API}")
        return False
    except requests.exceptions.ConnectionError:
        st.error(f"Connection error to ChromaDB at {CHROMADB_API}")
        return False
    except Exception as e:
        st.error(f"Unexpected error connecting to ChromaDB: {str(e)}")
        return False

st.set_page_config(page_title="Document Management", layout="wide")
st.title("Document & Collection Management")

# connection status 
with st.sidebar:
    st.subheader("Service Status")
    if test_chromadb_connection():
        st.success("✅ ChromaDB Connected")
    else:
        st.error("❌ ChromaDB Disconnected")
        st.info(f"Trying to connect to: {CHROMADB_API}")

# ---- COLLECTION MANAGEMENT ----
st.header("Manage Collections")
collections = fetch_collections()
col1, col2 = st.columns(2)

# Debugging information in sidebar
st.sidebar.header("Debug Info")
try:
    # Test ChromaDB direct
    response = requests.get(f"{CHROMADB_API}/collections", timeout=5)
    st.sidebar.write(f"ChromaDB Status: {response.status_code}")
    st.sidebar.write(f"Raw Response: {response.text}")
    
    # Test fetch_collections function
    collections = fetch_collections()
    st.sidebar.write(f"Parsed Collections: {collections}")
except Exception as e:
    st.sidebar.error(f"Debug Error: {str(e)}")

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
    uploaded_files = st.file_uploader("Drop files here", type=["pdf", "docx", "txt"], accept_multiple_files=True)
    if uploaded_files and st.button("Process & Store Documents"):
        try:
            store_files_in_chromadb(uploaded_files, collection_name)
            st.success(f"Documents stored in collection '{collection_name}'!")
        except Exception as e:
            st.error(f"Error storing documents: {str(e)}")
else:
    st.warning("No collections exist. Please create one first.")

# ---- VIEW DOCUMENTS ----
st.header("View Documents in a Collection")
selected_collection = st.selectbox("Select Collection to View Documents", fetch_collections())
if selected_collection and st.button("Fetch Documents"):
    try:
        chunks = list_all_chunks_with_scores(selected_collection)
        if chunks:
            st.table(chunks)
        else:
            st.write("No documents found in this collection.")
    except Exception as e:
        st.error(f"Error fetching documents: {str(e)}")