import streamlit as st
import requests
from utils import fetch_collections, store_files_in_chromadb, list_all_chunks_with_scores
import torch
torch.classes.__path__ = [] 


# FastAPI endpoint
CHROMADB_API = "http://chromadb:8020"

st.set_page_config(page_title="Document Management", layout="wide")

st.title("Document & Collection Management")

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
        response = requests.post(f"{CHROMADB_API}/collection/create", params={"collection_name": new_collection})
        if response.status_code == 200:
            st.success(f"Collection '{new_collection}' created!")
            st.rerun()
        else:
            st.error(response.json()["detail"])

# ---- DOCUMENT UPLOAD ----
st.header("Upload Documents to Vector Database (Chroma DB)")

if collections:
    collection_name = st.selectbox("Select Collection", collections)
    uploaded_files = st.file_uploader("Drop files here", type=["pdf", "docx", "txt"], accept_multiple_files=True)

    if uploaded_files and st.button("Process & Store Documents"):
        store_files_in_chromadb(uploaded_files, collection_name)
        st.success(f"Documents stored in collection '{collection_name}'!")
else:
    st.warning("No collections exist. Please create one first.")

# ---- VIEW DOCUMENTS ----
st.header("View Documents in a Collection")

selected_collection = st.selectbox("Select Collection to View Documents", fetch_collections())

if selected_collection and st.button("Fetch Documents"):
    chunks = list_all_chunks_with_scores(selected_collection)

    if chunks:
        st.table(chunks)
    else:
        st.write("No documents found in this collection.")

