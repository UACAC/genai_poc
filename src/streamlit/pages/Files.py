import streamlit as st
import requests
from utils import fetch_collections, store_files_in_chromadb, list_all_chunks_with_scores, export_to_docx, export_to_pdf, image_to_base64, render_reconstructed_document
import torch
import os
import base64
from io import BytesIO
from PIL import Image

torch.classes.__path__ = [] 

# This will resolve to the service discovery DNS name in AWS
CHROMADB_API = os.getenv("CHROMA_URL", "http://localhost:8020")


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
        store_images = st.checkbox("Store Images", value=True)
    with col2:
        chunk_size = st.number_input("Chunk Size", min_value=100, max_value=5000, value=1000)
        chunk_overlap = st.number_input("Chunk Overlap", min_value=0, max_value=500, value=200)
    
    if uploaded_files and st.button("Process & Store Documents"):
        # Check if we can proceed with selected model
        can_proceed = True
        
        if can_proceed:
            try:
                with st.spinner("Processing documents..."):
                    # Pass model configuration to the processing function
                    
                    results = store_files_in_chromadb(
                        uploaded_files, 
                        collection_name,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        store_images=store_images
                    )

                    st.success(f"Documents stored in collection '{collection_name}'!")
                    with st.expander("Uploaded Document Details"):
                        for doc in results.get("processed_files", []):
                            if doc.get("status") == "success":
                                st.markdown(f"""
                                **{doc['filename']}**  
                                - Status: {doc['status']}  
                                - Document ID: `{doc['document_id']}`  
                                - Chunks: {doc['chunks_created']}  
                                - Images Stored: {doc['images_stored']}
                                """)
                            else:
                                st.warning(f"{doc['filename']} — {doc.get('error', 'Unknown error')}")
                        
                        st.session_state['latest_doc_id'] = doc['document_id']

                st.success(f"Documents stored in collection '{collection_name}'!")
                
                # Show what was used
                with st.expander("Processing Details"):
                    st.write(f"**Chunk Size**: {chunk_size}")
                    st.write(f"**Chunk Overlap**: {chunk_overlap}")
                    st.write(f"**Images Stored**: {'Yes' if store_images else 'No'}")
                        
            except Exception as e:
                st.error(f"Error storing documents: {str(e)}")
                
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
    reconstruct_collection = st.selectbox("Select Collection", collections, key="reconstruct_collection")
    document_id = st.text_input(
        "Document ID",
        placeholder="Enter the document ID to reconstruct",
        value=st.session_state.get('latest_doc_id', "")
    )
    
    
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

                def image_to_base64(img_url):
                    try:
                        response = requests.get(img_url)
                        img = Image.open(BytesIO(response.content))
                        buffered = BytesIO()
                        img.save(buffered, format="PNG")
                        img_base64 = base64.b64encode(buffered.getvalue()).decode()
                        return f"![img](data:image/png;base64,{img_base64})"
                    except:
                        return ""

                # Build rich markdown with embedded images
                render_reconstructed_document(result)

                # ---- EXPORT DOCUMENTS ----
                with st.expander("Export Document"):
                    col1, col2 = st.columns(2)

                    with col1:
                        docx_path = export_to_docx(result)
                        with open(docx_path, "rb") as f:
                            st.download_button(
                                label="Download DOCX",
                                data=f,
                                file_name=f"{result['document_name']}.docx",
                                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                            )

                    with col2:
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

        except Exception as e:
            st.error(f"Error reconstructing document: {str(e)}")
