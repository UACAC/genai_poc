import streamlit as st
import requests
import os
import time
import utils
import re
import pandas as pd

CHROMADB_API = os.getenv("CHROMA_URL", "http://localhost:8020")

def render_upload_component(
    available_collections: list[str],
    load_collections_func: callable,
    create_collection_func: callable,
    upload_endpoint: str,
    job_status_endpoint: str,
    key_prefix: str = "",
):
    """
    Renders a unified document upload & ingestion UI.

    Parameters:
    - available_collections: current list of collection names
    - load_collections_func: fn() -> list[str], repopulates collections
    - create_collection_func: fn(name: str) -> dict, creates a new collection
    - upload_endpoint: URL to POST files to
    - job_status_endpoint: URL template to poll job status, e.g. .../jobs/{job_id}
    """
    def pref(k): return f"{key_prefix}_{k}" if key_prefix else k
    
    with st.expander("Upload & Ingestion"):
        # # Refresh collections
        col_refresh, _ = st.columns([1, 3])
        with col_refresh:
            if st.button("Refresh Collections", key=pref("refresh_collections")):
                try:
                    with st.spinner("Loading collections..."):
                        new = load_collections_func()
                        st.session_state.collections = new
                        st.success("Collections updated")
                        st.dataframe(new, use_container_width=True)
                except Exception as e:
                    st.error(f"Failed to load collections: {e}")
        

        # Choose to use or create
        action = st.radio(
            "Collection Action:",
            ["Use Existing Collection", "Create New Collection"],
            horizontal=True,
            key=pref("create_or_use")
        )

        target_collection = None
        if action == "Use Existing Collection":
            if available_collections:
                target_collection = st.selectbox(
                    "Select Collection:",
                    available_collections,
                    key=pref("browse_collection")
                )
            else:
                st.warning("No collections available. Create one below.")
        else:
            new_name = st.text_input(
                "New Collection Name:",
                placeholder="e.g. standards, templates, policies"
            )
            
            if new_name:
                # Normalize collection name according to ChromaDB rules
                normalized_name = new_name.strip().lower()
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
                if normalized_name != new_name:
                    st.info(f"Collection name will be normalized to: `{normalized_name}`")
                
                # Validate final name
                if len(normalized_name) < 3:
                    st.error("Collection name must be at least 3 characters after normalization")
                    new_name = None
                else:
                    new_name = normalized_name
                    
            if new_name:
                if st.button("Create Collection"):
                    try:
                        with st.spinner(f"Creating '{new_name}'..."):
                            create_collection_func(new_name)
                            st.success(f"Created collection '{new_name}'")
                            target_collection = new_name
                    except Exception as e:
                        st.error(f"Failed to create collection: {e}")

        st.markdown("---")
        st.subheader("Upload Documents")
        uploaded = st.file_uploader(
            "Select files to upload:",
            type=['pdf','docx','txt','xlsx','pptx','html','csv'],
            accept_multiple_files=True,
            key=pref("files")
        )
        
        # Vision model selection
        st.subheader("Vision Models")
        openai_v = st.checkbox("OpenAI Vision", value=False, key=pref("openai_vision"))
        ollama_v = st.checkbox("Ollama Vision", value=False, key=pref("ollama_vision"))
        hf_v = st.checkbox("HuggingFace BLIP Vision", value=False, key=pref("hf_vision"))
        enhanced_v = st.checkbox("Enhanced Vision Model", value=False, key=pref("enhanced_vision"))
        basic_v = st.checkbox("Basic Vision Model", value=False, key=pref("basic_vision"))
        vision_models = []
        if openai_v: vision_models.append("openai")
        if ollama_v: vision_models.append("ollama")
        if hf_v: vision_models.append("huggingface")
        if enhanced_v: vision_models.append("enhanced_local")
        if basic_v: vision_models.append("basic")

        # Chunk settings
        st.subheader("Chunk Settings")
        chunk_size = st.number_input("Chunk Size", min_value=500, max_value=5000, value=1000, key=pref("chunk_size"))
        chunk_overlap = st.number_input("Chunk Overlap", min_value=0, max_value=chunk_size//2, value=200, key=pref("chunk_overlap"))

        if st.button("Upload and Process", key=pref("upload_button")):
            if not target_collection:
                st.error("Please select or create a collection first.")
            elif not uploaded:
                st.error("Please choose at least one file.")
            else:
                # prepare payload
                files = [("files", (f.name, f.getvalue(), f.type)) for f in uploaded]
                params = {
                    "collection_name": target_collection,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "store_images": True,
                    "model_name": "enhanced",
                    "vision_models": ",".join(vision_models)
                }

                with st.spinner("Starting upload…"):
                    try:
                        resp = requests.post(upload_endpoint, files=files, params=params, timeout=10)
                        resp.raise_for_status()
                        job_id = resp.json().get("job_id")
                    except Exception as e:
                        st.error(f"Upload failed: {e}")
                        return
        
                # Poll job status
                progress_bar = st.progress(0)
                status_text = st.empty()
                total, processed = 0, 0
                while True:
                    time.sleep(1)
                    status = requests.get(job_status_endpoint.format(job_id=job_id)).json()
                    total = status.get("total_chunks", 0)
                    processed = status.get("processed_chunks", 0)
                    if total:
                        pct = int(processed / total * 100)
                    else:
                        pct = 0
                    progress_bar.progress(min(pct, 100))
                    status_text.text(f"{processed}/{total} chunks processed")

                    if status.get("status") in ("success", "failed"):
                        break

                if status.get("status") == "success":
                    latest = status.get("latest_document_id")
                    if latest:
                        st.session_state.latest_doc_id = latest
                    st.success("Ingestion complete!")
                    st.session_state['latest_doc_id'] = status.get("latest_document_id", "")
                    collections = utils.get_chromadb_collections()
                    st.session_state['collections'] = collections
                
                else:
                    st.error("Ingestion failed.")
                    
    # ---- QUERY DOCUMENTS ----
    with st.expander("Query Documents"):
        if st.session_state.collections:
            query_collection = st.selectbox("Select Collection to Query", st.session_state.collections, key=pref("query_collection"))
            query_text = st.text_input("Enter your search query", placeholder="e.g., 'dog with tongue out' or 'red square with text'")
            n_results = st.slider("Number of results", min_value=1, max_value=20, value=5)
            
            if query_text and st.button("Search Documents", key=pref("search_documents")):
                with st.spinner("Searching..."):
                    results = utils.query_documents(query_collection, query_text, n_results)
                    
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
                
    # ---- BROWSE DOCUMENTS ----
    with st.expander("Browse Documents in Collection"):
    
        if st.session_state.collections:
            col = st.selectbox("Select Collection to Browse", st.session_state.collections)
            if st.button("Load Documents", key=pref("load_documents")):
                st.session_state.documents = utils.get_all_documents_in_collection(col)

            if "documents" in st.session_state:
                docs = st.session_state.documents
                if docs:
                    df = pd.DataFrame(docs)
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("No documents found in this collection.")
        else:
            st.warning("No collections available. Upload or create one first.")

    # ---- RECONSTRUCT DOCUMENTS ----
    with st.expander("View Image Processed"):
        if st.session_state.collections:
            reconstruct_collection = st.selectbox("Select Collection", st.session_state.collections, key=pref("reconstruct_collection"))
            
            # Use selected document ID if available, otherwise latest uploaded
            default_doc_id = st.session_state.get('selected_doc_id', st.session_state.get('latest_doc_id', ""))
            
            document_id = st.text_input(
                "Document ID",
                placeholder="Enter the document ID to reconstruct (or select from Browse section above)",
                value=default_doc_id
            )
            
            if st.button("Reconstruct Document", disabled=not document_id, key="reconstruct_document"):
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
                        utils.render_reconstructed_document(result)

                        # ---- EXPORT DOCUMENTS ----
                        with st.expander("Export Document"):
                            col1, col2 = st.columns(2)

                            with col1:
                                if st.button("Generate DOCX", key=pref("generate_docx")):
                                    with st.spinner("Generating DOCX..."):
                                        docx_path = utils.export_to_docx(result)
                                        with open(docx_path, "rb") as f:
                                            st.download_button(
                                                label="Download DOCX",
                                                data=f,
                                                file_name=f"{result['document_name']}.docx",
                                                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                                            )

                            with col2:
                                if st.button("Generate PDF", key=pref("generate_pdf")):
                                    with st.spinner("Generating PDF..."):
                                        pdf_path = utils.export_to_pdf(result)
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
