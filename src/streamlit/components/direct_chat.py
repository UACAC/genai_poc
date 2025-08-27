import os
import streamlit as st
import requests
from utils import *
from components.upload_documents import render_upload_component, browse_documents
from components.history import Chat_History


# Constants
FASTAPI_API      = os.getenv("FASTAPI_URL", "http://localhost:9020")
CHROMADB_API     = os.getenv("CHROMA_URL", "http://localhost:8020")
CHAT_ENDPOINT    = f"{FASTAPI_API}/chat"
HISTORY_ENDPOINT = f"{FASTAPI_API}/chat-history"
EVALUATE_ENDPOINT = f"{FASTAPI_API}/evaluate_doc"

# Memoize document listings to avoid resetting on rerun
@st.cache_data(show_spinner=False)
def fetch_collections():
    return get_chromadb_collections()


def Direct_Chat():
    # Load collections once per session
    if "collections" not in st.session_state:
        st.session_state.collections = fetch_collections()

    collections = st.session_state.collections

    # Tabs
    chat_tab, eval_tab = st.tabs([
        "Chat Interface", "Evaluate Document"
    ])

    # --- Chat Interface ---
    with chat_tab:
        col1, col2 = st.columns([2, 1])
        with col1:
            mode = st.selectbox("Select AI Model:", list(model_key_map.keys()), key="chat_model")
            if model_key_map[mode] in model_descriptions:
                st.info(model_descriptions[model_key_map[mode]])

        use_rag = st.checkbox("Use RAG (Retrieval Augmented Generation)", key="chat_use_rag")
        collection_name = None
        if use_rag:
            if collections:
                collection_name = st.selectbox(
                    "Document Collection:", collections, key="chat_coll"
                )
            else:
                st.warning("No collections available. Upload docs first.")

        user_input = st.text_area(
            "Ask your question:", height=100,
            placeholder="e.g. Summarize the latest uploaded document"
        )

        if st.button("Get Analysis", type="primary", key="chat_button"):
            if not user_input:
                st.warning("Please enter a question.")
            elif use_rag and not collection_name:
                st.error("Please select a collection for RAG mode.")
            else:
                payload = {
                    "query": user_input,
                    "model": model_key_map[mode],
                    "use_rag": use_rag,
                    "collection_name": collection_name
                }
                with st.spinner(f"{mode} is analyzing..."):
                    try:
                        resp = requests.post(CHAT_ENDPOINT, json=payload, timeout=300)
                        if resp.ok:
                            data = resp.json()
                            result = data.get("response", "")
                            st.success("Analysis Complete:")
                            st.markdown(result)
                            if data.get("response_time_ms"):
                                rt = data["response_time_ms"]
                                st.caption(f"Response time: {rt/1000:.2f}s")
                        else:
                            detail = resp.json().get("detail", resp.text)
                            st.error(f"Error {resp.status_code}: {detail}")
                    except Exception as e:
                        st.error(f"Request failed: {e}")

        # Chat History
        Chat_History(key_prefix="chat_history")


    # --- Evaluate Document ---
    with eval_tab:
        st.header("Evaluate Document")
        use_rag_eval = st.checkbox(
            "Use RAG mode", value=True, key="eval_use_rag"
        )
        collections = st.session_state.collections
        if use_rag_eval:
            # Use the same document selection method as upload_documents.py
            browse_documents(key_prefix="select_eval_browse")
            
            with st.container(border=True, key="eval_params_container"):
                st.subheader("Evaluation Parameters")
                
                coll_name = st.selectbox(
                    "Select Collection:", collections, key="eval_collection"
                )

                # Enhanced document selection - user can now select from loaded documents
                doc_id = ""
                if "documents" in st.session_state and st.session_state.documents:
                    # Create options from loaded documents
                    doc_options = {}
                    for doc in st.session_state.documents:
                        doc_name = doc.get('document_name', 'Unknown')
                        doc_id_val = doc.get('id', doc.get('document_id', ''))
                        if doc_id_val:
                            display_name = f"{doc_name} (ID: {doc_id_val[:8]}...)"
                            doc_options[display_name] = doc_id_val
                    
                    if doc_options:
                        selected_display = st.selectbox(
                            "Select Document:",
                            options=list(doc_options.keys()),
                            key="eval_document_selector"
                        )
                        doc_id = doc_options[selected_display]
                        st.info(f"Selected Document ID: {doc_id}")
                    else:
                        st.warning("No documents found. Please load documents first using 'Load Documents' button above.")
                else:
                    st.info("Please load documents first using 'Load Documents' button above.")
                
                # Fallback manual entry
                manual_doc_id = st.text_input(
                    "Or enter Document ID manually:",
                    placeholder="e.g. 12345abcde",
                    key="manual_doc_id"
                )
                
                # Use manual entry if provided, otherwise use selected
                if manual_doc_id:
                    doc_id = manual_doc_id
                
                custom_prompt = st.text_area(
                    "Custom Prompt:", height=150, key="eval_prompt"
                )
                mode2 = st.selectbox(
                    "Select AI Model:", list(model_key_map.keys()), key="eval_model"
                )

                if st.button("Evaluate", type="primary", key="eval_button"):
                    if not custom_prompt:
                        st.warning("Please enter a prompt.")
                    elif use_rag_eval and (not coll_name or not doc_id):
                        st.error("Select both collection and document for RAG mode.")
                    else:
                        # pick endpoint & payload
                        if use_rag_eval:
                            endpoint = EVALUATE_ENDPOINT
                            payload = {
                                "document_id":     doc_id,
                                "collection_name": coll_name,
                                "prompt":          custom_prompt,
                                "top_k":           5,
                                "model_name":      model_key_map[mode2]
                            }
                        else:
                            endpoint = CHAT_ENDPOINT
                            payload = {
                                "query":           custom_prompt,
                                "model":           model_key_map[mode2],
                                "use_rag":         False,
                                "collection_name": None
                            }

                        with st.spinner("Evaluating document..."):
                            # 1) Fire the request
                            try:
                                resp = requests.post(endpoint, json=payload, timeout=300)
                            except requests.exceptions.RequestException as e:
                                st.error(f"Request failed: {e}")
                                # nothing else to do
                                return

                            # 2) Parse the response
                            if resp.ok:
                                data = resp.json()
                                answer     = data["response"]
                                rt_ms      = data["response_time_ms"]
                                session_id = data["session_id"]

                                st.success("Evaluation Complete:")
                                st.markdown(answer)
                                st.caption(f"Response time: {rt_ms/1000:.2f}s")
                                st.caption(f"Session ID: {session_id}")
                            else:
                                st.error(f"Error {resp.status_code}: {resp.text}")
                                
                                
            # chat history for evaluation
            Chat_History(key_prefix="eval_chat_history")

            # Render upload component for evaluation
            render_upload_component(
                available_collections=collections,
                load_collections_func=lambda: st.session_state.collections,
                create_collection_func=create_collection,
                upload_endpoint=f"{CHROMADB_API}/documents/upload-and-process",
                job_status_endpoint=f"{CHROMADB_API}/jobs/{{job_id}}",
                key_prefix="eval"
            )
            
        else:
            st.info("RAG disabled: evaluation will be pure LLM.")
            
        

