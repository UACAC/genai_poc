import os
import streamlit as st
import requests
from utils import *
from components.upload_documents import render_upload_component
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

@st.cache_data(show_spinner=False)
def fetch_doc_ids(collection_name):
    try:
        resp = requests.get(
            f"{CHROMADB_API}/documents",
            params={"collection_name": collection_name},
            timeout=5
        )
        resp.raise_for_status()
        docs = resp.json().get("documents", [])
        return [d.get("id") or d.get("document_id") for d in docs]
    except Exception:
        return []


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


    # # --- Document Upload ---
    # with upload_tab:
    #     st.subheader("Upload New Documents")
    #     # Pass unique prefix to avoid key collisions
    #     upload_tab_component = render_upload_component(
    #         available_collections=collections,
    #         load_collections_func=lambda: st.session_state.collections,
    #         create_collection_func=create_collection,
    #         upload_endpoint=f"{CHROMADB_API}/documents/upload-and-process",
    #         job_status_endpoint=f"{CHROMADB_API}/jobs/{{job_id}}",
    #         key_prefix="upload"
    #     )
    #     upload_tab_component
        
    #     st.markdown("---")
    #     st.subheader("Current Collections")
    #     if st.button("Refresh Collections", key="upload_refresh"):
    #         st.session_state.collections = fetch_collections()
    #         st.success(f"Found {len(st.session_state.collections)} collections")
    #     if collections:
    #         for i, col in enumerate(collections, 1):
    #             st.write(f"{i}. **{col}**")
    #     else:
    #         st.info("No collections found.")

    # --- Evaluate Document ---
    with eval_tab:
        st.header("Evaluate Document")
        use_rag_eval = st.checkbox(
            "Use RAG mode", value=True, key="eval_use_rag"
        )
        coll_name = None
        doc_id = None
        if use_rag_eval:
            coll_name = st.selectbox(
                "Select Collection:", collections, key="eval_coll"
            )
            if coll_name:
                doc_ids = fetch_doc_ids(coll_name)
                doc_id = st.selectbox(
                    "Select Document ID:", doc_ids, key="eval_doc"
                )
            # Inline upload if new doc
            st.markdown("**Or upload a new document to index below:**")
            render_upload_component(
                available_collections=collections,
                load_collections_func=lambda: st.session_state.collections,
                create_collection_func=create_collection,
                upload_endpoint=f"{CHROMADB_API}/documents/upload-and-process",
                job_status_endpoint=f"{CHROMADB_API}/jobs/{{job_id}}",
                key_prefix="eval"
            )
            if st.button("Refresh Collections", key="eval_refresh"):
                st.session_state.collections = fetch_collections()
                st.success(f"Found {len(st.session_state.collections)} collections")
        else:
            st.info("RAG disabled: evaluation will be pure LLM.")
            
        
        st.subheader("Evaluation Parameters")
        k = st.number_input(
            "Top K chunks:", min_value=1, value=5, key="eval_top_k"
        )
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
                if use_rag_eval:
                    endpoint = EVALUATE_ENDPOINT
                    payload = {
                        "document_id":     doc_id,
                        "collection_name": coll_name,
                        "prompt":          custom_prompt,
                        "top_k":           k,
                        "model_name":      model_key_map[mode2]
                    }
                else:
                    endpoint = CHAT_ENDPOINT
                    payload = {
                        "query": custom_prompt,
                        "model": model_key_map[mode2],
                        "use_rag": False,
                        "collection_name": None
                    }
                with st.spinner("Evaluating document..."):
                    try:
                        resp = requests.post(endpoint, json=payload, timeout=300)
                        if resp.ok:
                            res = resp.json()
                            answer = (
                                res.get("response")
                                or res.get("answer")
                                or res.get("response_text", "")
                            )
                            st.success("Evaluation Complete:")
                            st.markdown(answer)
                            if res.get("response_time_ms"):
                                rt = res["response_time_ms"]
                                st.caption(f"Response time: {rt/1000:.2f}s")
                        else:
                            st.error(f"Error {resp.status_code}: {resp.text}")
                    except Exception as e:
                        st.error(f"Request failed: {e}")
