import streamlit as st
import requests
from utils import * 


FASTAPI_API      = os.getenv("FASTAPI_URL", "http://localhost:9020")
CHROMADB_API     = os.getenv("CHROMA_URL", "http://localhost:8020")
CHAT_ENDPOINT    = f"{FASTAPI_API}/chat"
HISTORY_ENDPOINT = f"{FASTAPI_API}/chat-history"
EVALUATE_ENDPOINT = f"{FASTAPI_API}/evaluate_doc"


def Chat_History(key_prefix: str = "",):  
    def pref(k): return f"{key_prefix}_{k}" if key_prefix else k
    with st.container(border=True, key=pref("history_container")):
        st.header("Chat History")
        if st.button("Load Chat History", key=pref("history_button")):
            try:
                with st.spinner("Loading..."):
                    hist = requests.get(HISTORY_ENDPOINT, timeout=10).json()
                if not hist:
                    st.info("No history found.")
                else:
                    for rec in reversed(hist[-10:]):
                        with st.expander(rec['timestamp'][:19]):
                            st.markdown(f"**User:** {rec['user_query']}")
                            st.markdown(f"**Response:** {rec['response']}")
            except Exception as e:
                st.error(f"Failed to load history: {e}")
                