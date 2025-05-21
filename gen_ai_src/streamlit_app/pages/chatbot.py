import streamlit as st
import requests
import nest_asyncio
from utils import fetch_collections
import torch
torch.classes.__path__ = [] 

nest_asyncio.apply()

# FastAPI endpoints
HISTORY_ENDPOINT = "http://fastapi:9020/chat-history"
GPT4_API = "http://fastapi:9020/chat-gpt4"
LLAMA_API = "http://fastapi:9020/chat-llama"
MISTRAL_API = "http://fastapi:9020/chat-mistral"
GEMMA_API = "http://fastapi:9020/chat-gemma"
RAG_GPT4_API = "http://fastapi:9020/chat-gpt4-rag"
RAG_LLAMA_API = "http://fastapi:9020/chat-rag-llama"
RAG_MISTRAL_API = "http://fastapi:9020/chat-rag-mistral"
RAG_GEMMA_API = "http://fastapi:9020/chat-rag-gemma"
CHROMADB_API = "http://chromadb:8020"

st.set_page_config(page_title="Chatbot", layout="wide")
st.title("JITC GenAI Chatbot with RAG")

collections = fetch_collections()

mode = st.selectbox("Select Mode:", [
    "Direct GPT-4", "RAG (Chroma + GPT-4)", 
    "Direct LLaMA", "RAG (Chroma + LLaMA)", 
    "Direct Mistral", "RAG (Chroma + Mistral)", 
    "Direct Gemma", "RAG (Chroma + Gemma)"
])

collection_name = None
if "RAG" in mode and collections:
    collection_name = st.selectbox("Select a Vector Database Collection:", collections)

user_input = st.text_input("Ask me something:")

if st.button("Get Response"):
    if user_input:
        # Ensure valid input for RAG mode
        if "RAG" in mode and not collection_name:
            st.error("Please select a collection for RAG mode.")
        else:
            # Map mode selection to API endpoints
            api_map = {
                "Direct GPT-4": GPT4_API,
                "Direct LLaMA": LLAMA_API,
                "Direct Mistral": MISTRAL_API,
                "Direct Gemma": GEMMA_API,
                "RAG (Chroma + GPT-4)": RAG_GPT4_API,
                "RAG (Chroma + LLaMA)": RAG_LLAMA_API,
                "RAG (Chroma + Mistral)": RAG_MISTRAL_API,
                "RAG (Chroma + Gemma)": RAG_GEMMA_API
            }
            
            api_url = api_map.get(mode)
            payload = {"query": user_input}
            
            if "RAG" in mode:
                if not collection_name:
                    st.error("Please select a collection for RAG mode.")
                    st.stop()  
                payload["collection_name"] = collection_name 
                
            try:
                response = requests.post(api_url, json=payload)
                response_json = response.json()
                if response.status_code == 200:
                    st.success("Response:")
                    st.write(response_json.get("response", "No response received."))
                else:
                    st.error(f"Error {response.status_code}: {response_json.get('detail', response.text)}")
            except requests.exceptions.RequestException as e:
                st.error(f"Request failed: {e}")

# ---- LOAD CHAT HISTORY ----
st.header("Chat History")
if st.button("Load Chat History"):
    try:
        response = requests.get(HISTORY_ENDPOINT)
        if response.status_code == 200:
            data = response.json()
            if not data:
                st.info("No chat history found.")
            else:
                for record in data:
                    st.write(f"**User**: {record['user_query']}")
                    st.write(f"**Assistant**: {record['response']}")
                    st.write(f"_Timestamp_: {record['timestamp']}")
                    st.write("---")
        else:
            st.error(f"Failed to load chat history. Status: {response.status_code}")
    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {e}")