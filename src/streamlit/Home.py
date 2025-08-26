import streamlit as st
# import requests
import os
import nest_asyncio
# import datetime
from utils import * 
import torch
# import base64
# from components.upload_documents import render_upload_component
from components.healthcheck_sidebar import Healthcheck_Sidebar
from components.direct_chat import Direct_Chat
from components.agent_sim import Agent_Sim
from components.ai_agent import AI_Agent
from components.document_generator import Document_Generator
from components.rag_assessment import RAGAS_Dashboard
from components.session_history import Session_History

torch.classes.__path__ = []
nest_asyncio.apply()

#  API endpoints
FASTAPI_API = os.getenv("FASTAPI_URL", "http://localhost:9020")
CHROMADB_API = os.getenv("CHROMA_URL", "http://localhost:8020") 
CHAT_ENDPOINT = f"{FASTAPI_API}/chat"
HISTORY_ENDPOINT = f"{FASTAPI_API}/chat-history"
HEALTH_ENDPOINT = f"{FASTAPI_API}/health"
OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")

# THIS MUST BE THE VERY FIRST STREAMLIT COMMAND
st.set_page_config(page_title="AI Assistant", layout="wide")

st.title("AI Assistant")

# Initialize session state
if 'health_status' not in st.session_state:
    st.session_state.health_status = None
if 'available_models' not in st.session_state:
    st.session_state.available_models = []
if 'collections' not in st.session_state:
    st.session_state.collections = []
if 'agents_data' not in st.session_state:
    st.session_state.agents_data = []
if 'debate_sequence' not in st.session_state:
    st.session_state.debate_sequence = []
if 'upload_progress' not in st.session_state:
    st.session_state.upload_progress = {}

# ----------------------------------------------------------------------
# SIDEBAR - SYSTEM STATUS & CONTROLS
# ----------------------------------------------------------------------
Healthcheck_Sidebar()

# ----------------------------------------------------------------------
# MAIN INTERFACE
# ----------------------------------------------------------------------
# Chat mode selection
chat_mode = st.radio(
    "Select Mode:",
    ["Direct Chat", "AI Agent Simulation", "AI Agents", "Document Generator", "RAG Assessment", "Session History"],
    horizontal=True
)

# ----------------------------------------------------------------------
# DIRECT CHAT MODE
# ----------------------------------------------------------------------
if chat_mode == "Direct Chat":
    st.markdown("---")
    Direct_Chat()

# ----------------------------------------------------------------------
# AI AGENT SIMULATION MODE
# ----------------------------------------------------------------------
elif chat_mode == "AI Agent Simulation":
    st.markdown("---")
    Agent_Sim()

# ----------------------------------------------------------------------
# CREATE AGENT MODE (WITH MANAGEMENT SUB-MODES)
# ----------------------------------------------------------------------
elif chat_mode == "AI Agents":
    st.markdown("---")
    AI_Agent()
    
    # Footer for create agent section
    st.warning("**Agent Disclaimer**: All created agents provide analysis for informational purposes only and do not constitute advice.")
    st.info("**Data Security**: Ensure all content processed by agents complies with your organization's data protection and confidentiality policies.")
    
# ----------------------------------------------------------------------
# DOCUMENT GENERATOR MODE
# ----------------------------------------------------------------------
elif chat_mode == "Document Generator":
    st.markdown("---")
    Document_Generator()

# ----------------------------------------------------------------------
# RAG ASSESSMENT MODE
# ----------------------------------------------------------------------
elif chat_mode == "RAG Assessment":
    st.markdown("---")
    RAGAS_Dashboard()

# ----------------------------------------------------------------------
# SESSION HISTORY & ANALYTICS MODE
# ----------------------------------------------------------------------
elif chat_mode == "Session History":
    st.markdown("---")
    Session_History()
    


# Footer
st.markdown("---")
st.caption("This application processes documents and provide GenAI capabilitites. Ensure all data is handled according to your organization's data protection policies.")
