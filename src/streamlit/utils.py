import os
import re
import tempfile
import time
import requests
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from docx import Document  # For handling .docx files
import streamlit as st

# ChromaDB API endpoint
CHROMADB_API = os.getenv("CHROMA_URL", "http://localhost:8020")
LLM_API = os.getenv("LLM_API", "http://localhost:9020")  # Add this

# Load Sentence Transformer model (multi-qa-mpnet-base-dot-v1 outputs 768-d vectors)
embedding_model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')

def fetch_collections():
    """Fetch collections from ChromaDB with caching"""
    try:
        chroma_url = CHROMADB_API
        response = requests.get(f"{chroma_url}/collections", timeout=10)
        if response.status_code == 200:
            collections_data = response.json()
            
            # Handle new response format with "collections" key
            if isinstance(collections_data, dict) and "collections" in collections_data:
                collections = collections_data["collections"]
                print(f"Found collections (new format): {collections}")
                return collections if isinstance(collections, list) else []
            # Handle old format (direct list) - fallback
            elif isinstance(collections_data, list):
                print(f"Found collections (old format): {collections_data}")
                return collections_data
            else:
                print(f"Unexpected response format: {collections_data}")
                return []
        else:
            print(f"Failed to fetch collections: {response.status_code}")
            print(f"Response text: {response.text}")
            return []
    except Exception as e:
        print(f"Error fetching collections: {e}")
        return []
def get_available_models():
    """Get available models from the API - this was missing!"""
    try:
        # Try the health endpoint first
        response = requests.get(f"{LLM_API}/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            # Check if models are in the health response
            if "models" in health_data:
                available_models = [model for model, status in health_data["models"].items() 
                                    if "available" in status.lower()]
                if available_models:
                    return available_models
        
        # Fallback: try legal-models endpoint
        response = requests.get(f"{LLM_API}/legal-models", timeout=10)
        if response.status_code == 200:
            legal_models_data = response.json()
            all_models = []
            if "legal_models" in legal_models_data:
                models_dict = legal_models_data["legal_models"]
                if "openai_models" in models_dict:
                    all_models.extend(models_dict["openai_models"])
                if "ollama_models" in models_dict:
                    all_models.extend(models_dict["ollama_models"])
                if all_models:
                    return all_models
                    
    except Exception as e:
        print(f"Error fetching models: {e}")
    
    # Ultimate fallback list of models
    return ["gpt-4", "gpt-3.5-turbo", "llama3"]

def check_model_availability():
    """Test which models are actually working - this was missing!"""
    available_models = []
    test_models = ["gpt-4", "gpt-3.5-turbo", "llama3"]
    
    for model in test_models:
        try:
            # Test with a simple request
            test_payload = {
                "query": "Hello",
                "model": model,
                "use_rag": False
            }
            response = requests.post(f"{LLM_API}/chat", json=test_payload, timeout=30)
            if response.status_code == 200:
                available_models.append(model)
                print(f"✅ {model} is available")
            else:
                print(f"❌ {model} failed: {response.status_code}")
        except Exception as e:
            print(f"❌ {model} error: {e}")
    
    return available_models

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_available_models_cached():
    return get_available_models()

def extract_sections_from_pdf(pdf_path):
    """
    Extracts full text from a PDF and splits it into sections based on headings.
    """
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"

    sections = re.split(r'\n(?=\d+\.\s)', full_text) 
    sections = [section.strip() for section in sections if section.strip()]
    return sections

def extract_text_from_txt(txt_path):
    """
    Extracts text from a .txt file.
    """
    with open(txt_path, 'r', encoding="utf-8") as file:
        text = file.read()
    
    sections = text.split("\n\n")  
    sections = [section.strip() for section in sections if section.strip()]
    return sections

def extract_text_from_docx(docx_path):
    """
    Extracts text from a .docx file.
    """
    doc = Document(docx_path)
    full_text = [para.text.strip() for para in doc.paragraphs if para.text.strip()]
    
    sections = re.split(r'\n(?=[A-Z])', "\n".join(full_text))
    sections = [section.strip() for section in sections if section.strip()]
    return sections

def store_files_in_chromadb(uploaded_files, collection_name, distance_metric="cosine"):
    """Stores documents (.pdf, .docx, .txt) in ChromaDB by extracting and embedding text sections."""
    # Ensure the collection exists
    requests.post(f"{CHROMADB_API}/collection/create", params={"collection_name": collection_name})

    for uploaded_file in uploaded_files:
        file_extension = uploaded_file.name.lower().split('.')[-1]
        temp_file_path = ""

        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name

        if file_extension == "pdf":
            sections = extract_sections_from_pdf(temp_file_path)
        elif file_extension == "txt":
            sections = extract_text_from_txt(temp_file_path)
        elif file_extension == "docx":
            sections = extract_text_from_docx(temp_file_path)
        else:
            print(f"Unsupported file type: {file_extension}")
            os.remove(temp_file_path)
            continue

        if not sections:
            print(f"No text extracted from {uploaded_file.name}. Skipping storage.")
            os.remove(temp_file_path)
            continue

        embeddings = embedding_model.encode(sections).tolist()
        print("Embedding dimension:", len(embeddings[0]))  # Should print 768

        # Prepare metadata and IDs
        metadatas = [{"document_name": uploaded_file.name} for _ in range(len(sections))]
        ids = [f"{uploaded_file.name}_section_{i}" for i in range(len(sections))]

        payload = {
            "collection_name": collection_name,
            "documents": sections,
            "ids": ids,
            "metadatas": metadatas,
            "embeddings": embeddings
        }
        response = requests.post(f"{CHROMADB_API}/documents/add", json=payload)

        # Cleanup temporary file
        os.remove(temp_file_path)

        if response.status_code == 200:
            print(f"Stored {uploaded_file.name} in collection '{collection_name}'.")
        else:
            print(f"Error storing {uploaded_file.name}: {response.json()}")

def list_all_chunks_with_scores(collection_name, query_text=None):
    """Lists all stored document sections in a ChromaDB collection with optional query scores."""
    # First, get all documents
    response = requests.get(f"{CHROMADB_API}/documents", params={"collection_name": collection_name})
    if response.status_code != 200:
        print(f"Error fetching documents: {response.text}")
        return []

    docs = response.json()
    if "ids" not in docs or "documents" not in docs:
        return []

    scores_dict = {}
    if query_text:
        query_embedding = embedding_model.encode([query_text]).tolist()
        query_payload = {
            "query_embeddings": query_embedding,
            "n_results": len(docs["ids"]),  # Get scores for all documents
            "include": ["metadatas", "distances"]
        }
        score_response = requests.post(
            f"{CHROMADB_API}/documents/query",
            json={"collection_name": collection_name, **query_payload}
        )
        if score_response.status_code == 200:
            score_results = score_response.json()
            scores_dict = {
                doc_id: round(distance, 4)
                for doc_id, distance in zip(score_results["ids"][0], score_results["distances"][0])
            }

    return [
        {
            "Chunk ID": f"`{doc_id}`",
            "Document": f">{doc_text[:250] + '...' if len(doc_text) > 250 else doc_text}",
            "Metadata": f"**{(metadata or {}).get('document_name', 'Unknown')}**",
            "Score": scores_dict.get(doc_id, "N/A")
        }
        for doc_id, doc_text, metadata in zip(
            docs["ids"],
            docs["documents"],
            docs.get("metadatas", [{}] * len(docs["ids"]))
        )
    ]