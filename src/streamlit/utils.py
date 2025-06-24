import os
import re
import tempfile
import time
import requests
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from docx import Document
from docx.shared import Inches
from fpdf import FPDF  
import streamlit as st
import base64
from io import BytesIO
from PIL import Image
import datetime


# ChromaDB API endpoint
CHROMADB_API = os.getenv("CHROMA_URL", "http://localhost:8020")
LLM_API = os.getenv("LLM_API", "http://localhost:9020")

# Load Sentence Transformer model
embedding_model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')

def fetch_collections():
    try:
        response = requests.get(f"{CHROMADB_API}/collections", timeout=10)
        response.raise_for_status()
        return response.json().get("collections", [])
    except requests.exceptions.RequestException as e:
        print(f"Fetch collections failed: {e}")
        return []

def get_available_models():
    try:
        response = requests.get(f"{LLM_API}/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            if "models" in health_data:
                return [model for model, status in health_data["models"].items() if "available" in status.lower()]
    except Exception as e:
        print(f"Error fetching models: {e}")
    return ["gpt-4", "gpt-3.5-turbo", "llama3"]

def check_model_availability():
    available_models = []
    test_models = ["gpt-4", "gpt-3.5-turbo", "llama3"]
    for model in test_models:
        try:
            response = requests.post(f"{LLM_API}/chat", json={"query": "Hello", "model": model, "use_rag": False}, timeout=30)
            if response.status_code == 200:
                available_models.append(model)
        except Exception as e:
            print(f"{model} error: {e}")
    return available_models

@st.cache_data(ttl=300)
def get_available_models_cached():
    return get_available_models()

def extract_sections_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    full_text = "".join([page.extract_text() or "" for page in reader.pages])
    return [s.strip() for s in re.split(r'\n(?=\d+\.\s)', full_text) if s.strip()]

def extract_text_from_txt(txt_path):
    with open(txt_path, 'r', encoding="utf-8") as file:
        return [s.strip() for s in file.read().split("\n\n") if s.strip()]

def extract_text_from_docx(docx_path):
    doc = Document(docx_path)
    return [s.strip() for s in re.split(r'\n(?=[A-Z])', "\n".join([p.text.strip() for p in doc.paragraphs if p.text.strip()])) if s.strip()]

def store_files_in_chromadb(files, collection_name, model_name="none", openai_api_key=None, 
                            chunk_size=1000, chunk_overlap=200, store_images=True, 
                            debug_mode=False, run_all_vision_models=True):
    """Store uploaded files in ChromaDB with enhanced parameters"""
    files_data = [('files', (file.name, file.getvalue(), file.type)) for file in files]
    
    params = {
        "collection_name": collection_name,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "store_images": store_images,
        "model_name": model_name,
        "debug_mode": debug_mode,
        "run_all_vision_models": run_all_vision_models  # Add new parameter
    }
    
    # Prepare headers
    headers = {}
    if openai_api_key and model_name in ["gpt-4", "gpt-3.5-turbo"]:
        headers["X-OpenAI-API-Key"] = openai_api_key
    
    try:
        response = requests.post(
            f"{CHROMADB_API}/documents/upload-and-process", 
            params=params, 
            files=files_data, 
            headers=headers, 
            timeout=600  # Increased timeout for multi-model processing
        )
        
        if response.status_code != 200:
            raise Exception(f"Failed to store documents: {response.status_code} - {response.text}")
        
        return response.json()
        
    except requests.exceptions.Timeout:
        raise Exception("Request timed out. Multi-model processing takes longer but provides comprehensive analysis.")
    except requests.exceptions.ConnectionError:
        raise Exception(f"Could not connect to ChromaDB at {CHROMADB_API}")
    except Exception as e:
        raise Exception(f"Error storing files: {str(e)}")
    
def store_files_in_chromadb_selective(files, collection_name, model_name="none", openai_api_key=None, 
                                    chunk_size=1000, chunk_overlap=200, store_images=True, 
                                    debug_mode=False, selected_models=None):
    """Store uploaded files in ChromaDB with selective vision models"""
    files_data = [('files', (file.name, file.getvalue(), file.type)) for file in files]
    
    # Default to enhanced + basic if no models selected
    if not selected_models:
        selected_models = ["enhanced_local", "basic"]
    
    # Convert list to comma-separated string
    vision_models_str = ",".join(selected_models)
    
    params = {
        "collection_name": collection_name,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "store_images": store_images,
        "model_name": model_name,
        "debug_mode": debug_mode,
        "vision_models": vision_models_str  # Pass selected models
    }
    
    # Prepare headers
    headers = {}
    if openai_api_key and model_name in ["gpt-4", "gpt-3.5-turbo"]:
        headers["X-OpenAI-API-Key"] = openai_api_key
    
    try:
        # Adjust timeout based on number of models selected
        base_timeout = 300
        model_timeout = len(selected_models) * 60  # 1 minute per model
        total_timeout = min(base_timeout + model_timeout, 900)  # Max 15 minutes
        
        response = requests.post(
            f"{CHROMADB_API}/documents/upload-and-process", 
            params=params, 
            files=files_data, 
            headers=headers, 
            timeout=total_timeout
        )
        
        if response.status_code != 200:
            raise Exception(f"Failed to store documents: {response.status_code} - {response.text}")
        
        return response.json()
        
    except requests.exceptions.Timeout:
        raise Exception(f"Request timed out. Processing {len(selected_models)} vision models takes significant time.")
    except requests.exceptions.ConnectionError:
        raise Exception(f"Could not connect to ChromaDB at {CHROMADB_API}")
    except Exception as e:
        raise Exception(f"Error storing files: {str(e)}")

def query_documents_with_embedding(collection_name, query_text, n_results=5):
    """Query documents using text embedding"""
    try:
        # Generate embedding for the query
        query_embedding = embedding_model.encode([query_text]).tolist()
        
        # Query ChromaDB
        response = requests.post(
            f"{CHROMADB_API}/documents/query",
            json={
                "collection_name": collection_name,
                "query_embeddings": query_embedding,
                "n_results": n_results,
                "include": ["documents", "metadatas", "distances"]
            },
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Query failed: {response.text}")
            
    except Exception as e:
        raise Exception(f"Error querying documents: {str(e)}")

def get_all_documents_in_collection(collection_name):
    """Get all unique documents in a collection with their metadata"""
    try:
        response = requests.get(
            f"{CHROMADB_API}/documents",
            params={"collection_name": collection_name},
            timeout=60
        )
        
        if response.status_code != 200:
            raise Exception(f"Failed to fetch documents: {response.text}")
            
        data = response.json()
        
        # Group chunks by document_id to get unique documents
        documents = {}
        for i, doc_id in enumerate(data.get("ids", [])):
            metadata = data["metadatas"][i] if i < len(data.get("metadatas", [])) else {}
            document_id = metadata.get("document_id")
            document_name = metadata.get("document_name", "Unknown")
            
            if document_id and document_id not in documents:
                documents[document_id] = {
                    "document_id": document_id,
                    "document_name": document_name,
                    "file_type": metadata.get("file_type", ""),
                    "total_chunks": metadata.get("total_chunks", 0),
                    "has_images": metadata.get("has_images", False),
                    "image_count": metadata.get("image_count", 0),
                    "processing_timestamp": metadata.get("timestamp", ""),
                    "openai_api_used": metadata.get("openai_api_used", False)
                }
        
        return list(documents.values())
        
    except Exception as e:
        raise Exception(f"Error fetching documents: {str(e)}")

def list_all_chunks_with_scores(collection_name, query_text=None):
    """List all chunks with optional similarity scores"""
    response = requests.get(f"{CHROMADB_API}/documents", params={"collection_name": collection_name})
    if response.status_code != 200:
        return []
    
    docs = response.json()
    scores_dict = {}
    
    if query_text:
        query_embedding = embedding_model.encode([query_text]).tolist()
        score_response = requests.post(
            f"{CHROMADB_API}/documents/query", 
            json={
                "collection_name": collection_name, 
                "query_embeddings": query_embedding, 
                "n_results": len(docs["ids"]), 
                "include": ["metadatas", "distances"]
            }
        )
        if score_response.status_code == 200:
            results = score_response.json()
            scores_dict = {doc_id: round(distance, 4) for doc_id, distance in zip(results["ids"][0], results["distances"][0])}
    
    return [
        {
            "Collection": collection_name, 
            "Document Name": metadata.get("document_name", "Unknown"), 
            "Document ID": doc_id, 
            "Chunk ID": f"`{doc_id}`", 
            "Document Text": f">{doc_text[:250] + '...' if len(doc_text) > 250 else doc_text}", 
            "Metadata": f"**{(metadata or {}).get('document_name', 'Unknown')}**", 
            "Score": scores_dict.get(doc_id, "N/A"),
            "Has Images": metadata.get("has_images", False),
            "Image Count": metadata.get("image_count", 0)
        } 
        for doc_id, doc_text, metadata in zip(
            docs["ids"], 
            docs["documents"], 
            docs.get("metadatas", [{}] * len(docs["ids"]))
        ) 
    ]

def reconstruct_document_with_timeout(document_id, collection_name, timeout=300):
    """Reconstruct document with configurable timeout"""
    try:
        response = requests.get(
            f"{CHROMADB_API}/documents/reconstruct/{document_id}",
            params={"collection_name": collection_name},
            timeout=timeout
        )
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            raise Exception("Document not found")
        else:
            raise Exception(f"Error: {response.text}")
            
    except requests.exceptions.Timeout:
        raise Exception("Request timed out. The document might be very large or the server is busy.")
    except Exception as e:
        raise Exception(f"Error reconstructing document: {str(e)}")

def image_to_base64(img_url):
    """Convert image URL to base64 for embedding in markdown"""
    try:
        response = requests.get(img_url, timeout=30)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            return f"![img](data:image/png;base64,{img_base64})"
        else:
            return f"Image not found: {img_url}"
    except Exception as e:
        return f"Failed to load image: {img_url} — {str(e)}"

def get_image_from_chromadb(filename):
    """Get image from ChromaDB storage"""
    try:
        img_url = f"{CHROMADB_API}/images/{filename}"
        response = requests.get(img_url, timeout=30)
        
        if response.status_code == 200:
            return Image.open(BytesIO(response.content))
        else:
            return None
    except Exception as e:
        print(f"Error fetching image {filename}: {e}")
        return None

def render_reconstructed_document(result: dict):
    """Render reconstructed document with enhanced image handling"""
    rich_content = result.get("reconstructed_content", "")
    images = result.get("images", [])

    # Replace image markers with base64 images + description
    for image in images:
        marker = f"[IMAGE:{image['filename']}]"
        img_url = f"{CHROMADB_API}/images/{image['filename']}"
        base64_img = image_to_base64(img_url)
        description = image.get("description", "")
        
        # Enhanced replacement with better formatting
        replacement = f"""
{base64_img}

**Image Analysis:** {description}

---
"""
        rich_content = rich_content.replace(marker, replacement)

    # Display the content
    st.markdown(rich_content, unsafe_allow_html=True)

def clean_text(text):
    """Remove null bytes and non-XML-safe characters"""
    if not text:
        return ""
    return ''.join(c for c in text if c == '\n' or c == '\t' or (32 <= ord(c) <= 126))

def export_to_docx(result):
    """Export document to DOCX with enhanced image handling"""
    doc = Document()
    doc.add_heading(result["document_name"], 0)

    rich_content = result["reconstructed_content"]
    images = result.get("images", [])

    # Create image lookup for descriptions
    image_lookup = {img['filename']: img for img in images}

    for img in images:
        marker = f"[IMAGE:{img['filename']}]"
        if marker in rich_content:
            rich_content = rich_content.replace(marker, f"@@IMAGE:{img['filename']}@@")

    chunks = re.split(r'(@@IMAGE:.+?@@)', rich_content)

    for chunk in chunks:
        img_match = re.match(r'@@IMAGE:(.+?)@@', chunk)
        if img_match:
            filename = img_match.group(1)
            
            # Try to get and insert the image
            image_obj = get_image_from_chromadb(filename)
            if image_obj:
                try:
                    # Save image to temporary file for docx
                    temp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                    image_obj.save(temp_img.name)
                    
                    # Add image to document
                    doc.add_picture(temp_img.name, width=Inches(5.5))
                    
                    # Add image description if available
                    if filename in image_lookup:
                        desc = image_lookup[filename].get('description', '')
                        if desc:
                            doc.add_paragraph(clean_text(desc), style='Intense Quote')
                    
                    # Clean up temp file
                    os.unlink(temp_img.name)
                    
                except Exception as e:
                    doc.add_paragraph(f"[Image '{filename}' could not be inserted: {e}]")
            else:
                doc.add_paragraph(f"[Image '{filename}' not found]")
        else:
            cleaned = clean_text(chunk.strip())
            if cleaned:
                doc.add_paragraph(cleaned)

    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".docx")
    doc.save(temp_file.name)
    return temp_file.name

def export_to_pdf(result):
    """Export document to PDF with enhanced image handling"""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    content = result["reconstructed_content"]
    images = result.get("images", [])
    image_map = {img["filename"]: img["description"] for img in images}

    # Split content by image markers
    parts = content.split("[IMAGE:")
    
    # Add first part (before any images)
    if parts[0].strip():
        pdf.multi_cell(0, 10, clean_text(parts[0].strip()))

    # Process each part that contains an image
    for part in parts[1:]:
        if "]" in part:
            filename, rest = part.split("]", 1)
            
            # Try to get and insert the image
            image_obj = get_image_from_chromadb(filename)
            if image_obj:
                try:
                    # Save image to temporary file
                    temp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                    image_obj.save(temp_img.name)

                    # Add new page for image
                    pdf.add_page()
                    pdf.image(temp_img.name, w=150)
                    pdf.ln(5)
                    
                    # Add image description
                    if filename in image_map:
                        pdf.multi_cell(0, 10, clean_text(image_map[filename]))
                    
                    # Clean up temp file
                    os.unlink(temp_img.name)
                    
                except Exception as e:
                    pdf.multi_cell(0, 10, f"[Failed to insert image {filename}]: {e}")
            else:
                pdf.multi_cell(0, 10, f"[Image {filename} not found]")

            # Add remaining text after image
            if rest.strip():
                pdf.multi_cell(0, 10, clean_text(rest.strip()))
        else:
            # No image marker found, just add text
            if part.strip():
                pdf.multi_cell(0, 10, clean_text(part.strip()))

    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(temp_file.name)
    return temp_file.name

def check_chromadb_health():
    """Check ChromaDB health and return status info"""
    try:
        response = requests.get(f"{CHROMADB_API}/health", timeout=10)
        if response.status_code == 200:
            return True, response.json()
        return False, None
    except Exception as e:
        return False, str(e)