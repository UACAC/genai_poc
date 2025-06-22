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

def store_files_in_chromadb(files, collection_name, model_name="none", openai_api_key=None, chunk_size=1000, chunk_overlap=200, store_images=True):
    files_data = [('files', (file.name, file.getvalue(), file.type)) for file in files]
    params = {
        "collection_name": collection_name,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "store_images": store_images,
        "model_name": model_name
    }
    headers = {"X-OpenAI-API-Key": openai_api_key} if openai_api_key and model_name in ["gpt-4", "gpt-3.5-turbo"] else {}
    response = requests.post(f"{CHROMADB_API}/documents/upload-and-process", params=params, files=files_data, headers=headers, timeout=300)
    if response.status_code != 200:
        raise Exception(f"Failed to store documents: {response.text}")
    return response.json()

def list_all_chunks_with_scores(collection_name, query_text=None):
    response = requests.get(f"{CHROMADB_API}/documents", params={"collection_name": collection_name})
    if response.status_code != 200:
        return []
    docs = response.json()
    scores_dict = {}
    if query_text:
        query_embedding = embedding_model.encode([query_text]).tolist()
        score_response = requests.post(f"{CHROMADB_API}/documents/query", json={"collection_name": collection_name, "query_embeddings": query_embedding, "n_results": len(docs["ids"]), "include": ["metadatas", "distances"]})
        if score_response.status_code == 200:
            results = score_response.json()
            scores_dict = {doc_id: round(distance, 4) for doc_id, distance in zip(results["ids"][0], results["distances"][0])}
    return [
        {
            "Collection": collection_name, 
            "Document Name": metadata.get("document_name", "Unknown"), 
            "Document ID": doc_id, "Chunk ID": f"`{doc_id}`", 
            "Document Text": f">{doc_text[:250] + '...' if len(doc_text) > 250 else doc_text}", 
            "Metadata": f"**{(metadata or {}).get('document_name', 'Unknown')}**", 
            "Score": scores_dict.get(doc_id, "N/A")
        } 
        for doc_id, doc_text, metadata in zip(docs["ids"], docs["documents"], docs.get("metadatas", [{}] * len(docs["ids"]))) 
    ]


def image_to_base64(img_url):
    try:
        response = requests.get(img_url)
        img = Image.open(BytesIO(response.content))
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        return f"![img](data:image/png;base64,{img_base64})"
    except Exception as e:
        return f"⚠️ Failed to load image: {img_url} — {str(e)}"

def render_reconstructed_document(result: dict):
    rich_content = result.get("reconstructed_content", "")
    images = result.get("images", [])

    # Replace image markers with base64 images + description
    for image in images:
        marker = f"[IMAGE:{image['filename']}]"
        img_url = f"{CHROMADB_API}/images/{image['filename']}"
        base64_img = image_to_base64(img_url)
        description = image.get("description", "")
        replacement = f"{base64_img}\n\n**Image Explanation:** {description}"
        rich_content = rich_content.replace(marker, replacement)

    st.markdown(rich_content, unsafe_allow_html=True)

def clean_text(text):
    # Remove null bytes and non-XML-safe characters
    return ''.join(c for c in text if c == '\n' or (32 <= ord(c) <= 126))

def export_to_docx(result):
    doc = Document()
    doc.add_heading(result["document_name"], 0)

    rich_content = result["reconstructed_content"]
    images = result.get("images", [])
    chromadb_url = os.getenv("CHROMA_URL", "http://localhost:8020")

    for img in images:
        marker = f"[IMAGE:{img['filename']}]"
        if marker in rich_content:
            # Replace the marker with a placeholder for post-processing
            rich_content = rich_content.replace(marker, f"@@IMAGE:{img['filename']}@@")

    chunks = re.split(r'(@@IMAGE:.+?@@)', rich_content)

    for chunk in chunks:
        img_match = re.match(r'@@IMAGE:(.+?)@@', chunk)
        if img_match:
            filename = img_match.group(1)
            img_url = f"{chromadb_url}/images/{filename}"

            try:
                response = requests.get(img_url)
                if response.status_code == 200:
                    image_stream = BytesIO(response.content)
                    img = Image.open(image_stream)

                    doc.add_picture(image_stream, width=Inches(5.5))
                    # Add explanation after image if it exists
                    desc = next((i['description'] for i in images if i['filename'] == filename), None)
                    if desc:
                        doc.add_paragraph(clean_text(desc), style='Intense Quote')
            except Exception as e:
                doc.add_paragraph(f"[Image '{filename}' could not be retrieved: {e}]")
        else:
            cleaned = clean_text(chunk.strip())
            if cleaned:
                doc.add_paragraph(cleaned)

        doc.add_page_break()

    # Save
    output_path = f"/tmp/{result['document_name']}.docx"
    doc.save(output_path)
    return output_path


def export_to_pdf(result):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    content = result["reconstructed_content"]
    image_map = {img["filename"]: img["description"] for img in result.get("images", [])}

    parts = content.split("[IMAGE:")
    pdf.multi_cell(0, 10, parts[0].strip())

    for part in parts[1:]:
        if "]" in part:
            filename, rest = part.split("]", 1)
            image_url = f"{CHROMADB_API}/images/{filename}"

            try:
                response = requests.get(image_url)
                img = Image.open(BytesIO(response.content))
                temp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                img.save(temp_img.name)

                pdf.add_page()
                pdf.image(temp_img.name, w=150)
                pdf.ln(5)
                if filename in image_map:
                    pdf.multi_cell(0, 10, image_map[filename])
            except Exception as e:
                pdf.multi_cell(0, 10, f"[Failed to insert image {filename}]: {e}")

            pdf.multi_cell(0, 10, rest.strip())
        else:
            pdf.multi_cell(0, 10, part.strip())

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(temp_file.name)
    return temp_file.name
