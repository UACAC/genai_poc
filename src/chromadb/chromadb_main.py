import os
import uvicorn
from fastapi import FastAPI, Query, HTTPException, Body
from pydantic import BaseModel
from chromadb.config import Settings
from chromadb import Client
import json
import uuid
import tempfile
import shutil
import base64
from pathlib import Path
from typing import List, Optional, Dict, Any
from fastapi import UploadFile, File, Response
from sentence_transformers import SentenceTransformer
from markitdown import MarkItDown
from langchain_ollama import OllamaLLM
from langchain_openai import ChatOpenAI
from PyPDF2 import PdfReader
from PIL import Image
import pandas as p
from io import BytesIO
from fastapi import Request
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Where ChromaDB should persist data
PERSIST_DIR = os.getenv("CHROMADB_PERSIST_DIRECTORY", "/app/chroma_db_data")

# Configure Chroma
settings = Settings(
    persist_directory=PERSIST_DIR,
    anonymized_telemetry=False
)

# Create a global ChromaDB client (reuse instead of creating a new one each route)
chroma_client = Client(settings)

# Initialize embedding model and MarkItDown
embedding_model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')

open_api_key = os.getenv("OPENAI_API_KEY")

# Image storage directory
IMAGES_DIR = os.path.join(os.getcwd(), "stored_images")
os.makedirs(IMAGES_DIR, exist_ok=True)

def get_markitdown_instance(model_name="none", openai_api_key=open_api_key):
    """
    Create a MarkItDown instance based on model selection
    """
    model_name = model_name.lower()
    ollama_host = os.getenv("LLM_OLLAMA_HOST", "http://ollama:11434")
    
    # OpenAI models
    if model_name in ["gpt4", "gpt-4", "gpt-3.5-turbo"]:
        if openai_api_key:
            try:
                openai_client = ChatOpenAI(
                    model=model_name, 
                    openai_api_key=openai_api_key, 
                    temperature=0.7
                )
                return MarkItDown(llm_client=openai_client, llm_model=model_name)
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client: {e}. Using basic extraction.")
                return MarkItDown()
        else:
            logger.warning("No OpenAI API key provided. Using basic extraction.")
            return MarkItDown()
    
    # Ollama models
    elif model_name in ["llama", "llama3", "llama3:8b"]:
        actual_model = "llama3" if model_name in ["llama", "llama3"] else model_name
        try:
            llama_client = OllamaLLM(
                model=actual_model, 
                base_url=ollama_host, 
                temperature=0.7
            )
            return MarkItDown(llm_client=llama_client, llm_model=actual_model)
        except Exception as e:
            logger.warning(f"Failed to initialize Ollama: {e}. Using basic extraction.")
            return MarkItDown()
    
    # No LLM or unknown model
    else:
        logger.info(f"Using basic MarkItDown without LLM support for model: {model_name}")
        return MarkItDown()

# Create a standard FastAPI app
app = FastAPI(title="ChromaDB Dockerized")
md = get_markitdown_instance("none")

def extract_and_store_images_from_pdf(file_content: bytes, filename: str, temp_dir: str, doc_id: str, md) -> List[Dict]:
    """Extract images from PDF and store them with metadata"""
    images_data = []
    
    try:
        temp_pdf_path = os.path.join(temp_dir, filename)
        with open(temp_pdf_path, 'wb') as f:
            f.write(file_content)
        
        reader = PdfReader(temp_pdf_path)
        
        for page_num, page in enumerate(reader.pages, 1):
            resources = page.get("/Resources")
            if resources and "/XObject" in resources:
                xobjects = resources["/XObject"].get_object()
                
                for obj_name in xobjects:
                    xobj = xobjects[obj_name]
                    if xobj.get("/Subtype") == "/Image":
                        try:
                            filters = xobj.get('/Filter')
                            data = xobj.get_data()
                            img_ext = 'jpg' if filters == '/DCTDecode' else 'png'
                            
                            # Create unique image filename
                            img_filename = f"{doc_id}_page_{page_num}_{obj_name[1:]}.{img_ext}"
                            img_storage_path = os.path.join(IMAGES_DIR, img_filename)
                            
                            # Save image to storage directory
                            with open(img_storage_path, "wb") as img_file:
                                img_file.write(data)
                            
                            # Get image description using MarkItDown
                            try:
                                result = md.convert(img_storage_path)
                                description = result.text_content if hasattr(result, 'text_content') else str(result)
                            except:
                                description = f"Image: {img_filename}"
                            
                            images_data.append({
                                "filename": img_filename,
                                "storage_path": img_storage_path,
                                "page": page_num,
                                "description": description,
                                "position_marker": f"[IMAGE:{img_filename}]"
                            })
                            
                            logger.info(f"Stored image: {img_filename}")
                            
                        except Exception as e:
                            logger.error(f"Failed to extract image {obj_name}: {e}")
    
    except Exception as e:
        logger.error(f"Error extracting images from {filename}: {e}")
    
    return images_data

def process_document_with_context(file_content: bytes, filename: str, temp_dir: str, doc_id: str, md) -> Dict:
    """Process document maintaining context and storing images"""
    
    file_extension = Path(filename).suffix.lower()
    images_data = []
    
    # Extract images first for PDFs
    if file_extension == '.pdf':
        images_data = extract_and_store_images_from_pdf(file_content, filename, temp_dir, doc_id, md)
    
    # Process document with MarkItDown
    temp_file_path = os.path.join(temp_dir, filename)
    with open(temp_file_path, 'wb') as f:
        f.write(file_content)
    
    try:
        result = md.convert(temp_file_path)
        content = result.text_content if hasattr(result, 'text_content') else str(result)
    except Exception as e:
        logger.error(f"MarkItDown processing failed for {filename}: {e}")
        if file_extension == '.txt':
            content = file_content.decode('utf-8', errors='ignore')
        else:
            raise e
    
    # Insert image markers in content at logical positions
    if images_data and content:
        # Simple strategy: insert image markers at paragraph breaks
        paragraphs = content.split('\n\n')
        
        enhanced_content = []
        img_index = 0
        
        for i, paragraph in enumerate(paragraphs):
            enhanced_content.append(paragraph)
            
            # Insert image markers strategically
            if img_index < len(images_data) and i % max(2, len(paragraphs) // len(images_data)) == 1:
                img = images_data[img_index]
                enhanced_content.append(f"\n{img['position_marker']}\n{img['description']}\n")
                img_index += 1
        
        content = '\n\n'.join(enhanced_content)
    
    return {
        "content": content,
        "images_data": images_data,
        "file_type": file_extension
    }

def smart_chunk_with_context(content: str, images_data: List[Dict], chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
    """Chunk content while preserving image context and references"""
    
    chunks = []
    
    # Find image marker positions
    image_positions = {}
    for img in images_data:
        marker = img['position_marker']
        pos = content.find(marker)
        if pos != -1:
            image_positions[pos] = img
    
    # Split content into chunks
    start = 0
    chunk_index = 0
    
    while start < len(content):
        end = start + chunk_size
        chunk_text = content[start:end]
        
        # Adjust boundaries to preserve context
        if end < len(content):
            # Try to break at sentence or paragraph boundaries
            last_period = chunk_text.rfind('.')
            last_newline = chunk_text.rfind('\n\n')
            break_point = max(last_period, last_newline)
            
            if break_point > start + chunk_size // 2:
                chunk_text = content[start:break_point + 1]
                end = break_point + 1
        
        # Find images in this chunk
        chunk_images = []
        for pos, img_data in image_positions.items():
            if start <= pos < end:
                chunk_images.append(img_data)
        
        chunks.append({
            "content": chunk_text.strip(),
            "chunk_index": chunk_index,
            "start_position": start,
            "end_position": end,
            "images": chunk_images,
            "has_images": len(chunk_images) > 0
        })
        
        chunk_index += 1
        start = end - overlap
        
        if start >= len(content):
            break
    
    return chunks

### Health Checks ###

@app.get("/")
def root_health_check():
    """Basic health check."""
    return {"status": "ok", "detail": "ChromaDB custom server running."}

@app.get("/health")
def health_check():
    """Enhanced health check endpoint."""
    return {
        "status": "ok",
        "markitdown_available": md is not None,
        "openai_available": open_api_key is not None,
        "supported_formats": ["pdf", "docx", "xlsx", "csv", "txt", "pptx", "html"],
        "embedding_model": "multi-qa-mpnet-base-dot-v1",
        "images_directory": IMAGES_DIR
    }


### Collection Endpoints ###

@app.get("/collections")
def list_collections():
    try:
        collection_names = chroma_client.list_collections()  # Now returns just names
        return {"collections": collection_names}
    except Exception as e:
        logger.error(f"Error listing collections: {e}")
        raise HTTPException(status_code=500, detail="Failed to list collections")



@app.post("/collection/create")
def create_collection(collection_name: str = Query(...)):
    """
    Create a ChromaDB collection with the given name.
    """
    try:
        # Get existing collection names safely
        existing_names = chroma_client.list_collections()
        
        if collection_name in existing_names:
            raise HTTPException(
                status_code=400,
                detail=f"Collection '{collection_name}' already exists."
            )
        
        chroma_client.create_collection(collection_name)
        logger.info(f"Created collection: {collection_name}")
        return {"created": collection_name}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating collection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating collection: {str(e)}")


@app.get("/collection")
def get_collection_info(collection_name: str = Query(...)):
    """
    Get basic info about a single collection.
    """
    try:
        # Check if collection exists
        existing_names = chroma_client.list_collections()
                
        if collection_name not in existing_names:
            raise HTTPException(
                status_code=404,
                detail=f"Collection '{collection_name}' not found."
            )
        
        collection = chroma_client.get_collection(name=collection_name)
        return {"name": collection.name}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting collection info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting collection info: {str(e)}")


@app.delete("/collection")
def delete_collection(collection_name: str = Query(...)):
    """
    Delete a ChromaDB collection by name.
    """
    try:
        # Get existing collection names safely
        existing_collections = chroma_client.list_collections()


        if collection_name not in existing_collections:
            raise HTTPException(
                status_code=404,
                detail=f"Collection '{collection_name}' not found."
            )

        chroma_client.delete_collection(collection_name)
        logger.info(f"Deleted collection: {collection_name}")
        return {"deleted": collection_name}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting collection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting collection: {str(e)}")


@app.put("/collection/edit")
def edit_collection_name(old_name: str = Query(...), new_name: str = Query(...)):
    """
    Rename a ChromaDB collection from 'old_name' to 'new_name'.
    """
    try:
        # Get existing collection names
        existing_names = chroma_client.list_collections()

        if old_name not in existing_names:
            raise HTTPException(
                status_code=404,
                detail=f"Collection '{old_name}' not found."
            )

        if new_name in existing_names:
            raise HTTPException(
                status_code=400,
                detail=f"Collection '{new_name}' already exists. Choose a different name."
            )

        # Retrieve the old collection
        collection = chroma_client.get_collection(name=old_name)

        # Create a new collection with the new name
        new_collection = chroma_client.create_collection(name=new_name)

        # Retrieve all documents from the old collection
        old_docs = collection.get()
        if old_docs and "ids" in old_docs and "documents" in old_docs:
            new_collection.add(ids=old_docs["ids"], documents=old_docs["documents"])

        # Delete the old collection
        chroma_client.delete_collection(old_name)

        return {"old_name": old_name, "new_name": new_name}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error editing collection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error editing collection: {str(e)}")


### Document Endpoints ###
class DocumentAddRequest(BaseModel):
    collection_name: str
    documents: list[str]
    ids: list[str]
    embeddings: list[list[float]] = None  
    metadatas: list[dict] = None           

@app.post("/documents/add")
def add_documents(req: DocumentAddRequest):
    try:
        # Check if the collection exists
        existing_names = chroma_client.list_collections()
                
        if req.collection_name not in existing_names:
            raise HTTPException(
                status_code=404,
                detail=f"Collection '{req.collection_name}' not found."
            )
        
        # Retrieve the collection
        collection = chroma_client.get_collection(req.collection_name)
        
        # Add documents along with embeddings and metadatas
        collection.add(
            documents=req.documents,
            ids=req.ids,
            embeddings=req.embeddings,
            metadatas=req.metadatas
        )
        return {
            "collection": req.collection_name,
            "added_count": len(req.documents),
            "ids": req.ids
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error adding documents: {str(e)}")


class DocumentRemoveRequest(BaseModel):
    collection_name: str
    ids: list[str]

@app.post("/documents/remove")
def remove_documents(req: DocumentRemoveRequest):
    """
    Remove documents by ID from a given collection.
    """
    try:
        # Check if the collection exists first
        existing_names = chroma_client.list_collections()
                
        if req.collection_name not in existing_names:
            raise HTTPException(
                status_code=404,
                detail=f"Collection '{req.collection_name}' not found."
            )

        # Now, safely retrieve the collection (since we verified it exists)
        collection = chroma_client.get_collection(req.collection_name)

        # Ensure at least one of the documents exists before attempting to delete
        existing_docs = collection.get()
        existing_ids = set(existing_docs.get("ids", []))

        if not any(doc_id in existing_ids for doc_id in req.ids):
            raise HTTPException(
                status_code=404,
                detail=f"None of the provided document IDs {req.ids} exist in collection '{req.collection_name}'."
            )

        # Delete the specified document(s)
        collection.delete(ids=req.ids)
        
        return {
            "collection": req.collection_name,
            "removed_ids": req.ids
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error removing documents: {str(e)}")


class DocumentEditRequest(BaseModel):
    collection_name: str
    doc_id: str
    new_document: str

@app.post("/documents/edit")
def edit_document(req: DocumentEditRequest):
    """
    Replace the content of an existing document by ID.
    """
    try:
        # Check if the collection exists first
        existing_names = chroma_client.list_collections()
                
        if req.collection_name not in existing_names:
            raise HTTPException(
                status_code=404,
                detail=f"Collection '{req.collection_name}' not found."
            )

        # Now, safely retrieve the collection
        collection = chroma_client.get_collection(req.collection_name)

        # Ensure the document exists before attempting to update
        existing_docs = collection.get()
        if req.doc_id not in existing_docs.get("ids", []):
            raise HTTPException(
                status_code=404,
                detail=f"Document '{req.doc_id}' not found in collection '{req.collection_name}'."
            )

        # Delete the old document and re-add with new content
        collection.delete(ids=[req.doc_id])
        collection.add(documents=[req.new_document], ids=[req.doc_id])

        return {
            "collection": req.collection_name,
            "updated_id": req.doc_id,
            "new_document": req.new_document
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error editing document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error editing document: {str(e)}")


@app.get("/documents")
def list_documents(collection_name: str = Query(...)):
    """
    Get all documents (and their IDs) in a collection.
    """
    try:
        # Check if the collection exists first
        existing_names = chroma_client.list_collections()
                
        if collection_name not in existing_names:
            raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found.")

        # Now, safely retrieve the collection
        collection = chroma_client.get_collection(name=collection_name)

        # Retrieve documents
        docs = collection.get()
        return docs
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")


class DocumentQueryRequest(BaseModel):
    collection_name: str
    query_embeddings: list[list[float]]
    n_results: int = 5
    include: list[str] = ["documents", "metadatas", "distances"]

@app.post("/documents/query")
def query_documents(req: DocumentQueryRequest):
    try:
        # Check if the collection exists first
        existing_names = chroma_client.list_collections()
                
        if req.collection_name not in existing_names:
            raise HTTPException(
                status_code=404,
                detail=f"Collection '{req.collection_name}' not found."
            )

        # Retrieve the collection
        collection = chroma_client.get_collection(req.collection_name)

        # Perform the query using the provided embeddings and parameters
        query_result = collection.query(
            query_embeddings=req.query_embeddings,
            n_results=req.n_results,
            include=req.include
        )
        return query_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error querying documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error querying documents: {str(e)}")


@app.post("/documents/upload-and-process")
async def upload_and_process_documents(
    files: List[UploadFile] = File(...),
    collection_name: str = Query(...),
    chunk_size: int = Query(1000),
    chunk_overlap: int = Query(200),
    store_images: bool = Query(True),
    model_name: str = Query("none"),
    request: Request = None  # Add request parameter
):
    """
    Upload and process documents with image storage and context preservation.
    """
    try:
        # Get OpenAI API key from header if present
        openai_api_key = request.headers.get("X-OpenAI-API-Key") if request else None
        
        # Create MarkItDown instance for this request
        md = get_markitdown_instance(model_name, openai_api_key)
        
        # Check if collection exists
        existing_names = chroma_client.list_collections()
                
        if collection_name not in existing_names:
            raise HTTPException(
                status_code=404,
                detail=f"Collection '{collection_name}' not found."
            )
        
        collection = chroma_client.get_collection(name=collection_name)
        processed_files = []
        total_chunks = 0
        total_images = 0
        
        with tempfile.TemporaryDirectory() as temp_dir:
            for file in files:
                logger.info(f"Processing file: {file.filename} with model: {model_name}")
                
                try:
                    file_content = await file.read()
                    doc_id = f"{Path(file.filename).stem}_{uuid.uuid4().hex[:8]}"
                    
                    # Process document with context preservation
                    # Pass md instance to the processing function
                    doc_data = process_document_with_context(
                        file_content, 
                        file.filename, 
                        temp_dir, 
                        doc_id,
                        md  # Pass the md instance
                    )
                    
                    if not doc_data["content"].strip():
                        processed_files.append({
                            "filename": file.filename,
                            "status": "skipped",
                            "error": "No content extracted"
                        })
                        continue
                    
                    # Smart chunking with context
                    chunks = smart_chunk_with_context(
                        doc_data["content"], 
                        doc_data["images_data"], 
                        chunk_size, 
                        chunk_overlap
                    )
                    
                    file_chunks = 0
                    file_images = len(doc_data["images_data"])
                    
                    for chunk_data in chunks:
                        # Generate embedding
                        embedding = embedding_model.encode([chunk_data["content"]])[0].tolist()
                        
                        # Prepare comprehensive metadata
                        metadata = {
                            "document_id": doc_id,
                            "document_name": file.filename,
                            "file_type": doc_data["file_type"],
                            "chunk_index": chunk_data["chunk_index"],
                            "total_chunks": len(chunks),
                            "start_position": chunk_data["start_position"],
                            "end_position": chunk_data["end_position"],
                            "has_images": chunk_data["has_images"],
                            "image_count": len(chunk_data["images"]),
                            "processed_with": "markitdown_enhanced",
                            "model_used": model_name,
                            "timestamp": str(uuid.uuid4()),
                            "images_stored": store_images
                        }
                        
                        # Add image metadata
                        if chunk_data["images"]:
                            metadata["image_filenames"] = json.dumps([img["filename"] for img in chunk_data["images"]])
                            metadata["image_descriptions"] = json.dumps([img["description"] for img in chunk_data["images"]])
                            metadata["image_storage_paths"] = json.dumps([img["storage_path"] for img in chunk_data["images"]])
                        
                        # Generate unique document ID
                        chunk_id = f"{doc_id}_chunk_{chunk_data['chunk_index']}"
                        
                        # Add to collection
                        collection.add(
                            documents=[chunk_data["content"]],
                            ids=[chunk_id],
                            metadatas=[metadata],
                            embeddings=[embedding]
                        )
                        
                        file_chunks += 1
                        total_chunks += 1
                    
                    total_images += file_images
                    
                    processed_files.append({
                        "filename": file.filename,
                        "document_id": doc_id,
                        "file_type": doc_data["file_type"],
                        "chunks_created": file_chunks,
                        "images_stored": file_images if store_images else 0,
                        "status": "success"
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing file {file.filename}: {e}")
                    processed_files.append({
                        "filename": file.filename,
                        "status": "error",
                        "error": str(e)
                    })
        
        return {
            "collection": collection_name,
            "model_used": model_name,
            "total_files_processed": len([f for f in processed_files if f["status"] == "success"]),
            "total_chunks_created": total_chunks,
            "total_images_stored": total_images if store_images else 0,
            "images_directory": IMAGES_DIR,
            "processed_files": processed_files
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in document processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing documents: {str(e)}")

@app.get("/documents/reconstruct/{document_id}")
def reconstruct_document(document_id: str, collection_name: str = Query(...)):
    """
    Reconstruct original document from stored chunks and images.
    """
    try:
        collection = chroma_client.get_collection(name=collection_name)
        
        # Get all chunks for this document
        results = collection.get(
            where={"document_id": document_id},
            include=["documents", "metadatas"]
        )
        
        if not results["ids"]:
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
        
        # Sort chunks by position
        chunks_data = []
        for i, chunk_id in enumerate(results["ids"]):
            metadata = results["metadatas"][i]
            chunks_data.append({
                "chunk_index": metadata.get("chunk_index", 0),
                "content": results["documents"][i],
                "metadata": metadata
            })
        
        chunks_data.sort(key=lambda x: x["chunk_index"])
        
        # Reconstruct content
        reconstructed_content = ""
        all_images = []
        
        for chunk in chunks_data:
            reconstructed_content += chunk["content"] + "\n\n"
            
            # Collect image information
            if chunk["metadata"].get("has_images"):
                try:
                    image_filenames = json.loads(chunk["metadata"].get("image_filenames", "[]"))
                    image_paths = json.loads(chunk["metadata"].get("image_storage_paths", "[]"))
                    image_descriptions = json.loads(chunk["metadata"].get("image_descriptions", "[]"))
                    
                    for filename, path, desc in zip(image_filenames, image_paths, image_descriptions):
                        all_images.append({
                            "filename": filename,
                            "storage_path": path,
                            "description": desc,
                            "exists": os.path.exists(path)
                        })
                except:
                    pass
        
        return {
            "document_id": document_id,
            "document_name": chunks_data[0]["metadata"].get("document_name", "Unknown"),
            "total_chunks": len(chunks_data),
            "reconstructed_content": reconstructed_content.strip(),
            "images": all_images,
            "metadata": {
                "file_type": chunks_data[0]["metadata"].get("file_type"),
                "total_images": len(all_images),
                "processing_timestamp": chunks_data[0]["metadata"].get("timestamp")
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reconstructing document: {str(e)}")

@app.get("/images/{image_filename}")
def get_stored_image(image_filename: str):
    """
    Retrieve a stored image file.
    """
    image_path = os.path.join(IMAGES_DIR, image_filename)
    
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")
    
    try:
        with open(image_path, "rb") as f:
            image_data = f.read()
        
        # Determine content type
        ext = Path(image_filename).suffix.lower()
        content_type = "image/jpeg" if ext == ".jpg" else "image/png"
        
        return Response(content=image_data, media_type=content_type)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading image: {str(e)}")

### Run with Uvicorn if called directly ###
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8020)