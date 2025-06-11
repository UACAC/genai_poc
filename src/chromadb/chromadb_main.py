import os
import uvicorn
from fastapi import FastAPI, Query, HTTPException, Body
from pydantic import BaseModel
from chromadb.config import Settings
from chromadb import Client
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

# Create a standard FastAPI app
app = FastAPI(title="ChromaDB Dockerized")


### Health Checks ###

@app.get("/")
def root_health_check():
    """Basic health check."""
    return {"status": "ok", "detail": "ChromaDB custom server running."}

@app.get("/health")
def health_check():
    """Another health check endpoint."""
    return {"status": "ok"}


### Collection Endpoints ###

@app.get("/collections")
def list_collections():
    """
    List all ChromaDB collections (returns the list of names).
    """
    try:
        # FIXED: Extract just the collection names, not the full objects
        collections = chroma_client.list_collections()
        logger.info(f"Raw collections response: {collections}")
        
        # Handle different ChromaDB versions
        if isinstance(collections, list):
            if len(collections) > 0 and hasattr(collections[0], 'name'):
                # Collections are objects with .name attribute
                collection_names = [col.name for col in collections]
            else:
                # Collections are already strings
                collection_names = collections
        else:
            # Fallback for unexpected return types
            collection_names = []
            
        logger.info(f"Returning collection names: {collection_names}")
        return {"collections": collection_names}
        
    except Exception as e:
        logger.error(f"Error listing collections: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing collections: {str(e)}")


@app.post("/collection/create")
def create_collection(collection_name: str = Query(...)):
    """
    Create a ChromaDB collection with the given name.
    """
    try:
        # Get existing collection names safely
        existing_collections = chroma_client.list_collections()
        existing_names = []
        
        if isinstance(existing_collections, list):
            if len(existing_collections) > 0 and hasattr(existing_collections[0], 'name'):
                existing_names = [col.name for col in existing_collections]
            else:
                existing_names = existing_collections
        
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
        existing_collections = chroma_client.list_collections()
        existing_names = []
        
        if isinstance(existing_collections, list):
            if len(existing_collections) > 0 and hasattr(existing_collections[0], 'name'):
                existing_names = [col.name for col in existing_collections]
            else:
                existing_names = existing_collections
                
        if collection_name not in existing_names:
            raise HTTPException(
                status_code=404,
                detail=f"Collection '{collection_name}' not found."
            )
        
        collection = chroma_client.get_collection(collection_name)
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
        existing_names = []
        
        if isinstance(existing_collections, list):
            if len(existing_collections) > 0 and hasattr(existing_collections[0], 'name'):
                existing_names = [col.name for col in existing_collections]
            else:
                existing_names = existing_collections

        if collection_name not in existing_names:
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
        existing_collections = chroma_client.list_collections()
        existing_names = []
        
        if isinstance(existing_collections, list):
            if len(existing_collections) > 0 and hasattr(existing_collections[0], 'name'):
                existing_names = [col.name for col in existing_collections]
            else:
                existing_names = existing_collections

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
        collection = chroma_client.get_collection(old_name)

        # Create a new collection with the new name
        new_collection = chroma_client.create_collection(new_name)

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
        existing_collections = chroma_client.list_collections()
        existing_names = []
        
        if isinstance(existing_collections, list):
            if len(existing_collections) > 0 and hasattr(existing_collections[0], 'name'):
                existing_names = [col.name for col in existing_collections]
            else:
                existing_names = existing_collections
                
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
        existing_collections = chroma_client.list_collections()
        existing_names = []
        
        if isinstance(existing_collections, list):
            if len(existing_collections) > 0 and hasattr(existing_collections[0], 'name'):
                existing_names = [col.name for col in existing_collections]
            else:
                existing_names = existing_collections
                
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
        existing_collections = chroma_client.list_collections()
        existing_names = []
        
        if isinstance(existing_collections, list):
            if len(existing_collections) > 0 and hasattr(existing_collections[0], 'name'):
                existing_names = [col.name for col in existing_collections]
            else:
                existing_names = existing_collections
                
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
        existing_collections = chroma_client.list_collections()
        existing_names = []
        
        if isinstance(existing_collections, list):
            if len(existing_collections) > 0 and hasattr(existing_collections[0], 'name'):
                existing_names = [col.name for col in existing_collections]
            else:
                existing_names = existing_collections
                
        if collection_name not in existing_names:
            raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found.")

        # Now, safely retrieve the collection
        collection = chroma_client.get_collection(collection_name)

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
        existing_collections = chroma_client.list_collections()
        existing_names = []
        
        if isinstance(existing_collections, list):
            if len(existing_collections) > 0 and hasattr(existing_collections[0], 'name'):
                existing_names = [col.name for col in existing_collections]
            else:
                existing_names = existing_collections
                
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


### Run with Uvicorn if called directly ###
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8020)