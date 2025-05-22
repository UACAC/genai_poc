import os
import uvicorn
from fastapi import FastAPI, Query, HTTPException, Body
from pydantic import BaseModel
from chromadb.config import Settings
from chromadb import Client

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
    # In v0.6+, list_collections() already returns a list of string collection names.
    return chroma_client.list_collections()


@app.post("/collection/create")
def create_collection(collection_name: str = Query(...)):
    """
    Create a ChromaDB collection with the given name.
    """
    # ✅ 'list_collections()' now returns just the names (strings).
    existing_collections = chroma_client.list_collections()
    
    if collection_name in existing_collections:
        raise HTTPException(
            status_code=400,
            detail=f"Collection '{collection_name}' already exists."
        )
    
    chroma_client.create_collection(collection_name)
    return {"created": collection_name}


@app.get("/collection")
def get_collection_info(collection_name: str = Query(...)):
    """
    Get basic info about a single collection.
    In Chroma v0.6+, get_collection() doesn't accept `allow_create`,
    so we check existence ourselves via list_collections().
    """
    existing_collections = chroma_client.list_collections()
    if collection_name not in existing_collections:
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{collection_name}' not found."
        )
    
    # Now we know it exists, so get_collection() won't auto-create a new one.
    collection = chroma_client.get_collection(collection_name)
    return {"name": collection.name}

@app.delete("/collection")
def delete_collection(collection_name: str = Query(...)):
    """
    Delete a ChromaDB collection by name.
    """
    # Just get the list of strings (collection names)
    existing_collections = chroma_client.list_collections()

    # Ensure the collection exists before attempting deletion
    if collection_name not in existing_collections:
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{collection_name}' not found."
        )

    # Now, safely delete the collection
    chroma_client.delete_collection(collection_name)

    return {"deleted": collection_name}

@app.put("/collection/edit")
def edit_collection_name(old_name: str = Query(...), new_name: str = Query(...)):
    """
    Rename a ChromaDB collection from 'old_name' to 'new_name'.
    """
    # Get existing collections
    existing_collections = chroma_client.list_collections()

    # Ensure the old collection exists
    if old_name not in existing_collections:
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{old_name}' not found."
        )

    # Ensure the new name does not already exist
    if new_name in existing_collections:
        raise HTTPException(
            status_code=400,
            detail=f"Collection '{new_name}' already exists. Choose a different name."
        )

    # Retrieve the old collection (without `allow_create`)
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

### Document Endpoints ###
class DocumentAddRequest(BaseModel):
    collection_name: str
    documents: list[str]
    ids: list[str]
    embeddings: list[list[float]] = None  
    metadatas: list[dict] = None           

@app.post("/documents/add")
def add_documents(req: DocumentAddRequest):
    # Check if the collection exists
    if req.collection_name not in chroma_client.list_collections():
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
class DocumentRemoveRequest(BaseModel):
    collection_name: str
    ids: list[str]

@app.post("/documents/remove")
def remove_documents(req: DocumentRemoveRequest):
    """
    Remove documents by ID from a given collection.
    """
    # Check if the collection exists first
    existing_collections = chroma_client.list_collections()
    if req.collection_name not in existing_collections:
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


class DocumentEditRequest(BaseModel):
    collection_name: str
    doc_id: str
    new_document: str

@app.post("/documents/edit")
def edit_document(req: DocumentEditRequest):
    """
    Replace the content of an existing document by ID.
    Internally, you can delete and re-add or do a custom approach.
    """
    # Check if the collection exists first
    existing_collections = chroma_client.list_collections()
    if req.collection_name not in existing_collections:
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{req.collection_name}' not found."
        )

    # Now, safely retrieve the collection (since we verified it exists)
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


@app.get("/documents")
def list_documents(collection_name: str = Query(...)):
    """
    Get all documents (and their IDs) in a collection.
    Note: Be mindful if the collection is huge (this could be large).
    """
    # Check if the collection exists first
    existing_collections = chroma_client.list_collections()
    if collection_name not in existing_collections:
        raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found.")

    # Now, safely retrieve the collection (since we verified it exists)
    collection = chroma_client.get_collection(collection_name)

    # Retrieve documents
    docs = collection.get()
    return docs  # Returns documents and associated metadata

class DocumentQueryRequest(BaseModel):
    collection_name: str
    query_embeddings: list[list[float]]
    n_results: int = 5
    include: list[str] = ["documents", "metadatas", "distances"]

@app.post("/documents/query")
def query_documents(req: DocumentQueryRequest):
    # Check if the collection exists first.
    existing_collections = chroma_client.list_collections()
    if req.collection_name not in existing_collections:
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{req.collection_name}' not found."
        )

    # Retrieve the collection.
    collection = chroma_client.get_collection(req.collection_name)

    # Perform the query using the provided embeddings and parameters.
    query_result = collection.query(
        query_embeddings=req.query_embeddings,
        n_results=req.n_results,
        include=req.include
    )
    return query_result

### Run with Uvicorn if called directly ###
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8020)
