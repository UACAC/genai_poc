import requests
import json

BASE_URL = "http://localhost:8020"


def test_health():
    """Check that the server is up and /health returns status: ok."""
    response = requests.get(f"{BASE_URL}/health")
    print("Health endpoint response:", response.status_code, response.text)
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_create_collection():
    """Create a new collection named 'test_collection'."""
    collection_name = "test_collection"
    response = requests.post(f"{BASE_URL}/collection/create?collection_name={collection_name}")
    print("Create collection response:", response.status_code, response.text)
    assert response.status_code == 200
    # Optionally check the response JSON
    data = response.json()
    assert data.get("created") == collection_name

def test_list_collections():
    """Verify that 'test_collection' is now in the list of all collections."""
    response = requests.get(f"{BASE_URL}/collections")
    print("List collections response:", response.status_code, response.text)
    assert response.status_code == 200
    collections = response.json()
    assert "test_collection" in collections

def test_get_collection_info():
    """Get info about 'test_collection'."""
    params = {"collection_name": "test_collection"}
    response = requests.get(f"{BASE_URL}/collection", params=params)
    print("Get collection info response:", response.status_code, response.text)
    assert response.status_code == 200
    data = response.json()
    assert data.get("name") == "test_collection"

def test_add_documents():
    """Add multiple documents to 'test_collection'."""
    documents = {
        "collection_name": "test_collection",
        "documents": [
            "The quick brown fox jumps over the lazy dog",
            "Pack my box with five dozen liquor jugs",
            "How vexingly quick daft zebras jump"
        ],
        "ids": ["doc1", "doc2", "doc3"]
    }
    response = requests.post(f"{BASE_URL}/documents/add", json=documents)
    print("Add documents response:", response.status_code, response.text)
    assert response.status_code == 200
    data = response.json()
    assert data.get("collection") == "test_collection"
    assert data.get("added_count") == 3

def test_list_documents():
    """Retrieve all documents in 'test_collection'."""
    params = {"collection_name": "test_collection"}
    response = requests.get(f"{BASE_URL}/documents", params=params)
    print("List documents response:", response.status_code, response.text)
    assert response.status_code == 200
    data = response.json()
    # data typically contains something like {"ids": [...], "documents": [...], ...}
    assert "ids" in data
    assert "documents" in data
    assert len(data["ids"]) == 3
    assert len(data["documents"]) == 3

def test_edit_document():
    """
    Edit (replace) one document by ID.
    The endpoint typically removes the old doc and re-adds the new one.
    """
    body = {
        "collection_name": "test_collection",
        "doc_id": "doc1",
        "new_document": "Updated content for doc1"
    }
    response = requests.post(f"{BASE_URL}/documents/edit", json=body)
    print("Edit document response:", response.status_code, response.text)
    assert response.status_code == 200
    data = response.json()
    assert data["updated_id"] == "doc1"

    # Optionally retrieve documents to confirm the change
    params = {"collection_name": "test_collection"}
    resp_docs = requests.get(f"{BASE_URL}/documents", params=params)
    doc_data = resp_docs.json()
    assert "Updated content for doc1" in doc_data["documents"]

def test_remove_documents():
    """Remove one of the documents by ID."""
    body = {
        "collection_name": "test_collection",
        "ids": ["doc2"]
    }
    response = requests.post(f"{BASE_URL}/documents/remove", json=body)
    print("Remove documents response:", response.status_code, response.text)
    assert response.status_code == 200
    data = response.json()
    assert "doc2" in data["removed_ids"]

    # Verify doc2 no longer appears
    resp_docs = requests.get(f"{BASE_URL}/documents", params={"collection_name": "test_collection"})
    doc_data = resp_docs.json()
    if "ids" in doc_data:
        assert "doc2" not in doc_data["ids"]

def test_rename_collection():
    """
    Rename 'test_collection' -> 'test_collection_renamed'.
    """
    old_name = "test_collection"
    new_name = "test_collection_renamed"

    # Check old collection exists before renaming
    list_resp = requests.get(f"{BASE_URL}/collections")
    collections = list_resp.json()
    assert old_name in collections, f"Collection '{old_name}' does not exist before rename test."

    # Rename collection
    params = {"old_name": old_name, "new_name": new_name}
    response = requests.put(f"{BASE_URL}/collection/edit", params=params)
    print("Rename collection response:", response.status_code, response.text)

    # Verify successful rename
    assert response.status_code == 200, f"Unexpected status code: {response.status_code}, response: {response.text}"
    data = response.json()
    assert data["old_name"] == old_name
    assert data["new_name"] == new_name

    # Check new collection is in the list
    list_resp = requests.get(f"{BASE_URL}/collections")
    collections = list_resp.json()
    print("Collections after rename:", collections)
    assert new_name in collections, f"New collection '{new_name}' was not created."
    assert old_name not in collections, f"Old collection '{old_name}' was not deleted."

def test_delete_collection():
    """Delete the renamed collection and ensure it's gone."""
    new_name = "test_collection_renamed"

    # Check collection exists before attempting to delete
    list_resp = requests.get(f"{BASE_URL}/collections")
    collections = list_resp.json()
    assert new_name in collections, f"Collection '{new_name}' does not exist before deletion test."

    # Delete collection
    params = {"collection_name": new_name}
    response = requests.delete(f"{BASE_URL}/collection", params=params)
    print("Delete collection response:", response.status_code, response.text)

    # Verify successful deletion
    assert response.status_code == 200, f"Unexpected status code: {response.status_code}, response: {response.text}"
    data = response.json()
    assert data.get("deleted") == new_name

    # Confirm collection no longer exists
    list_resp = requests.get(f"{BASE_URL}/collections")
    collections = list_resp.json()
    print("Collections after delete:", collections)
    assert new_name not in collections, f"Collection '{new_name}' was not deleted."



def run_tests():
    print("Running tests...")

    test_health()
    print("✓ Health check passed")

    test_create_collection()
    print("✓ Collection creation passed")

    test_list_collections()
    print("✓ Collections listed successfully")

    test_get_collection_info()
    print("✓ Retrieved collection info")

    test_add_documents()
    print("✓ Documents added")

    test_list_documents()
    print("✓ Documents listed successfully")

    test_edit_document()
    print("✓ Document edited successfully")

    test_remove_documents()
    print("✓ Document removal passed")

    test_rename_collection()
    print("✓ Collection renamed successfully")

    test_delete_collection()
    print("✓ Collection deletion passed")

    print("\nAll tests passed!")

if __name__ == "__main__":
    run_tests()
