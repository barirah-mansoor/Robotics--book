from qdrant_client import QdrantClient, models

def get_qdrant_client():
    # Initialize Qdrant client. For local/Docker, use host="localhost", port=6333
    # For cloud, replace with your Qdrant Cloud URL and API key
    client = QdrantClient(host="localhost", port=6333)

    # Define collection name (can be moved to config)
    collection_name = "textbook_chunks"

    # Create collection if it doesn't exist
    if not client.collection_exists(collection_name=collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE) # size should match embedding model output dimension
        )
    return client
