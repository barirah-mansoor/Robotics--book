from sentence_transformers import SentenceTransformer

def get_embedding_model():
    # Load the all-MiniLM-L6-v2 model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model

def embed_text(text: str):
    model = get_embedding_model()
    embeddings = model.encode(text, convert_to_tensor=False)
    return embeddings.tolist()
