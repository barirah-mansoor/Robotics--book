import os
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# LlamaIndex imports
from llam-index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llam-index.llms.openai import OpenAI
from llam-index.embeddings.openai import OpenAIEmbedding
from llam-index.vector_stores.qdrant import QdrantVectorStore
from llam-index.core.storage.storage_context import StorageContext

# Load environment variables from .env file
load_dotenv()

# --- Environment Variables ---
QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # For OpenAI LLM and Embeddings

if not QDRANT_HOST or not QDRANT_API_KEY:
    raise ValueError("QDRANT_HOST and QDRANT_API_KEY must be set in the .env file")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY must be set in the .env file for LLM and embeddings")

# --- Initialize Qdrant Client ---
qdrant_client = QdrantClient(host=QDRANT_HOST, api_key=QDRANT_API_KEY)
COLLECTION_NAME = "humanoid_robotics_book"
DOCS_DIR = "../website/docs"

# Create collection if it doesn't exist
if not qdrant_client.collection_exists(collection_name=COLLECTION_NAME):
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE), # OpenAI embeddings are 1536 dim
    )
    print(f"Collection '{COLLECTION_NAME}' created in Qdrant.")
else:
    print(f"Collection '{COLLECTION_NAME}' already exists in Qdrant.")

# --- LlamaIndex Setup ---
# Configure LLM and Embedding Model (using OpenAI as an example)
llm = OpenAI(model="gpt-3.5-turbo", api_key=OPENAI_API_KEY)
embed_model = OpenAIEmbedding(api_key=OPENAI_API_KEY)

# Configure LlamaIndex ServiceContext
service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)

# Configure Qdrant as LlamaIndex VectorStore
vector_store = QdrantVectorStore(client=qdrant_client, collection_name=COLLECTION_NAME)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Load documents from Docusaurus docs directory
print(f"Loading documents from {DOCS_DIR}...")
documents = SimpleDirectoryReader(DOCS_DIR).load_data()
print(f"Loaded {len(documents)} documents.")

# Create/Rebuild LlamaIndex
print("Creating/Rebuilding LlamaIndex and storing embeddings in Qdrant...")
index = VectorStoreIndex.from_documents(
    documents,
    service_context=service_context,
    storage_context=storage_context,
    show_progress=True
)
print("LlamaIndex created/rebuilt and documents indexed in Qdrant.")
