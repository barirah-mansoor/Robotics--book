from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.storage.storage_context import StorageContext
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# --- Environment Variables ---
QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not QDRANT_HOST or not QDRANT_API_KEY:
    raise ValueError("QDRANT_HOST and QDRANT_API_KEY must be set in the .env file")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY must be set in the .env file for LLM and embeddings")

# --- Initialize Qdrant Client ---
qdrant_client = QdrantClient(host=QDRANT_HOST, api_key=QDRANT_API_KEY)
COLLECTION_NAME = "humanoid_robotics_book"

# Create collection if it doesn't exist
if not qdrant_client.collection_exists(collection_name=COLLECTION_NAME):
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE), # Gemini embeddings are 768 dim
    )
    print(f"Collection '{COLLECTION_NAME}' created in Qdrant.")
else:
    print(f"Collection '{COLLECTION_NAME}' already exists in Qdrant.")

# --- LlamaIndex Setup ---
# Configure LLM and Embedding Model
llm = Gemini(model="gemini-2.0-flash", api_key=GEMINI_API_KEY)
embed_model = GeminiEmbedding(model_name="text-embedding-005", api_key=GEMINI_API_KEY)

# Configure LlamaIndex ServiceContext
service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)

# Configure Qdrant as LlamaIndex VectorStore
vector_store = QdrantVectorStore(client=qdrant_client, collection_name=COLLECTION_NAME)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Load documents from the Docusaurus docs directory
print("Loading documents from Docusaurus docs directory...")
docs_path = "../website/docs"  # Adjust this path based on your directory structure
if os.path.exists(docs_path):
    documents = SimpleDirectoryReader(
        input_dir=docs_path,
        required_exts=[".md", ".mdx"]
    ).load_data()
    print(f"Loaded {len(documents)} documents from {docs_path}")

    # Add metadata to documents to track their source
    for doc in documents:
        # Extract chapter info from filename
        filename = os.path.basename(doc.metadata.get('file_name', ''))
        if filename.startswith('0'):
            chapter_num = filename.split('-')[0]
            doc.metadata['chapter_id'] = chapter_num
            doc.metadata['section_title'] = filename
            doc.metadata['page_number'] = 0  # Placeholder, could extract from content if needed

    # Create index and store in Qdrant
    print("Creating index and storing in Qdrant...")
    index = VectorStoreIndex.from_documents(
        documents,
        service_context=service_context,
        storage_context=storage_context
    )
    print("Indexing completed successfully!")

else:
    print(f"Documents directory {docs_path} does not exist. Creating empty index.")
    # Create an empty index if no documents are found
    index = VectorStoreIndex.from_documents([], service_context=service_context, storage_context=storage_context)

print("Content indexing complete!")