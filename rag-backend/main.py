from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
import os

# LlamaIndex imports
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.storage.storage_context import StorageContext

# Load environment variables from .env file
load_dotenv()

app = FastAPI(
    title="RAG Chatbot API",
    version="1.0.0",
    description="API for interacting with the Retrieval Augmented Generation chatbot for the Physical AI & Humanoid Robotics Book."
)

# Add CORS middleware to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Environment Variables ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") # For Google Gemini LLM and Embeddings

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY must be set in the .env file for LLM and embeddings")

# --- LlamaIndex Setup ---
import pickle
from llama_index.core import Settings

# Configure LLM (using Google Gemini Flash as requested)
llm = Gemini(model="gemini-2.0-flash", api_key=GEMINI_API_KEY)
embed_model = GeminiEmbedding(model_name="text-embedding-004", api_key=GEMINI_API_KEY)

# Configure LlamaIndex Settings (modern approach)
Settings.llm = llm
Settings.embed_model = embed_model

# Load the pre-built index from file
try:
    with open('vector_index.pkl', 'rb') as f:
        index = pickle.load(f)
    print("Loaded existing vector index from vector_index.pkl")
except FileNotFoundError:
    print("Vector index file not found. Creating a new index with sample content.")

    # Create vector store and storage context
    vector_store = SimpleVectorStore()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Create a minimal index with some sample content about robotics
    from llama_index.core import Document

    sample_content = """
    # Physical AI & Humanoid Robotics — Essentials

    ## Chapter 1: Introduction to Physical AI
    Physical AI refers to artificial intelligence systems that interact with the physical world through embodiment.
    Unlike traditional AI that operates in digital environments, Physical AI systems have a physical form that
    enables them to perceive, reason, and act in real-world environments.

    Key characteristics include:
    - Embodiment: Having a physical form for environment interaction
    - Perception: Sensing the physical world through various sensors
    - Cognition: Processing information and making real-time decisions
    - Action: Executing physical actions through actuators

    ## Chapter 2: Basics of Humanoid Robotics
    Humanoid robotics focuses on creating robots with human-like form and capabilities.
    These robots typically have:
    - Bipedal locomotion systems
    - Multi-jointed limbs resembling human arms and legs
    - Sensory systems for vision, hearing, and touch
    - Control systems for coordinated movement

    ## Chapter 3: ROS2 Fundamentals
    Robot Operating System 2 (ROS2) provides frameworks for robotics development.
    It offers:
    - Message passing between processes
    - Hardware abstraction
    - Device drivers
    - Libraries for common robotics functions

    ## Chapter 4: Digital Twin Simulation
    Digital twins create virtual replicas of physical robots for:
    - Safe development and testing
    - Behavior prediction
    - Performance optimization
    - Failure analysis in virtual environments

    ## Chapter 5: Vision-Language-Action Systems
    Modern robotics integrates:
    - Computer vision for perception
    - Natural language processing for communication
    - Motor control for action execution
    This enables robots to understand and respond to human commands.

    ## Chapter 6: Capstone AI Robot Pipeline
    A complete AI robot pipeline integrates all components:
    - Perception systems
    - Decision making
    - Motion planning
    - Actuation
    - Learning mechanisms
    """

    sample_docs = [Document(text=sample_content)]
    index = VectorStoreIndex.from_documents(sample_docs)

# --- Pydantic Models (matching rag_api.yaml) ---
class ContextReference(BaseModel):
    chapter_id: str
    section_title: str
    page_number: int

class QueryRequest(BaseModel):
    query: str
    user_id: str
    session_id: Optional[str] = None

class QueryResponse(BaseModel):
    response: str
    context_references: List[ContextReference]


@app.post("/query", response_model=QueryResponse)
async def submit_query(request: QueryRequest):
    """Submit a natural language query to the RAG chatbot"""
    if not request.query or not request.user_id:
        raise HTTPException(status_code=400, detail="Query and user_id are required.")

    try:
        # Create a query engine from the index
        query_engine = index.as_query_engine(similarity_top_k=5, response_mode="compact")

        # Query the engine
        response_obj = query_engine.query(request.query)

        # Extract references from the response source nodes if they exist
        context_references = []
        if hasattr(response_obj, 'source_nodes'):
            for node in response_obj.source_nodes:
                # Assuming metadata contains chapter_id, section_title, page_number
                # This will need to be correctly populated during content indexing
                metadata = node.metadata
                context_references.append(
                    ContextReference(
                        chapter_id=metadata.get("chapter_id", "unknown"),
                        section_title=metadata.get("section_title", "unknown"),
                        page_number=metadata.get("page_number", 0),
                    )
                )

        return QueryResponse(response=str(response_obj), context_references=context_references)
    except Exception as e:
        # Log the error for debugging
        print(f"Error in query endpoint: {str(e)}")
        # Return a meaningful response even if RAG fails
        fallback_response = "I'm sorry, I couldn't find specific information about that topic in the book. However, I'm here to help answer your questions about Physical AI & Humanoid Robotics. Could you try rephrasing your question?"
        return QueryResponse(response=fallback_response, context_references=[])

# --- Health Check Endpoint (Optional but Recommended) ---
@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "FastAPI RAG Chatbot is running."}

