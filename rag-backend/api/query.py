import os
import pickle
from typing import List, Optional
from pydantic import BaseModel
from llama_index.core import VectorStoreIndex, Settings
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from fastapi import HTTPException
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# --- Environment Variables ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY must be set in the environment for LLM and embeddings")

# --- LlamaIndex Setup ---
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

# --- Pydantic Models ---
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

# Vercel serverless function
def handler(request):
    if request.method == "POST":
        try:
            import json
            from urllib.parse import urlparse

            # Parse the request body
            body = json.loads(request.body.decode('utf-8'))
            query = body.get('query', '')
            user_id = body.get('user_id', '')
            session_id = body.get('session_id', None)

            if not query or not user_id:
                return {
                    'statusCode': 400,
                    'headers': {
                        'Content-Type': 'application/json',
                        'Access-Control-Allow-Origin': '*',
                        'Access-Control-Allow-Methods': 'POST',
                        'Access-Control-Allow-Headers': 'Content-Type',
                    },
                    'body': json.dumps({'error': 'Query and user_id are required.'})
                }

            # Create a query engine from the index
            query_engine = index.as_query_engine(similarity_top_k=5, response_mode="compact")

            # Query the engine
            response_obj = query_engine.query(query)

            # Extract references from the response source nodes if they exist
            context_references = []
            if hasattr(response_obj, 'source_nodes'):
                for node in response_obj.source_nodes:
                    # Assuming metadata contains chapter_id, section_title, page_number
                    metadata = node.metadata
                    context_references.append(
                        {
                            'chapter_id': metadata.get("chapter_id", "unknown"),
                            'section_title': metadata.get("section_title", "unknown"),
                            'page_number': metadata.get("page_number", 0),
                        }
                    )

            response_data = {
                'response': str(response_obj),
                'context_references': context_references
            }

            return {
                'statusCode': 200,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'POST',
                    'Access-Control-Allow-Headers': 'Content-Type',
                },
                'body': json.dumps(response_data)
            }
        except Exception as e:
            # Log the error for debugging
            print(f"Error in query endpoint: {str(e)}")
            # Return a meaningful response even if RAG fails
            fallback_response = "I'm sorry, I couldn't find specific information about that topic in the book. However, I'm here to help answer your questions about Physical AI & Humanoid Robotics. Could you try rephrasing your question?"
            response_data = {
                'response': fallback_response,
                'context_references': []
            }

            return {
                'statusCode': 500,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Headers': 'Content-Type',
                },
                'body': json.dumps(response_data)
            }
    else:
        return {
            'statusCode': 405,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'POST',
                'Access-Control-Allow-Headers': 'Content-Type',
            },
            'body': json.dumps({'error': 'Method not allowed'})
        }

# Export the handler for Vercel
def main(request):
    return handler(request)