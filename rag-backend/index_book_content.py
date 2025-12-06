"""
Script to index the Physical AI & Humanoid Robotics book content into the vector store
"""
import os
import sys
from dotenv import load_dotenv
from pathlib import Path

# Add the current directory to the path so we can import the main modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

# Import after adding to path
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.storage.storage_context import StorageContext

def index_book_content():
    # Get the GEMINI_API_KEY from environment
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY must be set in the .env file")

    print("Initializing Gemini models...")
    # Initialize the LLM and embedding model
    llm = Gemini(model="gemini-2.0-flash", api_key=gemini_api_key)
    embed_model = GeminiEmbedding(model_name="text-embedding-004", api_key=gemini_api_key)

    # Update global settings
    Settings.llm = llm
    Settings.embed_model = embed_model

    print("Setting up vector store...")
    # Create vector store and storage context
    vector_store = SimpleVectorStore()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print("Looking for book content...")
    # Find the book content - look for the docs directory in the website
    docs_dir = Path("../../website/docs")
    if not docs_dir.exists():
        # Try alternative locations
        docs_dir = Path("../website/docs")
        if not docs_dir.exists():
            docs_dir = Path("../../../website/docs")
            if not docs_dir.exists():
                print("Could not find docs directory. Looking for markdown files in common locations...")
                # Search for markdown files recursively
                md_files = list(Path("..").rglob("*.md"))
                if not md_files:
                    print("No markdown files found for indexing")
                    return
                # Use parent directory of first markdown file as source
                source_dir = md_files[0].parent
                print(f"Using {source_dir} as source directory")
            else:
                source_dir = docs_dir
        else:
            source_dir = docs_dir
    else:
        source_dir = docs_dir

    print(f"Reading documents from {source_dir}...")
    # Read all markdown files from the docs directory
    reader = SimpleDirectoryReader(
        input_dir=str(source_dir),
        required_exts=['.md', '.mdx'],
        recursive=True
    )

    print("Loading documents...")
    documents = reader.load_data()
    print(f"Loaded {len(documents)} documents")

    if len(documents) == 0:
        print("No documents found to index. Checking for specific book content...")
        # Look specifically for the book chapters
        book_dirs = [
            Path("../../website/docs"),
            Path("../../../website/docs"),
            Path("../../humaniod-robotics-book/website/docs"),
            Path("../humaniod-robotics-book/website/docs")
        ]

        for book_dir in book_dirs:
            if book_dir.exists():
                print(f"Checking {book_dir} for book content...")
                reader = SimpleDirectoryReader(
                    input_dir=str(book_dir),
                    required_exts=['.md', '.mdx'],
                    recursive=True
                )

                documents = reader.load_data()
                if len(documents) > 0:
                    print(f"Found {len(documents)} documents in {book_dir}")
                    break

    if len(documents) == 0:
        print("No book content found to index. Creating a minimal index with sample content.")
        # Create minimal content for the index to work
        from llama_index.core import Document

        # Sample content about Physical AI & Humanoid Robotics
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

        documents = [Document(text=sample_content, metadata={"source": "book_intro"})]
        print(f"Created sample document with book content.")

    print("Creating index with documents...")
    # Create the index with documents
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )

    # Save the index to a file for reuse
    import pickle
    with open('vector_index.pkl', 'wb') as f:
        pickle.dump(index, f)

    print("Book content indexed successfully!")
    print(f"Indexed {len(documents)} documents")
    print("Index saved to vector_index.pkl")

if __name__ == "__main__":
    index_book_content()