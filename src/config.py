import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY    = os.getenv("GROQ_API_KEY")
QDRANT_URL      = os.getenv("QDRANT_URL")
QDRANT_API_KEY  = os.getenv("QDRANT_API_KEY")

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "llama-3.1-8b-instant"

COLLECTION_NAME = "rag_docs"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 80
TOP_K = 4