import sys
import os
sys.path.insert(0, ".")          # lets Python find src/config.py
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings import FastEmbedEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_core.documents import Document
import config

def load_document(file_path: str) -> list[Document]:
    print(f"\n📄 Parsing: {file_path}")

    # Disable OCR — our PDF has real selectable text, no OCR needed.
    # This skips the OCR engine initialisation that was causing the error.
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = False

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        }
    )

    result = converter.convert(file_path)
    text = result.document.export_to_markdown()

    doc = Document(
        page_content=text,
        metadata={"source": os.path.basename(file_path)}
    )
    print(f"   Extracted {len(text)} characters")
    return [doc]

def split_into_chunks(docs: list[Document]) -> list[Document]:
    """
    Cut documents into smaller chunks.
    WHY RecursiveCharacterTextSplitter?
    - Tries to split on paragraphs first, then sentences, then words.
    - Smarter than splitting at fixed character positions.
    - Respects natural language boundaries wherever possible.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")
    return chunks


def store_in_qdrant(chunks: list[Document]):
    """
    Convert chunks to vectors and store in Qdrant cloud.
    WHY FastEmbedEmbeddings?
    - fastembed runs locally — no API calls, no cost, no rate limits
    - BAAI/bge-small-en-v1.5 is downloaded once and cached
    - Qdrant built this — it's designed to pair with Qdrant
    """
    print(f"Embedding {len(chunks)} chunks with fastembed...")
    print("(First run downloads the model ~134MB — takes a moment)")

    embedding_model = FastEmbedEmbeddings(
        model_name=config.EMBED_MODEL
    )

    # Create Qdrant client (connects to your cloud instance)
    client = QdrantClient(
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY
    )

    # Create collection if it doesn't exist yet.
    # A collection = a table. We define the vector size (384 for bge-small)
    # and the distance metric (Cosine = measures angle between vectors).
    # Cosine similarity is standard for semantic search.
    collections = [c.name for c in client.get_collections().collections]
    if config.COLLECTION_NAME not in collections:
        print(f"Creating collection: {config.COLLECTION_NAME}")
        client.create_collection(
            collection_name=config.COLLECTION_NAME,
            vectors_config=VectorParams(
                size=384,              # bge-small-en-v1.5 output dimension
                distance=Distance.COSINE
            )
        )

    # QdrantVectorStore.from_documents() does two things:
    # 1. Calls fastembed to convert each chunk to a vector
    # 2. Uploads all vectors + original text to Qdrant cloud
    QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embedding_model,
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
        collection_name=config.COLLECTION_NAME,
        force_recreate=False    # don't wipe existing data — append instead
    )
    print(f"Stored in Qdrant collection: {config.COLLECTION_NAME}")


def ingest(file_path: str):
    """Main function — orchestrates the full pipeline."""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        sys.exit(1)

    docs   = load_document(file_path)
    chunks = split_into_chunks(docs)
    store_in_qdrant(chunks)
    print(f"Ingestion complete — {len(chunks)} chunks ready to query")


# This block only runs when you execute this file directly:
#   python src/ingest.py data/file.pdf
# It does NOT run when another file imports ingest.py
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/ingest.py <path-to-file>")
        sys.exit(1)
    ingest(sys.argv[1])
