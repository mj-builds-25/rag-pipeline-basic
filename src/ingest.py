# src/ingest.py
import sys
import os
sys.path.insert(0, ".")

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from fastembed import TextEmbedding
from typing import List
import config


# ── Custom fastembed wrapper ─────────────────────────────────────
# WHY write our own wrapper?
# langchain_community's FastEmbedEmbeddings has a pickling bug.
# HuggingFaceEmbeddings pulls in PyTorch (~2GB). 
# This wrapper uses fastembed directly — lightweight, no bugs, same model.
# It implements LangChain's Embeddings interface so it works everywhere.
class FastEmbedWrapper(Embeddings):
    def __init__(self, model_name: str):
        self.model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Converts a list of strings → list of vectors
        return [list(v) for v in self.model.embed(texts)]

    def embed_query(self, text: str) -> List[float]:
        # Converts a single query string → one vector
        return list(self.model.embed([text]))[0].tolist()


def load_document(file_path: str) -> list[Document]:
    print(f"\n📄 Parsing: {file_path}")

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False             # skip OCR — text PDF
    pipeline_options.do_table_structure = False  # skip table model (needs libGL)

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
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    print(f"   Split into {len(chunks)} chunks")
    return chunks


def store_in_qdrant(chunks: list[Document]):
    print(f"\n🧮 Embedding {len(chunks)} chunks with fastembed...")

    embedding_model = FastEmbedWrapper(model_name=config.EMBED_MODEL)

    client = QdrantClient(
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY
    )

    collections = [c.name for c in client.get_collections().collections]
    if config.COLLECTION_NAME not in collections:
        print(f"   Creating collection: {config.COLLECTION_NAME}")
        client.create_collection(
            collection_name=config.COLLECTION_NAME,
            vectors_config=VectorParams(
                size=384,
                distance=Distance.COSINE
            )
        )

    QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embedding_model,
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
        collection_name=config.COLLECTION_NAME,
        force_recreate=False
    )
    print(f"   ✅ Stored in Qdrant collection: {config.COLLECTION_NAME}")


def ingest(file_path: str):
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        sys.exit(1)

    docs   = load_document(file_path)
    chunks = split_into_chunks(docs)
    store_in_qdrant(chunks)
    print(f"\n✅ Ingestion complete — {len(chunks)} chunks ready to query")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/ingest.py <path-to-file>")
        sys.exit(1)
    ingest(sys.argv[1])