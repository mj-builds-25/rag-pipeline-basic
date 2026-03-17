# src/query.py
import sys
sys.path.insert(0, ".")

from langchain_qdrant import QdrantVectorStore
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.embeddings import Embeddings
from fastembed import TextEmbedding
from typing import List
import config


# ── Same wrapper as ingest.py ────────────────────────────────────
# MUST use the same model as ingestion — consistency is critical.
# Different models produce incompatible vectors → wrong search results.
class FastEmbedWrapper(Embeddings):
    def __init__(self, model_name: str):
        self.model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [list(v) for v in self.model.embed(texts)]

    def embed_query(self, text: str) -> List[float]:
        return list(self.model.embed([text]))[0].tolist()


def build_retriever():
    embedding_model = FastEmbedWrapper(model_name=config.EMBED_MODEL)

    vectorstore = QdrantVectorStore.from_existing_collection(
        embedding=embedding_model,
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
        collection_name=config.COLLECTION_NAME,
    )

    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": config.TOP_K}
    )


def build_prompt():
    template = """You are a helpful assistant that answers questions
strictly based on the provided document context.

RULES:
- Answer ONLY using information from the context below.
- If the answer is not in the context, say:
  "I couldn't find that in the provided documents."
- Do not make up information or use outside knowledge.
- Cite which part of the context supports your answer when possible.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""
    return ChatPromptTemplate.from_template(template)


def format_docs(docs) -> str:
    return "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in docs
    )


def build_rag_chain(retriever):
    llm = ChatOpenAI(
    model=config.LLM_MODEL,
    api_key=config.GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
    temperature=0,
    )

    prompt = build_prompt()

    chain = (
        {
            "context":  retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def ask(question: str, chain, retriever) -> dict:
    source_docs = retriever.invoke(question)
    answer = chain.invoke(question)

    return {
        "answer": answer,
        "sources": [
            {
                "source": doc.metadata.get("source", "unknown"),
                "preview": doc.page_content[:120].replace("\n", " ")
            }
            for doc in source_docs
        ]
    }


def main():
    print("\n=== RAG Pipeline — Query Mode ===")
    print("Building retriever and chain...", end=" ", flush=True)
    retriever = build_retriever()
    chain     = build_rag_chain(retriever)
    print("Ready!\n")

    while True:
        question = input("Your question (or 'quit'): ").strip()
        if question.lower() in ("quit", "q", "exit"):
            break
        if not question:
            continue

        result = ask(question, chain, retriever)

        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nRetrieved from:")
        for s in result["sources"]:
            print(f"   [{s['source']}] {s['preview']}...")
        print()


if __name__ == "__main__":
    main()