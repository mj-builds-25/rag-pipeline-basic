import sys
sys.path.insert(0, ".")

from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import config

def build_retriever():
    """
    Connect to the existing Qdrant collection and return a retriever.
    WHY build this once?  The retriever is stateless — safe to reuse
    across multiple queries without reconnecting each time.
    """
    embedding_model = FastEmbedEmbeddings(model_name=config.EMBED_MODEL)

    # Connect to the EXISTING collection (ingest.py must have run first)
    vectorstore = QdrantVectorStore.from_existing_collection(
        embedding=embedding_model,
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
        collection_name=config.COLLECTION_NAME,
    )

    # as_retriever() wraps the vectorstore in a standard interface.
    # search_kwargs={"k": TOP_K} means "return top 4 results per query"
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": config.TOP_K}
    )


def build_prompt():
    """
    The system prompt is the MOST IMPORTANT part of a RAG pipeline.
    WHY this exact wording?
    - 'ONLY use the context below' prevents hallucination.
    - 'If not in context, say so' prevents confident wrong answers.
    - Providing context + question in structure helps the LLM parse it.
    Without this guardrail, LLMs will happily invent plausible-sounding
    answers from their training data — not from your document.
    """
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
    """
    Join retrieved chunks into one string for the prompt.
    WHY format_docs?  The retriever returns a list of Document objects.
    The prompt expects a single string. This bridges that gap.
    We also add a separator so the LLM can tell chunks apart.
    """
    return "

---

".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}]
{doc.page_content}"
        for doc in docs
    )


def build_rag_chain(retriever):
    """
    Build the full RAG chain using LangChain Expression Language (LCEL).
    WHY LCEL (the | pipe syntax)?
    It's LangChain's modern way to compose components declaratively.
    Each step's output becomes the next step's input.
    It's also easy to swap any piece (e.g., change the LLM) in one line.
    """
    llm = ChatGroq(
        model=config.LLM_MODEL,
        api_key=config.GROQ_API_KEY,
        temperature=0,      # 0 = deterministic. For RAG: always use 0.
                            # Higher temp = more creative but less factual.
    )

    prompt = build_prompt()

    # Chain breakdown (read left to right):
    # 1. {"context": retriever | format_docs, "question": passthrough}
    #    → retriever searches Qdrant, format_docs joins results
    #    → passthrough passes the question through unchanged
    # 2. | prompt  → fills {context} and {question} into the template
    # 3. | llm     → sends completed prompt to Groq
    # 4. | StrOutputParser()  → extracts just the text from the response
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
    """Run one question through the chain and return answer + sources."""
    # Get source docs for transparency
    source_docs = retriever.invoke(question)

    # Run the full chain
    answer = chain.invoke(question)

    return {
        "answer": answer,
        "sources": [
            {
                "source": doc.metadata.get("source", "unknown"),
                "preview": doc.page_content[:120].replace("
", " ")
            }
            for doc in source_docs
        ]
    }


def main():
    """Interactive query loop."""
    print("
=== RAG Pipeline — Query Mode ===")
    print("Building retriever and chain...", end=" ", flush=True)
    retriever = build_retriever()
    chain     = build_rag_chain(retriever)
    print("Ready!
")

    while True:
        question = input("Your question (or 'quit'): ").strip()
        if question.lower() in ("quit", "q", "exit"):
            break
        if not question:
            continue

        result = ask(question, chain, retriever)

        print(f"
Answer:
{result['answer']}")
        print(f"Retrieved from:")
        for s in result["sources"]:
            print(f"   [{s['source']}] {s['preview']}...")
        print()


if __name__ == "__main__":
    main()