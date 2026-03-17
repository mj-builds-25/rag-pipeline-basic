# RAG Pipeline — Basic

A production-style RAG (Retrieval-Augmented Generation) pipeline
that answers natural language questions from any PDF document.


## What It Does

Upload a PDF → ask questions in plain English → get grounded answers
with source citations. The LLM can only answer from the document —
no hallucination, no outside knowledge.

## Architecture
```
PDF → [Docling] → Clean Text
                       ↓
            [RecursiveCharacterTextSplitter]
                       ↓ 433 chunks
            [fastembed — BAAI/bge-small-en-v1.5]
                       ↓ 384-dim vectors
              [Qdrant Cloud — cosine similarity]
                       ↓ top-4 chunks retrieved
         [LangChain LCEL chain + anti-hallucination prompt]
                       ↓
           [Groq — Llama 3.1 8B via OpenAI-compatible API]
                       ↓
              Grounded answer + source citations
```

## Tech Stack

| Tool | Role | Why |
|------|------|-----|
| Docling (IBM) | PDF parser | Handles complex layouts, tables, headings |
| fastembed | Local embeddings | Free, no API key, no PyTorch overhead |
| Qdrant Cloud | Vector database | Production-grade, free tier, cloud-hosted |
| LangChain | Pipeline orchestration | LCEL chain, prompt management |
| Groq + Llama 3.1 8B | LLM inference | Free tier, fast, OpenAI-compatible API |
| uv | Package manager | 10-100x faster than pip |

## Project Structure
```
rag-pipeline-basic/
├── data/                  ← drop PDF files here
├── src/
│   ├── config.py          ← all settings (models, chunk size, keys)
│   ├── ingest.py          ← PDF → chunks → vectors → Qdrant
│   └── query.py           ← question → search → LLM → answer
├── .env.example           ← API key template
├── requirements.txt       ← pinned dependencies
└── README.md
```

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/mj-builds-25/rag-pipeline-basic
cd rag-pipeline-basic
```

### 2. Install uv (if not installed)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

### 3. Create and activate virtual environment
```bash
uv venv
source .venv/bin/activate        # Mac / Linux / Codespaces
source .venv/Scripts/activate    # Git Bash on Windows
```

### 4. Install dependencies
```bash
uv pip install -r requirements.txt
```

### 5. Set up environment variables
```bash
cp .env.example .env
# Edit .env and add your keys:
# GROQ_API_KEY    → console.groq.com
# QDRANT_URL      → cloud.qdrant.io → your cluster URL
# QDRANT_API_KEY  → cloud.qdrant.io → API Keys tab
```

### 6. Ingest a PDF
```bash
python src/ingest.py data/your-file.pdf
```

### 7. Query it
```bash
python src/query.py
```

## Example Output
```
=== RAG Pipeline — Query Mode ===

Your question: What is artificial intelligence?

Answer:
Artificial intelligence (AI) is the capability of computational
systems to perform tasks typically associated with human intelligence,
such as learning, reasoning, problem-solving, and decision-making.

Retrieved from:
   [ai-wikipedia.pdf] Artificial intelligence (AI) is the capability
   of computational systems to perform tasks typically associated...
   [ai-wikipedia.pdf] McCarthy defines intelligence as the
   computational part of the ability to achieve goals...
```

## Key Design Decisions

**Why fastembed over OpenAI embeddings?**
Zero API cost, runs locally, no rate limits. Same quality for RAG use cases.

**Why Groq via OpenAI-compatible endpoint?**
`langchain-groq` had version conflicts with the groq SDK. Using
`langchain-openai` pointed at Groq's OpenAI-compatible API is
more stable and equally performant.

**Why disable OCR and table structure in Docling?**
The Wikipedia test PDF has selectable text — OCR is unnecessary and
requires system libraries not available in Codespaces. Table structure
detection requires OpenCV (libGL). 

**Anti-hallucination prompt:**
The system prompt explicitly instructs the LLM to answer ONLY from
the retrieved context. If the answer isn't in the document, it says so.

## What I Learned

- How RAG pipelines work end-to-end
- Why the same embedding model must be used for ingest and query
- How vector similarity search works (cosine distance)
- How LangChain LCEL chains compose components with the pipe operator
- Real-world dependency version conflict debugging
- How Groq's OpenAI-compatible API works as a drop-in replacement

