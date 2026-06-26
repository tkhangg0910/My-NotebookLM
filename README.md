# 📚 MyNotebookLM

A local RAG (Retrieval-Augmented Generation) system that transforms your PDF documents into an interactive learning assistant. Ask questions, generate summaries, create quizzes, and build flashcard decks — all grounded in your own documents.

---

## Features

- **Grounded Q&A** — Ask questions and get answers cited directly from your PDFs
- **Summarization** — Generate structured summaries with key points for any document, topic, or the entire corpus
- **Quiz Generation** — Produce multiple-choice quizzes from document content
- **Flashcard Generation** — Create study flashcards automatically
- **Vision/Multimodal RAG** — Optional image-aware retrieval using ColQwen2 (for charts, figures, scanned pages)
- **Multiple LLM backends** — HuggingFace local, Gemini, vLLM, llama.cpp
- **Streamlit UI** + **REST API** + **CLI** — Use whichever interface you prefer

---

## Architecture

```
PDFs
 └─► Indexing (PyPDF + chunking)
      └─► Qdrant (vector store)
           └─► Retrieval + Reranker (BGE CrossEncoder)
                └─► LLM (HF / Gemini / vLLM / llama.cpp)
                     └─► Answer / Summary / Quiz / Flashcards
```

**Key components:**

| Module | Responsibility |
|---|---|
| `indexing.py` | PDF loading, chunking, embedding, Qdrant upsert |
| `store.py` | Qdrant client, embedding model, collection management |
| `rag.py` | Retrieval, reranking, prompt rendering, answer generation |
| `learning.py` | Summarization (map-reduce), quiz & flashcard generation |
| `vision.py` | ColQwen2 image embedding for multimodal retrieval |
| `reranker.py` | BGE CrossEncoder reranker |
| `filters.py` | Metadata filtering (filename, page, section, document_id) |
| `schemas.py` | Pydantic models for all data structures |
| `config.py` | All settings via environment variables / `.env` |
| `api.py` | FastAPI REST endpoints |
| `cli.py` | Typer CLI client |
| `ui.py` | Streamlit web interface |

---

## Installation

**Requirements:** Python 3.10+, CUDA (optional but recommended)

```bash
# Clone the repository
git clone <repo-url>
cd my-notebook-lm

# Install dependencies
pip install -r requirements.txt
```

---

## Configuration

All settings are read from environment variables with the `RAG_` prefix, or from a `.env` file in the project root.

Create a `.env` file:

```dotenv
# LLM provider: hf_local | gemini | vllm | llamacpp
RAG_LLM_PROVIDER=gemini
GOOGLE_API_KEY=your_google_api_key_here

# RAG mode: text_only | text_vl | hybrid
RAG_RAG_MODE=text_only

# Directories
RAG_DATA_DIR=data
RAG_STORAGE_DIR=storage/qdrant
RAG_IMAGE_DIR=storage/images

# Chunking
RAG_CHUNK_SIZE=1000
RAG_CHUNK_OVERLAP=150
RAG_CHUNKER_TYPE=recursive   # recursive | semantic

# Retrieval
RAG_TOP_K=5
RAG_TOP_C=20

# Models
RAG_EMBEDDING_MODEL=GreenNode/GreenNode-Embedding-Large-VN-Mixed-V1
RAG_RERANKER_MODEL=BAAI/bge-reranker-base
RAG_GEMINI_MODEL=gemini-2.5-flash
```

### LLM Provider Options

| Provider | Setting | Notes |
|---|---|---|
| HuggingFace local | `RAG_LLM_PROVIDER=hf_local` | Set `RAG_HF_MODEL`, `RAG_HF_DEVICE` |
| Google Gemini | `RAG_LLM_PROVIDER=gemini` | Requires `GOOGLE_API_KEY` |
| vLLM server | `RAG_LLM_PROVIDER=vllm` | Set `RAG_VLLM_API_BASE` |
| llama.cpp | `RAG_LLM_PROVIDER=llamacpp` | Set `RAG_GGUF_MODEL_PATH` |

### RAG Mode Options

| Mode | Description |
|---|---|
| `text_only` | Text retrieval only (default) |
| `text_vl` | Text + page image context passed to LLM |
| `hybrid` | Text + ColQwen2 visual retrieval |

---

## Quick Start

### 1. Start the API server

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

### 2. Launch the Streamlit UI

```bash
streamlit run src/ui.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### 3. Or use the CLI

```bash
# Upload and index a PDF
python -m src.cli upload path/to/document.pdf

# Ask a question
python -m src.cli ask "What is the main conclusion of the paper?"

# Summarize a document
python -m src.cli summarize --document document.pdf

# Generate a quiz
python -m src.cli quiz --document document.pdf

# Generate flashcards
python -m src.cli flashcards --query "neural networks"
```

---

## REST API

The API runs on `http://localhost:8000`. Interactive docs at `/docs`.

### Endpoints

#### `GET /health`
Health check.

#### `POST /upload`
Upload and index a PDF file.

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@document.pdf"
```

```json
{ "filename": "document.pdf", "chunks_indexed": 142 }
```

#### `POST /ask`
Ask a question over indexed documents.

```json
{
  "question": "What are the key findings?",
  "k": 5,
  "filters": { "filename": "document.pdf" }
}
```

#### `POST /summarize`
Summarize a document, a topic, or the full corpus.

```json
{
  "document": "document.pdf",
  "query": "machine learning methods",
  "k": 12
}
```

#### `POST /quiz`
Generate a multiple-choice quiz.

```json
{
  "query": "neural networks",
  "count": 8
}
```

#### `POST /flashcard`
Generate study flashcards.

```json
{
  "document": "document.pdf",
  "count": 15
}
```

#### `POST /reindex-images`
Re-embed all page images for multimodal retrieval (requires `text_vl` or `hybrid` mode).

---

## Metadata Filtering

All endpoints accept a `filters` object to scope retrieval:

```json
{
  "question": "...",
  "filters": {
    "filename": "lecture_notes.pdf",
    "page": 5
  }
}
```

Supported filter fields: `filename`, `filenames` (list), `page`, `section`, `document_id`.

---

## Summarization Strategy

The summarizer automatically selects a strategy based on the number of retrieved chunks:

- **Single pass** — used when chunks ≤ `RAG_SUMMARIZE_BATCH_SIZE` (default: 10)
- **Map-reduce** — used for larger document sets: each batch is summarized independently, then reduced into a final summary

---

## Project Structure

```
.
├── data/                   # Place your PDF files here
├── storage/
│   ├── qdrant/             # Qdrant vector database
│   └── images/             # Extracted page images
├── src/
│   ├── prompts/            # Jinja2 prompt templates
│   │   ├── answer.j2
│   │   ├── summary_single.j2
│   │   ├── summary_map.j2
│   │   ├── summary_reduce.j2
│   │   ├── quiz.j2
│   │   └── flashcard.j2
│   ├── config.py
│   ├── schemas.py
│   ├── store.py
│   ├── indexing.py
│   ├── rag.py
│   ├── reranker.py
│   ├── learning.py
│   ├── vision.py
│   ├── filters.py
│   ├── export.py
│   ├── api.py
│   ├── cli.py
│   └── ui.py
├── .env
└── README.md
```

---

## Default Models

| Role | Default Model |
|---|---|
| Embedding | `GreenNode/GreenNode-Embedding-Large-VN-Mixed-V1` |
| Reranker | `BAAI/bge-reranker-base` |
| LLM (local) | `Qwen/Qwen3-4B-Instruct-2507` |
| Vision | `vidore/colqwen2-v1.0` |
| LLM (cloud) | `gemini-2.5-flash` |

The embedding model is optimized for Vietnamese and English mixed content.

---

## Notes

- PDFs are automatically discovered from `RAG_DATA_DIR` (recursive scan).
- Re-uploading the same PDF via `/upload` is safe — chunks are upserted by deterministic ID.
- Vision features require a CUDA GPU with sufficient VRAM for ColQwen2.
- The Qdrant database is file-based (no separate server needed).
