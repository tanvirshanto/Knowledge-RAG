# Medical RAG Service (Docling + BGE-M3 + Qdrant + Gemini)

Production FastAPI backend for medical textbook RAG: local Docling PDF parsing (with OCR), BGE-M3 embeddings, Qdrant Cloud vectors, and Google Gemini for reasoning.

## Quick start

```bash
cd "e:\Projects\RAG AI"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
# Edit .env with your API keys
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

For production throughput:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

> **Note:** Job status is stored in-process. With multiple workers, poll `/status/{job_id}` on the same worker or use a single worker for ingestion until you add Redis/shared state.

## Frontend (Next.js)

UI for PDF upload, ingestion status polling, and streamed Q&A.

```powershell
cd "e:\Projects\RAG AI\frontend"
copy .env.local.example .env.local
# NEXT_PUBLIC_API_URL=http://localhost:8000
npm install
npm run dev
```

Open http://localhost:3000 (API must be running on port 8000).

| Area | Features |
|------|----------|
| Upload | Drag-and-drop PDF → `/upload-pdf` |
| Status | Live pipeline progress per job (polls every 2s) |
| Ask | Streaming answers via SSE (`/ask?stream=true`) |

Job history is stored in `localStorage` so refreshes keep the list (status re-polls only for active jobs).

## Docker

### Prerequisites

1. [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows)
2. Copy and fill `.env`:

```powershell
cd "e:\Projects\RAG AI"
copy .env.example .env
# Edit .env with your API keys
```

### Run with Compose (recommended)

API + UI:

```powershell
docker compose up --build
```

- API: http://localhost:8000  
- UI: http://localhost:3000  

Detached:

```powershell
docker compose up --build -d
docker compose logs -f api
```

Stop:

```powershell
docker compose down
```

API: http://localhost:8000 — docs at http://localhost:8000/docs

### Run with Docker only

```powershell
docker build -t medical-rag .
docker run --rm -p 8000:8000 --env-file .env -v rag_uploads:/app/tmp_uploads medical-rag
```

### Notes

- The image runs **one uvicorn worker** so `/status/{job_id}` stays consistent with background ingestion.
- Upload temp files use volume `rag_uploads` (Compose) so PDFs survive container restarts until processing finishes.
- Rebuild after code changes: `docker compose up --build`.

## API

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/upload-pdf` | Upload PDF; background parse → chunk → embed → Qdrant |
| `GET` | `/status/{job_id}` | Ingestion progress |
| `POST` | `/ask` | RAG question answering |
| `GET` | `/health` | Liveness |

### Upload PDF

```bash
curl -X POST "http://localhost:8000/upload-pdf" -F "file=@textbook.pdf"
```

```json
{ "job_id": "abc123...", "status": "Queued" }
```

### Status

```bash
curl "http://localhost:8000/status/{job_id}"
```

States: `Queued` → `Parsing...` → `Chunking...` → `Embedding...` → `Indexing...` → `Completed` | `Failed`

### Ask (streaming default)

SSE stream (`stream=true`, default):

```bash
curl -N -X POST "http://localhost:8000/ask?stream=true" \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"What are the CT patterns of pneumonia?\"}"
```

Events: `start` → `token` (repeated) → `done`. Each line is `data: {"type": "...", ...}`.

Non-streaming JSON (`stream=false`):

```bash
curl -X POST "http://localhost:8000/ask?stream=false" \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"What are the CT patterns of pneumonia?\"}"
```

## Configuration

See `.env.example`. Important settings:

- `EMBEDDING_MODEL`: default `BAAI/bge-m3` (1024-dim, same model for ingest and query)
- `EMBEDDING_BATCH_SIZE`: encode batch size (default `32`; lower if CPU RAM is tight)
- `CHUNK_SIZE` / `CHUNK_OVERLAP`: context-injected chunks (defaults `1000` / `150`)
- `RETRIEVAL_TOP_K`: 5–8 recommended (default `6`)
- `GEMINI_API_KEY`: from [Google AI Studio](https://aistudio.google.com/app/apikey)
- `GEMINI_MODEL`: default `gemini-2.0-flash`
- `LLM_TEMPERATURE`: `0.1` for low hallucination risk

## Architecture

```
PDF → Docling (markdown + OCR) → context-injected markdown chunking
    → BGE-M3 embed (document) → Qdrant (cosine, 1024-dim)
Query → BGE-M3 embed (query) → Qdrant top-k → Gemini LLM (context-only prompt)
```

Temporary PDFs are deleted in a `finally` block after ingestion completes or fails.
