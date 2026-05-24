# Medical RAG Service (Docling + BGE-M3 + Qdrant + Gemini)

Production FastAPI backend for medical textbook RAG: local Docling PDF parsing (with OCR), BGE-M3 embeddings, Qdrant Cloud vectors, and Google Gemini for reasoning. Includes JWT authentication, RBAC, user management, and a background ingestion worker — all backed by Supabase.

## Prerequisites

- Python 3.10+
- [Supabase](https://supabase.com) project
- Qdrant Cloud instance
- Google Gemini API key

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

## Database Setup

Before starting the application, run this SQL in your Supabase SQL Editor:

```sql
CREATE TABLE IF NOT EXISTS users (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'user',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS upload_jobs (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    original_filename TEXT NOT NULL,
    uploaded_by TEXT,
    status TEXT NOT NULL DEFAULT 'QUEUED',
    error_message TEXT,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    total_pages INTEGER,
    total_chunks INTEGER
);

CREATE TABLE IF NOT EXISTS system_logs (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    level TEXT NOT NULL,
    message TEXT NOT NULL,
    traceback TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

The application will automatically seed the default admin user on first startup:
- Username: `admin`
- Password: `Admin@1234`
- Role: `maintainer`

## API Reference

### Authentication

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/auth/login` | No | Login, returns JWT token |

### Upload (Maintainer only)

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/uploads/upload-pdf` | Bearer | Upload PDF(s), creates QUEUED jobs |
| `GET` | `/uploads` | Bearer | List upload jobs (filter by `?status=`) |
| `GET` | `/uploads/running` | Bearer | Get currently running job |
| `GET` | `/uploads/{id}` | Bearer | Get single job details |

### Users (Maintainer only)

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/users` | Bearer | Create user |
| `GET` | `/users` | Bearer | List all users |
| `GET` | `/users/{id}` | Bearer | Get user by ID |
| `PATCH` | `/users/{id}` | Bearer | Update user (role, active status, password) |
| `DELETE` | `/users/{id}` | Bearer | Delete user |

### Ask (Authenticated — both roles)

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/ask` | Bearer | RAG question answering (SSE streaming by default) |

### Health

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/health` | No | Liveness check |

## Authentication

All protected endpoints require a Bearer token obtained via `/auth/login`:

```bash
# Login
curl -X POST "http://localhost:8000/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "Admin@1234"}'

# Response: {"access_token": "...", "token_type": "bearer"}

# Use token for protected endpoints
TOKEN="your-access-token"
curl -H "Authorization: Bearer $TOKEN" "http://localhost:8000/users"
```

### Roles

| Role | Permissions |
|------|-------------|
| `maintainer` | Upload PDFs, manage uploads, run ingestion, view statuses, manage users |
| `user` | Ask questions only |

## Upload Pipeline

1. `POST /uploads/upload-pdf` accepts one or more PDF files
2. Files are saved locally and a `QUEUED` job record is created in Supabase
3. A background worker polls for QUEUED jobs and processes them **one at a time**
4. Ingestion: Parse → Chunk → Embed → Index → Complete
5. Status is updated in Supabase at each stage

**States:** `QUEUED` → `RUNNING` → `COMPLETED` | `FAILED`

```bash
# Upload single or multiple PDFs
curl -X POST "http://localhost:8000/uploads/upload-pdf" \
  -H "Authorization: Bearer $TOKEN" \
  -F "files=@textbook1.pdf" \
  -F "files=@textbook2.pdf"

# Check status
curl -H "Authorization: Bearer $TOKEN" \
  "http://localhost:8000/uploads/{job_id}"
```

## Ask (Streaming)

SSE stream (default):

```bash
curl -N -X POST "http://localhost:8000/ask?stream=true" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the CT patterns of pneumonia?"}'
```

Non-streaming JSON:

```bash
curl -X POST "http://localhost:8000/ask?stream=false" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the CT patterns of pneumonia?"}'
```

## Docker

### Prerequisites

1. Copy `.env.example` to `.env` and fill in all required values
2. Run the SQL from the Database Setup section in Supabase

### Run with Compose

```bash
docker compose up --build
```

- API: http://localhost:8000
- Docs: http://localhost:8000/docs
- UI: http://localhost:3000 (if frontend is present)

Detached:

```bash
docker compose up --build -d
docker compose logs -f api
```

### Notes

- The image runs **one uvicorn worker** — the background ingestion worker runs in-process via asyncio
- Upload temp files use the `rag_uploads` Docker volume for persistence
- The worker processes **one file at a time** sequentially, no parallel ingestion

## Configuration

See `.env.example`. All settings:

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `SUPABASE_URL` | — | Yes | Supabase project URL |
| `SUPABASE_SERVICE_KEY` | — | Yes | Supabase service role key |
| `JWT_SECRET` | — | Yes | Secret for signing JWT tokens |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `30` | No | JWT token expiry |
| `QDRANT_URL` | — | Yes | Qdrant Cloud URL |
| `QDRANT_API_KEY` | — | Yes | Qdrant API key |
| `QDRANT_COLLECTION_NAME` | `medical_textbooks` | No | Vector collection name |
| `GEMINI_API_KEY` | — | Yes | Google Gemini API key |
| `GEMINI_MODEL` | `gemini-2.0-flash` | No | Gemini model name |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | No | Sentence transformer model |
| `CHUNK_SIZE` | `1000` | No | Text chunk size |
| `CHUNK_OVERLAP` | `300` | No | Chunk overlap |
| `RETRIEVAL_TOP_K` | `20` | No | Top-k for retrieval |
| `WORKER_POLL_INTERVAL_SECONDS` | `5` | No | Worker poll interval |
| `MAX_UPLOAD_SIZE_MB` | `50` | No | Max PDF upload size |
| `SEED_ADMIN_USERNAME` | `admin` | No | Default admin username |
| `SEED_ADMIN_PASSWORD` | `Admin@1234` | No | Default admin password |

## Architecture

```
Upload → Local Storage → Supabase (QUEUED)
                                 ↑
Worker polls → RUNNING → run_ingestion_pipeline()
    PDF → Docling (markdown + OCR) → context-injected chunking
    → BGE-M3 embed → Qdrant (cosine, 1024-dim) → COMPLETED

Query → BGE-M3 embed → Qdrant top-k → Gemini LLM (context-only prompt)
```

### Project Structure

```
├── main.py                  # FastAPI app entrypoint, lifespan
├── seed.py                  # Admin user seed
├── api/                     # Route handlers
│   ├── auth.py              # POST /auth/login
│   ├── ask.py               # POST /ask
│   ├── users.py             # CRUD /users
│   ├── uploads.py           # POST /uploads/upload-pdf, GET /uploads/*
│   └── router.py            # Router aggregation
├── auth/                    # Authentication
│   ├── security.py          # bcrypt hashing
│   ├── jwt.py               # Token create/decode
│   └── dependencies.py      # get_current_user, require_role
├── middleware/               # ASGI middleware
│   ├── logging.py           # Request/response logging
│   └── exception.py         # Global exception handlers
├── repositories/            # Supabase data access
│   ├── base.py              # Client singleton + BaseRepository
│   ├── user_repository.py   # users table
│   └── upload_repository.py # upload_jobs table
├── services/                # Business logic
│   ├── user_service.py      # User CRUD + auth
│   └── upload_service.py    # Upload orchestration
├── schemas/                 # Pydantic request/response models
│   ├── auth.py
│   ├── user.py
│   └── upload.py
├── workers/                 # Background processing
│   └── ingestion_worker.py  # Sequential ingestion loop
├── utils/                   # Helpers
│   ├── logging_config.py
│   └── file_storage.py
├── app/                     # Existing RAG pipeline (unchanged)
│   ├── config.py            # Settings + new Supabase/JWT config
│   ├── state.py             # Supabase-backed job tracking wrapper
│   ├── models.py            # JobStatus enum, request/response models
│   ├── pipelines/
│   │   ├── ingestion.py     # run_ingestion_pipeline()
│   │   └── retrieval.py     # retrieve_contexts(), answer_question()
│   └── services/
│       ├── parsing.py       # Docling PDF → Markdown
│       ├── chunking.py      # Markdown chunking with context injection
│       ├── embeddings.py    # BGE-M3 via sentence-transformers
│       ├── llm.py           # Gemini LLM
│       └── vector_store.py  # Qdrant vector store
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── requirements.txt
```
