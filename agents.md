# Medical RAG Service — Agent Context

## Overview

Production FastAPI backend for medical textbook RAG. Accepts PDF uploads, parses them with Docling (OCR + table extraction), chunks with context-injected breadcrumb headers, embeds via local BGE-M3 (1024-dim), indexes into Qdrant Cloud, and answers questions via Google Gemini with strict context-only prompting. Supports multi-turn chat history with per-user conversations stored in Supabase. JWT auth with RBAC (`maintainer`/`user`), Supabase-backed state, and an in-process background ingestion worker.

**Tech stack:** Python 3.10+, FastAPI, Supabase, Qdrant Cloud, Google Gemini, sentence-transformers (BGE-M3), Docling, langchain_text_splitters, JWT (python-jose), bcrypt, SSE streaming.

## Project Structure

```
├── main.py                  # FastAPI app entrypoint, lifespan (worker start/stop, embedding pre-warm, Qdrant collection ensure)
├── seed.py                  # Admin user seed (disabled in lifespan — runs via cli)
├── api/                     # Route handlers
│   ├── router.py            # Aggregates all routers (/auth, /ask, /users, /uploads, /conversations)
│   ├── auth.py              # POST /auth/login
│   ├── ask.py               # POST /ask (SSE streaming, auto-creates conversation, saves messages)
│   ├── users.py             # CRUD /users (maintainer-only)
│   ├── uploads.py           # POST /uploads/upload-pdf, GET /uploads/*
│   └── conversations.py     # CRUD /conversations (list, get with messages, rename, delete)
├── auth/                    # Authentication
│   ├── security.py          # bcrypt hashing
│   ├── jwt.py               # Token create/decode
│   └── dependencies.py      # get_current_user, require_role, require_maintainer
├── middleware/               # ASGI middleware
│   ├── logging.py           # Request/response logging
│   └── exception.py         # Global exception handlers
├── repositories/            # Supabase data access
│   ├── base.py              # Client singleton + BaseRepository
│   ├── user_repository.py   # users table
│   ├── upload_repository.py # upload_jobs table
│   ├── conversation_repository.py # conversations table
│   └── message_repository.py # messages table
├── services/                # Business logic
│   ├── user_service.py      # User CRUD + auth
│   ├── upload_service.py    # Upload orchestration
│   └── conversation_service.py # Chat history management
├── schemas/                 # Pydantic request/response models
│   ├── auth.py
│   ├── user.py
│   ├── upload.py
│   └── conversation.py      # Conversation + message models
├── workers/
│   └── ingestion_worker.py  # Sequential ingestion loop (poll Supabase)
├── app/                     # Core RAG pipeline
│   ├── config.py            # Pydantic Settings (all env vars)
│   ├── state.py             # Supabase-backed job tracking wrapper
│   ├── models.py            # JobStatus enum, AskRequest/AskResponse
│   ├── pipelines/
│   │   ├── ingestion.py     # run_ingestion_pipeline()
│   │   └── retrieval.py     # retrieve_contexts(), stream_answer_question(), answer_question()
│   └── services/
│       ├── parsing.py       # Docling PDF → Markdown (singleton converter)
│       ├── chunking.py      # Markdown chunking with context injection
│       ├── embeddings.py    # Local BGE-M3 via sentence-transformers (singleton)
│       ├── llm.py           # Gemini LLM with medical system prompt
│       └── vector_store.py  # Qdrant vector store (singleton)
```

## Key Architecture Decisions

1. **BGE-M3 runs locally on CPU** — The SentenceTransformer model loads once at startup and stays in RAM as a class-level singleton (`LocalEmbeddingService._model_instance`). No GPU, no external embedding API.

2. **Context-injected chunking** — Before embedding, every chunk gets a breadcrumb header like `Context: # Chapter > ## Section > ### Subsection` prepended to the text. This is the single most important design choice for retrieval quality. The `clean_text` function is only used for page-matching heuristics, not for the actual chunk text.

3. **Singleton pattern everywhere** — `SentenceTransformer`, `DocumentConverter` (Docling), `QdrantVectorStore`, `LocalEmbeddingService`, and `Supabase Client` are all singletons. Use `get_embedding_service(settings)`, `get_vector_store(settings)`, `get_supabase()` — never instantiate directly.

4. **In-process ingestion worker** — `IngestionWorker` runs via `asyncio.create_task` in the FastAPI lifespan. It polls `upload_jobs` for `QUEUED` records, downloads from Supabase Storage bucket `mediRag`, processes one file at a time sequentially. No separate process/queue.

5. **Supabase Storage for PDFs** — Uploads go directly to Supabase Storage bucket `mediRag` via `get_supabase().storage.from_("mediRag").upload()`. The worker downloads them back when processing. Local `tmp_uploads/` is just a transient staging area.

6. **Retrieval has figure-aware boosting** — If the question mentions figures/pages or diagnosis keywords, the pipeline performs a secondary search for `"Legend for Figure {X}"` and boosts context scores for chunks containing "answer guide" or "legends for introductory figures" by +0.5, then re-sorts.

7. **Multi-turn chat history** — `POST /ask` accepts an optional `conversation_id`. If omitted, a new conversation is auto-created (title = first 50 chars of question). Previous messages (up to `CHAT_HISTORY_MAX_TURNS * 2`) are loaded and sent to Gemini as multi-turn `contents` with alternating `user`/`model` roles. The current question's RAG context is always the final user message. Both user and assistant messages are saved after each interaction. The `conversation_id` is emitted in SSE `start` and `done` events so the frontend can track it.

## Data Flow — Full Pipeline

```
POST /uploads/upload-pdf (maintainer, multipart PDFs)
  → Validate (size, type)
  → Upload to Supabase Storage "mediRag" (path: uploads/{job_id}.pdf)
  → Insert upload_jobs row (status=QUEUED, storage_path set)
  → Return job IDs

IngestionWorker (background, polls every WORKER_POLL_INTERVAL_SECONDS)
  → upload_repository.get_next_queued() (atomically sets QUEUED→RUNNING)
  → Download from Supabase Storage
  → run_ingestion_pipeline():
     1. PARSE: Docling (do_ocr=True, do_table_structure=True) → Markdown + DoclingDocument
     2. CHUNK: normalize_markdown_headers() → MarkdownHeaderTextSplitter (h1/h2/h3)
        → RecursiveCharacterTextSplitter → _inject_context(breadcrumb prefix)
     3. EMBED: Local BGE-M3 (1024-dim, cosine normalized)
     4. INDEX: QdrantCloudVectors.upsert (batch size 64, deterministic UUID via MD5 of filename+chunk_index)
  → Update upload_jobs.status=COMPLETED|FAILED
  → Delete local PDF temp

POST /ask (authenticated, SSE streaming default)
  → Resolve user_id from JWT username via UserRepository
  → If conversation_id provided: verify ownership, load last CHAT_HISTORY_MAX_TURNS*2 messages
  → If no conversation_id: auto-create conversation (title = question[:50])
  → BGE-M3 embed query
  → Qdrant search (top_k=RETRIEVAL_TOP_K, default 20, cosine)
  → Figure-aware secondary search + legend boost if figure/diagnosis mention detected
  → Build Gemini prompt: history messages + RAG context + current question
  → Stream response via SSE (emits conversation_id in start/done events)
  → Save user message + assistant response to messages table

GET /conversations (authenticated)
  → List user's conversations ordered by updated_at DESC

POST /conversations (authenticated)
  → Create conversation with optional title

GET /conversations/{id} (authenticated)
  → Verify ownership
  → Return conversation with all messages (ordered by created_at ASC)

PATCH /conversations/{id} (authenticated)
  → Verify ownership
  → Update title

DELETE /conversations/{id} (authenticated)
  → Verify ownership
  → Delete conversation (cascades to messages via FK)
```

## Supabase Schema

### `users` table
| Column | Type | Notes |
|--------|------|-------|
| id | UUID | PK, gen_random_uuid() |
| username | TEXT | UNIQUE NOT NULL |
| password_hash | TEXT | NOT NULL |
| role | TEXT | NOT NULL DEFAULT 'user' |
| is_active | BOOLEAN | DEFAULT TRUE |
| created_at | TIMESTAMPTZ | DEFAULT NOW() |

### `upload_jobs` table
| Column | Type | Notes |
|--------|------|-------|
| id | TEXT | PK (uuid hex) |
| filename | TEXT | NOT NULL |
| original_filename | TEXT | NOT NULL |
| uploaded_by | TEXT | |
| status | TEXT | NOT NULL DEFAULT 'QUEUED' |
| error_message | TEXT | |
| started_at | TIMESTAMPTZ | |
| completed_at | TIMESTAMPTZ | |
| created_at | TIMESTAMPTZ | DEFAULT NOW() |
| total_pages | INTEGER | |
| total_chunks | INTEGER | |
| storage_path | TEXT | Supabase Storage path |

### `system_logs` table
| Column | Type | Notes |
|--------|------|-------|
| id | UUID | PK |
| level | TEXT | NOT NULL |
| message | TEXT | NOT NULL |
| traceback | TEXT | |
| created_at | TIMESTAMPTZ | DEFAULT NOW() |

### `conversations` table
| Column | Type | Notes |
|--------|------|-------|
| id | UUID | PK, gen_random_uuid() |
| user_id | UUID | FK → users.id, NOT NULL |
| title | TEXT | NOT NULL (auto-generated from first question or user-provided) |
| created_at | TIMESTAMPTZ | DEFAULT NOW() |
| updated_at | TIMESTAMPTZ | DEFAULT NOW(), updated on new message |

### `messages` table
| Column | Type | Notes |
|--------|------|-------|
| id | UUID | PK, gen_random_uuid() |
| conversation_id | UUID | FK → conversations.id ON DELETE CASCADE, NOT NULL |
| role | TEXT | NOT NULL, CHECK IN ('user', 'assistant') |
| content | TEXT | NOT NULL |
| created_at | TIMESTAMPTZ | DEFAULT NOW() |

## Status Flow for Upload Jobs

```
QUEUED → RUNNING → PARSING → CHUNKING → EMBEDDING → INDEXING → COMPLETED
                                                      ↓
                                                  FAILED (at any point)
```

Note: `run_ingestion_pipeline` uses statuses from `app.models.JobStatus` enum (e.g., `"PARSING"`, `"CHUNKING"`), while the worker and repository use `"QUEUED"`, `"RUNNING"`, `"COMPLETED"`, `"FAILED"`. The `update_status` method in `UploadRepository` handles both sets — they're strings in the DB.

## Configuration (all env vars via `app/config.py` `Settings`)

| Variable | Required | Default |
|----------|----------|---------|
| SUPABASE_URL | Yes | — |
| SUPABASE_SERVICE_KEY | Yes | — |
| JWT_SECRET | Yes | — |
| QDRANT_URL | Yes | — |
| QDRANT_API_KEY | Yes | — |
| QDRANT_COLLECTION_NAME | No | medical_textbooks |
| GEMINI_API_KEY | Yes | — |
| GEMINI_MODEL | No | gemini-2.0-flash |
| EMBEDDING_MODEL | No | BAAI/bge-m3 |
| CHUNK_SIZE | No | 1000 |
| CHUNK_OVERLAP | No | 300 |
| RETRIEVAL_TOP_K | No | 20 |
| WORKER_POLL_INTERVAL_SECONDS | No | 60 |
| MAX_UPLOAD_SIZE_MB | No | 50 |
| ACCESS_TOKEN_EXPIRE_MINUTES | No | 30 |
| LLM_TEMPERATURE | No | 0.1 |
| SEED_ADMIN_USERNAME | No | admin |
| SEED_ADMIN_PASSWORD | No | Admin@1234 |
| TEMP_UPLOAD_DIR | No | tmp_uploads |
| CHAT_HISTORY_MAX_TURNS | No | 10 |

## Roles & Permissions

| Role | Permissions |
|------|-------------|
| `maintainer` | Upload PDFs, manage uploads, manage users, chat history |
| `user` | Ask questions, manage own chat history |

Authorization pattern: `Depends(require_maintainer)` on routes, or `Depends(get_current_user)` for any authenticated user.

## Qdrant Vector Details

- **Distance metric:** Cosine
- **Dimension:** 1024 (BGE-M3)
- **Payload per point:** text, page, chapter, section, subsection, job_id, filename
- **Point IDs:** Deterministic UUIDs from MD5 hash of `"{filename}_{chunk_index}"`
- **Upsert batch size:** 64

## Gemini System Prompt (Critical)

The LLM is instructed to:
- Use ONLY retrieved context (no prior knowledge)
- Return `"Information not found in the provided medical context."` for unknown info
- Cite chapters/pages only when metadata is present, never invent citations
- Never generate `(Document)`, `(Source)`, or `(Textbook)` as citations
- Never use page numbers from `"# CONTENTS"` breadcrumbs for medical facts
- Mention conflicts if context is contradictory
- Not provide diagnosis/treatment recommendations unless context explicitly states them

## Chunking Details

`chunk_markdown()`:
1. Normalizes headers: ensures `# TEXT` format (space after #), adds blank lines before headers
2. Splits by `MarkdownHeaderTextSplitter` on headers: `#` → chapter, `##` → section, `###` → subsection
3. Builds breadcrumb: `"# {chapter} > ## {section} > ### {subsection}"`
4. Resets section/subsection when encountering `# PART` or `# INTRODUCTION` headers
5. Runs `RecursiveCharacterTextSplitter` with separators: `\n\n`, `\n`, `|`, `. `, ` `, ``
6. Each chunk gets `CONTEXT_PREFIX` + breadcrumb prepended
7. Page numbers extracted via Docling item text matching (with word-overlap fallback) or regex

## Important Notes for Changes

- **Never** introduce multi-worker parallelism — the system is designed for sequential processing.
- **Never** remove context injection from chunks — it's critical for retrieval quality.
- **Never** touch the Gemini system prompt's citation rules without understanding the `"(Document)"` bug it fixes.
- **Never** instantiate services directly — always use the singleton accessors (`get_*()` functions).
- Qdrant collection name must match `QDRANT_COLLECTION_NAME` env var.
- Supabase Storage bucket is hardcoded as `"mediRag"` in `api/uploads.py` and `workers/ingestion_worker.py`.
- Admin seed in `seed.py` is called from the root module name `"seed"`, not `"seed.seed"`.
- The `app/state.py` module is a thin wrapper around `UploadRepository` — prefer using the repository/service layers directly for new code.
- CORS is configured for `localhost:3000` and `127.0.0.1:3000` only.
- **Chat history**: Conversations are scoped to users via `user_id` FK. All `/conversations` endpoints enforce ownership checks. Deleting a conversation cascades to all messages.
- **Multi-turn LLM**: Gemini API uses `"model"` role (not `"assistant"`) for conversation history. The `llm.py` service maps `"assistant"` → `"model"` when building multi-turn `contents`.
- **History limit**: `CHAT_HISTORY_MAX_TURNS` controls how many message pairs are sent to Gemini (default 10 turns = 20 messages). Adjust via env var without code changes.

## Docker

`Dockerfile` uses a multi-stage build to install PyTorch + BGE-M3 dependencies. `docker-compose.yml` exposes port 8000 for the API. The worker runs in-process (no separate container). Upload files use the `rag_uploads` volume.
