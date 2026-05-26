import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.router import api_router
from app.config import get_settings
from middleware.exception import register_exception_handlers
from middleware.logging import LoggingMiddleware
from seed import seed_admin_user
from utils.logging_config import setup_logging
from utils.file_storage import ensure_upload_dir
from workers.ingestion_worker import IngestionWorker

worker: IngestionWorker | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global worker

    setup_logging()
    logger = logging.getLogger(__name__)
    settings = get_settings()

    logger.info("Starting Medical RAG Service...")

    ensure_upload_dir(settings.temp_upload_dir)

    from app.services.embeddings import get_embedding_service
    logger.info("Pre-warming LocalEmbeddingService (loading %s)...", settings.embedding_model)
    get_embedding_service(settings)
    logger.info("LocalEmbeddingService pre-warmed successfully.")

    from app.services.parsing import get_converter
    logger.info("Pre-warming DocumentConverter...")
    get_converter()
    logger.info("DocumentConverter pre-warmed successfully.")

    from app.services.vector_store import get_vector_store
    logger.info("Ensuring Qdrant collection '%s' exists...", settings.qdrant_collection_name)
    get_vector_store(settings).ensure_collection()
    logger.info("Qdrant collection check complete.")

    # await seed_admin_user()

    worker = IngestionWorker(poll_interval=settings.worker_poll_interval_seconds)
    await worker.start()

    logger.info("Medical RAG Service started successfully.")
    yield

    if worker:
        await worker.stop()
    logger.info("Medical RAG Service shut down.")


app = FastAPI(
    title="Medical RAG Service",
    description="Production RAG backend for medical textbooks (Docling + BGE-M3 + Qdrant + Gemini)",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(LoggingMiddleware)

register_exception_handlers(app)

app.include_router(api_router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
