from datetime import datetime
import logging
from pathlib import Path

from app.config import Settings, get_settings
from app.models import JobStatus
from app.services.chunking import chunk_markdown
from app.services.embeddings import get_embedding_service
from app.services.parsing import parse_pdf_to_markdown
from app.services.vector_store import get_vector_store
from app.state import update_job

logger = logging.getLogger(__name__)

def run_ingestion_pipeline(job_id: str, pdf_path: Path, filename: str = "document.pdf") -> None:
    settings = get_settings()
    try:
        start_time = datetime.now()
        logger.info("Starting ingestion pipeline for job %s", job_id)
        update_job(job_id, status=JobStatus.PARSING)
        markdown = parse_pdf_to_markdown(pdf_path, settings)

        update_job(job_id, status=JobStatus.CHUNKING)
        logger.info("Chunking markdown...")
        chunks = chunk_markdown(markdown, settings)
        logger.info("Markdown chunked successfully.")
        if not chunks:
            raise ValueError("No chunks produced from document.")

        embedder = get_embedding_service(settings)
        logger.info("Embedder initialized successfully.")
        store = get_vector_store(settings)
        logger.info("Vector store initialized successfully.")

        update_job(job_id, status=JobStatus.EMBEDDING)
        logger.info("Embedding chunks...")
        texts = [c.text for c in chunks]
        all_vectors = embedder.embed_documents(texts)
        logger.info("Chunks embedded successfully.")

        update_job(job_id, status=JobStatus.INDEXING)
        logger.info("Indexing chunks...")
        indexed = store.upsert_chunks(chunks, all_vectors, job_id=job_id, filename=filename)
        logger.info("Successfully indexed %d chunks into '%s' for job %s", indexed, settings.qdrant_collection_name, job_id)
        logger.info("Total Time Taken: %s", datetime.now() - start_time)

        update_job(
            job_id,
            status=JobStatus.COMPLETED,
            chunks_indexed=indexed,
            detail=f"Indexed {indexed} chunks into '{settings.qdrant_collection_name}'.",
        )
    except Exception as exc:
        logger.exception("Ingestion failed for job %s", job_id)
        update_job(job_id, status=JobStatus.FAILED, detail=str(exc))
        raise
    finally:
        _safe_delete(pdf_path)


def _safe_delete(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except OSError as exc:
        logger.warning("Could not delete temp file %s: %s", path, exc)
