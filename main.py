import json
import logging
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.config import get_settings
from app.models import AskRequest, AskResponse, JobStatus, StatusResponse, UploadResponse
from app.pipelines.ingestion import run_ingestion_pipeline
from app.pipelines.retrieval import answer_question, stream_answer_question
from app.state import create_job, get_job

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Medical RAG Service",
    description="Production RAG backend for medical textbooks (Docling + BGE-M3 + Qdrant + Command Code)",
    version="1.0.0",
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


@app.on_event("startup")
def startup_event() -> None:
    settings = get_settings()
    Path(settings.temp_upload_dir).mkdir(parents=True, exist_ok=True)
    
    # Pre-warm LocalEmbeddingService on startup
    logger.info("Pre-warming LocalEmbeddingService (loading BAAI/bge-m3)...")
    from app.services.embeddings import get_embedding_service
    get_embedding_service(settings)
    logger.info("LocalEmbeddingService pre-warmed successfully.")

    # Ensure Qdrant collection exists on startup
    logger.info("Ensuring Qdrant collection '%s' exists...", settings.qdrant_collection_name)
    from app.services.vector_store import get_vector_store
    get_vector_store(settings).ensure_collection()
    logger.info("Qdrant collection check complete.")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/upload-pdf", response_model=UploadResponse)
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> UploadResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    settings = get_settings()
    job_id = uuid.uuid4().hex
    create_job(job_id)

    temp_dir = Path(settings.temp_upload_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    dest = temp_dir / f"{job_id}.pdf"

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    dest.write_bytes(content)

    background_tasks.add_task(run_ingestion_pipeline, job_id, dest, file.filename)
    logger.info("Queued ingestion job %s for %s", job_id, file.filename)

    return UploadResponse(job_id=job_id, status="Queued")


@app.get("/status/{job_id}", response_model=StatusResponse)
def get_status(job_id: str) -> StatusResponse:
    record = get_job(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return StatusResponse(
        job_id=job_id,
        status=record.status,
        detail=record.detail,
        chunks_indexed=record.chunks_indexed,
    )


def _sse_event(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.post("/ask")
def ask(
    body: AskRequest,
    stream: bool = Query(default=True, description="Stream tokens via SSE"),
):
    if stream:
        def event_generator():
            try:
                yield _sse_event({"type": "start", "question": body.question})
                for token in stream_answer_question(body.question):
                    yield _sse_event({"type": "token", "content": token})
                yield _sse_event({"type": "done", "question": body.question})
            except Exception as exc:
                logger.exception("Ask stream failed")
                yield _sse_event({"type": "error", "detail": str(exc)})

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    try:
        answer = answer_question(body.question)
    except Exception as exc:
        logger.exception("Ask pipeline failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return AskResponse(question=body.question, answer=answer)
