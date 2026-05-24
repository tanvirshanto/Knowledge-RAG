import logging
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status

from app.config import get_settings
from auth.dependencies import get_current_user, require_maintainer
from schemas.upload import UploadBulkResponse, UploadJobResponse, UploadListResponse
from services.upload_service import UploadService
from utils.file_storage import ensure_upload_dir, save_upload

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(get_current_user)])


@router.post("/upload-pdf", response_model=UploadBulkResponse, status_code=status.HTTP_201_CREATED)
async def upload_pdf(
    files: List[UploadFile] = File(..., description="PDF files to upload"),
    current_user: dict = Depends(require_maintainer),
):
    settings = get_settings()
    service = UploadService()
    ensure_upload_dir(settings.temp_upload_dir)

    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    created_jobs = []
    for file in files:
        if not file.filename:
            raise HTTPException(status_code=400, detail="File without a filename received")

        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail=f"Only PDF files are supported. '{file.filename}' is not a PDF.",
            )

        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail=f"File '{file.filename}' is empty")

        if len(content) > settings.max_upload_size_mb * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail=f"File '{file.filename}' exceeds maximum size of {settings.max_upload_size_mb}MB",
            )

        job = service.create_upload_job(
            original_filename=file.filename,
            uploaded_by=current_user["sub"],
        )

        save_upload(content, job.id, settings.temp_upload_dir)
        created_jobs.append(job)
        logger.info(
            "Created upload job %s for '%s' by %s",
            job.id, file.filename, current_user["sub"],
        )

    return UploadBulkResponse(jobs=created_jobs)


@router.get("", response_model=UploadListResponse)
def list_uploads(
    status_filter: str | None = Query(default=None, alias="status"),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    service = UploadService()
    return service.list_jobs(status_filter=status_filter, limit=limit, offset=offset)


@router.get("/running", response_model=UploadJobResponse | None)
def get_running():
    service = UploadService()
    job = service.get_running_job()
    return job


@router.get("/{job_id}", response_model=UploadJobResponse)
def get_upload(job_id: str):
    service = UploadService()
    job = service.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Upload job not found")
    return job
