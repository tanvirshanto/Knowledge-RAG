import logging
import uuid
from typing import Optional

from repositories.upload_repository import UploadRepository
from schemas.upload import UploadJobResponse, UploadListResponse

logger = logging.getLogger(__name__)


class UploadService:
    def __init__(self, upload_repo: UploadRepository | None = None):
        self.upload_repo = upload_repo or UploadRepository()

    def create_upload_job(
        self,
        original_filename: str,
        uploaded_by: str,
        job_id: Optional[str] = None,
        storage_path: Optional[str] = None,
    ) -> UploadJobResponse:
        if not job_id:
            job_id = uuid.uuid4().hex
        filename = f"{job_id}.pdf"
        if not storage_path:
            storage_path = f"uploads/{job_id}.pdf"
        record = self.upload_repo.create(
            job_id=job_id,
            filename=filename,
            original_filename=original_filename,
            uploaded_by=uploaded_by,
            status="QUEUED",
            storage_path=storage_path,
        )
        return self._to_response(record)

    def get_job(self, job_id: str) -> Optional[UploadJobResponse]:
        record = self.upload_repo.get_by_id(job_id)
        if record is None:
            return None
        return self._to_response(record)

    def list_jobs(
        self,
        status_filter: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> UploadListResponse:
        rows, total = self.upload_repo.list_all(status_filter, limit, offset)
        return UploadListResponse(
            jobs=[self._to_response(r) for r in rows],
            total=total,
        )

    def get_running_job(self) -> Optional[UploadJobResponse]:
        record = self.upload_repo.get_running()
        if record is None:
            return None
        return self._to_response(record)

    def get_next_queued(self) -> Optional[dict]:
        return self.upload_repo.get_next_queued()

    def update_job_status(self, job_id: str, status: str, **fields) -> None:
        self.upload_repo.update_status(job_id, status, **fields)

    def retry_job(self, job_id: str) -> Optional[UploadJobResponse]:
        record = self.upload_repo.get_by_id(job_id)
        if record is None:
            return None
        self.upload_repo.update_status(
            job_id,
            "QUEUED",
            error_message=None,
            started_at=None,
            completed_at=None,
            total_pages=None,
            total_chunks=None,
        )
        updated_record = self.upload_repo.get_by_id(job_id)
        return self._to_response(updated_record) if updated_record else None

    def _to_response(self, record: dict) -> UploadJobResponse:
        return UploadJobResponse(
            id=record.get("id", ""),
            filename=record.get("filename", ""),
            original_filename=record.get("original_filename", ""),
            uploaded_by=record.get("uploaded_by"),
            status=record.get("status", ""),
            error_message=record.get("error_message"),
            started_at=record.get("started_at"),
            completed_at=record.get("completed_at"),
            created_at=record.get("created_at", ""),
            total_pages=record.get("total_pages"),
            total_chunks=record.get("total_chunks"),
            storage_path=record.get("storage_path"),
        )
