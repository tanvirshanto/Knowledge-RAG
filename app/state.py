import logging
from dataclasses import dataclass
from typing import Optional

from app.config import get_settings
from app.models import JobStatus

logger = logging.getLogger(__name__)


@dataclass
class JobRecord:
    status: str = "QUEUED"
    detail: Optional[str] = None
    chunks_indexed: Optional[int] = None


def _get_repo():
    from repositories.upload_repository import UploadRepository
    return UploadRepository()


def create_job(job_id: str) -> JobRecord:
    return JobRecord()


def get_job(job_id: str) -> Optional[JobRecord]:
    repo = _get_repo()
    record = repo.get_by_id(job_id)
    if record is None:
        return None
    return JobRecord(
        status=record.get("status", "QUEUED"),
        detail=record.get("error_message"),
        chunks_indexed=record.get("total_chunks"),
    )


def update_job(
    job_id: str,
    *,
    status: Optional[JobStatus] = None,
    detail: Optional[str] = None,
    chunks_indexed: Optional[int] = None,
) -> None:
    fields = {}
    if status is not None:
        fields["status"] = status.value if hasattr(status, "value") else str(status)
    if detail is not None:
        fields["error_message"] = detail
    if chunks_indexed is not None:
        fields["total_chunks"] = chunks_indexed
    if status == JobStatus.COMPLETED:
        from datetime import datetime, timezone
        fields["completed_at"] = datetime.now(timezone.utc).isoformat()
    if fields:
        try:
            _get_repo().update(job_id, **fields)
        except Exception as exc:
            logger.warning("Failed to update job %s in Supabase: %s", job_id, exc)
