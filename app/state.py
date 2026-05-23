import threading
from dataclasses import dataclass, field
from typing import Dict, Optional

from app.models import JobStatus


@dataclass
class JobRecord:
    status: JobStatus = JobStatus.QUEUED
    detail: Optional[str] = None
    chunks_indexed: Optional[int] = None


_lock = threading.Lock()
_jobs: Dict[str, JobRecord] = {}


def create_job(job_id: str) -> JobRecord:
    with _lock:
        record = JobRecord()
        _jobs[job_id] = record
        return record


def get_job(job_id: str) -> Optional[JobRecord]:
    with _lock:
        return _jobs.get(job_id)


def update_job(
    job_id: str,
    *,
    status: Optional[JobStatus] = None,
    detail: Optional[str] = None,
    chunks_indexed: Optional[int] = None,
) -> None:
    with _lock:
        record = _jobs.get(job_id)
        if record is None:
            return
        if status is not None:
            record.status = status
        if detail is not None:
            record.detail = detail
        if chunks_indexed is not None:
            record.chunks_indexed = chunks_indexed
