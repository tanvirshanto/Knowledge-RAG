import logging
from datetime import datetime, timezone
from typing import Optional

from repositories.base import BaseRepository

logger = logging.getLogger(__name__)


class UploadRepository(BaseRepository):
    def __init__(self):
        super().__init__("upload_jobs")

    def create(
        self,
        job_id: str,
        filename: str,
        original_filename: str,
        uploaded_by: str,
        status: str = "QUEUED",
        storage_path: Optional[str] = None,
    ) -> dict:
        now = datetime.now(timezone.utc).isoformat()
        result = self.table.insert({
            "id": job_id,
            "filename": filename,
            "original_filename": original_filename,
            "uploaded_by": uploaded_by,
            "status": status,
            "created_at": now,
            "storage_path": storage_path,
        }).execute()
        return self._execute_single(result)

    def get_by_id(self, job_id: str) -> Optional[dict]:
        result = self.table.select("*").eq("id", job_id).execute()
        return self._execute_single(result)

    def list_all(
        self,
        status_filter: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict], int]:
        query = self.table.select("*", count="exact")
        if status_filter:
            query = query.eq("status", status_filter)
        query = query.order("created_at", desc=True).range(offset, offset + limit - 1)
        result = query.execute()
        rows = self._execute(result)
        total = result.count if hasattr(result, "count") and result.count is not None else len(rows)
        return rows, total

    def get_next_queued(self) -> Optional[dict]:
        result = (
            self.table.select("*")
            .eq("status", "QUEUED")
            .order("created_at", desc=False)
            .limit(1)
            .execute()
        )
        job = self._execute_single(result)
        if job:
            now = datetime.now(timezone.utc).isoformat()
            self.update_status(
                job["id"],
                "RUNNING",
                started_at=now,
            )
            job["status"] = "RUNNING"
            job["started_at"] = now
        return job

    def get_running(self) -> Optional[dict]:
        result = (
            self.table.select("*")
            .eq("status", "RUNNING")
            .limit(1)
            .execute()
        )
        return self._execute_single(result)

    def update(self, job_id: str, **fields) -> Optional[dict]:
        result = self.table.update(fields).eq("id", job_id).execute()
        return self._execute_single(result)

    def update_status(self, job_id: str, status: str, **extra_fields) -> Optional[dict]:
        fields = {"status": status, **extra_fields}
        return self.update(job_id, **fields)

    def table_exists(self) -> bool:
        try:
            self.table.select("id").limit(1).execute()
            return True
        except Exception:
            return False
