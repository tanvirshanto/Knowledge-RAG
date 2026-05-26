import logging
from datetime import datetime, timezone
from typing import Optional

from repositories.base import BaseRepository
from utils.retry import retry_sync

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
        return self._execute_with_retry(
            lambda: self.table.insert({
                "id": job_id,
                "filename": filename,
                "original_filename": original_filename,
                "uploaded_by": uploaded_by,
                "status": status,
                "created_at": now,
                "storage_path": storage_path,
            }).execute(),
            single=True,
        )

    def get_by_id(self, job_id: str) -> Optional[dict]:
        return self._execute_with_retry(
            lambda: self.table.select("*").eq("id", job_id).execute(),
            single=True,
        )

    def list_all(
        self,
        status_filter: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict], int]:
        def _query():
            query = self.table.select("*", count="exact")
            if status_filter:
                query = query.eq("status", status_filter)
            query = query.order("created_at", desc=True).range(offset, offset + limit - 1)
            return query.execute()

        result = retry_sync(_query, f"supabase.{self.table_name}.list_all")
        rows = self._execute(result)
        total = result.count if hasattr(result, "count") and result.count is not None else len(rows)
        return rows, total

    def get_next_queued(self) -> Optional[dict]:
        job = self._execute_with_retry(
            lambda: (
                self.table.select("*")
                .eq("status", "QUEUED")
                .order("created_at", desc=False)
                .limit(1)
                .execute()
            ),
            single=True,
        )
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
        return self._execute_with_retry(
            lambda: (
                self.table.select("*")
                .eq("status", "RUNNING")
                .limit(1)
                .execute()
            ),
            single=True,
        )

    def update(self, job_id: str, **fields) -> Optional[dict]:
        return self._execute_with_retry(
            lambda: self.table.update(fields).eq("id", job_id).execute(),
            single=True,
        )

    def update_status(self, job_id: str, status: str, **extra_fields) -> Optional[dict]:
        fields = {"status": status, **extra_fields}
        return self.update(job_id, **fields)

    def table_exists(self) -> bool:
        try:
            self._execute_with_retry(
                lambda: self.table.select("id").limit(1).execute(),
            )
            return True
        except Exception:
            return False
