import logging
from typing import Optional

from repositories.base import BaseRepository

logger = logging.getLogger(__name__)


class SystemLogRepository(BaseRepository):
    def __init__(self):
        super().__init__("system_logs")

    def create(self, level: str, message: str, traceback: Optional[str] = None) -> dict:
        return self._execute_with_retry(
            lambda: self.table.insert({
                "level": level,
                "message": message,
                "traceback": traceback,
            }).execute(),
            single=True,
        )

    def list_all(self, limit: int = 100, offset: int = 0) -> list[dict]:
        return self._execute_with_retry(
            lambda: self.table
            .select("*")
            .order("created_at", desc=True)
            .range(offset, offset + limit - 1)
            .execute(),
        )

    def get_by_id(self, log_id: str) -> Optional[dict]:
        return self._execute_with_retry(
            lambda: self.table.select("*").eq("id", log_id).execute(),
            single=True,
        )

    def table_exists(self) -> bool:
        try:
            self._execute_with_retry(
                lambda: self.table.select("id").limit(1).execute(),
            )
            return True
        except Exception:
            return False
