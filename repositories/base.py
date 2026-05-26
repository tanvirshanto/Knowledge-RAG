import logging
from typing import Callable, Optional, TypeVar

from supabase import Client, create_client

from app.config import get_settings
from utils.retry import retry_sync

logger = logging.getLogger(__name__)

_supabase_client: Optional[Client] = None

T = TypeVar("T")


def get_supabase() -> Client:
    global _supabase_client
    if _supabase_client is None:
        settings = get_settings()
        _supabase_client = create_client(settings.supabase_url, settings.supabase_service_key)
        logger.info("Supabase client initialized")
    return _supabase_client


class BaseRepository:
    def __init__(self, table_name: str):
        self.table_name = table_name

    @property
    def supabase(self) -> Client:
        return get_supabase()

    @property
    def table(self):
        return self.supabase.table(self.table_name)

    def _execute(self, result):
        if hasattr(result, "error") and result.error is not None:
            logger.error("Supabase error on %s: %s", self.table_name, result.error)
            raise RuntimeError(f"Database error: {result.error}")
        return result.data or []

    def _execute_single(self, result) -> Optional[dict]:
        data = self._execute(result)
        return data[0] if data else None

    def _execute_with_retry(
        self,
        query_callable: Callable[[], T],
        *,
        single: bool = False,
    ):
        result = retry_sync(query_callable, f"supabase.{self.table_name}")
        return self._execute_single(result) if single else self._execute(result)
