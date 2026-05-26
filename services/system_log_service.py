import logging
import traceback
from typing import Optional

from repositories.system_log_repository import SystemLogRepository
from schemas.system_log import SystemLogResponse

logger = logging.getLogger(__name__)


class SystemLogService:
    def __init__(self, log_repo: SystemLogRepository | None = None):
        self.log_repo = log_repo or SystemLogRepository()

    def log_error(
        self,
        level: str,
        message: str,
        exception: Optional[Exception] = None,
    ) -> SystemLogResponse:
        tb = None
        if exception:
            tb = "".join(traceback.format_exception(
                type(exception), exception, exception.__traceback__
            ))
        record = self.log_repo.create(
            level=level,
            message=message,
            traceback=tb,
        )
        return self._to_response(record)

    def list_logs(self, limit: int = 100, offset: int = 0) -> list[SystemLogResponse]:
        records = self.log_repo.list_all(limit=limit, offset=offset)
        return [self._to_response(record) for record in records]

    def _to_response(self, record: dict) -> SystemLogResponse:
        return SystemLogResponse(
            id=str(record.get("id", "")),
            level=record.get("level", ""),
            message=record.get("message", ""),
            traceback=record.get("traceback"),
            created_at=record.get("created_at", ""),
        )

