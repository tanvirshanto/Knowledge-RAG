import logging
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from services.system_log_service import SystemLogService


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path == "/health":
            return await call_next(request)

        logger = logging.getLogger("api.access")
        start = time.monotonic()
        response = await call_next(request)
        elapsed_ms = round((time.monotonic() - start) * 1000, 2)

        status_code = response.status_code
        if status_code >= 500:
            logger.error("%s %s → %d (%sms)", request.method, request.url.path, status_code, elapsed_ms)
            _log_to_db(
                level="ERROR",
                message=f"{request.method} {request.url.path} → {status_code} ({elapsed_ms}ms)",
            )
        elif status_code >= 400:
            logger.warning("%s %s → %d (%sms)", request.method, request.url.path, status_code, elapsed_ms)
        else:
            logger.info("%s %s → %d (%sms)", request.method, request.url.path, status_code, elapsed_ms)

        return response


def _log_to_db(*, level: str, message: str) -> None:
    try:
        log_service = SystemLogService()
        log_service.log_error(level=level, message=message)
    except Exception:
        logger = logging.getLogger("api.access")
        logger.warning("Failed to persist system log to Supabase", exc_info=True)
