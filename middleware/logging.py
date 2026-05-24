import logging
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


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
        elif status_code >= 400:
            logger.warning("%s %s → %d (%sms)", request.method, request.url.path, status_code, elapsed_ms)
        else:
            logger.info("%s %s → %d (%sms)", request.method, request.url.path, status_code, elapsed_ms)

        return response
