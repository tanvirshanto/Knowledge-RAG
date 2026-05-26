import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from services.system_log_service import SystemLogService

logger = logging.getLogger(__name__)


def register_exception_handlers(app: FastAPI) -> None:
    log_service = SystemLogService()

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        if exc.status_code >= 500:
            _log_to_db(
                log_service,
                level="ERROR",
                message=f"HTTP {exc.status_code} on {request.method} {request.url.path}: {exc.detail}",
                exception=exc,
            )
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        logger.warning("Validation error on %s %s: %s", request.method, request.url.path, exc.errors())
        _log_to_db(
            log_service,
            level="WARNING",
            message=f"Validation error on {request.method} {request.url.path}: {exc.errors()}",
        )
        return JSONResponse(
            status_code=422,
            content={
                "detail": "Validation error",
                "errors": exc.errors(),
            },
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
        _log_to_db(
            log_service,
            level="ERROR",
            message=f"Unhandled exception on {request.method} {request.url.path}: {exc}",
            exception=exc,
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )


def _log_to_db(
    log_service: SystemLogService,
    *,
    level: str,
    message: str,
    exception: Exception | None = None,
) -> None:
    try:
        log_service.log_error(level=level, message=message, exception=exception)
    except Exception:
        logger.warning("Failed to persist system log to Supabase", exc_info=True)
