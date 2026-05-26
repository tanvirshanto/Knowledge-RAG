import logging
import time
from typing import Callable, TypeVar

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)

T = TypeVar("T")

_retryable_list: list[type[BaseException]] = [
    httpx.RequestError,
    ConnectionError,
    TimeoutError,
    OSError,
]

# Dynamically add Qdrant exceptions if available
try:
    from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
    _retryable_list.extend([ResponseHandlingException, UnexpectedResponse])
except ImportError:
    pass

# Dynamically add Google GenAI exceptions if available
try:
    from google.genai.errors import ServerError
    _retryable_list.append(ServerError)
except ImportError:
    pass

DEFAULT_RETRYABLE = tuple(_retryable_list)


def _is_non_retryable_unexpected_response(exc: BaseException) -> bool:
    try:
        from qdrant_client.http.exceptions import UnexpectedResponse
        if isinstance(exc, UnexpectedResponse):
            # Only retry on transient/timeout statuses: 408 (Timeout), 500, 502, 503, 504.
            # Do NOT retry client/validation errors like 400, 401, 403, 404, 409 etc.
            if exc.status_code not in {408, 500, 502, 503, 504}:
                return True
    except ImportError:
        pass
    return False


def retry_sync(
    operation: Callable[[], T],
    operation_name: str,
    retryable_exceptions: tuple[type[BaseException], ...] | None = None,
) -> T:
    settings = get_settings()
    max_retries = settings.max_retries
    base_delay = settings.retry_base_delay_seconds
    multiplier = settings.retry_backoff_multiplier
    max_delay = settings.retry_max_delay_seconds

    if retryable_exceptions is None:
        retryable_exceptions = DEFAULT_RETRYABLE

    last_exception: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            return operation()
        except retryable_exceptions as exc:
            # Prevent retrying on non-retryable Qdrant HTTP status codes
            if _is_non_retryable_unexpected_response(exc):
                logger.error(
                    "Non-retryable UnexpectedResponse for '%s' — %s: %s",
                    operation_name,
                    type(exc).__name__,
                    exc,
                )
                raise exc

            last_exception = exc
            if attempt < max_retries:
                delay = min(base_delay * (multiplier ** attempt), max_delay)
                logger.warning(
                    "Retry %d/%d for '%s' after %.1fs — %s: %s",
                    attempt + 1,
                    max_retries,
                    operation_name,
                    delay,
                    type(exc).__name__,
                    exc,
                )
                time.sleep(delay)
            else:
                logger.error(
                    "All %d retries exhausted for '%s' — %s: %s",
                    max_retries,
                    operation_name,
                    type(exc).__name__,
                    exc,
                )

    raise RuntimeError(
        f"Operation '{operation_name}' failed after {max_retries} retries. "
        f"Last error: {type(last_exception).__name__}: {last_exception}"
    )
