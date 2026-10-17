"""Shared retry logic for inference providers.

This module provides common retry utilities, status codes, and error handling
for API-based inference providers (LiteLLM, vLLM Server, etc.).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

logger = logging.getLogger(__name__)

# HTTP status codes that should trigger a retry
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

# HTTP status codes that should never be retried. OpenRouter returns 402 when a
# key is out of credits.
NON_RETRYABLE_STATUS_CODES = frozenset({400, 401, 402, 403, 404, 422})

# Exception type names that should never be retried (request itself is wrong)
NEVER_RETRY_EXCEPTION_TYPES = (
    "AuthenticationError",  # 401 - bad API key
    "BadRequestError",  # 400 - invalid params, content policy, etc.
    "NotFoundError",  # 404 - wrong model / endpoint
    "UnprocessableEntityError",  # 422 - semantic validation failure
)

# Exception type names that are always transient and should be retried
ALWAYS_RETRY_EXCEPTION_TYPES = (
    "RateLimitError",  # 429
    "APITimeoutError",  # request timed out (OpenAI SDK)
    "Timeout",  # request timed out (litellm)
    "APIConnectionError",  # connection-level failure
    "ServiceUnavailableError",  # 503
    "InternalServerError",  # 500
)


def _format_cause(cause: BaseException) -> str:
    """Format the cause of an exception with fallback for empty strings."""
    cause_type = type(cause).__qualname__
    cause_str = str(cause)

    # If str(cause) is non-empty, use it
    if cause_str:
        return f"{cause_type}: {cause_str}"

    # Try args (often contains the real message)
    if cause.args:
        args_str = ", ".join(str(a) for a in cause.args if a)
        if args_str:
            return f"{cause_type}: {args_str}"

    # Just return the type name
    return cause_type


def format_error(exc: Exception) -> str:
    """Build a detailed, single-log-entry description of an exception.

    Args:
        exc: The exception to format.

    Returns:
        Multi-line string with error details.
    """
    parts: list[str] = [f"  type: {type(exc).__qualname__}"]

    # HTTP status code
    status = getattr(exc, "status_code", None)
    if status is not None:
        parts.append(f"  status_code: {status}")

    # Provider and model info (litellm-specific)
    for attr in ("llm_provider", "model"):
        val = getattr(exc, attr, None)
        if val is not None:
            parts.append(f"  {attr}: {val}")

    # Request URL from response
    response = getattr(exc, "response", None)
    if response is not None:
        url = getattr(response, "url", None)
        if url is not None:
            parts.append(f"  url: {url}")

    # Error message
    message = getattr(exc, "message", None) or str(exc)
    # Truncate very long messages (e.g., full HTML error pages)
    if len(message) > 500:
        message = message[:500] + "…"
    parts.append(f"  message: {message}")

    # The wrapped cause often has the real reason (e.g., httpx.ReadTimeout)
    cause = exc.__cause__
    if cause is not None:
        parts.append(f"  cause: {_format_cause(cause)}")

    return "\n".join(parts)


def is_retryable_error(
    exc: Exception,
    sdk_module: Any | None = None,
) -> bool:
    """Determine whether an exception should be retried.

    Uses typed exception hierarchy (if sdk_module provided) to classify errors:
    - Never retry: AuthenticationError, BadRequestError, NotFoundError,
      UnprocessableEntityError
    - Always retry: RateLimitError, APITimeoutError, Timeout, APIConnectionError,
      ServiceUnavailableError, InternalServerError
    - Falls back to HTTP status code for unknown subtypes.

    Args:
        exc: The exception to check.
        sdk_module: Optional SDK module (e.g., openai, litellm) for type checking.

    Returns:
        True if the error should be retried, False otherwise.
    """
    # Check against SDK exception types if module provided
    if sdk_module is not None:
        # Never retry these - the request itself is wrong
        for attr in NEVER_RETRY_EXCEPTION_TYPES:
            cls = getattr(sdk_module, attr, None)
            if cls is not None and isinstance(exc, cls):
                return False

        # Always retry these - transient server/network issues
        for attr in ALWAYS_RETRY_EXCEPTION_TYPES:
            cls = getattr(sdk_module, attr, None)
            if cls is not None and isinstance(exc, cls):
                return True

    # Check exception type name as fallback
    error_type = type(exc).__name__
    if error_type in NEVER_RETRY_EXCEPTION_TYPES:
        return False
    if error_type in ALWAYS_RETRY_EXCEPTION_TYPES:
        return True

    # Fall back to HTTP status code
    status = getattr(exc, "status_code", None)
    if status is not None:
        status = int(status)
        if status in NON_RETRYABLE_STATUS_CODES:
            return False
        if status in RETRYABLE_STATUS_CODES:
            return True

    # String matching fallback for errors without status codes
    error_str = str(exc).lower()

    # Timeout errors are retryable
    if "timeout" in error_str or "timed out" in error_str:
        return True

    # Connection errors are retryable
    if "connection" in error_str:
        return True

    # Rate limit errors are retryable
    return "rate" in error_str and "limit" in error_str


async def retry_with_backoff[T](
    func: Callable[[], Awaitable[T]],
    *,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    context: str = "",
    sdk_module: Any | None = None,
) -> T:
    """Execute an async function with exponential backoff for retryable errors.

    Args:
        func: Async callable to execute.
        max_retries: Maximum number of retries (default 3).
        retry_delay: Base delay in seconds between retries (exponential backoff).
        context: Optional human-readable label (e.g., "generate model=llama3")
                 included in log messages.
        sdk_module: Optional SDK module for exception type checking.

    Returns:
        Result of the function call.

    Raises:
        Exception: If all retries are exhausted or a non-retryable error occurs.
    """
    last_exception: Exception | None = None
    ctx = f" [{context}]" if context else ""

    for attempt in range(max_retries + 1):
        try:
            return await func()
        except Exception as e:
            last_exception = e
            detail = format_error(e)

            # Authentication errors: fail immediately with actionable guidance
            if sdk_module is not None:
                auth_cls = getattr(sdk_module, "AuthenticationError", None)
                if auth_cls is not None and isinstance(e, auth_cls):
                    logger.error(
                        f"Authentication failed{ctx}:\n{detail}\n"
                        f"  Verify the API key environment variable is set correctly."
                    )
                    raise

                # Not-found errors: fail immediately with actionable guidance
                not_found_cls = getattr(sdk_module, "NotFoundError", None)
                if not_found_cls is not None and isinstance(e, not_found_cls):
                    logger.error(
                        f"Resource not found{ctx}:\n{detail}\n"
                        f"  Verify the model name and API endpoint are correct."
                    )
                    raise

            retryable = is_retryable_error(e, sdk_module)

            if not retryable or attempt >= max_retries:
                if retryable:
                    logger.error(f"Retries exhausted{ctx} after {attempt + 1} attempts:\n{detail}")
                else:
                    logger.error(f"Non-retryable error{ctx}:\n{detail}")
                raise

            delay = retry_delay * (2**attempt)
            logger.warning(
                f"Retryable error{ctx} "
                f"(attempt {attempt + 1}/{max_retries + 1}):\n{detail}\n"
                f"  retrying in {delay:.1f}s …"
            )
            await asyncio.sleep(delay)

    # Should not reach here, but raise last exception if we do
    if last_exception:
        raise last_exception
    raise RuntimeError("Unexpected retry loop exit")


__all__ = [
    "RETRYABLE_STATUS_CODES",
    "NON_RETRYABLE_STATUS_CODES",
    "NEVER_RETRY_EXCEPTION_TYPES",
    "ALWAYS_RETRY_EXCEPTION_TYPES",
    "format_error",
    "is_retryable_error",
    "retry_with_backoff",
]
