"""Error envelope and exception handlers.

Every non-2xx response has the body ``{"error": {"code", "message", "request_id", "details"?}}``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import DBAPIError
from starlette.exceptions import HTTPException as StarletteHTTPException

from olmo_eval_api.log_config import request_id_var

logger = logging.getLogger(__name__)

_STATUS_CODES = {
    400: "bad_request",
    401: "unauthenticated",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    413: "payload_too_large",
    422: "validation_error",
    429: "rate_limited",
    503: "unavailable",
}


class ApiError(Exception):
    def __init__(
        self,
        status_code: int,
        message: str,
        *,
        code: str | None = None,
        details: Any = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code or _STATUS_CODES.get(status_code, "error")
        self.message = message
        self.details = details
        self.headers = headers


def bad_request(message: str, details: Any = None) -> ApiError:
    return ApiError(400, message, details=details)


def not_found(message: str) -> ApiError:
    return ApiError(404, message)


def forbidden(message: str) -> ApiError:
    return ApiError(403, message)


def conflict(message: str) -> ApiError:
    return ApiError(409, message)


def error_body(code: str, message: str, details: Any = None) -> dict[str, Any]:
    body: dict[str, Any] = {
        "code": code,
        "message": message,
        "request_id": request_id_var.get() or "",
    }
    if details is not None:
        body["details"] = details
    return {"error": body}


def error_response(
    status_code: int,
    code: str,
    message: str,
    details: Any = None,
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    return JSONResponse(
        error_body(code, message, details),
        status_code=status_code,
        headers=dict(headers) if headers else None,
    )


def _jsonable_errors(errors: Sequence[Any]) -> list[dict[str, Any]]:
    out = []
    for err in errors:
        item = {k: err[k] for k in ("type", "loc", "msg") if k in err}
        if "input" in err and isinstance(err["input"], str | int | float | bool | None):
            item["input"] = err["input"]
        out.append(item)
    return out


def is_parameter_error(exc: DBAPIError) -> bool:
    """True when asyncpg refused to encode a query parameter (wrong type or out of range).

    asyncpg raises the base ``DataError`` class itself for these, before the query reaches
    Postgres. Server-side data errors (division by zero and the like) are subclasses and stay
    500s because they point at a server bug.
    """
    import asyncpg

    cause = exc.orig.__cause__ if exc.orig is not None else None
    return type(cause) is asyncpg.exceptions.DataError


def install_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError) -> JSONResponse:
        return error_response(exc.status_code, exc.code, exc.message, exc.details, exc.headers)

    from olmo_eval_api.stats.paired import WorkBudgetExceeded

    @app.exception_handler(WorkBudgetExceeded)
    async def _too_much_work(_: Request, exc: WorkBudgetExceeded) -> JSONResponse:
        return error_response(400, "bad_request", str(exc))

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        return error_response(
            422, "validation_error", "request validation failed", _jsonable_errors(exc.errors())
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = _STATUS_CODES.get(exc.status_code, "error")
        message = exc.detail if isinstance(exc.detail, str) else code
        return error_response(exc.status_code, code, message, headers=exc.headers)

    @app.exception_handler(DBAPIError)
    async def _db_error(request: Request, exc: DBAPIError) -> JSONResponse:
        if is_parameter_error(exc):
            # A cursor or numeric parameter that does not fit its column; the client sent it.
            logger.info("rejected parameter on %s %s: %s", request.method, request.url.path, exc)
            return error_response(
                400, "bad_request", "a parameter value is invalid or out of range"
            )
        logger.exception("unhandled error on %s %s", request.method, request.url.path)
        return error_response(500, "internal", "internal server error")

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error on %s %s", request.method, request.url.path)
        return error_response(500, "internal", "internal server error")
