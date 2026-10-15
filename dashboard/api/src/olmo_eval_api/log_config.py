"""JSON logging to stdout with a per-request context.

Cloud Logging parses one JSON object per line and maps ``severity`` and ``message``.
"""

from __future__ import annotations

import contextvars
import logging
import sys
from datetime import UTC, datetime

import orjson

request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None
)
principal_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "principal", default=None
)

_RESERVED = set(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {
    "message",
    "color_message",  # uvicorn's ANSI-colored duplicate of the message
}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, object] = {
            "time": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "severity": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        request_id = request_id_var.get()
        if request_id:
            entry["request_id"] = request_id
        principal = principal_var.get()
        if principal:
            entry["principal"] = principal
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                entry[key] = value
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return orjson.dumps(entry, default=str).decode()


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    for name in ("uvicorn", "uvicorn.error"):
        logger = logging.getLogger(name)
        logger.handlers[:] = []
        logger.propagate = True
    # The request middleware writes one access line per request with the request ID.
    access = logging.getLogger("uvicorn.access")
    access.handlers[:] = []
    access.propagate = False
    # The connector and google-auth log at INFO on every refresh. httpx logs every request URL
    # at INFO, and those URLs can carry credentials.
    for name in ("google", "httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)
