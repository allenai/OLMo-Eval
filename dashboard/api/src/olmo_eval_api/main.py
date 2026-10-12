"""FastAPI app factory.

One image serves two Cloud Run services. API_MODE picks the routes:

- ``dashboard``: /health and /api/** (behind IAP, read by the UI)
- ``ingest``: /health and /v1/** (Google access-token auth, written by olmo-eval)
- ``all``: everything (SKIFF_ENV=local only)

With local file storage (SKIFF_ENV=local), /_local/objects/** stands in for GCS signed URLs.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from olmo_eval_api import __version__
from olmo_eval_api.auth.google_token import TokenVerifier
from olmo_eval_api.auth.iap import IapVerifier
from olmo_eval_api.db.engine import make_database
from olmo_eval_api.db.grants import grant_dashboard_access
from olmo_eval_api.db.migrate import prune_stats_cache, run_migrations
from olmo_eval_api.errors import error_body, install_exception_handlers
from olmo_eval_api.log_config import configure_logging, principal_var, request_id_var
from olmo_eval_api.settings import Settings, get_settings
from olmo_eval_api.storage.base import Storage

logger = logging.getLogger(__name__)

MAX_BODY_BYTES = 16 * 1024 * 1024
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9-]{8,64}$")
_UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
CSRF_HEADER = b"x-requested-with"
# IAP answers XMLHttpRequest-marked requests with 401 instead of a sign-in redirect, so the
# UI can show "session expired". Any custom header also forces a CORS preflight.
CSRF_VALUE = b"XMLHttpRequest"


def make_storage(settings: Settings) -> Storage:
    if settings.storage_backend == "local":
        from olmo_eval_api.storage.local import LocalStorage

        return LocalStorage(
            settings.local_storage_dir,
            settings.public_base_url or "",
            bucket=settings.results_bucket,
        )
    from olmo_eval_api.storage.gcs import GcsStorage

    return GcsStorage(settings.results_bucket, settings.signer_service_account)


class RequestContextMiddleware:
    """Request IDs, access logs, body size limit, CSRF check, and the 500 envelope.

    A pure ASGI middleware so the request ID is still set while an unhandled exception is
    turned into the error envelope.
    """

    def __init__(self, app: ASGIApp, *, csrf: bool) -> None:
        self.app = app
        self.csrf = csrf

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers") or [])
        incoming = headers.get(b"x-request-id", b"").decode("latin-1")
        request_id = incoming if _REQUEST_ID_RE.match(incoming) else uuid.uuid4().hex
        rid_token = request_id_var.set(request_id)
        principal_token = principal_var.set(None)
        method = scope["method"]
        path = scope["path"]
        start = time.perf_counter()
        status = {"code": 500, "started": False}

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
                status["started"] = True
                message.setdefault("headers", [])
                message["headers"] = [
                    *message["headers"],
                    (b"x-request-id", request_id.encode()),
                ]
            await send(message)

        async def reply(code: int, error: str, message: str) -> None:
            import orjson

            body = orjson.dumps(error_body(error, message))
            await send_wrapper(
                {
                    "type": "http.response.start",
                    "status": code,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(body)).encode()),
                    ],
                }
            )
            await send_wrapper({"type": "http.response.body", "body": body})

        try:
            length = headers.get(b"content-length")
            if length is not None and length.isdigit() and int(length) > MAX_BODY_BYTES:
                await reply(413, "payload_too_large", "request body exceeds 16 MiB")
                return
            if (
                self.csrf
                and method in _UNSAFE_METHODS
                and path.startswith("/api/")
                and headers.get(CSRF_HEADER) != CSRF_VALUE
            ):
                await reply(403, "forbidden", "missing X-Requested-With: XMLHttpRequest header")
                return

            received = 0

            async def limited_receive() -> Message:
                nonlocal received
                message = await receive()
                if message["type"] == "http.request":
                    received += len(message.get("body", b""))
                    if received > MAX_BODY_BYTES:
                        raise _BodyTooLarge
                return message

            await self.app(scope, limited_receive, send_wrapper)
        except _BodyTooLarge:
            if not status["started"]:
                await reply(413, "payload_too_large", "request body exceeds 16 MiB")
        except Exception:
            logger.exception("unhandled error on %s %s", method, path)
            if not status["started"]:
                await reply(500, "internal", "internal server error")
        finally:
            if path != "/health" or status["code"] != 200:
                logger.info(
                    "%s %s %s",
                    method,
                    path,
                    status["code"],
                    extra={
                        "http_method": method,
                        "path": path,
                        "status": status["code"],
                        "duration_ms": round((time.perf_counter() - start) * 1000, 1),
                    },
                )
            principal_var.reset(principal_token)
            request_id_var.reset(rid_token)


class _BodyTooLarge(Exception):
    pass


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.ready = False
        db = await make_database(settings)
        app.state.db = db
        app.state.storage = make_storage(settings)
        app.state.token_verifier = TokenVerifier(settings)
        app.state.iap_verifier = IapVerifier(settings.iap_audience)
        try:
            if settings.run_migrations:
                await run_migrations(db.engine)
                await prune_stats_cache(db.engine)
                await grant_dashboard_access(db.engine, settings.dashboard_db_user)
            app.state.ready = True
            logger.info(
                "started",
                extra={
                    "mode": settings.api_mode,
                    "skiff_env": settings.skiff_env,
                    "git_sha": settings.git_sha,
                    "db_name": settings.db_name,
                    "storage": settings.storage_backend,
                },
            )
            yield
        finally:
            await app.state.token_verifier.aclose()
            await app.state.iap_verifier.aclose()
            await app.state.storage.aclose()
            await db.close()

    # The ingest service is reachable without IAP, so it serves no docs outside local dev.
    if settings.api_mode != "ingest":
        docs_url, openapi_url = "/api/docs", "/api/openapi.json"
    elif settings.is_local:
        docs_url, openapi_url = "/docs", "/openapi.json"
    else:
        docs_url = openapi_url = None
    app = FastAPI(
        title=f"olmo-eval API ({settings.api_mode})",
        version=__version__,
        lifespan=lifespan,
        docs_url=docs_url,
        openapi_url=openapi_url,
        redoc_url=None,
    )
    app.state.settings = settings
    install_exception_handlers(app)
    app.add_middleware(RequestContextMiddleware, csrf=settings.api_mode != "ingest")
    _mount_routes(app, settings)
    return app


def _mount_routes(app: FastAPI, settings: Settings) -> None:
    from olmo_eval_api.routers import health

    app.include_router(health.router)
    mode = settings.api_mode
    if mode in ("dashboard", "all"):
        from olmo_eval_api.routers.api import router as api_router

        app.include_router(api_router)
    if mode in ("ingest", "all"):
        from olmo_eval_api.routers import ingest

        app.include_router(ingest.router)
    if settings.is_local and settings.storage_backend == "local" and mode != "dashboard":
        from olmo_eval_api.routers import local_storage

        app.include_router(local_storage.router)


def __getattr__(name: str) -> Any:
    # ``uvicorn olmo_eval_api.main:app`` builds the app on first access so importing this
    # module (tests, tooling) does not read the environment.
    if name == "app":
        application = create_app()
        globals()["app"] = application
        return application
    raise AttributeError(name)
