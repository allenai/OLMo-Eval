"""Health endpoint used by the Cloud Run startup probe."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict
from sqlalchemy import text

from olmo_eval_api.errors import ApiError
from olmo_eval_api.settings import Settings


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"]
    mode: Literal["dashboard", "ingest", "all"]
    git_sha: str | None
    skiff_env: str
    db: Literal["ok"]


router = APIRouter()


async def health(request: Request) -> HealthResponse:
    """200 only after migrations finished and the database answers ``SELECT 1``."""
    settings: Settings = request.app.state.settings
    if not getattr(request.app.state, "ready", False):
        raise ApiError(503, "starting up")
    try:
        async with request.app.state.db.engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception as exc:
        raise ApiError(503, "database unavailable") from exc
    return HealthResponse(
        status="ok",
        mode=settings.api_mode,
        git_sha=settings.git_sha or None,
        skiff_env=settings.skiff_env,
        db="ok",
    )


router.add_api_route("/health", health, methods=["GET"], response_model=HealthResponse)
