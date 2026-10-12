"""Shared fixtures.

Database tests need TEST_DATABASE_URL, e.g.
``postgresql+asyncpg://olmo@localhost:5433/olmo_eval_test`` from
``docker compose -f dashboard/docker-compose.yml up db``. Without it they are skipped. The
schema is dropped and migrated once per session; every table is truncated after each test.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine

from olmo_eval_api.db.migrate import run_migrations
from olmo_eval_api.db.models import ALL_TABLES
from olmo_eval_api.main import create_app
from olmo_eval_api.services.cache import clear_memory_cache
from olmo_eval_api.settings import Settings

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")

DEFAULT_HEADERS = {
    "X-Goog-Authenticated-User-Email": "accounts.google.com:tester@allenai.org",
    "X-Requested-With": "XMLHttpRequest",
    "X-Olmo-Eval-Token": "dev:tester@allenai.org",
}


def _require_db() -> str:
    if not TEST_DATABASE_URL:
        pytest.skip(
            "TEST_DATABASE_URL is not set; start Postgres with "
            "`docker compose -f dashboard/docker-compose.yml up -d db` and export "
            "TEST_DATABASE_URL=postgresql+asyncpg://olmo@localhost:5433/olmo_eval_test"
        )
    return TEST_DATABASE_URL


@pytest.fixture(scope="session")
async def engine() -> AsyncIterator[AsyncEngine]:
    url = _require_db()
    eng = create_async_engine(url)
    async with eng.begin() as conn:
        await conn.execute(text("DROP SCHEMA public CASCADE"))
        await conn.execute(text("CREATE SCHEMA public"))
    await run_migrations(eng)
    yield eng
    await eng.dispose()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        api_mode="all",
        skiff_env="local",
        db_url=_require_db(),
        storage_backend="local",
        local_storage_dir=str(tmp_path / "storage"),
        ingest_dev_auth=True,
        run_migrations=False,
        public_base_url="http://test",
    )


@pytest.fixture
async def app(engine: AsyncEngine, settings: Settings) -> AsyncIterator[Any]:
    application = create_app(settings)
    async with application.router.lifespan_context(application):
        yield application
    async with engine.begin() as conn:
        await conn.execute(text(f"TRUNCATE {', '.join(ALL_TABLES)} RESTART IDENTITY CASCADE"))
    # The database is emptied, so cached stats from this test must not reach the next one.
    clear_memory_cache()


@pytest.fixture
async def client(app: Any) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", headers=DEFAULT_HEADERS
    ) as c:
        yield c


@pytest.fixture
async def session(app: Any) -> AsyncIterator[AsyncSession]:
    async with app.state.db.sessionmaker() as s:
        yield s
