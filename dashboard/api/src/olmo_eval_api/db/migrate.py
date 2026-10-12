"""Run alembic migrations at startup under a Postgres advisory lock.

The ingest service migrates at startup (the dashboard's database user cannot create tables,
so RUN_MIGRATIONS defaults to off in dashboard mode). The lock serializes concurrent
instances; the startup probe keeps traffic away until ``/health`` sees a migrated database.
"""

from __future__ import annotations

import logging
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import Connection, text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

logger = logging.getLogger(__name__)

ADVISORY_LOCK_ID = 727274001
MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"


def alembic_config(connection: Connection | None = None) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS_DIR))
    if connection is not None:
        cfg.attributes["connection"] = connection
    return cfg


def _upgrade_head(connection: Connection) -> None:
    command.upgrade(alembic_config(connection), "head")


async def run_migrations(engine: AsyncEngine) -> None:
    async with engine.connect() as conn:
        # A session-level lock, so it outlives the transactions alembic opens and commits.
        await conn.execute(text(f"SELECT pg_advisory_lock({ADVISORY_LOCK_ID})"))
        await conn.commit()
        try:
            await conn.run_sync(_upgrade_head)
            await conn.commit()
        except BaseException:
            # The transaction may be aborted; roll back before unlocking, and never let a
            # failed unlock hide the migration error.
            await conn.rollback()
            await _unlock(conn, quiet=True)
            raise
        await _unlock(conn)


async def _unlock(conn: AsyncConnection, *, quiet: bool = False) -> None:
    try:
        await conn.execute(text(f"SELECT pg_advisory_unlock({ADVISORY_LOCK_ID})"))
        await conn.commit()
    except Exception:
        if not quiet:
            raise
        # Closing the connection releases the lock anyway.
        logger.exception("could not release the migration lock")


async def prune_stats_cache(engine: AsyncEngine, days: int = 30) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM stats_cache WHERE created_at < now() - make_interval(days => :d)"),
            {"d": days},
        )
