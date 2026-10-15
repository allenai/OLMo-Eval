"""Alembic environment.

The API passes its own connection through ``config.attributes["connection"]``. The alembic
CLI has none, so this builds an engine from Settings (``DB_URL`` or the Cloud SQL connector).
"""

from __future__ import annotations

import asyncio

from alembic import context
from sqlalchemy import Connection

from olmo_eval_api.db import models

target_metadata = models.metadata
config = context.config


def _run(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        transaction_per_migration=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def _run_with_own_engine() -> None:
    from olmo_eval_api.db.engine import make_database
    from olmo_eval_api.settings import Settings

    db = await make_database(Settings())
    try:
        async with db.engine.connect() as conn:
            await conn.run_sync(_run)
            await conn.commit()
    finally:
        await db.close()


connection = config.attributes.get("connection")
if connection is not None:
    _run(connection)
else:
    asyncio.run(_run_with_own_engine())
