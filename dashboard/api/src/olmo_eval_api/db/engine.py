"""Database engine and session dependency.

In Cloud Run the engine connects through the Cloud SQL Python Connector with IAM database
authentication (the runtime service account is the database user, no password). Local
development and tests set ``DB_URL`` to a plain ``postgresql+asyncpg://`` URL instead.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import orjson
from fastapi import Request
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from olmo_eval_api.settings import Settings

logger = logging.getLogger(__name__)


def _json_serializer(obj: Any) -> str:
    return orjson.dumps(obj, option=orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY).decode()


@dataclass
class Database:
    engine: AsyncEngine
    sessionmaker: async_sessionmaker[AsyncSession]
    connector: Any = None

    async def close(self) -> None:
        await self.engine.dispose()
        if self.connector is not None:
            await self.connector.close_async()


async def make_database(settings: Settings) -> Database:
    common: dict[str, Any] = {
        "pool_size": settings.db_pool_size,
        "max_overflow": settings.db_max_overflow,
        "pool_pre_ping": True,
        "pool_recycle": 1800,
        "json_serializer": _json_serializer,
        "json_deserializer": orjson.loads,
    }
    connector = None
    if settings.db_url:
        engine = create_async_engine(settings.db_url, **common)
    else:
        from google.cloud.sql.connector import IPTypes, create_async_connector

        connector = await create_async_connector(refresh_strategy="lazy")
        instance = settings.db_instance_connection_name

        async def getconn() -> Any:
            return await connector.connect_async(
                instance,
                "asyncpg",
                user=settings.db_iam_user,
                db=settings.db_name,
                enable_iam_auth=True,
                ip_type=IPTypes.PUBLIC,
            )

        engine = create_async_engine("postgresql+asyncpg://", async_creator=getconn, **common)
        logger.info(
            "using Cloud SQL connector",
            extra={"instance": instance, "db_name": settings.db_name, "user": settings.db_iam_user},
        )
    sessionmaker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    return Database(engine=engine, sessionmaker=sessionmaker, connector=connector)


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    db: Database = request.app.state.db
    async with db.sessionmaker() as session:
        yield session
