"""Database privileges of the dashboard's IAM user, granted by the ingest service.

The ingest user owns every table (it runs the migrations). After migrating, it grants the
dashboard user SELECT on every table, SELECT on tables it creates later, and the few writes the
dashboard makes. infra/terraform/scripts/grant_db.py applies the same grants from a workstation;
keep DASHBOARD_WRITES identical in both.

Granting is idempotent. Where the dashboard role does not exist (local development, tests) the
step is skipped.
"""

from __future__ import annotations

import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

logger = logging.getLogger(__name__)

# What the dashboard writes beyond SELECT: table -> (privileges, columns); no columns means the
# whole table. Saved views (routers/api/misc.py), run tags and notes (routers/api/runs.py), and
# the stats cache (services/cache.py; DELETE covers the 30-day prune in db/migrate.py).
DASHBOARD_WRITES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "saved_views": (("INSERT", "UPDATE", "DELETE"), ()),
    "stats_cache": (("INSERT", "UPDATE", "DELETE"), ()),
    "runs": (("UPDATE",), ("tags", "notes", "search_text", "updated_at")),
}


def quote_ident(ident: str) -> str:
    return '"' + ident.replace('"', '""') + '"'


def grant_statements(role: str, tables: set[str]) -> list[str]:
    """The GRANT statements for ``role``, limited to the write tables that exist."""
    who = quote_ident(role)
    statements = [
        f"GRANT SELECT ON ALL TABLES IN SCHEMA public TO {who}",
        f"ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO {who}",
    ]
    for table, (privileges, columns) in DASHBOARD_WRITES.items():
        if table in tables:
            cols = f" ({', '.join(map(quote_ident, columns))})" if columns else ""
            grant = ", ".join(p + cols for p in privileges)
            statements.append(f"GRANT {grant} ON public.{quote_ident(table)} TO {who}")
    return statements


async def grant_dashboard_access(engine: AsyncEngine, role: str) -> bool:
    """Grant ``role`` the dashboard's privileges. Returns False when the role does not exist."""
    async with engine.begin() as conn:
        exists = (
            await conn.execute(text("SELECT 1 FROM pg_roles WHERE rolname = :r"), {"r": role})
        ).first()
        if exists is None:
            logger.info(
                "dashboard database role does not exist; skipping grants", extra={"role": role}
            )
            return False
        tables = set(
            (
                await conn.execute(
                    text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
                )
            )
            .scalars()
            .all()
        )
        for statement in grant_statements(role, tables):
            await conn.execute(text(statement))
    logger.info("granted dashboard database access", extra={"role": role})
    return True
