# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "cloud-sql-python-connector[asyncpg]>=1.18",
#     "google-auth>=2.40",
# ]
# ///
"""Grant the database privileges the ingest and dashboard IAM users need.

Run after `terraform apply` creates the IAM database users, as a member of var.debug_iam_users:

    uv run infra/terraform/scripts/grant_db.py --user you@allenai.org

The script is idempotent. For each database it runs two steps:

1. As you (a cloudsqlsuperuser member, so an owner of schema public): grant USAGE and CREATE on
   schema public to the ingest user, which migrations need now that it is not a cloudsqlsuperuser
   member, and USAGE to the dashboard user. It also makes sure pg_trgm is installed, since the
   ingest user can no longer create extensions.
2. As the ingest service account (impersonated; it owns the tables): grant the dashboard user
   SELECT on every table, SELECT on tables the ingest user creates later, and the dashboard's
   writes (DASHBOARD_WRITES). The ingest service re-applies these grants after each migration.

Terraform removes the ingest user from cloudsqlsuperuser (`google_sql_user.api.database_roles =
[]`); the script warns if it is still a member.
"""

import argparse
import asyncio
import sys

import google.auth
from google.auth import impersonated_credentials
from google.cloud.sql.connector import Connector, IPTypes

INSTANCE = "ai2-skiff2-olmo-eval:us-west1:olmo-eval-db"
API_SA = "olmo-eval-api@ai2-skiff2-olmo-eval.iam.gserviceaccount.com"
DASHBOARD_SA = "olmo-eval-dashboard@ai2-skiff2-olmo-eval.iam.gserviceaccount.com"
API_USER = API_SA.removesuffix(".gserviceaccount.com")
DASHBOARD_USER = DASHBOARD_SA.removesuffix(".gserviceaccount.com")
DATABASES = ["olmo_eval", "olmo_eval_dev"]
SCOPES = [
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/sqlservice.login",
]

# What the dashboard writes, beyond SELECT on every table (dashboard/api routers/api: saved
# views, run tags and notes, and the stats cache).
# Each entry is table -> (privileges, columns); no columns means the whole table.
DASHBOARD_WRITES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "saved_views": (("INSERT", "UPDATE", "DELETE"), ()),
    "stats_cache": (("INSERT", "UPDATE", "DELETE"), ()),
    "runs": (("UPDATE",), ("tags", "notes", "search_text", "updated_at")),
}


def _q(ident: str) -> str:
    return '"' + ident.replace('"', '""') + '"'


async def _connect(connector: Connector, user: str, database: str):
    return await connector.connect_async(
        INSTANCE, "asyncpg", user=user, db=database, enable_iam_auth=True, ip_type=IPTypes.PUBLIC
    )


async def _as_schema_owner(connector: Connector, user: str, database: str) -> None:
    conn = await _connect(connector, user, database)
    try:
        await conn.execute(f"GRANT USAGE, CREATE ON SCHEMA public TO {_q(API_USER)}")
        await conn.execute(f"GRANT USAGE ON SCHEMA public TO {_q(DASHBOARD_USER)}")
        await conn.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
        member = await conn.fetchval(
            "SELECT pg_has_role($1, 'cloudsqlsuperuser', 'MEMBER')", API_USER
        )
        if member:
            # Only Cloud SQL's admin can revoke it, through google_sql_user.api.database_roles.
            print(f"{database}: WARNING {API_USER} is still a cloudsqlsuperuser member")
        print(f"{database}: schema public grants OK")
    finally:
        await conn.close()


async def _as_table_owner(connector: Connector, database: str) -> None:
    conn = await _connect(connector, API_USER, database)
    try:
        dash = _q(DASHBOARD_USER)
        await conn.execute(f"GRANT SELECT ON ALL TABLES IN SCHEMA public TO {dash}")
        await conn.execute(
            f"ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO {dash}"
        )
        tables = {
            r["tablename"]
            for r in await conn.fetch("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
        }
        for table, (privileges, columns) in DASHBOARD_WRITES.items():
            if table in tables:
                cols = f" ({', '.join(map(_q, columns))})" if columns else ""
                grant = ", ".join(p + cols for p in privileges)
                await conn.execute(f"GRANT {grant} ON public.{_q(table)} TO {dash}")
        missing = sorted(set(DASHBOARD_WRITES) - tables)
        note = f" (not migrated yet: {', '.join(missing)})" if missing else ""
        print(f"{database}: dashboard grants OK{note}")
    finally:
        await conn.close()


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--user", required=True, help="your IAM database user (your email)")
    args = parser.parse_args()

    source, _ = google.auth.default(scopes=SCOPES)
    api_creds = impersonated_credentials.Credentials(
        source_credentials=source, target_principal=API_SA, target_scopes=SCOPES
    )
    loop = asyncio.get_running_loop()
    async with (
        Connector(credentials=source, refresh_strategy="lazy", loop=loop) as as_self,
        Connector(credentials=api_creds, refresh_strategy="lazy", loop=loop) as as_api,
    ):
        for database in DATABASES:
            await _as_schema_owner(as_self, args.user, database)
            await _as_table_owner(as_api, database)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
