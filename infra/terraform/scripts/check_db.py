# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "cloud-sql-python-connector[asyncpg]>=1.18",
#     "google-auth>=2.40",
# ]
# ///
"""Check that the IAM database users can connect to Cloud SQL with the privileges they need.

Run after `terraform apply` and scripts/grant_db.py:

    uv run infra/terraform/scripts/check_db.py                    # as the ingest (api) account
    uv run infra/terraform/scripts/check_db.py --as dashboard     # as the dashboard account
    uv run infra/terraform/scripts/check_db.py --as-self --user you@allenai.org  # as your ADC user

The default impersonates a runtime service account, which needs
roles/iam.serviceAccountTokenCreator on it (granted to var.debug_iam_users). For each database:

- api: checks the user has CREATE on schema public and is not a cloudsqlsuperuser member, creates
  and drops a temp table and a table in public, and runs CREATE EXTENSION IF NOT EXISTS pg_trgm.
  --as-self runs the same checks for your own user, except the cloudsqlsuperuser one.
- dashboard: checks the user is not a cloudsqlsuperuser member, cannot create in public, can
  SELECT every table in public and has the writes in grant_db.DASHBOARD_WRITES.

It exits non-zero on any failure.
"""

import argparse
import asyncio
import sys
from pathlib import Path

import google.auth
from google.auth import impersonated_credentials
from google.cloud.sql.connector import Connector, IPTypes

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grant_db import DASHBOARD_WRITES  # noqa: E402

INSTANCE = "ai2-skiff2-olmo-eval:us-west1:olmo-eval-db"
SERVICE_ACCOUNTS = {
    "api": "olmo-eval-api@ai2-skiff2-olmo-eval.iam.gserviceaccount.com",
    "dashboard": "olmo-eval-dashboard@ai2-skiff2-olmo-eval.iam.gserviceaccount.com",
}
DATABASES = ["olmo_eval", "olmo_eval_dev"]
SCOPES = [
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/sqlservice.login",
]


def _credentials(as_self: bool, account: str):
    source, _ = google.auth.default(scopes=SCOPES)
    if as_self:
        return source, None
    sa = SERVICE_ACCOUNTS[account]
    creds = impersonated_credentials.Credentials(
        source_credentials=source, target_principal=sa, target_scopes=SCOPES
    )
    return creds, sa.removesuffix(".gserviceaccount.com")


async def _check_owner(conn, database: str, as_self: bool) -> bool:
    row = await conn.fetchrow(
        "SELECT current_user, "
        "pg_has_role(current_user, 'cloudsqlsuperuser', 'MEMBER') AS superuser, "
        "has_schema_privilege('public', 'CREATE') AS schema_create"
    )
    print(
        f"{database}: user={row['current_user']} cloudsqlsuperuser={row['superuser']} "
        f"schema_create={row['schema_create']}"
    )
    await conn.execute("CREATE TEMP TABLE check_db_probe (id int)")
    await conn.execute("DROP TABLE check_db_probe")
    await conn.execute("CREATE TABLE public.check_db_probe (id int)")
    await conn.execute("DROP TABLE public.check_db_probe")
    await conn.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    print(f"{database}: temp table, public table and pg_trgm OK")
    return bool(row["schema_create"]) and (as_self or not row["superuser"])


async def _check_dashboard(conn, database: str) -> bool:
    row = await conn.fetchrow(
        "SELECT current_user, "
        "pg_has_role(current_user, 'cloudsqlsuperuser', 'MEMBER') AS superuser, "
        "has_schema_privilege('public', 'CREATE') AS schema_create"
    )
    print(
        f"{database}: user={row['current_user']} cloudsqlsuperuser={row['superuser']} "
        f"schema_create={row['schema_create']}"
    )
    ok = not row["superuser"] and not row["schema_create"]
    tables = [
        r["tablename"]
        for r in await conn.fetch(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY 1"
        )
    ]
    for table in tables:
        # A real read, so a missing grant fails here rather than in the dashboard.
        await conn.fetch(f'SELECT 1 FROM public."{table}" LIMIT 1')
    print(f"{database}: SELECT OK on {len(tables)} tables")
    for table, (privileges, columns) in DASHBOARD_WRITES.items():
        if table not in tables:
            continue
        for privilege in privileges:
            if columns:
                checks = [
                    await conn.fetchval(
                        "SELECT has_column_privilege($1, $2, $3)", f"public.{table}", c, privilege
                    )
                    for c in columns
                ]
                has = all(checks)
            else:
                has = await conn.fetchval(
                    "SELECT has_table_privilege($1, $2)", f"public.{table}", privilege
                )
            print(
                f"{database}: {privilege} on {table}{' ' + str(columns) if columns else ''}: {has}"
            )
            ok = ok and bool(has)
    return ok


async def _check(
    connector: Connector, user: str, database: str, dashboard: bool, as_self: bool
) -> bool:
    conn = await connector.connect_async(
        INSTANCE, "asyncpg", user=user, db=database, enable_iam_auth=True, ip_type=IPTypes.PUBLIC
    )
    try:
        if dashboard:
            return await _check_dashboard(conn, database)
        return await _check_owner(conn, database, as_self)
    finally:
        await conn.close()


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--as",
        dest="account",
        choices=sorted(SERVICE_ACCOUNTS),
        default="api",
        help="runtime service account to impersonate (default: api)",
    )
    parser.add_argument("--as-self", action="store_true", help="connect as your own ADC user")
    parser.add_argument("--user", help="IAM database user (default: derived from credentials)")
    args = parser.parse_args()

    creds, user = _credentials(args.as_self, args.account)
    user = args.user or user
    if user is None:
        parser.error("--as-self needs --user <your email>")
    dashboard = args.account == "dashboard" and not args.as_self

    ok = True
    async with Connector(
        credentials=creds, refresh_strategy="lazy", loop=asyncio.get_running_loop()
    ) as connector:
        for database in DATABASES:
            ok = await _check(connector, user, database, dashboard, args.as_self) and ok
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
