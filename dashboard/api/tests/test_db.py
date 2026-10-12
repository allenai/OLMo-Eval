"""Migrations, model/migration drift, and GCS signing with a mocked client."""

from __future__ import annotations

from typing import Any

import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError

from olmo_eval_api.db.grants import grant_dashboard_access
from olmo_eval_api.db.migrate import run_migrations
from olmo_eval_api.db.models import metadata
from olmo_eval_api.storage.gcs import GcsStorage


async def test_migrations_are_idempotent(engine: Any) -> None:
    await run_migrations(engine)
    async with engine.connect() as conn:
        version = (await conn.execute(text("SELECT version_num FROM alembic_version"))).scalar()
        assert version == "0005"
        ext = (
            await conn.execute(text("SELECT 1 FROM pg_extension WHERE extname = 'pg_trgm'"))
        ).scalar()
        assert ext == 1


async def test_models_match_migrations(engine: Any) -> None:
    def diff(sync_conn: Any) -> list:
        ctx = MigrationContext.configure(sync_conn, opts={"compare_type": True})
        return compare_metadata(ctx, metadata)

    async with engine.connect() as conn:
        changes = await conn.run_sync(diff)
    assert changes == []


class FakeBlob:
    def __init__(self, name: str, calls: list) -> None:
        self.name = name
        self.calls = calls

    def generate_signed_url(self, **kwargs: Any) -> str:
        self.calls.append((self.name, kwargs))
        return f"https://storage.googleapis.com/bucket/{self.name}?X-Goog-Signature=x"


class FakeBucket:
    def __init__(self, calls: list) -> None:
        self.calls = calls

    def blob(self, name: str) -> FakeBlob:
        return FakeBlob(name, self.calls)


class FakeClient:
    def __init__(self) -> None:
        self.calls: list = []

    def bucket(self, name: str) -> FakeBucket:
        return FakeBucket(self.calls)


class FakeCreds:
    token = "runtime-token"
    valid = True
    expiry = None


async def test_gcs_signing_uses_signblob() -> None:
    client = FakeClient()
    storage = GcsStorage("bucket", "olmo-eval-api@proj.iam.gserviceaccount.com", client=client)
    storage._credentials = FakeCreds()
    signed = await storage.sign_upload(
        "dev/runs/abc/metrics.json", "application/json", "x" * 22 + "==", 1234
    )
    assert signed.headers == {
        "Content-Type": "application/json",
        "Content-MD5": "x" * 22 + "==",
        "x-goog-content-length-range": "1234,1234",
    }
    name, kwargs = client.calls[0]
    assert name == "dev/runs/abc/metrics.json"
    assert kwargs["version"] == "v4" and kwargs["method"] == "PUT"
    assert kwargs["service_account_email"] == "olmo-eval-api@proj.iam.gserviceaccount.com"
    assert kwargs["access_token"] == "runtime-token"
    assert kwargs["content_md5"] == "x" * 22 + "=="
    # The size is part of the signature, so GCS rejects a body of any other size.
    assert kwargs["headers"] == {"x-goog-content-length-range": "1234,1234"}
    assert kwargs["expiration"].total_seconds() == 3600

    download = await storage.sign_download("dev/runs/abc/metrics.json")
    _, kwargs = client.calls[1]
    assert kwargs["method"] == "GET"
    assert kwargs["expiration"].total_seconds() == 900
    assert kwargs["response_disposition"] == 'attachment; filename="metrics.json"'
    assert download.url.startswith("https://storage.googleapis.com/")


async def test_failed_migration_reports_its_error_and_releases_the_lock(
    engine: Any, monkeypatch: Any
) -> None:
    from olmo_eval_api.db import migrate

    def broken(connection: Any) -> None:
        connection.execute(text("SELECT * FROM no_such_table"))  # aborts the transaction

    monkeypatch.setattr(migrate, "_upgrade_head", broken)
    with pytest.raises(ProgrammingError, match="no_such_table"):
        await run_migrations(engine)
    async with engine.connect() as conn:
        locked = await conn.scalar(text(f"SELECT pg_try_advisory_lock({migrate.ADVISORY_LOCK_ID})"))
        assert locked is True
        await conn.execute(text(f"SELECT pg_advisory_unlock({migrate.ADVISORY_LOCK_ID})"))


async def test_grant_dashboard_access(engine: Any) -> None:
    role = "olmo-eval-dashboard-test@example.iam"
    assert await grant_dashboard_access(engine, role) is False  # no role: skipped
    async with engine.begin() as conn:
        await conn.execute(text(f'CREATE ROLE "{role}"'))
    try:
        for _ in range(2):  # idempotent
            assert await grant_dashboard_access(engine, role) is True
        async with engine.begin() as conn:

            async def has(sql: str, *args: str) -> bool:
                return bool(
                    await conn.scalar(text(sql), dict(zip("abc", (role, *args), strict=False)))
                )

            table = "SELECT has_table_privilege(:a, :b, :c)"
            column = "SELECT has_column_privilege(:a, 'runs', :b, 'UPDATE')"
            for t in ("runs", "task_results", "instance_results", "models", "stats_cache"):
                assert await has(table, t, "SELECT")
            for priv in ("INSERT", "UPDATE", "DELETE"):
                assert await has(table, "saved_views", priv)
                assert await has(table, "stats_cache", priv)
            assert not await has(table, "runs", "DELETE")
            assert not await has(table, "runs", "INSERT")
            assert not await has(table, "task_results", "UPDATE")
            assert not await has(table, "runs", "UPDATE")  # only some columns
            for col in ("tags", "notes", "search_text", "updated_at"):
                assert await has(column, col)
            assert not await has(column, "author")
            # Tables created later by the owner are readable too.
            await conn.execute(text("CREATE TABLE grants_probe (x int)"))
            assert await has(table, "grants_probe", "SELECT")
            await conn.execute(text("DROP TABLE grants_probe"))
    finally:
        async with engine.begin() as conn:
            await conn.execute(text(f'DROP OWNED BY "{role}"'))
            await conn.execute(text(f'DROP ROLE "{role}"'))
