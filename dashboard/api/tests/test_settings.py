"""Defaults that depend on API_MODE: database user, URL signer, migrations, IAP audience."""

from __future__ import annotations

from typing import Any

import pytest

from olmo_eval_api.settings import Settings

API_SA = "olmo-eval-api@ai2-skiff2-olmo-eval.iam.gserviceaccount.com"
DASHBOARD_SA = "olmo-eval-dashboard@ai2-skiff2-olmo-eval.iam.gserviceaccount.com"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "API_MODE", "SKIFF_ENV", "K_SERVICE", "DB_IAM_USER", "SIGNER_SERVICE_ACCOUNT",
        "RUN_MIGRATIONS", "IAP_AUDIENCE",
    ):  # fmt: skip
        monkeypatch.delenv(name, raising=False)


def test_dashboard_defaults() -> None:
    s = Settings(api_mode="dashboard", skiff_env="prod", k_service="ui")
    assert s.db_iam_user == "olmo-eval-dashboard@ai2-skiff2-olmo-eval.iam"
    assert s.signer_service_account == DASHBOARD_SA
    assert s.run_migrations is False
    assert s.verify_iap
    assert s.iap_audience == "/projects/354333262681/locations/us-west1/services/ui"


def test_ingest_defaults() -> None:
    s = Settings(api_mode="ingest", skiff_env="prod", k_service="api")
    assert s.db_iam_user == "olmo-eval-api@ai2-skiff2-olmo-eval.iam"
    assert s.signer_service_account == API_SA
    assert s.run_migrations is True
    assert not s.verify_iap
    assert s.dashboard_db_user == "olmo-eval-dashboard@ai2-skiff2-olmo-eval.iam"


def test_local_defaults_are_unchanged() -> None:
    for mode in ("dashboard", "ingest", "all"):
        s = Settings(api_mode=mode, skiff_env="local")
        assert s.db_iam_user == "olmo-eval-api@ai2-skiff2-olmo-eval.iam"
        assert s.signer_service_account == API_SA
        assert s.run_migrations is True
        assert not s.verify_iap


def test_environment_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_MODE", "dashboard")
    monkeypatch.setenv("SKIFF_ENV", "prod")
    monkeypatch.setenv("K_SERVICE", "ui")
    monkeypatch.setenv("DB_IAM_USER", "someone@proj.iam")
    monkeypatch.setenv("SIGNER_SERVICE_ACCOUNT", "signer@proj.iam.gserviceaccount.com")
    monkeypatch.setenv("RUN_MIGRATIONS", "true")
    monkeypatch.setenv("IAP_AUDIENCE", "/projects/1/global/backendServices/2")
    s = Settings()
    assert s.db_iam_user == "someone@proj.iam"
    assert s.signer_service_account == "signer@proj.iam.gserviceaccount.com"
    assert s.run_migrations is True
    assert s.iap_audience == "/projects/1/global/backendServices/2"


def test_ingest_serves_no_docs_outside_local() -> None:
    from olmo_eval_api.main import create_app

    common: dict[str, Any] = {
        "db_url": "postgresql+asyncpg://nobody@127.0.0.1:1/none",
        "storage_backend": "local",
    }
    ingest = create_app(Settings(api_mode="ingest", skiff_env="prod", k_service="api", **common))
    assert ingest.docs_url is None and ingest.openapi_url is None
    local = create_app(Settings(api_mode="ingest", skiff_env="local", **common))
    assert local.docs_url == "/docs" and local.openapi_url == "/openapi.json"
    dashboard = create_app(
        Settings(api_mode="dashboard", skiff_env="prod", k_service="ui", **common)
    )
    assert dashboard.openapi_url == "/api/openapi.json"
