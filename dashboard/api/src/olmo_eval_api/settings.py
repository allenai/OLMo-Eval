"""Runtime settings.

Every value has a default derived from SKIFF_ENV, which Skiff2 sets on every Cloud Run
container (``prod`` or the sanitized branch name). Nothing needs to be configured in
Cloud Run for the service to start.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

ApiMode = Literal["dashboard", "ingest", "all"]

GCP_PROJECT = "ai2-skiff2-olmo-eval"
GCP_PROJECT_NUMBER = "354333262681"
GCP_REGION = "us-west1"
# The ingest service runs as this account. Its database user owns the schema and migrates it.
API_SERVICE_ACCOUNT = f"olmo-eval-api@{GCP_PROJECT}.iam.gserviceaccount.com"
# The ui service (and its API_MODE=dashboard sidecar) runs as this account. Its database user
# reads every table and writes only what db/grants.py grants it.
DASHBOARD_SERVICE_ACCOUNT = f"olmo-eval-dashboard@{GCP_PROJECT}.iam.gserviceaccount.com"
# Beaker jobs upload as this account (infra/terraform/uploader.tf). It has no GCP roles.
UPLOADER_SERVICE_ACCOUNT = f"olmo-eval-uploader@{GCP_PROJECT}.iam.gserviceaccount.com"


def iam_db_user(service_account: str) -> str:
    """Cloud SQL IAM database user of a service account (the email without the suffix)."""
    return service_account.removesuffix(".gserviceaccount.com")


def _split_csv(value: object) -> object:
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return value


class Settings(BaseSettings):
    model_config = SettingsConfigDict(case_sensitive=False, extra="ignore")

    api_mode: ApiMode = "dashboard"
    skiff_env: str = "local"
    git_sha: str = ""
    port: int = 8000

    db_url: str | None = None
    db_instance_connection_name: str = f"{GCP_PROJECT}:us-west1:olmo-eval-db"
    # Empty means the default for the mode: the dashboard account's database user in dashboard
    # mode, the ingest account's otherwise.
    db_iam_user: str = ""
    # The database role the ingest service grants dashboard access to after migrating.
    dashboard_db_user: str = iam_db_user(DASHBOARD_SERVICE_ACCOUNT)
    db_name: str | None = None
    db_pool_size: int = 5
    db_max_overflow: int = 5
    # None means the default for the mode: off for the dashboard outside local development,
    # since its database user cannot create tables.
    run_migrations: bool | None = None

    storage_backend: Literal["gcs", "local"] | None = None
    results_bucket: str = "ai2-skiff2-olmo-eval-results"
    results_prefix: str | None = None
    local_storage_dir: str = "/tmp/olmo-eval-api-storage"
    # Empty means the service's own account (by mode, like db_iam_user).
    signer_service_account: str = ""
    public_base_url: str | None = None
    dashboard_base_url: str | None = None

    ingest_allowed_domains: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["allenai.org"]
    )
    ingest_allowed_service_accounts: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: [UPLOADER_SERVICE_ACCOUNT]
    )
    ingest_dev_auth: bool = False
    dashboard_dev_user: str = "dev@allenai.org"
    # Outside local development the dashboard verifies IAP's signed header (auth/iap.py) and
    # accepts only these email domains.
    dashboard_allowed_domains: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["allenai.org"]
    )
    # Expected audience of the IAP JWT; empty means the Cloud Run service from K_SERVICE.
    iap_audience: str = ""

    log_level: str = "INFO"
    # Cloud Run sets K_SERVICE on every container. Used only to refuse local mode there.
    k_service: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _parse_lists(cls, data: object) -> object:
        if isinstance(data, dict):
            for key in (
                "ingest_allowed_domains",
                "ingest_allowed_service_accounts",
                "dashboard_allowed_domains",
            ):
                for k in (key, key.upper()):
                    if k in data:
                        data[k] = _split_csv(data[k])
        return data

    @model_validator(mode="after")
    def _derive_defaults(self) -> Settings:
        prod = self.skiff_env == "prod"
        local = self.is_local
        dashboard = self.api_mode == "dashboard" and not local
        account = DASHBOARD_SERVICE_ACCOUNT if dashboard else API_SERVICE_ACCOUNT
        if not self.db_iam_user:
            self.db_iam_user = iam_db_user(account)
        if not self.signer_service_account:
            self.signer_service_account = account
        if self.run_migrations is None:
            self.run_migrations = not dashboard
        if not self.iap_audience and self.k_service:
            self.iap_audience = (
                f"/projects/{GCP_PROJECT_NUMBER}/locations/{GCP_REGION}/services/{self.k_service}"
            )
        if self.db_name is None:
            self.db_name = "olmo_eval" if prod else "olmo_eval_dev"
        if self.storage_backend is None:
            self.storage_backend = "local" if (local and self.db_url) else "gcs"
        if self.results_prefix is None:
            self.results_prefix = "" if prod else "dev/"
        if self.public_base_url is None:
            self.public_base_url = f"http://localhost:{self.port}"
        if self.dashboard_base_url is None:
            if prod:
                self.dashboard_base_url = "https://olmo-eval.allen.ai"
            elif local:
                self.dashboard_base_url = "http://localhost:5173"
            else:
                self.dashboard_base_url = f"https://{self.skiff_env}-ui.olmo-eval.apps.allenai.org"
        self.public_base_url = self.public_base_url.rstrip("/")
        self.dashboard_base_url = self.dashboard_base_url.rstrip("/")
        self.ingest_allowed_domains = [d.lower() for d in self.ingest_allowed_domains]
        self.dashboard_allowed_domains = [d.lower() for d in self.dashboard_allowed_domains]
        self.ingest_allowed_service_accounts = [
            s.lower() for s in self.ingest_allowed_service_accounts
        ]
        if self.api_mode == "all" and not local:
            raise ValueError("API_MODE=all is allowed only when SKIFF_ENV=local")
        if local and self.k_service:
            # Local mode turns on the dev dashboard user and dev ingest tokens.
            raise ValueError("SKIFF_ENV is unset or local, but this container runs on Cloud Run")
        return self

    @property
    def is_local(self) -> bool:
        return self.skiff_env == "local"

    @property
    def verify_iap(self) -> bool:
        """Whether dashboard requests must carry a valid IAP JWT (auth/iap.py)."""
        return self.api_mode == "dashboard" and not self.is_local

    @property
    def dev_auth_enabled(self) -> bool:
        return self.is_local and self.ingest_dev_auth

    def gcs_prefix(self, run_id: str) -> str:
        return f"gs://{self.results_bucket}/{self.object_prefix(run_id)}"

    def object_prefix(self, run_id: str) -> str:
        return f"{self.results_prefix}runs/{run_id}/"

    def object_key(self, run_id: str, relative_path: str) -> str:
        return self.object_prefix(run_id) + relative_path

    def dashboard_run_url(self, run_id: str) -> str:
        return f"{self.dashboard_base_url}/runs/{run_id}"


@lru_cache
def get_settings() -> Settings:
    return Settings()
