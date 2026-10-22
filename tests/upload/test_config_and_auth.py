"""Upload settings and Google token acquisition."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from olmo_eval.upload.auth import (
    NO_CREDENTIALS_MESSAGE,
    SCOPES,
    GoogleTokenProvider,
    UploadAuthError,
    load_google_credentials,
)
from olmo_eval.upload.config import (
    DEFAULT_API_URL,
    InvalidApiUrl,
    UploadConfig,
    resolve_upload_config,
    retry_command,
    upload_param_hint,
    validate_api_url,
    validate_tags,
)

# ---------------------------------------------------------------------- config


def test_defaults_upload_to_production(monkeypatch) -> None:
    monkeypatch.delenv("OLMO_EVAL_UPLOAD", raising=False)
    monkeypatch.setattr("olmo_eval.upload.auth.has_upload_credentials", lambda: True)
    config = resolve_upload_config()
    assert config.enabled is True
    assert config.notice is None
    assert config.api_url == DEFAULT_API_URL
    assert config.timeout_s == 1800


def test_default_upload_falls_back_to_local_without_credentials(monkeypatch) -> None:
    monkeypatch.delenv("OLMO_EVAL_UPLOAD", raising=False)
    monkeypatch.setattr("olmo_eval.upload.auth.has_upload_credentials", lambda: False)
    config = resolve_upload_config()
    assert config.enabled is False
    assert config.notice is not None and "gcloud auth application-default login" in config.notice


@pytest.mark.parametrize(("env", "cli"), [("1", None), (None, True)])
def test_requested_upload_stays_on_without_credentials(monkeypatch, env, cli) -> None:
    if env is None:
        monkeypatch.delenv("OLMO_EVAL_UPLOAD", raising=False)
    else:
        monkeypatch.setenv("OLMO_EVAL_UPLOAD", env)
    monkeypatch.setattr("olmo_eval.upload.auth.has_upload_credentials", lambda: False)
    config = resolve_upload_config(cli_upload=cli)
    assert config.enabled is True and config.notice is None


def test_uploader_key_counts_as_credentials(monkeypatch) -> None:
    from olmo_eval.upload import auth

    monkeypatch.setattr(auth, "has_local_google_credentials", lambda: False)
    monkeypatch.delenv("OLMO_EVAL_UPLOAD_CREDENTIALS", raising=False)
    assert auth.has_upload_credentials() is False
    monkeypatch.setenv("OLMO_EVAL_UPLOAD_CREDENTIALS", "/secrets/key.json")
    assert auth.has_upload_credentials() is True


@pytest.mark.parametrize(
    ("env", "cli", "expected"),
    [("0", None, False), ("false", None, False), ("1", None, True), ("0", True, True)],
)
def test_cli_flag_beats_env(monkeypatch, env, cli, expected) -> None:
    monkeypatch.setenv("OLMO_EVAL_UPLOAD", env)
    assert resolve_upload_config(cli_upload=cli).enabled is expected


def test_api_url_and_timeout_from_env(monkeypatch) -> None:
    monkeypatch.setenv("OLMO_EVAL_API_URL", "https://dev-ingest.example/")
    monkeypatch.setenv("OLMO_EVAL_UPLOAD_TIMEOUT", "60")
    config = resolve_upload_config()
    assert config.api_url == "https://dev-ingest.example"
    assert config.timeout_s == 60
    assert resolve_upload_config(cli_api_url="http://localhost:8000").api_url == (
        "http://localhost:8000"
    )


@pytest.mark.parametrize(
    "url",
    [
        "https://prod-ingest.olmo-eval.apps.allenai.org",
        "https://dev-ingest.example:8443/",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://[::1]:8000",
    ],
)
def test_api_url_accepts_https_and_local_http(url) -> None:
    assert validate_api_url(url) == url.rstrip("/")


@pytest.mark.parametrize(
    "url",
    [
        "http://dev-ingest.example",
        "http://localhost.evil.example",
        "http://10.0.0.5:8000",
        "ftp://ingest.example",
        "prod-ingest.olmo-eval.apps.allenai.org",
        "https://",
    ],
)
def test_api_url_rejects_non_https(url) -> None:
    with pytest.raises(InvalidApiUrl, match="https|Invalid"):
        validate_api_url(url)


def test_resolve_rejects_http_api_url_from_env(monkeypatch) -> None:
    monkeypatch.delenv("OLMO_EVAL_UPLOAD", raising=False)
    monkeypatch.setattr("olmo_eval.upload.auth.has_upload_credentials", lambda: True)
    monkeypatch.setenv("OLMO_EVAL_API_URL", "http://ingest.example")
    with pytest.raises(InvalidApiUrl, match="OLMO_EVAL_API_URL") as error:
        resolve_upload_config()
    assert upload_param_hint(error.value) == "--api-url"
    # A disabled upload sends no token, so the URL does not block the run.
    assert resolve_upload_config(cli_upload=False).enabled is False


def test_client_rejects_http_api_url() -> None:
    from olmo_eval.upload.client import IngestClient

    with pytest.raises(InvalidApiUrl):
        IngestClient("http://ingest.example", token_provider=None)  # type: ignore[arg-type]


def test_tags_are_validated_and_deduplicated() -> None:
    assert validate_tags(["a", "b:c", "a"]) == ("a", "b:c")
    with pytest.raises(ValueError, match="Invalid tag") as error:
        validate_tags(["has space"])
    assert upload_param_hint(error.value) == "--tag"
    with pytest.raises(ValueError, match="At most"):
        validate_tags([f"t{i}" for i in range(51)])


def test_retry_command() -> None:
    assert retry_command("/results", UploadConfig()) == "olmo-eval results upload /results"
    custom = UploadConfig(api_url="http://localhost:8000", tags=("x",))
    assert retry_command("/tmp/my results", custom) == (
        "olmo-eval results upload '/tmp/my results' --api-url http://localhost:8000 --tag x"
    )


# ---------------------------------------------------------------------- auth


class FakeCredentials:
    def __init__(self, token=None, valid=False, expiry=None, refresh_token="fresh"):
        self.token = token
        self.valid = valid
        self.expiry = expiry
        self.refreshes = 0
        self._refresh_token = refresh_token

    def refresh(self, request) -> None:
        self.refreshes += 1
        self.token = self._refresh_token
        self.valid = self._refresh_token is not None
        self.expiry = datetime.now(UTC).replace(tzinfo=None) + timedelta(hours=1)


def test_token_refreshes_when_missing() -> None:
    creds = FakeCredentials()
    provider = GoogleTokenProvider(creds)
    assert provider.token() == "fresh"
    assert provider.token() == "fresh"
    assert creds.refreshes == 1


def test_valid_token_is_reused() -> None:
    expiry = datetime.now(UTC).replace(tzinfo=None) + timedelta(minutes=30)
    creds = FakeCredentials(token="cached", valid=True, expiry=expiry)
    assert GoogleTokenProvider(creds).token() == "cached"
    assert creds.refreshes == 0


def test_token_refreshes_within_five_minutes_of_expiry() -> None:
    expiry = datetime.now(UTC) + timedelta(minutes=2)  # tz-aware expiry works too
    creds = FakeCredentials(token="old", valid=True, expiry=expiry)
    assert GoogleTokenProvider(creds).token() == "fresh"


def test_empty_token_after_refresh_raises() -> None:
    with pytest.raises(UploadAuthError, match="no access token"):
        GoogleTokenProvider(FakeCredentials(refresh_token=None)).token()


def test_refresh_error_becomes_upload_auth_error() -> None:
    from google.auth.exceptions import RefreshError

    class Broken(FakeCredentials):
        def refresh(self, request) -> None:
            raise RefreshError("invalid_grant")

    with pytest.raises(UploadAuthError, match="application-default login"):
        GoogleTokenProvider(Broken()).token()


def test_no_credentials_message() -> None:
    from google.auth.exceptions import DefaultCredentialsError

    with (
        patch("google.auth.default", side_effect=DefaultCredentialsError("none")),
        pytest.raises(UploadAuthError) as excinfo,
    ):
        GoogleTokenProvider().token()
    assert str(excinfo.value) == NO_CREDENTIALS_MESSAGE


def test_requests_the_ingest_scopes() -> None:
    with patch("google.auth.default", return_value=(object(), None)) as default:
        load_google_credentials()
    assert default.call_args.kwargs["scopes"] == list(SCOPES)
    assert "https://www.googleapis.com/auth/userinfo.email" in SCOPES


def test_authorized_user_adc_file(tmp_path: Path, monkeypatch) -> None:
    from google.oauth2.credentials import Credentials

    path = tmp_path / "adc.json"
    path.write_text(
        json.dumps(
            {
                "type": "authorized_user",
                "client_id": "id.apps.googleusercontent.com",
                "client_secret": "secret",
                "refresh_token": "refresh",
            }
        )
    )
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(path))
    creds = load_google_credentials()
    assert isinstance(creds, Credentials)
    assert creds.refresh_token == "refresh"


def test_service_account_key_file(tmp_path: Path, monkeypatch) -> None:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from google.oauth2 import service_account

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()
    path = tmp_path / "sa.json"
    path.write_text(
        json.dumps(
            {
                "type": "service_account",
                "project_id": "p",
                "private_key_id": "k",
                "private_key": pem,
                "client_email": "ci@p.iam.gserviceaccount.com",
                "client_id": "1",
                "token_uri": "https://oauth2.googleapis.com/token",
            }
        )
    )
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(path))
    creds = load_google_credentials()
    assert isinstance(creds, service_account.Credentials)
    assert set(SCOPES) <= set(creds.scopes or [])


def _service_account_key_json() -> str:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()
    return json.dumps(
        {
            "type": "service_account",
            "project_id": "p",
            "private_key_id": "k",
            "private_key": pem,
            "client_email": "olmo-eval-uploader@p.iam.gserviceaccount.com",
            "client_id": "1",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
    )


@pytest.mark.parametrize("as_path", [False, True])
def test_uploader_key_env_beats_adc(tmp_path: Path, monkeypatch, as_path: bool) -> None:
    from google.oauth2 import service_account

    key = _service_account_key_json()
    if as_path:
        path = tmp_path / "uploader.json"
        path.write_text(key)
        key = str(path)
    monkeypatch.setenv("OLMO_EVAL_UPLOAD_CREDENTIALS", key)
    with patch("google.auth.default") as default:
        creds = load_google_credentials()
    default.assert_not_called()
    assert isinstance(creds, service_account.Credentials)
    assert creds.service_account_email == "olmo-eval-uploader@p.iam.gserviceaccount.com"
    assert set(SCOPES) <= set(creds.scopes or [])


def test_malformed_uploader_key_is_an_upload_auth_error(monkeypatch) -> None:
    monkeypatch.setenv("OLMO_EVAL_UPLOAD_CREDENTIALS", '{"type": "service_account"}')
    with pytest.raises(UploadAuthError, match="OLMO_EVAL_UPLOAD_CREDENTIALS"):
        load_google_credentials()


def test_uploader_key_is_hidden_from_subprocesses(monkeypatch) -> None:
    import subprocess
    import sys

    from google.oauth2 import service_account

    from olmo_eval.upload.auth import take_upload_credentials_from_env

    monkeypatch.setenv("OLMO_EVAL_UPLOAD_CREDENTIALS", _service_account_key_json())
    take_upload_credentials_from_env()

    assert "OLMO_EVAL_UPLOAD_CREDENTIALS" not in os.environ
    child = subprocess.run(
        [sys.executable, "-c", "import os; print(os.environ.get('OLMO_EVAL_UPLOAD_CREDENTIALS'))"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert child.stdout.strip() == "None"
    with patch("google.auth.default") as default:
        creds = load_google_credentials()
    default.assert_not_called()
    assert isinstance(creds, service_account.Credentials)


def test_cli_moves_the_uploader_key_out_of_the_environment(monkeypatch) -> None:
    from click.testing import CliRunner

    from olmo_eval.cli import main
    from olmo_eval.upload import auth

    monkeypatch.setenv("OLMO_EVAL_UPLOAD_CREDENTIALS", "/secrets/key.json")
    CliRunner().invoke(main, ["results", "--help"])

    assert "OLMO_EVAL_UPLOAD_CREDENTIALS" not in os.environ
    assert auth.has_upload_credentials() is True
