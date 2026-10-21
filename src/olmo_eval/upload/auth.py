"""Google credentials for the ingest service.

The ingest service verifies a Google OAuth access token on every request. Local
runs use Application Default Credentials (``gcloud auth application-default
login``). Beaker jobs use the shared uploader service account key, which
``olmo-eval beaker launch`` injects as OLMO_EVAL_UPLOAD_CREDENTIALS, so personal
credentials never reach Beaker.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

SCOPES = (
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/cloud-platform",
)
NO_CREDENTIALS_MESSAGE = (
    "No Google credentials found. Run `gcloud auth application-default login`, or pass --no-upload."
)
REFRESH_MARGIN = timedelta(minutes=5)

# The uploader key once take_upload_credentials_from_env has moved it out of the environment.
_uploader_key: str | None = None


class UploadAuthError(Exception):
    """Google credentials are missing or cannot produce an access token."""


def take_upload_credentials_from_env() -> None:
    """Move the uploader key out of the environment, keeping it for this process only.

    Evals start subprocesses (inference servers, code sandboxes, external benchmarks) that
    inherit the environment and may run model-written code, so they must not see the key.
    """
    global _uploader_key
    from olmo_eval.upload.config import UPLOAD_CREDENTIALS_ENV

    key = os.environ.pop(UPLOAD_CREDENTIALS_ENV, "").strip()
    if key:
        _uploader_key = key


def _uploader_key_value() -> str:
    from olmo_eval.upload.config import UPLOAD_CREDENTIALS_ENV

    return _uploader_key or os.environ.get(UPLOAD_CREDENTIALS_ENV, "").strip()


def load_google_credentials() -> Any:
    """Load credentials with the scopes the ingest service needs.

    Uses the uploader service account key from OLMO_EVAL_UPLOAD_CREDENTIALS when set,
    else Application Default Credentials.

    Raises:
        UploadAuthError: If no credentials are configured or the key is unreadable.
    """
    import google.auth
    from google.auth.exceptions import DefaultCredentialsError

    from olmo_eval.upload.config import UPLOAD_CREDENTIALS_ENV

    key = _uploader_key_value()
    if key:
        return _service_account_credentials(key, UPLOAD_CREDENTIALS_ENV)
    try:
        credentials, _project = google.auth.default(scopes=list(SCOPES))
    except DefaultCredentialsError as e:
        raise UploadAuthError(NO_CREDENTIALS_MESSAGE) from e
    return credentials


def _service_account_credentials(key: str, env_name: str) -> Any:
    """Build service account credentials from a JSON key or a path to one."""
    from google.oauth2 import service_account

    try:
        info = json.loads(key if key.startswith("{") else Path(key).read_text())
        return service_account.Credentials.from_service_account_info(info, scopes=list(SCOPES))
    except (OSError, ValueError, KeyError) as e:
        raise UploadAuthError(f"{env_name} does not hold a valid service account key: {e}") from e


def has_upload_credentials() -> bool:
    """Whether an uploader key or Application Default Credentials are configured."""
    return bool(_uploader_key_value()) or has_local_google_credentials()


def has_local_google_credentials() -> bool:
    """Whether Application Default Credentials are configured on this machine."""
    import google.auth
    from google.auth.exceptions import DefaultCredentialsError

    try:
        google.auth.default(scopes=list(SCOPES))
    except DefaultCredentialsError:
        return False
    return True


def _expires_soon(expiry: datetime | None) -> bool:
    if expiry is None:
        return False
    if expiry.tzinfo is None:
        expiry = expiry.replace(tzinfo=UTC)
    return expiry - datetime.now(UTC) < REFRESH_MARGIN


class GoogleTokenProvider:
    """Thread-safe source of Google access tokens.

    Credentials load lazily on the first call, and the token refreshes when it is
    missing or expires within five minutes.
    """

    def __init__(self, credentials: Any | None = None) -> None:
        self._credentials = credentials
        self._lock = threading.Lock()

    def token(self) -> str:
        """Return a valid access token.

        Raises:
            UploadAuthError: If credentials are missing or the refresh fails.
        """
        with self._lock:
            if self._credentials is None:
                self._credentials = load_google_credentials()
            creds = self._credentials
            if not creds.token or not creds.valid or _expires_soon(creds.expiry):
                self._refresh(creds)
            if not creds.token:
                raise UploadAuthError(
                    "Google credentials returned no access token. Run "
                    "`gcloud auth application-default login`, or pass --no-upload."
                )
            return creds.token

    @staticmethod
    def _refresh(creds: Any) -> None:
        from google.auth.exceptions import GoogleAuthError
        from google.auth.transport.requests import Request

        try:
            creds.refresh(Request())
        except GoogleAuthError as e:
            raise UploadAuthError(
                f"Could not refresh Google credentials ({e}). Run "
                "`gcloud auth application-default login`, or pass --no-upload."
            ) from e
