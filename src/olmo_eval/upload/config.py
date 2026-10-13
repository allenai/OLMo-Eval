"""Upload settings resolved from CLI flags and environment variables."""

from __future__ import annotations

import os
import re
import shlex
from collections.abc import Sequence
from dataclasses import dataclass
from urllib.parse import urlsplit

DEFAULT_API_URL = "https://prod-ingest.olmo-eval.apps.allenai.org"
DEFAULT_TIMEOUT_S = 1800.0

UPLOAD_ENV = "OLMO_EVAL_UPLOAD"
API_URL_ENV = "OLMO_EVAL_API_URL"
TIMEOUT_ENV = "OLMO_EVAL_UPLOAD_TIMEOUT"
LAUNCH_ID_ENV = "OLMO_EVAL_LAUNCH_ID"
# Uploader service account key (JSON content or a file path). Beaker jobs get it from the
# shared olmo_eval_uploader_key secret; when unset, uploads use local credentials.
UPLOAD_CREDENTIALS_ENV = "OLMO_EVAL_UPLOAD_CREDENTIALS"

MAX_TAGS = 50
_TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+-]{0,63}$")
_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}

LOCAL_ONLY_NOTICE = (
    "No Google credentials found, so results are saved locally and not uploaded. "
    "To upload, run `gcloud auth application-default login`. "
    f"Set {UPLOAD_ENV}=0 or pass --no-upload to hide this message."
)


@dataclass(frozen=True)
class UploadConfig:
    """Where and whether to upload results.

    Attributes:
        enabled: Upload results to the ingest service.
        api_url: Base URL of the ingest service.
        tags: Free-form labels attached to the run.
        timeout_s: Overall deadline for one upload, in seconds.
        notice: Why uploads were turned off without being asked, for the CLI to show.
    """

    enabled: bool = True
    api_url: str = DEFAULT_API_URL
    tags: tuple[str, ...] = ()
    timeout_s: float = DEFAULT_TIMEOUT_S
    notice: str | None = None

    @property
    def is_default_api_url(self) -> bool:
        return self.api_url.rstrip("/") == DEFAULT_API_URL


def parse_bool(value: str | None) -> bool | None:
    """Parse a boolean environment value. Returns None when unset or unrecognized."""
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered in _TRUE:
        return True
    if lowered in _FALSE:
        return False
    return None


def env_upload_enabled() -> bool | None:
    """Return the OLMO_EVAL_UPLOAD setting, or None when it is unset."""
    return parse_bool(os.environ.get(UPLOAD_ENV))


def validate_tags(tags: Sequence[str]) -> tuple[str, ...]:
    """Check tags against the ingest contract and drop duplicates.

    Raises:
        ValueError: If a tag is malformed or there are too many.
    """
    unique = tuple(dict.fromkeys(tags))
    bad = [tag for tag in unique if not _TAG_RE.match(tag)]
    if bad:
        raise ValueError(
            f"Invalid tag(s): {', '.join(bad)}. Tags start with a letter or digit and use "
            "only letters, digits and . _ : / + - (at most 64 characters)."
        )
    if len(unique) > MAX_TAGS:
        raise ValueError(f"At most {MAX_TAGS} tags are allowed, got {len(unique)}.")
    return unique


class InvalidApiUrl(ValueError):
    """The ingest service URL is malformed or not https."""


def validate_api_url(url: str) -> str:
    """Check that the ingest service URL is safe to send a Google access token to.

    The client sends a cloud-platform-scoped access token with every request, so the
    URL must use https. Plain http is allowed only for localhost (local development).

    Returns:
        The URL without a trailing slash.

    Raises:
        InvalidApiUrl: If the URL is malformed or uses http for a non-local host.
    """
    url = url.strip().rstrip("/")
    try:
        parts = urlsplit(url)
        host = parts.hostname
    except ValueError as e:
        raise InvalidApiUrl(f"Invalid ingest service URL {url!r}: {e}") from None
    if not host:
        raise InvalidApiUrl(f"Invalid ingest service URL {url!r}: expected https://<host>")
    if parts.scheme == "https":
        return url
    if parts.scheme == "http" and host in _LOCAL_HOSTS:
        return url
    raise InvalidApiUrl(
        f"Ingest service URL {url!r} must use https (plain http is allowed only for "
        f"localhost, 127.0.0.1 and ::1). Check --api-url or ${API_URL_ENV}."
    )


def upload_param_hint(error: ValueError) -> str:
    """The CLI option to blame for an error from ``resolve_upload_config``."""
    return "--api-url" if isinstance(error, InvalidApiUrl) else "--tag"


def resolve_upload_config(
    cli_upload: bool | None = None,
    cli_api_url: str | None = None,
    cli_tags: Sequence[str] = (),
) -> UploadConfig:
    """Combine CLI flags and environment variables. CLI flags win.

    Uploads are on by default. When neither a flag nor the environment asks for them and
    no Google credentials are configured, they are turned off and ``notice`` says why, so
    the eval still runs and keeps its results locally.

    Args:
        cli_upload: Value of --upload/--no-upload, or None when not passed.
        cli_api_url: Value of --api-url, or None when not passed.
        cli_tags: Values of --tag.

    Raises:
        InvalidApiUrl: If uploads are on and the API URL is not https (or local http).
        ValueError: If a tag is malformed.
    """
    enabled = cli_upload if cli_upload is not None else env_upload_enabled()
    api_url = cli_api_url or os.environ.get(API_URL_ENV) or DEFAULT_API_URL
    timeout_s = DEFAULT_TIMEOUT_S
    raw_timeout = os.environ.get(TIMEOUT_ENV)
    if raw_timeout:
        try:
            timeout_s = max(1.0, float(raw_timeout))
        except ValueError:
            timeout_s = DEFAULT_TIMEOUT_S
    notice = None
    if enabled is None:
        from olmo_eval.upload.auth import has_upload_credentials

        enabled = has_upload_credentials()
        notice = None if enabled else LOCAL_ONLY_NOTICE
    # Only an enabled upload sends a token, so a disabled one does not fail on the URL.
    api_url = validate_api_url(api_url) if enabled else api_url.rstrip("/")
    return UploadConfig(
        enabled=enabled,
        api_url=api_url,
        tags=validate_tags(cli_tags),
        timeout_s=timeout_s,
        notice=notice,
    )


def retry_command(output_dir: str, config: UploadConfig) -> str:
    """The command that re-uploads a results directory."""
    parts = ["olmo-eval", "results", "upload", output_dir]
    if not config.is_default_api_url:
        parts += ["--api-url", config.api_url]
    for tag in config.tags:
        parts += ["--tag", tag]
    return shlex.join(parts)
