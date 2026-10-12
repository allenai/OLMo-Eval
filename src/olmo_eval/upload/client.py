"""HTTP client for the ingest service (protocol version 1)."""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import httpx

from olmo_eval.common.logging import get_logger

logger = get_logger("upload.client")

TOKEN_HEADER = "X-Olmo-Eval-Token"
RUN_SECRET_HEADER = "X-Olmo-Eval-Run-Secret"
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
BACKOFF_SECONDS = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
CONNECT_TIMEOUT_S = 10.0
READ_TIMEOUT_S = 120.0


class TokenProvider(Protocol):
    def token(self) -> str: ...


class IngestError(Exception):
    """A request to the ingest service or a signed upload failed."""

    def __init__(self, message: str, status: int | None = None, code: str | None = None):
        super().__init__(message)
        self.status = status
        self.code = code


class SignedUrlRejected(IngestError):
    """GCS rejected a signed upload with 400 or 403, usually because the URL expired."""


class DeadlineExceeded(IngestError):
    """The overall upload deadline passed."""


def _client_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("olmo-eval")
    except PackageNotFoundError:
        return "unknown"


def _error_message(response: httpx.Response) -> tuple[str, str | None]:
    try:
        body = response.json()
    except ValueError:
        text = response.text.strip()[:500]
        return (text or response.reason_phrase or f"HTTP {response.status_code}"), None
    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, dict):
        message = str(error.get("message") or response.reason_phrase)
        request_id = error.get("request_id")
        if request_id:
            message = f"{message} (request_id={request_id})"
        return message, error.get("code")
    return str(body)[:500], None


class IngestClient:
    """One method per ingest endpoint, with retries for transient failures.

    Connection errors, timeouts and HTTP 429/500/502/503/504 are retried with
    exponential backoff (1, 2, 4, 8, 16, 32 s plus up to 20% jitter). Other 4xx
    responses fail immediately with the server's error message.

    Args:
        api_url: Base URL of the ingest service.
        token_provider: Source of Google access tokens.
        deadline_s: Seconds from now after which requests stop being attempted.
        transport: Optional httpx transport (tests use httpx.MockTransport).
        sleep: Sleep function used between retries.
        max_retries: Retries after the first attempt.
        run_secret: The run's write secret, sent with every ingest request. It lets the
            shared Beaker service account change a run it uploaded earlier.
    """

    def __init__(
        self,
        api_url: str,
        token_provider: TokenProvider,
        *,
        deadline_s: float | None = None,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        max_retries: int = len(BACKOFF_SECONDS),
        run_secret: str | None = None,
    ) -> None:
        from olmo_eval.upload.config import validate_api_url

        self.api_url = validate_api_url(api_url)
        self._tokens = token_provider
        self._deadline = time.monotonic() + deadline_s if deadline_s is not None else None
        self._sleep = sleep
        self._max_retries = max_retries
        self.run_secret = run_secret
        self._http = httpx.Client(
            transport=transport,
            timeout=httpx.Timeout(READ_TIMEOUT_S, connect=CONNECT_TIMEOUT_S),
            headers={"User-Agent": f"olmo-eval/{_client_version()}"},
            follow_redirects=False,
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> IngestClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ endpoints

    def whoami(self) -> dict[str, Any]:
        return self._json("GET", "/v1/whoami")

    def upsert_run(self, run_id: str, body: dict[str, Any]) -> dict[str, Any]:
        return self._json("PUT", f"/v1/runs/{run_id}", body)

    def sign_artifacts(self, run_id: str, body: dict[str, Any]) -> dict[str, Any]:
        return self._json("POST", f"/v1/runs/{run_id}/artifacts:sign", body)

    def upsert_task_result(self, run_id: str, body: dict[str, Any]) -> dict[str, Any]:
        return self._json("POST", f"/v1/runs/{run_id}/task-results", body)

    def post_instances(self, task_result_id: int, body: dict[str, Any]) -> dict[str, Any]:
        return self._json("POST", f"/v1/task-results/{task_result_id}/instances", body)

    def put_inference(self, run_id: str, body: dict[str, Any]) -> dict[str, Any]:
        return self._json("PUT", f"/v1/runs/{run_id}/inference", body)

    def complete(self, run_id: str, body: dict[str, Any]) -> dict[str, Any]:
        return self._json("POST", f"/v1/runs/{run_id}/complete", body)

    def put_signed(
        self,
        url: str,
        headers: dict[str, str],
        file_path: Path,
        expected_size: int | None = None,
    ) -> None:
        """Upload a file to a V4 signed URL, sending exactly the signed headers.

        The body is the file, sent with an explicit Content-Length equal to its size
        (never chunked), since the signature may cover the exact size.

        Args:
            url: Signed URL.
            headers: Headers the URL was signed with.
            file_path: File to upload.
            expected_size: Size the URL was signed for. The upload fails without being
                sent if the file has changed size since.

        Raises:
            SignedUrlRejected: On HTTP 400/403 (the caller re-signs once).
            IngestError: On any other failure after retries.
        """
        size = file_path.stat().st_size
        if expected_size is not None and size != expected_size:
            raise IngestError(
                f"{file_path.name} changed size since it was signed "
                f"({expected_size} -> {size} bytes); not uploading it"
            )
        read_timeout = max(READ_TIMEOUT_S, size / 1_000_000 * 2)
        send_headers = {k: v for k, v in headers.items() if k.lower() != "content-length"}
        send_headers["Content-Length"] = str(size)

        def send() -> httpx.Response:
            with file_path.open("rb") as f:
                return self._http.request(
                    "PUT",
                    url,
                    content=f,
                    headers=send_headers,
                    timeout=self._timeout(read_timeout),
                )

        response = self._with_retries(send, f"upload {file_path.name}")
        if response.status_code in (400, 403):
            raise SignedUrlRejected(
                f"GCS rejected the upload of {file_path.name} "
                f"(HTTP {response.status_code}): {response.text.strip()[:300]}",
                status=response.status_code,
            )
        if response.status_code >= 300:
            raise IngestError(
                f"Upload of {file_path.name} failed (HTTP {response.status_code}): "
                f"{response.text.strip()[:300]}",
                status=response.status_code,
            )

    # ------------------------------------------------------------------ internals

    def _remaining(self) -> float | None:
        if self._deadline is None:
            return None
        return self._deadline - time.monotonic()

    def _timeout(self, read_timeout: float) -> httpx.Timeout:
        remaining = self._remaining()
        if remaining is not None:
            read_timeout = max(1.0, min(read_timeout, remaining))
        return httpx.Timeout(read_timeout, connect=min(CONNECT_TIMEOUT_S, read_timeout))

    def _json(self, method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        def send() -> httpx.Response:
            headers = {TOKEN_HEADER: self._tokens.token()}
            if self.run_secret:
                headers[RUN_SECRET_HEADER] = self.run_secret
            return self._http.request(
                method,
                f"{self.api_url}{path}",
                json=body,
                headers=headers,
                timeout=self._timeout(READ_TIMEOUT_S),
            )

        response = self._with_retries(send, f"{method} {path}")
        if response.status_code >= 400:
            message, code = _error_message(response)
            raise IngestError(
                f"{method} {path} failed (HTTP {response.status_code}): {message}",
                status=response.status_code,
                code=code,
            )
        if response.status_code == 204 or not response.content:
            return {}
        try:
            data = response.json()
        except ValueError as e:
            raise IngestError(f"{method} {path} returned invalid JSON") from e
        return data if isinstance(data, dict) else {}

    def _with_retries(self, send: Callable[[], httpx.Response], label: str) -> httpx.Response:
        attempt = 0
        while True:
            remaining = self._remaining()
            if remaining is not None and remaining <= 0:
                raise DeadlineExceeded(f"Upload deadline exceeded before {label}")
            error: str
            try:
                response = send()
            except httpx.TransportError as e:
                error = f"{type(e).__name__}: {e}"
            else:
                if response.status_code not in RETRY_STATUSES:
                    return response
                if attempt >= self._max_retries:
                    return response
                error = f"HTTP {response.status_code}"
            if attempt >= self._max_retries:
                raise IngestError(f"{label} failed after {attempt + 1} attempts: {error}")
            delay = BACKOFF_SECONDS[min(attempt, len(BACKOFF_SECONDS) - 1)]
            delay *= 1 + random.uniform(0, 0.2)
            remaining = self._remaining()
            if remaining is not None and delay >= remaining:
                raise DeadlineExceeded(f"Upload deadline exceeded while retrying {label}: {error}")
            logger.warning(f"{label}: {error}; retrying in {delay:.1f}s")
            self._sleep(delay)
            attempt += 1
