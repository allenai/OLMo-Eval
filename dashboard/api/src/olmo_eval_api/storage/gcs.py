"""GCS storage with V4 signed URLs.

The runtime service account has no key file, so URLs are signed through the IAM signBlob API
(``service_account_email`` + ``access_token``). The service account needs
``iam.serviceAccounts.signBlob`` on itself.
"""

from __future__ import annotations

import asyncio
import re
import threading
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from typing import Any

import google.auth
from google.api_core import exceptions as gexc
from google.auth.transport.requests import Request as AuthRequest
from google.cloud import storage

from olmo_eval_api.storage.base import LENGTH_RANGE_HEADER, ObjectInfo, SignedUrl, upload_headers

UPLOAD_TTL = timedelta(hours=1)
DOWNLOAD_TTL = timedelta(minutes=15)
STAT_CONCURRENCY = 16
_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
_UNSAFE_FILENAME = re.compile(r"[^A-Za-z0-9._+=@-]")


def attachment_disposition(name: str) -> str:
    """Content-Disposition for a download; characters that could break the header become _."""
    safe = _UNSAFE_FILENAME.sub("_", name)[:200] or "download"
    return f'attachment; filename="{safe}"'


class GcsStorage:
    def __init__(self, bucket: str, signer_email: str, client: Any = None) -> None:
        self.bucket = bucket
        self.signer_email = signer_email
        self._client = client
        self._credentials: Any = None
        self._lock = threading.Lock()

    def _gcs(self) -> Any:
        if self._client is None:
            self._client = storage.Client()
        return self._client

    def _access_token(self) -> str:
        with self._lock:
            if self._credentials is None:
                self._credentials, _ = google.auth.default(scopes=_SCOPES)
            creds = self._credentials
            expiry = getattr(creds, "expiry", None)
            near_expiry = expiry is not None and expiry - datetime.now(UTC).replace(
                tzinfo=None
            ) < timedelta(minutes=5)
            if not creds.token or not creds.valid or near_expiry:
                creds.refresh(AuthRequest())
            return creds.token

    def _sign(self, key: str, method: str, ttl: timedelta, **kwargs: Any) -> str:
        blob = self._gcs().bucket(self.bucket).blob(key)
        return blob.generate_signed_url(
            version="v4",
            method=method,
            expiration=ttl,
            service_account_email=self.signer_email,
            access_token=self._access_token(),
            **kwargs,
        )

    async def list_objects(self, prefix: str) -> dict[str, ObjectInfo]:
        def run() -> dict[str, ObjectInfo]:
            out = {}
            for blob in self._gcs().list_blobs(self.bucket, prefix=prefix):
                out[blob.name] = ObjectInfo(size=int(blob.size or 0), md5_b64=blob.md5_hash)
            return out

        return await asyncio.to_thread(run)

    async def stat_objects(self, keys: Iterable[str]) -> dict[str, ObjectInfo]:
        limiter = asyncio.Semaphore(STAT_CONCURRENCY)

        def stat(key: str) -> ObjectInfo | None:
            blob = self._gcs().bucket(self.bucket).get_blob(key)
            if blob is None:
                return None
            return ObjectInfo(size=int(blob.size or 0), md5_b64=blob.md5_hash)

        async def one(key: str) -> tuple[str, ObjectInfo | None]:
            async with limiter:
                return key, await asyncio.to_thread(stat, key)

        found = await asyncio.gather(*(one(k) for k in dict.fromkeys(keys)))
        return {k: info for k, info in found if info is not None}

    async def sign_upload(
        self, key: str, content_type: str, md5_b64: str, size_bytes: int
    ) -> SignedUrl:
        expires_at = datetime.now(UTC) + UPLOAD_TTL
        headers = upload_headers(content_type, md5_b64, size_bytes)
        url = await asyncio.to_thread(
            self._sign,
            key,
            "PUT",
            UPLOAD_TTL,
            content_type=content_type,
            content_md5=md5_b64,
            headers={LENGTH_RANGE_HEADER: headers[LENGTH_RANGE_HEADER]},
        )
        return SignedUrl(url=url, headers=headers, expires_at=expires_at)

    async def sign_download(self, key: str, filename: str | None = None) -> SignedUrl:
        expires_at = datetime.now(UTC) + DOWNLOAD_TTL
        name = filename or key.rsplit("/", 1)[-1]
        url = await asyncio.to_thread(
            self._sign,
            key,
            "GET",
            DOWNLOAD_TTL,
            response_disposition=attachment_disposition(name),
        )
        return SignedUrl(url=url, headers={}, expires_at=expires_at)

    async def read_range(self, key: str, start: int, length: int) -> bytes:
        if start < 0 or length <= 0:
            # end = start - 1 would make an invalid Range header, which can return everything.
            raise ValueError(f"invalid byte range start={start} length={length}")

        def run() -> bytes:
            blob = self._gcs().bucket(self.bucket).blob(key)
            try:
                return blob.download_as_bytes(start=start, end=start + length - 1)
            except gexc.NotFound as exc:
                raise FileNotFoundError(key) from exc

        return await asyncio.to_thread(run)

    async def delete_prefix(self, prefix: str) -> int:
        def run() -> int:
            client = self._gcs()
            blobs = list(client.list_blobs(self.bucket, prefix=prefix))
            for i in range(0, len(blobs), 100):
                with client.batch():
                    for blob in blobs[i : i + 100]:
                        blob.delete()
            return len(blobs)

        return await asyncio.to_thread(run)

    async def aclose(self) -> None:
        if self._client is not None:
            await asyncio.to_thread(self._client.close)
