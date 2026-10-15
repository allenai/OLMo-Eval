"""Storage protocol for run artifacts."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol


@dataclass(frozen=True)
class ObjectInfo:
    size: int
    md5_b64: str | None


# GCS rejects a signed PUT whose body size is outside this header's "min,max" range. The
# header is part of the signature, so the uploader cannot drop or change it.
LENGTH_RANGE_HEADER = "x-goog-content-length-range"


def upload_headers(content_type: str, md5_b64: str, size_bytes: int) -> dict[str, str]:
    return {
        "Content-Type": content_type,
        "Content-MD5": md5_b64,
        LENGTH_RANGE_HEADER: f"{size_bytes},{size_bytes}",
    }


@dataclass(frozen=True)
class SignedUrl:
    url: str
    headers: dict[str, str]
    expires_at: datetime


class Storage(Protocol):
    bucket: str

    async def list_objects(self, prefix: str) -> dict[str, ObjectInfo]:
        """Map of object key -> info for every object under ``prefix``."""
        ...

    async def stat_objects(self, keys: Iterable[str]) -> dict[str, ObjectInfo]:
        """Info for each of ``keys`` that exists."""
        ...

    async def sign_upload(
        self, key: str, content_type: str, md5_b64: str, size_bytes: int
    ) -> SignedUrl:
        """A PUT URL that accepts only a body of exactly ``size_bytes`` with that MD5.

        The returned headers must be sent with the PUT.
        """
        ...

    async def sign_download(self, key: str, filename: str | None = None) -> SignedUrl: ...

    async def read_range(self, key: str, start: int, length: int) -> bytes:
        """Read ``length`` bytes starting at ``start``. Raises FileNotFoundError."""
        ...

    async def delete_prefix(self, prefix: str) -> int: ...

    async def aclose(self) -> None: ...
