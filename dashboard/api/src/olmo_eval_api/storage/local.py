"""Filesystem storage for local development and tests (SKIFF_ENV=local).

"Signed" URLs point at the API's own ``/_local/objects/{key}`` route.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import shutil
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import quote

from olmo_eval_api.storage.base import ObjectInfo, SignedUrl, upload_headers


def md5_b64(data: bytes) -> str:
    return base64.b64encode(hashlib.md5(data).digest()).decode()


class LocalStorage:
    def __init__(self, root: str | Path, public_base_url: str, bucket: str = "local") -> None:
        self.root = Path(root)
        self.public_base_url = public_base_url.rstrip("/")
        self.bucket = bucket

    def path_for(self, key: str) -> Path:
        path = (self.root / key).resolve()
        root = self.root.resolve()
        if root != path and root not in path.parents:
            raise ValueError(f"key escapes the storage root: {key}")
        return path

    def object_url(self, key: str) -> str:
        return f"{self.public_base_url}/_local/objects/{quote(key)}"

    async def list_objects(self, prefix: str) -> dict[str, ObjectInfo]:
        def run() -> dict[str, ObjectInfo]:
            out: dict[str, ObjectInfo] = {}
            base = self.root
            if not base.exists():
                return out
            for path in base.rglob("*"):
                if path.is_file():
                    key = path.relative_to(base).as_posix()
                    if key.startswith(prefix):
                        data = path.read_bytes()
                        out[key] = ObjectInfo(size=len(data), md5_b64=md5_b64(data))
            return out

        return await asyncio.to_thread(run)

    async def stat_objects(self, keys: Iterable[str]) -> dict[str, ObjectInfo]:
        def run() -> dict[str, ObjectInfo]:
            out: dict[str, ObjectInfo] = {}
            for key in keys:
                path = self.path_for(key)
                if path.is_file():
                    data = path.read_bytes()
                    out[key] = ObjectInfo(size=len(data), md5_b64=md5_b64(data))
            return out

        return await asyncio.to_thread(run)

    async def sign_upload(
        self, key: str, content_type: str, md5_b64: str, size_bytes: int
    ) -> SignedUrl:
        headers = upload_headers(content_type, md5_b64, size_bytes)
        return SignedUrl(
            url=self.object_url(key),
            headers=headers,
            expires_at=datetime.now(UTC) + timedelta(hours=1),
        )

    async def sign_download(self, key: str, filename: str | None = None) -> SignedUrl:
        return SignedUrl(
            url=self.object_url(key),
            headers={},
            expires_at=datetime.now(UTC) + timedelta(minutes=15),
        )

    def write(self, key: str, data: bytes) -> None:
        path = self.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def read(self, key: str) -> bytes:
        return self.path_for(key).read_bytes()

    async def read_range(self, key: str, start: int, length: int) -> bytes:
        def run() -> bytes:
            path = self.path_for(key)
            if not path.is_file():
                raise FileNotFoundError(key)
            with path.open("rb") as f:
                f.seek(start)
                return f.read(length)

        return await asyncio.to_thread(run)

    async def delete_prefix(self, prefix: str) -> int:
        def run() -> int:
            target = self.path_for(prefix.rstrip("/"))
            if not target.exists():
                return 0
            count = sum(1 for p in target.rglob("*") if p.is_file())
            shutil.rmtree(target)
            return count

        return await asyncio.to_thread(run)

    async def aclose(self) -> None:
        return None
