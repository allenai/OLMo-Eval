"""Object routes that stand in for GCS signed URLs when STORAGE_BACKEND=local."""

from __future__ import annotations

from fastapi import APIRouter, Request, Response

from olmo_eval_api.errors import ApiError
from olmo_eval_api.storage.base import LENGTH_RANGE_HEADER
from olmo_eval_api.storage.local import LocalStorage, md5_b64

router = APIRouter(prefix="/_local/objects", include_in_schema=False)


def _storage(request: Request) -> LocalStorage:
    return request.app.state.storage


@router.put("/{key:path}")
async def put_object(key: str, request: Request) -> Response:
    data = await request.body()
    expected = request.headers.get("content-md5")
    if expected and expected != md5_b64(data):
        raise ApiError(400, "Content-MD5 does not match the body", code="bad_digest")
    # Like GCS: a signed upload names the exact size it accepts.
    length_range = request.headers.get(LENGTH_RANGE_HEADER)
    if length_range:
        low, _, high = length_range.partition(",")
        if not (low.isdigit() and high.isdigit() and int(low) <= len(data) <= int(high)):
            raise ApiError(400, "body size is outside x-goog-content-length-range")
    try:
        _storage(request).write(key, data)
    except ValueError as exc:
        raise ApiError(400, str(exc)) from exc
    return Response(status_code=200)


@router.get("/{key:path}")
async def get_object(key: str, request: Request) -> Response:
    try:
        data = _storage(request).read(key)
    except (FileNotFoundError, IsADirectoryError, ValueError) as exc:
        raise ApiError(404, f"object {key} not found") from exc
    return Response(content=data, media_type="application/octet-stream")
