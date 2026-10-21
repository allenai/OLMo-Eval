"""Signed artifact uploads and contract validation fallbacks."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from olmo_eval.upload.client import IngestClient, IngestError
from olmo_eval.upload.validation import PayloadValidator
from tests.upload.fixtures import CONTRACT_SCHEMA

SIGNED_URL = "https://storage.googleapis.com/bucket/object?X-Goog-Signature=abc"


def _client(seen: list[httpx.Request]) -> IngestClient:
    def handler(request: httpx.Request) -> httpx.Response:
        request.read()
        seen.append(request)
        return httpx.Response(200)

    return IngestClient(
        "https://ingest.example", token_provider=None, transport=httpx.MockTransport(handler)
    )  # type: ignore[arg-type]


def test_signed_put_sends_exact_content_length(tmp_path: Path) -> None:
    path = tmp_path / "predictions.jsonl"
    path.write_bytes(b"x" * 12345)
    seen: list[httpx.Request] = []

    _client(seen).put_signed(
        SIGNED_URL,
        {"Content-Type": "application/jsonl", "content-length": "1"},
        path,
        expected_size=12345,
    )

    request = seen[0]
    assert request.headers.get_list("content-length") == ["12345"]
    assert "transfer-encoding" not in request.headers
    assert request.headers["content-type"] == "application/jsonl"
    assert len(request.content) == 12345


def test_signed_put_refuses_a_file_that_changed_size(tmp_path: Path) -> None:
    path = tmp_path / "predictions.jsonl"
    path.write_bytes(b"x" * 10)
    seen: list[httpx.Request] = []

    with pytest.raises(IngestError, match="changed size"):
        _client(seen).put_signed(SIGNED_URL, {}, path, expected_size=9)
    assert seen == []


def test_missing_schema_is_reported_and_logged_once() -> None:
    from olmo_eval.upload import validation

    validation._warn_degraded.cache_clear()
    with (
        patch.object(validation, "find_contract_schema", return_value=None),
        patch.object(validation.logger, "warning") as warning,
    ):
        first = PayloadValidator()
        PayloadValidator()

    assert first.degraded_reason is not None
    assert "not found" in first.degraded_reason
    assert not first.uses_schema
    warning.assert_called_once()
    assert first.errors("CompleteRequest", {}) != []


def test_repository_schema_is_not_degraded() -> None:
    assert PayloadValidator(CONTRACT_SCHEMA).degraded_reason is None
