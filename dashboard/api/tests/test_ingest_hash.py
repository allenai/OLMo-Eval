"""Compatibility of task-result upload identity."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from olmo_eval_api.schemas import ingest as s
from olmo_eval_api.services.ingest import _content_hash
from tests.helpers import load_example

LEGACY_HASH = "3c39b95e373799588fefbcff9c621445d645f2bed5415451ffc247e6fca76e5e"


@pytest.mark.parametrize("explicit_null", [False, True])
async def test_absent_diagnostics_preserve_the_legacy_content_hash(explicit_null: bool) -> None:
    payload = load_example("ingest/task-result.request.json")
    payload.update(predictions_path=None, requests_path=None)
    if explicit_null:
        payload["generation_counts"] = None
        payload["scoring_seconds"] = None
    request = s.TaskResultIn.model_validate(payload)

    assert await _content_hash(AsyncMock(), "legacy", request) == LEGACY_HASH


async def test_changed_diagnostics_change_upload_identity() -> None:
    payload = load_example("ingest/task-result.request.json")
    payload.update(predictions_path=None, requests_path=None)
    payload["generation_counts"] = {"generations": 3, "cap_hit": 1}
    request = s.TaskResultIn.model_validate(payload)
    first = await _content_hash(AsyncMock(), "legacy", request)
    request.generation_counts = {"generations": 3, "cap_hit": 2}
    second = await _content_hash(AsyncMock(), "legacy", request)

    assert first != LEGACY_HASH and second != first


async def test_scoring_seconds_change_upload_identity() -> None:
    payload = load_example("ingest/task-result.request.json")
    payload.update(predictions_path=None, requests_path=None, scoring_seconds=3.0)
    request = s.TaskResultIn.model_validate(payload)
    first = await _content_hash(AsyncMock(), "legacy", request)
    request.scoring_seconds = 4.0

    assert first != LEGACY_HASH
    assert await _content_hash(AsyncMock(), "legacy", request) != first
