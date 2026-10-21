"""JSONL writers: lazy image loaders in request records, strictness elsewhere."""

from __future__ import annotations

import functools
import json
from pathlib import Path

import pytest

from olmo_eval.runners.io.writers import write_predictions_jsonl, write_requests_jsonl


def _load_image(path: str) -> None:
    return None


def _written(tmp_path: Path) -> list[dict]:
    (path,) = tmp_path.rglob("*.jsonl")
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_request_records_keep_a_placeholder_for_lazy_loaders(tmp_path: Path) -> None:
    loader = functools.partial(_load_image, "cat.png")
    write_requests_jsonl(str(tmp_path), "task", [{"doc": {"images": loader, "q": "?"}}], "m")
    assert _written(tmp_path) == [{"doc": {"images": "<partial>", "q": "?"}}]


def test_request_records_still_reject_other_non_json_values(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="set"):
        write_requests_jsonl(str(tmp_path), "task", [{"doc": {"tags": {"a"}}}], "m")


def test_prediction_records_stay_strict(tmp_path: Path) -> None:
    loader = functools.partial(_load_image, "cat.png")
    with pytest.raises(TypeError):
        write_predictions_jsonl(str(tmp_path), "task", [{"metrics": loader}], "m")
