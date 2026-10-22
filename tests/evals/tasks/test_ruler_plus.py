"""Tests for the ruler-plus loader, task and suites."""

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from huggingface_hub.errors import RepositoryNotFoundError

from olmo_eval.data import ruler_plus_loader
from olmo_eval.data.ruler_plus_loader import (
    ROW_FIELD,
    RULER_PLUS_DATA_DIR_ENV_VAR,
    RULER_PLUS_REPO_ID,
    RULER_PLUS_REVISION,
    get_ruler_plus_data_file,
    load_ruler_plus_shard,
)
from olmo_eval.data.ruler_plus_tasks import CONTEXT_SIZES, RULER_PLUS_TASKS
from olmo_eval.data.ruler_tasks import RULER_TASKS
from olmo_eval.evals.suites import get_suite
from olmo_eval.evals.tasks.common import get_task, list_tasks

SHARD = "4096/niah_multikey_3/validation.jsonl"


def _write_shard(path: Path, records: list[dict[str, Any]], blank_every: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for i, record in enumerate(records):
        if blank_every and i % blank_every == 0:
            lines.append("")
        lines.append(json.dumps(record))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _records(n: int) -> list[dict[str, Any]]:
    # Like the published NIAH shards, `index` repeats across distinct records
    return [
        {
            "index": 636,
            "input": f"prompt {i}",
            "answer_prefix": " Answer:",
            "outputs": [str(i)],
        }
        for i in range(n)
    ]


@pytest.fixture
def local_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    _write_shard(tmp_path / SHARD, _records(20), blank_every=4)
    monkeypatch.setenv(RULER_PLUS_DATA_DIR_ENV_VAR, str(tmp_path))
    return tmp_path


class TestLoadShard:
    def test_no_limit_returns_every_row_in_order(self, tmp_path: Path):
        path = tmp_path / "shard.jsonl"
        _write_shard(path, _records(10), blank_every=3)
        rows = load_ruler_plus_shard(str(path))
        assert [r["outputs"] for r in rows] == [[str(i)] for i in range(10)]
        assert [r[ROW_FIELD] for r in rows] == list(range(10))

    def test_limit_above_shard_size_returns_everything(self, tmp_path: Path):
        path = tmp_path / "shard.jsonl"
        _write_shard(path, _records(10))
        assert len(load_ruler_plus_shard(str(path), max_samples=50)) == 10

    def test_sample_is_capped_and_deterministic(self, tmp_path: Path):
        path = tmp_path / "shard.jsonl"
        _write_shard(path, _records(100), blank_every=7)
        first = load_ruler_plus_shard(str(path), max_samples=10, seed=3)
        assert len(first) == 10
        assert first == load_ruler_plus_shard(str(path), max_samples=10, seed=3)
        assert first != load_ruler_plus_shard(str(path), max_samples=10, seed=4)

    def test_row_is_shard_position_whatever_the_limit(self, tmp_path: Path):
        path = tmp_path / "shard.jsonl"
        _write_shard(path, _records(100), blank_every=5)
        for row in load_ruler_plus_shard(str(path), max_samples=10, seed=0):
            assert row["outputs"] == [str(row[ROW_FIELD])]


class TestDataFile:
    def test_downloads_pinned_revision(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv(RULER_PLUS_DATA_DIR_ENV_VAR, raising=False)
        calls: list[dict[str, Any]] = []

        def fake_download(**kwargs: Any) -> str:
            calls.append(kwargs)
            return "/cache/shard.jsonl"

        monkeypatch.setattr(ruler_plus_loader, "hf_hub_download", fake_download)
        data_file = get_ruler_plus_data_file(SHARD)

        assert calls == [
            {
                "repo_id": RULER_PLUS_REPO_ID,
                "repo_type": "dataset",
                "filename": f"data/{SHARD}",
                "revision": RULER_PLUS_REVISION,
            }
        ]
        assert data_file.path == "/cache/shard.jsonl"
        assert data_file.source == f"hf:{RULER_PLUS_REPO_ID}@{RULER_PLUS_REVISION}/data/{SHARD}"

    def test_private_repo_explains_access(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv(RULER_PLUS_DATA_DIR_ENV_VAR, raising=False)

        def fake_download(**kwargs: Any) -> str:
            request = httpx.Request("GET", "https://huggingface.co")
            raise RepositoryNotFoundError(
                "404 Client Error", response=httpx.Response(404, request=request)
            )

        monkeypatch.setattr(ruler_plus_loader, "hf_hub_download", fake_download)
        with pytest.raises(PermissionError, match="internal-only"):
            get_ruler_plus_data_file(SHARD)

    def test_env_override_reads_local_file(self, local_data: Path):
        data_file = get_ruler_plus_data_file(SHARD)
        assert data_file.path == str(local_data / SHARD)
        assert data_file.source == f"local:{local_data / SHARD}"

    def test_env_override_missing_file(self, local_data: Path):
        with pytest.raises(FileNotFoundError, match=RULER_PLUS_DATA_DIR_ENV_VAR):
            get_ruler_plus_data_file("4096/missing/validation.jsonl")


class TestTask:
    def test_ids_are_unique_when_index_repeats(self, local_data: Path):
        instances = list(get_task("ruler_plus_niah_mk_3__4096").instances)
        ids = [i.metadata["id"] for i in instances]
        assert len(ids) == 20
        assert sorted(ids) == list(range(20))

    def test_id_is_stable_across_limits(self, local_data: Path):
        full = {
            i.metadata["id"]: i.gold_answer
            for i in get_task("ruler_plus_niah_mk_3__4096").instances
        }
        limited = get_task("ruler_plus_niah_mk_3__4096", config_overrides={"limit": 5})
        sampled = list(limited.instances)
        assert len(sampled) == 5
        for instance in sampled:
            assert full[instance.metadata["id"]] == instance.gold_answer

    def test_prompt_and_metadata(self, local_data: Path):
        instance = next(iter(get_task("ruler_plus_niah_mk_3__4096").instances))
        row = instance.metadata["id"]
        assert instance.question == f"prompt {row} Answer:"
        assert instance.gold_answer == [str(row)]
        assert instance.metadata["all_gold_answers"] == [str(row)]
        assert instance.metadata["data_source"] == f"local:{local_data / SHARD}"

    def test_record_without_input_is_skipped(self):
        task = get_task("ruler_plus_niah_mk_3__4096")
        assert task.process_doc({"outputs": ["1"], ROW_FIELD: 0}) is None


def test_every_task_and_suite_is_registered():
    registered = set(list_tasks())
    assert {f"ruler_plus_{name}" for name in RULER_PLUS_TASKS} <= registered
    assert len(RULER_PLUS_TASKS) == 13 * len(CONTEXT_SIZES)
    # the prefixes nest, so check no ruler-plus name doubles as a RULER one
    assert not {f"ruler_plus_{name}" for name in RULER_PLUS_TASKS} & {
        f"ruler_{name}" for name in RULER_TASKS
    }
    for size in CONTEXT_SIZES:
        assert len(get_suite(f"ruler_plus_all__{size}").tasks) == 13
