"""Upload mapping for run timing, per-task cost and per-instance prompt tokens."""

from __future__ import annotations

import json
from pathlib import Path

from olmo_eval.upload.extract import iter_instances
from olmo_eval.upload.uploader import build_plan, dry_run
from tests.upload.fixtures import ARC_EASY, make_arc_run, mc_predictions, write_jsonl

RUN_KEYS = ("startup_seconds", "processing_started_at", "processing_seconds")
TASK_KEYS = (
    "first_request_at",
    "last_completed_at",
    "prompt_tokens_total",
    "completion_tokens_total",
    "attributed_inference_seconds",
)


def _add_cost(out: Path) -> None:
    path = out / "metrics.json"
    metrics = json.loads(path.read_text())
    metrics.update(
        startup_seconds=95.5,
        processing_started_at="2026-10-01T09:59:00+00:00",
        processing_seconds=21.0,
    )
    for row in metrics["tasks"]:
        row.update(
            first_request_at="2026-10-01T09:59:00.500000+00:00",
            last_completed_at="2026-10-01T09:59:17+00:00",
            prompt_tokens_total=4321,
            completion_tokens_total=0,
            attributed_inference_seconds=6.75,
        )
    path.write_text(json.dumps(metrics))


def test_new_fields_map_into_the_payload(tmp_path: Path) -> None:
    out = make_arc_run(tmp_path / "run")
    _add_cost(out)

    report = dry_run(out)

    assert report.errors == []
    run = report.plan.run_request["run"]
    assert {key: run[key] for key in RUN_KEYS} == {
        "startup_seconds": 95.5,
        "processing_started_at": "2026-10-01T09:59:00+00:00",
        "processing_seconds": 21.0,
    }
    payload = next(t.payload for t in report.plan.tasks if t.payload["task_name"] == ARC_EASY)
    assert {key: payload[key] for key in TASK_KEYS} == {
        "first_request_at": "2026-10-01T09:59:00.500000+00:00",
        "last_completed_at": "2026-10-01T09:59:17+00:00",
        "prompt_tokens_total": 4321,
        "completion_tokens_total": 0,
        "attributed_inference_seconds": 6.75,
    }


def test_old_results_without_new_fields_still_upload(tmp_path: Path) -> None:
    out = make_arc_run(tmp_path / "run")

    report = dry_run(out)

    assert report.errors == []
    run = report.plan.run_request["run"]
    assert all(run[key] is None for key in RUN_KEYS)
    for task in report.plan.tasks:
        assert all(task.payload[key] is None for key in TASK_KEYS)


def test_degraded_upload_without_manifest_maps_timing(tmp_path: Path) -> None:
    out = make_arc_run(tmp_path / "run", with_manifest=False)
    _add_cost(out)

    plan = build_plan(out)

    assert plan.run_request["run"]["startup_seconds"] == 95.5
    assert all(t.payload["prompt_tokens_total"] == 4321 for t in plan.tasks)


def test_malformed_values_become_null(tmp_path: Path) -> None:
    out = make_arc_run(tmp_path / "run")
    path = out / "metrics.json"
    metrics = json.loads(path.read_text())
    metrics.update(startup_seconds="soon", processing_started_at="not a date")
    metrics["tasks"][0].update(prompt_tokens_total=-3, attributed_inference_seconds=float("inf"))
    path.write_text(json.dumps(metrics))

    report = dry_run(out)

    assert report.errors == []
    run = report.plan.run_request["run"]
    assert run["startup_seconds"] is None and run["processing_started_at"] is None
    first = report.plan.tasks[0].payload
    assert first["prompt_tokens_total"] is None
    assert first["attributed_inference_seconds"] is None


def _rows(tmp_path: Path, records: list[dict]) -> list[dict]:
    path = tmp_path / "p.jsonl"
    write_jsonl(path, records)
    return list(iter_instances(path, None, None))


def test_loglikelihood_instance_prompt_tokens_use_the_first_choice(tmp_path: Path) -> None:
    records = mc_predictions(2)
    for i, rec in enumerate(records):
        for c, output in enumerate(rec["model_output"]):
            output["num_tokens_all"] = 40 + i + c
    rows = _rows(tmp_path, records)
    assert [r["prompt_tokens"] for r in rows] == [40, 41]


def test_loglikelihood_without_context_count_is_null(tmp_path: Path) -> None:
    records = mc_predictions(1)
    for output in records[0]["model_output"]:
        output["num_tokens_all"] = output["num_tokens"]
    assert _rows(tmp_path, records)[0]["prompt_tokens"] is None
    # Older predictions without num_tokens_all
    assert _rows(tmp_path, mc_predictions(1))[0]["prompt_tokens"] is None


def test_generation_instance_prompt_tokens(tmp_path: Path) -> None:
    with_usage = {
        "doc_id": 0,
        "model_output": [
            {"text": "x", "finish_reason": "stop", "completion_tokens": 5, "prompt_tokens": 77}
        ],
    }
    without_usage = {
        "doc_id": 1,
        "model_output": [{"text": "y", "finish_reason": "stop", "completion_tokens": 5}],
    }
    multi_turn = {
        "doc_id": 2,
        "model_output": [{"text": "z", "finish_reason": "stop", "prompt_tokens": 900}],
        "trajectory": {"turns": [{"role": "assistant"}, {"role": "user"}, {"role": "assistant"}]},
    }
    rows = _rows(tmp_path, [with_usage, without_usage, multi_turn])
    assert [r["prompt_tokens"] for r in rows] == [77, None, None]
