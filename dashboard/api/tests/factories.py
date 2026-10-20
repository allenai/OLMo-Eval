"""Seed runs through the real ingest endpoints."""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

import httpx

from tests.helpers import load_example


def _check(response: httpx.Response) -> dict[str, Any]:
    assert response.status_code < 300, f"{response.request.url}: {response.text}"
    return response.json() if response.content else {}


def run_payload(
    *,
    run_id: str,
    model_name: str,
    model_path: str | None = None,
    model_hash: str | None = None,
    step: int | None = None,
    group: str | None = "test-group",
    author: str = "tester",
    status: str = "complete",
    tags: Sequence[str] = (),
    task_specs: Sequence[str] = (),
    launch_id: str | None = None,
    provider_config: Mapping[str, Any] | None = None,
    started_at: datetime | None = None,
) -> dict[str, Any]:
    base = load_example("ingest/run-upsert-final.request.json")
    payload = copy.deepcopy(base)
    run = payload["run"]
    run.update(
        run_id=run_id,
        launch_id=launch_id,
        experiment_name=f"exp-{run_id}",
        experiment_group=group,
        status=status,
        author=author,
        tags=list(tags),
        task_specs=list(task_specs),
        started_at=(started_at or datetime.now(UTC)).isoformat(),
    )
    run["beaker"]["experiment_id"] = f"01EXP{run_id.upper()}"
    config = dict(provider_config or {"kind": "vllm", "model": model_path or model_name})
    payload["model"] = {
        "name": model_name,
        "path": model_path or model_name,
        "model_hash": model_hash or hashlib.sha256(model_name.encode()).hexdigest()[:16],
        "revision": None,
        "provider_kind": "vllm",
        "provider_config": config,
        "step": step,
    }
    return payload


def task_payload(
    task_name: str,
    *,
    task_hash: str,
    primary_metric: str,
    corpus_score: float | None,
    n: int,
    error: str | None = None,
    suites: Sequence[str] = (),
    higher_is_better: bool | None = True,
    display_format: str | None = "percent",
    limit: int | None = None,
) -> dict[str, Any]:
    metric, scorer = primary_metric.rsplit(":", 1)
    return {
        "task_name": task_name,
        "task_hash": task_hash,
        "base_task": task_name.split(":", 1)[0],
        "primary_metric": primary_metric,
        "metrics": {metric: {scorer: corpus_score}},
        "metric_meta": {
            metric: {
                "higher_is_better": higher_is_better,
                "display_format": display_format,
                "unit": None,
            }
        },
        "config": {
            "name": task_name,
            "limit": limit,
            "sampling_params": {"max_tokens": 256},
        },
        "num_fewshot": 0,
        "limit": limit,
        "split": "test",
        "num_instances": n,
        "instances_processed": n,
        "instances_failed": 0,
        "error": error,
        "error_summary": None,
        "duration_seconds": 12.5,
        "predictions_path": f"predictions/{task_name.replace(':', '_')}-predictions.jsonl",
        "requests_path": None,
        "suites": list(suites),
        "instance_count": n,
    }


def instance_rows(scores: Sequence[float | None], primary_metric: str) -> list[dict[str, Any]]:
    rows = []
    for i, score in enumerate(scores):
        rows.append(
            {
                "native_id": f"q{i}",
                "doc_id": i,
                "primary_score": score,
                "metrics": {primary_metric: score} if score is not None else {},
                "label": f"label {i}",
                "finish_reason": "length" if i % 10 == 9 else "stop",
                "completion_tokens": 10 + i,
                "prompt_tokens": None,
                "num_outputs": 1,
                "extracted_answer": f"answer {i}",
                "prompt_preview": f"question {i}",
                "output_preview": f"output {i}",
                "judge_verdict": None,
                "has_scoring_error": False,
                "has_execution_result": False,
                "has_judge_result": False,
                "has_trajectory": False,
                "pred_offset": None,
                "pred_length": None,
                "req_offset": None,
                "req_length": None,
            }
        )
    return rows


async def seed_run(
    client: httpx.AsyncClient,
    *,
    run_id: str,
    model_name: str = "test-org/test-model",
    model_path: str | None = None,
    step: int | None = None,
    group: str | None = "test-group",
    author: str = "tester",
    status: str = "complete",
    tasks: Mapping[str, Sequence[float | None]],
    task_hashes: Mapping[str, str] | None = None,
    primary_metric: str = "acc:default",
    corpus_scores: Mapping[str, float] | None = None,
    errors: Mapping[str, str] | None = None,
    suites: Sequence[dict] = (),
    tags: Sequence[str] = (),
    created_at: datetime | None = None,
    launch_id: str | None = None,
    session: Any = None,
    started_at: datetime | None = None,
    task_limits: Mapping[str, int] | None = None,
) -> str:
    """Upload a synthetic run through the ingest endpoints and return the run_id.

    ``created_at`` needs ``session`` (an AsyncSession) to patch the timestamps after complete.
    """
    task_specs = [x["name"] for x in suites if not x.get("parent")] + list(tasks)
    body = run_payload(
        run_id=run_id,
        model_name=model_name,
        model_path=model_path,
        step=step,
        group=group,
        author=author,
        status="running",
        tags=tags,
        task_specs=task_specs,
        launch_id=launch_id,
        started_at=started_at,
    )
    _check(await client.put(f"/v1/runs/{run_id}", json=body))
    body["run"]["status"] = status
    _check(await client.put(f"/v1/runs/{run_id}", json=body))
    for task_name, scores in tasks.items():
        values = [v for v in scores if v is not None]
        default = sum(values) / len(values) if values else None
        corpus = corpus_scores.get(task_name, default) if corpus_scores else default
        task_hash = (task_hashes or {}).get(task_name) or hashlib.sha256(
            task_name.encode()
        ).hexdigest()[:16]
        tr = _check(
            await client.post(
                f"/v1/runs/{run_id}/task-results",
                json=task_payload(
                    task_name,
                    task_hash=task_hash,
                    primary_metric=primary_metric,
                    corpus_score=corpus,
                    n=len(scores),
                    error=(errors or {}).get(task_name),
                    limit=(task_limits or {}).get(task_name),
                ),
            )
        )
        rows = instance_rows(scores, primary_metric)
        for i in range(0, len(rows), 2000):
            _check(
                await client.post(
                    f"/v1/task-results/{tr['task_result_id']}/instances",
                    json={"batch_index": i // 2000, "instances": rows[i : i + 2000]},
                )
            )
    _check(
        await client.post(
            f"/v1/runs/{run_id}/complete",
            json={
                "status": status if status != "running" else "complete",
                "finished_at": None,
                "duration_seconds": 100.0,
                "suites": list(suites),
                "expected_task_results": len(tasks),
                "expected_artifacts": 0,
            },
        )
    )
    if status == "running":
        # Leave the run visible as running (complete forces a final status).
        from sqlalchemy import text

        assert session is not None, "status='running' needs session"
        await session.execute(
            text("UPDATE runs SET status = 'running' WHERE run_id = :r"), {"r": run_id}
        )
        await session.commit()
    if created_at is not None:
        from sqlalchemy import text

        assert session is not None, "created_at needs session"
        await session.execute(
            text("UPDATE runs SET created_at = :t WHERE run_id = :r"),
            {"t": created_at, "r": run_id},
        )
        await session.execute(
            text("UPDATE task_results SET run_created_at = :t WHERE run_id = :r"),
            {"t": created_at, "r": run_id},
        )
        await session.commit()
    return run_id


def suite(name: str, children: Sequence[str], score: float | None = None, **kw: Any) -> dict:
    return {
        "name": name,
        "aggregation": kw.get("aggregation", "average"),
        "parent": kw.get("parent"),
        "description": kw.get("description"),
        "children": [
            {
                "type": "suite" if c.startswith("suite:") else "task",
                "name": c.removeprefix("suite:"),
            }
            for c in children
        ],
        "metrics": {"primary_score": {"average": score}},
        "primary_metric": "primary_score:average",
        "score": score,
        "num_tasks": len(children),
    }


INGEST_ORDER = [
    "ingest/run-upsert-start.request.json",
    "ingest/run-upsert-final.request.json",
    "ingest/artifacts-sign.request.json",
    "ingest/task-result.request.json",
    "ingest/instances.request.json",
    "ingest/inference.request.json",
    "ingest/complete.request.json",
]


async def seed_contract_example_run(client: httpx.AsyncClient) -> str:
    """Send dashboard/contract/examples/ingest/* in protocol order and return the run_id.

    The artifact files are not in the repo, so nothing is PUT to the signed URLs and
    ``complete`` lists them in ``artifacts_missing``.
    """
    start = load_example("ingest/run-upsert-start.request.json")
    run_id = start["run"]["run_id"]
    _check(await client.put(f"/v1/runs/{run_id}", json=start))
    _check(
        await client.put(
            f"/v1/runs/{run_id}", json=load_example("ingest/run-upsert-final.request.json")
        )
    )
    _check(
        await client.post(
            f"/v1/runs/{run_id}/artifacts:sign",
            json=load_example("ingest/artifacts-sign.request.json"),
        )
    )
    tr = _check(
        await client.post(
            f"/v1/runs/{run_id}/task-results", json=load_example("ingest/task-result.request.json")
        )
    )
    _check(
        await client.post(
            f"/v1/task-results/{tr['task_result_id']}/instances",
            json=load_example("ingest/instances.request.json"),
        )
    )
    _check(
        await client.put(
            f"/v1/runs/{run_id}/inference", json=load_example("ingest/inference.request.json")
        )
    )
    _check(
        await client.post(
            f"/v1/runs/{run_id}/complete", json=load_example("ingest/complete.request.json")
        )
    )
    return run_id
