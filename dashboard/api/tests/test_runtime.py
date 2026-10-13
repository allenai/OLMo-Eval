"""Task runtime: the derivation rules and the endpoints that return runtime."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest
from sqlalchemy import text

from olmo_eval_api.services.runtime import RunTiming, derive_runtime, run_startup_seconds
from tests.factories import seed_run, suite
from tests.test_api_endpoints import check, get, post

T0 = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)


def tr(**kw: Any) -> dict[str, Any]:
    row = {
        "prompt_tokens_total": None,
        "completion_tokens_total": None,
        "attributed_inference_seconds": None,
        "first_request_at": None,
        "last_completed_at": None,
        "num_instances": 200,
    }
    row.update(kw)
    return row


# ---------------------------------------------------------------------------
# derive_runtime
# ---------------------------------------------------------------------------


def test_measured_single_task() -> None:
    run = RunTiming(
        startup_seconds=30.0,
        processing_started_at=T0,
        processing_seconds=100.0,
        n_task_results=1,
        all_have_tokens=True,
        total_tokens=400,
    )
    rt = derive_runtime(
        tr(
            prompt_tokens_total=300,
            completion_tokens_total=100,
            attributed_inference_seconds=80.0,
            first_request_at=T0 + timedelta(seconds=1),
            last_completed_at=T0 + timedelta(seconds=99),
        ),
        run,
    )
    # Measured wins over attributed.
    assert rt.basis == "measured"
    assert rt.inference_seconds == 100.0
    assert rt.with_startup_seconds == 130.0
    assert rt.token_share == 1.0
    assert rt.seconds_per_1k_instances == 500.0
    assert rt.span_start_s == 1.0 and rt.span_end_s == 99.0


def test_attributed_beats_estimated() -> None:
    run = RunTiming(
        processing_seconds=100.0, n_task_results=3, all_have_tokens=True, total_tokens=1000
    )
    rt = derive_runtime(
        tr(prompt_tokens_total=200, completion_tokens_total=50, attributed_inference_seconds=12.5),
        run,
    )
    assert rt.basis == "attributed"
    assert rt.inference_seconds == 12.5
    assert rt.token_share == 0.25
    # No startup recorded: with_startup is unknown.
    assert rt.startup_seconds is None and rt.with_startup_seconds is None


def test_estimated_from_token_share() -> None:
    run = RunTiming(
        startup_seconds=10.0,
        processing_seconds=200.0,
        n_task_results=2,
        all_have_tokens=True,
        total_tokens=400,
    )
    rt = derive_runtime(tr(prompt_tokens_total=250, completion_tokens_total=50), run)
    assert rt.basis == "estimated"
    assert rt.inference_seconds == 150.0
    assert rt.with_startup_seconds == 160.0
    assert rt.span_start_s is None


def test_missing_tokens_in_one_task_is_not_recorded() -> None:
    run = RunTiming(
        processing_seconds=200.0, n_task_results=2, all_have_tokens=False, total_tokens=300
    )
    rt = derive_runtime(tr(prompt_tokens_total=250, completion_tokens_total=50), run)
    assert rt.basis == "not_recorded"
    assert rt.inference_seconds is None and rt.token_share is None
    assert rt.prompt_tokens_total == 250
    assert rt.seconds_per_1k_instances is None


def test_zero_run_tokens_is_not_recorded() -> None:
    run = RunTiming(
        processing_seconds=200.0, n_task_results=2, all_have_tokens=True, total_tokens=0
    )
    rt = derive_runtime(tr(prompt_tokens_total=0, completion_tokens_total=0), run)
    assert rt.basis == "not_recorded"
    assert rt.token_share is None


def test_zero_task_tokens_estimates_zero() -> None:
    run = RunTiming(
        processing_seconds=200.0, n_task_results=2, all_have_tokens=True, total_tokens=50
    )
    rt = derive_runtime(tr(prompt_tokens_total=0, completion_tokens_total=0, num_instances=0), run)
    assert rt.basis == "estimated" and rt.inference_seconds == 0.0
    assert rt.seconds_per_1k_instances is None  # no instances


def test_old_run_with_provider_init_only() -> None:
    startup = run_startup_seconds(None, {"w0": 70.5, "w1": 72.25, "w2": float("nan")})
    assert startup == 72.25
    run = RunTiming(startup_seconds=startup, n_task_results=1)
    rt = derive_runtime(tr(), run)
    assert rt.basis == "not_recorded"
    assert rt.startup_seconds == 72.25 and rt.with_startup_seconds is None
    assert run_startup_seconds(None, None) is None
    assert run_startup_seconds(5.0, {"w0": 70.0}) == 5.0


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


async def _set_run(session: Any, run_id: str, *, gpu: str, count: int, **cols: Any) -> None:
    values = {"startup_seconds": None, "processing_started_at": None, "processing_seconds": None}
    values.update(cols)
    await session.execute(
        text(
            "UPDATE runs SET startup_seconds = :startup_seconds, "
            "processing_started_at = :processing_started_at, "
            "processing_seconds = :processing_seconds, provider_init_seconds = NULL, "
            "environment = environment || jsonb_build_object('gpu_type', CAST(:gpu AS text), "
            "'gpu_count', CAST(:count AS int)) WHERE run_id = :r"
        ),
        {**values, "gpu": gpu, "count": count, "r": run_id},
    )


async def _set_task(session: Any, run_id: str, task: str, **cols: Any) -> None:
    sets = ", ".join(f"{k} = :{k}" for k in cols)
    await session.execute(
        text(f"UPDATE task_results SET {sets} WHERE run_id = :r AND task_name = :t"),
        {**cols, "r": run_id, "t": task},
    )


@pytest.fixture
async def runtime_runs(client: httpx.AsyncClient, session: Any) -> dict[str, str]:
    """Runs of task "arc" with every runtime basis.

    A (model one, H100 x4): only task, measured 100 s, startup 30 s.
    B (model one, H100 x4): arc and gsm, 200 s split 3:1 by tokens, startup 20 s.
    C (model two, A100 x8): arc attributed 40 s; gsm lacks tokens.
    D (model two, H100 x4): nothing recorded.
    """
    scores = [1.0, 0.0, 1.0, 1.0]
    ids = {}
    specs = [
        ("A", "rta00001", "org/model-one", {"arc": scores}, []),
        (
            "B",
            "rtb00001",
            "org/model-one",
            {"arc": scores, "gsm": scores},
            [suite("rtcore", ["arc", "gsm"])],
        ),
        ("C", "rtc00001", "org/model-two", {"arc": scores, "gsm": scores}, []),
        ("D", "rtd00001", "org/model-two", {"arc": scores}, []),
    ]
    for i, (name, run_id, model, tasks, suites) in enumerate(specs):
        ids[name] = await seed_run(
            client,
            run_id=run_id,
            model_name=model,
            tasks=tasks,
            suites=suites,
            created_at=T0 + timedelta(hours=i),
            session=session,
        )
    await _set_run(
        session, ids["A"], gpu="H100", count=4, startup_seconds=30.0, processing_seconds=100.0
    )
    await _set_run(
        session,
        ids["B"],
        gpu="H100",
        count=4,
        startup_seconds=20.0,
        processing_started_at=T0,
        processing_seconds=200.0,
    )
    await _set_task(
        session,
        ids["B"],
        "arc",
        prompt_tokens_total=250,
        completion_tokens_total=50,
        first_request_at=T0 + timedelta(seconds=2),
        last_completed_at=T0 + timedelta(seconds=180),
    )
    await _set_task(session, ids["B"], "gsm", prompt_tokens_total=80, completion_tokens_total=20)
    await _set_run(
        session, ids["C"], gpu="A100", count=8, startup_seconds=60.0, processing_seconds=90.0
    )
    await _set_task(session, ids["C"], "arc", attributed_inference_seconds=40.0)
    await _set_task(
        session, ids["C"], "gsm", prompt_tokens_total=None, completion_tokens_total=None
    )
    await _set_run(session, ids["D"], gpu="H100", count=4)
    await session.commit()
    return ids


async def test_task_results_include_runtime(
    client: httpx.AsyncClient, runtime_runs: dict[str, str]
) -> None:
    body = await get(
        client, f"/api/runs/{runtime_runs['B']}/task-results", "RunTaskResultsResponse"
    )
    by_task = {i["task_name"]: i["runtime"] for i in body["items"]}
    assert by_task["arc"]["basis"] == "estimated"
    assert math.isclose(by_task["arc"]["inference_seconds"], 150.0)
    assert math.isclose(by_task["arc"]["with_startup_seconds"], 170.0)
    assert math.isclose(by_task["arc"]["token_share"], 0.75)
    assert by_task["arc"]["span_start_s"] == 2.0 and by_task["arc"]["span_end_s"] == 180.0
    assert math.isclose(by_task["arc"]["seconds_per_1k_instances"], 150.0 / 4 * 1000)
    assert math.isclose(by_task["gsm"]["inference_seconds"], 50.0)

    detail = await get(client, f"/api/runs/{runtime_runs['B']}", "RunDetail")
    assert detail["startup_seconds"] == 20.0
    assert detail["processing_seconds"] == 200.0
    assert detail["processing_started_at"] is not None

    c = await get(client, f"/api/runs/{runtime_runs['C']}/task-results", "RunTaskResultsResponse")
    by_task = {i["task_name"]: i["runtime"] for i in c["items"]}
    assert by_task["arc"]["basis"] == "attributed" and by_task["arc"]["inference_seconds"] == 40.0
    assert by_task["gsm"]["basis"] == "not_recorded"


async def test_task_leaderboard_runtime_and_gpu(
    client: httpx.AsyncClient, runtime_runs: dict[str, str]
) -> None:
    body = await get(client, "/api/tasks/leaderboard", "LeaderboardResponse", task="arc")
    by_run = {i["run_id"]: i for i in body["items"]}
    assert set(by_run) == {runtime_runs["B"], runtime_runs["D"]}  # latest run per model
    b = by_run[runtime_runs["B"]]
    assert b["runtime"]["basis"] == "estimated" and b["gpu_type"] == "H100" and b["gpu_count"] == 4
    assert by_run[runtime_runs["D"]]["runtime"]["basis"] == "not_recorded"

    body = await get(
        client, "/api/tasks/leaderboard", "LeaderboardResponse", task="arc", gpu_type="A100"
    )
    assert [i["run_id"] for i in body["items"]] == [runtime_runs["C"]]
    assert body["items"][0]["runtime"]["inference_seconds"] == 40.0
    assert body["items"][0]["gpu_count"] == 8

    body = await get(client, "/api/suites/leaderboard", "LeaderboardResponse", suite="rtcore")
    assert body["items"] and all(i["runtime"] is None for i in body["items"])


async def test_task_runtime_endpoint(
    client: httpx.AsyncClient, runtime_runs: dict[str, str]
) -> None:
    body = await get(client, "/api/tasks/arc/runtime", "TaskRuntimeResponse")
    assert body["task_hash"] is not None
    assert body["gpu_types"] == ["H100", "A100"]
    assert body["not_recorded"] == 1
    # Newest first; D has no runtime.
    assert [p["run_id"] for p in body["points"]] == [
        runtime_runs["C"],
        runtime_runs["B"],
        runtime_runs["A"],
    ]
    rows = {(r["model"]["name"], r["gpu_type"], r["gpu_count"]): r for r in body["rows"]}
    one = rows[("org/model-one", "H100", 4)]
    assert one["runs"] == 2
    assert one["inference"]["n"] == 2
    assert math.isclose(one["inference"]["median"], 125.0)
    assert math.isclose(one["inference"]["p90"], 145.0)
    assert one["inference"]["min"] == 100.0 and one["inference"]["max"] == 150.0
    assert math.isclose(one["with_startup"]["median"], 150.0)
    assert math.isclose(one["seconds_per_1k_instances"]["max"], 150.0 / 4 * 1000)
    assert one["latest_score"] == pytest.approx(0.75)
    two = rows[("org/model-two", "A100", 8)]
    assert two["runs"] == 1 and two["inference"]["median"] == 40.0
    # Fastest first.
    assert body["rows"][0]["model"]["name"] == "org/model-two"

    body = await get(client, "/api/tasks/arc/runtime", "TaskRuntimeResponse", gpu_type="A100")
    assert [p["run_id"] for p in body["points"]] == [runtime_runs["C"]]
    assert body["not_recorded"] == 0
    assert body["gpu_types"] == ["H100", "A100"]  # the filter list ignores the filter

    body = await get(
        client,
        "/api/tasks/arc/runtime",
        "TaskRuntimeResponse",
        hash=body["task_hash"],
        family="no-such-family",
    )
    assert body["rows"] == [] and body["points"] == [] and body["not_recorded"] == 0

    response = await client.get("/api/tasks/no-such-task/runtime")
    assert response.status_code == 404


async def test_compare_matrix_runtime(
    client: httpx.AsyncClient, runtime_runs: dict[str, str]
) -> None:
    a, b = f"r:{runtime_runs['A']}", f"r:{runtime_runs['B']}"
    body = {"subjects": [a, b], "baseline": a, "scope": "task:arc", "metric": "runtime:inference"}
    matrix = await post(client, "/api/compare/matrix", body, "MatrixResponse")
    row = matrix["rows"][0]
    assert row["meta"]["higher_is_better"] is False and row["meta"]["unit"] == "s"
    assert row["metric_key"] == "runtime:inference"
    cell_a, cell_b = row["cells"]
    assert cell_a["score"] == 100.0 and cell_a["delta"] is None
    assert math.isclose(cell_b["score"], 150.0)
    assert math.isclose(cell_b["delta"]["delta"], 50.0)
    assert cell_b["delta"]["improved"] is False
    assert cell_b["delta"]["ci_low"] is None and cell_b["delta"]["method"] == "none"

    body["metric"] = "runtime:with_startup"
    matrix = await post(client, "/api/compare/matrix", body, "MatrixResponse")
    assert [c["score"] for c in matrix["rows"][0]["cells"]] == [130.0, 170.0]

    # A suite cell sums its tasks and adds the largest startup once.
    body = {"subjects": [b, a], "scope": "suite:rtcore", "metric": "runtime:with_startup"}
    matrix = await post(client, "/api/compare/matrix", body, "MatrixResponse")
    suite_row = next(r for r in matrix["rows"] if r["kind"] == "suite")
    cell_b, cell_a = suite_row["cells"]
    assert math.isclose(cell_b["score"], 220.0) and cell_b["status"] == "ok"
    assert cell_a["status"] == "partial" and cell_a["children_missing"] == 1
    assert math.isclose(cell_a["score"], 130.0)

    # D has no runtime: its cell is missing.
    d = f"r:{runtime_runs['D']}"
    body = {"subjects": [a, d], "scope": "task:arc", "metric": "runtime:inference"}
    matrix = await post(client, "/api/compare/matrix", body, "MatrixResponse")
    assert [c["status"] for c in matrix["rows"][0]["cells"]] == ["ok", "missing"]
    assert matrix["coverage"]["missing"] == 1

    body["metric"] = "runtime:bogus"
    assert (await client.post("/api/compare/matrix", json=body)).status_code == 400


async def test_runtime_metric_rejected_where_pairing_is_needed(
    client: httpx.AsyncClient, runtime_runs: dict[str, str]
) -> None:
    a, b = f"r:{runtime_runs['A']}", f"r:{runtime_runs['B']}"
    pairwise = {"subjects": [a, b], "scope": "task:arc", "metric": "runtime:inference"}
    response = await client.post("/api/compare/pairwise", json=pairwise)
    assert response.status_code == 400
    check("ErrorResponse", response.json())
    assert "runtime" in response.json()["error"]["message"]
    contingency = {"a": a, "b": b, "scope": "task:arc", "metric": "runtime:inference"}
    assert (await client.post("/api/compare/contingency", json=contingency)).status_code == 400
    instances = {
        "subjects": [a, b],
        "task_name": "arc",
        "metric": "runtime:with_startup",
        "filter": "all",
    }
    assert (await client.post("/api/compare/instances", json=instances)).status_code == 400
