"""Hostile or malformed input to the read API: 400s instead of 500s, bounded work."""

from __future__ import annotations

import base64
import json
from typing import Any

import httpx
import numpy as np
import pytest
from sqlalchemy import event, text

from olmo_eval_api.routers.api import run_tabs
from olmo_eval_api.services.subjects import resolve_subjects
from olmo_eval_api.storage.gcs import attachment_disposition
from tests.factories import seed_run, suite


def cursor(values: Any) -> str:
    return base64.urlsafe_b64encode(json.dumps(values).encode()).decode().rstrip("=")


@pytest.fixture
async def two_runs(client: httpx.AsyncClient) -> dict[str, Any]:
    rng = np.random.default_rng(3)
    ids = {}
    for name, p in (("base", 0.5), ("cand", 0.7)):
        arc = (rng.random(40) < p).astype(float).tolist()
        gsm = (rng.random(30) < p).astype(float).tolist()
        ids[name] = await seed_run(
            client,
            run_id=f"{name}00000001",
            model_name=f"org/{name}-model",
            tasks={"arc": arc, "gsm": gsm},
            suites=[suite("core", ["arc", "gsm"], score=(np.mean(arc) + np.mean(gsm)) / 2)],
        )
    trs = (await client.get(f"/api/runs/{ids['cand']}/task-results")).json()["items"]
    ids["tr"] = next(t["task_result_id"] for t in trs if t["task_name"] == "arc")
    return ids


async def assert_400(response: httpx.Response) -> None:
    assert response.status_code == 400, response.text
    assert response.json()["error"]["code"] == "bad_request"


@pytest.mark.parametrize(
    "params",
    [
        {"cursor": cursor([{"a": 1}, "x"])},  # wrong type for a timestamp key
        {"cursor": cursor(["not-a-date", "x"])},
        {"sort": "-step", "cursor": cursor(["abc", "x"])},
        {"sort": "score:task:arc", "cursor": cursor(["abc", "x"])},
        {"step_min": str(2**70)},  # does not fit bigint
    ],
)
async def test_runs_list_rejects_bad_parameters(
    client: httpx.AsyncClient, two_runs: dict, params: dict
) -> None:
    await assert_400(await client.get("/api/runs", params=params))


async def test_other_lists_reject_bad_cursors(client: httpx.AsyncClient, two_runs: dict) -> None:
    run_id, tr = two_runs["cand"], two_runs["tr"]
    cases = [
        ("/api/tasks/leaderboard", {"task": "arc", "cursor": cursor(["x"])}),
        ("/api/tasks/leaderboard", {"task": "arc", "cursor": cursor([-5])}),
        ("/api/suites", {"cursor": cursor([7])}),
        ("/api/groups", {"cursor": cursor([1, 2])}),
        ("/api/groups", {"active_days": str(10**12)}),
        ("/api/activity", {"cursor": cursor(["x", "y"])}),
        (f"/api/runs/{run_id}/task-results/{tr}/instances", {"cursor": cursor(["x", "y"])}),
        (f"/api/runs/{run_id}/task-results/{tr}/instances", {"len_min": str(2**40)}),
        ("/api/instances/detail", {"task_result_id": str(2**70), "native_id": "q1"}),
    ]
    for url, params in cases:
        await assert_400(await client.get(url, params=params))


async def test_stats_parameters_are_validated(client: httpx.AsyncClient, two_runs: dict) -> None:
    run_id, base = two_runs["cand"], f"r:{two_runs['base']}"
    for params in ({"seed": "-1"}, {"seed": str(2**70)}, {"threshold": "1.5"}):
        await assert_400(
            await client.get(
                f"/api/runs/{run_id}/task-results", params={"baseline": base, **params}
            )
        )
    await assert_400(
        await client.get(f"/api/runs/{run_id}/suites", params={"baseline": base, "seed": "-5"})
    )
    response = await client.get(
        f"/api/runs/{run_id}/task-results/{two_runs['tr']}/instances", params={"threshold": "nan"}
    )
    assert response.status_code == 422
    subjects = [f"r:{run_id}", base]
    for path, body in (
        ("matrix", {"subjects": subjects, "scope": "all", "metric": "primary", "seed": -1}),
        ("pairwise", {"subjects": subjects, "scope": "all", "metric": "primary", "seed": 2**40}),
        ("contingency", {"a": subjects[0], "b": subjects[1], "scope": "all", "metric": "primary",
                         "threshold": 2.0}),
    ):  # fmt: skip
        await assert_400(await client.post(f"/api/compare/{path}", json=body))
    body = {"subjects": subjects, "task_name": "arc", "metric": "primary", "filter": "all"}
    await assert_400(await client.post("/api/compare/instances", json={**body, "cursor": "!!"}))
    await assert_400(
        await client.post("/api/compare/instances", json={**body, "cursor": cursor([-3])})
    )
    ok = await client.post("/api/compare/instances", json={**body, "limit": 5})
    assert ok.status_code == 200
    nxt = ok.json()["next_cursor"]
    page2 = await client.post("/api/compare/instances", json={**body, "limit": 5, "cursor": nxt})
    assert page2.status_code == 200
    assert page2.json()["items"][0]["native_id"] != ok.json()["items"][0]["native_id"]


async def test_zero_length_record_is_not_fetched(
    client: httpx.AsyncClient, app: Any, two_runs: dict
) -> None:
    # A zero-length range would turn into an invalid Range header, which GCS may answer with
    # the whole predictions file.
    tr = two_runs["tr"]
    async with app.state.db.sessionmaker() as s:
        await s.execute(
            text(
                "UPDATE instance_results SET pred_offset = 0, pred_length = 0 "
                "WHERE task_result_id = :t AND native_id = 'q1'"
            ),
            {"t": tr},
        )
        await s.commit()
    calls = []

    async def read_range(key: str, start: int, length: int) -> bytes:
        calls.append((key, start, length))
        return b"{}"

    app.state.storage.read_range = read_range
    response = await client.get(
        "/api/instances/detail", params={"task_result_id": tr, "native_id": "q1"}
    )
    assert response.status_code == 200
    assert response.json()["prediction"] is None
    assert "empty" in response.json()["unavailable_reason"]
    assert calls == []


def test_record_cache_is_bounded_by_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_tabs, "_RECORD_CACHE", type(run_tabs._RECORD_CACHE)())
    monkeypatch.setattr(run_tabs, "_record_cache_bytes", 0)
    big = run_tabs._RECORD_CACHE_MAX_BYTES // 8
    for i in range(20):
        run_tabs._cache_record((f"k{i}", 0, big, ""), {"i": i})
    assert run_tabs._record_cache_bytes <= run_tabs._RECORD_CACHE_MAX_BYTES
    assert len(run_tabs._RECORD_CACHE) == 8
    run_tabs._cache_record(("huge", 0, big + 1, ""), {})  # too large to cache at all
    assert ("huge", 0, big + 1) not in run_tabs._RECORD_CACHE


def test_download_filename_cannot_break_the_header() -> None:
    assert attachment_disposition("preds.jsonl") == 'attachment; filename="preds.jsonl"'
    value = attachment_disposition('a"b\r\nSet-Cookie: x;.jsonl')
    assert value.count('"') == 2 and "\r" not in value and "\n" not in value
    assert attachment_disposition("") == 'attachment; filename="download"'


async def test_subject_resolution_uses_a_fixed_number_of_queries(
    client: httpx.AsyncClient, app: Any, two_runs: dict, session: Any
) -> None:
    model_ids = (await session.execute(text("SELECT model_id FROM models"))).scalars().all()
    model_keys = [f"m:{m}" for m in model_ids]
    assert len(model_keys) == 2
    statements: list[str] = []

    def count(*args: Any) -> None:
        statements.append(args[2])

    sync_engine = app.state.db.engine.sync_engine
    event.listen(sync_engine, "before_cursor_execute", count)
    try:
        await resolve_subjects(session, [f"r:{two_runs['base']}"])
        one = len(statements)
        statements.clear()
        keys = [f"r:{two_runs['base']}", f"r:{two_runs['cand']}", *model_keys]
        subjects, missing = await resolve_subjects(session, keys)
        many = len(statements)
    finally:
        event.remove(sync_engine, "before_cursor_execute", count)
    assert not missing and len(subjects) == len(keys)
    assert all(s.trs for s in subjects)
    assert many <= one + 2  # one extra query for models, one for their task results
