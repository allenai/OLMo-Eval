"""Every read endpoint on a seeded database, validated against api-v1.schema.json."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import numpy as np
import pytest
from jsonschema import Draft7Validator, FormatChecker
from sqlalchemy import text

from tests.factories import seed_contract_example_run, seed_run, suite
from tests.helpers import load_schema

API_SCHEMA = load_schema("api-v1.schema.json")


def check(definition: str, payload: Any) -> Any:
    schema = {**API_SCHEMA, "$ref": f"#/definitions/{definition}"}
    errors = sorted(
        Draft7Validator(schema, format_checker=FormatChecker()).iter_errors(payload), key=str
    )
    assert not errors, f"{definition}: " + "\n".join(
        f"{list(e.absolute_path)}: {e.message}" for e in errors[:5]
    )
    return payload


async def get(client: httpx.AsyncClient, url: str, definition: str, **params: Any) -> Any:
    response = await client.get(url, params=params)
    assert response.status_code == 200, f"{url}: {response.text}"
    return check(definition, response.json())


async def post(client: httpx.AsyncClient, url: str, body: Any, definition: str) -> Any:
    response = await client.post(url, json=body)
    assert response.status_code == 200, f"{url}: {response.text}"
    return check(definition, response.json())


def binary(rng: np.random.Generator, n: int, p: float) -> list[float]:
    return (rng.random(n) < p).astype(float).tolist()


@pytest.fixture
async def seeded(client: httpx.AsyncClient, session: Any) -> dict[str, Any]:
    rng = np.random.default_rng(1)
    now = datetime.now(UTC)
    ids = {}
    suites = [suite("core", ["arc", "gsm"], score=None)]
    for i, (step, p) in enumerate([(1000, 0.5), (2000, 0.6), (3000, 0.7)]):
        arc = binary(rng, 80, p)
        gsm = binary(rng, 60, p - 0.1)
        suites[0]["score"] = (np.mean(arc) + np.mean(gsm)) / 2
        ids[f"step{step}"] = await seed_run(
            client,
            run_id=f"olmo{step:08d}",
            model_name=f"allenai/OLMo-test-step{step}",
            group="olmo-sweep",
            tasks={"arc": arc, "gsm": gsm, "unbounded": rng.normal(0, 3, 40).tolist()},
            suites=suites,
            tags=["sweep"],
            created_at=now - timedelta(hours=10 - i),
            launch_id="launch01",
            session=session,
        )
    ids["qwen"] = await seed_run(
        client,
        run_id="qwen00000001",
        model_name="Qwen/Qwen3-8B",
        group="other-group",
        author="someone",
        tasks={"arc": binary(rng, 80, 0.8), "gsm": binary(rng, 60, 0.75)},
        suites=[suite("core", ["arc", "gsm"], score=0.775)],
        errors=None,
        session=session,
        created_at=now - timedelta(hours=2),
    )
    ids["failed"] = await seed_run(
        client,
        run_id="fail00000001",
        model_name="Qwen/Qwen3-8B",
        group="other-group",
        status="partial",
        tasks={"arc": binary(rng, 30, 0.8), "mmlu": []},
        errors={"mmlu": "CUDA out of memory"},
        session=session,
        created_at=now - timedelta(hours=1),
    )
    ids["example"] = await seed_contract_example_run(client)
    # Runs are ordered by when they started; make the contract example the newest.
    for sql in (
        "UPDATE runs SET created_at = :t WHERE run_id = :r",
        "UPDATE task_results SET run_created_at = :t WHERE run_id = :r",
    ):
        await session.execute(text(sql), {"t": now, "r": ids["example"]})
    await session.commit()
    return ids


async def test_runs_list_and_facets(client: httpx.AsyncClient, seeded: dict) -> None:
    body = await get(client, "/api/runs", "RunsListResponse")
    assert body["total"] == 6
    assert body["items"][0]["run_id"] == seeded["example"]  # newest first
    body = await get(
        client,
        "/api/runs",
        "RunsListResponse",
        cols="task:arc,suite:core",
        baseline=f"r:{seeded['step1000']}",
        group="olmo-sweep",
        sort="-score:task:arc",
    )
    assert body["total"] == 3
    assert [c["key"] for c in body["columns"]] == ["task:arc", "suite:core"]
    scores = [row["scores"]["task:arc"]["score"] for row in body["items"]]
    assert scores == sorted(scores, reverse=True)
    deltas = [row["scores"]["task:arc"]["delta"] for row in body["items"]]
    assert all(d["method"] == "unpaired" for d in deltas)
    assert body["baseline"]["key"] == f"r:{seeded['step1000']}"

    facets = await get(client, "/api/runs/facets", "RunsFacetsResponse")
    groups = {f["value"]: f["count"] for f in facets["group"]}
    assert groups["olmo-sweep"] == 3
    statuses = {f["value"] for f in facets["status"]}
    assert {"complete", "partial"} <= statuses
    facets = await get(client, "/api/runs/facets", "RunsFacetsResponse", group="olmo-sweep")
    # A facet ignores its own filter, so every group still shows.
    assert len(facets["group"]) >= 3
    assert {f["value"] for f in facets["model"]} == {
        "allenai/OLMo-test-step1000",
        "allenai/OLMo-test-step2000",
        "allenai/OLMo-test-step3000",
    }


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        ({"model": "*olmo*"}, 3),
        ({"task": "mmlu"}, 1),
        ({"user": "me"}, 6),
        ({"user": "someone"}, 1),
        ({"tag": "sweep"}, 3),
        ({"not_tag": "sweep"}, 3),
        ({"status": "partial"}, 1),
        ({"has_failures": "true"}, 1),
        ({"suite": "core"}, 4),
        ({"suite": "core", "suite_coverage": "any"}, 5),
        ({"q": "qwen other"}, 2),
        ({"date": "24h"}, 6),
        ({"step_min": 2000}, 2),
        ({"launch": "launch01"}, 3),
        ({"commit": "e81a69bf"}, 6),
    ],
)
async def test_runs_filters(
    client: httpx.AsyncClient, seeded: dict, params: dict, expected: int
) -> None:
    body = await get(client, "/api/runs", "RunsListResponse", **params)
    assert body["total"] == expected, params


async def test_runs_pagination_is_stable(client: httpx.AsyncClient, seeded: dict) -> None:
    for sort in ("-created_at", "model", "-step", "-score:task:arc"):
        seen = []
        cursor = None
        while True:
            params = {"limit": 2, "sort": sort}
            if cursor:
                params["cursor"] = cursor
            body = await get(client, "/api/runs", "RunsListResponse", **params)
            seen.extend(r["run_id"] for r in body["items"])
            cursor = body["next_cursor"]
            if not cursor:
                break
        assert len(seen) == len(set(seen)) == 6, sort


async def test_run_detail_patch_and_tags(client: httpx.AsyncClient, seeded: dict) -> None:
    run_id = seeded["step2000"]
    body = await get(client, f"/api/runs/{run_id}", "RunDetail")
    assert body["model"]["step"] == 2000
    assert body["model"]["series"] == "allenai/OLMo-test"
    assert [s["name"] for s in body["suites_used"]] == ["core"]
    assert {s["run_id"] for s in body["siblings"]} == {seeded["step1000"], seeded["step3000"]}
    assert body["links"]["beaker_experiment"].startswith("https://beaker.org/ex/")
    assert body["reproduce_command"].startswith("olmo-eval run")

    response = await client.patch(f"/api/runs/{run_id}", json={"tags": ["a", "b"], "notes": "hi"})
    assert response.status_code == 200
    body = check("RunDetail", response.json())
    assert body["tags"] == ["a", "b"] and body["notes"] == "hi"
    response = await client.post(
        "/api/runs/tags", json={"run_ids": [run_id, seeded["qwen"]], "add": ["x"], "remove": ["a"]}
    )
    assert check("BulkTagResponse", response.json())["updated"] == 2
    body = await get(client, f"/api/runs/{run_id}", "RunDetail")
    assert body["tags"] == ["b", "x"]
    assert (await client.get("/api/runs/zzzzzzzzzzzz")).status_code == 404
    response = await client.patch(f"/api/runs/{run_id}", json={"tags": ["bad tag"]})
    assert response.status_code == 400


async def test_tag_note_and_view_caps(
    client: httpx.AsyncClient, seeded: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = seeded["step2000"]
    many = [f"t{i}" for i in range(51)]
    assert (await client.patch(f"/api/runs/{run_id}", json={"tags": many})).status_code == 400
    assert (await client.patch(f"/api/runs/{run_id}", json={"tags": many[:50]})).status_code == 200
    response = await client.post(
        "/api/runs/tags", json={"run_ids": [run_id], "add": ["extra"], "remove": []}
    )
    assert response.status_code == 400 and run_id in response.json()["error"]["message"]
    swap = {"run_ids": [run_id], "add": ["extra"], "remove": ["t0"]}
    assert (await client.post("/api/runs/tags", json={**swap})).status_code == 200
    notes = {"notes": "x" * 10_001}
    assert (await client.patch(f"/api/runs/{run_id}", json=notes)).status_code == 400

    from olmo_eval_api.routers.api import misc

    monkeypatch.setattr(misc, "MAX_VIEWS_PER_USER", 1)
    view = {"name": "v", "page": "runs", "query": "", "shared": False}
    assert (await client.post("/api/views", json=view)).status_code == 200
    assert (await client.post("/api/views", json=view)).status_code == 400


async def test_run_task_results_with_baseline(client: httpx.AsyncClient, seeded: dict) -> None:
    run_id = seeded["step3000"]
    body = await get(client, f"/api/runs/{run_id}/task-results", "RunTaskResultsResponse")
    assert {r["task_name"] for r in body["items"]} == {"arc", "gsm", "unbounded"}
    assert all(r["baseline"] is None for r in body["items"])
    body = await get(
        client,
        f"/api/runs/{run_id}/task-results",
        "RunTaskResultsResponse",
        baseline=f"r:{seeded['step1000']}",
    )
    arc = next(r for r in body["items"] if r["task_name"] == "arc")
    assert arc["baseline"]["delta"]["method"] == "paired_bootstrap"
    assert arc["baseline"]["delta"]["n_shared"] == 80
    assert arc["baseline"]["contingency"]["n_shared"] == 80
    assert arc["score_is_mean"] is True and arc["stderr"] > 0
    unb = next(r for r in body["items"] if r["task_name"] == "unbounded")
    assert unb["baseline"]["contingency"] is None
    # Cached responses are identical apart from nothing at all.
    again = await get(
        client,
        f"/api/runs/{run_id}/task-results",
        "RunTaskResultsResponse",
        baseline=f"r:{seeded['step1000']}",
    )
    assert again == body
    model_baseline = body["items"][0]["baseline"]
    assert model_baseline is not None


async def test_run_suites(client: httpx.AsyncClient, seeded: dict) -> None:
    run_id = seeded["step3000"]
    body = await get(
        client,
        f"/api/runs/{run_id}/suites",
        "RunSuitesResponse",
        baseline=f"r:{seeded['step1000']}",
    )
    assert [s["name"] for s in body["items"]] == ["core"]
    core = body["items"][0]
    assert core["depth"] == 0 and core["stderr"] > 0
    assert core["baseline"]["delta"]["method"] == "paired_bootstrap"
    assert body["unsuited_tasks"] == ["unbounded"]


async def test_instances_filters_and_detail(
    client: httpx.AsyncClient, seeded: dict, app: Any
) -> None:
    run_id = seeded["step3000"]
    trs = (await client.get(f"/api/runs/{run_id}/task-results")).json()["items"]
    arc = next(r for r in trs if r["task_name"] == "arc")
    url = f"/api/runs/{run_id}/task-results/{arc['task_result_id']}/instances"
    body = await get(client, url, "InstancesResponse")
    assert body["total"] == 80 and body["kind"] == "binary"
    n_correct = (await get(client, url, "InstancesResponse", correct="correct"))["total"]
    n_wrong = (await get(client, url, "InstancesResponse", correct="incorrect"))["total"]
    assert n_correct + n_wrong == 80
    body = await get(
        client, url, "InstancesResponse", baseline=f"r:{seeded['step1000']}", vs="gained"
    )
    assert all(r["correct"] and r["baseline_correct"] is False for r in body["items"])
    body = await get(
        client, url, "InstancesResponse", baseline=f"r:{seeded['step1000']}", sort="-delta"
    )
    deltas = [r["delta"] for r in body["items"]]
    assert deltas == sorted(deltas, reverse=True)
    body = await get(client, url, "InstancesResponse", q="question 1", limit=5)
    assert all("question 1" in r["prompt_preview"] for r in body["items"])
    assert body["next_cursor"] is not None
    page2 = await get(
        client, url, "InstancesResponse", q="question 1", limit=5, cursor=body["next_cursor"]
    )
    assert not {r["native_id"] for r in body["items"]} & {r["native_id"] for r in page2["items"]}
    # Later pages reuse the first page's total from the cursor instead of counting again.
    assert page2["total"] == body["total"]
    from olmo_eval_api.services.queries import decode_cursor, encode_cursor

    values = decode_cursor(body["next_cursor"])
    assert values is not None and values[-1] == body["total"]
    forged = encode_cursor([*values[:-1], 999])
    page2 = await get(client, url, "InstancesResponse", q="question 1", limit=5, cursor=forged)
    assert page2["total"] == 999
    body = await get(client, url, "InstancesResponse", finish_reason="length", len_min=15)
    assert all(r["finish_reason"] == "length" for r in body["items"])
    unb = next(r for r in trs if r["task_name"] == "unbounded")
    response = await client.get(
        f"/api/runs/{run_id}/task-results/{unb['task_result_id']}/instances",
        params={"correct": "correct"},
    )
    assert response.status_code == 400

    detail = await get(
        client,
        "/api/instances/detail",
        "InstanceDetailResponse",
        task_result_id=arc["task_result_id"],
        native_id="q3",
    )
    assert detail["prediction"] is None
    assert "offsets were not recorded" in detail["unavailable_reason"]


async def test_instance_detail_reads_byte_ranges(client: httpx.AsyncClient, app: Any) -> None:
    run_id = await seed_contract_example_run(client)
    storage = app.state.storage
    prefix = f"dev/runs/{run_id}/"
    lines = [b'{"native_id": "1", "model_output": [{"text": "a"}]}', b'{"native_id": "2"}']
    preds = b"\n".join(lines) + b"\n"
    storage.write(
        prefix
        + "predictions/meta-llama_Llama-3.1-8B-Instruct/omniscience_judge_afd898-predictions.jsonl",
        preds,
    )
    trs = (await client.get(f"/api/runs/{run_id}/task-results")).json()["items"]
    tr_id = trs[0]["task_result_id"]
    # Point the first instance at the first line.
    await app.state.db.engine.dispose()
    async with app.state.db.sessionmaker() as s:
        from sqlalchemy import text

        await s.execute(
            text(
                "UPDATE instance_results SET pred_offset = 0, pred_length = :n "
                "WHERE task_result_id = :t AND native_id = '1'"
            ),
            {"n": len(lines[0]), "t": tr_id},
        )
        await s.commit()
    detail = await get(
        client,
        "/api/instances/detail",
        "InstanceDetailResponse",
        task_result_id=tr_id,
        native_id="1",
    )
    assert detail["prediction"] == {"native_id": "1", "model_output": [{"text": "a"}]}
    assert detail["request"] is None
    assert "request:" in detail["unavailable_reason"]

    # A re-upload that rewrites the file is not served from the record cache.
    pred_key = (
        prefix
        + "predictions/meta-llama_Llama-3.1-8B-Instruct/omniscience_judge_afd898-predictions.jsonl"
    )
    async with app.state.db.sessionmaker() as s:
        await s.execute(
            text("UPDATE artifacts SET uploaded = true WHERE run_id = :r"), {"r": run_id}
        )
        await s.commit()
    params = {"task_result_id": tr_id, "native_id": "1"}
    await get(client, "/api/instances/detail", "InstanceDetailResponse", **params)  # cached now
    storage.write(pred_key, b'{"native_id": "1", "model_output": [{"text": "b"}]}\n')
    async with app.state.db.sessionmaker() as s:
        await s.execute(
            text("UPDATE artifacts SET md5_b64 = 'new' WHERE run_id = :r AND kind = 'predictions'"),
            {"r": run_id},
        )
        await s.commit()
    detail = await get(client, "/api/instances/detail", "InstanceDetailResponse", **params)
    assert detail["prediction"]["model_output"] == [{"text": "b"}]


async def test_histograms_inference_configs_artifacts(
    client: httpx.AsyncClient, seeded: dict
) -> None:
    run_id = seeded["step3000"]
    trs = (await client.get(f"/api/runs/{run_id}/task-results")).json()["items"]
    arc = next(r for r in trs if r["task_name"] == "arc")
    body = await get(
        client,
        f"/api/runs/{run_id}/task-results/{arc['task_result_id']}/histograms",
        "HistogramsResponse",
        baseline=f"r:{seeded['step1000']}",
        bins=10,
    )
    assert sum(body["score"]["counts"]) == 80
    assert body["max_tokens"] == 256
    assert sum(body["completion_tokens"]["all"]["counts"]) == 80

    example = seeded["example"]
    body = await get(client, f"/api/runs/{example}/inference", "InferenceResponse")
    assert body["available"] is True
    assert body["kpis"]["total_requests"] == 192
    assert body["kpis"]["gpu_hours"] == pytest.approx(1341.517121553421 / 3600 * 4)
    assert body["series"] and body["gpu_devices"] is not None
    body = await get(
        client, f"/api/runs/{run_id}/inference", "InferenceResponse", baseline=f"r:{example}"
    )
    assert body["available"] is False and body["baseline_kpis"]["total_requests"] == 192

    body = await get(client, f"/api/runs/{example}/configs", "RunConfigsResponse")
    assert body["model_config"]["kind"] == "vllm_server"
    assert body["task_configs"][0]["config"]["name"] == "omniscience"

    body = await get(client, f"/api/runs/{example}/artifacts", "ArtifactsResponse")
    assert len(body["items"]) == 4 and not any(i["uploaded"] for i in body["items"])
    assert body["links"]["beaker_result_dataset"].endswith("01M3TFPE2X873CCF4KMSKRME4D")

    sign = await post(
        client,
        "/api/artifacts/sign",
        {"gs_uri": body["items"][0]["gs_uri"]},
        "SignDownloadResponse",
    )
    assert sign["url"].startswith("http://test/_local/objects/")
    response = await client.post(
        "/api/artifacts/sign", json={"gs_uri": "gs://other-bucket/runs/x/y"}
    )
    assert response.status_code == 400


async def test_baseline_suggestions(client: httpx.AsyncClient, seeded: dict) -> None:
    body = await get(
        client,
        f"/api/runs/{seeded['step3000']}/baseline-suggestions",
        "BaselineSuggestionsResponse",
    )
    reasons = [i["reason"] for i in body["items"]]
    assert reasons[0] == "previous_checkpoint"
    assert body["items"][0]["model"]["step"] == 2000
    assert "same_group" in reasons


async def test_subjects(client: httpx.AsyncClient, seeded: dict) -> None:
    body = await post(
        client,
        "/api/subjects/resolve",
        {"subjects": [f"r:{seeded['qwen']}", "m:000000000000", f"r:{seeded['failed']}"]},
        "ResolveSubjectsResponse",
    )
    assert body["missing"] == ["m:000000000000"]
    labels = [i["label"] for i in body["items"]]
    assert labels[0] != labels[1]  # same model, so the experiment name disambiguates
    model_id = body["items"][0]["model"]["model_id"]
    body = await get(
        client,
        "/api/subjects/task-result",
        "SubjectTaskResultResponse",
        subject=f"m:{model_id}",
        task="arc",
    )
    # The newest finalized result without an error wins.
    assert body["run_id"] == seeded["failed"]
    response = await client.get(
        "/api/subjects/task-result", params={"subject": "m:000000000000", "task": "arc"}
    )
    assert response.status_code == 404


async def test_compare_matrix(client: httpx.AsyncClient, seeded: dict) -> None:
    subjects = [f"r:{seeded['step1000']}", f"r:{seeded['step3000']}", f"r:{seeded['qwen']}"]
    body = await post(
        client,
        "/api/compare/matrix",
        {"subjects": subjects, "scope": "suite:core", "metric": "primary", "baseline": subjects[0]},
        "MatrixResponse",
    )
    assert [r["key"] for r in body["rows"]] == ["suite:core", "task:arc", "task:gsm"]
    arc = body["rows"][1]
    assert arc["cells"][0]["delta"] is None
    assert arc["cells"][1]["delta"]["method"] == "paired_bootstrap"
    assert body["rows"][0]["cells"][2]["delta"]["method"] == "paired_bootstrap"
    assert body["mde80"] is not None
    assert body["coverage"] == {
        "n_subjects": 3,
        "n_tasks": 2,
        "complete": 6,
        "missing": 0,
        "failed": 0,
        "hash_mismatch": 0,
    }
    body = await post(
        client,
        "/api/compare/matrix",
        {"subjects": subjects, "scope": "all", "metric": "primary", "shared_only": True},
        "MatrixResponse",
    )
    assert {r["name"] for r in body["rows"]} == {"arc", "gsm", "unbounded"}
    unb = next(r for r in body["rows"] if r["name"] == "unbounded")
    assert unb["cells"][2]["status"] == "missing"
    response = await client.post(
        "/api/compare/matrix",
        json={"subjects": ["r:nonexistent1"], "scope": "all", "metric": "primary"},
    )
    assert response.status_code == 400


async def test_cached_compare_reflects_tag_and_suite_changes(
    client: httpx.AsyncClient, seeded: dict, session: Any
) -> None:
    from sqlalchemy import text

    subjects = [f"r:{seeded['step1000']}", f"r:{seeded['step3000']}"]
    matrix = {"subjects": subjects, "scope": "suite:core", "metric": "primary"}
    pairwise = {"subjects": subjects, "scope": "suite:core", "metric": "primary"}
    m1 = await post(client, "/api/compare/matrix", matrix, "MatrixResponse")
    p1 = await post(client, "/api/compare/pairwise", pairwise, "PairwiseResponse")
    assert m1 == await post(client, "/api/compare/matrix", matrix, "MatrixResponse")  # cached
    n_cached = await session.scalar(text("SELECT count(*) FROM stats_cache"))

    # A tag edit shows up in the cached responses without recomputing them.
    run_id = seeded["step1000"]
    assert (await client.patch(f"/api/runs/{run_id}", json={"tags": ["fresh"]})).status_code == 200
    m2 = await post(client, "/api/compare/matrix", matrix, "MatrixResponse")
    p2 = await post(client, "/api/compare/pairwise", pairwise, "PairwiseResponse")
    for body in (m2, p2):
        tags = {s["key"]: s["run"]["tags"] for s in body["subjects"]}
        assert tags[f"r:{run_id}"] == ["fresh"]
    assert m2["rows"] == m1["rows"] and m2["computed_at"] == m1["computed_at"]
    assert p2["pairs"] == p1["pairs"]
    assert await session.scalar(text("SELECT count(*) FROM stats_cache")) == n_cached

    # A new definition of the suite is a new cache key.
    await seed_run(
        client,
        run_id="suitechange1",
        tasks={"arc": [1.0, 0.0] * 15},
        suites=[suite("core", ["arc"], score=0.5)],
    )
    m3 = await post(client, "/api/compare/matrix", matrix, "MatrixResponse")
    assert [r["key"] for r in m3["rows"]] == ["suite:core", "task:arc"]
    p3 = await post(client, "/api/compare/pairwise", pairwise, "PairwiseResponse")
    assert p3["tasks_used"] == ["arc"]


async def test_compare_pairwise_rejects_too_much_work(
    client: httpx.AsyncClient, seeded: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    from olmo_eval_api.routers.api import compare

    monkeypatch.setattr(compare, "MAX_PAIRWISE_WORK", 1000)
    subjects = [f"r:{seeded['step1000']}", f"r:{seeded['step3000']}"]
    response = await client.post(
        "/api/compare/pairwise",
        json={"subjects": subjects, "scope": "suite:core", "metric": "primary"},
    )
    assert response.status_code == 400
    assert "bootstrap cells" in response.json()["error"]["message"]


async def test_compare_pairwise_contingency_instances(
    client: httpx.AsyncClient, seeded: dict
) -> None:
    subjects = [f"r:{seeded['step1000']}", f"r:{seeded['step3000']}", f"r:{seeded['qwen']}"]
    body = await post(
        client,
        "/api/compare/pairwise",
        {"subjects": subjects, "scope": "suite:core", "metric": "primary"},
        "PairwiseResponse",
    )
    assert len(body["pairs"]) == 6
    assert body["tasks_used"] == ["arc", "gsm"]
    rates = [r["mean_win_rate"] for r in body["rows"]]
    assert rates == sorted(rates, reverse=True)
    pair = {(p["row"], p["col"]): p for p in body["pairs"]}
    ab = pair[(subjects[0], subjects[1])]
    ba = pair[(subjects[1], subjects[0])]
    assert ab["wins"] == ba["losses"] and ab["delta"] == pytest.approx(-ba["delta"])

    body = await post(
        client,
        "/api/compare/contingency",
        {"a": subjects[1], "b": subjects[0], "scope": "all", "metric": "primary"},
        "ContingencyResponse",
    )
    nets = [i["contingency"]["net"] for i in body["items"] if i["contingency"]]
    assert nets == sorted(nets)
    assert body["totals"]["n_shared"] == 140

    body = await post(
        client,
        "/api/compare/instances",
        {
            "subjects": subjects,
            "task_name": "arc",
            "metric": "primary",
            "filter": "disagree",
            "include_previews": True,
            "limit": 10,
        },
        "CompareInstancesResponse",
    )
    assert all(0 < r["n_correct"] < r["n_present"] for r in body["items"])
    assert body["items"][0]["prompt_preview"].startswith("question")
    assert sum(body["counts_by_n_correct"]) == body["total"]
    body = await post(
        client,
        "/api/compare/instances",
        {
            "subjects": subjects,
            "task_name": "arc",
            "metric": "primary",
            "filter": "cell",
            "cell": "only_a",
            "a": subjects[1],
            "b": subjects[0],
            "sort": "native_id",
        },
        "CompareInstancesResponse",
    )
    assert all(r["correct"][1] and not r["correct"][0] for r in body["items"])


async def test_compare_instances_baseline_wrong(client: httpx.AsyncClient, seeded: dict) -> None:
    full, short = f"r:{seeded['step1000']}", f"r:{seeded['failed']}"  # 80 vs 30 arc instances
    body = await post(
        client,
        "/api/compare/instances",
        {
            "subjects": [full, short],
            "task_name": "arc",
            "metric": "primary",
            "filter": "baseline_wrong",
            "baseline": short,
            "limit": 1000,
        },
        "CompareInstancesResponse",
    )
    assert 0 < body["total"] <= 30
    assert all(r["scores"][1] is not None and r["correct"][1] is False for r in body["items"])

    steps = [f"r:{seeded['step1000']}", f"r:{seeded['step2000']}"]
    body = await post(
        client,
        "/api/compare/instances",
        {
            "subjects": steps,
            "task_name": "unbounded",
            "metric": "primary",
            "filter": "baseline_wrong",
            "baseline": steps[1],
        },
        "CompareInstancesResponse",
    )
    assert body["total"] == 0


async def test_models_tasks_suites_groups(client: httpx.AsyncClient, seeded: dict) -> None:
    body = await get(client, "/api/models", "ModelsListResponse")
    series = {r["series"]: r for r in body["items"]}
    assert series["allenai/OLMo-test"]["n_checkpoints"] == 3
    assert series["allenai/OLMo-test"]["latest_step"] == 3000
    body = await get(
        client, "/api/models/series", "ModelSeriesDetailResponse", series="allenai/OLMo-test"
    )
    assert [c["model"]["step"] for c in body["checkpoints"]] == [1000, 2000, 3000]
    body = await get(
        client,
        "/api/models/progression",
        "ProgressionResponse",
        series="allenai/OLMo-test",
        references=f"r:{seeded['qwen']}",
    )
    assert body["panels"][0]["key"] == "suite:core"
    assert [p["step"] for p in body["panels"][0]["points"]] == [1000, 2000, 3000]
    assert body["panels"][0]["references"][0]["score"] is not None
    body = await get(
        client,
        "/api/models/progression",
        "ProgressionResponse",
        series="allenai/OLMo-test",
        tasks="arc,gsm",
        merge="all",
    )
    assert [p["key"] for p in body["panels"]] == ["task:arc", "task:gsm"]

    body = await get(client, "/api/tasks", "TasksListResponse")
    assert body["items"][0]["task_name"] == "arc"  # most runs
    body = await get(client, "/api/tasks", "TasksListResponse", suite="core", sort="task_name")
    assert [r["task_name"] for r in body["items"]] == ["arc", "gsm"]
    body = await get(client, "/api/tasks/detail", "TaskDetailResponse", task="arc")
    assert body["variants"][0]["n_runs"] == 5
    body = await get(client, "/api/tasks/leaderboard", "LeaderboardResponse", task="arc")
    assert body["items"][0]["rank"] == 1
    assert body["items"][0]["tied_with_leader"] is True
    scores = [r["score"] for r in body["items"]]
    assert scores == sorted(scores, reverse=True)
    body = await get(
        client,
        "/api/tasks/leaderboard",
        "LeaderboardResponse",
        task="arc",
        per_model="all",
        limit=2,
    )
    assert body["total"] == 5 and body["next_cursor"]

    body = await get(client, "/api/suites", "SuitesListResponse")
    assert body["items"][0]["suite_name"] == "core" and body["items"][0]["n_runs"] == 4
    body = await get(client, "/api/suites/detail", "SuiteDetailResponse", suite="core")
    assert [c["name"] for c in body["tree"]["children"]] == ["arc", "gsm"]
    body = await get(client, "/api/suites/leaderboard", "LeaderboardResponse", suite="core")
    assert body["items"][0]["child_scores"].keys() == {"task:arc", "task:gsm"}

    body = await post(
        client,
        "/api/tasks/distributions",
        {"items": [{"task_name": "arc", "task_hash": None, "metric": "primary"}]},
        "DistributionsResponse",
    )
    assert body["items"][0]["n"] == 4  # latest per model
    assert body["items"][0]["top"]["score"] == max(body["items"][0]["scores"])

    body = await get(client, "/api/groups", "GroupsListResponse", include_heatmap="true")
    group = next(g for g in body["items"] if g["name"] == "olmo-sweep")
    assert group["n_runs"] == 3 and group["heatmap"]["columns"][0]["key"] == "suite:core"
    body = await get(client, "/api/groups/detail", "GroupDetailResponse", group="olmo-sweep")
    assert len(body["subjects"]) == 3 and body["tasks"] == ["arc", "gsm", "unbounded"]
    assert len(body["coverage"]) == 9


async def test_home_search_views(client: httpx.AsyncClient, seeded: dict) -> None:
    body = await get(client, "/api/activity", "ActivityResponse", limit=3)
    assert body["items"][0]["key"] == "5e0c7a9d13f2"
    keys = [i["key"] for i in body["items"]]
    body2 = await get(
        client, "/api/activity", "ActivityResponse", limit=3, cursor=body["next_cursor"]
    )
    assert not set(keys) & {i["key"] for i in body2["items"]}
    failed = next(i for i in body["items"] + body2["items"] if i["status"] == "partial")
    assert failed["failure_reason"] == "CUDA out of memory"
    launch = next(i for i in body["items"] + body2["items"] if i["key"] == "launch01")
    assert len(launch["runs"]) == 3

    body = await get(client, "/api/stats/summary", "StatsSummaryResponse", window_days=3)
    assert body["runs"] == 6 and len(body["daily"]) == 3

    body = await get(client, "/api/search", "SearchResponse", q="olmo")
    types = {g["type"] for g in body["groups"]}
    assert {"run", "model", "group"} <= types
    body = await get(client, "/api/search", "SearchResponse", q="01M3TFPE2PQ9DDAZD3YT52V0JZ")
    assert body["groups"][0]["type"] == "id"
    body = await get(client, "/api/search", "SearchResponse", q=seeded["qwen"], types="id")
    assert body["groups"][0]["items"][0]["href"] == f"/runs/{seeded['qwen']}"
    body = await get(client, "/api/search", "SearchResponse", q="e81a69b", types="id")
    assert len(body["groups"][0]["items"]) == 5

    me = await get(client, "/api/me", "MeResponse")
    assert me["username"] == "tester"
    view = await post(
        client,
        "/api/views",
        {"name": "Sweep", "page": "runs", "query": "?group=olmo-sweep", "shared": True},
        "SavedView",
    )
    assert view["query"] == "group=olmo-sweep"
    other = {"X-Goog-Authenticated-User-Email": "accounts.google.com:other@allenai.org"}
    response = await client.get("/api/views", headers=other)
    body = check("SavedViewsResponse", response.json())
    assert body["mine"] == [] and body["shared"][0]["id"] == view["id"]
    response = await client.patch(f"/api/views/{view['id']}", json={"name": "x"}, headers=other)
    assert response.status_code == 403
    response = await client.patch(f"/api/views/{view['id']}", json={"name": "Renamed"})
    assert check("SavedView", response.json())["name"] == "Renamed"
    assert (await client.delete(f"/api/views/{view['id']}")).status_code == 204
    assert (await client.get("/api/views")).json()["mine"] == []


async def test_errors_use_envelope(client: httpx.AsyncClient) -> None:
    response = await client.get("/api/runs", params={"sort": "nope"})
    assert response.status_code == 400
    check("ErrorResponse", response.json())
    response = await client.get("/api/runs", params={"limit": 0})
    assert response.status_code == 400
    response = await client.get("/api/search")
    assert response.status_code == 422
    body = check("ErrorResponse", response.json())
    assert body["error"]["code"] == "validation_error"
    assert response.headers["x-request-id"] == body["error"]["request_id"]
    response = await client.get("/api/runs", params={"cursor": "!!!"})
    assert response.status_code == 400
    assert (await get(client, "/api/health", "HealthResponse"))["mode"] == "all"


async def _latest_arc_score(client: httpx.AsyncClient) -> float:
    body = await get(client, "/api/tasks/leaderboard", "LeaderboardResponse", task="arc")
    assert len(body["items"]) == 1
    return body["items"][0]["score"]


async def test_latest_result_is_the_last_to_run_not_the_last_uploaded(
    client: httpx.AsyncClient, session: Any
) -> None:
    now = datetime.now(UTC)
    await seed_run(client, run_id="ran000000002", tasks={"arc": [1.0, 1.0]}, started_at=now)
    late = now - timedelta(days=2)
    await seed_run(client, run_id="ran000000001", tasks={"arc": [0.0, 0.0]}, started_at=late)

    assert await _latest_arc_score(client) == 1.0
    created = await session.scalar(
        text("SELECT created_at FROM runs WHERE run_id = 'ran000000001'")
    )
    assert created == late


async def test_latest_result_prefers_runs_without_an_instance_limit(
    client: httpx.AsyncClient,
) -> None:
    now = datetime.now(UTC)
    await seed_run(
        client,
        run_id="full00000001",
        tasks={"arc": [1.0, 1.0, 1.0]},
        started_at=now - timedelta(days=1),
    )
    await seed_run(
        client,
        run_id="smoke0000001",
        tasks={"arc": [0.0]},
        task_hashes={"arc": "smoke-hash"},
        task_limits={"arc": 1},
        started_at=now,
    )

    assert await _latest_arc_score(client) == 1.0
