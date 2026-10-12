"""Ingest endpoints end to end against Postgres with local storage."""

from __future__ import annotations

import base64
import copy
import hashlib
import math
from typing import Any

import httpx
import numpy as np
import pytest
from jsonschema import Draft7Validator
from sqlalchemy import text

from olmo_eval_api.services.derive import instance_key_hash
from olmo_eval_api.settings import UPLOADER_SERVICE_ACCOUNT
from tests.factories import (
    instance_rows,
    run_payload,
    seed_contract_example_run,
    seed_run,
    suite,
    task_payload,
)
from tests.helpers import load_example, load_schema

INGEST_SCHEMA = load_schema("ingest-v1.schema.json")
RUN_ID = "db385601e335"


def assert_valid(definition: str, payload: Any) -> None:
    schema = {**INGEST_SCHEMA, "$ref": f"#/definitions/{definition}"}
    errors = sorted(Draft7Validator(schema).iter_errors(payload), key=str)
    assert not errors, "\n".join(e.message for e in errors[:5])


async def table_counts(session: Any) -> dict[str, int]:
    out = {}
    for table in (
        "runs",
        "models",
        "task_variants",
        "task_results",
        "instance_results",
        "task_result_vectors",
        "artifacts",
        "inference_batches",
        "run_inference",
        "suite_results",
        "suite_defs",
    ):
        out[table] = (await session.execute(text(f"SELECT count(*) FROM {table}"))).scalar_one()
    return out


async def test_health(client: httpx.AsyncClient) -> None:
    response = await client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok" and body["mode"] == "all" and body["db"] == "ok"
    assert response.headers["x-request-id"]


async def test_request_id_passthrough(client: httpx.AsyncClient) -> None:
    response = await client.get("/health", headers={"X-Request-Id": "abcdef12-3456"})
    assert response.headers["x-request-id"] == "abcdef12-3456"
    response = await client.get("/health", headers={"X-Request-Id": "bad id!"})
    assert response.headers["x-request-id"] != "bad id!"


async def test_whoami(client: httpx.AsyncClient) -> None:
    response = await client.get("/v1/whoami")
    assert response.status_code == 200
    body = response.json()
    assert_valid("WhoAmIResponse", body)
    assert body["email"] == "tester@allenai.org"
    assert body["principal_type"] == "user"
    assert body["protocol_versions"] == [1]


async def test_contract_example_round_trip(client: httpx.AsyncClient, session: Any) -> None:
    start = load_example("ingest/run-upsert-start.request.json")
    response = await client.put(f"/v1/runs/{RUN_ID}", json=start)
    assert response.status_code == 200, response.text
    body = response.json()
    assert_valid("RunUpsertResponse", body)
    assert body["created"] is True
    assert body["status"] == "running"
    assert body["upload_state"] == "uploading"
    assert body["uploaded_by"] == "tester@allenai.org"
    assert body["gcs_prefix"] == f"gs://ai2-skiff2-olmo-eval-results/dev/runs/{RUN_ID}/"
    assert body["dashboard_url"] == f"http://localhost:5173/runs/{RUN_ID}"

    final = load_example("ingest/run-upsert-final.request.json")
    body = (await client.put(f"/v1/runs/{RUN_ID}", json=final)).json()
    assert body["created"] is False and body["status"] == "complete"
    assert body["upload_state"] == "uploading"

    sign = (
        await client.post(
            f"/v1/runs/{RUN_ID}/artifacts:sign",
            json=load_example("ingest/artifacts-sign.request.json"),
        )
    ).json()
    assert_valid("SignArtifactsResponse", sign)
    assert len(sign["uploads"]) == 4
    for upload in sign["uploads"]:
        assert upload["skip"] is False
        assert upload["url"].startswith("http://test/_local/objects/dev/runs/")
        assert set(upload["headers"]) == {
            "Content-Type",
            "Content-MD5",
            "x-goog-content-length-range",
        }

    tr = (
        await client.post(
            f"/v1/runs/{RUN_ID}/task-results", json=load_example("ingest/task-result.request.json")
        )
    ).json()
    assert_valid("TaskResultUpsertResponse", tr)
    assert tr["created"] is True and tr["instances_cleared"] is False

    batch = (
        await client.post(
            f"/v1/task-results/{tr['task_result_id']}/instances",
            json=load_example("ingest/instances.request.json"),
        )
    ).json()
    assert_valid("InstanceBatchResponse", batch)
    assert batch == {"task_result_id": tr["task_result_id"], "received": 3, "total_stored": 3}

    inference = (
        await client.put(
            f"/v1/runs/{RUN_ID}/inference", json=load_example("ingest/inference.request.json")
        )
    ).json()
    assert_valid("InferenceUploadResponse", inference)
    assert inference["batches_stored"] == 3

    done = (
        await client.post(
            f"/v1/runs/{RUN_ID}/complete", json=load_example("ingest/complete.request.json")
        )
    ).json()
    assert_valid("CompleteResponse", done)
    assert done["status"] == "complete" and done["upload_state"] == "complete"
    assert done["task_results"] == 1 and done["instances"] == 3
    assert len(done["artifacts_missing"]) == 4
    assert any("expected 600 instances" in w for w in done["warnings"])

    run = (
        (await session.execute(text("SELECT * FROM runs WHERE run_id = :r"), {"r": RUN_ID}))
        .mappings()
        .one()
    )
    assert run["status"] == "complete"
    assert run["upload_state"] == "complete"
    assert run["beaker_experiment_id"] == "01M3TFPE2PQ9DDAZD3YT52V0JZ"
    assert run["beaker_workspace"] == "ai2/olmo-eval-debug"
    assert run["headline"]["kind"] == "task"
    assert run["headline"]["name"] == "omniscience:judge"
    assert math.isclose(run["headline"]["score"], -33.44481605351171)
    assert "omniscience" in run["search_text"]

    model = (await session.execute(text("SELECT * FROM models"))).mappings().one()
    assert model["family"] == "llama3.1"
    assert model["series_label"] == "Llama-3.1-8B-Instruct"

    row = (
        (
            await session.execute(
                text("SELECT * FROM task_results WHERE id = :i"), {"i": tr["task_result_id"]}
            )
        )
        .mappings()
        .one()
    )
    # The omniscience index is not the mean of its per-instance values.
    assert row["score_is_mean"] is False
    assert row["stderr"] is None
    assert row["metric_kind"] == "unbounded"
    assert row["display_format"] == "raw"
    assert row["metric_meta"]["any__any__hallucination_rate:omniscience_judge"]["kind"] == "binary"
    assert (
        row["metric_meta"]["any__any__hallucination_rate:omniscience_judge"]["higher_is_better"]
        is False
    )
    assert row["finish_reason_counts"] == {"stop": 3}
    # The client's task-level totals win over the sums of the three stored instances.
    assert row["completion_tokens_total"] == 398112
    assert row["prompt_tokens_total"] == 61234
    assert row["reported_completion_tokens_total"] == 398112
    assert row["first_request_at"] is not None and row["last_completed_at"] is not None
    assert row["attributed_inference_seconds"] is None
    assert row["finalized_at"] is not None
    assert run["startup_seconds"] == 78.4
    assert run["processing_seconds"] == 1266.1
    assert run["processing_started_at"] is not None


async def test_old_client_payloads_ingest(client: httpx.AsyncClient, session: Any) -> None:
    """Payloads without the runtime fields still ingest; token totals come from instances."""
    final = load_example("ingest/run-upsert-final.request.json")
    for key in ("startup_seconds", "processing_started_at", "processing_seconds"):
        del final["run"][key]
    assert (await client.put(f"/v1/runs/{RUN_ID}", json=final)).status_code == 200
    task = load_example("ingest/task-result.request.json")
    for key in (
        "first_request_at",
        "last_completed_at",
        "prompt_tokens_total",
        "completion_tokens_total",
        "attributed_inference_seconds",
    ):
        del task[key]
    tr = (await client.post(f"/v1/runs/{RUN_ID}/task-results", json=task)).json()
    response = await client.post(
        f"/v1/task-results/{tr['task_result_id']}/instances",
        json=load_example("ingest/instances.request.json"),
    )
    assert response.status_code == 200
    response = await client.post(
        f"/v1/runs/{RUN_ID}/complete", json=load_example("ingest/complete.request.json")
    )
    assert response.status_code == 200
    row = (await session.execute(text("SELECT * FROM task_results"))).mappings().one()
    assert row["completion_tokens_total"] == 59 + 9 + 2
    assert row["reported_completion_tokens_total"] is None
    assert row["first_request_at"] is None
    run = (await session.execute(text("SELECT * FROM runs"))).mappings().one()
    assert run["startup_seconds"] is None and run["processing_seconds"] is None
    results = (await client.get(f"/api/runs/{RUN_ID}/task-results")).json()
    runtime = results["items"][0]["runtime"]
    assert runtime["basis"] == "not_recorded"
    assert runtime["inference_seconds"] is None
    # Older runs fall back to the slowest provider init for startup.
    assert runtime["startup_seconds"] == max(final["run"]["provider_init_seconds"].values())


async def test_reupload_is_idempotent(client: httpx.AsyncClient, session: Any) -> None:
    await seed_contract_example_run(client)
    first = await table_counts(session)
    vec1 = (await session.execute(text("SELECT key_hashes, scores FROM task_result_vectors"))).one()
    run1 = (
        await session.execute(text("SELECT headline, num_instances, uploaded_by FROM runs"))
    ).one()
    await seed_contract_example_run(client)
    second = await table_counts(session)
    vec2 = (await session.execute(text("SELECT key_hashes, scores FROM task_result_vectors"))).one()
    run2 = (
        await session.execute(text("SELECT headline, num_instances, uploaded_by FROM runs"))
    ).one()
    assert first == second
    assert first["instance_results"] == 3
    assert bytes(vec1[0]) == bytes(vec2[0]) and bytes(vec1[1]) == bytes(vec2[1])
    assert run1 == run2


async def test_reupload_task_result_clears_instances(client: httpx.AsyncClient) -> None:
    await seed_contract_example_run(client)
    tr = (
        await client.post(
            f"/v1/runs/{RUN_ID}/task-results", json=load_example("ingest/task-result.request.json")
        )
    ).json()
    assert tr["created"] is False and tr["instances_cleared"] is True


async def test_retried_instance_batch_is_harmless(client: httpx.AsyncClient) -> None:
    await seed_contract_example_run(client)
    tr = (
        await client.post(
            f"/v1/runs/{RUN_ID}/task-results", json=load_example("ingest/task-result.request.json")
        )
    ).json()
    batch = load_example("ingest/instances.request.json")
    for _ in range(3):
        body = (
            await client.post(f"/v1/task-results/{tr['task_result_id']}/instances", json=batch)
        ).json()
    assert body["total_stored"] == 3


async def test_model_conflict(client: httpx.AsyncClient) -> None:
    body = load_example("ingest/run-upsert-start.request.json")
    assert (await client.put(f"/v1/runs/{RUN_ID}", json=body)).status_code == 200
    other = copy.deepcopy(body)
    other["model"]["name"] = "someone/else"
    response = await client.put(f"/v1/runs/{RUN_ID}", json=other)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "conflict"


async def test_status_never_downgraded(client: httpx.AsyncClient) -> None:
    await seed_contract_example_run(client)
    start = load_example("ingest/run-upsert-start.request.json")
    body = (await client.put(f"/v1/runs/{RUN_ID}", json=start)).json()
    assert body["status"] == "complete"
    assert body["upload_state"] == "complete"


async def test_reupload_keeps_a_complete_run_visible(client: httpx.AsyncClient) -> None:
    await seed_contract_example_run(client)
    final = load_example("ingest/run-upsert-final.request.json")
    final["run"]["status"] = "partial"
    body = (await client.put(f"/v1/runs/{RUN_ID}", json=final)).json()
    assert body["status"] == "partial" and body["upload_state"] == "complete"


async def test_identical_task_result_reupload_keeps_instances(
    client: httpx.AsyncClient, session: Any
) -> None:
    run_id = "same00000001"
    await seed_run(client, run_id=run_id, tasks={"gsm8k": [1.0, 0.0, 1.0]})
    finalized = text("SELECT finalized_at FROM task_results WHERE run_id = :r")
    before = await session.scalar(finalized, {"r": run_id})
    payload = task_payload(
        "gsm8k",
        task_hash=hashlib.sha256(b"gsm8k").hexdigest()[:16],
        primary_metric="acc:default",
        corpus_score=2 / 3,
        n=3,
    )
    # An interrupted re-upload: the run and the task result are sent again, nothing else.
    body = run_payload(run_id=run_id, model_name="test-org/test-model", task_specs=["gsm8k"])
    assert (await client.put(f"/v1/runs/{run_id}", json=body)).json()["upload_state"] == "complete"
    response = (await client.post(f"/v1/runs/{run_id}/task-results", json=payload)).json()
    assert response["unchanged"] is True and response["instances_cleared"] is False
    counts = await table_counts(session)
    assert counts["instance_results"] == 3 and counts["task_result_vectors"] == 1
    assert await session.scalar(finalized, {"r": run_id}) == before

    payload["metrics"]["acc"]["default"] = 0.5
    response = (await client.post(f"/v1/runs/{run_id}/task-results", json=payload)).json()
    assert response["unchanged"] is False and response["instances_cleared"] is True


async def test_changed_predictions_file_is_not_unchanged(client: httpx.AsyncClient) -> None:
    run_id = "same00000002"
    await seed_run(client, run_id=run_id, tasks={"gsm8k": [1.0, 0.0]})
    payload = task_payload(
        "gsm8k",
        task_hash=hashlib.sha256(b"gsm8k").hexdigest()[:16],
        primary_metric="acc:default",
        corpus_score=0.5,
        n=2,
    )
    artifact = {
        "path": payload["predictions_path"],
        "size_bytes": 10,
        "md5_b64": base64.b64encode(hashlib.md5(b"new").digest()).decode(),
        "content_type": "application/jsonl",
        "kind": "predictions",
        "task_name": "gsm8k",
    }
    response = await client.post(
        f"/v1/runs/{run_id}/artifacts:sign", json={"artifacts": [artifact]}
    )
    assert response.status_code == 200, response.text
    response = (await client.post(f"/v1/runs/{run_id}/task-results", json=payload)).json()
    assert response["unchanged"] is False


async def test_run_id_mismatch_and_protocol(client: httpx.AsyncClient) -> None:
    body = load_example("ingest/run-upsert-start.request.json")
    response = await client.put("/v1/runs/aaaaaaaaaaaa", json=body)
    assert response.status_code == 400
    bad = copy.deepcopy(body)
    bad["protocol_version"] = 2
    response = await client.put(f"/v1/runs/{RUN_ID}", json=bad)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
    response = await client.put("/v1/runs/BAD_ID", json=body)
    assert response.status_code == 422


async def test_missing_run_404(client: httpx.AsyncClient) -> None:
    task = load_example("ingest/task-result.request.json")
    response = await client.post("/v1/runs/abcdef123456/task-results", json=task)
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"
    response = await client.post(
        "/v1/task-results/999999/instances", json=load_example("ingest/instances.request.json")
    )
    assert response.status_code == 404
    response = await client.post(
        "/v1/runs/abcdef123456/artifacts:sign",
        json=load_example("ingest/artifacts-sign.request.json"),
    )
    assert response.status_code == 404


async def test_payload_too_large(client: httpx.AsyncClient) -> None:
    response = await client.put(
        f"/v1/runs/{RUN_ID}",
        content=b"x" * (16 * 1024 * 1024 + 1),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "payload_too_large"


async def test_artifact_upload_and_skip(client: httpx.AsyncClient) -> None:
    await client.put(
        f"/v1/runs/{RUN_ID}", json=load_example("ingest/run-upsert-start.request.json")
    )
    data = b'{"hello": "world"}\n'
    md5 = base64.b64encode(hashlib.md5(data).digest()).decode()
    request = {
        "artifacts": [
            {
                "path": "metrics.json",
                "size_bytes": len(data),
                "md5_b64": md5,
                "content_type": "application/json",
                "kind": "metrics",
                "task_name": None,
            }
        ]
    }
    upload = (await client.post(f"/v1/runs/{RUN_ID}/artifacts:sign", json=request)).json()[
        "uploads"
    ][0]
    assert upload["skip"] is False
    bad = await client.put(upload["url"], content=b"tampered", headers=upload["headers"])
    assert bad.status_code == 400
    ok = await client.put(upload["url"], content=data, headers=upload["headers"])
    assert ok.status_code == 200
    again = (await client.post(f"/v1/runs/{RUN_ID}/artifacts:sign", json=request)).json()
    assert again["uploads"][0]["skip"] is True
    assert again["uploads"][0]["url"] is None
    done = (
        await client.post(
            f"/v1/runs/{RUN_ID}/complete",
            json={**load_example("ingest/complete.request.json"), "expected_task_results": 0},
        )
    ).json()
    assert done["artifacts_missing"] == []


async def test_artifact_path_validation(client: httpx.AsyncClient) -> None:
    await client.put(
        f"/v1/runs/{RUN_ID}", json=load_example("ingest/run-upsert-start.request.json")
    )
    request = load_example("ingest/artifacts-sign.request.json")
    request["artifacts"][0]["path"] = "../escape.json"
    response = await client.post(f"/v1/runs/{RUN_ID}/artifacts:sign", json=request)
    assert response.status_code == 422


async def test_delete_run(client: httpx.AsyncClient, session: Any, app: Any) -> None:
    await seed_contract_example_run(client)
    response = await client.delete(
        f"/v1/runs/{RUN_ID}", headers={"X-Olmo-Eval-Token": "dev:other@allenai.org"}
    )
    assert response.status_code == 403
    response = await client.delete(f"/v1/runs/{RUN_ID}")
    assert response.status_code == 204
    counts = await table_counts(session)
    assert counts["runs"] == 0 and counts["task_results"] == 0 and counts["instance_results"] == 0
    assert counts["models"] == 1  # models and task variants outlive runs


async def test_reupload_cannot_take_over_a_service_account_run(
    client: httpx.AsyncClient, session: Any
) -> None:
    uploader = {"X-Olmo-Eval-Token": f"dev:{UPLOADER_SERVICE_ACCOUNT}"}
    alice = {"X-Olmo-Eval-Token": "dev:alice@allenai.org"}
    run_id = "spoof0000001"
    body = run_payload(run_id=run_id, model_name="org/model", author="bob")
    assert (await client.put(f"/v1/runs/{run_id}", json=body, headers=uploader)).status_code == 200
    spoofed = run_payload(run_id=run_id, model_name="org/model", author="alice")
    for headers in (alice, uploader):
        response = await client.put(f"/v1/runs/{run_id}", json=spoofed, headers=headers)
        assert response.status_code == 403, response.text
        sql = text("SELECT author FROM runs WHERE run_id = :r")
        assert await session.scalar(sql, {"r": run_id}) == "bob"
    assert (await client.delete(f"/v1/runs/{run_id}", headers=alice)).status_code == 403
    bob = {"X-Olmo-Eval-Token": "dev:bob@allenai.org"}
    assert (await client.delete(f"/v1/runs/{run_id}", headers=bob)).status_code == 204


async def test_writes_to_another_users_run_are_forbidden(client: httpx.AsyncClient) -> None:
    await seed_contract_example_run(client)
    other = {"X-Olmo-Eval-Token": "dev:other@allenai.org"}
    task_result_id = await _first_task_result_id(client)
    attempts = [
        ("PUT", f"/v1/runs/{RUN_ID}", load_example("ingest/run-upsert-final.request.json")),
        (
            "POST",
            f"/v1/runs/{RUN_ID}/artifacts:sign",
            load_example("ingest/artifacts-sign.request.json"),
        ),
        (
            "POST",
            f"/v1/runs/{RUN_ID}/task-results",
            load_example("ingest/task-result.request.json"),
        ),
        (
            "POST",
            f"/v1/task-results/{task_result_id}/instances",
            load_example("ingest/instances.request.json"),
        ),
        ("PUT", f"/v1/runs/{RUN_ID}/inference", load_example("ingest/inference.request.json")),
        ("POST", f"/v1/runs/{RUN_ID}/complete", load_example("ingest/complete.request.json")),
    ]
    for method, path, body in attempts:
        response = await client.request(method, path, json=body, headers=other)
        assert response.status_code == 403, (path, response.text)
        assert response.json()["error"]["code"] == "forbidden"


async def test_service_account_writes_need_the_run_secret(
    client: httpx.AsyncClient, session: Any
) -> None:
    uploader = {"X-Olmo-Eval-Token": f"dev:{UPLOADER_SERVICE_ACCOUNT}"}
    secret = {**uploader, "X-Olmo-Eval-Run-Secret": "s3cret-value"}
    run_id = "secret000001"
    body = run_payload(run_id=run_id, model_name="org/model", author="bob")
    assert (await client.put(f"/v1/runs/{run_id}", json=body, headers=secret)).status_code == 200
    stored = await session.scalar(
        text("SELECT write_secret_hash FROM runs WHERE run_id = :r"), {"r": run_id}
    )
    assert stored == hashlib.sha256(b"s3cret-value").hexdigest()

    wrong = {**uploader, "X-Olmo-Eval-Run-Secret": "guess"}
    for headers in (uploader, wrong):
        response = await client.put(f"/v1/runs/{run_id}", json=body, headers=headers)
        assert response.status_code == 403
        response = await client.post(
            f"/v1/runs/{run_id}/task-results",
            json=load_example("ingest/task-result.request.json"),
            headers=headers,
        )
        assert response.status_code == 403

    assert (await client.put(f"/v1/runs/{run_id}", json=body, headers=secret)).status_code == 200
    response = await client.post(
        f"/v1/runs/{run_id}/task-results",
        json=load_example("ingest/task-result.request.json"),
        headers=secret,
    )
    assert response.status_code == 200, response.text
    # The author of a launched run and anyone holding the secret may re-upload it.
    bob = {"X-Olmo-Eval-Token": "dev:bob@allenai.org"}
    carol = {"X-Olmo-Eval-Token": "dev:carol@allenai.org", "X-Olmo-Eval-Run-Secret": "s3cret-value"}
    for headers in (bob, carol):
        response = await client.put(f"/v1/runs/{run_id}", json=body, headers=headers)
        assert response.status_code == 200, response.text


async def test_first_secret_is_kept(client: httpx.AsyncClient, session: Any) -> None:
    run_id = "secret000002"
    body = run_payload(run_id=run_id, model_name="org/model")
    assert (await client.put(f"/v1/runs/{run_id}", json=body)).status_code == 200
    sql = text("SELECT write_secret_hash FROM runs WHERE run_id = :r")
    assert await session.scalar(sql, {"r": run_id}) is None
    for value in (b"first", b"second"):
        headers = {"X-Olmo-Eval-Run-Secret": value.decode()}
        assert (
            await client.put(f"/v1/runs/{run_id}", json=body, headers=headers)
        ).status_code == 200
    assert await session.scalar(sql, {"r": run_id}) == hashlib.sha256(b"first").hexdigest()


async def _first_task_result_id(client: httpx.AsyncClient) -> int:
    response = await client.post(
        f"/v1/runs/{RUN_ID}/task-results", json=load_example("ingest/task-result.request.json")
    )
    assert response.status_code == 200, response.text
    return response.json()["task_result_id"]


async def test_uploader_may_correct_the_author_of_their_own_run(
    client: httpx.AsyncClient, session: Any
) -> None:
    run_id = "ownrun000001"
    body = run_payload(run_id=run_id, model_name="org/model", author="typo")
    assert (await client.put(f"/v1/runs/{run_id}", json=body)).status_code == 200
    body = run_payload(run_id=run_id, model_name="org/model", author="tester")
    assert (await client.put(f"/v1/runs/{run_id}", json=body)).status_code == 200
    author = await session.scalar(text("SELECT author FROM runs WHERE run_id = :r"), {"r": run_id})
    assert author == "tester"


async def test_latest_task_hash_is_the_last_uploaded(client: httpx.AsyncClient) -> None:
    """complete_run finalizes every task result at once; that must not tie their updated_at."""
    run_id = "latest000001"
    await seed_run(client, run_id=run_id, tasks={"t": [1.0] * 25}, task_hashes={"t": "h1"})
    await seed_run(client, run_id=run_id, tasks={"t": [0.0] * 25}, task_hashes={"t": "h2"})
    # Re-upload h1 after h2: h1 is now the most recent, although h2 has the larger id.
    await seed_run(client, run_id=run_id, tasks={"t": [1.0] * 25}, task_hashes={"t": "h1"})
    for _ in range(2):  # and completing again does not change that
        response = await client.get(
            "/api/subjects/task-result", params={"subject": f"r:{run_id}", "task": "t"}
        )
        assert response.status_code == 200, response.text
        assert response.json()["task_hash"] == "h1"
        done = await client.post(
            f"/v1/runs/{run_id}/complete",
            json={**load_example("ingest/complete.request.json"), "expected_task_results": 2},
        )
        assert done.status_code == 200, done.text


async def test_complete_derivations(client: httpx.AsyncClient, session: Any) -> None:
    rng = np.random.default_rng(0)
    a = rng.integers(0, 2, size=50).astype(float).tolist()
    b = rng.integers(0, 2, size=40).astype(float).tolist()
    pct = rng.integers(0, 2, size=30).astype(float).tolist()
    tasks = {"task_a": a, "task_b": b, "task_pct": pct}
    await seed_run(
        client,
        run_id="run000000001",
        tasks=tasks,
        corpus_scores={
            "task_a": sum(a) / len(a),
            "task_b": sum(b) / len(b),
            "task_pct": 100 * sum(pct) / len(pct),
        },
        suites=[suite("suite_ab", ["task_a", "task_b"], score=0.5)],
    )
    rows = {
        r["task_name"]: r
        for r in (
            await session.execute(text("SELECT * FROM task_results ORDER BY task_name"))
        ).mappings()
    }
    ra = rows["task_a"]
    assert ra["score_is_mean"] is True and ra["instance_scale"] == 1
    expected_se = float(np.std(a, ddof=1) / math.sqrt(len(a)))
    assert math.isclose(ra["stderr"], expected_se, rel_tol=1e-9)
    assert ra["metric_kind"] == "binary" and ra["display_format"] == "percent"
    assert ra["truncation_rate"] == pytest.approx(5 / 50)
    rp = rows["task_pct"]
    assert rp["score_is_mean"] is True and rp["instance_scale"] == 100
    assert math.isclose(rp["stderr"], 100 * float(np.std(pct, ddof=1) / math.sqrt(30)))

    vec = (
        (
            await session.execute(
                text("SELECT * FROM task_result_vectors WHERE task_result_id = :i"), {"i": ra["id"]}
            )
        )
        .mappings()
        .one()
    )
    keys = np.frombuffer(vec["key_hashes"], "<i8")
    scores = np.frombuffer(vec["scores"], "<f8")
    assert vec["n"] == 50 and list(keys) == sorted(keys)
    by_key = {instance_key_hash(f"q{i}"): v for i, v in enumerate(a)}
    assert all(by_key[int(k)] == s for k, s in zip(keys, scores, strict=True))

    sr = (await session.execute(text("SELECT * FROM suite_results"))).mappings().one()
    se_b = float(np.std(b, ddof=1) / math.sqrt(len(b)))
    assert math.isclose(sr["stderr"], math.sqrt(0.25 * expected_se**2 + 0.25 * se_b**2))
    assert sr["tasks_missing"] == [] and sr["display_format"] == "percent"
    assert sr["n_instances"] == 90

    run = (await session.execute(text("SELECT * FROM runs"))).mappings().one()
    assert run["headline"]["kind"] == "suite" and run["headline"]["name"] == "suite_ab"
    assert run["num_tasks"] == 3 and run["num_instances"] == 120


async def test_suite_with_missing_task_and_mean_headline(
    client: httpx.AsyncClient, session: Any
) -> None:
    await seed_run(
        client,
        run_id="run000000002",
        tasks={"t1": [1.0, 0.0] * 10, "t2": [1.0] * 20},
        suites=[suite("s", ["t1", "t3"], score=None, aggregation="display_only")],
    )
    sr = (await session.execute(text("SELECT * FROM suite_results"))).mappings().one()
    assert sr["tasks_missing"] == ["t3"]
    run = (await session.execute(text("SELECT headline FROM runs"))).scalar_one()
    assert run["kind"] == "mean" and math.isclose(run["score"], 0.75)


async def test_nested_average_suite_weighs_every_task_equally(
    client: httpx.AsyncClient, session: Any
) -> None:
    t1, t2, t3 = [1.0, 0.0] * 10, [1.0, 1.0, 0.0] * 10, [0.0, 1.0, 1.0, 1.0] * 10
    await seed_run(
        client,
        run_id="run000000004",
        tasks={"t1": t1, "t2": t2, "t3": t3},
        suites=[
            suite("outer", ["suite:inner", "t3"], score=0.6),
            suite("inner", ["t1", "t2"], score=0.6, parent="outer"),
        ],
    )
    rows = {
        r["suite_name"]: r
        for r in (await session.execute(text("SELECT * FROM suite_results"))).mappings()
    }
    ses = [float(np.std(x, ddof=1) / math.sqrt(len(x))) for x in (t1, t2, t3)]
    expected = math.sqrt(sum(se**2 for se in ses) / 9)
    assert math.isclose(rows["outer"]["stderr"], expected)
    assert rows["outer"]["n_instances"] == 20 + 30 + 40


async def test_complete_warns_about_undefined_child_suites(client: httpx.AsyncClient) -> None:
    await client.put(f"/v1/runs/{RUN_ID}", json=run_payload(run_id=RUN_ID, model_name="m"))
    response = await client.post(
        f"/v1/runs/{RUN_ID}/complete",
        json={
            "status": "complete",
            "finished_at": None,
            "duration_seconds": 1.0,
            "suites": [suite("lonely", ["suite:ghost"])],
            "expected_task_results": 0,
            "expected_artifacts": 0,
        },
    )
    assert response.status_code == 200
    assert any("ghost" in w for w in response.json()["warnings"])


async def test_upsert_keeps_tags_added_in_the_dashboard(client: httpx.AsyncClient) -> None:
    body = run_payload(run_id=RUN_ID, model_name="m", tags=["cli"], status="running")
    assert (await client.put(f"/v1/runs/{RUN_ID}", json=body)).status_code in (200, 201)
    response = await client.patch(f"/api/runs/{RUN_ID}", json={"tags": ["cli", "baseline"]})
    assert response.status_code == 200
    body["run"]["status"] = "complete"
    body["run"]["tags"] = ["cli", "final"]
    assert (await client.put(f"/v1/runs/{RUN_ID}", json=body)).status_code == 200
    detail = (await client.get(f"/api/runs/{RUN_ID}")).json()
    assert detail["tags"] == ["cli", "baseline", "final"]


async def test_reupload_does_not_restore_tags_removed_in_the_dashboard(
    client: httpx.AsyncClient,
) -> None:
    body = run_payload(run_id=RUN_ID, model_name="m", tags=["cli", "smoke"])
    assert (await client.put(f"/v1/runs/{RUN_ID}", json=body)).status_code == 200
    response = await client.patch(f"/api/runs/{RUN_ID}", json={"tags": ["cli"]})
    assert response.status_code == 200
    body["run"]["tags"] = ["cli", "smoke", "new"]
    assert (await client.put(f"/v1/runs/{RUN_ID}", json=body)).status_code == 200
    detail = (await client.get(f"/api/runs/{RUN_ID}")).json()
    assert detail["tags"] == ["cli", "new"]


async def test_null_scores_and_failed_task(client: httpx.AsyncClient, session: Any) -> None:
    await seed_run(
        client,
        run_id="run000000003",
        tasks={"ok": [1.0, None, 0.0, 1.0], "bad": []},
        errors={"bad": "CUDA out of memory"},
        status="partial",
    )
    run = (await session.execute(text("SELECT * FROM runs"))).mappings().one()
    assert run["status"] == "partial" and run["num_failed_tasks"] == 1
    ok = (
        (await session.execute(text("SELECT * FROM task_results WHERE task_name = 'ok'")))
        .mappings()
        .one()
    )
    assert ok["score_is_mean"] is True
    vec = (
        await session.execute(
            text("SELECT scores FROM task_result_vectors WHERE task_result_id = :i"),
            {"i": ok["id"]},
        )
    ).scalar_one()
    assert np.isnan(np.frombuffer(vec, "<f8")).sum() == 1


async def test_instance_batch_validation(client: httpx.AsyncClient) -> None:
    await client.put(f"/v1/runs/{RUN_ID}", json=run_payload(run_id=RUN_ID, model_name="m"))
    tr = (
        await client.post(
            f"/v1/runs/{RUN_ID}/task-results",
            json=task_payload("t", task_hash="abc", primary_metric="acc:x", corpus_score=1, n=1),
        )
    ).json()
    rows = instance_rows([1.0], "acc:x")
    rows[0]["prompt_preview"] = "x" * 301
    response = await client.post(
        f"/v1/task-results/{tr['task_result_id']}/instances",
        json={"batch_index": 0, "instances": rows},
    )
    assert response.status_code == 422
    response = await client.post(
        f"/v1/task-results/{tr['task_result_id']}/instances",
        json={"batch_index": 0, "instances": []},
    )
    assert response.status_code == 422


async def test_duplicate_native_ids_in_one_batch(client: httpx.AsyncClient) -> None:
    await client.put(f"/v1/runs/{RUN_ID}", json=run_payload(run_id=RUN_ID, model_name="m"))
    tr = (
        await client.post(
            f"/v1/runs/{RUN_ID}/task-results",
            json=task_payload("t", task_hash="abc", primary_metric="acc:x", corpus_score=1, n=2),
        )
    ).json()
    rows = instance_rows([1.0, 0.0], "acc:x")
    rows[1]["native_id"] = rows[0]["native_id"]
    body = (
        await client.post(
            f"/v1/task-results/{tr['task_result_id']}/instances",
            json={"batch_index": 0, "instances": rows},
        )
    ).json()
    assert body["total_stored"] == 1


async def test_task_variant_suites_union(client: httpx.AsyncClient, session: Any) -> None:
    await client.put(f"/v1/runs/{RUN_ID}", json=run_payload(run_id=RUN_ID, model_name="m"))
    p = task_payload("t", task_hash="abc", primary_metric="acc:x", corpus_score=1, n=0)
    p["suites"] = ["s1"]
    await client.post(f"/v1/runs/{RUN_ID}/task-results", json=p)
    p["suites"] = ["s2"]
    await client.post(f"/v1/runs/{RUN_ID}/task-results", json=p)
    suites = (await session.execute(text("SELECT suites FROM task_variants"))).scalar_one()
    assert suites == ["s1", "s2"]


async def test_ingest_mode_rejects_dashboard_routes(engine: Any, settings: Any) -> None:
    from olmo_eval_api.main import create_app

    ingest = create_app(settings.model_copy(update={"api_mode": "ingest"}))
    async with ingest.router.lifespan_context(ingest):
        transport = httpx.ASGITransport(app=ingest)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            assert (await c.get("/api/me")).status_code == 404
            assert (await c.get("/health")).json()["mode"] == "ingest"
            assert (await c.get("/v1/whoami")).status_code == 401
