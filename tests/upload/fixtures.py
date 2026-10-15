"""Synthetic results directories and a fake ingest service for upload tests."""

from __future__ import annotations

import base64
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import httpx

from olmo_eval.upload.manifest import build_manifest, build_model_info, build_run_info
from olmo_eval.upload.validation import PayloadValidator

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_SCHEMA = REPO_ROOT / "dashboard" / "contract" / "ingest-v1.schema.json"
API_URL = "https://ingest.test"
RUN_ID = "abc123def456"
MODEL = "allenai/OLMo-2-0425-1B"
MODEL_DIR = "allenai_OLMo-2-0425-1B"

ARC_EASY = "arc_easy:rc:olmo3base"
ARC_CHALLENGE = "arc_challenge:rc:olmo3base"
ARC_SUITE = "arc:rc:olmo3base"


def write_jsonl(path: Path, records: list[Any], newline: str = "\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        for rec in records:
            f.write((rec if isinstance(rec, str) else json.dumps(rec)) + newline)


def mc_predictions(n: int, correct_every: int = 2) -> list[dict[str, Any]]:
    """Loglikelihood multiple-choice records (four choices, integer labels)."""
    records = []
    for i in range(n):
        score = 1.0 if i % correct_every == 0 else 0.0
        records.append(
            {
                "doc_id": i,
                "native_id": f"Mercury_{i}",
                "label": i % 4,
                "model_output": [
                    {"sum_logits": -1.5 - c, "num_tokens": 3, "is_greedy": c == 0} for c in range(4)
                ],
                "instance_metrics": {
                    "accuracy": {"LogprobScorer": score},
                    "LogprobScorer": {"LogprobScorer": score},
                },
            }
        )
    return records


def mc_requests(n: int) -> list[dict[str, Any]]:
    return [
        {
            "request_type": "loglikelihood",
            "doc": {"query": f"Question {i}: which planet is closest to the sun?", "id": i},
            "request": {"context": f"Question {i}: which planet?\nAnswer:", "continuation": " A"},
            "idx": 0,
            "task_name": "arc_easy",
            "doc_id": i,
            "native_id": f"Mercury_{i}",
            "label": i % 4,
        }
        for i in range(n)
    ]


def generative_predictions(n: int) -> list[dict[str, Any]]:
    return [
        {
            "doc_id": i,
            "native_id": i + 1,
            "label": str(i * 2),
            "model_output": [
                {
                    "text": f"The answer is {i * 2}." + " pad" * 100,
                    "extracted_answer": str(i * 2),
                    "finish_reason": "stop" if i % 3 else "length",
                    "completion_tokens": 40 + i,
                }
            ],
            "instance_metrics": {"exact_match": {"exact_match": float(i % 2 == 0)}},
        }
        for i in range(n)
    ]


def chat_requests(n: int) -> list[dict[str, Any]]:
    return [
        {
            "request_type": "generate_until",
            "doc": {"id": i + 1},
            "request": {
                "context": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": f"What is {i} times two?"},
                ]
            },
            "doc_id": i,
            "native_id": i + 1,
        }
        for i in range(n)
    ]


def judge_predictions(n: int) -> list[dict[str, Any]]:
    verdicts = ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"]
    return [
        {
            "doc_id": i,
            "native_id": str(i),
            "label": {"answer": f"gold {i}"},
            "model_output": [{"text": f"answer {i}", "finish_reason": "stop", "num_tokens": 7}],
            "judge_result": {"verdict": verdicts[i % 3], "explanation": "..."},
            "instance_metrics": {
                "accuracy": {"judge": float(i % 3 == 0)},
                "hallucination_rate": {"judge": float(i % 3 == 1)},
            },
        }
        for i in range(n)
    ]


def multi_sample_predictions(n: int) -> list[dict[str, Any]]:
    records = []
    for i in range(n):
        outputs = [
            {
                "text": f"def f(): return {s}",
                "finish_reason": "stop",
                "completion_tokens": 12,
                "execution_result": {"passed": s == 0},
                "scoring_errors": ["timeout"] if (i == 0 and s == 2) else [],
            }
            for s in range(3)
        ]
        records.append(
            {
                "doc_id": i,
                "native_id": f"HumanEval/{i}",
                "label": None,
                "model_output": outputs,
                "trajectory": {"turns": []} if i == 1 else None,
                "instance_metrics": {
                    "pass_at_1": {"code": 1.0 / 3},
                    "flag": {"code": True},
                    "nan_metric": {"code": float("nan")},
                },
            }
        )
    return records


def task_row(
    spec: str,
    task_hash: str,
    metrics: dict[str, Any],
    num_instances: int,
    primary_metric: str | None,
    **extra: Any,
) -> dict[str, Any]:
    row = {
        "task": spec,
        "metrics": metrics,
        "num_instances": num_instances,
        "primary_metric": primary_metric,
        "config": {
            "name": spec.split(":")[0],
            "num_fewshot": 0,
            "limit": num_instances,
            "split": "test",
        },
        "duration_seconds": 12.5,
        "task_hash": task_hash,
        "instances_saved": num_instances,
        "instances_processed": num_instances,
        "instances_failed": 0,
    }
    row.update(extra)
    return row


def write_metrics(
    out: Path,
    tasks: list[dict[str, Any]],
    *,
    run_id: str = RUN_ID,
    errors: list[dict[str, Any]] | None = None,
    summary: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> None:
    metrics = {
        "timestamp": "2026-10-01T10:00:00.000000",
        "config": config
        or {
            "name": "default",
            "provider": {"kind": "vllm_server", "model": MODEL},
            "model_hash": "0123456789abcdef",
        },
        "tasks": tasks,
        "summary": summary or {},
        "errors": errors or [],
        "experiment_id": run_id,
        "experiment_name": "olmo-1b-arc",
        "experiment_group": "olmo-1b-arc-group",
        "experiment_duration_seconds": 120.0,
        "provider_init_seconds": {"OLMo-2-0425-1B-w0": 30.0},
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))


def write_manifest(
    out: Path,
    *,
    run_id: str = RUN_ID,
    status: str = "complete",
    tasks: dict[str, Any] | None = None,
    suites: list[dict[str, Any]] | None = None,
    task_specs: list[str] | None = None,
) -> dict[str, Any]:
    run = build_run_info(
        run_id=run_id,
        status=status,
        started_at="2026-10-01T09:58:00+00:00",
        finished_at=None if status == "running" else "2026-10-01T10:00:00+00:00",
        duration_seconds=None if status == "running" else 120.0,
        experiment_name="olmo-1b-arc",
        experiment_group="olmo-1b-arc-group",
        task_specs=task_specs or [ARC_SUITE],
        output_dir=str(out),
        harness_config={"name": "default"},
        tags=["smoke"],
        argv=["olmo-eval", "run", "-m", MODEL, "-t", ARC_SUITE],
        capture_provenance=False,
    )
    model = build_model_info(
        name=MODEL,
        path=MODEL,
        model_hash="0123456789abcdef",
        revision=None,
        provider_kind="vllm_server",
        provider_config={"kind": "vllm_server", "model": MODEL},
    )
    manifest = build_manifest(run, model, tasks or {}, suites or [])
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def make_arc_run(out: Path, n: int = 6, with_manifest: bool = True) -> Path:
    """A two-task suite run with predictions, requests, inference metrics and a log."""
    easy_hash, challenge_hash = "1111111111aaaaaa", "2222222222bbbbbb"
    write_jsonl(
        out
        / "predictions"
        / MODEL_DIR
        / f"arc_easy_rc_olmo3base_{easy_hash[-6:]}-predictions.jsonl",
        mc_predictions(n),
    )
    write_jsonl(
        out / "requests" / MODEL_DIR / f"arc_easy_rc_olmo3base_{easy_hash[-6:]}-requests.jsonl",
        mc_requests(n),
    )
    write_jsonl(
        out
        / "predictions"
        / MODEL_DIR
        / f"arc_challenge_rc_olmo3base_{challenge_hash[-6:]}-predictions.jsonl",
        mc_predictions(n, correct_every=3),
    )
    write_inference(out)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    (out / "logs" / "vllm_server_8000.log").write_text("INFO server started\n")
    metrics = {"accuracy": {"LogprobScorer": 0.5}}
    tasks = [
        task_row(ARC_EASY, easy_hash, metrics, n, "accuracy:LogprobScorer"),
        task_row(
            ARC_CHALLENGE,
            challenge_hash,
            {"accuracy": {"LogprobScorer": 2 / 6}},
            n,
            "accuracy:LogprobScorer",
        ),
    ]
    write_metrics(
        out,
        tasks,
        summary={
            ARC_EASY: {"metric": "accuracy:LogprobScorer", "score": 0.5},
            ARC_CHALLENGE: {"metric": "accuracy:LogprobScorer", "score": 2 / 6},
            ARC_SUITE: {"metric": "primary_score:average", "score": (0.5 + 2 / 6) / 2},
        },
    )
    if with_manifest:
        meta = {"accuracy": {"higher_is_better": True, "display_format": "percent", "unit": None}}
        write_manifest(
            out,
            tasks={
                ARC_EASY: {
                    "base_task": "arc_easy",
                    "metric_meta": meta,
                    "suites": [ARC_SUITE],
                    "error": None,
                },
                ARC_CHALLENGE: {
                    "base_task": "arc_challenge",
                    "metric_meta": meta,
                    "suites": [ARC_SUITE],
                    "error": None,
                },
            },
            suites=[
                {
                    "name": ARC_SUITE,
                    "aggregation": "average",
                    "parent": None,
                    "description": None,
                    "children": [
                        {"type": "task", "name": ARC_CHALLENGE},
                        {"type": "task", "name": ARC_EASY},
                    ],
                    "metrics": {"primary_score": {"average": (0.5 + 2 / 6) / 2}},
                    "primary_metric": "primary_score:average",
                    "score": (0.5 + 2 / 6) / 2,
                    "num_tasks": 2,
                }
            ],
        )
    return out


def write_inference(out: Path, batches: int = 3, with_requests: bool = True) -> None:
    lines = []
    for b in range(batches):
        lines.append(
            {
                "type": "batch",
                "data": {
                    "total_requests": 8,
                    "successful_requests": 8,
                    "failed_requests": 0,
                    "total_prompt_tokens": 800,
                    "total_completion_tokens": 80,
                    "wall_clock_time_s": 2.0,
                    "output_tokens_per_second": 40.0,
                    "mean_latency_s": 0.25,
                    "timestamp": f"2026-10-01T09:59:0{b}+00:00",
                    "gpu_summary": {"device_count": 1, "avg_utilization_pct": 50.0},
                    "gpu_devices": [
                        {
                            "device_id": 0,
                            "name": "NVIDIA H100 80GB HBM3",
                            "memory_total_mb": 81559.0,
                            "samples": [
                                {
                                    "utilization_pct": 40.0 + s,
                                    "memory_used_mb": 70000.0,
                                    "timestamp": f"2026-10-01T09:59:0{b}.{s}00000+00:00",
                                }
                                for s in range(3)
                            ],
                        }
                    ],
                    "requests": [
                        {
                            "end_to_end_latency_s": 0.1 * (r + 1),
                            "time_to_first_token_s": 0.01 * (r + 1),
                            "time_per_output_token_s": None,
                        }
                        for r in range(4)
                    ]
                    if with_requests
                    else [],
                    "experiment_id": RUN_ID,
                },
            }
        )
    write_jsonl(out / "metrics" / f"vllm_server_{MODEL_DIR}-inference.jsonl", lines)


def md5_b64(data: bytes) -> str:
    return base64.b64encode(hashlib.md5(data).digest()).decode()


class StaticTokens:
    def __init__(self, token: str = "test-token") -> None:
        self.value = token

    def token(self) -> str:
        return self.value


class FakeIngest:
    """In-memory ingest service and GCS bucket behind httpx.MockTransport.

    Every request body is validated against the contract schema; violations are
    collected in ``schema_errors``. A task result sent again with the same body is
    reported as unchanged. ``fail`` maps "METHOD /path" to a list of status
    codes returned (in order) before the real handler runs; ``gcs_fail`` does the
    same for signed uploads keyed by artifact path.
    """

    def __init__(self) -> None:
        self.validator = PayloadValidator(CONTRACT_SCHEMA)
        self.calls: list[tuple[str, str]] = []
        self.tokens: list[str | None] = []
        self.run_secrets: list[str | None] = []
        self.schema_errors: list[str] = []
        self.runs: dict[str, dict[str, Any]] = {}
        self.task_results: dict[int, dict[str, Any]] = {}
        self.instances: dict[int, list[dict[str, Any]]] = defaultdict(list)
        self.objects: dict[str, bytes] = {}
        self.inference: dict[str, Any] | None = None
        self.completed: dict[str, Any] | None = None
        self.sign_requests = 0
        self.fail: dict[str, list[int]] = {}
        self.gcs_fail: dict[str, list[int]] = {}
        self.error_body: dict[str, Any] = {}

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def _check(self, definition: str, body: Any) -> None:
        self.schema_errors.extend(self.validator.errors(definition, body))

    def _json(self, status: int, body: Any) -> httpx.Response:
        return httpx.Response(status, json=body)

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        key = f"{request.method} {path}"
        if request.url.host == "storage.test":
            return self._gcs(request)
        self.calls.append((request.method, path))
        self.tokens.append(request.headers.get("X-Olmo-Eval-Token"))
        self.run_secrets.append(request.headers.get("X-Olmo-Eval-Run-Secret"))
        queued = self.fail.get(key)
        if queued:
            status = queued.pop(0)
            body = self.error_body or {
                "error": {"code": "unavailable", "message": "try later", "request_id": "r1"}
            }
            return self._json(status, body)
        body = json.loads(request.content) if request.content else None
        if key == "GET /v1/whoami":
            return self._json(
                200,
                {
                    "email": "dev@allenai.org",
                    "principal_type": "user",
                    "expires_in": 3000,
                    "protocol_versions": [1],
                },
            )
        match = re.fullmatch(r"/v1/runs/([a-z0-9]+)", path)
        if match and request.method == "PUT":
            self._check("RunUpsertRequest", body)
            run_id = match.group(1)
            created = run_id not in self.runs
            self.runs[run_id] = body
            return self._json(
                200,
                {
                    "run_id": run_id,
                    "created": created,
                    "status": body["run"]["status"],
                    "upload_state": "uploading",
                    "gcs_prefix": f"gs://bucket/runs/{run_id}/",
                    "dashboard_url": f"https://dash.test/runs/{run_id}",
                    "uploaded_by": "dev@allenai.org",
                },
            )
        match = re.fullmatch(r"/v1/runs/([a-z0-9]+)/artifacts:sign", path)
        if match:
            self._check("SignArtifactsRequest", body)
            self.sign_requests += 1
            uploads = []
            for artifact in body["artifacts"]:
                existing = self.objects.get(artifact["path"])
                skip = existing is not None and md5_b64(existing) == artifact["md5_b64"]
                uploads.append(
                    {
                        "path": artifact["path"],
                        "gs_uri": f"gs://bucket/runs/{match.group(1)}/{artifact['path']}",
                        "skip": skip,
                        "url": None
                        if skip
                        else f"https://storage.test/{artifact['path']}?sig={self.sign_requests}",
                        "method": "PUT",
                        "headers": {}
                        if skip
                        else {
                            "Content-Type": artifact["content_type"],
                            "Content-MD5": artifact["md5_b64"],
                        },
                        "expires_at": None,
                    }
                )
            return self._json(200, {"uploads": uploads})
        match = re.fullmatch(r"/v1/runs/([a-z0-9]+)/task-results", path)
        if match:
            self._check("TaskResultIn", body)
            existing = [
                tid
                for tid, t in self.task_results.items()
                if (t["task_name"], t["task_hash"]) == (body["task_name"], body["task_hash"])
            ]
            tid = existing[0] if existing else len(self.task_results) + 1
            unchanged = bool(existing) and self.task_results[tid] == body
            if not unchanged:
                self.task_results[tid] = body
                self.instances[tid] = []
            return self._json(
                200,
                {
                    "task_result_id": tid,
                    "created": not existing,
                    "instances_cleared": False,
                    "unchanged": unchanged,
                },
            )
        match = re.fullmatch(r"/v1/task-results/(\d+)/instances", path)
        if match:
            self._check("InstanceBatchRequest", body)
            tid = int(match.group(1))
            self.instances[tid].extend(body["instances"])
            return self._json(
                200,
                {
                    "task_result_id": tid,
                    "received": len(body["instances"]),
                    "total_stored": len(self.instances[tid]),
                },
            )
        match = re.fullmatch(r"/v1/runs/([a-z0-9]+)/inference", path)
        if match:
            self._check("InferenceUploadRequest", body)
            self.inference = body
            return self._json(
                200,
                {"batches_stored": len(body["batches"]), "series_stored": len(body["series"])},
            )
        match = re.fullmatch(r"/v1/runs/([a-z0-9]+)/complete", path)
        if match:
            self._check("CompleteRequest", body)
            self.completed = body
            return self._json(
                200,
                {
                    "run_id": match.group(1),
                    "status": body["status"],
                    "upload_state": "complete",
                    "dashboard_url": f"https://dash.test/runs/{match.group(1)}",
                    "task_results": len(self.task_results),
                    "instances": sum(len(v) for v in self.instances.values()),
                    "artifacts_missing": [],
                    "warnings": [],
                },
            )
        return self._json(404, {"error": {"code": "not_found", "message": key, "request_id": "r"}})

    def _gcs(self, request: httpx.Request) -> httpx.Response:
        rel = request.url.path.lstrip("/")
        queued = self.gcs_fail.get(rel)
        if queued:
            return httpx.Response(queued.pop(0), text="<Error>ExpiredToken</Error>")
        data = request.read()
        if request.headers.get("Content-MD5") != md5_b64(data):
            return httpx.Response(400, text="<Error>BadDigest</Error>")
        if int(request.headers.get("Content-Length", -1)) != len(data):
            return httpx.Response(400, text="<Error>BadLength</Error>")
        self.objects[rel] = data
        return httpx.Response(200)
