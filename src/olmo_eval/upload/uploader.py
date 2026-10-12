"""Upload a results directory to the ingest service.

Sequence (protocol version 1):

1. ``PUT /v1/runs/{id}`` with the final run record.
2. Sign artifacts in batches of up to 100 and PUT each file to its signed URL.
3. For each task: ``POST /v1/runs/{id}/task-results``, then its instance rows in
   batches of up to 2,000.
4. ``PUT /v1/runs/{id}/inference`` when inference files exist.
5. ``POST /v1/runs/{id}/complete``.

The uploader only reads files. It never moves or deletes anything, and the public
functions never raise: a failed upload leaves the results on disk and logs the
command that retries it.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import math
import mimetypes
import os
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from olmo_eval.common.logging import get_logger
from olmo_eval.upload import provenance
from olmo_eval.upload.auth import GoogleTokenProvider, UploadAuthError
from olmo_eval.upload.client import IngestClient, IngestError, SignedUrlRejected
from olmo_eval.upload.config import UploadConfig, retry_command, validate_tags
from olmo_eval.upload.extract import batched, count_instances, find_task_file, iter_instances
from olmo_eval.upload.inference import build_inference_payload
from olmo_eval.upload.manifest import (
    MANIFEST_NAME,
    METRICS_NAME,
    RUN_SECRET_NAME,
    ManifestError,
    degraded_manifest,
    json_safe,
    read_manifest,
    read_metrics,
    run_secret,
    run_status,
    to_utc_iso,
    utc_now,
    write_manifest,
)
from olmo_eval.upload.metadata import (
    base_task_name,
    clean_metrics,
    metric_meta_for_spec,
    strip_priority,
    suites_containing,
)

logger = get_logger("upload")

PROTOCOL_VERSION = 1
SIGN_BATCH_SIZE = 100
PARALLEL_UPLOADS = 8
MAX_ARTIFACTS = 10000
MAX_ARTIFACT_BYTES = 5 * 1024**3
START_BUDGET_S = 15.0
FAILURE_BUDGET_S = 30.0
FINAL_STATUSES = ("complete", "partial", "failed")
# Clock slack when leaving out files that predate the run (see collect_artifacts).
STALE_FILE_SLACK = timedelta(minutes=1)


@dataclass
class UploadResult:
    """Outcome of an upload attempt."""

    ok: bool
    run_id: str | None = None
    dashboard_url: str | None = None
    error: str | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class TaskUpload:
    spec: str
    payload: dict[str, Any]
    predictions: Path | None
    requests: Path | None


@dataclass
class UploadPlan:
    """Everything an upload sends, built from files on disk."""

    output_dir: Path
    run_id: str
    run_request: dict[str, Any]
    tasks: list[TaskUpload]
    artifacts: list[dict[str, Any]]
    artifact_files: dict[str, Path]
    inference: dict[str, Any] | None
    complete: dict[str, Any]
    warnings: list[str] = field(default_factory=list)

    @property
    def instance_count(self) -> int:
        return sum(task.payload["instance_count"] for task in self.tasks)

    @property
    def artifact_bytes(self) -> int:
        return sum(a["size_bytes"] for a in self.artifacts)

    def iter_instance_batches(self, task: TaskUpload) -> Iterator[list[dict[str, Any]]]:
        if task.predictions is None:
            return iter(())
        rows = iter_instances(task.predictions, task.requests, task.payload["primary_metric"])
        return batched(rows)


# ---------------------------------------------------------------------- plan


def _client_info() -> dict[str, str]:
    return {"name": "olmo-eval", "version": provenance.olmo_eval_version() or "unknown"}


def _nonneg_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, float) and value.is_integer() and value >= 0:
        return int(value)
    return None


def _nonneg_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and number >= 0 else None


def _str_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def run_timing(metrics: Mapping[str, Any] | None) -> dict[str, Any]:
    """RunInfo timing fields from metrics.json, null when absent (older results)."""
    metrics = metrics or {}
    return {
        "startup_seconds": _nonneg_float(metrics.get("startup_seconds")),
        "processing_started_at": to_utc_iso(_str_or_none(metrics.get("processing_started_at"))),
        "processing_seconds": _nonneg_float(metrics.get("processing_seconds")),
    }


def _md5_b64(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return base64.b64encode(digest.digest()).decode()


def _artifact_kind(rel: str) -> str:
    parts = rel.split("/")
    name = parts[-1]
    if rel == METRICS_NAME:
        return "metrics"
    if rel == MANIFEST_NAME:
        return "manifest"
    if parts[0] == "predictions":
        return "predictions"
    if parts[0] == "requests":
        return "requests"
    if parts[0] == "metrics" and name.endswith(".jsonl"):
        return "inference_metrics"
    if parts[0] == "traces":
        return "traces"
    if parts[0] == "logs" or "logs" in parts[:-1] or name.endswith(".log"):
        return "logs"
    return "other"


def _content_type(rel: str) -> str:
    if rel.endswith(".jsonl"):
        return "application/x-ndjson"
    if rel.endswith(".json"):
        return "application/json"
    if rel.endswith((".log", ".txt")):
        return "text/plain"
    guessed, _ = mimetypes.guess_type(rel)
    return guessed or "application/octet-stream"


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def collect_artifacts(
    output_dir: Path,
    task_files: Mapping[str, str],
    warnings: list[str],
    since: datetime | None = None,
    skip: frozenset[str] = frozenset(),
) -> tuple[list[dict[str, Any]], dict[str, Path]]:
    """ArtifactIn entries for the regular files under the output directory.

    An output directory can be reused across runs (the default is /tmp/results/). When
    ``since`` is given, files last modified before it are left out, except metrics.json,
    manifest.json and the files the run's tasks reference. Paths in ``skip`` are never
    included.
    """
    artifacts: list[dict[str, Any]] = []
    files: dict[str, Path] = {}
    always = {METRICS_NAME, MANIFEST_NAME, *task_files}
    cutoff = (since - STALE_FILE_SLACK).timestamp() if since is not None else None
    stale = 0
    for root, dirs, names in os.walk(output_dir):
        dirs.sort()
        for name in sorted(names):
            path = Path(root) / name
            if (
                path.is_symlink()
                or not path.is_file()
                or name.endswith(".json.tmp")
                or name == RUN_SECRET_NAME
            ):
                continue
            rel = path.relative_to(output_dir).as_posix()
            if rel in skip:
                continue
            stat = path.stat()
            if cutoff is not None and rel not in always and stat.st_mtime < cutoff:
                stale += 1
                continue
            size = stat.st_size
            if size > MAX_ARTIFACT_BYTES:
                warnings.append(f"Skipped {rel}: larger than 5 GiB")
                continue
            if len(artifacts) >= MAX_ARTIFACTS:
                warnings.append(f"More than {MAX_ARTIFACTS} files; the rest were not uploaded")
                return artifacts, files
            artifacts.append(
                {
                    "path": rel,
                    "size_bytes": size,
                    "md5_b64": _md5_b64(path),
                    "content_type": _content_type(rel),
                    "kind": _artifact_kind(rel),
                    "task_name": task_files.get(rel),
                }
            )
            files[rel] = path
    if stale:
        warnings.append(
            f"Skipped {stale} file(s) in {output_dir} last modified before this run started"
        )
    return artifacts, files


def _task_payload(
    output_dir: Path,
    row: Mapping[str, Any],
    manifest_task: Mapping[str, Any] | None,
    run_errors: Mapping[str, str],
    summary: Mapping[str, Any],
    model_path: str | None = None,
    since: datetime | None = None,
    warnings: list[str] | None = None,
) -> TaskUpload:
    spec = str(row["task"])
    raw_config = row.get("config")
    config: Mapping[str, Any] = raw_config if isinstance(raw_config, Mapping) else {}
    task_hash = row.get("task_hash")
    if not isinstance(task_hash, str) or not task_hash:
        task_hash = hashlib.sha256(spec.encode()).hexdigest()[:16]
    raw_metrics = row.get("metrics")
    metrics: Mapping[str, Any] = raw_metrics if isinstance(raw_metrics, Mapping) else {}

    cutoff = (since - STALE_FILE_SLACK).timestamp() if since is not None else None
    raw_hash = row.get("task_hash")
    file_hash = raw_hash if isinstance(raw_hash, str) and raw_hash else None
    predictions = find_task_file(
        output_dir, "predictions", spec, file_hash, model_path, cutoff, warnings
    )
    requests = find_task_file(output_dir, "requests", spec, file_hash, model_path, cutoff, warnings)

    if manifest_task is not None:
        metric_meta = dict(manifest_task.get("metric_meta") or {})
        suites = list(manifest_task.get("suites") or [])
        base_task = manifest_task.get("base_task")
        error = manifest_task.get("error") or run_errors.get(spec)
    else:
        metric_meta = metric_meta_for_spec(spec, list(metrics))
        suites = suites_containing(spec)
        base_task = base_task_name(spec)
        error = run_errors.get(spec)

    primary_metric = row.get("primary_metric")
    if not primary_metric:
        entry = summary.get(spec)
        primary_metric = entry.get("metric") if isinstance(entry, Mapping) else None

    error_summary = row.get("error_summary")
    if isinstance(error_summary, str):
        error_summary = {"summary": error_summary}
    elif not isinstance(error_summary, Mapping):
        error_summary = None

    split = config.get("split")
    duration = row.get("duration_seconds")
    payload = {
        "task_name": strip_priority(spec),
        "task_hash": task_hash,
        "base_task": base_task,
        "primary_metric": primary_metric if isinstance(primary_metric, str) else None,
        "metrics": clean_metrics(metrics),
        "metric_meta": metric_meta,
        "config": json_safe(dict(config)),
        "num_fewshot": _nonneg_int(config.get("num_fewshot")),
        "limit": _nonneg_int(config.get("limit")),
        "split": split if isinstance(split, str) else None,
        "num_instances": _nonneg_int(row.get("num_instances")) or 0,
        "instances_processed": _nonneg_int(row.get("instances_processed")),
        "instances_failed": _nonneg_int(row.get("instances_failed")),
        "error": str(error) if error else None,
        "error_summary": json_safe(dict(error_summary)) if error_summary else None,
        "duration_seconds": _nonneg_float(duration),
        "first_request_at": to_utc_iso(_str_or_none(row.get("first_request_at"))),
        "last_completed_at": to_utc_iso(_str_or_none(row.get("last_completed_at"))),
        "prompt_tokens_total": _nonneg_int(row.get("prompt_tokens_total")),
        "completion_tokens_total": _nonneg_int(row.get("completion_tokens_total")),
        "attributed_inference_seconds": _nonneg_float(row.get("attributed_inference_seconds")),
        "predictions_path": predictions.relative_to(output_dir).as_posix() if predictions else None,
        "requests_path": requests.relative_to(output_dir).as_posix() if requests else None,
        "suites": suites,
        "instance_count": count_instances(predictions) if predictions else 0,
    }
    return TaskUpload(spec, payload, predictions, requests)


def load_manifest(output_dir: Path, metrics: Mapping[str, Any] | None) -> dict[str, Any]:
    manifest = read_manifest(output_dir)
    if manifest is not None:
        return manifest
    if metrics is None:
        raise ManifestError(f"{output_dir} has neither {MANIFEST_NAME} nor {METRICS_NAME}")
    return degraded_manifest(output_dir, metrics)


def build_plan(output_dir: str | Path, tags: Sequence[str] = ()) -> UploadPlan:
    """Read a results directory and build every payload except instance rows.

    Raises:
        ManifestError: If the directory cannot be identified as a run.
    """
    out = Path(output_dir)
    if not out.is_dir():
        raise ManifestError(f"{out} is not a directory")
    metrics = read_metrics(out)
    manifest = load_manifest(out, metrics)
    run = dict(manifest.get("run") or {})
    model = dict(manifest.get("model") or {})
    run_id = run.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ManifestError(f"{out / MANIFEST_NAME} has no run_id")
    warnings: list[str] = []
    skip: frozenset[str] = frozenset()
    metrics_run = metrics.get("experiment_id") if metrics is not None else None
    if metrics_run is not None and metrics_run != run_id:
        # A reused directory: this run stopped before writing metrics.json, and the file
        # left there belongs to an earlier run.
        warnings.append(f"Ignored {METRICS_NAME}: it belongs to run {metrics_run}, not {run_id}")
        metrics = None
        skip = frozenset({METRICS_NAME})
    run["tags"] = list(validate_tags([*(run.get("tags") or []), *tags]))
    run.update(run_timing(metrics))

    manifest_tasks = manifest.get("tasks") or {}
    run_errors = {
        str(e["task"]): str(e["error"])
        for e in (metrics or {}).get("errors") or []
        if isinstance(e, Mapping) and e.get("task") and e.get("error")
    }
    summary = (metrics or {}).get("summary") or {}
    started_at = _parse_time(run.get("started_at"))
    model_path = model.get("path") if isinstance(model.get("path"), str) else None
    tasks: list[TaskUpload] = []
    for row in (metrics or {}).get("tasks") or []:
        if not isinstance(row, Mapping) or not row.get("task"):
            continue
        spec = str(row["task"])
        tasks.append(
            _task_payload(
                out,
                row,
                manifest_tasks.get(spec),
                run_errors,
                summary,
                model_path=model_path,
                since=started_at,
                warnings=warnings,
            )
        )

    status = run.get("status")
    if status not in FINAL_STATUSES:
        errors = {task.spec: task.payload["error"] for task in tasks}
        status = run_status(errors, []) if tasks else "failed"
        run["status"] = status
    if not run.get("finished_at") and metrics is not None:
        run["finished_at"] = utc_now()

    task_files: dict[str, str] = {}
    for task in tasks:
        for key in ("predictions_path", "requests_path"):
            if task.payload[key]:
                task_files[task.payload[key]] = task.payload["task_name"]
    artifacts, artifact_files = collect_artifacts(
        out, task_files, warnings, since=started_at, skip=skip
    )

    inference = None
    try:
        model_names = [str(n) for n in (model.get("name"), model.get("path")) if n]
        inference = build_inference_payload(out, run.get("started_at"), model_names, warnings)
    except Exception as e:
        warnings.append(f"Skipped inference metrics: {e}")

    run_request = {
        "protocol_version": PROTOCOL_VERSION,
        "client": _client_info(),
        "run": run,
        "model": model,
    }
    complete = {
        "status": status,
        "finished_at": run.get("finished_at"),
        "duration_seconds": run.get("duration_seconds"),
        "suites": list(manifest.get("suites") or []),
        "expected_task_results": len(tasks),
        "expected_artifacts": len(artifacts),
    }
    return UploadPlan(
        output_dir=out,
        run_id=run_id,
        run_request=run_request,
        tasks=tasks,
        artifacts=artifacts,
        artifact_files=artifact_files,
        inference=inference,
        complete=complete,
        warnings=warnings,
    )


# ---------------------------------------------------------------------- protocol


def make_client(config: UploadConfig, deadline_s: float | None = None) -> IngestClient:
    return IngestClient(
        config.api_url,
        GoogleTokenProvider(),
        deadline_s=config.timeout_s if deadline_s is None else deadline_s,
    )


def _upload_artifacts(client: IngestClient, plan: UploadPlan) -> list[str]:
    """Sign and upload every artifact. Returns failure messages."""
    failures: list[str] = []
    by_path = {a["path"]: a for a in plan.artifacts}

    def upload_one(signed: Mapping[str, Any]) -> str | None:
        path = signed["path"]
        local = plan.artifact_files[path]
        try:
            size = by_path[path]["size_bytes"]
            try:
                client.put_signed(signed["url"], dict(signed.get("headers") or {}), local, size)
            except SignedUrlRejected:
                resigned = client.sign_artifacts(plan.run_id, {"artifacts": [by_path[path]]})
                fresh = (resigned.get("uploads") or [{}])[0]
                if fresh.get("skip"):
                    return None
                client.put_signed(fresh["url"], dict(fresh.get("headers") or {}), local, size)
        except (IngestError, OSError, KeyError) as e:
            return f"{path}: {e}"
        return None

    with ThreadPoolExecutor(max_workers=PARALLEL_UPLOADS) as pool:
        for start in range(0, len(plan.artifacts), SIGN_BATCH_SIZE):
            batch = plan.artifacts[start : start + SIGN_BATCH_SIZE]
            response = client.sign_artifacts(plan.run_id, {"artifacts": batch})
            pending = [u for u in response.get("uploads") or [] if not u.get("skip")]
            failures.extend(msg for msg in pool.map(upload_one, pending) if msg)
    return failures


def _send(client: IngestClient, plan: UploadPlan) -> UploadResult:
    warnings = list(plan.warnings)
    response = client.upsert_run(plan.run_id, plan.run_request)
    dashboard_url = response.get("dashboard_url")

    artifact_failures = _upload_artifacts(client, plan)

    for task in plan.tasks:
        created = client.upsert_task_result(plan.run_id, task.payload)
        if created.get("unchanged"):
            continue
        task_result_id = created["task_result_id"]
        for index, rows in enumerate(plan.iter_instance_batches(task)):
            client.post_instances(task_result_id, {"batch_index": index, "instances": rows})

    if plan.inference is not None:
        client.put_inference(plan.run_id, plan.inference)

    completed = client.complete(plan.run_id, plan.complete)
    dashboard_url = completed.get("dashboard_url") or dashboard_url
    warnings.extend(str(w) for w in completed.get("warnings") or [])
    missing = completed.get("artifacts_missing") or []
    if missing:
        warnings.append(f"{len(missing)} artifact(s) missing in GCS: {', '.join(missing[:5])}")
    if artifact_failures:
        return UploadResult(
            ok=False,
            run_id=plan.run_id,
            dashboard_url=dashboard_url,
            error=f"{len(artifact_failures)} file(s) failed to upload: "
            + "; ".join(artifact_failures[:3]),
            warnings=warnings,
        )
    return UploadResult(ok=True, run_id=plan.run_id, dashboard_url=dashboard_url, warnings=warnings)


def _log_failure(output_dir: Path, config: UploadConfig, error: str) -> None:
    location = os.path.abspath(output_dir)
    logger.error(f"Results upload failed: {error}")
    logger.error(
        f"Results are saved in {location}. Re-upload with: {retry_command(location, config)}"
    )


def upload_results_dir(
    output_dir: str | Path,
    config: UploadConfig,
    *,
    client: IngestClient | None = None,
) -> UploadResult:
    """Upload a results directory. Never raises."""
    out = Path(output_dir)
    owned = client is None
    try:
        plan = build_plan(out, config.tags)
        if client is None:
            client = make_client(config)
        client.run_secret = run_secret(out)
        logger.info(
            f"Uploading run {plan.run_id} to {config.api_url}: {len(plan.tasks)} task(s), "
            f"{plan.instance_count} instance(s), {len(plan.artifacts)} file(s)"
        )
        result = _send(client, plan)
    except (UploadAuthError, IngestError, ManifestError) as e:
        result = UploadResult(ok=False, error=str(e))
    except Exception as e:
        result = UploadResult(ok=False, error=f"{type(e).__name__}: {e}")
    finally:
        if owned and client is not None:
            client.close()

    for warning in result.warnings:
        logger.warning(f"Upload warning: {warning}")
    if result.ok:
        logger.info(f"Results uploaded: {result.dashboard_url or result.run_id}")
    else:
        if result.dashboard_url:
            logger.error(f"Partial upload visible at {result.dashboard_url}")
        _log_failure(out, config, result.error or "unknown error")
    return result


def register_run_start(
    output_dir: str | Path,
    config: UploadConfig,
    *,
    client: IngestClient | None = None,
) -> bool:
    """Tell the ingest service a run started. Best effort, 15 s budget, never raises."""
    if not config.enabled:
        return False
    owned = client is None
    try:
        manifest = read_manifest(output_dir)
        if manifest is None:
            return False
        run = dict(manifest["run"])
        run["tags"] = list(validate_tags([*(run.get("tags") or []), *config.tags]))
        if client is None:
            client = make_client(config, deadline_s=START_BUDGET_S)
        client.run_secret = run_secret(output_dir)
        client.whoami()
        response = client.upsert_run(
            run["run_id"],
            {
                "protocol_version": PROTOCOL_VERSION,
                "client": _client_info(),
                "run": run,
                "model": manifest["model"],
            },
        )
        logger.info(f"Registered run {run['run_id']}: {response.get('dashboard_url', '')}")
        return True
    except Exception as e:
        hint = "" if isinstance(e, UploadAuthError) else " (results will still be saved locally)"
        logger.warning(f"Could not register the run with {config.api_url}: {e}{hint}")
        return False
    finally:
        if owned and client is not None:
            client.close()


def mark_run_failed(
    output_dir: str | Path,
    config: UploadConfig,
    error: str,
    *,
    since: datetime | None = None,
    client: IngestClient | None = None,
) -> UploadResult:
    """Record a crash: rewrite the manifest as failed and report it. Never raises.

    Only acts when manifest.json exists with status "running" (the run crashed before
    it could finalize) and, when ``since`` is given, the run started after it, so a
    manifest left by an earlier run in a reused directory is not reported. 30 s budget.
    """
    out = Path(output_dir)
    owned = client is None
    try:
        manifest = read_manifest(out)
        if manifest is None or manifest.get("run", {}).get("status") != "running":
            return UploadResult(ok=False, error="no running manifest")
        run = manifest["run"]
        started = _parse_time(run.get("started_at"))
        if since is not None and (started is None or started < since - STALE_FILE_SLACK):
            return UploadResult(ok=False, error="manifest is from an earlier run")
        finished = datetime.now(UTC)
        run["status"] = "failed"
        run["finished_at"] = finished.isoformat()
        started = run.get("started_at")
        if started:
            with contextlib.suppress(ValueError):
                run["duration_seconds"] = (
                    finished - datetime.fromisoformat(started)
                ).total_seconds()
        run["errors"] = [*(run.get("errors") or []), {"task": None, "error": error[:4000]}]
        write_manifest(out, manifest)
        if not config.enabled:
            return UploadResult(ok=False, run_id=run["run_id"], error="upload disabled")

        run_upload = dict(run)
        run_upload["tags"] = list(validate_tags([*(run.get("tags") or []), *config.tags]))
        if client is None:
            client = make_client(config, deadline_s=FAILURE_BUDGET_S)
        client.run_secret = run_secret(out)
        client.upsert_run(
            run["run_id"],
            {
                "protocol_version": PROTOCOL_VERSION,
                "client": _client_info(),
                "run": run_upload,
                "model": manifest["model"],
            },
        )
        completed = client.complete(
            run["run_id"],
            {
                "status": "failed",
                "finished_at": run["finished_at"],
                "duration_seconds": run.get("duration_seconds"),
                "suites": [],
                "expected_task_results": 0,
                "expected_artifacts": 0,
            },
        )
        url = completed.get("dashboard_url")
        logger.info(f"Recorded failed run {run['run_id']}: {url or ''}")
        return UploadResult(ok=True, run_id=run["run_id"], dashboard_url=url)
    except Exception as e:
        logger.warning(f"Could not record the failed run with {config.api_url}: {e}")
        return UploadResult(ok=False, error=str(e))
    finally:
        if owned and client is not None:
            client.close()


@dataclass
class DryRunReport:
    """What an upload would send, with any contract violations."""

    plan: UploadPlan
    instances: int
    errors: list[str]
    used_schema: bool
    #: Why validation fell back to required keys only, or None with the full schema.
    degraded_reason: str | None = None


def dry_run(output_dir: str | Path, tags: Sequence[str] = ()) -> DryRunReport:
    """Build and validate every payload without network calls.

    Raises:
        ManifestError: If the directory cannot be identified as a run.
    """
    from olmo_eval.upload.validation import PayloadValidator

    validator = PayloadValidator()
    plan = build_plan(output_dir, tags)
    errors = list(validator.errors("RunUpsertRequest", plan.run_request))
    for start in range(0, len(plan.artifacts), SIGN_BATCH_SIZE):
        batch = {"artifacts": plan.artifacts[start : start + SIGN_BATCH_SIZE]}
        errors += validator.errors("SignArtifactsRequest", batch)
    instances = 0
    for task in plan.tasks:
        errors += validator.errors("TaskResultIn", task.payload)
        sent = 0
        for index, rows in enumerate(plan.iter_instance_batches(task)):
            sent += len(rows)
            errors += validator.errors(
                "InstanceBatchRequest", {"batch_index": index, "instances": rows}
            )
        if sent != task.payload["instance_count"]:
            errors.append(
                f"{task.spec}: instance_count {task.payload['instance_count']} != rows {sent}"
            )
        instances += sent
    if plan.inference is not None:
        errors += validator.errors("InferenceUploadRequest", plan.inference)
    errors += validator.errors("CompleteRequest", plan.complete)
    return DryRunReport(plan, instances, errors, validator.uses_schema, validator.degraded_reason)
