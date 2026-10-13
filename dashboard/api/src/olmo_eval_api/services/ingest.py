"""Ingest operations: run upsert, artifacts, task results, instances, inference, complete.

Every operation is idempotent (spec 2.4). ``complete_run`` recomputes all derived values from
stored rows (spec 2.5), so calling it twice gives the same result.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast

import numpy as np
from sqlalchemy import delete, func, literal_column, select, text, tuple_
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from olmo_eval_api.auth.google_token import Principal
from olmo_eval_api.db.models import (
    Artifact,
    Run,
    SuiteResult,
    TaskResult,
    TaskResultVector,
    artifacts_t,
    inference_batches_t,
    instance_results_t,
    models_t,
    run_inference_t,
    runs_t,
    suite_defs_t,
    suite_results_t,
    task_result_vectors_t,
    task_results_t,
    task_variants_t,
)
from olmo_eval_api.errors import bad_request, conflict, forbidden, not_found
from olmo_eval_api.schemas import ingest as s
from olmo_eval_api.services import derive
from olmo_eval_api.services.common import detect_scale, latest_by_task, now_utc
from olmo_eval_api.services.search_text import build_search_text
from olmo_eval_api.services.suites import (
    build_tree,
    current_suite_defs,
    leaf_weights,
    leaves,
    propagate_stderr,
    suite_definition_hash,
)
from olmo_eval_api.settings import Settings
from olmo_eval_api.storage.base import Storage

logger = logging.getLogger(__name__)

FINAL_STATUSES = {"complete", "partial", "failed"}
MAX_TASK_RESULTS_PER_RUN = 2000
MAX_INSTANCES_PER_TASK_RESULT = 1_000_000
MAX_ARTIFACTS_PER_RUN = 10_000
# Each artifact is at most schemas.ingest.MAX_ARTIFACT_BYTES (5 GiB), and the signed upload
# URL accepts only the declared size.
MAX_ARTIFACT_BYTES_PER_RUN = 50 * 1024**3
MAX_INSTANCES_PER_RUN = 10_000_000
_INSTANCE_CHUNK = 500
_BATCH_CHUNK = 1000
SIGN_CONCURRENCY = 16


async def get_run(session: AsyncSession, run_id: str, *, lock: bool = False) -> Run:
    stmt = select(Run).where(runs_t.c.run_id == run_id)
    if lock:
        stmt = stmt.with_for_update()
    run = (await session.execute(stmt)).scalars().first()
    if run is None:
        raise not_found(f"Run {run_id} not found")
    return run


async def _count(session: AsyncSession, stmt: Any) -> int:
    return int((await session.execute(stmt)).scalar_one())


# ---------------------------------------------------------------------------
# PUT /v1/runs/{run_id}
# ---------------------------------------------------------------------------


def model_values(model: s.ModelInfo, ts: datetime) -> dict[str, Any]:
    model_hash = model.model_hash or derive.fallback_model_hash(model.provider_config)
    series = model.series or derive.derive_series(model.name)
    step = model.step
    if step is None:
        step = derive.derive_step(model.revision, model.name, model.path)
    tokens = model.tokens_seen
    if tokens is None:
        tokens = derive.derive_tokens(model.revision, model.name, model.path)
    return {
        "model_id": derive.model_id(model.name, model_hash),
        "name": model.name,
        "model_hash": model_hash,
        "path": model.path,
        "revision": model.revision,
        "provider_kind": model.provider_kind,
        "provider_config": model.provider_config,
        "settings_hash": derive.settings_hash(model.provider_config),
        "series": series,
        "series_label": derive.series_label(series),
        "family": model.family or derive.derive_family(model.name, model.path),
        "step": step,
        "tokens_seen": tokens,
        "first_seen_at": ts,
        "last_seen_at": ts,
    }


def _run_fields(req: s.RunUpsertRequest) -> dict[str, Any]:
    run = req.run
    beaker = run.beaker
    return {
        "launch_id": run.launch_id,
        "model_name": req.model.name,
        "experiment_name": run.experiment_name,
        "experiment_group": run.experiment_group,
        "author": run.author,
        "tags": list(dict.fromkeys(run.tags)),
        "client_tags": list(dict.fromkeys(run.tags)),
        "started_at": run.started_at,
        "finished_at": run.finished_at,
        "duration_seconds": run.duration_seconds,
        "olmo_eval_version": run.olmo_eval_version,
        "git_repo": run.git.repo,
        "git_commit": run.git.commit,
        "git_branch": run.git.branch,
        "git_dirty": run.git.dirty,
        "beaker_experiment_id": beaker.experiment_id if beaker else None,
        "beaker_job_id": beaker.job_id if beaker else None,
        "beaker_result_dataset_id": beaker.result_dataset_id if beaker else None,
        "beaker_workspace": beaker.workspace if beaker else None,
        "beaker": beaker.model_dump(mode="json") if beaker else None,
        "environment": run.environment.model_dump(mode="json"),
        "argv": run.argv,
        "task_specs": run.task_specs,
        "output_dir": run.output_dir,
        "harness_config": run.harness_config,
        "provider_init_seconds": run.provider_init_seconds,
        "startup_seconds": derive.finite_or_none(run.startup_seconds),
        "processing_started_at": run.processing_started_at,
        "processing_seconds": derive.finite_or_none(run.processing_seconds),
        "errors": [e.model_dump(mode="json") for e in run.errors],
        "client": req.client.model_dump(mode="json"),
    }


async def upsert_run(
    session: AsyncSession,
    settings: Settings,
    run_id: str,
    req: s.RunUpsertRequest,
    principal: Principal,
    run_secret: str | None = None,
) -> s.RunUpsertResponse:
    if req.run.run_id != run_id:
        raise bad_request(f"body run.run_id {req.run.run_id!r} does not match the URL {run_id!r}")
    ts = now_utc()
    mvalues = model_values(req.model, ts)
    mid = mvalues["model_id"]

    model_stmt = pg_insert(models_t).values(**mvalues)
    await session.execute(
        model_stmt.on_conflict_do_update(
            index_elements=["model_id"],
            set_={
                k: model_stmt.excluded[k] for k in mvalues if k not in ("model_id", "first_seen_at")
            },
        )
    )
    fields = _run_fields(req)
    inserted = (
        await session.execute(
            pg_insert(runs_t)
            .values(
                run_id=run_id,
                model_id=mid,
                status=req.run.status,
                upload_state="uploading",
                uploaded_by=principal.email,
                write_secret_hash=hash_run_secret(run_secret) if run_secret else None,
                created_at=run_time(req.run.started_at, ts),
                updated_at=ts,
                gcs_prefix=settings.gcs_prefix(run_id),
                search_text="",
                **fields,
            )
            .on_conflict_do_nothing(index_elements=["run_id"])
            .returning(runs_t.c.run_id)
        )
    ).first()
    created = inserted is not None
    run = await get_run(session, run_id, lock=True)
    if not created and run.model_id != mid:
        stored_model = run.model_name
        await session.rollback()
        raise conflict(
            f"Run {run_id} was uploaded with a different model ({stored_model}); "
            "a run ID belongs to one model"
        )
    if not created and not can_write(principal, run, run_secret):
        await session.rollback()
        raise forbidden(write_denied_message(run_id))
    if not created and run.write_secret_hash is None and run_secret:
        run.write_secret_hash = hash_run_secret(run_secret)
    if not created:
        downgrade = run.status in FINAL_STATUSES and req.run.status == "running"
        if not downgrade:
            fields["tags"] = merge_tags(run.tags, run.client_tags, fields["client_tags"])
            if not may_change_author(principal, run.uploaded_by):
                # can_delete trusts the author of service-account uploads, so only the user
                # who uploaded a run may change it.
                fields.pop("author")
            for key, value in fields.items():
                setattr(run, key, value)
            run.status = req.run.status
            run.updated_at = ts
            created_at = run_time(run.started_at, run.created_at)
            if created_at != run.created_at:
                run.created_at = created_at
                await session.execute(
                    task_results_t.update()
                    .where(task_results_t.c.run_id == run_id)
                    .values(run_created_at=created_at)
                )
    run.search_text = build_search_text(run)
    response = s.RunUpsertResponse(
        run_id=run_id,
        created=created,
        status=cast(s.RunStatus, run.status),
        upload_state=cast(s.UploadState, run.upload_state),
        gcs_prefix=run.gcs_prefix,
        dashboard_url=settings.dashboard_run_url(run_id),
        uploaded_by=run.uploaded_by,
    )
    await session.commit()
    logger.info(
        "run upserted",
        extra={"run_id": run_id, "run_created": created, "run_status": response.status},
    )
    return response


def merge_tags(stored: Sequence[str], sent_before: Sequence[str], sent: Sequence[str]) -> list[str]:
    """A run's tags after a re-upload.

    The stored tags include edits made in the dashboard, which the client does not know about.
    Only tags the client had not sent before are added, so a tag removed in the dashboard
    stays removed.
    """
    previous = set(sent_before)
    return list(dict.fromkeys([*stored, *(tag for tag in sent if tag not in previous)]))


def run_time(started_at: datetime | None, received_at: datetime) -> datetime:
    """When a run happened: its start, unless that is missing or after ``received_at``.

    Runs are ordered by this time, so a run uploaded long after it ran (a failed start
    registration, a backfill) does not count as the newest result.
    """
    if started_at is None:
        return received_at
    if started_at.tzinfo is None:
        started_at = started_at.replace(tzinfo=UTC)
    return min(started_at, received_at)


def may_change_author(principal: Principal, uploaded_by: str) -> bool:
    """Whether a re-upload may change a run's author.

    Only a user re-uploading their own run may. Service-account uploads keep the author from
    the first upload: every Beaker job shares the uploader account, so any job could
    otherwise rename another job's run to its own user and then delete it.
    """
    return principal.principal_type == "user" and principal.email == uploaded_by


def hash_run_secret(secret: str) -> str:
    return hashlib.sha256(secret.encode()).hexdigest()


def can_write(principal: Principal, run: Run, run_secret: str | None) -> bool:
    """Whether a principal may change an existing run.

    The users who may delete a run may also change it. Anyone else, including the shared
    uploader service account, must present the run's write secret, which the client that
    first uploaded the run keeps in its results directory.
    """
    if can_delete(principal, run.uploaded_by, run.author):
        return True
    stored = run.write_secret_hash
    if not stored or not run_secret:
        return False
    return hmac.compare_digest(stored, hash_run_secret(run_secret))


def write_denied_message(run_id: str) -> str:
    return (
        f"run {run_id} can only be changed by the person who uploaded or launched it, "
        "or from its original results directory"
    )


async def require_write(
    session: AsyncSession, run_id: str, principal: Principal, run_secret: str | None
) -> None:
    """Raise unless the principal may change the run (see ``can_write``)."""
    run = await get_run(session, run_id)
    if not can_write(principal, run, run_secret):
        raise forbidden(write_denied_message(run_id))


async def require_task_result_write(
    session: AsyncSession, task_result_id: int, principal: Principal, run_secret: str | None
) -> None:
    """Raise unless the principal may change the run that owns the task result."""
    run_id = (
        await session.execute(
            select(task_results_t.c.run_id).where(task_results_t.c.id == task_result_id)
        )
    ).scalar_one_or_none()
    if run_id is None:
        raise not_found(f"Task result {task_result_id} not found")
    await require_write(session, run_id, principal, run_secret)


def can_delete(principal: Principal, uploaded_by: str, author: str | None) -> bool:
    """Whether a principal may delete a run.

    Service accounts never delete: every Beaker job shares the uploader account, so
    letting it delete would let any job delete any other job's run. A user may delete
    runs they uploaded, and runs a service account uploaded on their behalf, which
    record their Beaker username as the author. ``uploaded_by`` never changes after the
    first upload, and for service-account uploads neither does ``author``
    (``may_change_author``), so a requester cannot change either value to pass this check.
    """
    if principal.principal_type == "service_account":
        return False
    if principal.email == uploaded_by:
        return True
    username = principal.email.split("@", 1)[0]
    return uploaded_by.endswith(".gserviceaccount.com") and author == username


async def delete_run(
    session: AsyncSession, storage: Storage, settings: Settings, run_id: str, principal: Principal
) -> None:
    run = await get_run(session, run_id, lock=True)
    if not can_delete(principal, run.uploaded_by, run.author):
        raise forbidden(f"only the person who uploaded or launched run {run_id} can delete it")
    deleted = await storage.delete_prefix(settings.object_prefix(run_id))
    await session.execute(delete(runs_t).where(runs_t.c.run_id == run_id))
    await session.commit()
    logger.info("run deleted", extra={"run_id": run_id, "objects_deleted": deleted})


# ---------------------------------------------------------------------------
# POST /v1/runs/{run_id}/artifacts:sign
# ---------------------------------------------------------------------------


async def sign_artifacts(
    session: AsyncSession,
    storage: Storage,
    settings: Settings,
    run_id: str,
    req: s.SignArtifactsRequest,
) -> s.SignArtifactsResponse:
    await get_run(session, run_id)
    ac = artifacts_t.c
    known: dict[str, int] = {
        path: int(size)
        for path, size in await session.execute(
            select(ac.path, ac.size_bytes).where(ac.run_id == run_id)
        )
    }
    # The last entry for a path wins, as in the upsert below.
    requested = {a.path: a.size_bytes for a in req.artifacts}
    if len(known.keys() | requested.keys()) > MAX_ARTIFACTS_PER_RUN:
        raise bad_request(f"a run may have at most {MAX_ARTIFACTS_PER_RUN} artifacts")
    total_bytes = sum(v for k, v in known.items() if k not in requested) + sum(requested.values())
    if total_bytes > MAX_ARTIFACT_BYTES_PER_RUN:
        raise bad_request(
            f"a run's artifacts may total at most {MAX_ARTIFACT_BYTES_PER_RUN // 1024**3} GiB; "
            f"this request would bring it to {total_bytes / 1024**3:.1f} GiB"
        )

    # Check only the requested objects: listing the whole run prefix for every batch would be
    # quadratic in the number of artifacts.
    existing = await storage.stat_objects(
        settings.object_key(run_id, a.path) for a in req.artifacts
    )
    ts = now_utc()
    rows: dict[str, dict[str, Any]] = {}
    to_sign: list[s.ArtifactIn] = []
    for artifact in req.artifacts:
        info = existing.get(settings.object_key(run_id, artifact.path))
        skip = (
            info is not None
            and info.size == artifact.size_bytes
            and info.md5_b64 == artifact.md5_b64
        )
        rows[artifact.path] = {
            "run_id": run_id,
            "path": artifact.path,
            "kind": artifact.kind,
            "task_name": artifact.task_name,
            "size_bytes": artifact.size_bytes,
            "md5_b64": artifact.md5_b64,
            "content_type": artifact.content_type,
            "uploaded": skip,
            "created_at": ts,
            "updated_at": ts,
        }
        if not skip:
            to_sign.append(artifact)

    # Each GCS signature is an IAM signBlob call, so sign concurrently (bounded).
    limiter = asyncio.Semaphore(SIGN_CONCURRENCY)

    async def sign(artifact: s.ArtifactIn) -> tuple[str, Any]:
        async with limiter:
            key = settings.object_key(run_id, artifact.path)
            return artifact.path, await storage.sign_upload(
                key, artifact.content_type, artifact.md5_b64, artifact.size_bytes
            )

    signed_by_path = dict(await asyncio.gather(*(sign(x) for x in to_sign)))
    uploads: list[s.SignedUpload] = []
    for artifact in req.artifacts:
        gs_uri = f"gs://{settings.results_bucket}/{settings.object_key(run_id, artifact.path)}"
        signed = signed_by_path.get(artifact.path)
        if signed is None:
            uploads.append(
                s.SignedUpload(
                    path=artifact.path,
                    gs_uri=gs_uri,
                    skip=True,
                    url=None,
                    headers={},
                    expires_at=None,
                )
            )
        else:
            uploads.append(
                s.SignedUpload(
                    path=artifact.path,
                    gs_uri=gs_uri,
                    skip=False,
                    url=signed.url,
                    headers=signed.headers,
                    expires_at=signed.expires_at,
                )
            )
    stmt = pg_insert(artifacts_t).values(list(rows.values()))
    update_cols = (
        "kind",
        "task_name",
        "size_bytes",
        "md5_b64",
        "content_type",
        "uploaded",
        "updated_at",
    )
    await session.execute(
        stmt.on_conflict_do_update(
            index_elements=["run_id", "path"],
            set_={col: stmt.excluded[col] for col in update_cols},
        )
    )
    await session.commit()
    return s.SignArtifactsResponse(uploads=uploads)


# ---------------------------------------------------------------------------
# POST /v1/runs/{run_id}/task-results
# ---------------------------------------------------------------------------


async def upsert_task_result(
    session: AsyncSession, run_id: str, req: s.TaskResultIn
) -> s.TaskResultUpsertResponse:
    run = await get_run(session, run_id, lock=True)
    ts = now_utc()
    metric_meta = {k: v.model_dump(mode="json") for k, v in req.metric_meta.items()}

    variant = pg_insert(task_variants_t).values(
        task_name=req.task_name,
        task_hash=req.task_hash,
        base_task=req.base_task,
        config=req.config,
        primary_metric=req.primary_metric,
        metric_meta=metric_meta,
        num_fewshot=req.num_fewshot,
        task_limit=req.limit,
        split=req.split,
        suites=sorted(set(req.suites)),
        first_seen_at=ts,
        last_seen_at=ts,
    )
    tv = task_variants_t.c
    await session.execute(
        variant.on_conflict_do_update(
            index_elements=["task_name", "task_hash"],
            set_={
                "last_seen_at": variant.excluded.last_seen_at,
                "metric_meta": variant.excluded.metric_meta,
                "primary_metric": func.coalesce(variant.excluded.primary_metric, tv.primary_metric),
                "base_task": func.coalesce(variant.excluded.base_task, tv.base_task),
                "suites": text(
                    "ARRAY(SELECT DISTINCT x FROM unnest(task_variants.suites || excluded.suites)"
                    " AS x ORDER BY x)"
                ),
            },
        )
    )

    flat = derive.flatten_metrics(req.metrics)
    score = flat.get(req.primary_metric) if req.primary_metric else None
    client_primary = req.metric_meta.get(derive.metric_name(req.primary_metric or ""))
    fields: dict[str, Any] = {
        "model_id": run.model_id,
        "run_created_at": run.created_at,
        "primary_metric": req.primary_metric,
        "score": score,
        "stderr": None,
        "score_is_mean": False,
        "instance_scale": 1.0,
        "metric_kind": None,
        "higher_is_better": client_primary.higher_is_better if client_primary else None,
        "display_format": (client_primary.display_format if client_primary else None) or "raw",
        "metrics": flat,
        "metric_meta": {},
        "num_instances": req.num_instances,
        "instances_processed": req.instances_processed,
        "instances_failed": req.instances_failed,
        "instance_count_expected": req.instance_count,
        "instances_stored": 0,
        "error": req.error,
        "error_summary": req.error_summary,
        "duration_seconds": req.duration_seconds,
        "first_request_at": req.first_request_at,
        "last_completed_at": req.last_completed_at,
        "attributed_inference_seconds": derive.finite_or_none(req.attributed_inference_seconds),
        "reported_prompt_tokens_total": req.prompt_tokens_total,
        "reported_completion_tokens_total": req.completion_tokens_total,
        # Replaced when the run completes (finalize_task_result).
        "completion_tokens_total": req.completion_tokens_total,
        "prompt_tokens_total": req.prompt_tokens_total,
        "mean_completion_tokens": None,
        "truncation_rate": None,
        "finish_reason_counts": {},
        "predictions_path": req.predictions_path,
        "requests_path": req.requests_path,
        "updated_at": ts,
        "finalized_at": None,
    }
    fields["content_hash"] = await _content_hash(session, run_id, req)
    trc = task_results_t.c
    existing = (
        (
            await session.execute(
                select(TaskResult)
                .where(
                    trc.run_id == run_id,
                    trc.task_name == req.task_name,
                    trc.task_hash == req.task_hash,
                )
                .with_for_update()
            )
        )
        .scalars()
        .first()
    )
    if (
        existing is not None
        and existing.finalized_at is not None
        and existing.content_hash == fields["content_hash"]
        and existing.instances_stored == req.instance_count
    ):
        task_result_id = int(existing.id or 0)
        await session.commit()
        return s.TaskResultUpsertResponse(
            task_result_id=task_result_id, created=False, instances_cleared=False, unchanged=True
        )
    if existing is None:
        count = await _count(
            session, select(func.count()).select_from(task_results_t).where(trc.run_id == run_id)
        )
        if count >= MAX_TASK_RESULTS_PER_RUN:
            raise bad_request(f"a run may have at most {MAX_TASK_RESULTS_PER_RUN} task results")
        tr = TaskResult(
            run_id=run_id,
            task_name=req.task_name,
            task_hash=req.task_hash,
            created_at=ts,
            **fields,
        )
        session.add(tr)
        await session.flush()
        created, cleared = True, False
    else:
        tr = existing
        cleared = (
            await _count(
                session,
                select(func.count())
                .select_from(instance_results_t)
                .where(instance_results_t.c.task_result_id == tr.id),
            )
            > 0
        )
        await session.execute(
            delete(instance_results_t).where(instance_results_t.c.task_result_id == tr.id)
        )
        await session.execute(
            delete(task_result_vectors_t).where(task_result_vectors_t.c.task_result_id == tr.id)
        )
        for key, value in fields.items():
            setattr(tr, key, value)
        created = False
    run.updated_at = ts
    await session.flush()
    task_result_id = int(tr.id or 0)
    await session.commit()
    return s.TaskResultUpsertResponse(
        task_result_id=task_result_id, created=created, instances_cleared=cleared
    )


async def _content_hash(session: AsyncSession, run_id: str, req: s.TaskResultIn) -> str:
    """Hash of the request and the checksums of the files it points to."""
    paths = [p for p in (req.predictions_path, req.requests_path) if p]
    checksums: dict[str, str] = {}
    if paths:
        ac = artifacts_t.c
        rows = await session.execute(
            select(ac.path, ac.md5_b64).where(ac.run_id == run_id, ac.path.in_(paths))
        )
        checksums = {path: md5 for path, md5 in rows}
    body = {"request": req.model_dump(mode="json"), "files": {p: checksums.get(p) for p in paths}}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


# ---------------------------------------------------------------------------
# POST /v1/task-results/{task_result_id}/instances
# ---------------------------------------------------------------------------

_INSTANCE_UPDATE_COLS = [
    "key_hash", "doc_id", "primary_score", "metrics", "label", "extracted_answer",
    "finish_reason", "completion_tokens", "prompt_tokens", "num_outputs", "prompt_preview",
    "output_preview", "judge_verdict", "has_scoring_error", "has_execution_result",
    "has_judge_result", "has_trajectory", "pred_offset", "req_offset", "pred_length",
    "req_length",
]  # fmt: skip


def _instance_row(task_result_id: int, inst: s.InstanceIn) -> dict[str, Any]:
    metrics = {k: v for k, v in inst.metrics.items() if v is not None and math.isfinite(v)}
    score = inst.primary_score
    if score is not None and not math.isfinite(score):
        score = None
    return {
        "task_result_id": task_result_id,
        "native_id": inst.native_id,
        "key_hash": derive.instance_key_hash(inst.native_id),
        "doc_id": inst.doc_id,
        "primary_score": score,
        "metrics": metrics,
        "label": inst.label,
        "extracted_answer": inst.extracted_answer,
        "finish_reason": inst.finish_reason[:32] if inst.finish_reason else None,
        "completion_tokens": inst.completion_tokens,
        "prompt_tokens": inst.prompt_tokens,
        "num_outputs": min(inst.num_outputs, 32767),
        "prompt_preview": inst.prompt_preview,
        "output_preview": inst.output_preview,
        "judge_verdict": inst.judge_verdict,
        "has_scoring_error": inst.has_scoring_error,
        "has_execution_result": inst.has_execution_result,
        "has_judge_result": inst.has_judge_result,
        "has_trajectory": inst.has_trajectory,
        "pred_offset": inst.pred_offset,
        "req_offset": inst.req_offset,
        "pred_length": inst.pred_length,
        "req_length": inst.req_length,
    }


async def add_instances(
    session: AsyncSession, task_result_id: int, req: s.InstanceBatchRequest
) -> s.InstanceBatchResponse:
    tr = (
        (
            await session.execute(
                select(TaskResult).where(task_results_t.c.id == task_result_id).with_for_update()
            )
        )
        .scalars()
        .first()
    )
    if tr is None:
        raise not_found(f"Task result {task_result_id} not found")
    if tr.instances_stored + len(req.instances) > MAX_INSTANCES_PER_TASK_RESULT:
        raise bad_request(
            f"a task result may have at most {MAX_INSTANCES_PER_TASK_RESULT} instances"
        )
    trc = task_results_t.c
    run_total = await _count(
        session,
        select(func.coalesce(func.sum(trc.instances_stored), 0)).where(trc.run_id == tr.run_id),
    )
    if run_total + len(req.instances) > MAX_INSTANCES_PER_RUN:
        raise bad_request(f"a run may have at most {MAX_INSTANCES_PER_RUN:,} instances")
    # A batch with a repeated native_id would make ON CONFLICT touch one row twice.
    rows = list({i.native_id: _instance_row(task_result_id, i) for i in req.instances}.values())
    inserted = 0
    for start in range(0, len(rows), _INSTANCE_CHUNK):
        stmt = pg_insert(instance_results_t).values(rows[start : start + _INSTANCE_CHUNK])
        # xmax = 0 marks rows this statement inserted (as opposed to updated), so the stored
        # count stays exact without recounting the whole task result after every batch.
        result = await session.execute(
            stmt.on_conflict_do_update(
                index_elements=["task_result_id", "native_id"],
                set_={col: stmt.excluded[col] for col in _INSTANCE_UPDATE_COLS},
            ).returning(literal_column("(xmax = 0)"))
        )
        inserted += sum(1 for (is_new,) in result if is_new)
    total = tr.instances_stored + inserted
    tr.instances_stored = total
    tr.updated_at = now_utc()
    tr.finalized_at = None
    await session.commit()
    return s.InstanceBatchResponse(
        task_result_id=task_result_id, received=len(req.instances), total_stored=total
    )


# ---------------------------------------------------------------------------
# PUT /v1/runs/{run_id}/inference
# ---------------------------------------------------------------------------


async def put_inference(
    session: AsyncSession, run_id: str, req: s.InferenceUploadRequest
) -> s.InferenceUploadResponse:
    await get_run(session, run_id, lock=True)
    await session.execute(delete(inference_batches_t).where(inference_batches_t.c.run_id == run_id))
    batches = {b.seq: b for b in req.batches}
    rows = [
        {
            "run_id": run_id,
            "seq": b.seq,
            "ts": b.timestamp,
            "task_name": b.task_name,
            "total_requests": b.total_requests,
            "successful_requests": b.successful_requests,
            "failed_requests": b.failed_requests,
            "total_prompt_tokens": b.total_prompt_tokens,
            "total_completion_tokens": b.total_completion_tokens,
            "wall_clock_time_s": b.wall_clock_time_s,
            "output_tokens_per_second": b.output_tokens_per_second,
            "mean_latency_s": b.mean_latency_s,
            "gpu_summary": b.gpu_summary,
        }
        for b in batches.values()
    ]
    for start in range(0, len(rows), _BATCH_CHUNK):
        await session.execute(
            pg_insert(inference_batches_t).values(rows[start : start + _BATCH_CHUNK])
        )
    latency = req.request_latency.model_dump(mode="json") if req.request_latency else None
    values = {
        "run_id": run_id,
        "source_paths": req.source_paths,
        "series": [x.model_dump(mode="json") for x in req.series],
        "request_latency": latency,
        "gpu_devices": [d.model_dump(mode="json") for d in req.gpu_devices],
        "updated_at": now_utc(),
    }
    stmt = pg_insert(run_inference_t).values(**values)
    await session.execute(
        stmt.on_conflict_do_update(
            index_elements=["run_id"],
            set_={k: stmt.excluded[k] for k in values if k != "run_id"},
        )
    )
    await session.commit()
    return s.InferenceUploadResponse(batches_stored=len(rows), series_stored=len(req.series))


# ---------------------------------------------------------------------------
# POST /v1/runs/{run_id}/complete
# ---------------------------------------------------------------------------


def corpus_kind(value: float | None) -> derive.MetricKind:
    if value is not None and 0.0 <= value <= 1.0:
        return "bounded"
    return "unbounded"


def build_vector(key_hashes: Sequence[int], scores: Sequence[float | None]) -> tuple[bytes, bytes]:
    """Little-endian int64 key hashes sorted ascending, and aligned float64 scores (NaN = null)."""
    keys = np.asarray(key_hashes, dtype="<i8")
    vals = np.asarray([np.nan if v is None else v for v in scores], dtype="<f8")
    order = np.argsort(keys, kind="stable")
    return keys[order].tobytes(), vals[order].tobytes()


def _meta_entry(kind: derive.MetricKind, client: dict[str, Any]) -> dict[str, Any]:
    display = client.get("display_format") or ("percent" if kind != "unbounded" else "raw")
    return {
        "higher_is_better": client.get("higher_is_better"),
        "display_format": display,
        "unit": client.get("unit"),
        "kind": kind,
    }


_KINDS_SQL = text(
    """
    SELECT e.key,
           bool_and((e.value)::float8 IN (0, 1)) AS is_binary,
           bool_and((e.value)::float8 BETWEEN 0 AND 1) AS is_bounded
    FROM instance_results ir, jsonb_each(ir.metrics) e
    WHERE ir.task_result_id = :tid AND jsonb_typeof(e.value) = 'number'
    GROUP BY e.key
    """
)
_TOKENS_SQL = text(
    """
    SELECT count(*), sum(completion_tokens), avg(completion_tokens), sum(prompt_tokens),
           count(*) FILTER (WHERE finish_reason = 'length')
    FROM instance_results WHERE task_result_id = :tid
    """
)
_REASONS_SQL = text(
    "SELECT coalesce(finish_reason, 'unknown'), count(*) FROM instance_results "
    "WHERE task_result_id = :tid GROUP BY 1"
)


@dataclass(frozen=True)
class VectorStats:
    """What finalizing derives from a task result's per-instance primary scores."""

    n_rows: int
    n_scored: int
    score: float | None
    score_is_mean: bool
    instance_scale: float
    stderr: float | None
    primary_kind: derive.MetricKind | None
    key_bytes: bytes | None
    score_bytes: bytes | None


def vector_stats(rows: Sequence[Sequence[Any]], score: float | None) -> VectorStats:
    """Steps 2, 3 and the vector of step 7 of spec 2.5, from (key_hash, primary_score) rows.

    CPU-bound for large task results, so finalize_task_result runs it in a worker thread.
    """
    values = np.array([r[1] for r in rows if r[1] is not None], dtype=np.float64)
    n = int(values.size)
    # 2. Is the corpus score the mean of the per-instance values, and on which scale?
    is_mean, scale = False, 1.0
    if n:
        mean = float(values.mean())
        if score is None:
            score, is_mean = mean, True
        else:
            found = detect_scale(mean, score)
            if found is not None:
                is_mean, scale = True, found
    # 3. Standard error of the mean on the corpus scale.
    stderr = float(scale * values.std(ddof=1) / math.sqrt(n)) if is_mean and n >= 2 else None
    key_bytes = score_bytes = None
    if rows:
        key_bytes, score_bytes = build_vector([r[0] for r in rows], [r[1] for r in rows])
    return VectorStats(
        n_rows=len(rows),
        n_scored=n,
        score=score,
        score_is_mean=is_mean,
        instance_scale=scale,
        stderr=stderr,
        primary_kind=derive.metric_kind(values.tolist()) if n else None,
        key_bytes=key_bytes,
        score_bytes=score_bytes,
    )


def _first_not_none(reported: int | None, summed: Any) -> int | None:
    if reported is not None:
        return int(reported)
    return int(summed) if summed is not None else None


async def finalize_task_result(
    session: AsyncSession, tr: TaskResult, client_meta: dict[str, Any], ts: datetime
) -> None:
    """Steps 1-8 of spec 2.5 for one task result.

    ``updated_at`` is left alone: it records the last upload, which latest_by_task uses to pick
    between task hashes, and finalizing every task result of a run at once would tie them.
    The cache keys include ``finalized_at`` instead.
    """
    tid = tr.id
    irc = instance_results_t.c
    rows = (
        await session.execute(
            select(irc.key_hash, irc.primary_score).where(irc.task_result_id == tid)
        )
    ).all()
    stats = await asyncio.to_thread(vector_stats, rows, tr.score)
    del rows
    tr.instances_stored = stats.n_rows
    tr.score = stats.score
    tr.score_is_mean = stats.score_is_mean
    tr.instance_scale = stats.instance_scale
    tr.stderr = stats.stderr

    # 4-5. Metric kinds and metadata for every corpus metric key.
    kind_rows = (await session.execute(_KINDS_SQL, {"tid": tid})).all()
    instance_kinds: dict[str, derive.MetricKind] = {
        key: "binary" if is_binary else "bounded" if is_bounded else "unbounded"
        for key, is_binary, is_bounded in kind_rows
    }
    if stats.primary_kind is not None and tr.primary_metric:
        instance_kinds[tr.primary_metric] = stats.primary_kind
    meta: dict[str, dict[str, Any]] = {}
    keys = list(tr.metrics or {})
    if tr.primary_metric and tr.primary_metric not in tr.metrics and stats.n_scored:
        keys.append(tr.primary_metric)
    for key in keys:
        kind = instance_kinds.get(key) or corpus_kind((tr.metrics or {}).get(key))
        meta[key] = _meta_entry(kind, client_meta.get(derive.metric_name(key)) or {})
    tr.metric_meta = meta
    primary = meta.get(tr.primary_metric or "")
    if primary:
        tr.metric_kind = primary["kind"]
        tr.higher_is_better = primary["higher_is_better"]
        tr.display_format = primary["display_format"]

    # 6. Token and finish-reason statistics. The task-level totals the client reported cover
    # every request (failed instances too), so they win over the sums of stored instances.
    total, comp_sum, comp_avg, prompt_sum, n_length = (
        await session.execute(_TOKENS_SQL, {"tid": tid})
    ).one()
    tr.completion_tokens_total = _first_not_none(tr.reported_completion_tokens_total, comp_sum)
    tr.mean_completion_tokens = float(comp_avg) if comp_avg is not None else None
    tr.prompt_tokens_total = _first_not_none(tr.reported_prompt_tokens_total, prompt_sum)
    tr.truncation_rate = (n_length / total) if total else None
    reasons = (await session.execute(_REASONS_SQL, {"tid": tid})).all()
    tr.finish_reason_counts = {reason: int(count) for reason, count in reasons}

    # 7. Score vector for paired statistics.
    await session.execute(
        delete(task_result_vectors_t).where(task_result_vectors_t.c.task_result_id == tid)
    )
    if stats.key_bytes is not None and stats.score_bytes is not None:
        session.add(
            TaskResultVector(
                task_result_id=int(tid or 0),
                metric_key=tr.primary_metric or "primary_score",
                n=stats.n_rows,
                key_hashes=stats.key_bytes,
                scores=stats.score_bytes,
                built_at=ts,
            )
        )
    # 8.
    tr.finalized_at = ts


def _headline(run: Run, suites: list[SuiteResult], trs: list[TaskResult]) -> dict[str, Any] | None:
    order = {spec: i for i, spec in enumerate(run.task_specs or [])}
    top = [x for x in suites if x.parent_suite is None and x.score is not None]
    top.sort(key=lambda x: order.get(x.suite_name, len(order)))
    if top:
        x = top[0]
        return {
            "kind": "suite",
            "name": x.suite_name,
            "score": x.score,
            "stderr": x.stderr,
            "display_format": x.display_format,
        }
    if len(trs) == 1:
        tr = trs[0]
        return {
            "kind": "task",
            "name": tr.task_name,
            "score": tr.score,
            "stderr": tr.stderr,
            "display_format": tr.display_format,
        }
    scores = [tr.score for tr in trs if tr.score is not None]
    formats = {tr.display_format for tr in trs if tr.score is not None}
    directions = {tr.higher_is_better for tr in trs if tr.score is not None}
    if scores and len(formats) == 1 and len(directions) == 1:
        return {
            "kind": "mean",
            "name": None,
            "score": sum(scores) / len(scores),
            "stderr": None,
            "display_format": formats.pop(),
        }
    return None


async def complete_run(
    session: AsyncSession,
    storage: Storage,
    settings: Settings,
    run_id: str,
    req: s.CompleteRequest,
) -> s.CompleteResponse:
    run = await get_run(session, run_id, lock=True)
    ts = now_utc()
    trc = task_results_t.c
    trs = list(
        (
            await session.execute(
                select(TaskResult).where(trc.run_id == run_id).order_by(trc.id).with_for_update()
            )
        )
        .scalars()
        .all()
    )
    variant_meta: dict[tuple[str, str], dict[str, Any]] = {}
    if trs:
        tv = task_variants_t.c
        rows = (
            await session.execute(
                select(tv.task_name, tv.task_hash, tv.metric_meta).where(
                    tuple_(tv.task_name, tv.task_hash).in_(
                        [(tr.task_name, tr.task_hash) for tr in trs]
                    )
                )
            )
        ).all()
        variant_meta = {(r[0], r[1]): r[2] or {} for r in rows}
    for tr in trs:
        meta = variant_meta.get((tr.task_name, tr.task_hash), {})
        await finalize_task_result(session, tr, meta, ts)
    await session.flush()

    # 9. Suites.
    by_task = latest_by_task(trs)
    await session.execute(delete(suite_results_t).where(suite_results_t.c.run_id == run_id))
    request_defs = {
        x.name: (x.aggregation, [c.model_dump() for c in x.children]) for x in req.suites
    }
    child_suites = {c.name for x in req.suites for c in x.children if c.type == "suite"}
    stored = await current_suite_defs(session, child_suites - set(request_defs))
    defs: dict[str, tuple[str, list[dict]]] = {k: (v[0], v[1]) for k, v in stored.items()}
    defs.update(request_defs)
    undefined_suites = sorted(child_suites - set(defs))
    suite_rows: list[SuiteResult] = []
    for x in req.suites:
        children = [c.model_dump() for c in x.children]
        def_hash = suite_definition_hash(x.aggregation, children)
        stmt = pg_insert(suite_defs_t).values(
            suite_name=x.name,
            definition_hash=def_hash,
            aggregation=x.aggregation,
            description=x.description,
            children=children,
            first_seen_at=ts,
            last_seen_at=ts,
        )
        await session.execute(
            stmt.on_conflict_do_update(
                index_elements=["suite_name", "definition_hash"],
                set_={
                    "last_seen_at": stmt.excluded.last_seen_at,
                    "description": func.coalesce(
                        stmt.excluded.description, suite_defs_t.c.description
                    ),
                },
            )
        )
        try:
            tree = build_tree(x.name, defs)
        except ValueError as exc:
            raise bad_request(str(exc)) from exc
        leaf_names = leaves(tree)
        present = {name: by_task[name] for name in leaf_names if name in by_task}
        scored = {name for name, tr in present.items() if tr.score is not None}
        n_inst = {name: tr.num_instances for name, tr in present.items()}
        weights = leaf_weights(tree, n_inst, scored)
        formats = {tr.display_format for tr in present.values()}
        directions = {tr.higher_is_better for tr in present.values()}
        row = SuiteResult(
            run_id=run_id,
            suite_name=x.name,
            parent_suite=x.parent,
            aggregation=x.aggregation,
            description=x.description,
            children=children,
            definition_hash=def_hash,
            score=derive.finite_or_none(x.score),
            stderr=propagate_stderr(weights, {k: tr.stderr for k, tr in present.items()}),
            num_tasks=x.num_tasks,
            n_instances=sum(n_inst.values()),
            metrics=derive.flatten_metrics(x.metrics),
            primary_metric=x.primary_metric,
            display_format="percent" if formats == {"percent"} else "raw",
            higher_is_better=directions.pop() if len(directions) == 1 else None,
            tasks_missing=[name for name in leaf_names if name not in by_task],
        )
        session.add(row)
        suite_rows.append(row)

    # 10-12. Run summary.
    run.num_tasks = len(trs)
    run.num_failed_tasks = sum(1 for tr in trs if tr.error)
    run.num_instances = sum(tr.instances_stored for tr in trs)
    run.headline = _headline(run, suite_rows, trs)
    run.status = req.status
    if req.finished_at is not None:
        run.finished_at = req.finished_at
    if req.duration_seconds is not None:
        run.duration_seconds = req.duration_seconds
    run.upload_state = "complete"
    run.updated_at = ts
    run.search_text = build_search_text(run)

    # 13. Artifact verification.
    artifacts = list(
        (await session.execute(select(Artifact).where(artifacts_t.c.run_id == run_id)))
        .scalars()
        .all()
    )
    missing: list[str] = []
    if artifacts:
        found = await storage.list_objects(settings.object_prefix(run_id))
        for artifact in artifacts:
            ok = settings.object_key(run_id, artifact.path) in found
            if ok != artifact.uploaded:
                artifact.uploaded = ok
                artifact.updated_at = ts
            if not ok:
                missing.append(artifact.path)

    # 14. Warnings.
    warnings: list[str] = []
    if len(trs) != req.expected_task_results:
        warnings.append(f"expected {req.expected_task_results} task results, server has {len(trs)}")
    for tr in trs:
        if tr.instances_stored != tr.instance_count_expected:
            warnings.append(
                f"{tr.task_name}: expected {tr.instance_count_expected} instances, "
                f"server has {tr.instances_stored}"
            )
    if missing:
        warnings.append(f"{len(missing)} signed artifacts are missing from storage")
    if undefined_suites:
        warnings.append(
            "no definition for child suite(s) "
            f"{', '.join(undefined_suites)}; they count as empty until one is uploaded"
        )

    instances = run.num_instances
    await session.commit()
    logger.info(
        "run completed", extra={"run_id": run_id, "task_results": len(trs), "instances": instances}
    )
    return s.CompleteResponse(
        run_id=run_id,
        status=req.status,
        upload_state="complete",
        dashboard_url=settings.dashboard_run_url(run_id),
        task_results=len(trs),
        instances=instances,
        artifacts_missing=sorted(missing),
        warnings=warnings,
    )
