"""The run manifest (``<output_dir>/manifest.json``).

metrics.json alone cannot rebuild a run: it lacks the original task specs, suite
definitions, per-task errors, metric directions and provenance. The runner writes
this manifest next to it (once at start with status "running", again at the end),
and the uploader reads only files on disk, so ``olmo-eval results upload`` and the
in-run upload share one code path.
"""

from __future__ import annotations

import json
import math
import os
import re
import secrets
import sys
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from olmo_eval.upload import provenance
from olmo_eval.upload.config import LAUNCH_ID_ENV
from olmo_eval.upload.metadata import (
    base_task_name,
    metric_meta_for_spec,
    recompute_suite_aggregations,
    strip_priority,
    suite_results,
    suites_containing,
)

MANIFEST_NAME = "manifest.json"
METRICS_NAME = "metrics.json"
# Proves to the ingest service that an upload comes from the run's own results directory.
# Never uploaded.
RUN_SECRET_NAME = ".upload-secret"
MANIFEST_VERSION = 1
RUN_ID_RE = re.compile(r"^[a-z0-9]{6,32}$")


class ManifestError(Exception):
    """A results directory cannot be turned into an upload."""


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def to_utc_iso(value: str | None) -> str | None:
    """Parse an ISO timestamp, treating naive values as UTC."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).isoformat()


def run_status(task_errors: Mapping[str, str | None], run_errors: Sequence[Any] = ()) -> str:
    """complete when nothing failed, partial when some tasks succeeded, else failed."""
    failed = {task for task, error in task_errors.items() if error}
    succeeded = [task for task, error in task_errors.items() if not error]
    if not failed and not run_errors and succeeded:
        return "complete"
    if succeeded:
        return "partial"
    return "failed"


def current_argv() -> list[str]:
    return ["olmo-eval", *sys.argv[1:]]


REDACTED = "[redacted]"
# Config keys and CLI flags whose string values are credentials. Tokenizer special tokens
# ("eos_token" and the like) are not.
_SECRET_NAME_RE = re.compile(
    r"(^|_)(api_?key|access_?key|secret|secret_?key|client_?secret|password|passwd|"
    r"private_?key|token|auth_?token|access_?token|refresh_?token|bearer|authorization)$"
)
_TOKENIZER_TOKENS = frozenset(
    {"eos_token", "bos_token", "pad_token", "unk_token", "sep_token", "cls_token", "mask_token"}
)


def is_secret_name(name: str) -> bool:
    """Whether a config key or CLI flag name looks like it holds a credential."""
    key = name.strip().lstrip("-").rsplit(".", 1)[-1].lower().replace("-", "_")
    return key not in _TOKENIZER_TOKENS and bool(_SECRET_NAME_RE.search(key))


def redact_argv(argv: Sequence[str]) -> list[str]:
    """argv with the values of secret-looking options replaced.

    Handles ``key=value`` overrides (``-o provider.api_key=...``), ``--flag=value`` and
    ``--flag value``. An argument after a flag that starts with ``-`` is another option
    (the flag was boolean, like ``--hf-token``) and is kept.
    """
    out: list[str] = []
    redact_next = False
    for arg in argv:
        if redact_next and not arg.startswith("-"):
            out.append(REDACTED)
            redact_next = False
            continue
        redact_next = False
        name, sep, _value = arg.partition("=")
        if sep and is_secret_name(name):
            out.append(f"{name}={REDACTED}")
        elif arg.startswith("--") and not sep and is_secret_name(arg):
            out.append(arg)
            redact_next = True
        else:
            out.append(arg)
    return out


def json_safe(value: Any) -> Any:
    """Round-trip through JSON so configs with odd values still serialize.

    Non-finite floats become None, since the ingest API only accepts standard JSON, and
    string values under secret-looking keys (``api_key``, ``password``...) are redacted.
    """
    return _clean(json.loads(json.dumps(value, default=str)))


def _clean(value: Any, key: str | None = None) -> Any:
    if isinstance(value, str) and key is not None and value and is_secret_name(key):
        return REDACTED
    if isinstance(value, dict):
        return {k: _clean(item, str(k)) for k, item in value.items()}
    if isinstance(value, list):
        return [_clean(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def build_run_info(
    *,
    run_id: str,
    status: str,
    started_at: str | None,
    finished_at: str | None = None,
    duration_seconds: float | None = None,
    experiment_name: str | None = None,
    experiment_group: str | None = None,
    task_specs: Sequence[str] = (),
    output_dir: str | None = None,
    harness_config: Mapping[str, Any] | None = None,
    provider_init_seconds: Mapping[str, float] | None = None,
    errors: Iterable[Mapping[str, Any]] = (),
    tags: Sequence[str] = (),
    argv: Sequence[str] | None = None,
    capture_provenance: bool = True,
) -> dict[str, Any]:
    """RunInfo for the ingest contract, capturing provenance from this process."""
    if capture_provenance:
        git = provenance.git_info()
        beaker = provenance.beaker_info()
        environment = provenance.environment_info()
        author: str | None = provenance.get_author()
    else:
        git = {"repo": None, "commit": None, "branch": None, "dirty": None}
        beaker = None
        environment = {
            "hostname": None,
            "platform": None,
            "python_version": None,
            "packages": {},
            "cuda_version": None,
            "gpu_type": None,
            "gpu_count": None,
        }
        author = None
    return {
        "run_id": run_id,
        "launch_id": (os.environ.get(LAUNCH_ID_ENV) or None) if capture_provenance else None,
        "experiment_name": experiment_name,
        "experiment_group": experiment_group,
        "status": status,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_seconds": duration_seconds,
        "author": author,
        "tags": list(tags),
        "olmo_eval_version": provenance.olmo_eval_version() if capture_provenance else None,
        "git": git,
        "beaker": beaker,
        "environment": environment,
        "argv": redact_argv(argv if argv is not None else current_argv()),
        "task_specs": list(task_specs),
        "output_dir": output_dir,
        "harness_config": json_safe(dict(harness_config)) if harness_config else None,
        "provider_init_seconds": dict(provider_init_seconds) if provider_init_seconds else None,
        "errors": [
            {"task": e.get("task"), "error": str(e.get("error") or "")}
            for e in errors
            if e.get("error")
        ],
    }


def build_model_info(
    *,
    name: str,
    path: str,
    model_hash: str | None,
    revision: str | None,
    provider_kind: str,
    provider_config: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "name": name,
        "path": path,
        "model_hash": model_hash,
        "revision": revision,
        "provider_kind": provider_kind,
        "provider_config": json_safe(dict(provider_config)),
    }


def model_info_from_provider(
    provider_config: Any, attention_backend: str | None = None
) -> dict[str, Any]:
    """ModelInfo from a ProviderConfig, matching what metrics.json records."""
    from olmo_eval.common.types import compute_model_hash

    config = provider_config.to_dict()
    if attention_backend:
        config["attention_backend"] = attention_backend
    return build_model_info(
        name=provider_config.alias or provider_config.model,
        path=provider_config.model,
        model_hash=compute_model_hash(config),
        revision=provider_config.revision,
        provider_kind=str(provider_config.kind),
        provider_config=config,
    )


def build_task_entry(
    spec: str, metric_meta: Mapping[str, Any], error: str | None = None
) -> dict[str, Any]:
    return {
        "base_task": base_task_name(spec),
        "metric_meta": dict(metric_meta),
        "suites": suites_containing(spec),
        "error": error,
    }


def build_manifest(
    run: Mapping[str, Any],
    model: Mapping[str, Any],
    tasks: Mapping[str, Mapping[str, Any]] | None = None,
    suites: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    return {
        "manifest_version": MANIFEST_VERSION,
        "run": dict(run),
        "model": dict(model),
        "tasks": {k: dict(v) for k, v in (tasks or {}).items()},
        "suites": [dict(s) for s in suites],
    }


def write_manifest(output_dir: str | Path, manifest: Mapping[str, Any]) -> Path:
    """Write manifest.json atomically."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / MANIFEST_NAME
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    os.replace(tmp, path)
    return path


def run_secret(output_dir: str | Path) -> str | None:
    """The run's write secret, created on first use. None if the directory is not writable."""
    path = Path(output_dir) / RUN_SECRET_NAME
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        pass
    except OSError:
        return None
    else:
        with os.fdopen(fd, "w") as f:
            f.write(secrets.token_urlsafe(32))
    try:
        value = path.read_text().strip()
    except OSError:
        return None
    return value or None


def read_manifest(output_dir: str | Path) -> dict[str, Any] | None:
    path = Path(output_dir) / MANIFEST_NAME
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def read_metrics(output_dir: str | Path) -> dict[str, Any] | None:
    path = Path(output_dir) / METRICS_NAME
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        raise ManifestError(f"Could not read {path}: {e}") from e
    if not isinstance(data, dict):
        raise ManifestError(f"{path} is not a JSON object")
    return data


def _task_errors(metrics: Mapping[str, Any]) -> dict[str, str]:
    errors: dict[str, str] = {}
    for entry in metrics.get("errors") or []:
        if isinstance(entry, Mapping) and entry.get("task") and entry.get("error"):
            errors[str(entry["task"])] = str(entry["error"])
    return errors


def degraded_manifest(output_dir: str | Path, metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Best-effort manifest for a directory written before manifests existed.

    Provenance of the original run is unknown, so git, Beaker and environment are
    left empty instead of describing the machine doing the upload.
    """
    run_id = metrics.get("experiment_id")
    if not isinstance(run_id, str) or not RUN_ID_RE.match(run_id):
        raise ManifestError(
            f"metrics.json in {output_dir} has no usable experiment_id ({run_id!r}); "
            "cannot identify the run"
        )
    raw_config = metrics.get("config")
    config: Mapping[str, Any] = raw_config if isinstance(raw_config, Mapping) else {}
    provider = config.get("provider")
    if isinstance(provider, Mapping):
        harness_config: dict[str, Any] | None = {
            k: v for k, v in config.items() if k != "model_hash"
        }
        provider_config = dict(provider)
    else:
        harness_config = None
        provider_config = {k: v for k, v in config.items() if k != "model_hash"}
    model_path = str(provider_config.get("model") or "unknown")
    model = build_model_info(
        name=str(provider_config.get("alias") or model_path),
        path=model_path,
        model_hash=config.get("model_hash"),
        revision=provider_config.get("revision"),
        provider_kind=str(provider_config.get("kind") or "unknown"),
        provider_config=provider_config,
    )

    task_errors = _task_errors(metrics)
    task_rows = [t for t in metrics.get("tasks") or [] if isinstance(t, Mapping) and t.get("task")]
    tasks: dict[str, dict[str, Any]] = {}
    for row in task_rows:
        spec = str(row["task"])
        names = list((row.get("metrics") or {}).keys())
        meta = metric_meta_for_spec(spec, names)
        tasks[spec] = build_task_entry(spec, meta, task_errors.get(spec))

    from olmo_eval.evals.suites import suite_exists

    summary = metrics.get("summary") or {}
    suite_specs = [name for name in summary if suite_exists(strip_priority(name))]
    task_results = {str(row["task"]): dict(row) for row in task_rows}
    suites = suite_results(recompute_suite_aggregations(suite_specs, task_results), task_results)

    finished_at = to_utc_iso(metrics.get("timestamp"))
    duration = metrics.get("experiment_duration_seconds")
    started_at = None
    if finished_at and isinstance(duration, (int, float)):
        started_at = (datetime.fromisoformat(finished_at) - timedelta(seconds=duration)).isoformat()

    status = run_status({spec: task_errors.get(spec) for spec in tasks}, [])
    run = build_run_info(
        run_id=run_id,
        status=status,
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=float(duration) if isinstance(duration, (int, float)) else None,
        experiment_name=metrics.get("experiment_name"),
        experiment_group=metrics.get("experiment_group"),
        task_specs=[*suite_specs, *[s for s in tasks if s not in suite_specs]],
        output_dir=str(output_dir),
        harness_config=harness_config,
        provider_init_seconds=metrics.get("provider_init_seconds"),
        errors=[{"task": t, "error": e} for t, e in task_errors.items()],
        argv=[],
        capture_provenance=False,
    )
    return build_manifest(run, model, tasks, suites)
