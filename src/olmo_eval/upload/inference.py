"""Inference metrics files turned into the InferenceUploadRequest payload.

Sources: ``metrics/*-inference.jsonl`` (one BatchMetrics per line, written by the
file reporter) and ``metrics/vllm_server_metrics.jsonl`` (vLLM /metrics snapshots,
external runner only). Per-sample GPU arrays become downsampled series; the full
files are uploaded to GCS as artifacts.

The file reporter appends, and an output directory can be reused across runs and
models, so only this model's batch files and records stamped after the run started
are included.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

MAX_POINTS = 500
MAX_SERIES = 64
MAX_BATCHES = 20000
# Clock slack for records that predate the run's recorded start (matches the uploader).
STALE_RECORD_SLACK = timedelta(minutes=1)
SUFFIX = "-inference.jsonl"
VLLM_METRICS_FILE = "vllm_server_metrics.jsonl"
_VLLM_SERIES = (
    ("vllm:num_requests_running", "vllm_num_requests_running", None, 1.0),
    ("vllm:num_requests_waiting", "vllm_num_requests_waiting", None, 1.0),
    ("vllm:gpu_cache_usage_perc", "vllm_kv_cache_usage_pct", "%", 100.0),
    ("vllm:kv_cache_usage_perc", "vllm_kv_cache_usage_pct", "%", 100.0),
)


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _iter_json_lines(path: Path) -> Iterator[Mapping[str, Any]]:
    with path.open("rb") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if isinstance(rec, Mapping):
                yield rec


def _nonneg_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return 0
    return max(0, int(value))


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return 0.0
    return float(value)


def _gpu_summary(value: Any) -> dict[str, float | None] | None:
    if not isinstance(value, Mapping):
        return None
    summary: dict[str, float | None] = {}
    for key, raw in value.items():
        if isinstance(raw, bool):
            continue
        if isinstance(raw, (int, float)):
            summary[str(key)] = float(raw) if math.isfinite(raw) else None
        elif raw is None:
            summary[str(key)] = None
    return summary


def downsample(points: list[tuple[float, float]], max_points: int = MAX_POINTS) -> list:
    """Average points into at most max_points equal time buckets."""
    if len(points) <= max_points:
        return [[round(x, 3), y] for x, y in sorted(points)]
    points = sorted(points)
    start, end = points[0][0], points[-1][0]
    width = (end - start) / max_points or 1.0
    sums: dict[int, list[float]] = {}
    for x, y in points:
        bucket = min(int((x - start) / width), max_points - 1)
        acc = sums.setdefault(bucket, [0.0, 0.0, 0.0])
        acc[0] += x
        acc[1] += y
        acc[2] += 1
    return [[round(acc[0] / acc[2], 3), acc[1] / acc[2]] for _, acc in sorted(sums.items())]


def _quantiles(values: list[float]) -> dict[str, Any] | None:
    finite = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    if not finite:
        return None
    p5, p25, p50, p75, p95 = (float(q) for q in np.percentile(finite, [5, 25, 50, 75, 95]))
    return {
        "p5": p5,
        "p25": p25,
        "p50": p50,
        "p75": p75,
        "p95": p95,
        "mean": float(np.mean(finite)),
        "n": len(finite),
    }


def _safe_name(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_")


def find_inference_files(
    output_dir: Path, model_names: Sequence[str] = ()
) -> tuple[list[Path], Path | None]:
    """Batch files and the vLLM metrics file under ``metrics/``.

    Batch files are named ``<provider>_<model>-inference.jsonl``, where the model is the
    alias or the provider model. When any file matches one of ``model_names``, files for
    other models are left out.
    """
    metrics_dir = output_dir / "metrics"
    if not metrics_dir.is_dir():
        return [], None
    batch_files = sorted(metrics_dir.glob(f"*{SUFFIX}"))
    endings = tuple(f"_{_safe_name(n)}{SUFFIX}" for n in model_names if n)
    if endings:
        ours = [p for p in batch_files if p.name.endswith(endings)]
        if ours:
            batch_files = ours
    vllm = metrics_dir / VLLM_METRICS_FILE
    return batch_files, vllm if vllm.is_file() else None


def build_inference_payload(
    output_dir: Path,
    started_at: str | None = None,
    model_names: Sequence[str] = (),
    warnings: list[str] | None = None,
) -> dict[str, Any] | None:
    """Build InferenceUploadRequest, or None when the run has no inference records.

    What does not fit the request's limits is left out and described in ``warnings``.
    """
    batch_files, vllm_file = find_inference_files(output_dir, model_names)
    if not batch_files and vllm_file is None:
        return None

    origin = _parse_time(started_at)
    cutoff = origin - STALE_RECORD_SLACK if origin is not None else None
    used: set[Path] = set()
    batches: list[dict[str, Any]] = []
    dropped_batches = 0
    gpu_samples: dict[tuple[str, int], list[tuple[datetime, float]]] = {}
    devices: dict[int, dict[str, Any]] = {}
    latencies: dict[str, list[float]] = {"e2e": [], "ttft": [], "tpot": []}
    earliest: datetime | None = None

    for path in batch_files:
        for rec in _iter_json_lines(path):
            if rec.get("type") != "batch" or not isinstance(rec.get("data"), Mapping):
                continue
            data = rec["data"]
            timestamp = _parse_time(data.get("timestamp"))
            if timestamp is None or (cutoff is not None and timestamp < cutoff):
                continue
            used.add(path)
            if len(batches) >= MAX_BATCHES:
                dropped_batches += 1
            else:
                batches.append(
                    {
                        "seq": len(batches),
                        "timestamp": timestamp.isoformat(),
                        "task_name": data.get("task_name")
                        if isinstance(data.get("task_name"), str)
                        else None,
                        "total_requests": _nonneg_int(data.get("total_requests")),
                        "successful_requests": _nonneg_int(data.get("successful_requests")),
                        "failed_requests": _nonneg_int(data.get("failed_requests")),
                        "total_prompt_tokens": _nonneg_int(data.get("total_prompt_tokens")),
                        "total_completion_tokens": _nonneg_int(data.get("total_completion_tokens")),
                        "wall_clock_time_s": _number(data.get("wall_clock_time_s")),
                        "output_tokens_per_second": _number(data.get("output_tokens_per_second")),
                        "mean_latency_s": _number(data.get("mean_latency_s")),
                        "gpu_summary": _gpu_summary(data.get("gpu_summary")),
                    }
                )
            for device in data.get("gpu_devices") or []:
                if not isinstance(device, Mapping):
                    continue
                device_id = _nonneg_int(device.get("device_id"))
                memory_total = device.get("memory_total_mb")
                devices.setdefault(
                    device_id,
                    {
                        "device_id": device_id,
                        "name": str(device.get("name") or "unknown"),
                        "memory_total_mb": float(memory_total)
                        if isinstance(memory_total, (int, float))
                        else None,
                    },
                )
                for sample in device.get("samples") or []:
                    if not isinstance(sample, Mapping):
                        continue
                    ts = _parse_time(sample.get("timestamp"))
                    if ts is None or (cutoff is not None and ts < cutoff):
                        continue
                    earliest = ts if earliest is None or ts < earliest else earliest
                    for field, name in (
                        ("utilization_pct", "gpu_utilization_pct"),
                        ("memory_used_mb", "gpu_memory_used_mb"),
                    ):
                        value = sample.get(field)
                        if isinstance(value, (int, float)) and math.isfinite(value):
                            gpu_samples.setdefault((name, device_id), []).append((ts, value))
            for request in data.get("requests") or []:
                if not isinstance(request, Mapping):
                    continue
                for key, field in (
                    ("e2e", "end_to_end_latency_s"),
                    ("ttft", "time_to_first_token_s"),
                    ("tpot", "time_per_output_token_s"),
                ):
                    value = request.get(field)
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        latencies[key].append(float(value))

    vllm_samples: dict[str, tuple[str | None, list[tuple[datetime, float]]]] = {}
    if vllm_file is not None:
        for rec in _iter_json_lines(vllm_file):
            if rec.get("error"):
                continue
            ts = _parse_time(rec.get("timestamp"))
            metrics = rec.get("metrics")
            if ts is None or not isinstance(metrics, Mapping):
                continue
            if cutoff is not None and ts < cutoff:
                continue
            used.add(vllm_file)
            earliest = ts if earliest is None or ts < earliest else earliest
            for source, name, unit, scale in _VLLM_SERIES:
                entry = metrics.get(source)
                value = entry.get("value") if isinstance(entry, Mapping) else None
                if isinstance(value, (int, float)) and math.isfinite(value):
                    vllm_samples.setdefault(name, (unit, []))[1].append((ts, value * scale))

    if not used:
        return None
    base = origin or earliest or datetime.now(UTC)
    series: list[dict[str, Any]] = []
    for (name, device_id), samples in sorted(gpu_samples.items(), key=lambda kv: kv[0][::-1]):
        points = [((ts - base).total_seconds(), value) for ts, value in samples]
        series.append(
            {
                "name": name,
                "unit": "%" if name == "gpu_utilization_pct" else "MB",
                "device": str(device_id),
                "points": downsample(points),
            }
        )
    for name, (unit, samples) in sorted(vllm_samples.items()):
        points = [((ts - base).total_seconds(), value) for ts, value in samples]
        series.append({"name": name, "unit": unit, "device": None, "points": downsample(points)})

    request_latency = None
    if any(latencies.values()):
        request_latency = {
            "end_to_end_s": _quantiles(latencies["e2e"]),
            "ttft_s": _quantiles(latencies["ttft"]),
            "tpot_s": _quantiles(latencies["tpot"]),
        }

    if warnings is not None:
        if dropped_batches:
            warnings.append(
                f"Inference metrics: kept the first {MAX_BATCHES} batches and left out "
                f"{dropped_batches}; request latency still covers every batch"
            )
        if len(series) > MAX_SERIES:
            warnings.append(f"Inference metrics: kept {MAX_SERIES} of {len(series)} time series")
    sources = [p for p in [*batch_files, vllm_file] if p is not None and p in used]
    return {
        "source_paths": [p.relative_to(output_dir).as_posix() for p in sources],
        "batches": batches,
        "series": series[:MAX_SERIES],
        "request_latency": request_latency,
        "gpu_devices": [devices[k] for k in sorted(devices)],
    }
