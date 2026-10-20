"""Compact per-instance rows extracted from predictions and requests JSONL files.

The dashboard stores one small row per instance (scores, flags, short previews and
byte offsets). Full records stay in GCS and are read back by byte range.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from olmo_eval.runners.common.types import PREDICTIONS_SUFFIX, REQUESTS_SUFFIX
from olmo_eval.runners.common.usage import instance_prompt_tokens
from olmo_eval.runners.processing.utils import sanitize_spec_for_filename

BATCH_SIZE = 2000
PREVIEW_CHARS = 300
MAX_METRICS = 200
MAX_NATIVE_ID = 512


@dataclass(frozen=True)
class RequestRef:
    offset: int
    length: int
    preview: str | None


def find_task_file(
    output_dir: Path,
    kind: str,
    spec: str,
    task_hash: str | None,
    model_path: str | None = None,
    since: float | None = None,
    warnings: list[str] | None = None,
) -> Path | None:
    """Locate the predictions or requests file this run wrote for a task.

    Files live at ``<kind>/<sanitized model path>/<sanitized spec>[_<hash6>]-<kind>.jsonl``.
    An output directory can hold files from earlier runs and other models, so files
    modified before ``since`` (a POSIX timestamp) are ignored, the run's own model
    directory (``model_path``, the provider model) is searched first, and an ambiguous
    match returns None with a warning rather than guessing.
    """
    suffix = PREDICTIONS_SUFFIX if kind == "predictions" else REQUESTS_SUFFIX
    base = sanitize_spec_for_filename(spec)
    names = [f"{base}_{task_hash[-6:]}{suffix}"] if task_hash else []
    names.append(f"{base}{suffix}")
    root = output_dir / kind
    if not root.is_dir():
        return None

    def fresh(path: Path) -> bool:
        return path.is_file() and (since is None or path.stat().st_mtime >= since)

    if model_path:
        model_dir = root / sanitize_spec_for_filename(model_path)
        if model_dir.is_dir():
            for name in names:
                if fresh(model_dir / name):
                    return model_dir / name
            return None
    for name in names:
        matches = [p for p in root.glob(f"*/{name}") if fresh(p)]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            if warnings is not None:
                listed = ", ".join(sorted(p.relative_to(output_dir).as_posix() for p in matches))
                warnings.append(f"Skipped {kind} for {spec}: several candidate files ({listed})")
            return None
    return None


def _line_length(line: bytes) -> int:
    if line.endswith(b"\r\n"):
        return len(line) - 2
    if line.endswith(b"\n"):
        return len(line) - 1
    return len(line)


def _iter_lines(path: Path) -> Iterator[tuple[int, bytes]]:
    offset = 0
    with path.open("rb") as f:
        for line in f:
            yield offset, line
            offset += len(line)


def _get(obj: Any, key: str) -> Any:
    return obj.get(key) if isinstance(obj, Mapping) else None


def _message_text(content: Any) -> str | None:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = [_get(part, "text") for part in content]
        texts = [text for text in texts if isinstance(text, str)]
        return " ".join(texts) if texts else None
    return None


def _request_preview(rec: Mapping[str, Any]) -> str | None:
    query = _get(rec.get("doc"), "query")
    if isinstance(query, str) and query:
        return query[:PREVIEW_CHARS]
    context = _get(rec.get("request"), "context")
    if isinstance(context, list):
        for message in reversed(context):
            if _get(message, "role") == "user":
                text = _message_text(_get(message, "content"))
                return text[:PREVIEW_CHARS] if text else None
        return None
    if isinstance(context, str) and context:
        return context[-PREVIEW_CHARS:]
    return None


def index_requests(path: Path) -> tuple[dict[str, RequestRef], dict[int, RequestRef]]:
    """One streaming pass over a requests file: native_id and doc_id to line refs."""
    by_native: dict[str, RequestRef] = {}
    by_doc: dict[int, RequestRef] = {}
    for offset, line in _iter_lines(path):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if not isinstance(rec, Mapping):
            continue
        ref = RequestRef(offset, _line_length(line), _request_preview(rec))
        native_id = rec.get("native_id")
        if native_id is not None:
            by_native.setdefault(str(native_id), ref)
        doc_id = rec.get("doc_id")
        if isinstance(doc_id, int) and not isinstance(doc_id, bool):
            by_doc.setdefault(doc_id, ref)
    return by_native, by_doc


def _finite(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


def _flat_metrics(instance_metrics: Any, primary_key: str | None) -> dict[str, float]:
    flat: dict[str, float] = {}
    if not isinstance(instance_metrics, Mapping):
        return flat
    for metric, scorers in instance_metrics.items():
        if not isinstance(scorers, Mapping):
            continue
        for scorer, value in scorers.items():
            number = _finite(value)
            if number is not None:
                flat[f"{metric}:{scorer}"] = number
    keys = sorted(flat)
    if primary_key in flat:
        keys.remove(primary_key)
        keys.insert(0, primary_key)
    return {key: flat[key] for key in keys[:MAX_METRICS]}


INT32_MAX = 2**31 - 1


def _int32(value: int | None) -> int | None:
    """The value if it fits the ingest API's 32-bit fields, else None."""
    return value if value is not None and value <= INT32_MAX else None


def _nonneg_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, float) and value.is_integer() and value >= 0:
        return int(value)
    return None


def _label(value: Any) -> str | None:
    if value is None:
        return None
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return text[:256]


def _judge_verdict(judge: Any) -> str | None:
    if judge is None:
        return None
    if isinstance(judge, str):
        return judge[:64]
    if isinstance(judge, Mapping):
        verdict = judge.get("verdict") or judge.get("result") or judge.get("label")
        if verdict is None:
            return None
        return str(verdict)[:64]
    return None


def _output_tokens(output: Mapping[str, Any]) -> int | None:
    tokens = _nonneg_int(output.get("completion_tokens"))
    if tokens is None:
        tokens = _nonneg_int(output.get("num_tokens"))
    return tokens


def _prompt_tokens(rec: Mapping[str, Any], outputs: list[Mapping[str, Any]]) -> int | None:
    """Prompt tokens for one instance (rules in runners/common/usage.py).

    Agent records with more than one assistant turn are left null: their outputs
    describe only the last model call.
    """
    turns = _get(rec.get("trajectory"), "turns")
    if isinstance(turns, list):
        assistant_turns = [t for t in turns if _get(t, "role") == "assistant"]
        if len(assistant_turns) > 1:
            return None
    return instance_prompt_tokens(outputs)


def _unique_native_id(base: str, counts: dict[str, int], used: set[str]) -> str:
    """Return ``base``, or ``base#<n>`` for repeats, unique among ``used``.

    ``base`` is already at most MAX_NATIVE_ID characters. The suffix replaces its tail
    when needed so the result stays within the limit, and a candidate that is already
    taken (including by a literal id such as "x#2") moves on to the next number.
    """
    n = counts.get(base, 0)
    while True:
        n += 1
        if n == 1:
            candidate = base
        else:
            suffix = f"#{n}"
            candidate = base[: MAX_NATIVE_ID - len(suffix)] + suffix
        if candidate not in used:
            counts[base] = n
            used.add(candidate)
            return candidate


def build_instance_row(
    rec: Mapping[str, Any],
    native_id: str,
    primary_metric: str | None,
    pred_offset: int,
    pred_length: int,
    request: RequestRef | None,
) -> dict[str, Any]:
    """One InstanceIn row from a predictions record."""
    outputs = rec.get("model_output") or []
    if not isinstance(outputs, list):
        outputs = []
    outputs = [o for o in outputs if isinstance(o, Mapping)]
    out0: Mapping[str, Any] = outputs[0] if outputs else {}

    primary_score = None
    primary_key = None
    instance_metrics = rec.get("instance_metrics")
    if primary_metric and ":" in primary_metric:
        metric, scorer = primary_metric.rsplit(":", 1)
        primary_key = primary_metric
        if isinstance(instance_metrics, Mapping):
            scorers = instance_metrics.get(metric)
            if isinstance(scorers, Mapping):
                primary_score = _finite(scorers.get(scorer))

    # For generated samples, count every sample, so pass@k tasks (num_outputs > 1)
    # report all generated tokens and any truncated sample. The server sums
    # completion_tokens and counts finish_reason == "length" per row for the task's
    # token and truncation statistics. Multiple-choice outputs are one per choice, not
    # samples, so they keep the first output's token count.
    finish_reason = out0.get("finish_reason")
    samples = [o for o in outputs if "completion_tokens" in o or "finish_reason" in o]
    if any(o.get("finish_reason") == "length" for o in samples):
        finish_reason = "length"
    if samples:
        counted = [n for n in map(_output_tokens, samples) if n is not None]
        completion_tokens = sum(counted) if counted else None
    else:
        completion_tokens = _output_tokens(out0)
    extracted = out0.get("extracted_answer")
    text = out0.get("text") or rec.get("final_output") or ""
    judge = rec.get("judge_result", out0.get("judge_result"))

    return {
        "native_id": native_id,
        "doc_id": _int32(_nonneg_int(rec.get("doc_id"))),
        "primary_score": primary_score,
        "metrics": _flat_metrics(instance_metrics, primary_key),
        "label": _label(rec.get("label")),
        "finish_reason": finish_reason[:32] if isinstance(finish_reason, str) else None,
        "completion_tokens": _int32(completion_tokens),
        "prompt_tokens": _int32(_prompt_tokens(rec, outputs)),
        "num_outputs": len(outputs),
        "extracted_answer": None if extracted is None else str(extracted)[:256],
        "prompt_preview": request.preview[:PREVIEW_CHARS] if request and request.preview else None,
        "output_preview": text[:PREVIEW_CHARS] if isinstance(text, str) and text else None,
        "judge_verdict": _judge_verdict(judge),
        "has_scoring_error": any(bool(o.get("scoring_errors")) for o in outputs),
        "has_execution_result": any(o.get("execution_result") is not None for o in outputs),
        "has_judge_result": judge is not None,
        "has_trajectory": rec.get("trajectory") is not None,
        "pred_offset": pred_offset,
        "pred_length": _int32(pred_length),
        "req_offset": request.offset if request else None,
        "req_length": _int32(request.length) if request else None,
    }


def count_instances(predictions_path: Path) -> int:
    """Number of non-blank lines, which is the number of rows iter_instances yields."""
    count = 0
    with predictions_path.open("rb") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def iter_instances(
    predictions_path: Path,
    requests_path: Path | None,
    primary_metric: str | None,
) -> Iterator[dict[str, Any]]:
    """Yield one InstanceIn row per predictions line.

    native_ids are cut to MAX_NATIVE_ID characters, then duplicates within the file get
    "#2", "#3", ... suffixes (within the limit) so every row has a unique key. Lines
    that are not JSON objects yield a minimal row so the count matches count_instances().
    """
    by_native: dict[str, RequestRef] = {}
    by_doc: dict[int, RequestRef] = {}
    if requests_path is not None and requests_path.is_file():
        by_native, by_doc = index_requests(requests_path)

    counts: dict[str, int] = {}
    used: set[str] = set()
    line_number = 0
    for offset, line in _iter_lines(predictions_path):
        if not line.strip():
            continue
        line_number += 1
        try:
            rec = json.loads(line)
        except ValueError:
            rec = None
        if not isinstance(rec, Mapping):
            rec = {}
        raw_native = rec.get("native_id")
        doc_id = rec.get("doc_id")
        original = str(raw_native) if raw_native is not None else None
        if original is None:
            original = f"doc_{doc_id}" if doc_id is not None else f"line_{line_number}"
        base = original[:MAX_NATIVE_ID] or f"line_{line_number}"
        native_id = _unique_native_id(base, counts, used)

        request = by_native.get(original)
        if request is None and isinstance(doc_id, int) and not isinstance(doc_id, bool):
            request = by_doc.get(doc_id)
        yield build_instance_row(
            rec, native_id, primary_metric, offset, _line_length(line), request
        )


def batched(rows: Iterator[dict[str, Any]], size: int = BATCH_SIZE) -> Iterator[list[dict]]:
    batch: list[dict[str, Any]] = []
    for row in rows:
        batch.append(row)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch
