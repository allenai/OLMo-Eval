"""Identity hashes and values derived from model names (spec 2.6)."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable
from typing import Any, Literal

MetricKind = Literal["binary", "bounded", "unbounded"]


def canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()


def sha256_hex(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode()
    return hashlib.sha256(data).hexdigest()


def model_id(name: str, model_hash: str) -> str:
    return sha256_hex(f"{name}\x00{model_hash}")[:12]


def instance_key_hash(native_id: str) -> int:
    """Signed int64 hash of a native_id; pairs instances across task results."""
    digest = hashlib.blake2b(native_id.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=True)


def fallback_model_hash(provider_config: dict[str, Any]) -> str:
    return sha256_hex(canonical_json(provider_config))[:16]


_SETTINGS_DROP = {
    "model",
    "alias",
    "revision",
    "tokenizer",
    "base_url",
    "api_base",
    "force_download",
    "max_concurrency",
    "num_instances",
    "required_secrets",
}
_SETTINGS_DROP_KWARGS = {
    "attention_backend",
    "tensor_parallel_size",
    "pipeline_parallel_size",
    "expert_parallel_size",
    "enable_prefix_caching",
    "gpu_memory_utilization",
    "max_num_batched_tokens",
    "max_num_seqs",
}


def settings_hash(provider_config: dict[str, Any]) -> str:
    """Hash of the provider config without path-like and operational keys.

    Checkpoints of one series evaluated with the same settings share this hash.
    """
    rest = {k: v for k, v in provider_config.items() if k not in _SETTINGS_DROP}
    kwargs = rest.get("kwargs")
    if isinstance(kwargs, dict):
        rest["kwargs"] = {k: v for k, v in kwargs.items() if k not in _SETTINGS_DROP_KWARGS}
    return sha256_hex(canonical_json(rest))[:12]


_STEP_RE = re.compile(r"(?i)(?:^|[^a-z0-9])step[-_]?(\d+)")
_TOKENS_RE = re.compile(r"(?i)tokens[-_]?(\d+(?:\.\d+)?)([KMBT])")
_SERIES_STRIP_RE = re.compile(
    r"(?i)[-_/.]?(?:stage\d+[-_])?step[-_]?\d+(?:[-_]tokens[-_]?\d+(?:\.\d+)?[KMBT])?(?:[-_]?hf)?"
)
_FAMILY_RE = re.compile(
    r"^(olmo|llama|qwen|gemma|mistral|mixtral|phi|deepseek|gpt|claude|gemini|tulu|smollm|granite"
    r"|command|yi)[-_ ]?(\d+(?:\.\d+)?)?"
)
_TOKEN_SCALE = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}


def derive_step(*candidates: str | None) -> int | None:
    for candidate in candidates:
        if candidate:
            match = _STEP_RE.search(candidate)
            if match:
                return int(match.group(1))
    return None


def derive_tokens(*candidates: str | None) -> int | None:
    for candidate in candidates:
        if candidate:
            match = _TOKENS_RE.search(candidate)
            if match:
                return int(round(float(match.group(1)) * _TOKEN_SCALE[match.group(2).upper()]))
    return None


def derive_series(name: str) -> str:
    stripped = _SERIES_STRIP_RE.sub("", name, count=1).rstrip("-_/.")
    return stripped or name


def series_label(series: str) -> str:
    return series.rsplit("/", 1)[-1] if "/" in series else series


def derive_family(name: str, path: str) -> str | None:
    base = (name or path or "").rstrip("/").rsplit("/", 1)[-1].lower()
    if not base:
        return None
    match = _FAMILY_RE.match(base)
    if match:
        return re.sub(r"[-_ ]", "", match.group(1) + (match.group(2) or ""))
    head = re.split(r"[-_]", base, maxsplit=1)[0]
    return head or None


def metric_kind(values: Iterable[float]) -> MetricKind:
    finite = [v for v in values if v is not None and math.isfinite(v)]
    if finite and all(v in (0.0, 1.0) for v in finite):
        return "binary"
    if all(0.0 <= v <= 1.0 for v in finite):
        return "bounded"
    return "unbounded"


def flatten_metrics(nested: dict[str, Any]) -> dict[str, float | None]:
    """{metric: {scorer: value}} -> {"metric:scorer": value}. Non-finite values become null."""
    flat: dict[str, float | None] = {}
    for metric, scorers in nested.items():
        if isinstance(scorers, dict):
            for scorer, value in scorers.items():
                flat[f"{metric}:{scorer}"] = finite_or_none(value)
        else:
            flat[metric] = finite_or_none(scorers)
    return flat


def finite_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def metric_name(metric_key: str) -> str:
    """ "acc:default" -> "acc"."""
    return metric_key.rsplit(":", 1)[0] if ":" in metric_key else metric_key
