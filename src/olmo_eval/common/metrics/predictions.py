"""Helpers for augmenting prediction payloads with exact per-instance metric keys."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from olmo_eval.common.metrics.base import (
    BPBMetricByteAvg,
    BPBMetricInstanceAvg,
    CorpusPerplexityMetric,
    GreedyAccuracyMetric,
    LogprobMCAccuracyMetric,
    LogprobPerCharMCAccuracyMetric,
    LogprobPerTokenMCAccuracyMetric,
    LogprobUncondMCAccuracyMetric,
    MeanPerplexityMetric,
    Metric,
    PassAtKMetric,
    PassPowKMetric,
)
from olmo_eval.common.types import Response
from olmo_eval.common.utils import compute_pass_at_k, compute_pass_pow_k


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    if math.isnan(value):
        return None
    return value


def _metric_scorer_name(metric: Metric) -> str | None:
    try:
        return metric.scorer().name
    except Exception:
        return None


def _nested_metric_value(
    instance_metrics: Mapping[str, Any],
    outer_key: str,
    inner_key: str,
) -> float | None:
    outer = instance_metrics.get(outer_key)
    if not isinstance(outer, Mapping):
        return None
    return _to_float(outer.get(inner_key))


def _default_metric_value(
    instance_metrics: Mapping[str, Any],
    *,
    metric_name: str,
    scorer_name: str,
    allow_scorer_fallback: bool,
) -> float | None:
    exact_metric_value = _nested_metric_value(instance_metrics, metric_name, scorer_name)
    if exact_metric_value is not None:
        return exact_metric_value

    metric_named_value = _nested_metric_value(instance_metrics, metric_name, metric_name)
    if metric_named_value is not None:
        return metric_named_value

    if allow_scorer_fallback:
        return _nested_metric_value(instance_metrics, scorer_name, scorer_name)

    return None


def normalize_prediction_instance_metrics(
    prediction: dict[str, Any],
) -> dict[str, dict[str, float]]:
    """Canonicalize stored instance metrics to ``{metric: {scorer: value}}``.

    Older payloads may store flat ``{metric: value}`` entries. We normalize those to
    ``{metric: {metric: value}}`` so downstream code can rely on one consistent shape.
    Non-numeric values are ignored.
    """
    raw_instance_metrics = prediction.get("instance_metrics")
    if not isinstance(raw_instance_metrics, Mapping):
        prediction["instance_metrics"] = {}
        return prediction["instance_metrics"]

    normalized_instance_metrics: dict[str, dict[str, float]] = {}
    for raw_outer_key, raw_value in raw_instance_metrics.items():
        outer_key = str(raw_outer_key)
        if isinstance(raw_value, Mapping):
            nested_scores = {
                str(raw_inner_key): score
                for raw_inner_key, inner_value in raw_value.items()
                if (score := _to_float(inner_value)) is not None
            }
            if nested_scores:
                normalized_instance_metrics[outer_key] = nested_scores
            continue

        scalar_score = _to_float(raw_value)
        if scalar_score is not None:
            normalized_instance_metrics[outer_key] = {outer_key: scalar_score}

    prediction["instance_metrics"] = normalized_instance_metrics
    return normalized_instance_metrics


def _prediction_outputs(prediction: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    outputs = prediction.get("model_output")
    if not isinstance(outputs, list):
        return []
    return [output for output in outputs if isinstance(output, Mapping)]


def _gold_idx(prediction: Mapping[str, Any], outputs: Sequence[Mapping[str, Any]]) -> int | None:
    label = prediction.get("label")
    if isinstance(label, int) and 0 <= label < len(outputs):
        return label
    if outputs:
        return 0
    return None


def _sum_logits(output: Mapping[str, Any]) -> float | None:
    return _to_float(output.get("sum_logits"))


def _num_tokens(output: Mapping[str, Any]) -> int | None:
    value = output.get("num_tokens")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    num_tokens = int(value)
    return num_tokens if num_tokens > 0 else None


def _num_chars(output: Mapping[str, Any]) -> int:
    value = output.get("num_chars")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return max(int(value), 0)


def _sample_metric_score(output: Mapping[str, Any], scorer_name: str) -> float | None:
    sample_metrics = output.get("sample_metrics")
    if not isinstance(sample_metrics, Mapping):
        return None
    outer = sample_metrics.get(scorer_name)
    if not isinstance(outer, Mapping):
        return None
    return _to_float(outer.get(scorer_name))


def _compute_metric_value_from_prediction(
    prediction: Mapping[str, Any],
    metric: Metric,
    *,
    scorer_name: str,
    instance_metrics: Mapping[str, Any],
) -> float | None:
    outputs = _prediction_outputs(prediction)

    if isinstance(metric, (PassAtKMetric, PassPowKMetric)):
        sample_scores = [
            score
            for output in outputs
            if (score := _sample_metric_score(output, scorer_name)) is not None
        ]
        if not sample_scores:
            fallback = _nested_metric_value(instance_metrics, scorer_name, scorer_name)
            if fallback is None or len(outputs) > 1:
                return None
            sample_scores = [fallback]

        n = len(sample_scores)
        c = sum(1 for score in sample_scores if score > 0.5)
        if isinstance(metric, PassAtKMetric):
            return float(compute_pass_at_k(n, c, min(metric.k, n)))
        return float(compute_pass_pow_k(n, c, metric.k))

    if isinstance(metric, GreedyAccuracyMetric):
        gold_idx = _gold_idx(prediction, outputs)
        if gold_idx is None:
            return None
        return 1.0 if outputs[gold_idx].get("is_greedy", False) else 0.0

    if isinstance(metric, (BPBMetricInstanceAvg, BPBMetricByteAvg)):
        gold_idx = _gold_idx(prediction, outputs)
        if gold_idx is None:
            return None
        return _to_float(outputs[gold_idx].get("bits_per_byte"))

    if isinstance(metric, MeanPerplexityMetric):
        gold_idx = _gold_idx(prediction, outputs)
        if gold_idx is None:
            return None
        output = outputs[gold_idx]
        sum_logits = _sum_logits(output)
        num_tokens = _num_tokens(output)
        if sum_logits is None or num_tokens is None:
            return None
        return float(math.exp(-(sum_logits / num_tokens)))

    if isinstance(metric, CorpusPerplexityMetric):
        if not outputs:
            return None
        sum_logits = _sum_logits(outputs[0])
        num_tokens = _num_tokens(outputs[0])
        if sum_logits is None or num_tokens is None:
            return None
        return float(math.exp(-(sum_logits / num_tokens)))

    if isinstance(metric, LogprobMCAccuracyMetric):
        gold_idx = _gold_idx(prediction, outputs)
        scores = [_sum_logits(output) for output in outputs]
        if gold_idx is None or any(score is None for score in scores):
            return None
        resolved_scores = [float(score) for score in scores if score is not None]
        return 1.0 if resolved_scores.index(max(resolved_scores)) == gold_idx else 0.0

    if isinstance(metric, LogprobPerCharMCAccuracyMetric):
        gold_idx = _gold_idx(prediction, outputs)
        normalized_scores: list[float] = []
        if gold_idx is None or not outputs:
            return None
        for output in outputs:
            sum_logits = _sum_logits(output)
            if sum_logits is None:
                return None
            normalized_scores.append(sum_logits / max(_num_chars(output), 1))
        return 1.0 if normalized_scores.index(max(normalized_scores)) == gold_idx else 0.0

    if isinstance(metric, LogprobPerTokenMCAccuracyMetric):
        gold_idx = _gold_idx(prediction, outputs)
        normalized_scores: list[float] = []
        if gold_idx is None or not outputs:
            return None
        for output in outputs:
            sum_logits = _sum_logits(output)
            if sum_logits is None:
                return None
            normalized_scores.append(sum_logits / max(_num_tokens(output) or 0, 1))
        return 1.0 if normalized_scores.index(max(normalized_scores)) == gold_idx else 0.0

    if isinstance(metric, LogprobUncondMCAccuracyMetric):
        gold_idx = _gold_idx(prediction, outputs)
        if gold_idx is None or not outputs or len(outputs) % 2 != 0:
            return None
        num_choices = len(outputs) // 2
        cond_outputs = outputs[:num_choices]
        uncond_outputs = outputs[num_choices:]
        scores: list[float] = []
        for cond_output, uncond_output in zip(cond_outputs, uncond_outputs, strict=True):
            cond_logits = _sum_logits(cond_output)
            uncond_logits = _sum_logits(uncond_output)
            if cond_logits is None or uncond_logits is None:
                return None
            scores.append(cond_logits - uncond_logits)
        return 1.0 if scores.index(max(scores)) == gold_idx else 0.0

    return _default_metric_value(
        instance_metrics,
        metric_name=metric.name,
        scorer_name=scorer_name,
        allow_scorer_fallback=metric.supports_scorer_fallback(),
    )


def augment_prediction_instance_metrics(
    prediction: dict[str, Any],
    metrics: Sequence[Metric],
    *,
    response: Response | None = None,
) -> None:
    """Populate exact ``metric:scorer``-style instance keys in a prediction payload.

    ``build_predictions()`` can supply the original ``Response`` for the most faithful
    computation. Without it, the helper works from the prediction payload alone, which
    still covers logprob-MC and other metrics derivable from the stored output summary.
    """
    instance_metrics = normalize_prediction_instance_metrics(prediction)

    for metric in metrics:
        scorer_name = _metric_scorer_name(metric)
        if scorer_name is None:
            continue

        exact_value: float | None = None
        if response is not None:
            exact_value = metric.compute_instance(response)
        if exact_value is None:
            if _nested_metric_value(instance_metrics, metric.name, scorer_name) is not None:
                continue
            exact_value = _compute_metric_value_from_prediction(
                prediction,
                metric,
                scorer_name=scorer_name,
                instance_metrics=instance_metrics,
            )
        if exact_value is None:
            continue

        instance_metrics.setdefault(metric.name, {})[scorer_name] = float(exact_value)
