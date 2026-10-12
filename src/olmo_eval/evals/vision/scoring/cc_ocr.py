"""Scoring for CC-OCR's multi-scene OCR track, vendored from the official evaluator.

Source: ``Benchmarks/CC-OCR/evaluation/evaluator/ocr_evaluator.py`` of
https://github.com/AlibabaResearch/AdvancedLiterateMachinery (MIT). Behavior is preserved
exactly; the one mechanical change is that a sample is scored on its own and returns its
sufficient statistics, so the harness can score responses as they arrive.

A prediction is compared with the reference as a multiset of basic units: characters for the
Chinese sub-datasets (those whose name contains ``zh``), lower-cased alphanumeric-only words
for the rest. ``###`` / ``***`` markers and whitespace layout are ignored.

:class:`CcOcrScorer` records each sample's statistics as its scorer result (saved with the
predictions), and the metrics reduce them the way the official evaluator does: a sub-dataset's
score is its ``macro_f1`` (mean per-image F1), and the track score is the unweighted mean over
sub-datasets. The evaluator's secondary ``micro_f1`` (pooled over a sub-dataset's images) is
reported alongside.

Required ``instance.metadata``: ``dataset`` (the benchmark's ``split`` column) and ``answer``.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from olmo_eval.common.metrics.base import Metric
from olmo_eval.common.scorers.base import Scorer, get_scorer_result, set_scorer_result
from olmo_eval.common.types import Instance, LMOutput, Response
from olmo_eval.evals.vision.scoring.common import response_text

SCORER_NAME = "cc_ocr"


def ocr_eval_config(dataset: str) -> dict[str, bool]:
    """The official tokenization switches for a multi-scene sub-dataset."""
    is_word_level = "zh" not in dataset
    return {"word_level": is_word_level, "alphanum_only": is_word_level, "lowercase": True}


def _token_normalize(token_text: str, is_lower: bool, is_alphanum_only: bool) -> str:
    if is_lower:
        token_text = token_text.lower()
    if is_alphanum_only:
        token_text = re.sub("[^A-Za-z0-9]+", "", token_text)
    return token_text


def text_normalize_and_tokenize(
    text: str, is_keep_blank: bool = True, is_lower: bool = True, is_alphanum_only: bool = False
) -> list[str]:
    text = text.replace("\t", " ").replace("\n", " ").replace("###", "").replace("***", "")
    text = re.sub(r"\s+", " ", text)
    if not is_keep_blank:
        text = text.replace(" ", "")
    text_tokens = text.split(" ") if is_keep_blank else list(text)
    normalized = [_token_normalize(t, is_lower, is_alphanum_only) for t in text_tokens]
    return [x for x in normalized if len(x) > 0]


def score_ocr_sample(prediction: str, answer: str, dataset: str) -> dict[str, float]:
    """Multiset unit overlap for one image, plus its per-image recall/precision/f1."""
    config = ocr_eval_config(dataset)
    args = (config["word_level"], config["lowercase"], config["alphanum_only"])
    preds = text_normalize_and_tokenize(str(prediction).strip(), *args)
    gts = text_normalize_and_tokenize(str(answer).strip(), *args)

    pred_counter = Counter(preds)
    right_num = sum(min(count, pred_counter.get(token, 0)) for token, count in Counter(gts).items())
    recall = right_num / (len(gts) + 1e-9)
    precision = right_num / (len(preds) + 1e-9)
    return {
        "right_num": right_num,
        "gt_num": len(gts),
        "pred_num": len(preds),
        "recall": recall,
        "precision": precision,
        "f1": 2 * recall * precision / (recall + precision + 1e-9),
    }


@dataclass(frozen=True, slots=True)
class CcOcrScorer(Scorer):
    """Per-image F1 of the multiset overlap between prediction and reference units."""

    name: str = SCORER_NAME

    def score(self, instance: Instance, output: LMOutput) -> float:
        meta = instance.metadata
        stats = score_ocr_sample(response_text(output), meta["answer"], meta["dataset"])
        set_scorer_result(output, self.name, {"dataset": meta["dataset"], **stats})
        return float(stats["f1"])


def _result(response: Response) -> dict[str, Any] | None:
    return get_scorer_result(response.outputs[0], SCORER_NAME) if response.outputs else None


def _results_by_dataset(responses: Sequence[Response]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for response in responses:
        result = _result(response)
        if result:
            grouped.setdefault(result["dataset"], []).append(result)
    return grouped


def _dataset_value(results: list[dict], kind: Literal["macro_f1", "micro_f1"]) -> float:
    """One sub-dataset's score, with the official evaluator's smoothing constants."""
    if kind == "macro_f1":
        return sum(r["f1"] for r in results) / (len(results) + 1e-9)
    if kind == "micro_f1":
        right = sum(r["right_num"] for r in results)
        recall = right / (sum(r["gt_num"] for r in results) + 1e-9)
        precision = right / (sum(r["pred_num"] for r in results) + 1e-9)
        return 2 * recall * precision / (recall + precision + 1e-9)
    raise ValueError(f"unknown CC-OCR statistic {kind!r}")


@dataclass(frozen=True)
class CcOcrDatasetMetric(Metric):
    """Macro F1 of one sub-dataset (e.g. ``TotalText``)."""

    name: str  # type: ignore[misc]
    scorer: Scorer  # type: ignore[misc]
    dataset: str = ""

    def compute(self, responses: Sequence[Response]) -> float:
        results = _results_by_dataset(responses).get(self.dataset)
        return _dataset_value(results, "macro_f1") if results else 0.0

    def compute_instance(self, response: Response) -> float | None:
        """The image's F1, for images of this sub-dataset only."""
        result = _result(response)
        return float(result["f1"]) if result and result["dataset"] == self.dataset else None

    def supports_scorer_fallback(self) -> bool:
        return False


@dataclass(frozen=True)
class CcOcrTrackMetric(Metric):
    """Unweighted mean of the sub-dataset scores, for one statistic."""

    name: str  # type: ignore[misc]
    scorer: Scorer  # type: ignore[misc]
    kind: Literal["macro_f1", "micro_f1"] = "macro_f1"

    def compute(self, responses: Sequence[Response]) -> float:
        values = [_dataset_value(r, self.kind) for r in _results_by_dataset(responses).values()]
        return sum(values) / len(values) if values else 0.0

    def compute_instance(self, response: Response) -> float | None:
        """The image's F1: on a single image, both statistics reduce to it."""
        result = _result(response)
        return float(result["f1"]) if result else None

    def supports_scorer_fallback(self) -> bool:
        # The track averages sub-datasets, not images.
        return False
