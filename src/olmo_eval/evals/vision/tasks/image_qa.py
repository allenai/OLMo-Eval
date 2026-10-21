"""The image-QA task base and its metric families."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from olmo_eval.common.metrics.base import Metric
from olmo_eval.common.scorers.base import Scorer
from olmo_eval.common.types import Response
from olmo_eval.evals.vision.tasks.base import VisionTask


class ImageQATask(VisionTask):
    """Base class for the single-image QA benchmarks."""


@dataclass(frozen=True)
class MeanScorerMetric(Metric):
    """Mean of a scorer's per-response score (the mm_olmo ``global_mean``)."""

    name: str  # type: ignore[misc]
    scorer: Scorer  # type: ignore[misc]

    def compute(self, responses: Sequence[Response]) -> float:
        if not responses:
            return 0.0
        scorer_name = self.scorer().name
        return sum(r.scores.get(scorer_name, 0.0) for r in responses) / len(responses)


@dataclass(frozen=True)
class ChartQaSubsetMetric(Metric):
    """ChartQA metric over all / human / augmented examples.

    Subset membership comes from ``instance.metadata["is_human"]``, matching
    the ``_human`` / ``_aug`` breakdowns of mm_olmo's ``VqaEval``.
    """

    name: str  # type: ignore[misc]
    scorer: Scorer  # type: ignore[misc]
    subset: str = "all"  # all | human | aug

    def compute(self, responses: Sequence[Response]) -> float:
        scorer_name = self.scorer().name
        vals = [r.scores.get(scorer_name, 0.0) for r in responses if self._in_subset(r)]
        return sum(vals) / len(vals) if vals else 0.0

    def _in_subset(self, response: Response) -> bool:
        if self.subset == "all":
            return True
        is_human = bool(response.instance.metadata.get("is_human"))
        return is_human if self.subset == "human" else not is_human

    def compute_instance(self, response: Response) -> float | None:
        """The example's own score, or ``None`` when it is outside this subset."""
        is_human = bool(response.instance.metadata.get("is_human"))
        if self.subset == "human" and not is_human:
            return None
        if self.subset == "augmented" and is_human:
            return None
        value = response.scores.get(self.scorer().name)
        return float(value) if isinstance(value, (int, float)) else None

    def supports_scorer_fallback(self) -> bool:
        return False


@dataclass(frozen=True)
class Ai2dMetric(Metric):
    """AI2D accuracy split by box rendering.

    abc-label questions count toward exactly one variant (transparent or
    opaque, per ``has_transparent_box``); questions without abc labels count
    toward both — matching ``mc_ai2d_opaque`` / ``mc_ai2d_transparent`` in
    mm_olmo's ``VqaEval``.
    """

    name: str  # type: ignore[misc]
    scorer: Scorer  # type: ignore[misc]
    transparent: bool = False

    def compute(self, responses: Sequence[Response]) -> float:
        vals: list[float] = []
        for response in responses:
            for output in response.outputs:
                if not output.metadata or "ai2d_result" not in output.metadata:
                    continue
                result = output.metadata["ai2d_result"]
                if result["abc_label"]:
                    if self.transparent and not result["has_transparent_box"]:
                        continue
                    if not self.transparent and result["has_transparent_box"]:
                        continue
                vals.append(result["is_correct"])
        return sum(vals) / len(vals) if vals else 0.0

    def compute_instance(self, response: Response) -> float | None:
        for output in response.outputs:
            if not output.metadata or "ai2d_result" not in output.metadata:
                continue
            result = output.metadata["ai2d_result"]
            if result["abc_label"] and result["has_transparent_box"] is not self.transparent:
                return None
            return float(result["is_correct"])
        return None

    def supports_scorer_fallback(self) -> bool:
        return False


def _point_count_results(responses: Sequence[Response]) -> Iterator[tuple[Response, dict]]:
    for response in responses:
        for output in response.outputs:
            if output.metadata and "point_count_result" in output.metadata:
                yield response, output.metadata["point_count_result"]


def _point_count_result(response: Response) -> dict | None:
    for output in response.outputs:
        if output.metadata and "point_count_result" in output.metadata:
            return output.metadata["point_count_result"]
    return None


@dataclass(frozen=True)
class PointCountMetric(Metric):
    """Mean of one field (``correct`` / ``close`` / ``valid``) of the count result."""

    name: str  # type: ignore[misc]
    scorer: Scorer  # type: ignore[misc]
    kind: str = "correct"

    def compute(self, responses: Sequence[Response]) -> float:
        vals = [result[self.kind] for _, result in _point_count_results(responses)]
        return sum(vals) / len(vals) if vals else 0.0

    def compute_instance(self, response: Response) -> float | None:
        result = _point_count_result(response)
        return float(result[self.kind]) if result is not None else None

    def supports_scorer_fallback(self) -> bool:
        # The scorer channel is `correct`; close/valid are their own fields.
        return False


@dataclass(frozen=True)
class PointCountPerCountMetric(Metric):
    """Counting accuracy restricted to examples with ground-truth count ``k``."""

    name: str  # type: ignore[misc]
    scorer: Scorer  # type: ignore[misc]
    k: int = 0

    def compute(self, responses: Sequence[Response]) -> float:
        vals = [
            result["correct"]
            for response, result in _point_count_results(responses)
            if int(response.instance.metadata["count"]) == self.k
        ]
        return sum(vals) / len(vals) if vals else 0.0

    def compute_instance(self, response: Response) -> float | None:
        result = _point_count_result(response)
        if result is None or int(response.instance.metadata["count"]) != self.k:
            return None
        return float(result["correct"])

    def supports_scorer_fallback(self) -> bool:
        return False


@dataclass(frozen=True)
class PointCountCategoryAverageMetric(Metric):
    """Macro average of per-count accuracies over the counts present."""

    name: str  # type: ignore[misc]
    scorer: Scorer  # type: ignore[misc]

    def compute(self, responses: Sequence[Response]) -> float:
        by_count: dict[int, list[float]] = {}
        for response, result in _point_count_results(responses):
            by_count.setdefault(int(response.instance.metadata["count"]), []).append(
                result["correct"]
            )
        if not by_count:
            return 0.0
        per_count = [sum(v) / len(v) for v in by_count.values()]
        return sum(per_count) / len(per_count)

    def compute_instance(self, response: Response) -> float | None:
        # A macro average over counts has no exact per-instance decomposition.
        return None

    def supports_scorer_fallback(self) -> bool:
        return False


# Counts present in the CountBench QA / PixMo Count eval sets.
POINT_COUNT_KS: tuple[int, ...] = tuple(range(2, 11))


def point_count_metrics(scorer: Scorer) -> tuple[Metric, ...]:
    """The full mm_olmo ``PointCountEval`` metric family for one shared scorer."""
    return (
        PointCountMetric(name="correct", scorer=scorer, kind="correct"),
        PointCountMetric(name="close", scorer=scorer, kind="close"),
        PointCountMetric(name="valid", scorer=scorer, kind="valid"),
        *(
            PointCountPerCountMetric(name=f"correct_{k}", scorer=scorer, k=k)
            for k in POINT_COUNT_KS
        ),
        PointCountCategoryAverageMetric(name="per_category_average", scorer=scorer),
    )
