"""The judge-graded multimodal safety task base and its metric families.

Each safety benchmark grades a response with its own judge prompts and stores
one result dict per response (``output.metadata["safety_result"]``) holding the
benchmark's per-example verdicts, each ``1.0``/``0.0`` or ``None`` when the
judge returned nothing usable. Rates are means over the examples that have a
verdict, so a judge outage shows up in the ``*_judge_errors`` counts rather
than as a shifted rate.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from olmo_eval.common.metrics.base import Metric
from olmo_eval.common.scorers.base import Scorer
from olmo_eval.common.types import Instance, LMOutput, Response
from olmo_eval.evals.vision.tasks.image_qa import ImageQATask

RESULT_KEY = "safety_result"


@dataclass(frozen=True)
class SafetyJudgeScorer(Scorer):
    """Score channel for a benchmark's judge.

    Grading happens task-level (judge calls are gathered under the runner's
    scoring concurrency) and stores ``score:<name>`` per output; this scorer
    exposes the stored value so the metric plumbing stays uniform.
    """

    name: str = "safety_judge"

    def score(self, instance: Instance, output: LMOutput) -> float:
        value = (output.metadata or {}).get(f"score:{self.name}", 0.0)
        return float(value) if isinstance(value, (int, float)) else 0.0


def store_result(response: Response, scorer: Scorer, result: dict[str, Any], key: str) -> None:
    """Attach ``result`` to the response's first output; ``result[key]`` is its score."""
    value = result.get(key)
    score = float(value) if value is not None else 0.0
    response.scores[scorer.name] = score
    if not response.outputs:
        return
    output = response.outputs[0]
    if output.metadata is None:
        output.metadata = {}
    output.metadata[RESULT_KEY] = result
    output.metadata[f"score:{scorer.name}"] = score


def _result_for(response: Response) -> dict | None:
    for output in response.outputs:
        if output.metadata and RESULT_KEY in output.metadata:
            return output.metadata[RESULT_KEY]
    return None


def _results(responses: Sequence[Response]) -> Iterator[tuple[Response, dict]]:
    for response in responses:
        result = _result_for(response)
        if result is not None:
            yield response, result


@dataclass(frozen=True)
class SafetyRateMetric(Metric):
    """Mean of one verdict over the examples the judge graded.

    ``where`` restricts the mean to examples whose instance metadata matches
    every given ``field: value`` pair (a category, an image type, ...).
    """

    name: str  # type: ignore[misc]
    scorer: Scorer  # type: ignore[misc]
    key: str = "unsafe"
    where: tuple[tuple[str, Any], ...] = field(default=())

    def _in_slice(self, response: Response) -> bool:
        meta = response.instance.metadata
        return all(meta.get(name) == value for name, value in self.where)

    def compute(self, responses: Sequence[Response]) -> float:
        values = [
            float(result[self.key])
            for response, result in _results(responses)
            if result.get(self.key) is not None and self._in_slice(response)
        ]
        return sum(values) / len(values) if values else 0.0

    def compute_instance(self, response: Response) -> float | None:
        result = _result_for(response)
        if result is None or result.get(self.key) is None or not self._in_slice(response):
            return None
        return float(result[self.key])

    def supports_pairwise_scorer_fallback(self) -> bool:
        return False


@dataclass(frozen=True)
class MacroRateMetric(Metric):
    """Unweighted mean, over the values of metadata field ``group``, of each group's rate.

    ``where`` restricts the examples first, as in :class:`SafetyRateMetric`.
    """

    name: str  # type: ignore[misc]
    scorer: Scorer  # type: ignore[misc]
    key: str = "unsafe"
    group: str = "category"
    where: tuple[tuple[str, Any], ...] = field(default=())

    def compute(self, responses: Sequence[Response]) -> float:
        by_group: dict[Any, list[float]] = {}
        for response, result in _results(responses):
            meta = response.instance.metadata
            if result.get(self.key) is None:
                continue
            if not all(meta.get(name) == value for name, value in self.where):
                continue
            by_group.setdefault(meta.get(self.group), []).append(float(result[self.key]))
        if not by_group:
            return 0.0
        rates = [sum(values) / len(values) for values in by_group.values()]
        return sum(rates) / len(rates)

    def compute_instance(self, response: Response) -> float | None:
        # A mean of group rates has no exact per-instance decomposition.
        return None

    def supports_pairwise_scorer_fallback(self) -> bool:
        return False


@dataclass(frozen=True)
class JudgeErrorCountMetric(Metric):
    """Number of examples that take a verdict but were left ungraded (judge error or
    unparsable reply)."""

    name: str  # type: ignore[misc]
    scorer: Scorer  # type: ignore[misc]
    key: str = "unsafe"

    def compute(self, responses: Sequence[Response]) -> float:
        return float(
            sum(
                1
                for _, result in _results(responses)
                if self.key in result and result[self.key] is None
            )
        )

    def compute_instance(self, response: Response) -> float | None:
        result = _result_for(response)
        if result is None or self.key not in result:
            return None
        return 1.0 if result[self.key] is None else 0.0

    def supports_pairwise_scorer_fallback(self) -> bool:
        return False


class SafetyJudgeTask(ImageQATask):
    """Base class for the judge-graded single-image safety benchmarks."""

    #: The judges call OpenAI.
    dependencies = ["pillow", "openai"]
    required_secrets = ("OPENAI_API_KEY",)


def response_text(response: Response, index: int = 0) -> str:
    """The text of output ``index`` (empty when the model produced none)."""
    if len(response.outputs) > index:
        return response.outputs[index].text or ""
    return ""
