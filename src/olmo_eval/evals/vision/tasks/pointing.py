"""Shared base task and metrics for the image-pointing benchmarks.

The pointing benchmarks (``pixmo_points_eval``, ``sa_co_gold_subset``) are a separate family from
the image-QA tasks: the model is asked ``"Point to <thing>."`` and scored by point-in-mask
precision/recall/f1 (see :class:`olmo_eval.common.scorers.pointing.PointingScorer`).  This module
provides:

* :class:`PointingTask` — caches instances and formats the CHAT+image request (its own base, not a
  subclass of the image-QA task).
* :class:`PointingMetric` + :func:`pointing_metrics` — frequency-bucketed and (optionally) weighted
  means over the per-example results, reproducing mm_olmo's ``SegmentionPointingEval`` /
  ``SACoGoldPointEvaluator`` metric key sets.

It reuses only generic, task-agnostic helpers (image loading + data-root resolution) imported from
the image-QA base; nothing image-QA-specific is shared.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from olmo_eval.common.metrics.base import Metric
from olmo_eval.common.scorers.base import Scorer
from olmo_eval.common.types import Instance, Response
from olmo_eval.evals.tasks.common.base import TaskConfig
from olmo_eval.evals.vision.scoring.pointing import require_scoring_dependencies
from olmo_eval.evals.vision.scoring.prompts import (
    POINTING_STYLE,
    apply_style_prefix,
    build_pointing_prompt,
)
from olmo_eval.evals.vision.tasks.base import VisionTask

__all__ = [
    "ModelPromptPointingTask",
    "PointingMetric",
    "PointingTask",
    "pointing_metrics",
    "presence_metrics",
]


class PointingTask(VisionTask):
    """Base class for the image-pointing benchmark tasks."""

    #: Image decoding builds the instances; the scorer decodes COCO-RLE masks with
    #: pycocotools and matches points to them with scipy. All three are needed
    #: whichever provider runs the task.
    dependencies = ["pillow", "pycocotools", "scipy"]

    @property
    def instances(self) -> Iterator[Instance]:
        # Probe the scoring backends while preparing instances, so an environment
        # missing pycocotools/scipy fails before a model is loaded instead of at
        # the first scored example.
        require_scoring_dependencies()
        return super().instances


class StylePrefixMixin:
    """Applies the checkpoint's ``"<style>:"`` prompt prefix to a pre-built question.

    For benchmarks that supply their own question, the prompt still has to follow the
    checkpoint: mm_olmo's ``DataFormatter`` keys the prefix off the example's *style*, so a
    ``style_and_length``/``_v2`` checkpoint saw ``"pointing: <q>"`` / ``"point_count: <q>"``
    in training while a ``demo_or_style_v*`` one saw the bare question. Set
    ``-o system_prompt_style=style_and_length_v2`` for a pretrain checkpoint; the default
    matches the instruction-tuned models, as in :class:`ModelPromptPointingTask`.

    Subclasses set :data:`style` to the mm_olmo style name for the benchmark.
    """

    #: mm_olmo style name for this benchmark's examples ("pointing" / "point_count").
    style: str = POINTING_STYLE
    default_system_prompt_style = "demo_or_style_v2"

    if TYPE_CHECKING:  # supplied by the Task this is mixed into
        config: TaskConfig

    def apply_family_prefix(self, question: str) -> str:
        """Return ``question`` prefixed for the checkpoint's prompt family."""
        return apply_style_prefix(
            question,
            self.config.system_prompt_style or self.default_system_prompt_style,
            self.style,
        )


class ModelPromptPointingTask(PointingTask):
    """Pointing task whose prompt the model's own formatter builds from a bare label.

    These are mm_olmo's ``_mp`` variants. The prompt form follows the checkpoint
    rather than the benchmark, so both settings are overridable per run
    (``-o prompt_templates=none -o system_prompt_style=style_and_length_v2`` for a
    pretrain checkpoint); the class defaults match the instruction-tuned models.
    """

    default_prompt_templates = "uber_model_v2"
    default_system_prompt_style = "demo_or_style_v2"

    def format_query(self, label: str, index: int) -> str:
        """Render the prompt for ``label`` at build-order position ``index``."""
        return build_pointing_prompt(
            label,
            index,
            prompt_templates=self.config.prompt_templates or self.default_prompt_templates,
            system_prompt=self.config.system_prompt_style or self.default_system_prompt_style,
        )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

# Frequency buckets keyed on the ground-truth instance count, matching mm_olmo
# (low/med_freq overlap intentionally, as in ``SegmentionPointingEval``).
_BUCKETS: dict[str, Callable[[int], bool]] = {
    "all": lambda c: True,
    "zero": lambda c: c == 0,
    "single": lambda c: c == 1,
    "low_freq": lambda c: c <= 3,
    "med_freq": lambda c: 3 <= c <= 10,
    "high_freq": lambda c: c >= 10,
}

_KINDS: tuple[str, ...] = ("precision", "recall", "f1")


def _pointing_results(responses: Sequence[Response]) -> Iterator[dict]:
    for response in responses:
        for output in response.outputs:
            if output.metadata and "pointing_result" in output.metadata:
                yield output.metadata["pointing_result"]


def _pointing_result(response: Response) -> dict | None:
    for output in response.outputs:
        if output.metadata and "pointing_result" in output.metadata:
            return output.metadata["pointing_result"]
    return None


@dataclass(frozen=True)
class PointingMetric(Metric):
    """Mean of one field (``precision`` / ``recall`` / ``f1``) over a count bucket.

    With ``weighted=True`` the mean is weighted by each example's ``weight`` (the SACoGold
    primary metrics); otherwise it is a simple mean (mm_olmo's ``global_mean``).
    """

    name: str  # type: ignore[misc]
    scorer: Scorer  # type: ignore[misc]
    kind: str = "f1"
    bucket: str = "all"
    weighted: bool = False

    def compute(self, responses: Sequence[Response]) -> float:
        in_bucket = _BUCKETS[self.bucket]
        num = 0.0
        den = 0.0
        for result in _pointing_results(responses):
            if not in_bucket(int(result["count"])):
                continue
            weight = float(result["weight"]) if self.weighted else 1.0
            num += float(result[self.kind]) * weight
            den += weight
        return num / den if den else 0.0

    def compute_instance(self, response: Response) -> float | None:
        """The example's own ``kind`` value, or ``None`` outside this metric's bucket.

        Weighted aggregates are not means of per-instance values, so weighted
        metrics store nothing rather than a misleading unweighted number.
        """
        if self.weighted:
            return None
        result = _pointing_result(response)
        if result is None or not _BUCKETS[self.bucket](int(result["count"])):
            return None
        return float(result[self.kind])

    def supports_scorer_fallback(self) -> bool:
        # The scorer channel is f1; it is not precision/recall or any bucketed value.
        return False


@dataclass(frozen=True)
class PointingErrorCountMetric(Metric):
    """How many examples failed to score (malformed ground truth).

    Reported alongside the point metrics so a partially unscorable run is visible
    in ``metrics.json`` instead of only in the logs.
    """

    name: str  # type: ignore[misc]
    scorer: Scorer  # type: ignore[misc]

    def compute(self, responses: Sequence[Response]) -> float:
        return float(sum(1 for result in _pointing_results(responses) if result.get("error")))

    def compute_instance(self, response: Response) -> float | None:
        result = _pointing_result(response)
        if result is None:
            return None
        return 1.0 if result.get("error") else 0.0

    def supports_scorer_fallback(self) -> bool:
        return False

    def display_format(self) -> str:
        return "raw"


def pointing_metrics(
    scorer: Scorer,
    *,
    buckets: Sequence[str],
    weighted_primary: bool = False,
) -> tuple[Metric, ...]:
    """Build the pointing metric family for one shared scorer.

    Returns the primary ``precision``/``recall``/``f1`` (over all examples, weighted iff
    ``weighted_primary``) followed by simple-mean ``{bucket}_{kind}`` metrics for each requested
    bucket — reproducing the exact key set of the corresponding mm_olmo evaluator.
    """
    metrics: list[Metric] = [
        PointingMetric(name=kind, scorer=scorer, kind=kind, bucket="all", weighted=weighted_primary)
        for kind in _KINDS
    ]
    for bucket in buckets:
        metrics.extend(
            PointingMetric(name=f"{bucket}_{kind}", scorer=scorer, kind=kind, bucket=bucket)
            for kind in _KINDS
        )
    metrics.append(PointingErrorCountMetric(name="n_scoring_errors", scorer=scorer))
    return tuple(metrics)


@dataclass(frozen=True)
class PresenceMetric(Metric):
    """Abstention accuracy over the examples where the phrase is wholly absent or present.

    ``is_absent_acc`` is the fraction of examples with no ground truth for any annotator
    where the model pointed at nothing; ``is_present_acc`` the fraction of examples with
    ground truth from every annotator where it pointed at something. Examples where the
    annotators disagree about presence count toward neither, as in mm_olmo's
    ``SACoGoldPointEvaluator``.
    """

    name: str  # type: ignore[misc]
    scorer: Scorer  # type: ignore[misc]
    #: ``"no_gt"`` selects the absent examples, ``"has_gt"`` the present ones.
    gt_field: str = "no_gt"
    #: Whether pointing at something is the correct behavior for those examples.
    expect_prediction: bool = False

    def compute(self, responses: Sequence[Response]) -> float:
        num = 0
        den = 0
        for result in _pointing_results(responses):
            if not result.get(self.gt_field):
                continue
            den += 1
            if bool(result.get("made_prediction")) is self.expect_prediction:
                num += 1
        return num / den if den else 0.0

    def compute_instance(self, response: Response) -> float | None:
        result = _pointing_result(response)
        if result is None or not result.get(self.gt_field):
            return None
        return 1.0 if bool(result.get("made_prediction")) is self.expect_prediction else 0.0

    def supports_scorer_fallback(self) -> bool:
        return False


def presence_metrics(scorer: Scorer) -> tuple[Metric, ...]:
    """The ``is_absent_acc`` / ``is_present_acc`` pair for one shared scorer."""
    return (
        PresenceMetric(
            name="is_absent_acc", scorer=scorer, gt_field="no_gt", expect_prediction=False
        ),
        PresenceMetric(
            name="is_present_acc", scorer=scorer, gt_field="has_gt", expect_prediction=True
        ),
    )
