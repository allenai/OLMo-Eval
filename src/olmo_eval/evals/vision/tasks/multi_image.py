"""Shared base task and metrics for the multi-image QA benchmarks.

The multi-image benchmarks (``muir_bench``, ``mmiu``, ``blink``) are a separate
family from the single-image QA tasks: each instance carries a *list* of images,
and scoring is the mm_olmo ``MuirBenchEval``-family protocol (response cleanup +
MMMU-style letter parsing; see
:class:`olmo_eval.evals.vision.scoring.multi_image.MultiImageMcScorer`).  This module
provides:

* :class:`MultiImageQATask` — caches instances and formats the CHAT request with
  the instance's image list, capped at ``max_images`` (mm_olmo's
  ``eval_molmo2.py`` sets ``mm_preprocessor.image.max_images = 20`` and the
  formatter truncates ``images[:max_images]``).
* :class:`MultiImageCategoryMetric` / :class:`MultiImageCountBucketMetric` —
  per-category and image-count-bucket accuracies reproducing the mm_olmo
  evaluator metric keys (``all``, per-task/sub-task/relationship, MMIU's
  ``num_images<=10`` / ``<=20`` / ``>20``).
* :func:`replace_images` — mm_olmo's ``<image>`` → ``"Image n"`` prompt rewrite
  (``academic_datasets.replace_images``).

It reuses only generic, task-agnostic helpers imported from the image-QA base;
nothing single-image-specific is shared.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from olmo_eval.common.metrics.base import Metric
from olmo_eval.common.scorers.base import Scorer
from olmo_eval.common.types import Instance, Response

# Generic (not single-image-QA-specific) data utilities.
from olmo_eval.evals.vision.data.images import capped_image_list
from olmo_eval.evals.vision.tasks.base import VisionTask

__all__ = [
    "MultiImageCategoryMetric",
    "MultiImageCountBucketMetric",
    "MultiImageQATask",
    "multi_image_mc_metrics",
    "replace_images",
]


def replace_images(question: str, options: list[str], max_images: int | None = None):
    """mm_olmo ``academic_datasets.replace_images``: rewrite ``<image>`` placeholders
    to ``"Image n"``, numbering sequentially through the question then the options."""
    all_strings = [question] + options
    total_images = sum(s.count("<image>") for s in all_strings)
    if max_images is not None:
        total_images = min(total_images, max_images)
    image_counter = 1

    def repl(match: re.Match[str]) -> str:
        # One counter across the question and every option, as mm_olmo numbers them.
        nonlocal image_counter
        if image_counter > total_images:
            return match.group(0)
        replacement = f"Image {image_counter}"
        image_counter += 1
        return replacement

    replaced = [re.sub(r"<image>", repl, s) for s in all_strings]
    return replaced[0], replaced[1:]


class MultiImageQATask(VisionTask):
    """Base class for the multi-image QA benchmark tasks."""

    #: mm_olmo eval-time image cap (``eval_molmo2.py`` sets ``max_images = 20``;
    #: the data formatter truncates the image list to this length).
    max_images: int = 20

    def _attach_images(self, instance: Instance) -> tuple[Any, ...] | None:
        images = instance.metadata.get("images")
        if images is None:
            return None
        # One lazy entry for the whole (capped) list: pixels decode in the provider,
        # never in the parent process or through the worker queue.
        return (capped_image_list(images, self.max_images),)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MultiImageCategoryMetric(Metric):
    """Mean scorer value over instances whose ``field`` metadata equals ``category``
    (``category=None`` means all instances — mm_olmo's ``all`` key)."""

    name: str  # type: ignore[misc]
    scorer: Scorer  # type: ignore[misc]
    field: str = "task"
    category: str | None = None

    def compute(self, responses: Sequence[Response]) -> float:
        scorer_name = self.scorer().name
        vals = [
            r.scores.get(scorer_name, 0.0)
            for r in responses
            if self.category is None or r.instance.metadata.get(self.field) == self.category
        ]
        return sum(vals) / len(vals) if vals else 0.0

    def compute_instance(self, response: Response) -> float | None:
        """The example's own score, or ``None`` when it is outside this category."""
        if (
            self.category is not None
            and response.instance.metadata.get(self.field) != self.category
        ):
            return None
        value = response.scores.get(self.scorer().name)
        return float(value) if isinstance(value, (int, float)) else None

    def supports_scorer_fallback(self) -> bool:
        return False


@dataclass(frozen=True)
class MultiImageCountBucketMetric(Metric):
    """Mean scorer value over instances bucketed by ``metadata["num_images"]``
    (mm_olmo ``MMIUEval``: ``<=10``, ``11-20``, ``>20``)."""

    name: str  # type: ignore[misc]
    scorer: Scorer  # type: ignore[misc]
    min_images: int = 0  # exclusive lower bound
    max_images: int | None = None  # inclusive upper bound, None = unbounded

    def compute(self, responses: Sequence[Response]) -> float:
        scorer_name = self.scorer().name
        vals = []
        for r in responses:
            num_images = int(r.instance.metadata["num_images"])
            if num_images <= self.min_images:
                continue
            if self.max_images is not None and num_images > self.max_images:
                continue
            vals.append(r.scores.get(scorer_name, 0.0))
        return sum(vals) / len(vals) if vals else 0.0

    def compute_instance(self, response: Response) -> float | None:
        """The example's own score, or ``None`` when it falls in another bucket."""
        num_images = int(response.instance.metadata["num_images"])
        if num_images <= self.min_images:
            return None
        if self.max_images is not None and num_images > self.max_images:
            return None
        value = response.scores.get(self.scorer().name)
        return float(value) if isinstance(value, (int, float)) else None

    def supports_scorer_fallback(self) -> bool:
        return False


def multi_image_mc_metrics(
    scorer: Scorer,
    categories: Sequence[str],
    field: str,
) -> tuple[Metric, ...]:
    """``all`` plus one per-category accuracy, using mm_olmo's metric key strings.

    The first metric (``all``) is the task's primary metric.
    """
    return (
        MultiImageCategoryMetric(name="all", scorer=scorer, field=field, category=None),
        *(
            MultiImageCategoryMetric(name=category, scorer=scorer, field=field, category=category)
            for category in categories
        ),
    )
