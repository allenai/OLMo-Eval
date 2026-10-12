"""Point-Bench (PointArena) scoring.

A port of the official evaluator (``model_evaluator.py`` in pointarena/pointarena): a prediction
succeeds when it has at least one point and every point it is scored on lies inside the
ground-truth mask. Outside the ``counting`` category only the first point is scored; in
``counting`` the number of points must also equal the annotated ``count``. Masks are binarized
as the official ``load_mask`` does (any channel > 127), and a point is inside when its truncated
integer pixel is in bounds and set.

mm_olmo's ``PointBenchEval`` scores the same predictions more strictly on multiple points (all
of them must be in the mask, in every category) and never checks the count; that rule is kept as
a secondary result so numbers can be set against mm_olmo-produced ones.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from olmo_eval.common.metrics.base import Metric
from olmo_eval.common.scorers.base import Scorer
from olmo_eval.common.types import Instance, LMOutput, Response
from olmo_eval.evals.vision.scoring.common import response_text
from olmo_eval.evals.vision.scoring.count_parsing import extract_image_points

__all__ = [
    "CATEGORIES",
    "PointBenchAccuracyMetric",
    "PointBenchScorer",
    "load_mask",
    "point_in_mask",
]

CATEGORIES: tuple[str, ...] = ("affordable", "spatial", "reasoning", "steerable", "counting")


def load_mask(path: str) -> np.ndarray:
    """A boolean ``HxW`` mask from a PNG, set where any channel exceeds 127."""
    from PIL import Image

    with Image.open(path) as image:
        array = np.array(image)
    if array.ndim == 2:
        return array > 127
    if array.ndim == 3:
        return np.any(array > 127, axis=2)
    raise ValueError(f"unexpected mask shape {array.shape} in {path}")


def point_in_mask(x: float, y: float, mask: np.ndarray, image_w: int, image_h: int) -> bool:
    """The official ``is_point_in_mask``: truncate, bounds-check against the image, look up."""
    px, py = int(x), int(y)
    if py < 0 or py >= image_h or px < 0 or px >= image_w:
        return False
    if py >= mask.shape[0] or px >= mask.shape[1]:
        # A few masks are smaller than their image; mm_olmo treats such points as misses.
        return False
    return bool(mask[py, px])


def score_point_bench(
    points: Sequence[tuple[float, float]],
    mask: np.ndarray,
    image_size: tuple[int, int],
    category: str,
    count: int | None,
) -> dict[str, Any]:
    """Official and mm_olmo-rule success for one example's predicted points."""
    image_w, image_h = image_size
    pts = list(points)
    if not pts or (category == "counting" and len(pts) != (count or 1)):
        official = False
    else:
        scored = pts if category == "counting" else pts[:1]
        official = all(point_in_mask(x, y, mask, image_w, image_h) for x, y in scored)
    all_points = bool(pts) and all(point_in_mask(x, y, mask, image_w, image_h) for x, y in pts)
    return {"official": official, "all_points": all_points, "n_points": len(pts)}


@dataclass(frozen=True, slots=True)
class PointBenchScorer(Scorer):
    """1.0 on an official Point-Bench success; both rules go to ``point_bench_result``."""

    name: str = "point_bench"

    def score(self, instance: Instance, output: LMOutput) -> float:
        meta = instance.metadata
        image_w, image_h = (int(v) for v in meta["image_size"])
        points = extract_image_points(response_text(output).strip(), image_w, image_h)
        result = score_point_bench(
            points,
            load_mask(meta["mask_path"]),
            (image_w, image_h),
            meta["category"],
            meta.get("count"),
        )
        if output.metadata is None:
            output.metadata = {}
        output.metadata["point_bench_result"] = result
        return float(result["official"])


def _result(response: Response) -> dict[str, Any] | None:
    for output in response.outputs:
        if output.metadata and "point_bench_result" in output.metadata:
            return output.metadata["point_bench_result"]
    return None


@dataclass(frozen=True)
class PointBenchAccuracyMetric(Metric):
    """Success rate under ``rule`` (``"official"`` or ``"all_points"``).

    With ``category`` set, the rate over that category; otherwise the unweighted mean of the five
    category rates, the ``Avg`` column of the Point-Bench leaderboard and of the Molmo papers.
    """

    name: str  # type: ignore[misc]
    scorer: Scorer  # type: ignore[misc]
    rule: str = "official"
    category: str | None = None

    def _rate(self, responses: Sequence[Response], category: str) -> float:
        vals = [
            float(res[self.rule])
            for r in responses
            if r.instance.metadata.get("category") == category and (res := _result(r)) is not None
        ]
        return sum(vals) / len(vals) if vals else 0.0

    def compute(self, responses: Sequence[Response]) -> float:
        if self.category is not None:
            return self._rate(responses, self.category)
        return sum(self._rate(responses, c) for c in CATEGORIES) / len(CATEGORIES)

    def compute_instance(self, response: Response) -> float | None:
        if self.category is None or response.instance.metadata.get("category") != self.category:
            return None
        res = _result(response)
        return None if res is None else float(res[self.rule])

    def supports_pairwise_scorer_fallback(self) -> bool:
        return False
