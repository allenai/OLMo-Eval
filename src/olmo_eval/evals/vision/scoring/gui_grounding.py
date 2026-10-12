"""Click-accuracy scoring for the GUI grounding benchmarks.

ScreenSpot-v2, ScreenSpot-Pro and OSWorld-G ask for one click target per instruction and count
a prediction correct when the model's first point lands inside the target region, as in the
benchmarks' own evaluation code (OSWorld-G's ``evaluation/eval.py``). A target is one of:

* a rectangle, ``{"type": "rect", "xyxy": [x1, y1, x2, y2]}`` in image pixels (edges inclusive);
* a polygon, ``{"type": "polygon", "coords": [x1, y1, x2, y2, ...]}`` (OSWorld-G), tested by ray
  casting exactly as OSWorld-G does;
* a refusal, ``{"type": "refusal"}`` (OSWorld-G): the instruction names no element on screen,
  so the answer is correct only when the model gives no point.

The scorer stores the number of predicted points in ``output.metadata["click_result"]`` so the
zero- and multiple-point rates can be reported next to the accuracies.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from olmo_eval.common.metrics.base import Metric
from olmo_eval.common.scorers.base import Scorer
from olmo_eval.common.types import Instance, LMOutput, Response
from olmo_eval.evals.vision.scoring.common import response_text
from olmo_eval.evals.vision.scoring.count_parsing import extract_image_points

__all__ = [
    "ClickAccuracyMetric",
    "ClickScorer",
    "MacroClickAccuracyMetric",
    "PointCountRateMetric",
    "point_in_polygon",
    "point_in_rect",
    "rect_from_xywh",
]


def rect_from_xywh(box: Sequence[float]) -> dict[str, Any]:
    """A rectangle target from ``[x, y, width, height]``."""
    x, y, w, h = (float(v) for v in box[:4])
    return {"type": "rect", "xyxy": [x, y, x + w, y + h]}


def point_in_rect(x: float, y: float, xyxy: Sequence[float]) -> bool:
    """Whether ``(x, y)`` lies inside the rectangle, edges included."""
    return xyxy[0] <= x <= xyxy[2] and xyxy[1] <= y <= xyxy[3]


def point_in_polygon(x: float, y: float, coords: Sequence[float]) -> bool:
    """Ray-casting point-in-polygon test over flat ``[x1, y1, x2, y2, ...]`` vertices.

    A line-for-line port of ``_is_point_in_polygon`` in OSWorld-G's ``evaluation/eval.py``,
    including its handling of points on an edge.
    """
    n = len(coords) // 2
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = coords[i * 2], coords[i * 2 + 1]
        xj, yj = coords[j * 2], coords[j * 2 + 1]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def _click_correct(target: dict[str, Any], points: list[tuple[float, float]]) -> bool:
    kind = target["type"]
    if kind == "refusal":
        return not points
    if not points:
        return False
    x, y = points[0]
    if kind == "rect":
        return point_in_rect(x, y, target["xyxy"])
    if kind == "polygon":
        return point_in_polygon(x, y, target["coords"])
    raise ValueError(f"unknown click target type {kind!r}")


@dataclass(frozen=True, slots=True)
class ClickScorer(Scorer):
    """1.0 when the first predicted point hits ``instance.metadata["target"]``, else 0.0.

    Points are parsed from the response in image pixels using ``instance.metadata["image_size"]``
    (``(width, height)``), the size the target coordinates are given in.
    """

    name: str = "click"

    def score(self, instance: Instance, output: LMOutput) -> float:
        meta = instance.metadata
        image_w, image_h = (float(v) for v in meta["image_size"])
        points = extract_image_points(response_text(output).strip(), image_w, image_h)
        if output.metadata is None:
            output.metadata = {}
        output.metadata["click_result"] = {"n_points": len(points)}
        return float(_click_correct(meta["target"], points))


def _member(response: Response, field: str | None, value: str | None) -> bool:
    if field is None:
        return True
    have = response.instance.metadata.get(field)
    if isinstance(have, (list, tuple)):
        return value in have
    return have == value


@dataclass(frozen=True)
class ClickAccuracyMetric(Metric):
    """Fraction of correct clicks, over all instances or those whose ``field`` matches ``value``.

    ``field`` may name a list-valued metadata entry (an OSWorld-G instance can belong to several
    capability groups); membership then means ``value`` is in the list.
    """

    name: str  # type: ignore[misc]
    scorer: Scorer  # type: ignore[misc]
    field: str | None = None
    value: str | None = None

    def compute(self, responses: Sequence[Response]) -> float:
        scorer_name = self.scorer().name
        vals = [
            r.scores.get(scorer_name, 0.0) for r in responses if _member(r, self.field, self.value)
        ]
        return sum(vals) / len(vals) if vals else 0.0

    def compute_instance(self, response: Response) -> float | None:
        if not _member(response, self.field, self.value):
            return None
        return response.scores.get(self.scorer().name)


@dataclass(frozen=True)
class MacroClickAccuracyMetric(Metric):
    """Unweighted mean of the per-``value`` accuracies of ``field`` (mm_olmo's ScreenSpot
    ``average``: the mean of the desktop, mobile and web accuracies)."""

    name: str  # type: ignore[misc]
    scorer: Scorer  # type: ignore[misc]
    field: str = ""
    values: tuple[str, ...] = ()

    def compute(self, responses: Sequence[Response]) -> float:
        parts = [
            ClickAccuracyMetric(name=v, scorer=self.scorer, field=self.field, value=v).compute(
                responses
            )
            for v in self.values
        ]
        return sum(parts) / len(parts) if parts else 0.0

    def compute_instance(self, response: Response) -> float | None:
        return None

    def supports_pairwise_scorer_fallback(self) -> bool:
        return False


def _n_points(response: Response, result_key: str) -> int | None:
    for output in response.outputs:
        if output.metadata and result_key in output.metadata:
            return int(output.metadata[result_key]["n_points"])
    return None


@dataclass(frozen=True)
class PointCountRateMetric(Metric):
    """Fraction of responses with no point (``mode="zero"``) or more than one (``"multiple"``)."""

    name: str  # type: ignore[misc]
    scorer: Scorer  # type: ignore[misc]
    mode: str = "zero"
    result_key: str = "click_result"

    def _flag(self, n: int) -> bool:
        return n == 0 if self.mode == "zero" else n > 1

    def compute(self, responses: Sequence[Response]) -> float:
        counts = [n for r in responses if (n := _n_points(r, self.result_key)) is not None]
        return sum(self._flag(n) for n in counts) / len(counts) if counts else 0.0

    def compute_instance(self, response: Response) -> float | None:
        n = _n_points(response, self.result_key)
        return None if n is None else float(self._flag(n))

    def supports_pairwise_scorer_fallback(self) -> bool:
        return False
