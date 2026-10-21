"""Count parsing for CountBench QA / PixMo Count (the ``point_count`` style).

Vendored from ``mm_olmo/olmo/eval/molmo_prediction_evaluators.py``
(``PointCountEval``) and the universal point-extraction regexes in
``mm_olmo/olmo/preprocessing/point_formatter.py`` (``UnifiedPointFormatter``
and ``PointFormattingV1``).  Behavior is preserved exactly.

The parse ladder for a predicted count:
    1. last whitespace token as int
    2. last token as a number word ("one" … "twenty")
    3. ``"a total of N"`` regex
    4. a bare "none" → 0
    5. fall back to counting the points the model emitted
"""

from __future__ import annotations

import contextlib
import re

WORD_TO_NUM = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "zero": 0,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}

# --- UnifiedPointFormatter regexes (coordinate_scale="1000", image_sep="\t") ---
_COORD_REGEX = re.compile(r"<(?:points|tracks|bboxes).*? coords=\"([0-9\t:;, .]+)\"/?>")
_FRAME_REGEX = re.compile(r"(?:^|\t|:|,|;)([0-9\.]+) ([0-9\. ]+)")
_POINTS_REGEX = re.compile(r"([0-9]+) ([0-9]{3,4}) ([0-9]{3,4})")


def _extract_points_unified(text: str, image_w: float, image_h: float) -> list[tuple[float, float]]:
    all_points: list[tuple[float, float]] = []
    for coord in _COORD_REGEX.finditer(text):
        for point_grp in _FRAME_REGEX.finditer(coord.group(1)):
            for triplet in _POINTS_REGEX.finditer(point_grp.group(2)):
                x = float(triplet.group(2)) / 1000 * image_w
                y = float(triplet.group(3)) / 1000 * image_h
                if 0 <= x <= image_w and 0 <= y <= image_h:
                    all_points.append((x, y))
    return all_points


def _extract_points_v1(text: str, image_w: float, image_h: float) -> list[tuple[float, float]]:
    """Legacy point-format fallback chain (``PointFormattingV1``)."""
    all_points: list[tuple[float, float]] = []

    def _scaled(x: float, y: float, bound: float, scale: float) -> tuple[float, float] | None:
        if max(x, y) > bound:
            return None  # treat as an invalid output
        return x / scale * image_w, y / scale * image_h

    for match in re.finditer(r"Click\(([0-9]+\.[0-9]), ?([0-9]+\.[0-9])\)", text):
        point = _scaled(float(match.group(1)), float(match.group(2)), 100, 100.0)
        if point is not None:
            all_points.append(point)
    if all_points:
        return all_points

    for match in re.finditer(r"[0-9]+ ([0-9]{3}) ([0-9]{3})", text):
        point = _scaled(float(match.group(1)), float(match.group(2)), 1000, 1000.0)
        if point is not None:
            all_points.append(point)
    if all_points:
        return all_points

    for match in re.finditer(r"[0-9]+ ([0-9]+\.[0-9]) ([0-9]+\.[0-9])", text):
        point = _scaled(float(match.group(1)), float(match.group(2)), 100, 100.0)
        if point is not None:
            all_points.append(point)
    if all_points:
        return all_points

    for match in re.finditer(r"\(([0-9]+\.[0-9]),? ?([0-9]+\.[0-9])\)", text):
        point = _scaled(float(match.group(1)), float(match.group(2)), 100, 100.0)
        if point is not None:
            all_points.append(point)
    for match in re.finditer(
        r'x\d*="\s*([0-9]+(?:\.[0-9]+)?)"\s+y\d*="\s*([0-9]+(?:\.[0-9]+)?)"', text
    ):
        point = _scaled(float(match.group(1)), float(match.group(2)), 100, 100.0)
        if point is not None:
            all_points.append(point)
    for match in re.finditer(r"(?:\d+|p)\s*=\s*([0-9]{3})\s*,\s*([0-9]{3})", text):
        point = _scaled(int(match.group(1)) / 10.0, int(match.group(2)) / 10.0, 100, 100.0)
        if point is not None:
            all_points.append(point)
    return all_points


def extract_image_points(
    text: str, image_w: float = 100, image_h: float = 100
) -> list[tuple[float, float]]:
    """Universal point extraction: unified format first, then legacy formats."""
    points = _extract_points_unified(text, image_w, image_h)
    if points:
        return points
    return _extract_points_v1(text, image_w, image_h)


def parse_count(original_pred: str) -> int:
    """Parse the predicted count from a ``point_count``-style response."""
    pred = original_pred.lower().rstrip(".").strip()
    pred_int: int | None = None
    parts = pred.split()

    if parts:
        with contextlib.suppress(ValueError):
            pred_int = int(parts[-1].strip(". "))

        if pred_int is None and parts[-1] in WORD_TO_NUM:
            pred_int = WORD_TO_NUM[parts[-1]]

    if pred_int is None:
        match = re.match(".*a total of ([0-9]+).*", pred)
        if match:
            pred_int = int(match.group(1))

    if pred_int is None:
        match = re.match(".*\\bnone\\b.*", pred, re.IGNORECASE)
        if match:
            pred_int = 0

    if pred_int is None:
        pred_int = len(extract_image_points(pred, 100, 100))

    return pred_int
