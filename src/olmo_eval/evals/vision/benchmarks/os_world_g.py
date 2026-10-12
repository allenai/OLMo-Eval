"""OSWorld-G — GUI grounding on 564 instructions over desktop screenshots.

The official benchmark from xlang-ai/OSWorld-G, pinned at commit ``daa6bd8e``: annotations
(``benchmark/OSWorld-G.json``, or ``OSWorld-G_refined.json`` for the refined instructions), the
264 screenshots, and the capability groups (``benchmark/classification_result.json``), all
downloaded once into the download cache (:mod:`olmo_eval.evals.vision.data.downloads`).

Scoring follows the official ``evaluation/eval.py``: the first predicted point must fall inside
the target, which is a box (470 instances), a polygon (40, ray-cast) or a refusal (54, where the
instruction names nothing on screen and only an answer with no point is correct; the official
code reads a refusal as negative coordinates, which a points-format answer cannot produce).
Primary ``accuracy`` is over all 564.

``MMInstruction/OSWorld-G`` on the Hub, which mm_olmo's ``OSWorldGConfig`` reads, is a different
cut: it drops the 54 refusals and replaces each polygon with its bounding box, and mm_olmo's
evaluator then reads those boxes as ``[x, y, w, h]``. ``accuracy_no_refusal`` (the 510 non-refusal
instances, polygons intact) is the closest number to results produced that way.

The question is ``Click <instruction>`` (``gui_point: <instruction>`` under the stage-1 family), as
for ScreenSpot. The instructions are sent verbatim, so one already starting with "Click" reads
``Click Click ...``, as in mm_olmo.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

from olmo_eval.common.metrics.base import Metric
from olmo_eval.common.types import Instance
from olmo_eval.evals.tasks.common import register
from olmo_eval.evals.vision.benchmarks.screen_spot import GuiGroundingTask, _rate_metrics
from olmo_eval.evals.vision.data.downloads import cached_download
from olmo_eval.evals.vision.scoring.gui_grounding import (
    ClickAccuracyMetric,
    ClickScorer,
    rect_from_xywh,
)

#: xlang-ai/OSWorld-G commit the annotations and images are read from.
REVISION = "daa6bd8e0e629f0917ad2984df930bf0bd967540"
_RAW = f"https://raw.githubusercontent.com/xlang-ai/OSWorld-G/{REVISION}/benchmark"

#: The official capability groups (``classification_result.json``); an instance can be in several.
GROUPS = (
    "text_matching",
    "element_recognition",
    "layout_understanding",
    "fine_grained_manipulation",
    "refusal",
)

_SCORER = ClickScorer()
_METRICS: tuple[Metric, ...] = (
    ClickAccuracyMetric(name="accuracy", scorer=_SCORER),
    ClickAccuracyMetric(
        name="accuracy_no_refusal", scorer=_SCORER, field="has_target", value="yes"
    ),
    *(ClickAccuracyMetric(name=g, scorer=_SCORER, field="groups", value=g) for g in GROUPS),
    *_rate_metrics(),
)


def _download(name: str) -> str:
    return str(cached_download(f"{_RAW}/{name}", f"osworld_g/{REVISION}/{name}"))


def _target(item: dict) -> dict:
    box_type = item["box_type"]
    if box_type == "bbox":
        return rect_from_xywh(item["box_coordinates"])
    if box_type == "polygon":
        return {"type": "polygon", "coords": [float(v) for v in item["box_coordinates"]]}
    if box_type == "refusal":
        return {"type": "refusal"}
    raise ValueError(f"unknown OSWorld-G box_type {box_type!r} for {item['id']}")


class _OSWorldGTask(GuiGroundingTask):
    metrics = _METRICS
    primary_metric = _METRICS[0]
    annotation_file: str = "OSWorld-G.json"

    def _build_instances(self) -> Iterator[Instance]:
        with open(_download(self.annotation_file)) as f:
            items = json.load(f)
        with open(_download("classification_result.json")) as f:
            classified = json.load(f)["classified"]
        groups: dict[str, list[str]] = {}
        for group, members in classified.items():
            for member in members:
                groups.setdefault(member["id"], []).append(group)
        # 264 screenshots (~128 MB); fetch them concurrently on first use.
        names = sorted({f"images/{item['image_path']}" for item in items})
        with ThreadPoolExecutor(max_workers=16) as pool:
            image_paths = dict(zip(names, pool.map(_download, names), strict=True))
        for item in items:
            yield Instance(
                question=self.gui_question(item["instruction"]),
                gold_answer=None,
                metadata={
                    "image_path": image_paths[f"images/{item['image_path']}"],
                    "image_size": tuple(int(v) for v in item["image_size"]),
                    "target": _target(item),
                    "box_type": item["box_type"],
                    "has_target": "no" if item["box_type"] == "refusal" else "yes",
                    "groups": groups.get(item["id"], []),
                    "gui_types": list(item.get("GUI_types") or []),
                    "instruction": item["instruction"],
                    "example_id": item["id"],
                },
            )


@register("os_world_g")
class OSWorldGTask(_OSWorldGTask):
    """OSWorld-G with the original instructions."""


@register("os_world_g_refined")
class OSWorldGRefinedTask(_OSWorldGTask):
    """OSWorld-G with the refined instructions (same instances and targets; 507 of 564 reworded)."""

    annotation_file = "OSWorld-G_refined.json"
