"""ScreenSpot-v2 and ScreenSpot-Pro — GUI grounding by single click.

Each instance is a screenshot and a natural-language instruction naming one UI element; the
model answers with a point, scored correct when the first point lies inside the element's box
(see :mod:`olmo_eval.evals.vision.scoring.gui_grounding`). Both benchmarks report overall
accuracy over all instances, as their official evaluations do, with per-platform /
per-application-group breakdowns split by element type (``text`` / ``icon``).

The data is read from ``$MOLMO_DATA_DIR/torch_datasets`` (the copies mm_olmo's
``ScreenSpotV2Config`` / ``ScreenSpotProConfig`` use): ``ScreenSpot-v2`` (OS-Copilot/ScreenSpot-v2,
1,272 instructions) and ``ScreenSpotPro`` (likaixin/ScreenSpot-Pro, 1,581).

For instruction-tuned checkpoints the question is ``Click <instruction>``: mm_olmo's ``click``
prompt, which the MolmoPoint paper's runs used (``screen_spot_v2_click`` and friends). Under
``-o system_prompt_style=style_and_length_v2`` it is ``gui_point: <instruction>`` instead, the tag
OLMo-core's stage 1 trains GUI instructions under.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from olmo_eval.common.metrics.base import Metric
from olmo_eval.common.types import Instance, SamplingParams, Split
from olmo_eval.evals.tasks.common import register
from olmo_eval.evals.vision.data.paths import torch_datasets_dir
from olmo_eval.evals.vision.scoring.gui_grounding import (
    ClickAccuracyMetric,
    ClickScorer,
    MacroClickAccuracyMetric,
    PointCountRateMetric,
    rect_from_xywh,
)
from olmo_eval.evals.vision.scoring.prompts import GUI_POINT_STYLE, STYLE_PREFIX_STYLES
from olmo_eval.evals.vision.tasks.base import VisionTask
from olmo_eval.evals.vision.tasks.pointing import StylePrefixMixin

_SCORER = ClickScorer()
_UI_TYPES = ("text", "icon")


def _rate_metrics() -> tuple[Metric, ...]:
    return (
        PointCountRateMetric(name="zero_points", scorer=_SCORER, mode="zero"),
        PointCountRateMetric(name="multiple_points", scorer=_SCORER, mode="multiple"),
    )


def _image_size(path: str) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as image:
        return image.size


class GuiGroundingTask(StylePrefixMixin, VisionTask):
    """A single-click GUI grounding benchmark."""

    style = GUI_POINT_STYLE
    sampling_params = SamplingParams(temperature=0.0, max_tokens=128)
    split = Split.TEST
    #: mm_olmo's ``click`` prompt (``_get_prompt``), prepended to the instruction verbatim.
    click_prefix = "Click "

    def gui_question(self, instruction: str) -> str:
        """``gui_point: <instruction>`` for the stage-1 family, else ``Click <instruction>``."""
        family = self.config.system_prompt_style or self.default_system_prompt_style
        if family in STYLE_PREFIX_STYLES:
            return self.apply_family_prefix(instruction)
        return self.click_prefix + instruction


# ---------------------------------------------------------------------------
# ScreenSpot-v2
# ---------------------------------------------------------------------------

_V2_KINDS = ("desktop", "web", "mobile")  # mm_olmo's build order
_V2_METRICS: tuple[Metric, ...] = (
    ClickAccuracyMetric(name="accuracy", scorer=_SCORER),
    MacroClickAccuracyMetric(name="average", scorer=_SCORER, field="kind", values=_V2_KINDS),
    *(ClickAccuracyMetric(name=k, scorer=_SCORER, field="kind", value=k) for k in _V2_KINDS),
    *(
        ClickAccuracyMetric(name=f"{k}_{t}", scorer=_SCORER, field="kind_type", value=f"{k}_{t}")
        for k in _V2_KINDS
        for t in _UI_TYPES
    ),
    *_rate_metrics(),
)


@register("screen_spot_v2")
class ScreenSpotV2Task(GuiGroundingTask):
    """ScreenSpot-v2: 1,272 instructions over desktop, web and mobile screenshots.

    Primary ``accuracy`` is over all instances (the official number). ``average`` is the mean of
    the desktop, web and mobile accuracies, mm_olmo's ``ScreenSpotEvaluator`` average.
    """

    metrics = _V2_METRICS
    primary_metric = _V2_METRICS[0]

    def _build_instances(self) -> Iterator[Instance]:
        root = torch_datasets_dir() / "ScreenSpot-v2"
        for kind in _V2_KINDS:
            with open(root / f"screenspot_{kind}_v2.json") as f:
                rows = json.load(f)
            for idx, ex in enumerate(rows):
                image_path = str(root / "screenspotv2_image" / ex["img_filename"])
                yield Instance(
                    question=self.gui_question(ex["instruction"]),
                    gold_answer=None,
                    metadata={
                        "image_path": image_path,
                        "image_size": _image_size(image_path),
                        "target": rect_from_xywh(ex["bbox"]),
                        "kind": kind,
                        "data_type": ex["data_type"],
                        "kind_type": f"{kind}_{ex['data_type']}",
                        "data_source": ex["data_source"],
                        "instruction": ex["instruction"],
                        "example_id": f"{kind}-{idx}",
                    },
                )


# ---------------------------------------------------------------------------
# ScreenSpot-Pro
# ---------------------------------------------------------------------------

_PRO_GROUPS = ("Dev", "Creative", "CAD", "Scientific", "Office", "OS")
_PRO_METRICS: tuple[Metric, ...] = (
    ClickAccuracyMetric(name="accuracy", scorer=_SCORER),
    *(ClickAccuracyMetric(name=t, scorer=_SCORER, field="ui_type", value=t) for t in _UI_TYPES),
    *(ClickAccuracyMetric(name=g, scorer=_SCORER, field="group", value=g) for g in _PRO_GROUPS),
    *(
        ClickAccuracyMetric(name=f"{g}_{t}", scorer=_SCORER, field="group_type", value=f"{g}_{t}")
        for g in _PRO_GROUPS
        for t in _UI_TYPES
    ),
    *_rate_metrics(),
)


@register("screen_spot_pro")
class ScreenSpotProTask(GuiGroundingTask):
    """ScreenSpot-Pro: 1,581 instructions on high-resolution professional-software screenshots.

    Primary ``accuracy`` is over all instances (the official ``Avg``), with breakdowns by
    application group and element type.
    """

    metrics = _PRO_METRICS
    primary_metric = _PRO_METRICS[0]

    def _build_instances(self) -> Iterator[Instance]:
        root = torch_datasets_dir() / "ScreenSpotPro"
        for ann_file in sorted(Path(root / "annotations").glob("*.json")):
            with open(ann_file) as f:
                rows = json.load(f)
            for ex in rows:
                width, height = ex["img_size"]
                yield Instance(
                    question=self.gui_question(ex["instruction"]),
                    gold_answer=None,
                    metadata={
                        "image_path": str(root / "images" / ex["img_filename"]),
                        "image_size": (int(width), int(height)),
                        "target": {"type": "rect", "xyxy": [float(v) for v in ex["bbox"]]},
                        "group": ex["group"],
                        "application": ex["application"],
                        "platform": ex["platform"],
                        "ui_type": ex["ui_type"],
                        "group_type": f"{ex['group']}_{ex['ui_type']}",
                        "instruction": ex["instruction"],
                        "example_id": ex["id"],
                    },
                )
