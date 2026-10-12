"""Point-Bench (PointArena) — 982 pointing queries in five categories.

The query, the expected ``count`` of the counting items and the existing points of the steerable
images come from the official ``PointArena/pointarena-data`` (``data.json``,
``pixmo_metadata.csv``) at a pinned revision; the images and masks from mm_olmo's copy of the same
release under ``$MOLMO_DATA_DIR/torch_datasets/point_arena``. Scoring is the official one (see
:mod:`olmo_eval.evals.vision.scoring.point_bench`); the primary metric is ``average``, the
unweighted mean of the five category success rates (the leaderboard ``Avg``).

As in the official evaluator, a ``steerable`` query is followed by the existing point it refers
to, in pixels: ``The image contains an existing original point at pixel coordinates: [x, y].
The query refers to this existing point.`` mm_olmo's ``PointBenchConfig`` sends the query alone
and scores with its stricter all-points rule (reported here as ``average_all_points``), so the
steerable numbers are not comparable with mm_olmo-produced ones.

The question carries the ``pointing`` style: bare for instruction-tuned checkpoints, ``pointing:
<query>`` under ``-o system_prompt_style=style_and_length_v2`` (the official Molmo prompt is also
``pointing: <query>``).
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator

from olmo_eval.common.metrics.base import Metric
from olmo_eval.common.types import Instance, SamplingParams, Split
from olmo_eval.evals.tasks.common import register
from olmo_eval.evals.vision.data.paths import torch_datasets_dir
from olmo_eval.evals.vision.scoring.gui_grounding import PointCountRateMetric
from olmo_eval.evals.vision.scoring.point_bench import (
    CATEGORIES,
    PointBenchAccuracyMetric,
    PointBenchScorer,
)
from olmo_eval.evals.vision.tasks.base import VisionTask
from olmo_eval.evals.vision.tasks.pointing import StylePrefixMixin

#: ``PointArena/pointarena-data`` revision the metadata is read from.
REVISION = "77ec5dca697b25025e655e2ec34fd2207856924c"

_SCORER = PointBenchScorer()
_METRICS: tuple[Metric, ...] = (
    PointBenchAccuracyMetric(name="average", scorer=_SCORER),
    *(PointBenchAccuracyMetric(name=c, scorer=_SCORER, category=c) for c in CATEGORIES),
    PointBenchAccuracyMetric(name="average_all_points", scorer=_SCORER, rule="all_points"),
    PointCountRateMetric(
        name="zero_points", scorer=_SCORER, mode="zero", result_key="point_bench_result"
    ),
)


def steerable_hint(points: list[dict], image_w: int, image_h: int) -> str:
    """The official ``get_original_points_info`` text for a steerable image's existing points."""
    if not points:
        return ""
    coords = ", ".join(
        f"[{p['x'] * image_w / 100:.1f}, {p['y'] * image_h / 100:.1f}]" for p in points
    )
    return (
        f"\nThe image contains an existing original point at pixel coordinates: {coords}."
        "\nThe query refers to this existing point."
    )


@register("point_bench")
class PointBenchTask(StylePrefixMixin, VisionTask):
    """Point-Bench: affordable, spatial, reasoning, steerable and counting queries."""

    sampling_params = SamplingParams(temperature=0.0, max_tokens=200)
    metrics = _METRICS
    primary_metric = _METRICS[0]
    split = Split.TEST
    dependencies = ["pillow", "numpy"]

    def _build_instances(self) -> Iterator[Instance]:
        from huggingface_hub import hf_hub_download
        from PIL import Image

        def fetch(name: str) -> str:
            return hf_hub_download(
                "PointArena/pointarena-data", name, repo_type="dataset", revision=REVISION
            )

        with open(fetch("data.json")) as f:
            items = json.load(f)
        with open(fetch("pixmo_metadata.csv"), newline="") as f:
            original_points = {
                row["image_filename"]: json.loads(row["points"]) for row in csv.DictReader(f)
            }
        root = torch_datasets_dir() / "point_arena"
        for idx, item in enumerate(items):
            category = item["category"]
            image_path = str(root / "selected_images" / category / item["image_filename"])
            with Image.open(image_path) as image:
                image_w, image_h = image.size
            question = item["user_input"]
            if category == "steerable":
                question += steerable_hint(
                    original_points.get(item["image_filename"], []), image_w, image_h
                )
            yield Instance(
                question=self.apply_family_prefix(question),
                gold_answer=None,
                metadata={
                    "image_path": image_path,
                    "image_size": (image_w, image_h),
                    "mask_path": str(root / "selected_masks" / category / item["mask_filename"]),
                    "category": category,
                    "count": item.get("count"),
                    "query": item["user_input"],
                    "example_id": idx,
                },
            )
