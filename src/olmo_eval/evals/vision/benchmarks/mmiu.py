"""MMIU — Multimodal Multi-image Understanding benchmark (test split, 11,698 examples).

Mirrors mm_olmo's ``MMIUConfig(format="multiple_choice")`` (task ``mmiu:test``):
the raw question is templated with lettered options parsed from the ``options``
string and the ``"Only return the correct answer option."`` instruction
(``uber_model_v2`` eval branch), no style tag; the ``context`` field is metadata
only and never shown to the model.  Each example carries 1-62 images (the model
sees at most ``max_images=20``, matching mm_olmo's eval config); gold answers
occasionally name a letter outside the listed options, so scoring matches the
raw answer letter rather than an option index.

Questions come from ``FanqingM/MMIU-Benchmark`` (set ``HF_DATASETS_CACHE`` for
offline loading); images are loose files under ``torch_datasets/mmiu/``.

Reference (Molmo2-4B ck2000): all=0.5549.
"""

from __future__ import annotations

import re
import string
from collections.abc import Iterator

from olmo_eval.common.metrics.base import Metric
from olmo_eval.common.types import Instance, SamplingParams, Split
from olmo_eval.evals.tasks.common import register
from olmo_eval.evals.vision.data.paths import torch_datasets_dir
from olmo_eval.evals.vision.scoring.multi_image import MultiImageMcScorer
from olmo_eval.evals.vision.scoring.prompt_templates import format_mc_question
from olmo_eval.evals.vision.tasks.multi_image import (
    MultiImageCountBucketMetric,
    MultiImageQATask,
    multi_image_mc_metrics,
)

# mm_olmo ``MMIUEval.RELATIONSHIPS`` (also the image subdirectory names).
MMIU_RELATIONSHIPS: tuple[str, ...] = (
    "2D-spatial",
    "3D-spatial",
    "Continuous-temporal",
    "Discrete-temporal",
    "High-level-obj-semantic",
    "High-level-sub-semantic",
    "Low-level-semantic",
)

#: Hub revision of FanqingM/MMIU-Benchmark the mm_olmo parity checks ran on.
_REVISION = "03bf7d143d920e97a757f606b6b7baee161b019b"

_SCORER = MultiImageMcScorer()
_METRICS: tuple[Metric, ...] = (
    *multi_image_mc_metrics(_SCORER, MMIU_RELATIONSHIPS, field="relationship"),
    # mm_olmo ``MMIUEval.NIMAGES`` buckets.
    MultiImageCountBucketMetric(name="num_images<=10", scorer=_SCORER, min_images=0, max_images=10),
    MultiImageCountBucketMetric(
        name="num_images<=20", scorer=_SCORER, min_images=10, max_images=20
    ),
    MultiImageCountBucketMetric(
        name="num_images>20", scorer=_SCORER, min_images=20, max_images=None
    ),
)


def _extract_options(option_string: str) -> list[str]:
    """mm_olmo ``MMIUConfig._extract_options``: parse ``"A: ...\\nB: ..."`` contents."""
    matches = []
    for ix, (_, letter, answer) in enumerate(
        re.findall(r"(^|\n)([A-Z]):\s?([^\n]+)", option_string, flags=re.DOTALL | re.MULTILINE)
    ):
        if letter != string.ascii_uppercase[ix]:
            raise ValueError(
                f"MMIU options are not in letter order at position {ix}: expected "
                f"{string.ascii_uppercase[ix]!r}, got {letter!r} in {option_string!r}"
            )
        matches.append(answer)
    return matches


def _split_image_path(path: str, idx: int) -> tuple[str, str]:
    """``(path under the images root, relationship)`` for one MMIU image path.

    MMIU lists images as ``./<relationship>/<task>/...``; anything else is a data error.
    """
    parts = path.split("/")
    if parts[0] != "." or len(parts) < 3:
        raise ValueError(f"MMIU row {idx} has an image path outside ./<relationship>/: {path!r}")
    return "/".join(parts[1:]), parts[1]


@register("mmiu")
class MmiuTask(MultiImageQATask):
    sampling_params = SamplingParams(temperature=0.0, max_tokens=32)
    metrics = _METRICS
    primary_metric = _METRICS[0]  # all
    split = Split.TEST

    def _build_instances(self) -> Iterator[Instance]:
        import datasets

        ds = datasets.load_dataset(
            "FanqingM/MMIU-Benchmark", split=self.config.split.value, revision=_REVISION
        )
        images_root = torch_datasets_dir() / "mmiu"

        for idx in range(len(ds)):
            ex = ds[idx]
            paths = [_split_image_path(path, idx) for path in ex["input_image_path"]]
            images = tuple(str(images_root / relative) for relative, _ in paths)
            relationships = {relationship for _, relationship in paths}
            if len(relationships) != 1:
                raise ValueError(
                    f"MMIU row {idx} spans multiple relationships {sorted(relationships)}: "
                    f"{ex['input_image_path']}"
                )
            relationship = relationships.pop()
            if relationship not in MMIU_RELATIONSHIPS:
                raise ValueError(f"MMIU row {idx} has an unknown relationship {relationship!r}")
            options = _extract_options(ex["options"])
            prompt, _ = format_mc_question(ex["question"], options)
            yield Instance(
                question=prompt,
                gold_answer=ex["output"],
                metadata={
                    "answer": ex["output"],
                    "options": options,
                    "example_id": str(idx),
                    "task": ex["task"],
                    "relationship": relationship,
                    "context": ex["context"],
                    "visual_input_component": ex["visual_input_component"],
                    "source": ex["source"],
                    "num_images": len(images),
                    "images": images,
                },
            )
