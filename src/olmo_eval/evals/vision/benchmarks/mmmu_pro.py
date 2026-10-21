"""MMMU-Pro (test) — a single image-QA task covering all three official settings.

One run produces one ``metrics.json`` with ``standard_10_accuracy``, ``standard_4_accuracy``,
``vision_accuracy`` and the primary **``overall`` = (standard_10 + vision) / 2** (the number SOTA
models report). The benchmark is decoupled from the existing ``mmmu`` task: prompts/parser/scorer
come from :mod:`olmo_eval.evals.vision.scoring.mmmu_pro`,
faithful to the official ``MMMU-Benchmark/MMMU`` ``mmmu-pro`` repo.

Settings (each the 1730-row ``test`` split of an ``MMMU/MMMU_Pro`` config):
* ``standard10`` / ``standard4`` — text question + lettered options (A–J / A–D), interleaved
  ``image_1..image_7`` attached in ``<image i>`` token-appearance order (options are shuffled);
* ``vision`` — a single screenshot image with the question+options rendered inside; the model is
  sent only the direct instruction + that screenshot.

Set ``HF_DATASETS_CACHE`` for offline loading (the dataset is cached like ``MMMU/MMMU``).
"""

from __future__ import annotations

import ast
import itertools
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from olmo_eval.common.metrics.base import Metric
from olmo_eval.common.scorers.base import Scorer
from olmo_eval.common.types import Instance, Response, SamplingParams, Split
from olmo_eval.evals.tasks.common import register
from olmo_eval.evals.vision.data.images import lazy_hf_image
from olmo_eval.evals.vision.scoring.mmmu_pro import (
    MMMU_PRO_VISION_DIRECT,
    MmmuProScorer,
    construct_standard_prompt,
    replace_images_tokens,
)
from olmo_eval.evals.vision.tasks.image_qa import ImageQATask

# Short setting key -> HF config name.
_SETTINGS: dict[str, str] = {
    "standard10": "standard (10 options)",
    "standard4": "standard (4 options)",
    "vision": "vision",
}

_SCORER = MmmuProScorer()


# ---------------------------------------------------------------------------
# Metrics (partition responses by the per-instance ``mmmu_pro_setting`` tag)
# ---------------------------------------------------------------------------


def _setting_mean(responses: Sequence[Response], scorer_name: str, setting: str) -> float | None:
    vals = [
        r.scores.get(scorer_name, 0.0)
        for r in responses
        if r.instance.metadata.get("mmmu_pro_setting") == setting
    ]
    return sum(vals) / len(vals) if vals else None


@dataclass(frozen=True)
class MmmuProSettingMetric(Metric):
    """Accuracy over one MMMU-Pro setting (``standard10`` / ``standard4`` / ``vision``)."""

    name: str  # type: ignore[misc]
    scorer: Scorer  # type: ignore[misc]
    setting: str = ""

    def compute(self, responses: Sequence[Response]) -> float:
        return _setting_mean(responses, self.scorer().name, self.setting) or 0.0

    def compute_instance(self, response: Response) -> float | None:
        """This setting's score for the example, or ``None`` for other settings."""
        if response.instance.metadata.get("mmmu_pro_setting") != self.setting:
            return None
        value = response.scores.get(self.scorer().name)
        return float(value) if isinstance(value, (int, float)) else None

    def supports_scorer_fallback(self) -> bool:
        return False


@dataclass(frozen=True)
class MmmuProOverallMetric(Metric):
    """MMMU-Pro Overall = mean(standard-10 accuracy, vision accuracy) — the SOTA-reported number."""

    name: str  # type: ignore[misc]
    scorer: Scorer  # type: ignore[misc]

    def compute(self, responses: Sequence[Response]) -> float:
        scorer_name = self.scorer().name
        parts = [
            _setting_mean(responses, scorer_name, "standard10"),
            _setting_mean(responses, scorer_name, "vision"),
        ]
        present = [p for p in parts if p is not None]
        return sum(present) / len(present) if present else 0.0

    def compute_instance(self, response: Response) -> float | None:
        # An average of two setting means has no exact per-instance value.
        return None

    def supports_scorer_fallback(self) -> bool:
        return False


_OVERALL = MmmuProOverallMetric(name="overall", scorer=_SCORER)
_METRICS: tuple[Metric, ...] = (
    _OVERALL,
    MmmuProSettingMetric(name="standard_10_accuracy", scorer=_SCORER, setting="standard10"),
    MmmuProSettingMetric(name="standard_4_accuracy", scorer=_SCORER, setting="standard4"),
    MmmuProSettingMetric(name="vision_accuracy", scorer=_SCORER, setting="vision"),
)


@register("mmmu_pro")
class MmmuProTask(ImageQATask):
    sampling_params = SamplingParams(temperature=0.0, max_tokens=64)
    metrics = _METRICS
    primary_metric = _OVERALL
    split = Split.TEST

    def _build_instances(self) -> Iterator[Instance]:
        import datasets

        per_setting: list[list[Instance]] = []
        for key, cfg in _SETTINGS.items():
            ds = datasets.load_dataset("MMMU/MMMU_Pro", cfg, split=self.config.split.value)
            if key == "vision":
                ds_nd = ds.cast_column("image", datasets.Image(decode=False))
                per_setting.append(list(self._vision_instances(ds_nd)))
            else:
                ds_nd = ds
                for col in (f"image_{i}" for i in range(1, 8)):
                    ds_nd = ds_nd.cast_column(col, datasets.Image(decode=False))
                per_setting.append(list(self._standard_instances(ds_nd, key)))

        # Round-robin interleave so a small ``limit`` samples all three settings.
        for group in itertools.zip_longest(*per_setting):
            for inst in group:
                if inst is not None:
                    yield inst

    @staticmethod
    def _standard_instances(ds_nd, setting: str) -> Iterator[Instance]:
        for idx in range(len(ds_nd)):
            ex = ds_nd[idx]
            options = ast.literal_eval(str(ex["options"]))
            text, image_order = replace_images_tokens(
                construct_standard_prompt(str(ex["question"]), options)
            )
            images = [
                lazy_hf_image(ds_nd, idx, f"image_{i}")
                for i in image_order
                if ex[f"image_{i}"] is not None
            ]
            yield Instance(
                question=text,
                gold_answer=str(ex["answer"]),
                metadata={
                    "mmmu_pro_setting": setting,
                    "answer": ex["answer"],
                    "options": options,
                    "example_id": ex["id"],
                    "subject": ex["subject"],
                    "images": images,
                },
            )

    @staticmethod
    def _vision_instances(ds_nd) -> Iterator[Instance]:
        for idx in range(len(ds_nd)):
            ex = ds_nd[idx]
            options = ast.literal_eval(str(ex["options"]))
            yield Instance(
                question=MMMU_PRO_VISION_DIRECT,
                gold_answer=str(ex["answer"]),
                metadata={
                    "mmmu_pro_setting": "vision",
                    "answer": ex["answer"],
                    "options": options,
                    "example_id": ex["id"],
                    "subject": ex["subject"],
                    "image": lazy_hf_image(ds_nd, idx, "image"),
                },
            )

    def _attach_images(self, instance: Instance) -> tuple[Any, ...] | None:
        """Lazy references only; the provider resolves them and drops ``None`` entries.

        The standard setting interleaves up to seven images in token order; the
        vision setting carries the single screenshot under ``image``.
        """
        meta = instance.metadata
        if "images" in meta:
            return tuple(meta["images"]) or None
        image = meta.get("image")
        return (image,) if image is not None else None
