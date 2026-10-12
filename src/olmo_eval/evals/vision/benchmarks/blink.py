"""BLINK — 14-subtask visual-perception benchmark (validation split, 1,901 examples).

Mirrors mm_olmo's ``BLINKConfig()`` (task ``blink:validation``): the official
``prompt`` field (question + embedded lettered choices + answer instruction) is
passed through verbatim, with no style tag.  Each example carries 1-4 images;
the subtasks are concatenated in ``NAMES`` order.  Loaded per-subtask from
``BLINK-Benchmark/BLINK`` (set ``HF_DATASETS_CACHE`` for offline loading).

Reference (Molmo2-4B ck2000): all=0.5713.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import ClassVar

from olmo_eval.common.types import Instance, SamplingParams, Split
from olmo_eval.evals.tasks.common import register
from olmo_eval.evals.vision.data.images import lazy_hf_image
from olmo_eval.evals.vision.scoring.multi_image import MultiImageMcScorer
from olmo_eval.evals.vision.tasks.multi_image import (
    MultiImageQATask,
    multi_image_mc_metrics,
)

# mm_olmo ``BLINKEval.TASKS`` (display names; dataset config names use underscores).
BLINK_SUBTASKS: tuple[str, ...] = (
    "Art Style",
    "Counting",
    "Forensic Detection",
    "Functional Correspondence",
    "IQ Test",
    "Jigsaw",
    "Multi-view Reasoning",
    "Object Localization",
    "Relative Depth",
    "Relative Reflectance",
    "Semantic Correspondence",
    "Spatial Relation",
    "Visual Correspondence",
    "Visual Similarity",
)

#: Hub revision of BLINK-Benchmark/BLINK the mm_olmo parity checks ran on.
_REVISION = "a3666eb249237ba3d5eca8db21176cc47967e040"

_SCORER = MultiImageMcScorer()
_METRICS = multi_image_mc_metrics(_SCORER, BLINK_SUBTASKS, field="sub_task")


@register("blink")
class BlinkTask(MultiImageQATask):
    sampling_params = SamplingParams(temperature=0.0, max_tokens=12)
    metrics = _METRICS
    primary_metric = _METRICS[0]  # all
    split = Split.VALIDATION

    # mm_olmo ``BLINKConfig.NAMES`` — dataset config load order.
    NAMES: ClassVar[tuple[str, ...]] = (
        "Art_Style",
        "Functional_Correspondence",
        "Multi-view_Reasoning",
        "Relative_Reflectance",
        "Visual_Correspondence",
        "Counting",
        "IQ_Test",
        "Object_Localization",
        "Semantic_Correspondence",
        "Visual_Similarity",
        "Forensic_Detection",
        "Jigsaw",
        "Relative_Depth",
        "Spatial_Relation",
    )

    _IMAGE_COLUMNS: ClassVar[tuple[str, ...]] = ("image_1", "image_2", "image_3", "image_4")

    def _build_instances(self) -> Iterator[Instance]:
        import datasets

        split = "val" if self.config.split == Split.VALIDATION else self.config.split.value
        parts = [
            datasets.load_dataset(
                "BLINK-Benchmark/BLINK", name=name, split=split, revision=_REVISION
            )
            for name in self.NAMES
        ]
        ds = datasets.concatenate_datasets(parts)
        ds_nodecode = ds
        for column in self._IMAGE_COLUMNS:
            ds_nodecode = ds_nodecode.cast_column(column, datasets.Image(decode=False))

        for idx in range(len(ds_nodecode)):
            ex = ds_nodecode[idx]
            sub_task = ex["sub_task"]
            if sub_task not in BLINK_SUBTASKS:
                raise ValueError(f"BLINK row {idx} has an unknown sub_task {sub_task!r}")
            answer = ex["answer"].replace("(", "").replace(")", "")
            images = tuple(
                lazy_hf_image(ds_nodecode, idx, column)
                for column in self._IMAGE_COLUMNS
                if ex[column] is not None
            )
            yield Instance(
                question=ex["prompt"],
                gold_answer=answer,
                metadata={
                    "answer": answer,
                    "answer_idx": ord(answer) - ord("A"),
                    "options": list(ex["choices"]),
                    "example_id": ex["idx"],
                    "sub_task": sub_task,
                    "images": images,
                },
            )
