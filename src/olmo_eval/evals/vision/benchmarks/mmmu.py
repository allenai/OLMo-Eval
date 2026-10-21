"""MMMU validation (all 30 subjects, 900 examples).

Mirrors mm_olmo's ``MMMUConfig()`` (task name ``mmmu_test`` — the validation
split is evaluated, as is standard practice).  Multiple-choice questions are
templated with lettered options and no style tag; open questions are prompted
with the ``vqa2`` style tag.  Following LLaVA (and mm_olmo), the image input
is dropped when more than one answer option embeds an image tag.

Set ``HF_DATASETS_CACHE`` to a local cache for offline loading.

Reference (Molmo2-4B ck2000): mmmu_score=0.5089.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator
from typing import ClassVar

from olmo_eval.common.types import Instance, SamplingParams, Split
from olmo_eval.evals.tasks.common import register
from olmo_eval.evals.vision.data.images import lazy_hf_image
from olmo_eval.evals.vision.scoring.image_qa import MmmuScorer
from olmo_eval.evals.vision.scoring.prompt_templates import format_mc_question
from olmo_eval.evals.vision.tasks.image_qa import ImageQATask, MeanScorerMetric

_METRIC = MeanScorerMetric(name="mmmu_score", scorer=MmmuScorer())

_IMG_OPTION_RE = re.compile(r"<img='(.*?)'>")


@register("mmmu")
class MmmuTask(ImageQATask):
    sampling_params = SamplingParams(temperature=0.0, max_tokens=12)
    metrics = (_METRIC,)
    primary_metric = _METRIC
    split = Split.VALIDATION

    SUBJECTS: ClassVar[list[str]] = [
        "Accounting",
        "Agriculture",
        "Architecture_and_Engineering",
        "Art",
        "Art_Theory",
        "Basic_Medical_Science",
        "Biology",
        "Chemistry",
        "Clinical_Medicine",
        "Computer_Science",
        "Design",
        "Diagnostics_and_Laboratory_Medicine",
        "Economics",
        "Electronics",
        "Energy_and_Power",
        "Finance",
        "Geography",
        "History",
        "Literature",
        "Manage",
        "Marketing",
        "Materials",
        "Math",
        "Mechanical_Engineering",
        "Music",
        "Pharmacy",
        "Physics",
        "Psychology",
        "Public_Health",
        "Sociology",
    ]

    def _build_instances(self) -> Iterator[Instance]:
        import datasets

        parts = [
            datasets.load_dataset("MMMU/MMMU", name=name, split=self.config.split.value)
            for name in self.SUBJECTS
        ]
        ds = datasets.concatenate_datasets(parts)
        image_cols = [f"image_{i}" for i in range(1, 8)]
        ds_nodecode = ds
        for col in image_cols:
            ds_nodecode = ds_nodecode.cast_column(col, datasets.Image(decode=False))

        for idx in range(len(ds_nodecode)):
            ex = ds_nodecode[idx]
            is_mc = ex["question_type"] == "multiple-choice"
            metadata = {
                "answer": ex["answer"],
                "example_id": ex["id"],
                "question_type": ex["question_type"],
            }
            if is_mc:
                options = ast.literal_eval(ex["options"])
                metadata["options"] = options
                # Following LLaVA, drop the image when multiple options embed
                # image paths (the images are the answer options themselves).
                n_img_options = sum(_IMG_OPTION_RE.match(opt) is not None for opt in options)
                if n_img_options <= 1 and ex["image_1"] is not None:
                    metadata["image"] = lazy_hf_image(ds_nodecode, idx, "image_1")
                question, option_names = format_mc_question(ex["question"], options)
                metadata["option_names"] = option_names
            else:
                if ex["image_1"] is not None:
                    metadata["image"] = lazy_hf_image(ds_nodecode, idx, "image_1")
                question = f"vqa2: {ex['question']}"
            yield Instance(
                question=question,
                gold_answer=ex["answer"],
                metadata=metadata,
            )
