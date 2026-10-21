"""TextVQA validation.

Mirrors mm_olmo's ``TextVqaConfig``: loads
``torch_datasets/text_vqa/TextVQA_0.5.1_val.json`` (val images live under
``train_images/``), prompts with the ``text_vqa`` style tag, scores the
official VQA accuracy.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

from olmo_eval.common.types import Instance, SamplingParams, Split
from olmo_eval.evals.tasks.common import register
from olmo_eval.evals.vision.data.paths import torch_datasets_dir
from olmo_eval.evals.vision.scoring.image_qa import VqaScoreScorer
from olmo_eval.evals.vision.tasks.image_qa import ImageQATask, MeanScorerMetric

_METRIC = MeanScorerMetric(name="vqa_score", scorer=VqaScoreScorer())

_NUM_ANSWERS_PER_QUESTION = 10


@register("text_vqa")
class TextVqaTask(ImageQATask):
    sampling_params = SamplingParams(temperature=0.0, max_tokens=12)
    metrics = (_METRIC,)
    primary_metric = _METRIC
    split = Split.VALIDATION

    def _build_instances(self) -> Iterator[Instance]:
        src_dir = torch_datasets_dir() / "text_vqa"
        with open(src_dir / "TextVQA_0.5.1_val.json") as f:
            data = json.load(f)["data"]
        for item in data:
            answers = item.get("answers", [""] * _NUM_ANSWERS_PER_QUESTION)
            image_subfolder = "train_images" if item["set_name"] != "test" else "test_images"
            yield Instance(
                question=f"text_vqa: {item['question']}",
                gold_answer=answers[0] if answers else None,
                metadata={
                    "answers": answers,
                    "example_id": item["question_id"],
                    "image_id": item["image_id"],
                    "image_path": str(src_dir / image_subfolder / f"{item['image_id']}.jpg"),
                },
            )
