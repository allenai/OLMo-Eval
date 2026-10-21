"""MuirBench — multi-image multiple-choice benchmark (test split, 2,600 examples).

Mirrors mm_olmo's ``MuirBenchConfig()`` (task ``muir_bench:test``): ``<image>``
placeholders in the question/options are rewritten to ``"Image n"``, and the
question is templated with lettered options and the ``"Only return the correct
answer option."`` instruction (``uber_model_v2`` eval branch), no style tag.
Each example carries 2-9 images.  Loads the prepared arrow dataset at
``torch_datasets/academic_datasets/muir_bench``.

Reference (Molmo2-4B ck2000): all=0.6046.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import ClassVar

from olmo_eval.common.types import Instance, SamplingParams, Split
from olmo_eval.evals.tasks.common import register
from olmo_eval.evals.vision.data.images import lazy_hf_image_list
from olmo_eval.evals.vision.data.paths import torch_datasets_dir
from olmo_eval.evals.vision.scoring.multi_image import MultiImageMcScorer
from olmo_eval.evals.vision.scoring.prompt_templates import format_mc_question
from olmo_eval.evals.vision.tasks.multi_image import (
    MultiImageQATask,
    multi_image_mc_metrics,
    replace_images,
)

# mm_olmo ``MuirBenchEval.TASKS``.
MUIR_BENCH_TASKS: tuple[str, ...] = (
    "Action Understanding",
    "Attribute Similarity",
    "Cartoon Understanding",
    "Counting",
    "Diagram Understanding",
    "Difference Spotting",
    "Geographic Understanding",
    "Image-Text Matching",
    "Ordering",
    "Scene Understanding",
    "Visual Grounding",
    "Visual Retrieval",
)

_SCORER = MultiImageMcScorer()
_METRICS = multi_image_mc_metrics(_SCORER, MUIR_BENCH_TASKS, field="task")


@register("muir_bench")
class MuirBenchTask(MultiImageQATask):
    sampling_params = SamplingParams(temperature=0.0, max_tokens=16)
    metrics = _METRICS
    primary_metric = _METRICS[0]  # all
    split = Split.TEST

    TASKS: ClassVar[tuple[str, ...]] = MUIR_BENCH_TASKS

    def _build_instances(self) -> Iterator[Instance]:
        import datasets

        ds = datasets.load_from_disk(
            str(torch_datasets_dir() / "academic_datasets" / "muir_bench")
        )[self.config.split.value]
        ds_nodecode = ds.cast_column("image_list", [datasets.Image(decode=False)])

        for idx in range(len(ds_nodecode)):
            ex = ds_nodecode[idx]
            task = ex["task"]
            if task not in self.TASKS:
                raise ValueError(f"MuirBench example {ex['idx']} has an unknown task {task!r}")
            question, options = replace_images(ex["question"], list(ex["options"]))
            answer_idx = ord(ex["answer"]) - ord("A")
            if not (0 <= answer_idx < len(options)):
                raise ValueError(f"Invalid answer idx in example: {ex['idx']}")
            prompt, _ = format_mc_question(question, options)
            yield Instance(
                question=prompt,
                gold_answer=ex["answer"],
                metadata={
                    "answer": ex["answer"],
                    "answer_idx": answer_idx,
                    "options": options,
                    "example_id": ex["idx"],
                    "task": task,
                    "image_relation": ex["image_relation"],
                    "image_type": ex["image_type"],
                    "counterpart_id": ex["counterpart_idx"],
                    "images": lazy_hf_image_list(ds_nodecode, idx, "image_list"),
                },
            )
