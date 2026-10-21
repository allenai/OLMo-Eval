"""RealWorldQA (xai-org/RealworldQA, test split — the only split).

Mirrors mm_olmo's ``RealWorldQaConfig(mode="no_instruction")`` (task name
``real_world_qa_no_instruction``): each question embeds one of two x.ai
instruction suffixes which determine the question type.  Short-answer
questions are truncated to their first line and prompted with the ``vqa2``
style tag; multiple-choice questions keep the full original prompt (embedded
options + letter instruction) with no style tag.

Set ``HF_DATASETS_CACHE`` to a local cache for offline loading.

Reference (Molmo2-4B ck2000): real_world_qa_score=0.7542.
"""

from __future__ import annotations

from collections.abc import Iterator

from olmo_eval.common.types import Instance, SamplingParams, Split
from olmo_eval.evals.tasks.common import register
from olmo_eval.evals.vision.data.images import lazy_hf_image
from olmo_eval.evals.vision.scoring.image_qa import RealWorldQaScorer
from olmo_eval.evals.vision.tasks.image_qa import ImageQATask, MeanScorerMetric

_METRIC = MeanScorerMetric(name="real_world_qa_score", scorer=RealWorldQaScorer())

_SHORT_ANSWER_INSTRUCTION = "Please answer directly with a single word or number."
_MC_INSTRUCTION = (
    "Please answer directly with only the letter of the correct option and nothing else."
)


@register("real_world_qa")
class RealWorldQaTask(ImageQATask):
    sampling_params = SamplingParams(temperature=0.0, max_tokens=12)
    metrics = (_METRIC,)
    primary_metric = _METRIC
    split = Split.TEST  # the dataset's only split

    def _build_instances(self) -> Iterator[Instance]:
        import datasets

        ds = datasets.load_dataset("xai-org/RealworldQA", split="test")
        ds_nodecode = ds.cast_column("image", datasets.Image(decode=False))
        for idx in range(len(ds_nodecode)):
            ex = ds_nodecode[idx]
            prompt: str = ex["question"]
            if _SHORT_ANSWER_INSTRUCTION in prompt:
                question_type = "short_answer"
                first_line = prompt.split("\n")[0]
                question = f"vqa2: {first_line}"
            else:
                if _MC_INSTRUCTION not in prompt:
                    raise ValueError(
                        f"RealWorldQA multiple-choice row {idx} lacks the expected "
                        f"instruction {_MC_INSTRUCTION!r}: {prompt!r}"
                    )
                question_type = "multiple_choice"
                question = prompt
            yield Instance(
                question=question,
                gold_answer=ex["answer"],
                metadata={
                    "answer": ex["answer"],
                    "question_type": question_type,
                    "example_id": idx,
                    "original_question": prompt,
                    "image": lazy_hf_image(ds_nodecode, idx, "image"),
                },
            )
