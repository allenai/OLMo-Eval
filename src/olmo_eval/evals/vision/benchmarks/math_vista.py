"""MathVista testmini (1,000 examples).

Mirrors mm_olmo's ``MathVistaConfig(simplify_question=True)`` (task name
``math_vista_v2``): the requested validation split maps to HF ``testmini``;
the ``Question:``/``Hint:`` boilerplate is stripped from the query.
Multiple-choice questions are templated with lettered options (no style tag);
free-form questions are prompted with the ``vqa2`` style tag.

Scoring follows the official MathVista protocol: deterministic short-circuits
(empty response, verbatim choice, int/float parse) and otherwise GPT-4
(``gpt-4-0613``) answer extraction — exactly mm_olmo's ``extract_answer``
(``quick_extract=False``).  This is the **default** task scorer (requires
``OPENAI_API_KEY``; responses cached per-run, see ``MathVistaGptScorer``).
The ``math_vista:offline`` variant replaces the GPT step with deterministic
letter/value extraction for API-free runs (an approximation, not the protocol).

Reference (Molmo2-4B ck2000, GPT extraction): score=0.5670.
"""

from __future__ import annotations

from collections.abc import Iterator

from olmo_eval.common.types import Instance, SamplingParams, Split
from olmo_eval.evals.tasks.common import register, register_variant
from olmo_eval.evals.vision.data.images import lazy_hf_image
from olmo_eval.evals.vision.scoring.image_qa import MathVistaGptScorer, MathVistaOfflineScorer
from olmo_eval.evals.vision.scoring.prompt_templates import format_mc_question
from olmo_eval.evals.vision.tasks.image_qa import ImageQATask, MeanScorerMetric

_OFFLINE_METRIC = MeanScorerMetric(name="score", scorer=MathVistaOfflineScorer())
_GPT_METRIC = MeanScorerMetric(name="score", scorer=MathVistaGptScorer())


@register("math_vista")
class MathVistaTask(ImageQATask):
    sampling_params = SamplingParams(temperature=0.0, max_tokens=32)
    metrics = (_GPT_METRIC,)
    primary_metric = _GPT_METRIC
    #: The default answer extraction calls OpenAI; without the client every
    #: instance records a failed extraction. `offline` needs no key.
    dependencies = ["pillow", "openai"]
    required_secrets = ("OPENAI_API_KEY",)
    split = Split.VALIDATION  # maps to the HF "testmini" split

    def _build_instances(self) -> Iterator[Instance]:
        import datasets

        split = "testmini" if self.config.split == Split.VALIDATION else self.config.split.value
        ds = datasets.load_dataset("AI4Math/MathVista", split=split)
        ds_nodecode = ds.cast_column("decoded_image", datasets.Image(decode=False))

        for idx in range(len(ds_nodecode)):
            ex = ds_nodecode[idx]
            # simplify_question=True: strip the "Question:"/"Hint:" wrappers
            question = ex["question"].split("Question:")[-1]
            question = question.split("Hint:")[0].strip()

            metadata = {
                "example_id": ex["pid"],
                "answer": ex["answer"],
                "precision": ex["precision"],
                "query": ex["question"],
                "choices": ex["choices"],
                "question_type": ex["question_type"],
                "answer_type": ex["answer_type"],
                "image": lazy_hf_image(ds_nodecode, idx, "decoded_image"),
            }
            if ex["question_type"] == "multi_choice":
                question, option_names = format_mc_question(question, ex["choices"])
                metadata["option_names"] = option_names
            else:
                question = f"vqa2: {question}"
            yield Instance(
                question=question,
                gold_answer=ex["answer"],
                metadata=metadata,
            )


# `:offline` = deterministic extraction (no API key, approximation of the
# protocol); `:gpt` is an explicit alias for the GPT-4 default.
register_variant(
    "math_vista", "offline", metrics=(_OFFLINE_METRIC,), primary_metric=_OFFLINE_METRIC
)
register_variant("math_vista", "gpt", metrics=(_GPT_METRIC,), primary_metric=_GPT_METRIC)
