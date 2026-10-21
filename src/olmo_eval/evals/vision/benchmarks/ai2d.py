"""AI2D (validation by default; ``ai2d:test`` for the official test split).

Mirrors mm_olmo's ``AI2DConfig(boxes="both")`` (task name
``ai2_diagram_v2_mix_transparent``): loads the prepared arrow dataset at
``torch_datasets/academic_datasets/ai2d``, where every abc-label question
appears twice — once with opaque answer boxes drawn on the diagram and once
with transparent ones (``has_transparent_box``).

Formatting follows ``AI2DConfig.format_example``: when a question's answer
options are (almost all) the on-diagram letters themselves, options are
listed without ``A./B.`` prefixes and the model must answer with the option
text (``ai2_diagram_no_letter``); otherwise standard lettered options are
used.  Multiple-choice style — no style tag.

Reference (Molmo2-4B ck2000, val): mc_ai2d_opaque=0.8537,
mc_ai2d_transparent=0.9481.
"""

from __future__ import annotations

from collections.abc import Iterator

from olmo_eval.common.types import Instance, SamplingParams, Split
from olmo_eval.evals.tasks.common import register, register_variant
from olmo_eval.evals.vision.data.images import lazy_hf_image
from olmo_eval.evals.vision.data.paths import torch_datasets_dir
from olmo_eval.evals.vision.scoring.image_qa import Ai2dScorer
from olmo_eval.evals.vision.scoring.prompt_templates import format_mc_question
from olmo_eval.evals.vision.tasks.image_qa import Ai2dMetric, ImageQATask

_SCORER = Ai2dScorer()
_OPAQUE = Ai2dMetric(name="mc_ai2d_opaque", scorer=_SCORER, transparent=False)
_TRANSPARENT = Ai2dMetric(name="mc_ai2d_transparent", scorer=_SCORER, transparent=True)


@register("ai2d")
class Ai2dTask(ImageQATask):
    sampling_params = SamplingParams(temperature=0.0, max_tokens=32)
    metrics = (_OPAQUE, _TRANSPARENT)
    primary_metric = _OPAQUE
    split = Split.VALIDATION

    def _build_instances(self) -> Iterator[Instance]:
        import datasets

        ds = datasets.load_from_disk(str(torch_datasets_dir() / "academic_datasets" / "ai2d"))
        ds = ds[self.config.split.value]
        ds_nodecode = ds.cast_column("image", datasets.Image(decode=False))

        for idx in range(len(ds_nodecode)):
            ex = ds_nodecode[idx]
            options = ex["answer_texts"]
            answer_idx = ex["correct_answer"]
            if ex["abc_label"] and sum(ex["option_is_abc"]) >= (len(options) - 1):
                # ai2_diagram_no_letter: unlabelled options, abc options uppercased
                unlabelled = [
                    opt.upper() if abc else opt
                    for opt, abc in zip(options, ex["option_is_abc"], strict=True)
                ]
                question, option_names = format_mc_question(
                    ex["question"], unlabelled, labelled=False
                )
                gold = unlabelled[answer_idx]
            else:
                question, option_names = format_mc_question(ex["question"], options)
                gold = option_names[answer_idx]
            yield Instance(
                question=question,
                gold_answer=gold,
                metadata={
                    "example_id": ex["question_id"],
                    "image_id": ex["image_id"],
                    "abc_label": ex["abc_label"],
                    "has_transparent_box": ex["has_transparent_box"],
                    "answer_idx": answer_idx,
                    "option_names": option_names,
                    "options": options,
                    "image": lazy_hf_image(ds_nodecode, idx, "image"),
                },
            )


register_variant("ai2d", "test", split=Split.TEST)
# `:transparent` makes mc_ai2d_transparent the primary metric (shown in the summary table);
# both metrics are still computed. Stack with `:test`, e.g. `ai2d:test:transparent`.
register_variant("ai2d", "transparent", primary_metric=_TRANSPARENT)
