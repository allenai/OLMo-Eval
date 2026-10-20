"""PixMo Count (validation by default; ``pixmo_count:test`` for the test split).

Mirrors mm_olmo's ``PixMoCountConfig(counting=True)`` (task name
``pixmo_count_counting``): loads the prepared arrow dataset at
``torch_datasets/pixmo_datasets/count`` and asks an RNG-templated counting
question per example (``point_count`` style — no style tag).

The question template is selected per example by the seeded RNG of mm_olmo's
eval data pipeline, which depends on the example's **arrow-order index** —
instances are therefore built strictly in arrow order (verified to reproduce
all 540 released validation prompts exactly).

The question follows the checkpoint's prompt family, as for the ``_mp`` pointing
tasks. The default (``uber_model_v2``, instruction-tuned checkpoints) is the
templated question above. ``-o prompt_templates=none`` (stage-1 checkpoints, with
``-o system_prompt_style=style_and_length_v2`` for the tag) sends the bare
lower-cased label, ``point_count: cows``: exactly what those checkpoints train on,
and what mm_olmo sends them.

Reference (Molmo2-4B ck2000, val): correct=0.9093.
"""

from __future__ import annotations

from collections.abc import Iterator

from olmo_eval.common.types import Instance, SamplingParams, Split
from olmo_eval.evals.tasks.common import register, register_variant
from olmo_eval.evals.vision.data.paths import rebase_data_path, torch_datasets_dir
from olmo_eval.evals.vision.scoring.counting import PointCountScorer
from olmo_eval.evals.vision.scoring.prompt_templates import pixmo_count_question
from olmo_eval.evals.vision.scoring.prompts import POINT_COUNT_STYLE
from olmo_eval.evals.vision.tasks.image_qa import ImageQATask, point_count_metrics
from olmo_eval.evals.vision.tasks.pointing import StylePrefixMixin

_SCORER = PointCountScorer()
_METRICS = point_count_metrics(_SCORER)


@register("pixmo_count")
class PixmoCountTask(StylePrefixMixin, ImageQATask):
    style = POINT_COUNT_STYLE
    sampling_params = SamplingParams(temperature=0.0, max_tokens=192)
    metrics = _METRICS
    primary_metric = _METRICS[0]  # correct
    split = Split.VALIDATION
    #: Prompt family assumed when the run does not say; matches the instruction-tuned
    #: checkpoints, like :class:`ModelPromptPointingTask`.
    default_prompt_templates = "uber_model_v2"

    def _question_for(self, label: str, idx: int) -> str:
        """The counting question, following the checkpoint's prompt family.

        mm_olmo's ``none`` family sends the bare lowercased label even for counting
        (``DataFormatter``'s ``point_count`` branch), so a pretrain checkpoint sees
        ``point_count: cats``; the templated families expand the label through the
        seeded counting templates as the instruction-tuned checkpoints were trained.
        """
        family = self.config.prompt_templates or self.default_prompt_templates
        if family == "none":
            question = label.lower()
        elif family in ("uber_model", "uber_model_v2"):
            question = pixmo_count_question(label, idx)
        else:
            raise ValueError(
                f"Unsupported prompt_templates {family!r} for pixmo_count; "
                "expected 'none', 'uber_model' or 'uber_model_v2'"
            )
        return self.apply_family_prefix(question)

    def _build_instances(self) -> Iterator[Instance]:
        import datasets

        ds = datasets.load_from_disk(str(torch_datasets_dir() / "pixmo_datasets" / "count"))
        ds = ds[self.config.split.value]
        # Arrow order is load-bearing: the per-example question template is
        # picked by an RNG seeded with the arrow index.
        for idx in range(len(ds)):
            ex = ds[idx]
            yield Instance(
                question=self._question_for(str(ex["label"]), idx),
                gold_answer=str(ex["count"]),
                metadata={
                    "count": ex["count"],
                    "label": ex["label"],
                    "arrow_idx": idx,
                    "example_id": ex["image_url"],
                    "image_url": ex["image_url"],
                    "image_path": rebase_data_path(ex["image"]),
                },
            )


register_variant("pixmo_count", "test", split=Split.TEST)
