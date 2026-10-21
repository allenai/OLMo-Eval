"""ChartQA (human + augmented parts).

Mirrors mm_olmo's ``ChartQaConfig`` (``parts="both"``): loads
``torch_datasets/chartqa/{split}/{split}_{human,augmented}.json`` (human part
first), prompts with the ``chart_qa`` style tag, and scores relaxed
correctness / scifi relaxed correctness / exact match with ``_human`` /
``_aug`` breakdowns.

Reference (Molmo2-4B ck2000, val): relaxed_correctness=0.8380, em=0.7490,
scifi_relaxed_correctness=0.8516.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

from olmo_eval.common.metrics.base import Metric
from olmo_eval.common.types import Instance, SamplingParams, Split
from olmo_eval.evals.tasks.common import register, register_variant
from olmo_eval.evals.vision.data.paths import torch_datasets_dir
from olmo_eval.evals.vision.scoring.image_qa import (
    EmScorer,
    RelaxedCorrectnessScorer,
    ScifiRelaxedScorer,
)
from olmo_eval.evals.vision.tasks.image_qa import ChartQaSubsetMetric, ImageQATask

_RELAXED = RelaxedCorrectnessScorer()
_SCIFI = ScifiRelaxedScorer()
_EM = EmScorer()

_METRICS: tuple[Metric, ...] = tuple(
    ChartQaSubsetMetric(name=scorer.name + suffix, scorer=scorer, subset=subset)
    for scorer in (_RELAXED, _SCIFI, _EM)
    for subset, suffix in (("all", ""), ("human", "_human"), ("aug", "_aug"))
)


@register("chart_qa")
class ChartQaTask(ImageQATask):
    sampling_params = SamplingParams(temperature=0.0, max_tokens=12)
    metrics = _METRICS
    primary_metric = _METRICS[0]  # relaxed_correctness (overall)
    split = Split.VALIDATION

    def _build_instances(self) -> Iterator[Instance]:
        split = "val" if self.config.split == Split.VALIDATION else "test"
        src_dir = torch_datasets_dir() / "chartqa" / split
        for part in ("human", "augmented"):
            with open(src_dir / f"{split}_{part}.json") as f:
                data = json.load(f)
            for ex_id, ex in enumerate(data):
                label = ex["label"]
                yield Instance(
                    question=f"chart_qa: {ex['query']}",
                    gold_answer=label if isinstance(label, str) else label[0],
                    metadata={
                        "answers": label,
                        "is_human": part == "human",
                        "example_id": ex_id,
                        "image_path": str(src_dir / "png" / ex["imgname"]),
                    },
                )


register_variant("chart_qa", "test", split=Split.TEST)
