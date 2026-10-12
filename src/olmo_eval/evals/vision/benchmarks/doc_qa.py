"""DocVQA (validation by default; ``doc_qa:test`` for the test split).

Mirrors mm_olmo's ``DocQaConfig``: loads
``torch_datasets/docqa/val_v1.0_withQT.json`` (manual RRC Task 1 download),
prompts with the ``doc_qa`` style tag, scores ANLS (primary) + exact match.

The ``doc_qa:test`` variant loads ``test_v1.0.json``, whose answers are not
public — like mm_olmo, instances carry placeholder ``[""]`` answers, so the
computed metrics are meaningless; run it to produce predictions for an RRC
evaluation-server submission (``questionId`` is in ``metadata["example_id"]``).
"""

from __future__ import annotations

import json
from collections.abc import Iterator

from olmo_eval.common.types import Instance, SamplingParams, Split
from olmo_eval.evals.tasks.common import register, register_variant
from olmo_eval.evals.vision.data.paths import torch_datasets_dir
from olmo_eval.evals.vision.scoring.image_qa import AnlsScorer, EmScorer
from olmo_eval.evals.vision.tasks.image_qa import ImageQATask, MeanScorerMetric

_ANLS_METRIC = MeanScorerMetric(name="ansl", scorer=AnlsScorer())
_EM_METRIC = MeanScorerMetric(name="em", scorer=EmScorer())


@register("doc_qa")
class DocQaTask(ImageQATask):
    sampling_params = SamplingParams(temperature=0.0, max_tokens=12)
    metrics = (_ANLS_METRIC, _EM_METRIC)
    primary_metric = _ANLS_METRIC
    split = Split.VALIDATION

    def _build_instances(self) -> Iterator[Instance]:
        src_dir = torch_datasets_dir() / "docqa"
        if self.config.split == Split.TEST:
            src = src_dir / "test_v1.0.json"
        else:
            src = src_dir / "val_v1.0_withQT.json"
        with open(src) as f:
            data = json.load(f)
        for ex in data["data"]:
            # The test split has no public answers; mm_olmo injects [""].
            answers = ex.get("answers") or [""]
            yield Instance(
                question=f"doc_qa: {ex['question']}",
                gold_answer=answers[0] if answers[0] else None,
                metadata={
                    "answers": answers,
                    "example_id": ex["questionId"],
                    "doc_id": ex["docId"],
                    "question_types": ex.get("question_types") or [""],
                    "image_path": str(src_dir / ex["image"]),
                },
            )


register_variant("doc_qa", "test", split=Split.TEST)
