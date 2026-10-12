"""VQA v2.0 validation (8,192-question subsample).

Mirrors mm_olmo's ``Vqa2Config(multi_question=False, sample=8192)``
(task name ``coco_2014_vqa_8192``): reads the prebuilt
``torch_datasets/vqa2/molmo_val.json`` manifest (never rebuilt here — the
loader is strictly read-only), flattens to one example per question, and
subsamples 8,192 questions with ``np.random.RandomState(9123)``.

Reference (Molmo2-4B ck2000): vqa_score=0.8582.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import numpy as np

from olmo_eval.common.types import Instance, SamplingParams, Split
from olmo_eval.evals.tasks.common import register
from olmo_eval.evals.vision.data.paths import molmo_data_dir, rebase_data_path, torch_datasets_dir
from olmo_eval.evals.vision.scoring.image_qa import VqaScoreScorer
from olmo_eval.evals.vision.tasks.image_qa import ImageQATask, MeanScorerMetric

_METRIC = MeanScorerMetric(name="vqa_score", scorer=VqaScoreScorer())


def _resolve_coco_image(path: str) -> str:
    """Resolve a COCO image path from the manifest.

    Prefers the recorded ``vqa2/<split>/`` location; falls back to the shared
    ``images/coco/<split>/`` tree (the ``vqa2`` image dirs are symlinks that
    may be broken on some mounts).
    """
    import os

    path = rebase_data_path(path)
    if os.path.exists(path):
        return path
    name = os.path.basename(path)  # e.g. COCO_val2014_000000123456.jpg
    coco_split = name.split("_")[1]
    fallback = molmo_data_dir() / "images" / "coco" / coco_split / name
    return str(fallback)


_SAMPLE_SIZE = 8192
_SAMPLE_SEED = 9123


@register("vqa2")
class Vqa2Task(ImageQATask):
    sampling_params = SamplingParams(temperature=0.0, max_tokens=12)
    metrics = (_METRIC,)
    primary_metric = _METRIC
    split = Split.VALIDATION

    def _build_instances(self) -> Iterator[Instance]:
        src = torch_datasets_dir() / "vqa2" / "molmo_val.json"
        if not src.exists():
            raise FileNotFoundError(
                f"{src} not found. This loader only reads the manifest cached by the "
                "original mm_olmo pipeline; it never (re)builds it."
            )
        with open(src) as f:
            data = json.load(f)

        flattened = []
        for item in data:
            for q in item["messages"]:
                flattened.append(
                    {
                        "question": q["question"],
                        "answers": q["answers"],
                        "image": item["image"],
                        "image_id": item["image_id"],
                        "question_id": q["question_id"],
                    }
                )
        # shuffle the list of dicts in place; the seeded RandomState ordering must
        # match mm_olmo exactly for dump parity, so keep np shuffle (not random.shuffle).
        np.random.RandomState(_SAMPLE_SEED).shuffle(flattened)  # ty: ignore[invalid-argument-type]
        flattened = flattened[:_SAMPLE_SIZE]

        for ex in flattened:
            answers = ex["answers"]
            yield Instance(
                question=f"vqa2: {ex['question']}",
                gold_answer=answers[0] if answers else None,
                metadata={
                    "answers": answers,
                    "image_id": ex["image_id"],
                    "example_id": ex["question_id"],
                    "image_path": _resolve_coco_image(ex["image"]),
                },
            )
