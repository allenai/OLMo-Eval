"""Real-data pipeline tests for the 11 Molmo2 image-QA tasks.

Opt-in (reads the shared mm-olmo data tree, strictly read-only):

    RUN_REAL_DATASET_TESTS=1 \
    HF_DATASETS_CACHE=/weka/oe-training-default/mm-olmo/hf_datasets \
    HF_DATASETS_OFFLINE=1 \
    pytest tests/evals/tasks/test_image_qa_pipeline.py -v

Each task is checked for (a) the exact instance count of the original
mm_olmo eval split, and (b) oracle behavior: gold-answer responses score near
1.0 on the primary metric while corrupted responses score much lower.
"""

from __future__ import annotations

import asyncio
import os

import pytest

from olmo_eval.common.types import LMOutput, Response

if not os.environ.get("RUN_REAL_DATASET_TESTS"):
    pytest.skip(
        "Set RUN_REAL_DATASET_TESTS=1 (and HF_DATASETS_CACHE for the HF-hub tasks) "
        "to run real-data image-QA pipeline tests",
        allow_module_level=True,
    )

from olmo_eval.evals.tasks.common.registry import get_task  # noqa: E402

# (task spec, expected instance count)
EXPECTED_COUNTS = [
    ("chart_qa", 1920),
    ("vqa2", 8192),
    ("doc_qa", 5349),
    ("info_qa", 2801),
    ("text_vqa", 5000),
    ("real_world_qa", 765),
    ("mmmu", 900),
    ("math_vista", 1000),
    ("countbench_qa", 490),
    ("pixmo_count", 540),
    ("ai2d", 1980),
]

# Unlabeled test-split variants (predictions for eval-server submission):
# instance counts only — their metrics are computed against placeholder answers.
TEST_VARIANT_COUNTS = [
    ("doc_qa:test", 5188),
    ("info_qa:test", 3288),
]

CORRUPTED_TEXT = "the wrong answer entirely 424242"


def _score(task, texts):
    responses = []
    for instance, text in zip(task.instances, texts, strict=True):
        responses.append(
            Response(
                instance=instance,
                request=task.format_request(instance),
                outputs=[LMOutput(text=text)],
            )
        )
    responses = asyncio.run(task.score_responses(responses))
    metrics = task.compute_metrics(responses)
    primary = task.config.get_primary_metric()
    return next(iter(metrics[primary.name].values()))


@pytest.mark.parametrize(("spec", "expected"), EXPECTED_COUNTS + TEST_VARIANT_COUNTS)
def test_instance_count(spec: str, expected: int) -> None:
    task = get_task(spec)
    assert len(list(task.instances)) == expected


@pytest.mark.parametrize(("spec", "_"), EXPECTED_COUNTS)
def test_oracle_beats_corrupted(spec: str, _: int) -> None:
    # A slice is enough to separate oracle from corrupted decisively.
    task = get_task(spec, {"limit": 64})
    instances = list(task.instances)
    golds = [inst.gold_answer if inst.gold_answer is not None else "" for inst in instances]

    oracle = _score(task, golds)
    corrupted = _score(task, [CORRUPTED_TEXT] * len(instances))

    assert oracle >= 0.85, f"{spec}: oracle primary metric unexpectedly low ({oracle})"
    assert corrupted <= oracle - 0.3, (
        f"{spec}: corrupted ({corrupted}) not clearly below oracle ({oracle})"
    )
