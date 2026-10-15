"""Tests that a zero ``num_fewshot`` yields no few-shot examples on every task."""

import pytest

from olmo_eval.common.types import Instance
from olmo_eval.evals.tasks.common import get_task

ZERO_SHOT_SPECS = [
    "arc_challenge:olmes",
    "arc_easy:olmes",
    "csqa:olmes",
    "drop:olmo3base",
    "drop:rc",
    "gsm_symbolic",
    "hellaswag:olmo3base",
    "jeopardy",
    "jeopardy:mc",
    "minerva_math_algebra:olmes",
    "naturalqs:olmo3base",
    "naturalqs:mc",
    "piqa:olmes",
    "socialiqa:olmes",
    "squad",
    "squad:mc",
    "winogrande:olmo3base",
    "mmlu_abstract_algebra:mc",
    "mmlu_abstract_algebra:rc",
    "bbq:base",
    "wmdp:base",
]


def _no_dataset_load(**kwargs):
    raise AssertionError("zero-shot must not load a few-shot split")


@pytest.mark.parametrize("task_spec", ZERO_SHOT_SPECS)
def test_zero_shot_override_returns_no_fewshot(
    task_spec: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = get_task(task_spec, {"num_fewshot": 0})
    monkeypatch.setattr(task, "_build_fewshot_from_source", _no_dataset_load)

    assert task.config.num_fewshot == 0
    assert task.get_fewshot() == []


@pytest.mark.parametrize(
    ("task_spec", "expected"),
    [("arc_challenge:olmes", 5), ("gsm_symbolic", 8), ("squad", 5), ("winogrande:olmo3base", 5)],
)
def test_fixed_pool_honors_positive_count(task_spec: str, expected: int) -> None:
    task = get_task(task_spec)

    assert task.config.num_fewshot == expected
    assert len(task.get_fewshot()) == expected


@pytest.mark.parametrize(
    "task_spec", ["mmlu_abstract_algebra:mc", "mmlu_abstract_algebra:rc", "bbq:base", "wmdp:base"]
)
def test_dev_split_tasks_take_first_k(task_spec: str, monkeypatch: pytest.MonkeyPatch) -> None:
    task = get_task(task_spec, {"num_fewshot": 2})
    pool = [Instance(question=f"q{i}", gold_answer="A") for i in range(6)]
    monkeypatch.setattr(task, "_build_fewshot_from_source", lambda **kwargs: pool)

    assert task.get_fewshot() == pool[:2]


def test_zero_shot_request_has_no_examples() -> None:
    task = get_task("gsm_symbolic", {"num_fewshot": 0})
    fewshot_question = get_task("gsm_symbolic").get_fewshot()[0].question
    instance = Instance(question="What is 2 + 2?", gold_answer="4")

    request = task.format_request(instance)

    assert fewshot_question not in str(request)
