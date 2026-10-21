"""Tests for the MMLU chain-of-thought tasks."""

import pytest

from olmo_eval.common.scorers.base import MultipleChoiceScorer
from olmo_eval.common.types import Instance, LMOutput, RequestType
from olmo_eval.evals.suites.registry import get_suite
from olmo_eval.evals.tasks.common import get_task
from olmo_eval.evals.tasks.mmlu import MMLU_SUBJECTS

_DOC = {
    "question": "What is 2 + 2?",
    "choices": ["3", "4", "5", "6"],
    "answer": 1,
}
_INSTANCE = Instance(question="q", gold_answer="B", metadata={})


def _score(text: str) -> float:
    task = get_task("mmlu_anatomy:cot")
    output = LMOutput(text=text)
    output.extracted_answer = task.extract_answer(output)
    return MultipleChoiceScorer().score(_INSTANCE, output)


def test_all_subjects_registered() -> None:
    assert len(MMLU_SUBJECTS) == 57
    for subject in MMLU_SUBJECTS:
        task = get_task(f"mmlu_{subject}:cot")
        assert task.request_type == RequestType.CHAT
        assert task.config.num_fewshot == 0


def test_sampling_matches_reasoning_regime() -> None:
    params = get_task("mmlu_anatomy:cot").config.sampling_params
    assert params.temperature == 0.6
    assert params.top_p == 0.95
    assert params.max_tokens is None


def test_suite_covers_every_subject() -> None:
    assert len(get_suite("mmlu:cot").tasks) == 57


def test_prompt_shape() -> None:
    task = get_task("mmlu_abstract_algebra:cot")
    instance = task.process_doc(dict(_DOC), index=0)
    assert instance is not None
    assert instance.question.startswith(
        "The following are multiple choice questions about abstract algebra."
    )
    assert "Question: What is 2 + 2?\n A. 3\n B. 4\n C. 5\n D. 6\n" in instance.question
    assert instance.gold_answer == "B"


def test_single_user_message() -> None:
    task = get_task("mmlu_anatomy:cot")
    instance = task.process_doc(dict(_DOC), index=0)
    assert instance is not None
    request = task.format_request(instance)
    assert request.request_type == RequestType.CHAT
    assert len(request.messages) == 1
    assert request.messages[0]["role"] == "user"


@pytest.mark.parametrize(
    "doc",
    [
        {"question": "", "choices": ["a"], "answer": 0},
        {"question": "q", "choices": [], "answer": 0},
        {"question": "q", "choices": ["a"], "answer": "B"},
    ],
)
def test_skips_malformed_docs(doc: dict) -> None:
    assert get_task("mmlu_anatomy:cot").process_doc(doc) is None


@pytest.mark.parametrize(
    "text",
    [
        "Reasoning.\n\nTherefore, the answer is: B",
        "Therefore, the answer is: (B)",
        "\\boxed{B}",
        "So the answer is B.",
        "The correct answer is: (B)",
        "the ANSWER is (b)",
        "<think>Could be A or C... no, D. Hmm.</think>Therefore, the answer is: B",
    ],
)
def test_extracts_letter(text: str) -> None:
    assert _score(text) == 1.0


@pytest.mark.parametrize(
    "text",
    ["Therefore, the answer is: C", "", "<think>Therefore, the answer is: B</think>"],
)
def test_wrong_and_empty(text: str) -> None:
    assert _score(text) == 0.0


def test_template_priority_beats_position() -> None:
    """The requested phrasing outranks a later, weaker one.

    The cascade tries templates in priority order, so "Therefore, the
    answer is: A" wins over a subsequent "the answer is B" — verified
    to match the reference implementation's extraction.
    """
    assert _score("Therefore, the answer is: A. Wait, no — the answer is B") == 0.0
    assert _score("Some reasoning. The answer is A. Therefore, the answer is: B") == 1.0
