"""Tests for the MMLU-Pro tasks."""

import pytest

from olmo_eval.common.metrics import BPBMetricInstanceAvg
from olmo_eval.common.types import Instance, LMOutput, RequestType
from olmo_eval.evals.suites.registry import get_suite
from olmo_eval.evals.tasks.common import get_task
from olmo_eval.evals.tasks.mmlu_pro import (
    _COT_DESCRIPTION,
    _COT_FINAL_DESCRIPTION,
    MMLU_PRO_CATEGORIES,
    MMLU_PRO_REVISION,
    MMLUProCoTExactMatchScorer,
    _extract_cot_answer,
    category_slug,
)

_OPTIONS = ["3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]
_DOC = {
    "question_id": 7,
    "question": "What is 2 + 2?",
    "options": _OPTIONS,
    "answer": "B",
    "answer_index": 1,
    "cot_content": "",
    "category": "math",
    "src": "test",
}
_SLUGS = tuple(category_slug(c) for c in MMLU_PRO_CATEGORIES)


def _doc(**overrides) -> dict:
    return {**_DOC, **overrides}


def test_all_categories_registered() -> None:
    assert len(MMLU_PRO_CATEGORIES) == 14
    for slug in _SLUGS:
        name = f"mmlu_pro_{slug}"
        mc = get_task(name)
        assert mc.request_type == RequestType.LOGLIKELIHOOD
        assert mc.config.num_fewshot == 5
        assert get_task(f"{name}:mc").config.name == name
        assert get_task(f"{name}:olmo3base").config.name == name

        rc = get_task(f"{name}:rc")
        assert rc.request_type == RequestType.LOGLIKELIHOOD
        assert rc.config.num_fewshot == 5
        assert get_task(f"{name}:rc:olmo3base").config.name == f"{name}:rc"
        bpb = get_task(f"{name}:rc:bpb")
        assert isinstance(bpb.config.primary_metric, BPBMetricInstanceAvg)

        cot = get_task(f"{name}:cot")
        assert cot.request_type == RequestType.CHAT
        assert cot.config.num_fewshot == 0
        assert cot.config.strip_thinking is True

        for task in (mc, rc, cot):
            assert task.config.data_source is not None
            assert task.config.data_source.revision == MMLU_PRO_REVISION


def test_computer_science_slug() -> None:
    task = get_task("mmlu_pro_computer_science")
    assert task.category == "computer science"
    assert task.process_doc(_doc(category="computer science")) is not None
    assert task.process_doc(_doc(category="computer_science")) is None


def test_suite_sizes() -> None:
    assert len(get_suite("mmlu_pro").tasks) == 14
    assert len(get_suite("mmlu_pro:cot").tasks) == 14
    bpb_tasks = get_suite("mmlu_pro:bpb").tasks
    assert len(bpb_tasks) == 14
    assert all(str(t).endswith(":rc:bpb") for t in bpb_tasks)


def test_mc_prompt_shape() -> None:
    instance = get_task("mmlu_pro_math").process_doc(_doc(), index=3)
    assert instance is not None
    expected_choices = "\n".join(
        f" {label}. {text}" for label, text in zip("ABCDEFGHIJ", _OPTIONS, strict=True)
    )
    assert instance.question == f"Question: What is 2 + 2?\n{expected_choices}\nAnswer:"
    assert instance.choices == tuple("ABCDEFGHIJ")
    assert instance.gold_answer == "B"
    assert instance.metadata == {"id": 7, "index": 3, "category": "math", "gold_idx": 1}


def test_mc_fewshot_request() -> None:
    task = get_task("mmlu_pro_math")
    ex1 = task.process_doc(_doc(question="First?", answer_index=2))
    ex2 = task.process_doc(_doc(question="Second?", answer_index=9))
    instance = task.process_doc(_doc())
    assert ex1 is not None and ex2 is not None and instance is not None
    task._fewshot_cache = [ex1, ex2]

    request = task.format_request(instance)
    assert request.request_type == RequestType.LOGLIKELIHOOD
    assert request.prompt == f"{ex1.question} C\n\n{ex2.question} J\n\n{instance.question}"
    assert request.prompt.startswith("Question: First?")
    assert request.continuations == tuple(f" {label}" for label in "ABCDEFGHIJ")


def test_rc_prompt_and_continuations() -> None:
    task = get_task("mmlu_pro_math:rc")
    example = task.process_doc(_doc(question="First?", answer_index=2))
    instance = task.process_doc(_doc())
    assert example is not None and instance is not None
    assert instance.question == "What is 2 + 2?"
    assert instance.gold_answer == "4"
    assert instance.choices == tuple(_OPTIONS)
    task._fewshot_cache = [example]

    request = task.format_request(instance)
    assert request.prompt == "Question: First?\nAnswer: 5\n\nQuestion: What is 2 + 2?\nAnswer:"
    assert request.continuations == tuple(f" {o}" for o in _OPTIONS)


def test_cot_message() -> None:
    task = get_task("mmlu_pro_math:cot")
    instance = task.process_doc(_doc())
    assert instance is not None
    assert instance.gold_answer == "B"
    assert instance.choices == tuple(_OPTIONS)
    assert instance.metadata == {"id": 7, "index": 0, "category": "math"}

    request = task.format_request(instance)
    assert request.request_type == RequestType.CHAT
    assert len(request.messages) == 1
    assert request.messages[0]["role"] == "user"
    query = "Question: What is 2 + 2?\n" + "".join(
        f" ({label}) {text}\n" for label, text in zip("ABCDEFGHIJ", _OPTIONS, strict=True)
    )
    assert request.messages[0]["content"] == _COT_DESCRIPTION + query + _COT_FINAL_DESCRIPTION

    params = task.config.sampling_params
    assert params.temperature == 0.6
    assert params.top_p == 0.95
    assert params.max_tokens is None
    assert params.stop_sequences is None


def test_cot_sampling_matches_mmlu_cot() -> None:
    assert (
        get_task("mmlu_pro_math:cot").config.sampling_params
        == get_task("mmlu_abstract_algebra:cot").config.sampling_params
    )


def test_category_filter_returns_none() -> None:
    assert get_task("mmlu_pro_law").process_doc(_doc()) is None
    assert get_task("mmlu_pro_law:rc").process_doc(_doc()) is None
    assert get_task("mmlu_pro_law:cot").process_doc(_doc()) is None


@pytest.mark.parametrize(
    "doc",
    [
        _doc(options=[]),
        _doc(question=""),
        _doc(answer_index=10),
        _doc(answer_index="B"),
        _doc(options=[str(i) for i in range(16)]),
    ],
)
def test_skips_malformed_docs(doc: dict) -> None:
    assert get_task("mmlu_pro_math").process_doc(doc) is None


def test_fewshot_slices_first_k(monkeypatch: pytest.MonkeyPatch) -> None:
    task = get_task("mmlu_pro_math")
    instances = [Instance(question=f"q{i}", gold_answer="A") for i in range(7)]
    monkeypatch.setattr(task, "_build_fewshot_from_source", lambda **kwargs: instances)
    assert task.get_fewshot() == instances[:5]


def test_cot_skips_fewshot_load(monkeypatch: pytest.MonkeyPatch) -> None:
    task = get_task("mmlu_pro_math:cot")

    def _fail(**kwargs):
        raise AssertionError("0-shot task should not load few-shot examples")

    monkeypatch.setattr(task, "_build_fewshot_from_source", _fail)
    assert task.get_fewshot() == []


@pytest.mark.parametrize(
    ("text", "answer", "format_correct"),
    [
        ("Therefore, the answer is (C). Wait. Therefore, the answer is (D)", "D", 1.0),
        ("Reasoning.\n\nTherefore, the answer is (B)", "B", 1.0),
        ("Therefore, the answer is (J)", "J", 1.0),
        ("therefore the answer is: G", "G", 1.0),
        ("So the answer is H.", "H", 0.5),
        ("The correct answer is: F", "F", 0.5),
        ("The answer is (B).", "B", 0.5),
        ("Answer: D", "D", 0.5),
        ("answer: A, no answer: C\nanswer: D", "D", 0.5),
        ("Option (E) is correct", "E", 0.5),
        (r"Final: \boxed{I}", "I", 0.5),
        ("Choose (J) here", "J", 0.5),
        ("the answer is: b", "b", 0.5),
        ("Therefore, the answer is (K)", "", 0.0),
        ("nothing here", "", 0.0),
        ("", "", 0.0),
        ("<think>(A) looks right</think>Therefore, the answer is (B)", "B", 1.0),
    ],
)
def test_extraction_cascade(text: str, answer: str, format_correct: float) -> None:
    assert _extract_cot_answer(text) == (answer, format_correct)


def test_scorer_records_format_correct() -> None:
    instance = Instance(question="q", gold_answer="B")
    scorer = MMLUProCoTExactMatchScorer()

    output = LMOutput(text="the answer is: b")
    assert scorer.score(instance, output) == 1.0
    assert output.metadata["answer_format_correct"] == 0.5

    output = LMOutput(text="Therefore, the answer is (C)")
    assert scorer.score(instance, output) == 0.0
    assert output.metadata["answer_format_correct"] == 1.0


def test_extract_answer_returns_letter() -> None:
    task = get_task("mmlu_pro_math:cot")
    assert task.extract_answer(LMOutput(text="Therefore, the answer is (J)")) == "J"
    assert task.extract_answer(LMOutput(text="")) is None


def test_task_configs_differ_between_recipes() -> None:
    specs = ("mmlu_pro_math", "mmlu_pro_math:rc", "mmlu_pro_math:rc:bpb", "mmlu_pro_math:cot")
    dicts = [get_task(spec).config.to_dict() for spec in specs]
    for i, left in enumerate(dicts):
        for right in dicts[i + 1 :]:
            assert left != right
