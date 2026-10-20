"""Tests for all task module registrations and basic functionality."""

import pytest

from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, Response
from olmo_eval.evals.tasks.common import OutputScoreAggregation, get_task, list_tasks
from olmo_eval.evals.tasks.gsm8k import (
    GSM8K_PLATINUM_REPO,
    GSM8K_PLATINUM_REVISION,
    _clean_short_answer,
    _extract_boxed_or_last_number,
    _extract_last_number,
)


class TestGSMNumberExtraction:
    """Tests for GSM8K/GSM-Symbolic number extraction helpers."""

    @pytest.mark.parametrize(
        "text, expected",
        [
            ("The answer is 42.", "42"),
            ("So the answer is 6.", "6"),
            ("5 + 4 = 9. So the answer is 9.", "9"),
            # Negative integers
            ("The temperature dropped to -3 degrees.", "-3"),
            ("The answer is -15.", "-15"),
            # Positive sign
            ("The result is +7.", "+7"),
            # Floats
            ("The answer is 3.14.", "3.14"),
            ("She has -2.5 dollars.", "-2.5"),
            ("A gain of +0.75 points.", "+0.75"),
            # Comma-separated numbers
            ("The population is 1,000,000.", "1000000"),
            ("He earned $2,500.", "2500"),
            # Last number wins
            ("First 10, then 20, finally 30.", "30"),
            ("Step 1: 100 - 60 = 40. Step 2: 40 - 8 = 32.", "32"),
            # No numbers
            ("There is no number here.", None),
            ("", None),
            # Edge: bare decimal
            (".5 liters", ".5"),
        ],
    )
    def test_extract_last_number(self, text: str, expected: str | None):
        assert _extract_last_number(text) == expected

    @pytest.mark.parametrize(
        "text, expected",
        [
            ("6", "6"),
            ("-3", "-3"),
            ("3.14", "3.14"),
            ("1,200", "1200"),
            # Falls back to original text when no number is found
            ("no number", "no number"),
        ],
    )
    def test_clean_short_answer(self, text: str, expected: str):
        assert _clean_short_answer(text) == expected


def test_gsm8k_olmo3base_uses_first_sample_exact_match() -> None:
    task = get_task("gsm8k:olmo3base")
    assert task.config.output_score_aggregation == OutputScoreAggregation.FIRST


def test_gsm8k_platinum_uses_pinned_platinum_data() -> None:
    task = get_task("gsm8k:platinum")
    source = task.config.data_source
    assert source.path == GSM8K_PLATINUM_REPO
    assert source.subset == "main"
    assert source.split == "test"
    assert source.revision == GSM8K_PLATINUM_REVISION


def test_gsm8k_platinum_keeps_base_task_settings() -> None:
    base = get_task("gsm8k").config
    platinum = get_task("gsm8k:platinum").config
    assert platinum.num_fewshot == base.num_fewshot
    assert platinum.sampling_params == base.sampling_params
    assert platinum.metrics == base.metrics


def test_gsm8k_platinum_process_doc_parses_revised_answer() -> None:
    task = get_task("gsm8k:platinum")
    doc = {
        "question": "How many apples?",
        "answer": "3 + 4 = <<3+4=7>>7\n#### 1,207",
        "cleaning_status": "revised",
    }

    instance = task.process_doc(doc, index=5)

    assert instance is not None
    assert instance.gold_answer == "1207"
    assert instance.metadata["id"] == 5


@pytest.mark.parametrize(
    "text, expected",
    [
        ("So the total is \\boxed{18}.", "18"),
        ("Step 1 gives 5. Final: \\boxed{\\$1,207} dollars and 3 cents", "1207"),
        ("First \\boxed{3}, corrected to \\boxed{4}.", "4"),
        # Trailing decimal zeros are dropped so money answers match integer golds
        ("She pays \\boxed{57.00} dollars.", "57"),
        ("\\boxed{2.50}", "2.5"),
        ("\\boxed{0.00}", "0"),
        ("The total is 12.0", "12"),
        # Falls back to the last number without a boxed answer
        ("The answer is 42.", "42"),
        # Falls back when the boxed answer holds no number
        ("\\boxed{none} but 7 remain", "7"),
        ("No answer here.", None),
    ],
)
def test_extract_boxed_or_last_number(text: str, expected: str | None) -> None:
    assert _extract_boxed_or_last_number(text) == expected


def test_gsm8k_chat_builds_zero_shot_chat_request() -> None:
    task = get_task("gsm8k:chat")
    instance = Instance(question="What is one plus one?", gold_answer="2")

    request = task.format_request(instance)

    assert request.request_type == RequestType.CHAT
    assert request.messages is not None
    assert len(request.messages) == 1
    content = request.messages[0]["content"]
    assert content.startswith("What is one plus one?")
    assert "\\boxed{}" in content


def test_gsm8k_chat_generation_settings() -> None:
    config = get_task("gsm8k:chat").config
    assert config.strip_thinking is True
    assert config.sampling_params is not None
    assert config.sampling_params.max_tokens is None
    assert not config.sampling_params.stop_sequences


def test_gsm8k_chat_extracts_boxed_answer_after_thinking() -> None:
    task = get_task("gsm8k:chat")
    output = LMOutput(text="<think>Maybe 5? No, 99.</think>\nThe answer is \\boxed{18}. Check: 2")
    instance = Instance(question="q", gold_answer="18")
    response = Response(instance=instance, request=task.format_request(instance), outputs=[output])

    task.strip_thinking_traces([response])

    assert task.extract_answer(response.outputs[0]) == "18"


def test_gsm8k_chat_composes_with_platinum() -> None:
    config = get_task("gsm8k:chat:platinum").config
    assert config.data_source.path == GSM8K_PLATINUM_REPO
    assert config.strip_thinking is True


@pytest.mark.parametrize("num_fewshot", [0, 3, 8])
def test_gsm8k_prompt_honors_num_fewshot(num_fewshot: int) -> None:
    task = get_task("gsm8k", config_overrides={"num_fewshot": num_fewshot})
    instance = Instance(question="What is one plus one?", gold_answer="2")

    prompt = task.format_request(instance).prompt

    assert prompt.count("Question:") == num_fewshot + 1
    assert prompt.endswith("Question: What is one plus one?\nAnswer:")


class TestTaskRegistration:
    """Tests for task registration across all modules."""

    @pytest.fixture(autouse=True)
    def setup_registry(self):
        """Ensure tasks are registered by importing modules."""
        import olmo_eval.evals.tasks  # noqa: F401

        yield

    # HumanEval
    def test_humaneval_registered(self):
        """Test that humaneval is registered."""
        assert "humaneval" in list_tasks()

    def test_humaneval_plus_registered(self):
        """Test that humaneval_plus is registered."""
        assert "humaneval_plus" in list_tasks()

    def test_get_humaneval(self):
        """Test getting humaneval task."""
        task = get_task("humaneval")
        assert task.config.name == "humaneval"

    def test_get_humaneval_plus(self):
        """Test getting humaneval_plus task."""
        task = get_task("humaneval_plus")
        assert task.config.name == "humaneval_plus"

    # MBPP
    def test_mbpp_registered(self):
        """Test that mbpp is registered."""
        assert "mbpp" in list_tasks()

    def test_mbpp_plus_registered(self):
        """Test that mbpp_plus is registered."""
        assert "mbpp_plus" in list_tasks()

    def test_get_mbpp(self):
        """Test getting mbpp task."""
        task = get_task("mbpp")
        assert task.config.name == "mbpp"

    def test_get_mbpp_plus(self):
        """Test getting mbpp_plus task."""
        task = get_task("mbpp_plus")
        assert task.config.name == "mbpp_plus"

    # Multilingual MBPP (sample languages)
    def test_mt_mbpp_python_registered(self):
        """Test that mt_mbpp_python is registered."""
        assert "mt_mbpp_python" in list_tasks()

    def test_mt_mbpp_javascript_registered(self):
        """Test that mt_mbpp_javascript is registered."""
        assert "mt_mbpp_javascript" in list_tasks()

    def test_mt_mbpp_v2fix_python_registered(self):
        """Test that mt_mbpp_v2fix_python is registered."""
        assert "mt_mbpp_v2fix_python" in list_tasks()

    def test_get_mt_mbpp_python(self):
        """Test getting mt_mbpp_python task."""
        task = get_task("mt_mbpp_python")
        assert task.config.name == "mt_mbpp_python"


class TestHumanEvalTask:
    """Tests for HumanEval task functionality."""

    @pytest.fixture
    def task(self):
        """Create a HumanEval task."""
        return get_task("humaneval")

    def test_format_request(self, task):
        """Test request formatting."""
        instance = Instance(
            question="```python\ndef add(a, b):",
            gold_answer="    return a + b",
            metadata={
                "id": "test_0",
                "entry_point": "add",
                "answer_prefix": "def add(a, b):",
                "test": "assert add(1, 2) == 3",
            },
        )

        request = task.format_request(instance)
        assert request.request_type == RequestType.COMPLETION
        assert "def add" in request.prompt


class TestMBPPTask:
    """Tests for MBPP task functionality."""

    @pytest.fixture
    def task(self):
        """Create a MBPP task."""
        return get_task("mbpp")

    def test_format_request(self, task):
        """Test request formatting."""
        instance = Instance(
            question="Write a function to add two numbers.\ndef add(a, b):",
            gold_answer="    return a + b",
            metadata={
                "id": 1,
                "answer_prefix": "def add(a, b):",
                "test": "assert add(1, 2) == 3",
            },
        )

        request = task.format_request(instance)
        assert request.request_type == RequestType.COMPLETION
        assert "add" in request.prompt

    @pytest.mark.anyio
    async def test_bpb_scoring_does_not_require_answer_prefix(self):
        """MBPP BPB scoring should not depend on completion-only metadata."""
        task = get_task("mbpp:bpb")
        instance = Instance(
            question="Write a function to add two numbers.\n```python\n",
            gold_answer="def add(a, b):\n    return a + b\n```",
            metadata={
                "id": 1,
                "test": "assert add(1, 2) == 3",
            },
        )
        request = LMRequest(
            request_type=RequestType.LOGLIKELIHOOD,
            prompt=instance.question,
            continuations=(instance.gold_answer,),
        )
        response = Response(
            instance=instance,
            request=request,
            outputs=[
                LMOutput(
                    text=instance.gold_answer,
                    logprobs=[
                        {"token": "def", "logprob": -1.0},
                        {"token": " add", "logprob": -1.0},
                    ],
                )
            ],
        )

        scored = await task.score_responses([response])

        assert scored[0].outputs[0].extracted_answer.startswith("def add(a, b):")
        assert "bits_per_byte" in scored[0].scores


class TestMultilingualMBPPTask:
    """Tests for Multilingual MBPP task functionality."""

    @pytest.fixture
    def task(self):
        """Create a Multilingual MBPP Python task."""
        return get_task("mt_mbpp_python")

    def test_format_request(self, task):
        """Test request formatting."""
        instance = Instance(
            question="Write a function to add two numbers.\n```python\n",
            gold_answer="def add(a, b):\n    return a + b\n```",
            metadata={
                "id": 1,
                "language": "python",
                "text": "Write a function to add two numbers.",
                "code": "def add(a, b):\n    return a + b",
            },
        )

        request = task.format_request(instance)
        assert request.request_type == RequestType.COMPLETION
        assert "python" in request.prompt

    def test_extract_answer(self, task):
        """Test answer extraction."""
        output = LMOutput(text="def add(a, b):\n    return a + b\n```")
        answer = task.extract_answer(output)
        assert "def add" in answer
        assert "return a + b" in answer
