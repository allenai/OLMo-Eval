"""Tests for LAB-Bench task logic."""

import pytest

from olmo_eval.common.scorers import MultipleChoiceScorer
from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, Response
from olmo_eval.evals.tasks.common import get_task, list_tasks
from olmo_eval.evals.tasks.lab_bench import CoverageMetric, PrecisionMetric


@pytest.fixture(autouse=True)
def _setup_registry():
    import olmo_eval.evals.tasks  # noqa: F401


_ALL_TASKS = (
    "lab_bench_litqa2",
    "lab_bench_dbqa",
    "lab_bench_seqqa",
    "lab_bench_protocolqa",
    "lab_bench_suppqa",
    "lab_bench_cloning_scenarios",
)


class TestLabBenchRegistration:
    """Tests for LAB-Bench task registration."""

    @pytest.mark.parametrize("task_name", _ALL_TASKS)
    def test_task_registered(self, task_name):
        assert task_name in list_tasks()

    @pytest.mark.parametrize("task_name", _ALL_TASKS)
    def test_get_task(self, task_name):
        task = get_task(task_name)
        assert task.config.name == task_name

    @pytest.mark.parametrize("task_name", _ALL_TASKS)
    def test_mc_variant(self, task_name):
        task = get_task(f"{task_name}:mc")
        assert task is not None

    @pytest.mark.parametrize("task_name", _ALL_TASKS)
    def test_bpb_variant(self, task_name):
        task = get_task(f"{task_name}:bpb")
        assert task is not None


class TestProcessDoc:
    """Tests for LabBenchTask.process_doc."""

    @pytest.fixture
    def task(self):
        return get_task("lab_bench_litqa2")

    def test_basic_conversion(self, task):
        doc = {
            "question": "What enzyme catalyzes X?",
            "ideal": "Kinase A",
            "distractors": ["Kinase B", "Kinase C"],
        }
        instance = task.process_doc(doc, index=0)

        assert instance is not None
        assert instance.question == "What enzyme catalyzes X?"
        assert len(instance.choices) == 4  # ideal + 2 distractors + refuse
        assert "Kinase A" in instance.choices
        assert "Kinase B" in instance.choices
        assert "Kinase C" in instance.choices
        assert "Insufficient information to answer the question" in instance.choices

    def test_gold_letter_points_to_ideal(self, task):
        doc = {
            "question": "What is the answer?",
            "ideal": "Correct answer",
            "distractors": ["Wrong 1", "Wrong 2"],
        }
        instance = task.process_doc(doc, index=42)

        gold_idx = instance.metadata["gold_idx"]
        assert instance.choices[gold_idx] == "Correct answer"
        assert instance.gold_answer == chr(ord("A") + gold_idx)

    def test_shuffle_determinism(self, task):
        doc = {
            "question": "Q?",
            "ideal": "Ideal",
            "distractors": ["D1", "D2", "D3"],
        }
        inst1 = task.process_doc(doc, index=7)
        inst2 = task.process_doc(doc, index=7)
        assert inst1.choices == inst2.choices
        assert inst1.gold_answer == inst2.gold_answer

    def test_different_indices_produce_different_shuffles(self, task):
        doc = {
            "question": "Q?",
            "ideal": "Ideal",
            "distractors": ["D1", "D2", "D3", "D4", "D5"],
        }
        inst0 = task.process_doc(doc, index=0)
        inst1 = task.process_doc(doc, index=1)
        assert inst0.choices != inst1.choices

    def test_deduplicates_ideal_in_distractors(self, task):
        doc = {
            "question": "Q?",
            "ideal": "Same",
            "distractors": ["Same", "Different"],
        }
        instance = task.process_doc(doc, index=0)
        assert instance.choices.count("Same") == 1
        assert len(instance.choices) == 3  # "Same" + "Different" + refuse option

    def test_metadata_contains_gold_idx_and_text(self, task):
        doc = {
            "question": "Q?",
            "ideal": "The right answer",
            "distractors": ["Wrong"],
        }
        instance = task.process_doc(doc, index=0)
        assert "gold_idx" in instance.metadata
        assert instance.metadata["gold_text"] == "The right answer"

    def test_metadata_contains_refuse_idx(self, task):
        doc = {
            "question": "Q?",
            "ideal": "Answer",
            "distractors": ["Wrong 1", "Wrong 2"],
        }
        instance = task.process_doc(doc, index=0)
        refuse_idx = instance.metadata["refuse_idx"]
        assert instance.choices[refuse_idx] == "Insufficient information to answer the question"

    def test_skip_missing_question(self, task):
        doc = {"question": "", "ideal": "Answer", "distractors": ["D1"]}
        assert task.process_doc(doc, index=0) is None

    def test_skip_missing_ideal(self, task):
        doc = {"question": "Q?", "ideal": "", "distractors": ["D1"]}
        assert task.process_doc(doc, index=0) is None


class TestProtocolQAProcessDoc:
    """Tests for ProtocolQA protocol injection."""

    @pytest.fixture
    def task(self):
        return get_task("lab_bench_protocolqa")

    def test_protocol_prepended_to_question(self, task):
        doc = {
            "question": "What went wrong?",
            "ideal": "Step 3 was skipped",
            "distractors": ["Step 1 was skipped"],
            "protocol": "Step 1: Mix reagents\nStep 2: Incubate\nStep 3: Centrifuge",
        }
        instance = task.process_doc(doc, index=0)
        assert "Protocol:" in instance.question
        assert "Step 1: Mix reagents" in instance.question
        assert "Question: What went wrong?" in instance.question

    def test_missing_protocol_uses_question_only(self, task):
        doc = {
            "question": "What went wrong?",
            "ideal": "Unknown",
            "distractors": ["Nothing"],
        }
        instance = task.process_doc(doc, index=0)
        assert instance.question == "What went wrong?"


class TestExtractAnswer:
    """Tests for LabBenchTask.extract_answer."""

    @pytest.fixture
    def task(self):
        return get_task("lab_bench_litqa2")

    def test_answer_pattern(self, task):
        output = LMOutput(text="The answer is clearly B because...\nANSWER: B")
        assert task.extract_answer(output) == "B"

    def test_answer_pattern_lowercase(self, task):
        output = LMOutput(text="answer: c")
        assert task.extract_answer(output) == "C"

    def test_answer_pattern_no_space(self, task):
        output = LMOutput(text="ANSWER:A")
        assert task.extract_answer(output) == "A"

    def test_answer_pattern_bold_label(self, task):
        output = LMOutput(text="**Answer:** B")
        assert task.extract_answer(output) == "B"

    def test_answer_pattern_followed_by_word_returns_none(self, task):
        output = LMOutput(text="ANSWER: the correct choice is C")
        assert task.extract_answer(output) is None

    def test_answer_pattern_letter_on_next_line(self, task):
        output = LMOutput(text="ANSWER:\nC")
        assert task.extract_answer(output) == "C"

    def test_answer_pattern_without_colon(self, task):
        output = LMOutput(text="Reasoning about the options.\n\n**Answer** C")
        assert task.extract_answer(output) == "C"

    def test_answer_pattern_without_colon_ignores_prose(self, task):
        output = LMOutput(text="To answer A good question needs care.")
        assert task.extract_answer(output) is None

    def test_last_answer_pattern_wins(self, task):
        output = LMOutput(text="ANSWER: A\nWait, actually ANSWER: C")
        assert task.extract_answer(output) == "C"

    def test_no_answer_pattern_returns_none(self, task):
        output = LMOutput(text="I think the answer is B")
        assert task.extract_answer(output) is None

    def test_empty_output(self, task):
        output = LMOutput(text="")
        assert task.extract_answer(output) is None

    def test_whitespace_only(self, task):
        output = LMOutput(text="   ")
        assert task.extract_answer(output) is None


class TestFormatRequest:
    """Tests for format_request delegation."""

    def test_chat_format(self):
        task = get_task("lab_bench_litqa2")
        instance = Instance(
            question="What enzyme?",
            gold_answer="A",
            choices=("Kinase A", "Kinase B"),
            metadata={"gold_idx": 0, "gold_text": "Kinase A"},
        )
        request = task.format_request(instance)
        assert request.request_type == RequestType.CHAT
        assert any("Kinase A" in m["content"] for m in request.messages if m["role"] == "user")

    def test_mc_format(self):
        task = get_task("lab_bench_litqa2:mc")
        instance = Instance(
            question="What enzyme?",
            gold_answer="A",
            choices=("Kinase A", "Kinase B"),
            metadata={"gold_idx": 0, "gold_text": "Kinase A"},
        )
        request = task.format_request(instance)
        assert request.request_type == RequestType.LOGLIKELIHOOD
        assert request.continuations is not None
        assert len(request.continuations) == 2


class TestPrecisionMetric:
    """Tests for PrecisionMetric (accuracy excluding refusals)."""

    @pytest.fixture
    def metric(self):
        return PrecisionMetric()

    def _make_response(
        self, gold: str, extracted: str, score: float, refuse_idx: int = 2
    ) -> Response:
        return Response(
            instance=Instance(
                question="Q?",
                gold_answer=gold,
                choices=("A", "B", "C"),
                metadata={"gold_idx": 0, "gold_text": "Answer", "refuse_idx": refuse_idx},
            ),
            request=LMRequest(request_type=RequestType.CHAT, messages=()),
            outputs=[LMOutput(text="", extracted_answer=extracted)],
            scores={MultipleChoiceScorer().name: score},
        )

    def test_all_committed(self, metric):
        """No refusals: precision == accuracy."""
        responses = [
            self._make_response("A", "A", 1.0),
            self._make_response("A", "B", 0.0),
        ]
        assert metric.compute(responses) == pytest.approx(0.5)

    def test_refusal_excluded(self, metric):
        """Refused response excluded from denominator."""
        responses = [
            self._make_response("A", "A", 1.0),  # correct
            self._make_response("A", "C", 0.0),  # refused (chose C = refuse option)
        ]
        # Only 1 committed response, and it's correct
        assert metric.compute(responses) == pytest.approx(1.0)

    def test_all_refused(self, metric):
        """All responses refused: returns 0.0."""
        responses = [
            self._make_response("A", "C", 0.0),
            self._make_response("A", "C", 0.0),
        ]
        assert metric.compute(responses) == pytest.approx(0.0)

    def test_empty_responses(self, metric):
        assert metric.compute([]) == pytest.approx(0.0)


class TestCoverageMetric:
    """Tests for CoverageMetric (fraction of non-refused responses)."""

    @pytest.fixture
    def metric(self):
        return CoverageMetric()

    def _make_response(
        self, gold: str, extracted: str, score: float, refuse_idx: int = 2
    ) -> Response:
        return Response(
            instance=Instance(
                question="Q?",
                gold_answer=gold,
                choices=("A", "B", "C"),
                metadata={"gold_idx": 0, "gold_text": "Answer", "refuse_idx": refuse_idx},
            ),
            request=LMRequest(request_type=RequestType.CHAT, messages=()),
            outputs=[LMOutput(text="", extracted_answer=extracted)],
            scores={MultipleChoiceScorer().name: score},
        )

    def test_all_committed(self, metric):
        responses = [
            self._make_response("A", "A", 1.0),
            self._make_response("A", "B", 0.0),
        ]
        assert metric.compute(responses) == pytest.approx(1.0)

    def test_some_refused(self, metric):
        responses = [
            self._make_response("A", "A", 1.0),  # committed
            self._make_response("A", "C", 0.0),  # refused
        ]
        assert metric.compute(responses) == pytest.approx(0.5)

    def test_all_refused(self, metric):
        responses = [
            self._make_response("A", "C", 0.0),
            self._make_response("A", "C", 0.0),
        ]
        assert metric.compute(responses) == pytest.approx(0.0)

    def test_empty_responses(self, metric):
        assert metric.compute([]) == pytest.approx(0.0)
