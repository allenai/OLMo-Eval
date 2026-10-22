"""Tests for the SciReas-Pro tasks."""

import asyncio
import json
from collections import Counter

import pytest

from olmo_eval.common.types import LMOutput, RequestType, Response
from olmo_eval.evals.tasks import scireas_pro
from olmo_eval.evals.tasks.common import get_task, task_exists
from olmo_eval.evals.tasks.frontierscience import FRONTIERSCIENCE_DEFAULT_JUDGE_MODEL
from olmo_eval.evals.tasks.scireas_pro import (
    FAST500_SEED,
    FAST500_SIZE,
    MCQ_RULES,
    OLYMPIADBENCH_REVISION,
    SCIREAS_PRO_REPO,
    SCIREAS_PRO_REVISION,
    SOURCES,
    UGPHYSICS_REVISION,
    allocate_proportional,
    choice_matches,
    extract_choice,
    fast500_ids,
    olympiad_correct,
    olympiad_precision,
    parse_ugphysics_verdict,
    scibench_correct,
    source_of,
    stratified_sample_ids,
    ugphysics_extract,
    ugphysics_judge_prompt,
    ugphysics_judge_template,
)
from olmo_eval.runners.processing.utils import compute_task_hash
from tests.evals.tasks.judge_stubs import scripted_judge, use_judge

pytestmark = pytest.mark.usefixtures("judge_test_env")

UGPHYSICS_INSTRUCTION = (
    "The following is an open-ended problem from Atomic Physics of the undergraduate-level "
    'Physics. Please end your solution with "So the final answer is \\boxed{answer}(unit)."'
)
UGPHYSICS_PROBLEM = "A light source emits 0.6 W at 600 nm.\n\nHow many photons per second?"

TRUE_REPLY = (
    "## Equivalence Judgement\nTRUE\n\n## Justification\nSame value.\n\n=== report over ==="
)
FALSE_REPLY = (
    "## Equivalence Judgement\nFALSE\n\n## Justification\nDifferent.\n\n=== report over ==="
)


def _doc(taskname: str, answer: str, doc_id: int = 3, **extra) -> dict:
    return {
        "question": f"Question for {taskname}",
        "answer": answer,
        "taskname": taskname,
        "doc_id": doc_id,
        **extra,
    }


def _ugphysics_doc(answer: str = "\\boxed{2 \\times 10^{18}}") -> dict:
    return {
        "question": f"{UGPHYSICS_INSTRUCTION}\n\n{UGPHYSICS_PROBLEM}",
        "answer": answer,
        "taskname": "ugphysics_AtomicPhysics",
        "doc_id": 7,
        "reference_problem": UGPHYSICS_PROBLEM,
        "reference_solution": "n = P lambda / (h c) = 1.8e18 per second.",
    }


def _response(task, doc: dict, text: str, **metadata) -> Response:
    instance = task.process_doc(doc)
    assert instance is not None
    return Response(
        instance=instance,
        request=task.format_request(instance),
        outputs=[LMOutput(text=text, metadata=dict(metadata))],
    )


def _score(task, *responses: Response) -> list[Response]:
    return list(asyncio.run(task.score_responses(list(responses))))


@pytest.fixture
def task():
    return get_task("scireas_pro")


class TestRegistration:
    def test_both_tasks_are_registered_with_the_pinned_dataset(self):
        for name in ("scireas_pro", "scireas_pro:fast500"):
            assert task_exists(name)
            source = get_task(name).config.data_source
            assert source.path == SCIREAS_PRO_REPO
            assert source.revision == SCIREAS_PRO_REVISION

    def test_defaults_follow_the_reference_sampling_with_no_token_cap(self, task):
        params = task.config.sampling_params
        assert params.temperature == 0.6
        assert params.max_tokens is None
        assert task.config.primary_metric.name == "accuracy"

    def test_judge_defaults_to_the_frontierscience_judge(self, task):
        assert task.config.judge_model == FRONTIERSCIENCE_DEFAULT_JUDGE_MODEL
        assert task.config.judge_reasoning_effort == "high"

    def test_auxiliary_pins_reach_the_task_hash(self, task):
        serialized = json.dumps(task.config.to_dict())
        assert OLYMPIADBENCH_REVISION in serialized
        assert UGPHYSICS_REVISION in serialized

    def test_fast_sample_hashes_apart_from_the_full_set(self, task):
        fast = get_task("scireas_pro:fast500")
        assert compute_task_hash(task.config.to_dict()) != compute_task_hash(fast.config.to_dict())

    def test_request_is_the_released_prompt_verbatim(self, task):
        instance = task.process_doc(_doc("gpqa", "A"))
        request = task.format_request(instance)
        assert request.request_type == RequestType.CHAT
        assert request.messages == ({"role": "user", "content": "Question for gpqa"},)


class TestSources:
    @pytest.mark.parametrize(
        ("taskname", "source"),
        [
            ("gpqa", "gpqa"),
            ("supergpqa_Power Engineering and Engineering Thermophysics:cot", "supergpqa"),
            ("lab_bench_SeqQA:cot", "lab_bench"),
            ("mmlu_pro_computer science:cot", "mmlu_pro"),
            ("olympiadbench_OE_TO_physics_en_COMP", "olympiadbench"),
            ("scieval_chemistry_knowledge_application", "scieval"),
            ("ugphysics_Solid-StatePhysics", "ugphysics"),
            ("scibench", "scibench"),
        ],
    )
    def test_source_of(self, taskname, source):
        assert source_of(taskname) == source

    def test_unknown_source_raises(self):
        with pytest.raises(ValueError, match="unknown"):
            source_of("arc_challenge")

    def test_instance_id_is_task_and_document(self, task):
        instance = task.process_doc(_doc("lab_bench_DbQA:cot", "D", doc_id=12))
        assert instance.metadata["id"] == "lab_bench_DbQA:cot:12"
        assert instance.metadata["source"] == "lab_bench"


class TestChoiceExtraction:
    def test_gpqa_reads_the_answer_line(self):
        rule = MCQ_RULES["gpqa"]
        assert extract_choice("Reasoning (A) is wrong.\nANSWER: C", rule) == "C"

    def test_gpqa_falls_back_to_a_parenthesized_letter(self):
        rule = MCQ_RULES["gpqa"]
        assert extract_choice("I pick option (B) here.", rule) == "B"

    def test_mmlu_pro_style_prefers_the_format_then_answer_is(self):
        rule = MCQ_RULES["mmlu_pro"]
        assert extract_choice("so the answer is (D).", rule) == "D"
        assert extract_choice("Steps.\nAnswer: J", rule) == "J"

    def test_supergpqa_reads_a_parenthesized_answer(self):
        assert extract_choice("ANSWER: (E)", MCQ_RULES["supergpqa"]) == "E"
        assert extract_choice("ANSWER: E", MCQ_RULES["supergpqa"]) == "E"

    def test_scieval_drops_emphasis_and_option_text(self):
        rule = MCQ_RULES["scieval"]
        assert extract_choice("**Answer: B. Hydrochlorothiazide**", rule) == "B"

    def test_scieval_molecule_answers_match_exactly_ignoring_case(self):
        rule = MCQ_RULES["scieval"]
        prediction = extract_choice("Answer: Brc1ccc2ncccc2c1", rule)
        assert choice_matches(prediction, "BRC1CCC2NCCCC2C1", rule)
        assert not choice_matches(prediction, "Ic1ccc2ncccc2c1", rule)

    def test_letters_ignore_case_and_punctuation(self):
        assert choice_matches("c.", "C", MCQ_RULES["lab_bench"])


class TestFreeFormScoring:
    def test_scibench_accepts_five_percent_relative_error(self):
        assert scibench_correct("Therefore, the final answer is: $\\boxed{-1.52}$", "-1.5")[0]
        assert not scibench_correct("Therefore, the final answer is: $\\boxed{-1.7}$", "-1.5")[0]

    def test_scibench_keeps_the_right_side_of_a_boxed_equation(self):
        correct, answer = scibench_correct("$\\boxed{W = 3857}$", "3857")
        assert correct
        assert answer == "3857"

    def test_olympiad_precision_follows_the_source_error_field(self):
        assert olympiad_precision("Numerical", None) == 1e-8
        assert olympiad_precision("Numerical", "1e-1") == 0.1
        assert olympiad_precision("Numerical", "1e-1,") == [0.1, 1e-8]
        assert olympiad_precision("Tuple", "1e-1") == 1e-8

    def test_olympiad_uses_the_final_answer_line_and_tolerance(self):
        text = "Work.\nTherefore, the final answer is: $\\boxed{0.033}$."
        assert olympiad_correct(text, "0.0320", 1e-2)[0]
        assert not olympiad_correct(text, "0.0320", 1e-4)[0]

    def test_olympiad_compares_expressions(self):
        text = "Therefore, the final answer is: $\\boxed{n^2-n+1}$"
        assert olympiad_correct(text, "$n^{2}-n+1$", 1e-8)[0]

    def test_ugphysics_extracts_the_last_boxed_answer(self):
        assert ugphysics_extract("So the final answer is \\boxed{0.07, 574}(K).") == "0.07, 574"
        assert ugphysics_extract("The answer is 42") == "42"
        assert ugphysics_extract("No conclusion reached.") == ""


class TestStrictScoring:
    def test_a_closed_trace_is_dropped_before_extraction(self, task):
        response = _response(task, _doc("gpqa", "B"), "<think>ANSWER: A</think>\nANSWER: B")
        [scored] = _score(task, response)
        assert scored.scores["accuracy"] == 1.0
        assert scored.scores["unclosed_think"] == 0.0

    def test_an_unclosed_trace_scores_zero(self, task):
        response = _response(task, _doc("gpqa", "B"), "<think>so ANSWER: B")
        [scored] = _score(task, response)
        assert scored.scores["accuracy"] == 0.0
        assert scored.scores["unclosed_think"] == 1.0
        assert scored.outputs[0].metadata["judge_result"]["method"] == "unfinished_reasoning"

    def test_a_trace_opened_by_the_template_that_hits_the_limit_scores_zero(self, task):
        response = _response(task, _doc("gpqa", "B"), "loop ANSWER: B", finish_reason="length")
        [scored] = _score(task, response)
        assert scored.scores["accuracy"] == 0.0

    def test_output_without_reasoning_that_finished_is_scored(self, task):
        response = _response(task, _doc("gpqa", "B"), "ANSWER: B", finish_reason="stop")
        [scored] = _score(task, response)
        assert scored.scores["accuracy"] == 1.0
        assert scored.scores["unclosed_think"] == 1.0

    def test_original_text_decides_closure_after_harness_stripping(self, task):
        response = _response(
            task,
            _doc("gpqa", "B"),
            "ANSWER: B",
            original_text="reasoning</think>ANSWER: B",
            finish_reason="length",
        )
        [scored] = _score(task, response)
        assert scored.scores["accuracy"] == 1.0


class TestUGPhysicsJudge:
    def test_rule_match_needs_no_judge(self, task, monkeypatch):
        judge, prompts = scripted_judge()
        use_judge(monkeypatch, judge)
        response = _response(
            task, _ugphysics_doc(), "</think>So the answer is \\boxed{2\\times10^{18}}."
        )
        [scored] = _score(task, response)
        assert scored.scores["accuracy"] == 1.0
        assert scored.scores["judge_called"] == 0.0
        assert prompts == []

    def test_rule_miss_goes_to_the_judge_with_the_reference_solution(self, task, monkeypatch):
        judge, prompts = scripted_judge(TRUE_REPLY)
        use_judge(monkeypatch, judge)
        response = _response(
            task, _ugphysics_doc(), "</think>So the final answer is \\boxed{1.8 \\times 10^{18}}."
        )
        [scored] = _score(task, response)
        assert scored.scores["accuracy"] == 1.0
        assert scored.scores["judge_called"] == 1.0
        [prompt] = prompts
        assert UGPHYSICS_PROBLEM in prompt
        assert "n = P lambda / (h c)" in prompt
        assert "**Student Answer(s)**:\n1.8 \\times 10^{18}" in prompt

    def test_judge_false_scores_zero(self, task, monkeypatch):
        judge, _ = scripted_judge(FALSE_REPLY)
        use_judge(monkeypatch, judge)
        response = _response(task, _ugphysics_doc(), "</think>\\boxed{5}")
        [scored] = _score(task, response)
        assert scored.scores["accuracy"] == 0.0

    def test_no_stated_answer_is_wrong_without_the_judge(self, task, monkeypatch):
        judge, prompts = scripted_judge()
        use_judge(monkeypatch, judge)
        response = _response(task, _ugphysics_doc(), "</think>I could not finish.")
        [scored] = _score(task, response)
        assert scored.scores["accuracy"] == 0.0
        assert prompts == []

    def test_unparseable_judge_replies_leave_the_item_unscored(self, task, monkeypatch):
        judge, _ = scripted_judge("no verdict", "still none", "nothing")
        use_judge(monkeypatch, judge)
        response = _response(task, _ugphysics_doc(), "</think>\\boxed{5}")
        with pytest.raises(Exception, match="no parseable verdict|judging failed"):
            _score(task, response)

    def test_verdict_parsing(self):
        assert parse_ugphysics_verdict(TRUE_REPLY) is True
        assert parse_ugphysics_verdict(FALSE_REPLY) is False
        assert parse_ugphysics_verdict("## Equivalence Judgement\n**TRUE**") is True
        assert parse_ugphysics_verdict("TRUE") is None

    def test_prompt_template_is_verbatim_and_filled(self):
        template = ugphysics_judge_template()
        assert template.count("{problem}") == 1
        prompt = ugphysics_judge_prompt("P", "S", "1, 2", "resp", "1,2")
        assert "{{RS}}" not in prompt
        assert "**Reference Answer(s)**:\n1, 2" in prompt


class TestMetrics:
    def test_source_slices_macro_average_and_judge_rate(self, task, monkeypatch):
        judge, _ = scripted_judge(FALSE_REPLY)
        use_judge(monkeypatch, judge)
        responses = _score(
            task,
            _response(task, _doc("gpqa", "A"), "</think>ANSWER: A"),
            _response(task, _doc("gpqa", "A", doc_id=4), "</think>ANSWER: B"),
            _response(task, _doc("scibench", "2.26"), "</think>$\\boxed{2.26}$"),
            _response(task, _ugphysics_doc(), "</think>\\boxed{7}"),
        )
        metrics = {metric.name: metric for metric in task.config.metrics}
        assert metrics["accuracy"].compute(responses) == pytest.approx(0.5)
        assert metrics["accuracy_gpqa"].compute(responses) == pytest.approx(0.5)
        assert metrics["accuracy_scibench"].compute(responses) == 1.0
        assert metrics["macro_accuracy"].compute(responses) == pytest.approx(0.5)
        assert metrics["judge_call_rate"].compute(responses) == pytest.approx(0.25)
        assert not metrics["unclosed_think_rate"].higher_is_better()


class TestFast500:
    def test_allocation_is_proportional_and_exact(self):
        allocation = allocate_proportional({"a": 50, "b": 30, "c": 20}, 7)
        assert sum(allocation.values()) == 7
        assert allocation == {"a": 4, "b": 2, "c": 1}

    def test_sampling_is_deterministic_and_stratified(self):
        items = [("a", i) for i in range(60)] + [("b", i) for i in range(40)]
        first = stratified_sample_ids(items, 10, seed=1)
        assert first == stratified_sample_ids(items, 10, seed=1)
        assert Counter(sample.split(":")[0] for sample in first) == {"a": 6, "b": 4}

    def test_shipped_sample_matches_its_recorded_procedure(self):
        resource = scireas_pro.files(scireas_pro.__package__).joinpath(scireas_pro.FAST500_FILE)
        recorded = json.loads(resource.read_text(encoding="utf-8"))
        assert recorded["revision"] == SCIREAS_PRO_REVISION
        assert recorded["seed"] == FAST500_SEED
        population = {name: counts[0] for name, counts in recorded["strata"].items()}
        assert sum(population.values()) == scireas_pro.SCIREAS_PRO_SIZE
        expected = allocate_proportional(population, FAST500_SIZE)
        drawn = Counter(sample_id.rsplit(":", 1)[0] for sample_id in fast500_ids())
        assert {name: drawn.get(name, 0) for name in population} == expected
        assert {source_of(name) for name in drawn} == set(SOURCES)

    def test_fast_task_keeps_the_listed_items_in_order(self, monkeypatch):
        ids = fast500_ids()
        rows = [
            {"question": "q", "answer": "A", "taskname": name, "doc_id": int(doc)}
            for name, doc in (sample_id.rsplit(":", 1) for sample_id in reversed(ids))
        ]
        task = get_task("scireas_pro:fast500")
        assert [task.process_doc(row).metadata["id"] for row in task._select(rows)] == list(ids)


class TestReferenceJoin:
    def _patch_sources(self, monkeypatch, olympiad_rows, ugphysics_rows):
        monkeypatch.setattr(scireas_pro, "_load_olympiadbench", lambda names: olympiad_rows)
        by_problem = {row["problem"].strip(): row for row in ugphysics_rows}
        monkeypatch.setattr(scireas_pro, "_load_ugphysics", lambda subjects: by_problem)

    def test_rows_get_their_tolerance_and_reference_solution(self, monkeypatch):
        self._patch_sources(
            monkeypatch,
            {
                "olympiadbench_OE_TO_physics_en_COMP": [
                    {"final_answer": ["1"], "answer_type": "Numerical", "error": None},
                    {"final_answer": ["0.0320"], "answer_type": "Numerical", "error": "1e-3"},
                ]
            },
            [{"problem": UGPHYSICS_PROBLEM, "solution": "S", "answers": "\\boxed{2}"}],
        )
        rows = [
            {
                "taskname": "olympiadbench_OE_TO_physics_en_COMP",
                "doc_id": 1,
                "answer": '["0.0320"]',
                "question": "q",
            },
            {**_ugphysics_doc("\\boxed{2}"), "reference_problem": None},
        ]
        scireas_pro.attach_references(rows)
        assert rows[0]["error"] == "1e-3"
        assert rows[1]["reference_problem"] == UGPHYSICS_PROBLEM
        assert rows[1]["reference_solution"] == "S"

    def test_a_row_that_does_not_join_raises(self, monkeypatch):
        self._patch_sources(monkeypatch, {}, [{"problem": "other", "answers": "\\boxed{2}"}])
        with pytest.raises(ValueError, match="no UGPhysics reference"):
            scireas_pro.attach_references([_ugphysics_doc("\\boxed{2}")])
