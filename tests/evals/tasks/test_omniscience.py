"""Tests for the AA-Omniscience task."""

import inspect
import logging
from dataclasses import replace

import pytest

from olmo_eval.common.execution import ScoringContext
from olmo_eval.common.scorers.llm_judge import build_openai_judge_fn
from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, Response, Split
from olmo_eval.evals.tasks import omniscience
from olmo_eval.evals.tasks.common import TaskConfig, get_task, task_exists
from olmo_eval.evals.tasks.omniscience import (
    DOMAINS,
    SYSTEM_PROMPT,
    HallucinationRateMetric,
    OmniscienceAccuracyMetric,
    OmniscienceIndexMetric,
    OmniscienceScorer,
)
from olmo_eval.runners.asynq.preparation import prepare_task_items
from olmo_eval.runners.io.builders import build_predictions, build_requests_from_responses
from olmo_eval.runners.processing.utils import compute_task_hash


async def named_judge(prompt: str) -> str:
    return "A"


@pytest.fixture
def task():
    return get_task("omniscience:judge")


def _doc(
    question: str = "Which ASC paragraph defines a series of distinct goods?",
    answer: str = "ASC 606-10-25-15",
    domain: str = "Finance",
    topic: str = "Accounting",
    subtopic: str = "Revenue Recognition",
    question_id: int = 7,
) -> dict:
    return {
        "question_id": question_id,
        "domain": domain,
        "topic": topic,
        "subtopic": subtopic,
        "question": question,
        "answer": answer,
    }


def _response(task, text: str = "ASC 606-10-25-15", domain: str = "Finance") -> Response:
    instance = task.process_doc(_doc(domain=domain))
    assert instance is not None
    return Response(
        instance=instance,
        request=task.format_request(instance),
        outputs=[LMOutput(text=text)],
    )


def _graded(grade: str | None, domain: str = "Finance") -> Response:
    """A response whose judge grade has already been recorded."""
    metadata: dict = {"domain": domain}
    if grade is not None:
        metadata["judge_result"] = {"grade": grade, "empty_response": False}
    return Response(
        instance=Instance(question="q", gold_answer="g", metadata=metadata),
        request=LMRequest(request_type=RequestType.CHAT, messages=()),
        outputs=[LMOutput(text="a")],
    )


def _replies_by_answer(replies: dict[str, str]):
    """A judge that grades by predicted answer and records every prompt it receives."""
    calls: list[str] = []

    async def judge(prompt: str) -> str:
        calls.append(prompt)
        for answer, reply in replies.items():
            if f"Predicted answer: {answer}\n" in prompt:
                return reply
        raise RuntimeError("judge unavailable")

    return judge, calls


@pytest.fixture
def patch_judge(monkeypatch):
    """Swap the judge callable on the scorer shared by the registered metrics."""

    def apply(judge) -> None:
        monkeypatch.setitem(omniscience._JUDGE_SCORER.__dict__, "judge_fn", judge)

    return apply


class TestRegistration:
    def test_base_task_and_judge_variant_are_registered(self):
        assert task_exists("omniscience")
        base = get_task("omniscience")
        judge = get_task("omniscience:judge")
        assert base.config.data_source.path == "ArtificialAnalysis/AA-Omniscience-Public"
        assert judge.config.data_source.path == base.config.data_source.path

    def test_uses_the_only_published_split(self, task):
        assert task.config.split == Split.TRAIN
        assert task.config.get_data_source().split == "train"

    def test_sampling_settings(self, task):
        params = task.config.sampling_params
        assert params.max_tokens is None
        assert params.temperature == 0.6
        assert params.top_p == 0.95

    def test_is_a_chat_task(self, task):
        assert task.request_type == RequestType.CHAT

    def test_judge_metrics(self, task):
        assert {metric.name for metric in task.config.metrics} == {
            "any__any__omniscience_index",
            "any__any__hallucination_rate",
            "any__any__accuracy",
            "judge_parsing_errors",
            *(f"domain__{domain}__accuracy" for domain in DOMAINS),
        }
        assert task.config.get_primary_metric().name == "any__any__omniscience_index"

    def test_domain_metrics_are_accuracy_percentages(self, task):
        domain_metrics = [m for m in task.config.metrics if m.name.startswith("domain__")]

        assert domain_metrics
        for metric in domain_metrics:
            assert isinstance(metric, OmniscienceAccuracyMetric)
            assert metric.display_format() == "percentage"

    def test_every_metric_shares_one_judge(self, task):
        scorers = {id(metric.scorer()) for metric in task.config.metrics}
        assert scorers == {id(omniscience._JUDGE_SCORER)}

    def test_declares_the_judge_api_key(self, task):
        assert task.config.required_secrets == ("OPENAI_API_KEY",)

    def test_task_hash_tracks_the_judge_model(self, task):
        def with_judge(model: str) -> TaskConfig:
            judge = OmniscienceScorer(
                judge_fn=build_openai_judge_fn(model=model, scorer_name="OmniscienceScorer")
            )
            return replace(
                task.config,
                metrics=tuple(replace(metric, scorer=judge) for metric in task.config.metrics),
            )

        default_model = inspect.signature(build_openai_judge_fn).parameters["model"].default
        default_hash = compute_task_hash(task.config.to_dict())

        assert compute_task_hash(with_judge(default_model).to_dict()) == default_hash
        assert compute_task_hash(with_judge("gpt-5.5").to_dict()) != default_hash

    @pytest.mark.parametrize(
        ("override", "setting", "value"),
        [
            ({"judge_model": "gpt-5.5"}, "model", "gpt-5.5"),
            ({"judge_reasoning_effort": "high"}, "reasoning_effort", "high"),
            ({"judge_max_tokens": 64}, "max_tokens", 64),
        ],
        ids=["model", "reasoning-effort", "max-tokens"],
    )
    def test_a_judge_override_sets_the_judge_that_grades(self, override, setting, value):
        task = get_task("omniscience:judge", override)

        judge = task._get_scorers()["omniscience_judge"]
        assert {id(metric.scorer()) for metric in task.config.metrics} == {id(judge)}
        assert task.config.get_primary_metric().scorer() is judge
        assert judge.judge_fn.__dict__["judge_config"] == {
            **build_openai_judge_fn().__dict__["judge_config"],
            setting: value,
        }

    def test_task_hash_tracks_a_judge_override(self, task):
        overridden = get_task("omniscience:judge", {"judge_model": "gpt-5.5"}).config.to_dict()

        assert compute_task_hash(overridden) != compute_task_hash(task.config.to_dict())
        assert {metric["scorer"]["judge_fn"]["model"] for metric in overridden["metrics"]} == {
            "gpt-5.5"
        }
        assert overridden["primary_metric"]["scorer"]["judge_fn"]["model"] == "gpt-5.5"


class TestProcessDoc:
    def test_maps_the_dataset_schema(self, task):
        instance = task.process_doc(_doc(), index=3)

        assert instance is not None
        assert instance.question == "Which ASC paragraph defines a series of distinct goods?"
        assert instance.gold_answer == "ASC 606-10-25-15"
        assert instance.metadata == {
            "id": 7,
            "domain": "Finance",
            "topic": "Accounting",
            "subtopic": "Revenue Recognition",
        }

    def test_uses_the_dataset_question_id_not_the_row_index(self, task):
        instance = task.process_doc(_doc(question_id=42), index=0)

        assert instance is not None
        assert instance.metadata["id"] == 42


class TestPrompts:
    def test_system_prompt_names_the_domain_and_subtopic(self, task):
        instance = task.process_doc(_doc())
        assert instance is not None
        request = task.format_request(instance)

        expected = SYSTEM_PROMPT.format(topic="Finance", category="Revenue Recognition")
        assert request.system_prompt == expected
        assert expected.startswith(
            "You are answering questions about Finance, and in particular Revenue Recognition."
        )
        assert request.messages[0] == {"role": "system", "content": expected}

    def test_question_is_sent_verbatim_as_the_only_user_turn(self, task):
        instance = task.process_doc(_doc())
        assert instance is not None
        request = task.format_request(instance)

        user_messages = [m for m in request.messages if m["role"] == "user"]
        assert user_messages == [{"role": "user", "content": instance.question}]

    def test_system_prompt_is_rebuilt_per_instance(self, task):
        first = task.process_doc(_doc(domain="Finance", subtopic="Revenue Recognition"))
        second = task.process_doc(_doc(domain="Law", subtopic="US Contract Law"))
        assert first is not None and second is not None

        assert "Finance" in task.format_request(first).system_prompt
        assert "US Contract Law" in task.format_request(second).system_prompt
        assert "Finance" not in task.format_request(second).system_prompt

    def test_judge_prompt_includes_question_gold_and_answer(self):
        instance = Instance(question="Who proposed fake barns?", gold_answer="Carl Ginet")
        output = LMOutput(text="raw", extracted_answer="  Carl Ginet \n")

        prompt = OmniscienceScorer(judge_fn=named_judge).format_judge_prompt(instance, output)

        assert "Question: Who proposed fake barns?\n" in prompt
        assert "Gold target: Carl Ginet\n" in prompt
        assert "Predicted answer: Carl Ginet\n" in prompt
        assert prompt.rstrip().endswith("with no text around it.")

    def test_judge_prompt_falls_back_to_the_full_output(self):
        """Without an extracted answer, the judge looks for one in the full output."""
        instance = Instance(question="q", gold_answer="g")
        output = LMOutput(text="reasoning ... so 42\n")

        prompt = OmniscienceScorer(judge_fn=named_judge).format_judge_prompt(instance, output)

        assert "Predicted answer: reasoning ... so 42\n" in prompt

    def test_judge_prompt_keeps_braces_in_substituted_values(self):
        instance = Instance(question="What does {x} mean?", gold_answer="{placeholder}")
        output = LMOutput(text="", extracted_answer="f'{value:.2f}'")

        prompt = OmniscienceScorer(judge_fn=named_judge).format_judge_prompt(instance, output)

        assert "Question: What does {x} mean?" in prompt
        assert "Gold target: {placeholder}" in prompt
        assert "Predicted answer: f'{value:.2f}'" in prompt


class TestFinalAnswer:
    @pytest.mark.parametrize(
        "text",
        [
            "<think>maybe 606-10-25-14</think>\n\nASC 606-10-25-15",
            "maybe 606-10-25-14</think>\n\nASC 606-10-25-15",
            "ASC 606-10-25-15",
        ],
        ids=["closed-trace", "opening-tag-in-prompt", "no-trace"],
    )
    def test_answers_are_found(self, task, text):
        output = LMOutput(text=text)
        output.extracted_answer = task.extract_answer(output)

        assert OmniscienceScorer(judge_fn=named_judge).final_answer(output) == "ASC 606-10-25-15"

    @pytest.mark.parametrize(
        "text",
        [
            "<think>maybe 606-10-25-14, or maybe",
            "<think>maybe 606-10-25-14</think>",
            "<think>maybe</think> ASC 606-10-25-15 <think>or maybe 606-10-25-14",
            "maybe 606-10-25-14</think>\n\n  ",
            "",
            " \n ",
        ],
        ids=[
            "unclosed-trace",
            "closed-trace-no-answer",
            "reopened-trace",
            "whitespace-answer",
            "empty",
            "blank",
        ],
    )
    def test_no_final_answer(self, task, text):
        output = LMOutput(text=text)
        output.extracted_answer = task.extract_answer(output)

        assert OmniscienceScorer(judge_fn=named_judge).final_answer(output) is None

    def test_empty_extraction_is_no_answer_even_with_output_text(self):
        output = LMOutput(text="<think>maybe 606-10-25-14", extracted_answer="")

        assert OmniscienceScorer(judge_fn=named_judge).final_answer(output) is None

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("<think>maybe 606-10-25-14</think>\n\nASC 606-10-25-15", "ASC 606-10-25-15"),
            ("maybe 606-10-25-14</think>\n\nASC 606-10-25-15", "ASC 606-10-25-15"),
            ("<think>maybe 606-10-25-14</think>", None),
            ("<think>maybe</think> ASC 606-10-25-15 <think>or maybe 606-10-25-14", None),
        ],
        ids=["closed-trace", "opening-tag-in-prompt", "closed-trace-no-answer", "reopened-trace"],
    )
    def test_stripping_thinking_first_gives_the_same_answer(self, task, text, expected):
        stripping = type(task)(replace(task.config, strip_thinking=True))
        response = _response(stripping, text=text)
        stripping.strip_thinking_traces([response])
        output = response.outputs[0]
        output.extracted_answer = stripping.extract_answer(output)

        assert "original_text" in output.metadata
        assert OmniscienceScorer(judge_fn=named_judge).final_answer(output) == expected


class TestJudgeParsing:
    @pytest.mark.parametrize(
        ("raw", "grade"),
        [
            ("A", "CORRECT"),
            ("B", "INCORRECT"),
            ("C", "PARTIAL_ANSWER"),
            ("D", "NOT_ATTEMPTED"),
            ("  b \n", "INCORRECT"),
            ("a", "CORRECT"),
            ("A: CORRECT", "CORRECT"),
            ("B: INCORRECT", "INCORRECT"),
            ("C: PARTIAL ANSWER", "PARTIAL_ANSWER"),
            ("D: NOT ATTEMPTED", "NOT_ATTEMPTED"),
            ("CORRECT", "CORRECT"),
            ("INCORRECT", "INCORRECT"),
            ("PARTIAL ANSWER", "PARTIAL_ANSWER"),
            ("PARTIAL_ANSWER", "PARTIAL_ANSWER"),
            ("NOT ATTEMPTED", "NOT_ATTEMPTED"),
            ("not_attempted", "NOT_ATTEMPTED"),
            ("“A”", "CORRECT"),
            ('"B"', "INCORRECT"),
            ("**C**", "PARTIAL_ANSWER"),
            ("(D)", "NOT_ATTEMPTED"),
            ("A.", "CORRECT"),
            ("ANSWER: B", "INCORRECT"),
            ("Grade: D", "NOT_ATTEMPTED"),
            ("This is a correct answer", "CORRECT"),
        ],
    )
    def test_recognized_grades(self, raw, grade):
        instance = Instance(question="q", gold_answer="g")

        score = OmniscienceScorer(judge_fn=named_judge).parse_judge_response(raw, instance)

        assert instance.metadata["judge_result"]["grade"] == grade
        assert score == (1.0 if grade == "CORRECT" else 0.0)
        assert instance.metadata["is_parsing_error"] is False

    @pytest.mark.parametrize(
        "raw",
        [
            "",
            "   ",
            "garbage",
            "E",
            "I cannot grade this",
            "CANNOT GRADE",
            "not a grade",
            "correct or incorrect",
            "partial answer, not attempted",
        ],
    )
    def test_unparseable_replies_are_a_parsing_error(self, raw):
        instance = Instance(question="q", gold_answer="g")

        score = OmniscienceScorer(judge_fn=named_judge).parse_judge_response(raw, instance)

        assert score == 0.0
        assert instance.metadata["judge_result"]["grade"] == "PARSING_ERROR"
        assert instance.metadata["is_parsing_error"] is True

    @pytest.mark.parametrize("raw", ["  b \n", "garbage"])
    def test_raw_judge_reply_is_kept_verbatim(self, raw):
        instance = Instance(question="q", gold_answer="g")

        OmniscienceScorer(judge_fn=named_judge).parse_judge_response(raw, instance)

        assert instance.metadata["judge_raw_response"] == raw


class TestScorer:
    @pytest.mark.anyio
    async def test_sends_the_formatted_prompt_to_the_judge(self):
        judge, calls = _replies_by_answer({"pred": "A"})
        scorer = OmniscienceScorer(judge_fn=judge)
        instance = Instance(question="q?", gold_answer="gold")
        output = LMOutput(text="pred", extracted_answer="pred")

        score = await scorer.ascore_with_context(instance, output, ScoringContext())

        assert score == 1.0
        assert calls == [scorer.format_judge_prompt(instance, output)]
        assert instance.metadata["judge_result"] == {"grade": "CORRECT", "empty_response": False}
        assert instance.metadata["judge_raw_response"] == "A"

    @pytest.mark.anyio
    async def test_clears_a_stale_parsing_error_flag(self):
        judge, _ = _replies_by_answer({"a": "B"})
        instance = Instance(question="q", gold_answer="g", metadata={"is_parsing_error": True})

        await OmniscienceScorer(judge_fn=judge).ascore_with_context(
            instance, LMOutput(text="a"), ScoringContext()
        )

        assert instance.metadata["is_parsing_error"] is False

    @pytest.mark.anyio
    async def test_flags_and_reraises_a_judge_failure(self):
        judge, _ = _replies_by_answer({})
        instance = Instance(question="q", gold_answer="g", metadata={"judge_raw_response": "stale"})

        with pytest.raises(RuntimeError, match="judge unavailable"):
            await OmniscienceScorer(judge_fn=judge).ascore_with_context(
                instance, LMOutput(text="a"), ScoringContext()
            )

        assert instance.metadata["is_parsing_error"] is True
        assert instance.metadata["judge_result"]["grade"] == "PARSING_ERROR"
        assert instance.metadata["judge_raw_response"] is None

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        "text",
        ["<think>still reasoning about 42", "<think>it is 42</think>", "", " \n "],
        ids=["unclosed-trace", "closed-trace-no-answer", "empty", "blank"],
    )
    async def test_no_final_answer_is_not_attempted_without_a_judge_call(self, text):
        judge, calls = _replies_by_answer({})
        instance = Instance(
            question="q", gold_answer="42", metadata={"judge_raw_response": "stale"}
        )

        score = await OmniscienceScorer(judge_fn=judge).ascore_with_context(
            instance, LMOutput(text=text, extracted_answer=""), ScoringContext()
        )

        assert score == 0.0
        assert calls == []
        assert instance.metadata["judge_result"] == {
            "grade": "NOT_ATTEMPTED",
            "empty_response": True,
        }
        assert instance.metadata["judge_raw_response"] is None
        assert instance.metadata["is_parsing_error"] is False


class TestMetrics:
    """Metrics are derived from the recorded judge grades."""

    @pytest.fixture
    def responses(self):
        # Finance: 2 correct, 1 incorrect, 1 not attempted, 1 parse error
        # Law: 1 correct, 1 incorrect, 1 partial, 1 not attempted, 1 parse error
        return [
            _graded("CORRECT", "Finance"),
            _graded("CORRECT", "Finance"),
            _graded("INCORRECT", "Finance"),
            _graded("NOT_ATTEMPTED", "Finance"),
            _graded("PARSING_ERROR", "Finance"),
            _graded("CORRECT", "Law"),
            _graded("INCORRECT", "Law"),
            _graded("PARTIAL_ANSWER", "Law"),
            _graded("NOT_ATTEMPTED", "Law"),
            _graded("PARSING_ERROR", "Law"),
        ]

    def test_omniscience_index(self, responses):
        # 8 graded: (3 correct - 2 incorrect) / 8
        assert OmniscienceIndexMetric().compute(responses) == pytest.approx(12.5)

    def test_hallucination_rate(self, responses):
        # 2 incorrect out of the 5 graded answers that were not correct
        assert HallucinationRateMetric().compute(responses) == pytest.approx(0.4)

    def test_accuracy(self, responses):
        assert OmniscienceAccuracyMetric().compute(responses) == pytest.approx(3 / 8)

    def test_parsing_errors_do_not_change_any_metric(self, responses):
        graded_only = [
            r for r in responses if r.instance.metadata["judge_result"]["grade"] != "PARSING_ERROR"
        ]
        for metric in (
            OmniscienceIndexMetric(),
            HallucinationRateMetric(),
            OmniscienceAccuracyMetric(),
            OmniscienceAccuracyMetric(name="domain__Law__accuracy"),
        ):
            assert metric.compute(responses) == metric.compute(graded_only)

    def test_domain_accuracy_uses_only_its_domain(self, responses):
        finance = OmniscienceAccuracyMetric(name="domain__Finance__accuracy")
        law = OmniscienceAccuracyMetric(name="domain__Law__accuracy")

        assert finance.compute(responses) == pytest.approx(2 / 4)
        assert law.compute(responses) == pytest.approx(1 / 4)

    def test_index_and_hallucination_support_domain_names(self, responses):
        index = OmniscienceIndexMetric(name="domain__Finance__omniscience_index")
        rate = HallucinationRateMetric(name="domain__Law__hallucination_rate")

        assert index.compute(responses) == pytest.approx(100 * (2 - 1) / 4)
        assert rate.compute(responses) == pytest.approx(1 / 3)

    def test_index_spans_minus_100_to_100(self):
        all_correct = [_graded("CORRECT") for _ in range(3)]
        all_incorrect = [_graded("INCORRECT") for _ in range(3)]

        assert OmniscienceIndexMetric().compute(all_correct) == 100.0
        assert OmniscienceIndexMetric().compute(all_incorrect) == -100.0

    def test_abstaining_scores_zero_on_the_index_and_no_hallucinations(self):
        abstained = [_graded("NOT_ATTEMPTED"), _graded("PARTIAL_ANSWER")]

        assert OmniscienceIndexMetric().compute(abstained) == 0.0
        assert HallucinationRateMetric().compute(abstained) == 0.0

    def test_hallucination_rate_is_zero_when_everything_is_correct(self):
        assert HallucinationRateMetric().compute([_graded("CORRECT")]) == 0.0

    @pytest.mark.parametrize(
        "responses",
        [
            [],
            [_graded("PARSING_ERROR"), _graded("PARSING_ERROR")],
            [_graded(None)],
        ],
        ids=["empty", "only-parsing-errors", "ungraded"],
    )
    def test_nothing_graded_scores_zero(self, responses):
        assert OmniscienceIndexMetric().compute(responses) == 0.0
        assert HallucinationRateMetric().compute(responses) == 0.0
        assert OmniscienceAccuracyMetric().compute(responses) == 0.0

    def test_empty_domain_scores_zero(self, responses):
        health = OmniscienceAccuracyMetric(name="domain__Health__accuracy")
        assert health.compute(responses) == 0.0

    def test_empty_domain_warns_that_its_score_is_not_measured(self, responses, caplog):
        health = OmniscienceAccuracyMetric(name="domain__Health__accuracy")

        with caplog.at_level(logging.WARNING, logger=omniscience.__name__):
            health.compute(responses)

        assert [r.getMessage() for r in caplog.records if r.name == omniscience.__name__] == [
            "domain__Health__accuracy has no graded responses; "
            "its value of 0.0 is not a measured score."
        ]

    def test_nothing_graded_warns_for_every_metric(self, caplog):
        responses = [_graded("PARSING_ERROR"), _graded(None)]

        with caplog.at_level(logging.WARNING, logger=omniscience.__name__):
            for metric in (
                OmniscienceIndexMetric(),
                HallucinationRateMetric(),
                OmniscienceAccuracyMetric(),
            ):
                metric.compute(responses)

        warned = [
            r.getMessage().split()[0] for r in caplog.records if r.name == omniscience.__name__
        ]
        assert warned == [
            "any__any__omniscience_index",
            "any__any__hallucination_rate",
            "any__any__accuracy",
        ]

    def test_graded_subsets_do_not_warn(self, responses, caplog):
        with caplog.at_level(logging.WARNING, logger=omniscience.__name__):
            OmniscienceAccuracyMetric(name="domain__Law__accuracy").compute(responses)
            HallucinationRateMetric().compute([_graded("CORRECT")])

        assert not [r for r in caplog.records if r.name == omniscience.__name__]


class TestInstanceMetrics:
    @pytest.mark.parametrize(
        ("grade", "index", "hallucination", "accuracy"),
        [
            ("CORRECT", 1.0, None, 1.0),
            ("INCORRECT", -1.0, 1.0, 0.0),
            ("PARTIAL_ANSWER", 0.0, 0.0, 0.0),
            ("NOT_ATTEMPTED", 0.0, 0.0, 0.0),
            ("PARSING_ERROR", None, None, None),
            (None, None, None, None),
        ],
    )
    def test_per_instance_values(self, grade, index, hallucination, accuracy):
        response = _graded(grade)

        assert OmniscienceIndexMetric().compute_instance(response) == index
        assert HallucinationRateMetric().compute_instance(response) == hallucination
        assert OmniscienceAccuracyMetric().compute_instance(response) == accuracy

    def test_other_domains_are_excluded(self):
        response = _graded("CORRECT", domain="Law")
        finance = "domain__Finance"

        assert (
            OmniscienceAccuracyMetric(name=f"{finance}__accuracy").compute_instance(response)
            is None
        )
        assert (
            OmniscienceIndexMetric(name=f"{finance}__omniscience_index").compute_instance(response)
            is None
        )
        assert (
            HallucinationRateMetric(name=f"{finance}__hallucination_rate").compute_instance(
                _graded("INCORRECT", domain="Law")
            )
            is None
        )

    def test_subset_and_category_come_from_the_name(self):
        metric = OmniscienceAccuracyMetric(name="domain__Law__accuracy")

        assert (metric.subset, metric.category) == ("domain", "Law")

    def test_renaming_a_metric_moves_it_to_the_new_subset(self):
        metric = replace(OmniscienceAccuracyMetric(), name="domain__Law__accuracy")

        assert metric.compute_instance(_graded("CORRECT", domain="Finance")) is None
        assert metric.compute_instance(_graded("CORRECT", domain="Law")) == 1.0

    @pytest.mark.parametrize(
        "name", ["accuracy", "domain__Law", "domain__Law__accuracy__extra", "domain____accuracy"]
    )
    def test_malformed_names_are_rejected_at_construction(self, name):
        with pytest.raises(ValueError, match="<subset>__<category>__<metric>"):
            OmniscienceAccuracyMetric(name=name)

    def test_comparison_settings(self):
        for metric in (
            OmniscienceIndexMetric(),
            HallucinationRateMetric(),
            OmniscienceAccuracyMetric(),
        ):
            assert metric.supports_scorer_fallback() is False

        assert OmniscienceIndexMetric().higher_is_better() is True
        assert OmniscienceAccuracyMetric().higher_is_better() is True
        assert HallucinationRateMetric().higher_is_better() is False

    def test_display_formats(self):
        assert OmniscienceIndexMetric().display_format() == "raw"
        assert OmniscienceAccuracyMetric().display_format() == "percentage"
        assert HallucinationRateMetric().display_format() == "percentage"


class TestEndToEnd:
    @pytest.mark.anyio
    async def test_scores_and_aggregates_a_run(self, task, patch_judge):
        responses = [
            _response(task, text="<think>hmm</think>\n\nASC 606-10-25-15", domain="Finance"),
            _response(task, text="ASC 606-10-25-2", domain="Finance"),
            _response(task, text="I don't know.", domain="Law"),
            _response(task, text="Something", domain="Law"),
        ]
        judge, calls = _replies_by_answer(
            {
                "ASC 606-10-25-15": "A",
                "ASC 606-10-25-2": "B",
                "I don't know.": "D",
                "Something": "garbage",
            }
        )
        patch_judge(judge)

        await task.score_responses(responses, ScoringContext())

        assert len(calls) == 4
        assert not any("hmm" in prompt for prompt in calls)
        assert [r.instance.metadata["judge_result"]["grade"] for r in responses] == [
            "CORRECT",
            "INCORRECT",
            "NOT_ATTEMPTED",
            "PARSING_ERROR",
        ]
        assert [r.scores["omniscience_judge"] for r in responses] == [1.0, 0.0, 0.0, 0.0]

        metrics = task.compute_metrics(responses)
        judge_name = "omniscience_judge"
        assert metrics["any__any__omniscience_index"][judge_name] == pytest.approx(0.0)
        assert metrics["any__any__hallucination_rate"][judge_name] == pytest.approx(0.5)
        assert metrics["any__any__accuracy"][judge_name] == pytest.approx(1 / 3)
        assert metrics["domain__Finance__accuracy"][judge_name] == pytest.approx(0.5)
        assert metrics["domain__Law__accuracy"][judge_name] == pytest.approx(0.0)
        assert metrics["judge_parsing_errors"][judge_name] == 1.0

    @pytest.mark.anyio
    async def test_no_final_answer_counts_as_not_attempted(self, task, patch_judge):
        responses = [
            _response(task, text="ASC 606-10-25-15"),
            _response(task, text="<think>ASC 606-10-25-15 or maybe"),
            _response(task, text="<think>ASC 606-10-25-15</think>\n\n"),
            _response(task, text=""),
        ]
        judge, calls = _replies_by_answer({"ASC 606-10-25-15": "A"})
        patch_judge(judge)

        await task.score_responses(responses, ScoringContext())

        assert len(calls) == 1
        assert [r.instance.metadata["judge_result"] for r in responses] == [
            {"grade": "CORRECT", "empty_response": False},
            {"grade": "NOT_ATTEMPTED", "empty_response": True},
            {"grade": "NOT_ATTEMPTED", "empty_response": True},
            {"grade": "NOT_ATTEMPTED", "empty_response": True},
        ]

        metrics = task.compute_metrics(responses)
        assert metrics["any__any__accuracy"]["omniscience_judge"] == pytest.approx(1 / 4)
        assert metrics["any__any__omniscience_index"]["omniscience_judge"] == pytest.approx(25.0)
        assert metrics["any__any__hallucination_rate"]["omniscience_judge"] == 0.0
        assert metrics["judge_parsing_errors"]["omniscience_judge"] == 0.0

    @pytest.mark.anyio
    async def test_a_failed_judge_call_is_logged_once_counted_and_excluded(
        self, task, patch_judge, caplog
    ):
        responses = [_response(task, text="answered"), _response(task, text="judge fails")]
        judge, _ = _replies_by_answer({"answered": "A"})
        patch_judge(judge)

        with caplog.at_level(logging.WARNING):
            scored = await task.score_responses(responses, ScoringContext())

        assert len(scored) == 2
        warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) == 1
        assert "omniscience_judge" in warnings[0]
        assert "judge unavailable" in warnings[0]

        failed = responses[1]
        assert failed.outputs[0].metadata["scoring_errors"]["omniscience_judge"]
        assert "omniscience_judge" not in failed.scores
        assert failed.instance.metadata["is_parsing_error"] is True
        assert failed.instance.metadata["judge_result"]["grade"] == "PARSING_ERROR"

        metrics = task.compute_metrics(responses)
        assert metrics["any__any__accuracy"]["omniscience_judge"] == 1.0
        assert metrics["any__any__omniscience_index"]["omniscience_judge"] == 100.0
        # The failed call is left out of every metric, its own error count included;
        # the runner reports it among the task's scoring failures.
        assert metrics["judge_parsing_errors"]["omniscience_judge"] == 0.0

    @pytest.mark.anyio
    async def test_no_warning_when_every_judge_call_succeeds(self, task, patch_judge, caplog):
        responses = [_response(task, text="answered"), _response(task, text="garbage reply")]
        judge, _ = _replies_by_answer({"answered": "A", "garbage reply": "not a grade"})
        patch_judge(judge)

        with caplog.at_level(logging.WARNING):
            await task.score_responses(responses, ScoringContext())

        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

    @pytest.mark.anyio
    async def test_judge_output_is_saved_to_the_requests_file(self, task, patch_judge):
        responses = [_response(task, text="graded"), _response(task, text="unparseable")]
        judge, _ = _replies_by_answer({"graded": "A", "unparseable": "not a grade"})
        patch_judge(judge)

        await task.score_responses(responses, ScoringContext())
        rows = build_requests_from_responses(responses, "omniscience:judge")

        assert rows[0]["doc"]["judge_raw_response"] == "A"
        assert rows[0]["doc"]["judge_result"]["grade"] == "CORRECT"
        assert rows[0]["doc"]["is_parsing_error"] is False
        assert rows[1]["doc"]["judge_raw_response"] == "not a grade"
        assert rows[1]["doc"]["judge_result"]["grade"] == "PARSING_ERROR"
        assert rows[1]["doc"]["is_parsing_error"] is True

    @pytest.mark.anyio
    async def test_empty_response_is_saved_to_the_predictions_file(self, task, patch_judge):
        responses = [_response(task, text="I don't know."), _response(task, text="<think>hmm")]
        judge, _ = _replies_by_answer({"I don't know.": "D"})
        patch_judge(judge)

        await task.score_responses(responses, ScoringContext())
        predictions = build_predictions(responses, metrics=task.config.metrics)

        assert [p["judge_result"] for p in predictions] == [
            {"grade": "NOT_ATTEMPTED", "empty_response": False},
            {"grade": "NOT_ATTEMPTED", "empty_response": True},
        ]


class TestSingleSample:
    """Omniscience grades exactly one answer per question."""

    @pytest.fixture
    def loads(self, monkeypatch):
        """Record dataset loads, serving one instance instead of downloading the dataset."""
        calls: list[str | None] = []

        def load(self, split=None):
            calls.append(split)
            yield self.process_doc(_doc())

        monkeypatch.setattr(omniscience.Omniscience, "_load_instances", load)
        return calls

    def test_several_samples_are_rejected_before_the_dataset_loads(self, loads):
        task = get_task(
            "omniscience:judge",
            {"sampling_params": replace(omniscience.DEFAULT_SAMPLING, num_samples=2)},
        )

        with pytest.raises(ValueError, match="num_samples must be 1"):
            list(task.instances)

        assert loads == []

    def test_a_num_samples_override_fails_before_any_request_is_built(self, loads, monkeypatch):
        formatted: list[Instance] = []
        monkeypatch.setattr(
            omniscience.Omniscience,
            "format_request",
            lambda self, instance: formatted.append(instance),
        )

        with pytest.raises(ValueError, match="num_samples must be 1"):
            prepare_task_items(
                "omniscience:judge", "model", None, sampling_overrides={"num_samples": 2}
            )

        assert loads == []
        assert formatted == []

    def test_a_single_sample_is_allowed(self, task, loads):
        assert [instance.metadata["id"] for instance in task.instances] == [7]
        assert len(loads) == 1
