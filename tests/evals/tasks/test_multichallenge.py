"""Tests for the MultiChallenge task."""

import pytest

from olmo_eval.common.types import LMOutput, RequestType, Response
from olmo_eval.evals.tasks import multichallenge
from olmo_eval.evals.tasks.common import OutputScoreAggregation, get_task, task_exists
from olmo_eval.evals.tasks.common import judge as judge_module
from olmo_eval.evals.tasks.multichallenge import (
    MULTICHALLENGE_AXES,
    MULTICHALLENGE_DEFAULT_JUDGE_MODEL,
    MULTICHALLENGE_JUDGE_PROMPT,
    parse_verdict,
)
from olmo_eval.runners.asynq.preparation import compute_task_metrics
from olmo_eval.runners.io.builders import build_predictions
from tests.evals.tasks.judge_stubs import score_like_the_runner, scripted_judge, use_judge

pytestmark = pytest.mark.usefixtures("judge_test_env")

TARGET_QUESTION = "Does the response only suggest venues within a 5-minute walk?"
CONVERSATION = [
    {"role": "user", "content": "I only go places within a 5-minute walk of the UN."},
    {"role": "assistant", "content": "Noted. Here are some nearby venues."},
    {"role": "user", "content": "Where should I have lunch on Friday?"},
]


def _doc(
    axis: str = "INFERENCE_MEMORY", pass_criteria: str = "YES", question_id: str = "q1"
) -> dict:
    # The Hub stores the conversation column-wise.
    return {
        "question_id": question_id,
        "axis": axis,
        "conversation": {
            "role": [turn["role"] for turn in CONVERSATION],
            "content": [turn["content"] for turn in CONVERSATION],
        },
        "target_question": TARGET_QUESTION,
        "pass_criteria": pass_criteria,
        "num_turns": 2,
    }


def _response(task, *texts: str, **doc_fields) -> Response:
    instance = task.process_doc(_doc(**doc_fields))
    assert instance is not None
    outputs = [LMOutput(text=text) for text in texts or ("Try the Delegates Dining Room.",)]
    return Response(instance=instance, request=task.format_request(instance), outputs=outputs)


def _reply(verdict: str) -> str:
    return f"Reasoning: The venue is across the street.\n\nVerdict: {verdict}"


@pytest.fixture
def task():
    return get_task("multichallenge")


class TestRegistration:
    def test_is_registered_against_the_pinned_scale_dataset(self, task):
        assert task_exists("multichallenge")
        source = task.config.data_source
        assert source.path == "ScaleAI/MultiChallenge"
        assert source.revision == multichallenge.MULTICHALLENGE_REVISION
        assert source.split == "test"

    def test_judge_defaults_and_secrets(self, task):
        config = task.config
        assert config.judge_model == MULTICHALLENGE_DEFAULT_JUDGE_MODEL
        assert config.judge_max_tokens == multichallenge.MULTICHALLENGE_JUDGE_MAX_TOKENS
        assert config.required_secrets == ("OPENAI_API_KEY",)
        assert config.output_score_aggregation == OutputScoreAggregation.MEAN

    def test_judge_settings_are_part_of_the_task_hash(self, task):
        serialized = task.config.to_dict()
        assert serialized["judge_model"] == MULTICHALLENGE_DEFAULT_JUDGE_MODEL
        assert serialized["judge_max_tokens"] == multichallenge.MULTICHALLENGE_JUDGE_MAX_TOKENS

    def test_metrics_cover_the_overall_score_and_every_axis(self, task):
        names = {metric.name for metric in task.config.metrics}
        assert task.config.get_primary_metric().name == "macro_accuracy"
        assert {"macro_accuracy", "accuracy", "truncation_rate"} <= names
        assert {f"accuracy_{suffix}" for suffix in MULTICHALLENGE_AXES.values()} <= names

    def test_generation_samples_up_to_the_context_limit(self, task):
        params = task.config.sampling_params
        assert params.temperature == 0.6
        assert params.top_p == 0.95
        assert params.max_tokens is None


class TestProcessDoc:
    def test_maps_the_schema(self, task):
        instance = task.process_doc(_doc(axis="SELF_COHERENCE"), index=4)

        assert instance is not None
        assert instance.question == TARGET_QUESTION
        assert instance.gold_answer == "YES"
        assert instance.metadata["id"] == "q1"
        assert instance.metadata["axis"] == "SELF_COHERENCE"
        assert instance.metadata["messages"] == tuple(CONVERSATION)
        assert instance.metadata["index"] == 4

    def test_accepts_a_list_of_messages(self, task):
        instance = task.process_doc({**_doc(), "conversation": CONVERSATION})

        assert instance is not None
        assert instance.metadata["messages"] == tuple(CONVERSATION)

    def test_normalizes_the_pass_criteria(self, task):
        instance = task.process_doc(_doc(pass_criteria=" no "))

        assert instance is not None
        assert instance.gold_answer == "NO"

    def test_skips_a_conversation_that_does_not_end_with_the_user(self, task):
        doc = {**_doc(), "conversation": CONVERSATION[:2]}

        assert task.process_doc(doc) is None

    def test_request_is_the_conversation_without_a_system_prompt(self, task):
        instance = task.process_doc(_doc())
        assert instance is not None

        request = task.format_request(instance)

        assert task.request_type == RequestType.CHAT
        assert request.request_type == RequestType.CHAT
        assert request.messages == tuple(CONVERSATION)


class TestParseVerdict:
    def test_reads_a_labeled_verdict(self):
        assert parse_verdict(_reply("YES")) == "YES"
        assert parse_verdict(_reply("NO")) == "NO"

    def test_tolerates_markdown_and_lowercase_labels(self):
        assert parse_verdict("**Verdict:** no") == "NO"
        assert parse_verdict('verdict: "yes"') == "YES"

    def test_takes_the_last_bare_verdict(self):
        assert parse_verdict("The answer to whether it is NO longer valid...\n\nYES") == "YES"

    def test_a_label_counts_only_at_the_start_of_a_line(self):
        assert parse_verdict("This clearly merits a YES verdict. No issues remain.") == "YES"
        assert parse_verdict("Final verdict: yes") == "YES"
        assert parse_verdict('> **Verdict**: "NO"') == "NO"
        assert parse_verdict("### Verdict\n\nNO") == "NO"

    def test_prose_about_a_verdict_is_unparseable(self):
        assert parse_verdict("I cannot give a verdict; no information.") is None

    def test_ignores_lowercase_words_in_unlabeled_reasoning(self):
        assert parse_verdict("yes, the response is fine, no issues\nNO") == "NO"
        assert parse_verdict("yes, the response is fine") is None

    def test_no_verdict_is_unparseable(self):
        assert parse_verdict("I cannot tell.") is None


class TestScoring:
    @pytest.mark.anyio
    async def test_judges_the_visible_response_against_the_rubric(self, task, monkeypatch):
        response = _response(task, "<think>hidden</think>\nTry the Delegates Dining Room.")
        judge, calls = scripted_judge(_reply("YES"))
        use_judge(monkeypatch, judge)

        await task.score_responses([response])

        assert calls == [
            MULTICHALLENGE_JUDGE_PROMPT.format(
                response="Try the Delegates Dining Room.", criteria=TARGET_QUESTION
            )
        ]
        assert response.scores["accuracy"] == 1.0
        saved = build_predictions([response])[0]["model_output"][0]["judge_result"]
        assert saved == {
            "scorer": "multichallenge_judge",
            "truncated": False,
            "pass_criteria": "YES",
            "raw_judge_response": _reply("YES"),
            "verdict": "YES",
            "passed": True,
            "parse_error": False,
        }

    @pytest.mark.anyio
    async def test_an_unparseable_reply_is_retried_not_scored(self, task, monkeypatch):
        response = _response(task)
        judge, calls = scripted_judge("I cannot give a verdict; no information.", _reply("YES"))
        use_judge(monkeypatch, judge)

        await task.score_responses([response])

        assert len(calls) == 2
        assert response.scores["accuracy"] == 1.0

    @pytest.mark.anyio
    async def test_a_verdict_that_misses_the_criteria_fails(self, task, monkeypatch):
        response = _response(task)
        judge, _ = scripted_judge(_reply("NO"))
        use_judge(monkeypatch, judge)

        await task.score_responses([response])

        assert response.scores["accuracy"] == 0.0
        assert response.outputs[0].metadata["judge_result"]["passed"] is False

    @pytest.mark.anyio
    async def test_passing_compares_against_the_expected_verdict(self, task, monkeypatch):
        response = _response(task, pass_criteria="NO")
        judge, _ = scripted_judge(_reply("NO"))
        use_judge(monkeypatch, judge)

        await task.score_responses([response])

        assert response.scores["accuracy"] == 1.0

    @pytest.mark.anyio
    async def test_unfinished_reasoning_fails_without_the_judge(self, task, monkeypatch):
        response = _response(task, "<think>the user wants somewhere close")
        judge, calls = scripted_judge()
        use_judge(monkeypatch, judge)

        await task.score_responses([response])

        assert calls == []
        assert response.scores["accuracy"] == 0.0
        judge_result = response.outputs[0].metadata["judge_result"]
        assert judge_result["no_final_response"] is True
        assert judge_result["passed"] is False

    @pytest.mark.anyio
    async def test_a_response_cut_off_at_the_token_limit_fails_without_the_judge(
        self, task, monkeypatch
    ):
        truncated = _response(task, "Try the Delegates")
        truncated.outputs[0].metadata["finish_reason"] = "length"
        judge, calls = scripted_judge()
        use_judge(monkeypatch, judge)

        await task.score_responses([truncated])

        assert calls == []
        assert truncated.scores == {"accuracy": 0.0, "truncation_rate": 1.0}
        assert truncated.outputs[0].metadata["judge_result"]["truncated"] is True

    @pytest.mark.anyio
    async def test_overall_score_averages_the_axes(self, task, monkeypatch):
        responses = [
            _response(task, axis="INFERENCE_MEMORY", question_id="m1"),
            _response(task, axis="INFERENCE_MEMORY", question_id="m2"),
            _response(task, axis="INFERENCE_MEMORY", question_id="m3"),
            _response(task, axis="SELF_COHERENCE", question_id="s1"),
        ]
        judge, _ = scripted_judge(_reply("YES"), _reply("YES"), _reply("YES"), _reply("NO"))
        use_judge(monkeypatch, judge)

        await task.score_responses(responses)

        metrics = {metric.name: metric.compute(responses) for metric in task.config.metrics}
        assert metrics["macro_accuracy"] == 0.5
        assert metrics["accuracy"] == 0.75
        assert metrics["accuracy_inference_memory"] == 1.0
        assert metrics["accuracy_self_coherence"] == 0.0
        assert metrics["truncation_rate"] == 0.0


@pytest.mark.anyio
async def test_an_unjudged_conversation_is_incomplete_not_failed(task, monkeypatch):
    judged = _response(task, question_id="judged")
    unjudged = _response(task, question_id="unjudged")
    judge, _ = scripted_judge(_reply("YES"), *["no verdict here"] * judge_module.JUDGE_ATTEMPTS)
    use_judge(monkeypatch, judge)

    scored = [await score_like_the_runner(task, judged)]
    scored.append(await score_like_the_runner(task, unjudged))
    result = compute_task_metrics("multichallenge", task, scored, {}, len(scored), 0.0)

    assert (
        result.error is not None
        and "could not be scored" in result.error
        and result.instances_unscored >= 1
    )
    assert result.metrics["macro_accuracy"] == {"multichallenge_judge": 1.0}
    assert result.metrics["accuracy"] == {"multichallenge_judge": 1.0}
    judged_prediction, unjudged_prediction = result.predictions
    assert judged_prediction["instance_metrics"]["accuracy"]["multichallenge_judge"] == 1.0
    assert "accuracy" not in unjudged_prediction["instance_metrics"]
