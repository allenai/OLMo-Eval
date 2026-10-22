"""Tests for the MT-Bench task."""

import re
from collections import Counter

import pytest

from olmo_eval.common.types import (
    FOLLOW_UP_TURNS_KEY,
    TURNS_KEY,
    LMOutput,
    RequestType,
    Response,
    SamplingParams,
)
from olmo_eval.evals.tasks import mt_bench
from olmo_eval.evals.tasks.common import OutputScoreAggregation, get_task, task_exists
from olmo_eval.evals.tasks.common import judge as judge_module
from olmo_eval.evals.tasks.mt_bench import (
    MT_BENCH_CATEGORIES,
    MT_BENCH_DEFAULT_JUDGE_MODEL,
    SECOND_TURN_PROMPT,
    SECOND_TURN_REFERENCE_PROMPT,
    SINGLE_TURN_PROMPT,
    SINGLE_TURN_REFERENCE_PROMPT,
    judge_prompt,
    parse_rating,
)
from olmo_eval.harness.config import HarnessConfig
from olmo_eval.harness.scaffolds import get_scaffold
from olmo_eval.runners.asynq.preparation import compute_task_metrics, prepare_task_items
from tests.evals.tasks.judge_stubs import score_like_the_runner, use_judge
from tests.harness.test_multi_turn_chat_scaffold import OVERFLOW, ScriptedProvider

pytestmark = pytest.mark.usefixtures("judge_test_env")

QUESTIONS = ["Write a haiku about rain.", "Now make it about snow."]
REFERENCE = ["Area is 3.", "Area is 5pi."]


def _doc(category: str = "writing", question_id: int = 81, reference=None) -> dict:
    doc = {"question_id": question_id, "category": category, "turns": QUESTIONS}
    if reference is not None:
        doc["reference"] = reference
    return doc


def _turns(*replies: str) -> list[dict[str, str]]:
    return [{"text": reply} for reply in replies]


def _response(task, replies: list[str] | None = None, **doc_fields) -> Response:
    instance = task.process_doc(_doc(**doc_fields))
    assert instance is not None
    replies = ["Rain falls.", "Snow falls."] if replies is None else replies
    output = LMOutput(text=(replies or [""])[-1], metadata={TURNS_KEY: _turns(*replies)})
    return Response(instance=instance, request=task.format_request(instance), outputs=[output])


_SECOND_TURN_SYSTEM_PROMPTS = {
    SECOND_TURN_PROMPT.system_prompt,
    SECOND_TURN_REFERENCE_PROMPT.system_prompt,
}


def _turn_judge(*replies_by_turn: list[str]):
    """A judge that answers each turn from its own replies, as the turns run concurrently.

    Returns the judge and the ``(turn, system_prompt, prompt)`` of each call.
    """
    remaining = {turn: iter(replies) for turn, replies in enumerate(replies_by_turn, start=1)}
    calls: list[tuple[int, str | None, str]] = []

    async def judge(prompt: str, *, system_prompt: str | None = None) -> str:
        turn = 2 if system_prompt in _SECOND_TURN_SYSTEM_PROMPTS else 1
        calls.append((turn, system_prompt, prompt))
        return next(remaining[turn])

    return judge, calls


def _rated(rating: float) -> str:
    return f"The answer is clear and correct. Rating: [[{rating:g}]]"


@pytest.fixture
def task():
    return get_task("mt_bench")


class TestRegistration:
    def test_loads_the_eighty_questions_with_fastchats_references(self, task):
        instances = list(task.instances)

        assert len(instances) == mt_bench.MT_BENCH_QUESTION_COUNT
        categories = Counter(instance.metadata["category"] for instance in instances)
        assert categories == dict.fromkeys(MT_BENCH_CATEGORIES, 10)
        for instance in instances:
            assert len(instance.metadata["turns"]) == 2
            has_reference = instance.metadata["reference"] is not None
            in_reference_category = (
                instance.metadata["category"] in mt_bench.MT_BENCH_REFERENCE_CATEGORIES
            )
            assert has_reference == in_reference_category

    def test_judge_defaults_and_secrets(self, task):
        config = task.config
        assert task_exists("mt_bench")
        assert config.judge_model == MT_BENCH_DEFAULT_JUDGE_MODEL
        assert config.judge_reasoning_effort == mt_bench.MT_BENCH_DEFAULT_JUDGE_REASONING_EFFORT
        assert config.judge_max_tokens == mt_bench.MT_BENCH_JUDGE_MAX_TOKENS
        assert config.required_secrets == ("OPENAI_API_KEY",)
        assert config.output_score_aggregation == OutputScoreAggregation.MEAN

    def test_judge_settings_are_part_of_the_task_hash(self, task):
        serialized = task.config.to_dict()
        assert serialized["judge_model"] == MT_BENCH_DEFAULT_JUDGE_MODEL
        assert serialized["judge_reasoning_effort"] == "medium"
        assert serialized["judge_max_tokens"] == mt_bench.MT_BENCH_JUDGE_MAX_TOKENS

    def test_metrics_cover_both_turns_and_every_category(self, task):
        names = {metric.name for metric in task.config.metrics}
        assert task.config.get_primary_metric().name == "score"
        assert {"score", "score_turn_1", "score_turn_2"} <= names
        assert {f"score_{category}" for category in MT_BENCH_CATEGORIES} <= names

    def test_scores_are_on_the_rating_scale(self, task):
        units = {metric.name: metric.metric_unit() for metric in task.config.metrics}
        assert units.pop("truncation_rate") == "proportion"
        assert set(units.values()) == {"mt_bench_rating"}
        for metric in task.config.metrics:
            if metric.name != "truncation_rate":
                assert metric.display_format() == "raw"

    def test_generation_samples_up_to_the_context_limit(self, task):
        params = task.config.sampling_params
        assert params.temperature == 0.6
        assert params.top_p == 0.95
        assert params.max_tokens is None
        instance = task.process_doc(_doc(category="math"))
        assert task.get_sampling_params(instance) == params


class TestPaperVariant:
    @pytest.mark.parametrize(
        ("category", "temperature"),
        [("writing", 0.7), ("roleplay", 0.7), ("stem", 0.1), ("humanities", 0.1)],
    )
    def test_samples_at_fastchats_category_temperature(self, category, temperature):
        task = get_task("mt_bench:paper")
        params = task.get_sampling_params(task.process_doc(_doc(category=category)))

        assert params is not None
        assert params.temperature == temperature
        assert params.do_sample is True
        assert params.max_tokens == mt_bench.FASTCHAT_MAX_NEW_TOKENS

    @pytest.mark.parametrize("category", ["math", "coding", "reasoning", "extraction"])
    def test_decodes_greedily_where_fastchat_does(self, category):
        task = get_task("mt_bench:paper")
        params = task.get_sampling_params(task.process_doc(_doc(category=category)))

        assert params is not None
        assert params.temperature == 0.0
        assert params.do_sample is False

    def test_runs_fastchats_temperatures_through_the_runner(self):
        _, items = prepare_task_items("mt_bench:paper", "mock", {"limit": 80}, None)

        temperatures = {
            item.instance.metadata["category"]: item.sampling_params.temperature for item in items
        }
        assert temperatures == mt_bench.FASTCHAT_TEMPERATURES

    def test_rejects_a_temperature_override(self):
        with pytest.raises(ValueError, match="drop the temperature override"):
            prepare_task_items("mt_bench:paper", "mock", None, {"temperature": 0.5})


class TestProcessDoc:
    def test_maps_the_schema(self, task):
        instance = task.process_doc(_doc(category="math", question_id=111, reference=REFERENCE))

        assert instance is not None
        assert instance.question == QUESTIONS[0]
        assert instance.metadata == {
            "id": "111",
            "category": "math",
            "turns": QUESTIONS,
            "reference": REFERENCE,
        }

    def test_a_question_without_a_reference(self, task):
        instance = task.process_doc(_doc())

        assert instance is not None
        assert instance.metadata["reference"] is None

    def test_rejects_a_question_without_two_turns(self, task):
        with pytest.raises(ValueError, match="has 1 turns"):
            task.process_doc({**_doc(), "turns": QUESTIONS[:1]})

    def test_request_asks_the_first_turn_and_scripts_the_second(self, task):
        request = task.format_request(task.process_doc(_doc()))

        assert request.request_type == RequestType.CHAT
        assert request.system_prompt is None
        assert request.messages == ({"role": "user", "content": QUESTIONS[0]},)
        assert request.metadata == {
            FOLLOW_UP_TURNS_KEY: [{"role": "user", "content": QUESTIONS[1]}]
        }


class TestExtractAnswer:
    def test_returns_each_turns_visible_answer(self, task):
        output = LMOutput(
            text="", metadata={TURNS_KEY: _turns("<think>plan</think>\n\nRain.", "<think>never")}
        )
        assert task.extract_answer(output) == ["Rain.", ""]

    def test_without_the_scaffolds_replies(self, task):
        assert task.extract_answer(LMOutput(text="Rain.")) is None


class TestParseRating:
    def test_reads_a_double_bracketed_rating(self):
        assert parse_rating("Clear and helpful. Rating: [[8]]") == 8.0

    def test_reads_a_decimal_rating(self):
        assert parse_rating("Rating: [[7.5]]") == 7.5

    def test_falls_back_to_single_brackets(self):
        assert parse_rating("Rating: [6]") == 6.0

    def test_takes_the_first_rating(self):
        assert parse_rating("Rating: [[3]]. On reflection, [[4]].") == 3.0

    @pytest.mark.parametrize("raw", ["Rating: 8", "Rating: [[11]]", "Rating: [[0]]"])
    def test_no_rating_in_range_is_unparseable(self, raw):
        assert parse_rating(raw) is None


class TestJudgePrompt:
    def test_first_turn_without_a_reference(self, task):
        instance = task.process_doc(_doc())

        system_prompt, prompt = judge_prompt(instance, ["Rain.", "Snow."], 1)

        assert system_prompt == SINGLE_TURN_PROMPT.system_prompt
        assert prompt == SINGLE_TURN_PROMPT.template.format(question=QUESTIONS[0], answer="Rain.")

    def test_first_turn_against_the_reference(self, task):
        instance = task.process_doc(_doc(category="math", reference=REFERENCE))

        system_prompt, prompt = judge_prompt(instance, ["Rain.", "Snow."], 1)

        assert system_prompt == SINGLE_TURN_REFERENCE_PROMPT.system_prompt
        assert "[The Start of Reference Answer]\nArea is 3.\n" in prompt
        assert "Area is 5pi." not in prompt

    def test_second_turn_shows_the_whole_conversation(self, task):
        instance = task.process_doc(_doc())

        system_prompt, prompt = judge_prompt(instance, ["Rain.", "Snow."], 2)

        assert system_prompt == SECOND_TURN_PROMPT.system_prompt
        assert prompt == SECOND_TURN_PROMPT.template.format(
            question_1=QUESTIONS[0], answer_1="Rain.", question_2=QUESTIONS[1], answer_2="Snow."
        )

    def test_second_turn_against_both_references(self, task):
        instance = task.process_doc(_doc(category="coding", reference=REFERENCE))

        system_prompt, prompt = judge_prompt(instance, ["Rain.", "Snow."], 2)

        assert system_prompt == SECOND_TURN_REFERENCE_PROMPT.system_prompt
        assert "### Reference answer:\nArea is 3.\n" in prompt
        assert "### Reference answer:\nArea is 5pi.\n" in prompt
        assert "### Assistant A:\nSnow.\n" in prompt

    def test_answers_with_braces_are_kept_verbatim(self, task):
        instance = task.process_doc(_doc())

        _, prompt = judge_prompt(instance, ["def f(): return {}", "{x}"], 2)

        assert "def f(): return {}" in prompt
        assert "### Assistant A:\n{x}\n" in prompt


class TestJudgeOverride:
    def test_an_override_with_an_effort_replaces_the_default_effort(self, monkeypatch):
        monkeypatch.setenv("OLMO_EVAL_JUDGE", "gpt-5-mini:low")

        config = get_task("mt_bench").config

        assert (config.judge_model, config.judge_reasoning_effort) == ("gpt-5-mini", "low")

    def test_an_override_without_an_effort_drops_the_default_effort(self, monkeypatch):
        monkeypatch.setenv("OLMO_EVAL_JUDGE", "gpt-4-0613")

        config = get_task("mt_bench").config

        assert (config.judge_model, config.judge_reasoning_effort) == ("gpt-4-0613", None)


class TestScoring:
    @pytest.mark.anyio
    async def test_rates_both_turns_with_fastchats_prompts(self, task, monkeypatch):
        response = _response(task, ["<think>hmm</think>Rain falls.", "Snow falls."])
        judge, calls = _turn_judge([_rated(8)], [_rated(5)])
        use_judge(monkeypatch, judge)

        await task.score_responses([response])

        answers = ["Rain falls.", "Snow falls."]
        expected = [(turn, *judge_prompt(response.instance, answers, turn)) for turn in (1, 2)]
        assert sorted(calls) == expected
        assert response.scores == {
            "score": 6.5,
            "turn_1": 8.0,
            "turn_2": 5.0,
            "truncation_rate": 0.0,
        }
        assert response.outputs[0].metadata["judge_result"] == {
            "scorer": "mt_bench_judge",
            "raw_judge_response_turn_1": _rated(8),
            "rating_turn_1": 8.0,
            "raw_judge_response_turn_2": _rated(5),
            "rating_turn_2": 5.0,
            "parse_error": False,
        }

    @pytest.mark.anyio
    async def test_a_turn_without_a_visible_answer_gets_the_lowest_rating(self, task, monkeypatch):
        response = _response(task, ["Rain falls.", "<think>the user wants snow"])
        judge, calls = _turn_judge([_rated(9)], [])
        use_judge(monkeypatch, judge)

        await task.score_responses([response])

        assert [turn for turn, _, _ in calls] == [1]
        assert response.scores["turn_1"] == 9.0
        assert response.scores["turn_2"] == 1.0
        assert response.scores["score"] == 5.0
        judge_result = response.outputs[0].metadata["judge_result"]
        assert judge_result["no_final_response_turn_2"] is True
        assert judge_result["rating_turn_2"] == 1.0

    @pytest.mark.anyio
    async def test_the_second_turn_is_judged_after_a_first_turn_without_an_answer(
        self, task, monkeypatch
    ):
        response = _response(task, ["<think>never closed", "Snow falls."])
        judge, calls = _turn_judge([], [_rated(6)])
        use_judge(monkeypatch, judge)

        await task.score_responses([response])

        assert [turn for turn, _, _ in calls] == [2]
        assert "### Assistant A:\n\n\n### User:" in calls[0][2]
        assert (response.scores["turn_1"], response.scores["turn_2"]) == (1.0, 6.0)

    @pytest.mark.anyio
    async def test_a_turn_the_conversation_never_reached_counts_as_truncated(
        self, task, monkeypatch
    ):
        response = _response(task, ["Rain falls."])
        response.outputs[0].metadata["finish_reason"] = "stop"
        judge, calls = _turn_judge([_rated(7)], [])
        use_judge(monkeypatch, judge)

        await task.score_responses([response])

        assert len(calls) == 1
        assert response.scores["turn_2"] == 1.0
        assert response.scores["truncation_rate"] == 1.0

    @pytest.mark.anyio
    async def test_a_question_without_output_gets_the_lowest_rating(self, task, monkeypatch):
        response = _response(task)
        response.outputs = []
        judge, calls = _turn_judge()
        use_judge(monkeypatch, judge)

        await task.score_responses([response])

        assert calls == []
        assert response.scores == {
            "score": 1.0,
            "turn_1": 1.0,
            "turn_2": 1.0,
            "truncation_rate": 1.0,
        }

    @pytest.mark.anyio
    async def test_samples_are_averaged_per_turn(self, task, monkeypatch):
        response = _response(task)
        response.outputs.append(
            LMOutput(text="", metadata={TURNS_KEY: _turns("Drizzle.", "Flurries.")})
        )
        judge, _ = _turn_judge([_rated(8), _rated(4)], [_rated(6), _rated(2)])
        use_judge(monkeypatch, judge)

        await task.score_responses([response])

        assert response.scores["turn_1"] == 6.0
        assert response.scores["turn_2"] == 4.0
        assert response.scores["score"] == 5.0

    @pytest.mark.anyio
    async def test_a_sample_that_fails_judging_is_left_out_of_every_turn(self, task, monkeypatch):
        response = _response(task)
        response.outputs.append(
            LMOutput(text="", metadata={TURNS_KEY: _turns("Drizzle.", "Flurries.")})
        )
        judge, _ = _turn_judge(
            [_rated(8), *["no rating"] * judge_module.JUDGE_ATTEMPTS], [_rated(6), _rated(2)]
        )
        use_judge(monkeypatch, judge)

        await task.score_responses([response])

        assert (response.scores["turn_1"], response.scores["turn_2"]) == (8.0, 6.0)
        assert response.scores["score"] == 7.0
        assert "score:turn_1" not in response.outputs[1].metadata

    @pytest.mark.anyio
    async def test_the_scaffolds_conversations_are_counted_as_truncated(self, task, monkeypatch):
        # Each question is played through the scaffold: the first stops at the token limit
        # on turn 1 only, the second outgrows the context on turn 2, the third finishes.
        def stop(text: str, finish_reason: str = "stop") -> LMOutput:
            return LMOutput(text=text, metadata={"finish_reason": finish_reason})

        conversations = {
            81: [stop("Rain falls.", "length"), stop("Snow falls.")],
            82: [stop("Rain falls."), OVERFLOW],
            83: [stop("Rain falls."), stop("Snow falls.")],
        }
        responses = []
        for question_id, replies in conversations.items():
            response = _response(task, question_id=question_id)
            scaffold = get_scaffold("multi_turn_chat")
            config = HarnessConfig(name="test")
            provider = ScriptedProvider(replies)
            result = await scaffold.run(provider, config, response.request, SamplingParams())
            response.outputs = [result.final_output]
            responses.append(response)
        judge, _ = _turn_judge([_rated(5)] * 3, [_rated(5)] * 2)
        use_judge(monkeypatch, judge)

        await task.score_responses(responses)

        assert [r.scores["truncation_rate"] for r in responses] == [1.0, 1.0, 0.0]
        metrics = {metric.name: metric for metric in task.config.metrics}
        assert metrics["truncation_rate"].compute(responses) == pytest.approx(2 / 3)

    @pytest.mark.anyio
    async def test_an_out_of_range_rating_is_retried(self, task, monkeypatch):
        response = _response(task)
        judge, calls = _turn_judge(["Rating: [[11]]", _rated(7)], [_rated(6)])
        use_judge(monkeypatch, judge)

        await task.score_responses([response])

        assert len(calls) == 3
        assert response.scores["turn_1"] == 7.0
        assert response.scores["turn_2"] == 6.0

    @pytest.mark.anyio
    async def test_an_unrated_first_turn_leaves_the_question_unscored(self, task, monkeypatch):
        response = _response(task)
        judge, _ = _turn_judge(["no rating"] * judge_module.JUDGE_ATTEMPTS, [_rated(6)])
        use_judge(monkeypatch, judge)

        await task.score_responses([response])

        judge_result = response.outputs[0].metadata["judge_result"]
        assert judge_result["parse_error"] is True
        assert judge_result["raw_judge_response_turn_1"] == "no rating"
        assert judge_result["rating_turn_2"] == 6.0
        assert "score" not in response.scores

    @pytest.mark.anyio
    async def test_a_failed_judge_call_leaves_the_question_unscored(self, task, monkeypatch):
        response = _response(task)

        async def judge(prompt: str, *, system_prompt: str | None = None) -> str:
            raise ConnectionError("judge unavailable")

        use_judge(monkeypatch, judge)

        await task.score_responses([response])

        error = response.outputs[0].metadata["scoring_errors"]["mt_bench_judge"]
        assert "judge unavailable" in error["message"]
        assert error["infrastructure"] == "true"
        assert "score" not in response.scores

    @pytest.mark.anyio
    async def test_requires_the_multi_turn_chat_scaffold(self, task, monkeypatch):
        response = _response(task)
        response.outputs = [LMOutput(text="Rain falls.")]
        judge, calls = _turn_judge()
        use_judge(monkeypatch, judge)

        with pytest.raises(ValueError, match="multi_turn_chat"):
            await task.score_responses([response])
        assert calls == []

    @pytest.mark.anyio
    async def test_metrics_average_turns_and_categories(self, task, monkeypatch):
        # Each turn's answer is its own rating, so the judge needs no script.
        responses = [
            _response(task, ["10", "8"], category="writing", question_id=81),
            _response(task, ["6", "4"], category="writing", question_id=82),
            _response(task, ["2", "6"], category="math", question_id=111, reference=REFERENCE),
        ]

        async def judge(prompt: str, *, system_prompt: str | None = None) -> str:
            # The judged answer is the last one in the prompt.
            answer = re.findall(r"(?:Assistant A:|Assistant's Answer\])\n(\d+)\n", prompt)[-1]
            return _rated(float(answer))

        use_judge(monkeypatch, judge)

        await task.score_responses(responses)

        metrics = {metric.name: metric.compute(responses) for metric in task.config.metrics}
        assert metrics["score"] == 6.0
        assert metrics["score_turn_1"] == 6.0
        assert metrics["score_turn_2"] == 6.0
        assert metrics["score_writing"] == 7.0
        assert metrics["score_math"] == 4.0
        assert metrics["score_coding"] == 0.0


@pytest.mark.anyio
async def test_an_unrated_question_is_incomplete_not_scored(task, monkeypatch):
    rated = _response(task, question_id=81)
    unrated = _response(task, question_id=82)
    judge, _ = _turn_judge(
        [_rated(8), _rated(7)], [_rated(6), *["no rating here"] * judge_module.JUDGE_ATTEMPTS]
    )
    use_judge(monkeypatch, judge)

    scored = [await score_like_the_runner(task, rated)]
    scored.append(await score_like_the_runner(task, unrated))
    result = compute_task_metrics("mt_bench", task, scored, {}, len(scored), 0.0)

    assert (
        result.error is not None
        and "could not be scored" in result.error
        and result.instances_unscored >= 1
    )
    assert result.metrics["score"] == {"mt_bench_judge": 7.0}
    assert result.metrics["score_turn_2"] == {"mt_bench_judge": 6.0}
    judge_result = unrated.outputs[0].metadata["judge_result"]
    assert judge_result["parse_error"] is True
    assert judge_result["raw_judge_response_turn_2"] == "no rating here"
    _, unrated_prediction = result.predictions
    assert "score" not in unrated_prediction["instance_metrics"]
