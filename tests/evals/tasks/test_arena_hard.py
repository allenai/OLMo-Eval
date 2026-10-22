"""Tests for the Arena-Hard v2.0 tasks."""

import math
from dataclasses import replace

import pytest
import tiktoken

from olmo_eval.common.types import LMOutput, RequestType, Response
from olmo_eval.common.types.base import compute_task_hash
from olmo_eval.evals.suites import get_suite
from olmo_eval.evals.tasks import arena_hard
from olmo_eval.evals.tasks.arena_hard import (
    ARENA_HARD_DEFAULT_JUDGE_MODEL,
    ARENA_HARD_JUDGE_MAX_TOKENS,
    CREATIVE_WRITING_JUDGE_SYSTEM_PROMPT,
    HARD_PROMPT_JUDGE_SYSTEM_PROMPT,
    StyleCounts,
    StyleModel,
    battle_outcomes,
    fit_win_rate,
    parse_verdict,
    style_counts,
    style_features,
)
from olmo_eval.evals.tasks.common import OutputScoreAggregation, get_task, task_exists
from olmo_eval.evals.tasks.common import judge as judge_module
from olmo_eval.runners.asynq.preparation import compute_task_metrics
from olmo_eval.runners.io.builders import build_predictions
from tests.evals.tasks.judge_stubs import score_like_the_runner, use_judge

pytestmark = pytest.mark.usefixtures("judge_test_env")

TASKS = ("arena_hard_v2:hard_prompt", "arena_hard_v2:creative_writing")
BASELINE = "The baseline answer."


@pytest.fixture(autouse=True)
def _word_tokens(monkeypatch):
    monkeypatch.setattr(arena_hard, "count_tokens", lambda text: len(text.split()))


def _task(name: str = "arena_hard_v2:hard_prompt"):
    task = get_task(name)
    task.baseline_answers = {"q1": BASELINE, "q2": BASELINE}
    return task


@pytest.fixture
def task():
    return _task()


def _doc(uid: str = "q1", category: str = "hard_prompt", subcategory: str = "coding") -> dict:
    return {"uid": uid, "category": category, "subcategory": subcategory, "prompt": "Write X."}


def _response(task, *texts: str, uid: str = "q1", subcategory: str = "coding") -> Response:
    instance = task.process_doc(_doc(uid, task.category.name, subcategory))
    assert instance is not None
    outputs = [LMOutput(text=text) for text in texts or ("The model answer.",)]
    return Response(instance=instance, request=task.format_request(instance), outputs=outputs)


def _scripted_judge(*replies: str):
    remaining = iter(replies)
    calls: list[dict] = []

    async def judge(prompt: str, **kwargs) -> str:
        calls.append({"prompt": prompt, **kwargs})
        return next(remaining)

    return judge, calls


def _metric(task, name: str):
    return next(metric for metric in task.config.metrics if metric.name == name)


class TestRegistration:
    def test_tasks_and_suite_are_registered_against_the_pinned_source(self):
        for name in TASKS:
            assert task_exists(name)
            source = get_task(name).config.data_source
            assert source.path == "lmarena-ai/arena-hard-auto"
            assert source.revision == arena_hard.ARENA_HARD_REVISION
            assert source.data_files == "data/arena-hard-v2.0/question.jsonl"
        assert get_suite("arena_hard_v2").expand() == TASKS

    def test_judge_defaults_and_secrets(self, task):
        assert task.config.judge_model == ARENA_HARD_DEFAULT_JUDGE_MODEL
        assert task.config.judge_max_tokens == ARENA_HARD_JUDGE_MAX_TOKENS
        assert task.config.required_secrets == ("OPENAI_API_KEY",)
        assert task.config.output_score_aggregation == OutputScoreAggregation.MEAN

    def test_judge_settings_and_style_constants_are_part_of_the_task_hash(self, task):
        serialized = task.config.to_dict()
        assert serialized["judge_model"] == ARENA_HARD_DEFAULT_JUDGE_MODEL
        assert serialized["judge_max_tokens"] == ARENA_HARD_JUDGE_MAX_TOKENS
        primary = serialized["primary_metric"]
        assert primary["style_model"]["coefs"] == arena_hard.HARD_PROMPT_STYLE_MODEL.coefs
        assert "style_model" not in arena_hard.ArenaHardWinRateMetric().to_dict()
        refit = replace(
            task.config.primary_metric,
            style_model=replace(arena_hard.HARD_PROMPT_STYLE_MODEL, coefs=(1.0, 0.0, 0.0, 0.0)),
        )
        changed = replace(task.config, primary_metric=refit)
        assert compute_task_hash(changed.to_dict()) != compute_task_hash(task.config.to_dict())

    def test_generation_samples_up_to_a_fitted_budget(self, task):
        params = task.config.sampling_params
        assert (params.temperature, params.top_p) == (0.6, 0.95)
        assert params.max_tokens == 32768
        assert params.fit_max_tokens_to_context

    def test_primary_metrics_follow_upstream_reporting(self):
        hard = get_task("arena_hard_v2:hard_prompt").config
        assert hard.get_primary_metric().name == "win_rate_style_controlled"
        assert hard.get_primary_metric().style_model == arena_hard.HARD_PROMPT_STYLE_MODEL
        assert [metric.name for metric in hard.metrics] == [
            "win_rate_style_controlled",
            "win_rate",
            "win_rate_coding",
            "win_rate_math",
            "truncation_rate",
        ]
        creative = get_task("arena_hard_v2:creative_writing").config
        assert creative.get_primary_metric().name == "win_rate"
        style = _metric(get_task("arena_hard_v2:creative_writing"), "win_rate_style_controlled")
        assert style.style_model == arena_hard.CREATIVE_WRITING_STYLE_MODEL


class TestProcessDoc:
    def test_keeps_only_its_category_and_attaches_the_baseline(self, task):
        instance = task.process_doc(_doc(subcategory="math"))
        assert instance is not None
        assert instance.question == "Write X."
        assert instance.metadata["id"] == "q1"
        assert instance.metadata["subcategory"] == "math"
        assert instance.metadata["baseline_model"] == "o3-mini-2025-01-31"
        assert instance.metadata["baseline_answer"] == BASELINE
        assert task.process_doc(_doc(category="creative_writing")) is None

    def test_missing_baseline_answer_is_an_error(self, task):
        with pytest.raises(ValueError, match="no o3-mini-2025-01-31 answer"):
            task.process_doc(_doc(uid="unknown"))

    def test_baseline_answers_come_from_the_configured_revision(self, monkeypatch):
        loaded = []

        def load(_loader, source):
            loaded.append(source)
            return [{"uid": "q1", "messages": [{"content": {"answer": BASELINE}}]}]

        monkeypatch.setattr(arena_hard.DataLoader, "load", load)
        source = replace(arena_hard.ArenaHardV2HardPrompt.data_source, revision="abc123")
        task = get_task("arena_hard_v2:hard_prompt", {"data_source": source})

        assert task.baseline_answers == {"q1": BASELINE}
        assert loaded[0].revision == "abc123"
        assert loaded[0].data_files == arena_hard.HARD_PROMPT.baseline_answers_file

    def test_request_is_the_prompt_alone(self, task):
        request = task.format_request(task.process_doc(_doc()))
        assert request.request_type == RequestType.CHAT
        assert request.messages == ({"role": "user", "content": "Write X."},)


class TestVerdicts:
    @pytest.mark.parametrize(
        ("reply", "verdict"),
        [
            ("My final verdict is tie: [[A=B]]", "A=B"),
            ("Considering [[A>B]] at first... final verdict: [[b>>a]]", "B>>A"),
            ("Final verdict: [A>B]", "A>B"),
            ("no verdict here", None),
            ("My final verdict: [[A>>>B]]", None),
        ],
    )
    def test_parse_verdict(self, reply, verdict):
        assert parse_verdict(reply) == verdict

    def test_battles_weight_decisive_verdicts_and_flip_the_baseline_game(self):
        # Baseline as A wins slightly; model as A wins decisively.
        assert battle_outcomes("A>B", "A>>B") == (1.0, 1.0, 1.0, 0.0)
        assert battle_outcomes("A=B", "B=A") == (0.5, 0.5)
        assert battle_outcomes("B>>A", "B>A") == (0.0, 1.0, 1.0, 1.0)


class TestStyle:
    def test_counts_markdown_outside_code_blocks(self):
        text = (
            "# Title\n## Section\n- item\n* item\n1. first\n**bold** and __bold__\n"
            "```python\n# not a header\n- not a list\n```"
        )
        counts = style_counts(text)
        assert (counts.headers, counts.lists, counts.bold) == (2, 3, 2)
        assert counts.tokens == len(text.split())

    def test_features_are_zero_for_matching_styles(self):
        same = StyleCounts(tokens=100, headers=2, lists=5, bold=1)
        assert style_features(same, same) == (0.0, 0.0, 0.0, 0.0)

    def test_features_follow_upstream_contrasts(self):
        model = StyleCounts(tokens=300, headers=3, lists=0, bold=0)
        baseline = StyleCounts(tokens=100, headers=0, lists=0, bold=0)
        length, headers, lists, bold = style_features(model, baseline)
        assert length == pytest.approx(0.5)
        density = 3 / 301
        assert headers == pytest.approx(density / (density + 1))
        assert (lists, bold) == (0.0, 0.0)

    def test_empty_responses_have_no_length_contrast(self):
        empty = StyleCounts(tokens=0, headers=0, lists=0, bold=0)
        assert style_features(empty, empty)[0] == 0.0


class TestWinRateFit:
    def test_without_offsets_it_is_the_raw_win_rate(self):
        outcomes = [1.0, 1.0, 0.0, 0.5]
        assert fit_win_rate(outcomes, [0.0] * 4) == pytest.approx(0.625)

    def test_style_offsets_explain_away_wins(self):
        # Wins that the style term already predicts leave less credit to the model.
        outcomes = [1.0, 1.0, 0.0, 0.0]
        controlled = fit_win_rate(outcomes, [2.0, 2.0, 0.0, 0.0])
        assert controlled < 0.5
        expected = sum(
            1 / (1 + math.exp(-(math.log(controlled / (1 - controlled)) + o)))
            for o in [2.0, 2.0, 0.0, 0.0]
        )
        assert expected == pytest.approx(sum(outcomes), abs=1e-6)

    def test_one_sided_battles_reach_the_limit(self):
        assert fit_win_rate([0.0, 0.0], [0.0, 0.0]) < 1e-6
        assert fit_win_rate([1.0], [0.0]) > 1 - 1e-6
        assert fit_win_rate([], []) == 0.0

    def test_style_model_offset_standardizes_features(self):
        model = StyleModel(means=(0.1, 0.0), stds=(0.2, 0.5), coefs=(1.0, -2.0))
        assert model.offset((0.5, 0.25)) == pytest.approx((0.4 / 0.2) - 2.0 * 0.5)


class TestScoring:
    @pytest.mark.anyio
    async def test_judges_both_orders_of_the_visible_answer(self, task, monkeypatch):
        response = _response(task, "<think>hidden</think>Visible answer here.")
        judge, calls = _scripted_judge("verdict [[A>B]]", "verdict [[A>>B]]")
        use_judge(monkeypatch, judge)

        await task.score_responses([response])

        assert [call["system_prompt"] for call in calls] == [HARD_PROMPT_JUDGE_SYSTEM_PROMPT] * 2
        baseline_first, model_first = (call["prompt"] for call in calls)
        assert "hidden" not in baseline_first + model_first
        assert baseline_first.startswith("<|User Prompt|>\nWrite X.\n\n")
        assert (
            "<|The Start of Assistant A's Answer|>\nThe baseline answer.\n"
            "<|The End of Assistant A's Answer|>" in baseline_first
        )
        assert (
            "<|The Start of Assistant A's Answer|>\nVisible answer here.\n"
            "<|The End of Assistant A's Answer|>" in model_first
        )
        output = response.outputs[0]
        assert response.scores["win_rate"] == pytest.approx(0.75)
        judge_result = output.metadata["judge_result"]
        assert judge_result["battle_outcomes"] == [1.0, 1.0, 1.0, 0.0]
        assert len(judge_result["style_features"]) == 4
        assert judge_result["verdict_baseline_first"] == "A>B"
        assert judge_result["verdict_model_first"] == "A>>B"
        assert judge_result["raw_judge_response_baseline_first"] == "verdict [[A>B]]"
        assert judge_result["raw_judge_response_model_first"] == "verdict [[A>>B]]"
        assert output.metadata["score:arena_hard_judge"] == pytest.approx(0.75)
        assert response.scores["truncation_rate"] == 0.0

    @pytest.mark.anyio
    async def test_an_unclosed_reasoning_block_is_judged_as_an_empty_answer(
        self, task, monkeypatch
    ):
        response = _response(task, "<think>never finished")
        judge, calls = _scripted_judge("[[A>>B]]", "[[B>>A]]")
        use_judge(monkeypatch, judge)

        await task.score_responses([response])

        assert "never finished" not in calls[0]["prompt"] + calls[1]["prompt"]
        assert response.scores["win_rate"] == 0.0
        # The style model would otherwise excuse the losses as those of a short answer.
        assert response.outputs[0].metadata["judge_result"]["style_features"] == [0.0] * 4

    @pytest.mark.anyio
    async def test_creative_writing_uses_its_own_judge_prompt(self, monkeypatch):
        task = _task("arena_hard_v2:creative_writing")
        response = _response(task, subcategory="creative_writing")
        judge, calls = _scripted_judge("[[A=B]]", "[[A=B]]")
        use_judge(monkeypatch, judge)

        await task.score_responses([response])

        assert calls[0]["system_prompt"] == CREATIVE_WRITING_JUDGE_SYSTEM_PROMPT
        assert "generating your own answer" not in CREATIVE_WRITING_JUDGE_SYSTEM_PROMPT
        assert response.scores["win_rate"] == 0.5

    @pytest.mark.anyio
    async def test_retries_an_unparseable_reply(self, task, monkeypatch):
        response = _response(task)
        judge, calls = _scripted_judge("thinking...", "[[B>A]]", "[[A>B]]")
        use_judge(monkeypatch, judge)

        await task.score_responses([response])

        assert len(calls) == 3
        assert response.scores["win_rate"] == 1.0

    @pytest.mark.anyio
    async def test_a_failed_game_waits_for_the_other_and_keeps_its_reply(self, task, monkeypatch):
        async def judge(prompt: str, **_kwargs) -> str:
            baseline_first = prompt.index(BASELINE) < prompt.index("The model answer.")
            return "no verdict" if baseline_first else "[[A>B]]"

        use_judge(monkeypatch, judge)

        scored = await score_like_the_runner(task, _response(task))

        judge_result = scored.outputs[0].metadata["judge_result"]
        assert judge_result["parse_error"] is True
        assert judge_result["verdict_model_first"] == "A>B"
        assert judge_result["raw_judge_response_baseline_first"] == "no verdict"
        assert "battle_outcomes" not in judge_result

    @pytest.mark.anyio
    async def test_style_is_measured_before_the_judge_is_paid(self, task, monkeypatch):
        def broken_tokenizer(_text: str) -> int:
            raise RuntimeError("no tokenizer")

        monkeypatch.setattr(arena_hard, "count_tokens", broken_tokenizer)
        judge, calls = _scripted_judge()
        use_judge(monkeypatch, judge)

        await score_like_the_runner(task, _response(task))

        assert calls == []

    @pytest.mark.anyio
    async def test_samples_are_averaged(self, task, monkeypatch):
        response = _response(task, "first", "second")
        judge, _ = _scripted_judge("[[B>A]]", "[[A>B]]", "[[A>B]]", "[[B>A]]")
        use_judge(monkeypatch, judge)

        await task.score_responses([response])

        assert response.scores["win_rate"] == 0.5


class TestMetrics:
    @pytest.mark.anyio
    async def test_win_rate_pools_weighted_battles_and_slices_subcategories(
        self, task, monkeypatch
    ):
        responses = [
            _response(task, uid="q1", subcategory="coding"),
            _response(task, uid="q2", subcategory="math"),
        ]
        # q1: model wins decisively in both orders (6 wins); q2: model loses slightly (2 losses).
        judge, _ = _scripted_judge("[[B>>A]]", "[[A>>B]]", "[[A>B]]", "[[B>A]]")
        use_judge(monkeypatch, judge)
        for response in responses:
            await task.score_responses([response])

        result = compute_task_metrics("arena_hard_v2:hard_prompt", task, responses, {}, 2, 0.0)

        assert result.error is None
        assert result.metrics["win_rate"] == {"arena_hard_judge": pytest.approx(6 / 8)}
        assert result.metrics["win_rate_coding"] == {"arena_hard_judge": 1.0}
        assert result.metrics["win_rate_math"] == {"arena_hard_judge": 0.0}
        controlled = result.metrics["win_rate_style_controlled"]["arena_hard_judge"]
        assert 0.0 < controlled < 1.0

        coding, maths = build_predictions(responses, metrics=task.config.metrics)
        saved = coding["model_output"][0]["judge_result"]
        assert saved["battle_outcomes"] == [1.0] * 6
        assert len(saved["style_features"]) == 4
        assert coding["instance_metrics"]["win_rate"]["arena_hard_judge"] == 1.0
        assert coding["instance_metrics"]["win_rate_coding"]["arena_hard_judge"] == 1.0
        assert "win_rate_math" not in coding["instance_metrics"]
        assert "win_rate_style_controlled" not in maths["instance_metrics"]

    def test_style_controlled_metric_applies_the_style_model(self, task):
        response = _response(task)
        response.outputs[0].metadata["judge_result"] = {
            "battle_outcomes": [1.0, 0.0],
            "style_features": [0.5, 0.0, 0.0, 0.0],
        }
        metric = arena_hard.ArenaHardWinRateMetric(
            style_model=StyleModel(means=(0.0,) * 4, stds=(1.0,) * 4, coefs=(1.0, 0, 0, 0))
        )
        assert metric.compute([response]) == pytest.approx(fit_win_rate([1.0, 0.0], [0.5, 0.5]))
        assert metric.compute([response]) < 0.5
        assert metric.higher_is_better()
        assert metric.display_format() == "percentage"
        assert metric.compute_instance(response) is None

    @pytest.mark.anyio
    async def test_an_unanswered_prompt_counts_as_decisive_losses(self, task, monkeypatch):
        answered = _response(task, uid="q1")
        unanswered = _response(task, uid="q2")
        unanswered.outputs = []
        judge, _ = _scripted_judge("[[B>A]]", "[[A>B]]")
        use_judge(monkeypatch, judge)

        await task.score_responses([answered, unanswered])
        result = compute_task_metrics(
            "arena_hard_v2:hard_prompt", task, [answered, unanswered], {}, 2, 0.0
        )

        assert unanswered.scores["win_rate"] == 0.0
        # Two wins for the answered prompt, six decisive losses for the unanswered one.
        assert result.metrics["win_rate"] == {"arena_hard_judge": pytest.approx(2 / 8)}
        assert result.metrics["truncation_rate"] == {"arena_hard_judge": 0.5}
        # Style control must not excuse the unanswered prompt's losses as a short answer.
        controlled = result.metrics["win_rate_style_controlled"]["arena_hard_judge"]
        assert controlled < 0.5

    @pytest.mark.anyio
    async def test_unjudged_question_is_incomplete_and_left_out(self, task, monkeypatch):
        responses = [_response(task, uid="q1"), _response(task, uid="q2")]
        judge, _ = _scripted_judge(
            "[[B>A]]", "[[A>B]]", *["garbage"] * (2 * judge_module.JUDGE_ATTEMPTS)
        )
        use_judge(monkeypatch, judge)

        scored = [await score_like_the_runner(task, response) for response in responses]
        result = compute_task_metrics("arena_hard_v2:hard_prompt", task, scored, {}, 2, 0.0)

        assert result.error is not None
        assert "could not be scored" in result.error and result.instances_unscored >= 1
        assert result.metrics["win_rate"] == {"arena_hard_judge": 1.0}

    @pytest.mark.anyio
    async def test_an_api_error_after_retries_is_incomplete_not_a_loss(self, task, monkeypatch):
        async def judge(_prompt: str, **_kwargs) -> str:
            raise ValueError("judge unavailable")

        use_judge(monkeypatch, judge)

        scored = [await score_like_the_runner(task, _response(task))]
        result = compute_task_metrics("arena_hard_v2:hard_prompt", task, scored, {}, 1, 0.0)

        assert result.error is not None
        error = scored[0].outputs[0].metadata["scoring_errors"]["arena_hard_judge"]
        assert "ValueError: judge unavailable" in error["message"]
        assert result.metrics["win_rate"] == {"arena_hard_judge": 0.0}


def test_style_tokenizer_is_the_one_upstream_uses():
    assert arena_hard._style_encoding().name == tiktoken.encoding_for_model("gpt-4o").name
    # Special-token text in a response is counted as ordinary text, not rejected.
    assert arena_hard._style_encoding().encode("<|endoftext|>", disallowed_special=())
