"""Tests for olmo_eval.core.llm_judge_scorer module."""

import asyncio
import inspect
import json
import sys
import types

import pytest

from olmo_eval.common.execution import ScoringContext
from olmo_eval.common.scorers import (
    RubricJudgeScorer,
    SimpleQAJudgeScorer,
)
from olmo_eval.common.scorers.llm_judge import ConfiguredJudgeFn, build_openai_judge_fn
from olmo_eval.common.types import Instance, LMOutput

_MAX_TOKENS_REJECTION = (
    "Unsupported parameter: 'max_tokens' is not supported with this model. "
    "Use 'max_completion_tokens' instead."
)


def _stub_openai(monkeypatch, create, clients: list[dict] | None = None):
    """Install a stub ``openai`` module whose completions call is ``create``.

    Client constructor arguments are appended to ``clients`` when it is given.
    """
    calls: list[dict] = []

    async def recording_create(**kwargs):
        calls.append(dict(kwargs))
        return await create(**kwargs)

    def make_message(content):
        message = types.SimpleNamespace(content=content)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            if clients is not None:
                clients.append(kwargs)
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=recording_create)
            )

    module = types.ModuleType("openai")
    module.AsyncOpenAI = FakeAsyncOpenAI
    monkeypatch.setitem(sys.modules, "openai", module)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    return calls, make_message


class TestBuildOpenAIJudgeFn:
    """Tests for the adaptive request shape in build_openai_judge_fn."""

    @pytest.mark.anyio
    async def test_switches_to_max_completion_tokens_on_rejection(self, monkeypatch):
        reply = []

        async def create(**kwargs):
            if "max_tokens" in kwargs:
                raise RuntimeError(_MAX_TOKENS_REJECTION)
            return reply[0]

        calls, make_message = _stub_openai(monkeypatch, create)
        reply.append(make_message("VERDICT: CORRECT"))

        judge = build_openai_judge_fn(model="gpt-5", max_tokens=64)

        assert await judge("grade this") == "VERDICT: CORRECT"
        assert [set(call) & {"max_tokens", "max_completion_tokens"} for call in calls] == [
            {"max_tokens"},
            {"max_completion_tokens"},
        ]

    @pytest.mark.anyio
    async def test_concurrent_first_calls_both_adapt(self, monkeypatch):
        """A call that loses the adaptation race must retry, not surface the error."""
        reply = []
        barrier = asyncio.Barrier(2)

        async def create(**kwargs):
            if "max_tokens" in kwargs:
                # Hold both first attempts in flight so they race on one fix.
                await barrier.wait()
                raise RuntimeError(_MAX_TOKENS_REJECTION)
            return reply[0]

        calls, make_message = _stub_openai(monkeypatch, create)
        reply.append(make_message("VERDICT: INCORRECT"))

        judge = build_openai_judge_fn(model="gpt-5", max_tokens=64)
        results = await asyncio.gather(judge("first"), judge("second"))

        assert results == ["VERDICT: INCORRECT", "VERDICT: INCORRECT"]
        assert sum("max_tokens" in call for call in calls) == 2
        assert sum("max_completion_tokens" in call for call in calls) == 2

    @pytest.mark.anyio
    async def test_unrelated_errors_still_surface(self, monkeypatch):
        async def create(**kwargs):
            raise RuntimeError("rate limit exceeded")

        _stub_openai(monkeypatch, create)
        judge = build_openai_judge_fn(model="gpt-5", max_tokens=64)

        with pytest.raises(RuntimeError, match="rate limit exceeded"):
            await judge("grade this")

    @pytest.mark.anyio
    async def test_falls_back_to_openrouter(self, monkeypatch):
        reply = []

        async def create(**kwargs):
            return reply[0]

        clients: list[dict] = []
        calls, make_message = _stub_openai(monkeypatch, create, clients)
        monkeypatch.delenv("OPENAI_API_KEY")
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
        reply.append(make_message("VERDICT: CORRECT"))

        judge = build_openai_judge_fn(model="gpt-4o-mini", max_tokens=64)

        assert await judge("grade this") == "VERDICT: CORRECT"
        assert clients == [{"api_key": "or-key", "base_url": "https://openrouter.ai/api/v1"}]
        assert calls[0]["model"] == "openai/gpt-4o-mini"

    @pytest.mark.anyio
    async def test_missing_keys_name_both_options(self, monkeypatch):
        async def create(**kwargs):
            raise AssertionError("no request should be sent")

        _stub_openai(monkeypatch, create)
        monkeypatch.delenv("OPENAI_API_KEY")
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        judge = build_openai_judge_fn(model="gpt-4o-mini", scorer_name="MyScorer")

        with pytest.raises(ValueError, match="OPENAI_API_KEY or OPENROUTER_API_KEY .*MyScorer"):
            await judge("grade this")

    def test_records_the_settings_that_affect_judge_output(self):
        judge = build_openai_judge_fn(
            model="gpt-5", max_tokens=64, temperature=0.5, reasoning_effort="low"
        )

        assert judge.judge_config == {
            "model": "gpt-5",
            "max_tokens": 64,
            "temperature": 0.5,
            "reasoning_effort": "low",
        }

    @pytest.mark.anyio
    async def test_sends_content_parts_and_a_response_format(self, monkeypatch):
        reply = []

        async def create(**kwargs):
            return reply[0]

        calls, make_message = _stub_openai(monkeypatch, create)
        reply.append(make_message('{"passed": true}'))
        response_format = {"type": "json_object"}
        judge = build_openai_judge_fn(model="gpt-5", response_format=response_format)
        content = [
            {"type": "text", "text": "grade this"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
        ]

        assert await judge(content, system_prompt="be strict") == '{"passed": true}'
        assert calls[0]["messages"] == [
            {"role": "system", "content": "be strict"},
            {"role": "user", "content": content},
        ]
        assert calls[0]["response_format"] == response_format
        assert judge.judge_config["response_format"] == response_format

    @pytest.mark.anyio
    async def test_judge_with_retries_asks_again_until_a_reply_parses(self, monkeypatch):
        from olmo_eval.common.scorers import llm_judge

        async def no_sleep(_):
            return None

        monkeypatch.setattr(llm_judge.asyncio, "sleep", no_sleep)
        replies = ["?", "VERDICT: 1"]
        sent = []

        async def judge(prompt, *, system_prompt=None):
            sent.append((prompt, system_prompt))
            return replies.pop(0)

        def parse(raw):
            return raw.split(": ")[1] if raw.startswith("VERDICT") else None

        verdict, raw = await llm_judge.judge_with_retries(
            judge, "grade", parse, limiter=asyncio.Semaphore(1), system_prompt="strict"
        )

        assert (verdict, raw) == ("1", "VERDICT: 1")
        assert sent == [("grade", "strict"), ("grade", "strict")]

    @pytest.mark.anyio
    async def test_judge_with_retries_gives_up_after_its_attempts(self, monkeypatch):
        from olmo_eval.common.scorers import llm_judge

        async def no_sleep(_):
            return None

        monkeypatch.setattr(llm_judge.asyncio, "sleep", no_sleep)

        async def judge(prompt, *, system_prompt=None):
            return "?"

        verdict, raw = await llm_judge.judge_with_retries(
            judge, "grade", lambda _: None, limiter=asyncio.Semaphore(1), attempts=2
        )

        assert (verdict, raw) == (None, "?")

    def test_records_a_response_format_only_when_set(self):
        assert "response_format" not in build_openai_judge_fn().judge_config

    def test_records_the_default_model(self):
        default_model = inspect.signature(build_openai_judge_fn).parameters["model"].default

        assert build_openai_judge_fn().judge_config["model"] == default_model

    def test_is_a_configured_judge(self):
        assert isinstance(build_openai_judge_fn(), ConfiguredJudgeFn)

    def test_a_plain_judge_function_is_not_a_configured_judge(self):
        async def custom_judge(prompt: str) -> str:
            return "A"

        assert not isinstance(custom_judge, ConfiguredJudgeFn)


class TestLLMJudgeScorerSerialization:
    """Tests for how LLM judge scorers describe their judge when serialized."""

    def test_records_the_judge_model(self):
        scorer = SimpleQAJudgeScorer(judge_fn=build_openai_judge_fn(model="gpt-5"))

        judge = scorer.to_dict()["judge_fn"]

        assert judge["name"] == "build_openai_judge_fn.<locals>.judge"
        assert judge["model"] == "gpt-5"

    def test_different_judge_models_serialize_differently(self):
        first = SimpleQAJudgeScorer(judge_fn=build_openai_judge_fn(model="gpt-4o-mini"))
        second = SimpleQAJudgeScorer(judge_fn=build_openai_judge_fn(model="gpt-5"))

        assert json.dumps(first.to_dict()) != json.dumps(second.to_dict())

    def test_custom_judge_is_recorded_by_name(self):
        async def custom_judge(prompt: str) -> str:
            return "A"

        assert SimpleQAJudgeScorer(judge_fn=custom_judge).to_dict()["judge_fn"] == {
            "name": custom_judge.__qualname__
        }

    def test_provider_judge_has_no_judge_fn(self):
        assert SimpleQAJudgeScorer(provider_name="judge").to_dict()["judge_fn"] is None


class TestSimpleQAJudgeScorer:
    """Tests for SimpleQAJudgeScorer."""

    @pytest.mark.anyio
    async def test_correct_grade_a(self):
        """Test parsing grade A (CORRECT)."""

        async def mock_judge(prompt: str) -> str:
            return "A"

        scorer = SimpleQAJudgeScorer(judge_fn=mock_judge)
        instance = Instance(question="What is 2+2?", gold_answer="4")
        output = LMOutput(text="4")
        output.extracted_answer = "4"

        score = await scorer.ascore_with_context(instance, output, ScoringContext())
        assert score == 1.0

    @pytest.mark.anyio
    async def test_incorrect_grade_b(self):
        """Test parsing grade B (INCORRECT)."""

        async def mock_judge(prompt: str) -> str:
            return "B"

        scorer = SimpleQAJudgeScorer(judge_fn=mock_judge)
        instance = Instance(question="What is 2+2?", gold_answer="4")
        output = LMOutput(text="5")
        output.extracted_answer = "5"

        score = await scorer.ascore_with_context(instance, output, ScoringContext())
        assert score == 0.0

    @pytest.mark.anyio
    async def test_not_attempted_grade_c(self):
        """Test parsing grade C (NOT_ATTEMPTED)."""

        async def mock_judge(prompt: str) -> str:
            return "C"

        scorer = SimpleQAJudgeScorer(judge_fn=mock_judge)
        instance = Instance(question="Complex question", gold_answer="Answer")
        output = LMOutput(text="I don't know")

        score = await scorer.ascore_with_context(instance, output, ScoringContext())
        assert score == 0.0

    @pytest.mark.anyio
    async def test_parse_correct_word(self):
        """Test parsing 'CORRECT' text."""

        async def mock_judge(prompt: str) -> str:
            return "The answer is CORRECT."

        scorer = SimpleQAJudgeScorer(judge_fn=mock_judge)
        instance = Instance(question="Q", gold_answer="A")
        output = LMOutput(text="A")

        score = await scorer.ascore_with_context(instance, output, ScoringContext())
        assert score == 1.0

    @pytest.mark.anyio
    async def test_parse_incorrect_word(self):
        """Test parsing 'INCORRECT' text."""

        async def mock_judge(prompt: str) -> str:
            return "The answer is INCORRECT."

        scorer = SimpleQAJudgeScorer(judge_fn=mock_judge)
        instance = Instance(question="Q", gold_answer="A")
        output = LMOutput(text="B")

        score = await scorer.ascore_with_context(instance, output, ScoringContext())
        assert score == 0.0

    @pytest.mark.anyio
    async def test_parse_not_attempted_text(self):
        """Test parsing 'NOT_ATTEMPTED' text."""

        async def mock_judge(prompt: str) -> str:
            return "NOT_ATTEMPTED"

        scorer = SimpleQAJudgeScorer(judge_fn=mock_judge)
        instance = Instance(question="Q", gold_answer="A")
        output = LMOutput(text="?")

        score = await scorer.ascore_with_context(instance, output, ScoringContext())
        assert score == 0.0

    def test_get_grade(self):
        """Test get_grade method."""
        scorer = SimpleQAJudgeScorer()
        assert scorer.get_grade("A") == "CORRECT"
        assert scorer.get_grade("B") == "INCORRECT"
        assert scorer.get_grade("C") == "NOT_ATTEMPTED"
        assert scorer.get_grade("CORRECT") == "CORRECT"
        assert scorer.get_grade("garbage") == "INCORRECT"

    def test_format_prompt(self):
        """Test prompt formatting."""
        scorer = SimpleQAJudgeScorer()
        instance = Instance(question="What is 2+2?", gold_answer="4")
        output = LMOutput(text="The answer is 4")
        output.extracted_answer = "4"

        prompt = scorer.format_judge_prompt(instance, output)
        assert "What is 2+2?" in prompt
        assert "4" in prompt
        assert "CORRECT" in prompt
        assert "INCORRECT" in prompt
        assert "NOT_ATTEMPTED" in prompt

    def test_scorer_name(self):
        """Test scorer name."""
        scorer = SimpleQAJudgeScorer()
        assert scorer.name == "simpleqa_judge"


class TestRubricJudgeScorer:
    """Tests for RubricJudgeScorer."""

    @pytest.mark.anyio
    async def test_extract_score(self):
        """Test extracting score from response."""

        async def mock_judge(prompt: str) -> str:
            return "The response is good. Score: 8"

        scorer = RubricJudgeScorer(judge_fn=mock_judge, max_score=10.0)
        instance = Instance(question="Q", gold_answer="A")
        output = LMOutput(text="Response")

        score = await scorer.ascore_with_context(instance, output, ScoringContext())
        assert score == 0.8

    @pytest.mark.anyio
    async def test_extract_decimal_score(self):
        """Test extracting decimal score."""

        async def mock_judge(prompt: str) -> str:
            return "Score: 7.5"

        scorer = RubricJudgeScorer(judge_fn=mock_judge, max_score=10.0)
        instance = Instance(question="Q", gold_answer="A")
        output = LMOutput(text="Response")

        score = await scorer.ascore_with_context(instance, output, ScoringContext())
        assert score == 0.75

    @pytest.mark.anyio
    async def test_custom_score_pattern(self):
        """Test custom score pattern."""

        async def mock_judge(prompt: str) -> str:
            return "Rating: 4/5"

        scorer = RubricJudgeScorer(
            judge_fn=mock_judge,
            score_pattern=r"Rating:\s*(\d+)/5",
            max_score=5.0,
        )
        instance = Instance(question="Q", gold_answer="A")
        output = LMOutput(text="Response")

        score = await scorer.ascore_with_context(instance, output, ScoringContext())
        assert score == 0.8

    @pytest.mark.anyio
    async def test_no_score_found(self):
        """Test when no score is found."""

        async def mock_judge(prompt: str) -> str:
            return "Good response with no score"

        scorer = RubricJudgeScorer(judge_fn=mock_judge, default_score=5.0, max_score=10.0)
        instance = Instance(question="Q", gold_answer="A")
        output = LMOutput(text="Response")

        score = await scorer.ascore_with_context(instance, output, ScoringContext())
        assert score == 0.5

    def test_get_raw_score(self):
        """Test getting raw score."""
        scorer = RubricJudgeScorer(max_score=10.0)
        raw = scorer.get_raw_score("The Score: 7 is given")
        assert raw == 7.0

    @pytest.mark.anyio
    async def test_custom_rubric(self):
        """Test with custom rubric."""
        custom_rubric = "Rate 0-5 based on accuracy"

        async def mock_judge(prompt: str) -> str:
            assert custom_rubric in prompt
            return "Score: 4"

        scorer = RubricJudgeScorer(judge_fn=mock_judge, rubric=custom_rubric, max_score=5.0)
        instance = Instance(question="Q", gold_answer="A")
        output = LMOutput(text="Response")

        score = await scorer.ascore_with_context(instance, output, ScoringContext())
        assert score == 0.8

    def test_format_prompt(self):
        """Test prompt formatting."""
        scorer = RubricJudgeScorer(rubric="Custom rubric here")
        instance = Instance(question="Test question", gold_answer="Expected answer")
        output = LMOutput(text="Model response")

        prompt = scorer.format_judge_prompt(instance, output)
        assert "Test question" in prompt
        assert "Expected answer" in prompt
        assert "Model response" in prompt
        assert "Custom rubric here" in prompt

    def test_default_rubric_used(self):
        """Test default rubric when none provided."""
        scorer = RubricJudgeScorer(max_score=10.0)
        instance = Instance(question="Q", gold_answer="A")
        output = LMOutput(text="Response")

        prompt = scorer.format_judge_prompt(instance, output)
        assert "10" in prompt
        assert "0" in prompt

    def test_scorer_name(self):
        """Test scorer name."""
        scorer = RubricJudgeScorer()
        assert scorer.name == "rubric_judge"


class TestLLMJudgeScorerIntegration:
    """Integration tests for LLM judge scorers."""

    @pytest.mark.anyio
    async def test_simpleqa_full_flow(self):
        """Test complete SimpleQA evaluation flow."""
        responses = []

        async def tracking_judge(prompt: str) -> str:
            responses.append(prompt)
            if "AI Assistant's Answer: 4" in prompt:
                return "A"
            return "B"

        scorer = SimpleQAJudgeScorer(judge_fn=tracking_judge)
        ctx = ScoringContext()

        # Correct answer
        instance1 = Instance(question="2+2=?", gold_answer="4")
        output1 = LMOutput(text="4")
        output1.extracted_answer = "4"
        score1 = await scorer.ascore_with_context(instance1, output1, ctx)

        # Incorrect answer
        instance2 = Instance(question="2+2=?", gold_answer="4")
        output2 = LMOutput(text="5")
        output2.extracted_answer = "5"
        score2 = await scorer.ascore_with_context(instance2, output2, ctx)

        assert score1 == 1.0
        assert score2 == 0.0
        assert len(responses) == 2

    @pytest.mark.anyio
    async def test_rubric_different_scales(self):
        """Test rubric scorer with different score scales."""
        scales = [(5.0, 4), (10.0, 8), (100.0, 80)]
        ctx = ScoringContext()

        for max_score, raw_score in scales:

            async def make_judge(rs):
                return f"Score: {rs}"

            scorer = RubricJudgeScorer(
                judge_fn=lambda p, rs=raw_score: make_judge(rs), max_score=max_score
            )
            instance = Instance(question="Q", gold_answer="A")
            output = LMOutput(text="R")

            score = await scorer.ascore_with_context(instance, output, ctx)
            assert score == 0.8
