"""Tests for the scripted multi-turn chat scaffold."""

from __future__ import annotations

import pytest

from olmo_eval.common.multi_turn import turn_replies
from olmo_eval.common.types import (
    FOLLOW_UP_TURNS_KEY,
    TURNS_KEY,
    LMOutput,
    LMRequest,
    RequestType,
    SamplingParams,
)
from olmo_eval.harness.config import HarnessConfig
from olmo_eval.harness.presets import get_harness_preset
from olmo_eval.harness.scaffolds import get_scaffold
from olmo_eval.inference.base import InferenceProvider
from olmo_eval.inference.errors import CONTEXT_OVERFLOW_KEY, REQUEST_ERROR_KEY


class ScriptedProvider(InferenceProvider):
    """Returns prepared replies in order, recording what it was asked."""

    def __init__(self, replies: list[LMOutput]) -> None:
        super().__init__("scripted")
        self.replies = list(replies)
        self.requests: list[LMRequest] = []

    def generate(self, requests, sampling_params=None):  # pragma: no cover - unused
        raise NotImplementedError

    def logprobs(self, requests, sampling_params=None):  # pragma: no cover - unused
        raise NotImplementedError

    async def agenerate(self, requests, sampling_params=None):
        self.requests.append(requests[0])
        return [[self.replies.pop(0)]]


def user(content: str) -> dict[str, str]:
    return {"role": "user", "content": content}


def request_for(*turns: str, system_prompt: str | None = None) -> LMRequest:
    return LMRequest(
        request_type=RequestType.CHAT,
        messages=(user(turns[0]),),
        system_prompt=system_prompt,
        metadata={FOLLOW_UP_TURNS_KEY: [user(turn) for turn in turns[1:]]},
    )


async def run(provider: InferenceProvider, request: LMRequest, config: HarnessConfig | None = None):
    return await get_scaffold("multi_turn_chat").run(
        provider, config or HarnessConfig(name="test"), request, SamplingParams()
    )


OVERFLOW = LMOutput(
    text="",
    metadata={REQUEST_ERROR_KEY: "context overflow: too long", CONTEXT_OVERFLOW_KEY: True},
)


@pytest.mark.anyio
async def test_each_turn_is_asked_with_the_models_earlier_replies() -> None:
    provider = ScriptedProvider([LMOutput(text="one"), LMOutput(text="two")])

    await run(provider, request_for("First?", "Second?", system_prompt="Be brief."))

    assert [r.messages for r in provider.requests] == [
        (user("First?"),),
        (user("First?"), {"role": "assistant", "content": "one"}, user("Second?")),
    ]
    assert all(r.system_prompt == "Be brief." for r in provider.requests)


@pytest.mark.anyio
async def test_the_harness_system_prompt_takes_precedence() -> None:
    provider = ScriptedProvider([LMOutput(text="one"), LMOutput(text="two")])
    config = HarnessConfig(name="test", system_prompt="From the harness.")

    await run(provider, request_for("First?", "Second?", system_prompt="From the task."), config)

    assert [r.system_prompt for r in provider.requests] == ["From the harness."] * 2


@pytest.mark.anyio
async def test_replies_are_returned_in_turn_order() -> None:
    provider = ScriptedProvider([LMOutput(text=t) for t in ("a", "b", "c")])

    result = await run(provider, request_for("1", "2", "3"))

    assert turn_replies(result.final_output) == ["a", "b", "c"]
    assert result.final_output.text == "c"
    assert result.metadata == {"turns": 3}


def turn(text: str, finish_reason: str | None = "stop", **tokens: int) -> LMOutput:
    return LMOutput(text=text, metadata={"finish_reason": finish_reason, **tokens})


@pytest.mark.anyio
async def test_each_turn_is_recorded_with_its_finish_reason_and_tokens() -> None:
    provider = ScriptedProvider(
        [
            turn("a", "length", prompt_tokens=10, completion_tokens=500),
            turn("b", prompt_tokens=520, completion_tokens=7),
            LMOutput(text="c"),
        ]
    )

    result = await run(provider, request_for("1", "2", "3"))

    assert result.final_output.metadata[TURNS_KEY] == [
        {"text": "a", "finish_reason": "length", "prompt_tokens": 10, "completion_tokens": 500},
        {"text": "b", "finish_reason": "stop", "prompt_tokens": 520, "completion_tokens": 7},
        {"text": "c", "finish_reason": None},
    ]


@pytest.mark.anyio
async def test_the_conversation_reports_the_tokens_of_every_turn() -> None:
    provider = ScriptedProvider(
        [
            turn("a", prompt_tokens=10, completion_tokens=500),
            turn("b", prompt_tokens=520, completion_tokens=7),
        ]
    )

    result = await run(provider, request_for("1", "2"))

    metadata = result.final_output.metadata
    assert (metadata["prompt_tokens"], metadata["completion_tokens"]) == (530, 507)


@pytest.mark.anyio
async def test_a_turn_without_token_counts_leaves_the_conversation_total_unknown() -> None:
    provider = ScriptedProvider([turn("a"), turn("b", prompt_tokens=520, completion_tokens=7)])

    result = await run(provider, request_for("1", "2"))

    assert "prompt_tokens" not in result.final_output.metadata
    assert "completion_tokens" not in result.final_output.metadata


@pytest.mark.anyio
async def test_a_turn_at_the_token_limit_marks_the_conversation_truncated() -> None:
    provider = ScriptedProvider([turn("a", "length"), turn("b", "stop")])

    result = await run(provider, request_for("1", "2"))

    assert result.final_output.metadata["finish_reason"] == "length"
    assert result.final_output.text == "b"


@pytest.mark.anyio
async def test_a_conversation_cut_short_by_the_context_is_truncated() -> None:
    provider = ScriptedProvider(
        [
            turn("a", prompt_tokens=10, completion_tokens=5),
            turn("b", prompt_tokens=20, completion_tokens=6),
            OVERFLOW,
        ]
    )

    result = await run(provider, request_for("1", "2", "3"))

    metadata = result.final_output.metadata
    assert metadata["finish_reason"] == "length"
    assert (metadata["prompt_tokens"], metadata["completion_tokens"]) == (30, 11)
    assert [t["finish_reason"] for t in metadata[TURNS_KEY]] == ["stop", "stop"]


@pytest.mark.anyio
async def test_an_overflow_on_the_first_turn_is_truncated() -> None:
    result = await run(ScriptedProvider([OVERFLOW]), request_for("1", "2"))

    assert result.final_output.metadata["finish_reason"] == "length"
    assert "completion_tokens" not in result.final_output.metadata


@pytest.mark.anyio
async def test_the_last_turns_finish_reason_stands_when_no_turn_hit_the_limit() -> None:
    provider = ScriptedProvider([turn("a", "stop"), turn("b", None)])

    result = await run(provider, request_for("1", "2"))

    assert result.final_output.metadata["finish_reason"] is None


@pytest.mark.anyio
async def test_a_request_without_follow_ups_is_one_turn() -> None:
    provider = ScriptedProvider([LMOutput(text="only")])

    result = await run(provider, LMRequest(request_type=RequestType.CHAT, messages=(user("Hi"),)))

    assert turn_replies(result.final_output) == ["only"]
    assert len(provider.requests) == 1


@pytest.mark.anyio
async def test_an_overflow_ends_the_conversation_without_failing_it() -> None:
    provider = ScriptedProvider([LMOutput(text="a"), OVERFLOW])

    result = await run(provider, request_for("1", "2", "3"))

    assert len(provider.requests) == 2
    assert turn_replies(result.final_output) == ["a"]
    assert REQUEST_ERROR_KEY not in result.final_output.metadata


@pytest.mark.anyio
async def test_an_overflow_on_the_first_turn_leaves_no_replies() -> None:
    provider = ScriptedProvider([OVERFLOW])

    result = await run(provider, request_for("1", "2"))

    assert len(provider.requests) == 1
    assert turn_replies(result.final_output) == []
    assert result.final_output.text == ""
    assert REQUEST_ERROR_KEY not in result.final_output.metadata


@pytest.mark.anyio
async def test_any_other_failed_turn_fails_the_instance() -> None:
    failed = LMOutput(text="", metadata={REQUEST_ERROR_KEY: "server unavailable"})
    provider = ScriptedProvider([LMOutput(text="a"), failed])

    with pytest.raises(RuntimeError, match="server unavailable"):
        await run(provider, request_for("1", "2"))


def test_preset_runs_the_scaffold() -> None:
    assert get_harness_preset("multi_turn_chat").scaffold == "multi_turn_chat"
