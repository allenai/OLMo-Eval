"""Tests for the BFCL multi-turn rollout scaffold."""

from __future__ import annotations

import pytest

from olmo_eval.common.types import LMOutput, LMRequest, RequestType, SamplingParams
from olmo_eval.common.types.tools import ToolCall
from olmo_eval.harness.scaffolds import get_scaffold
from olmo_eval.harness.scaffolds.bfcl_multi_turn import (
    ADDITIONAL_FUNCTION_PROMPT,
    MAXIMUM_STEP_LIMIT,
)
from olmo_eval.inference.base import InferenceProvider
from olmo_eval.inference.errors import CONTEXT_OVERFLOW_KEY, REQUEST_ERROR_KEY

FS_CONFIG = {
    "GorillaFileSystem": {
        "root": {"workspace": {"type": "directory", "contents": {}}},
    }
}


class ScriptedProvider(InferenceProvider):
    """Returns prepared replies in order, recording what it was asked."""

    supports_tools = True

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
        reply = self.replies.pop(0) if self.replies else LMOutput(text="Nothing further.")
        return [[reply]]


def request_for(turns, **payload) -> LMRequest:
    base = {
        "turns": turns,
        "initial_config": FS_CONFIG,
        "involved_classes": ["GorillaFileSystem"],
        "call_source": "text",
        "language": "python",
    }
    base.update(payload)
    return LMRequest(request_type=RequestType.CHAT, messages=(), metadata=base)


async def run(provider, request, **kwargs):
    return await get_scaffold("bfcl_multi_turn").run(
        provider, None, request, SamplingParams(), **kwargs
    )


@pytest.mark.anyio
async def test_a_turn_ends_when_the_model_stops_calling() -> None:
    provider = ScriptedProvider(
        [LMOutput(text="[mkdir(dir_name='temp')]"), LMOutput(text="All done.")]
    )

    result = await run(provider, request_for([[{"role": "user", "content": "Make temp."}]]))

    assert result.final_output.extracted_answer == [[[{"mkdir": {"dir_name": "temp"}}]]]
    assert len(provider.requests) == 2


@pytest.mark.anyio
async def test_several_steps_in_one_turn_are_kept_separately() -> None:
    provider = ScriptedProvider(
        [
            LMOutput(text="[mkdir(dir_name='temp')]"),
            LMOutput(text="[cd(folder='temp')]"),
            LMOutput(text="Done."),
        ]
    )

    result = await run(provider, request_for([[{"role": "user", "content": "Go."}]]))

    assert result.final_output.extracted_answer == [
        [[{"mkdir": {"dir_name": "temp"}}], [{"cd": {"folder": "temp"}}]]
    ]


@pytest.mark.anyio
async def test_execution_results_come_back_to_the_model() -> None:
    provider = ScriptedProvider([LMOutput(text="[ls()]"), LMOutput(text="Done.")])

    await run(provider, request_for([[{"role": "user", "content": "List."}]]))

    # The result comes back as a tool message named after the call that produced
    # it, as the reference implementation's handler for local models sends it.
    last = provider.requests[-1].messages[-1]
    assert last["role"] == "tool"
    assert last["name"] == "ls()"
    assert "current_directory_content" in last["content"]


@pytest.mark.anyio
async def test_each_result_is_paired_with_its_call() -> None:
    provider = ScriptedProvider(
        [LMOutput(text="[mkdir(dir_name='a'), mkdir(dir_name='b')]"), LMOutput(text="Done.")]
    )

    await run(provider, request_for([[{"role": "user", "content": "Make two."}]]))

    results = [m for m in provider.requests[-1].messages if m["role"] == "tool"]
    assert [m["name"] for m in results] == ["mkdir(dir_name='a')", "mkdir(dir_name='b')"]


@pytest.mark.anyio
async def test_state_carries_from_one_turn_to_the_next() -> None:
    provider = ScriptedProvider(
        [
            LMOutput(text="[mkdir(dir_name='temp')]"),
            LMOutput(text="Done."),
            LMOutput(text="[cd(folder='temp')]"),
            LMOutput(text="Done."),
        ]
    )

    await run(
        provider,
        request_for(
            [
                [{"role": "user", "content": "Make temp."}],
                [{"role": "user", "content": "Enter it."}],
            ]
        ),
    )

    # cd would fail had the directory not survived the first turn.
    assert "Error" not in provider.requests[-1].messages[-1]["content"]


@pytest.mark.anyio
async def test_a_turn_the_model_declines_records_no_calls() -> None:
    provider = ScriptedProvider([LMOutput(text="I need to know which file.")])

    result = await run(provider, request_for([[{"role": "user", "content": "Move one."}]]))

    assert result.final_output.extracted_answer == [[]]


@pytest.mark.anyio
async def test_held_back_functions_are_offered_at_their_turn() -> None:
    schema = {
        "type": "function",
        "function": {"name": "sort", "description": "Sort a file.", "parameters": {}},
    }
    provider = ScriptedProvider(
        [LMOutput(text="Done."), LMOutput(text="[sort(file_name='a.txt')]"), LMOutput(text="ok")]
    )

    await run(
        provider,
        request_for(
            [[{"role": "user", "content": "Do something."}], []],
            missed_function={"1": [schema]},
            missed_function_docs={"1": "[{'name': 'sort', 'description': 'Sort a file.'}]"},
        ),
    )

    contents = [m["content"] for r in provider.requests for m in r.messages if m["role"] == "user"]
    assert any(ADDITIONAL_FUNCTION_PROMPT in c for c in contents)
    # A prompted model has no tool list, so the offer writes the functions out
    # the way the system prompt wrote the rest, not as tool-API schemas.
    assert any("{'name': 'sort'" in c for c in contents)
    assert not any("'type': 'function'" in c for c in contents)
    # Sending schemas would also reach a server started for plain prompting,
    # which rejects a request carrying tools.
    assert all(r.tools is None for r in provider.requests)


@pytest.mark.anyio
async def test_a_held_back_function_reaches_the_tool_list_when_calling_natively() -> None:
    schema = {
        "type": "function",
        "function": {"name": "sort", "description": "Sort a file.", "parameters": {}},
    }
    provider = ScriptedProvider([LMOutput(text="Done."), LMOutput(text="ok"), LMOutput(text="ok")])

    await run(
        provider,
        request_for(
            [[{"role": "user", "content": "Do something."}], []],
            call_source="tool_calls",
            missed_function={"1": [schema]},
        ),
    )

    assert provider.requests[0].tools is None
    assert [t.name for t in provider.requests[-1].tools or ()] == ["sort"]


@pytest.mark.anyio
async def test_the_step_budget_stops_a_model_that_never_finishes() -> None:
    provider = ScriptedProvider([LMOutput(text="[ls()]") for _ in range(20)])

    result = await run(provider, request_for([[{"role": "user", "content": "Go."}]]), max_steps=3)

    assert result.max_turns_reached
    assert result.final_output.metadata["bfcl_step_budget_exhausted"] is True
    # The turn stops once more than the budget has run, as the reference counts.
    assert len(provider.requests) == 4


@pytest.mark.anyio
async def test_the_default_budget_allows_one_step_past_the_limit() -> None:
    provider = ScriptedProvider([LMOutput(text="[ls()]") for _ in range(30)])

    await run(provider, request_for([[{"role": "user", "content": "Go."}]]))

    assert len(provider.requests) == MAXIMUM_STEP_LIMIT + 1


@pytest.mark.anyio
async def test_a_budget_stop_before_the_last_turn_ends_the_rollout() -> None:
    provider = ScriptedProvider([LMOutput(text="[ls()]") for _ in range(10)])
    turns = [[{"role": "user", "content": "Go."}], [{"role": "user", "content": "Again."}]]

    result = await run(provider, request_for(turns), max_steps=2)

    assert len(result.final_output.extracted_answer) == 1


@pytest.mark.anyio
async def test_native_tool_calls_are_read_and_answered_as_tool_messages() -> None:
    call = ToolCall.create("call_1", "ls", {})
    provider = ScriptedProvider([LMOutput(text="", tool_calls=[call]), LMOutput(text="Done.")])

    result = await run(
        provider, request_for([[{"role": "user", "content": "List."}]], call_source="tool_calls")
    )

    assert result.final_output.extracted_answer == [[[{"ls": {}}]]]
    last = provider.requests[-1].messages
    assert last[-1]["role"] == "tool"
    assert last[-1]["tool_call_id"] == "call_1"


@pytest.mark.anyio
async def test_a_request_without_a_payload_is_refused() -> None:
    provider = ScriptedProvider([])

    with pytest.raises(ValueError, match="turns"):
        await run(provider, LMRequest(request_type=RequestType.CHAT, messages=()))


@pytest.mark.anyio
async def test_a_prompted_offer_without_written_out_functions_is_refused() -> None:
    schema = {
        "type": "function",
        "function": {"name": "sort", "description": "Sort a file.", "parameters": {}},
    }
    provider = ScriptedProvider([LMOutput(text="Done.")])

    with pytest.raises(ValueError, match="missed_function_docs"):
        await run(
            provider,
            request_for(
                [[{"role": "user", "content": "Do something."}], []],
                missed_function={"1": [schema]},
            ),
        )


OVERFLOW = LMOutput(
    text="",
    metadata={REQUEST_ERROR_KEY: "context overflow: too long", CONTEXT_OVERFLOW_KEY: True},
)


@pytest.mark.anyio
async def test_an_overflow_is_logged_so_it_can_be_counted(caplog) -> None:  # noqa: ANN001
    provider = ScriptedProvider([OVERFLOW])

    with caplog.at_level("WARNING"):
        await run(provider, request_for([[{"role": "user", "content": "Go."}]]))

    assert "outgrew the context window" in caplog.text


@pytest.mark.anyio
async def test_a_conversation_that_outgrows_the_window_ends_the_rollout() -> None:
    provider = ScriptedProvider([LMOutput(text="[mkdir(dir_name='temp')]"), OVERFLOW])
    turns = [
        [{"role": "user", "content": "Make temp."}],
        [{"role": "user", "content": "Now list it."}],
    ]

    result = await run(provider, request_for(turns))

    # The overflow is recorded for the scorer, and nothing after it is asked.
    assert result.final_output.metadata["bfcl_context_overflow"] == "context overflow: too long"
    assert len(provider.requests) == 2
    # The rollout's own reply carries no provider error, so the runner scores it
    # instead of setting the instance aside as failed.
    assert REQUEST_ERROR_KEY not in result.final_output.metadata


@pytest.mark.anyio
async def test_any_other_failed_step_still_fails_the_instance() -> None:
    failed = LMOutput(text="", metadata={REQUEST_ERROR_KEY: "server unavailable"})
    provider = ScriptedProvider([failed])

    with pytest.raises(RuntimeError, match="server unavailable"):
        await run(provider, request_for([[{"role": "user", "content": "Go."}]]))


@pytest.mark.anyio
async def test_a_step_with_no_reply_fails_with_a_reason() -> None:
    class SilentProvider(ScriptedProvider):
        async def agenerate(self, requests, sampling_params=None):
            return [[]]

    with pytest.raises(RuntimeError, match="no output"):
        await run(SilentProvider([]), request_for([[{"role": "user", "content": "Go."}]]))


@pytest.mark.anyio
async def test_a_native_rollout_refuses_a_provider_that_would_drop_its_tools() -> None:
    class ToollessProvider(ScriptedProvider):
        supports_tools = False

    request = request_for([[{"role": "user", "content": "Go."}]], call_source="tool_calls")

    with pytest.raises(ValueError, match="does not support tool requests"):
        await run(ToollessProvider([]), request)


@pytest.mark.anyio
async def test_a_prompted_rollout_needs_no_tool_support() -> None:
    class ToollessProvider(ScriptedProvider):
        supports_tools = False

    result = await run(
        ToollessProvider([LMOutput(text="Done.")]),
        request_for([[{"role": "user", "content": "Go."}]]),
    )

    assert result.final_output.extracted_answer == [[]]


@pytest.mark.anyio
async def test_a_call_with_malformed_arguments_is_recorded_as_text_and_the_rollout_goes_on() -> (
    None
):
    """vLLM rejects a history whose tool-call arguments are not a JSON object (400), which
    would drop the entry from the run; the scaffold keeps the step as text instead."""
    from olmo_eval.common.types.tools import Function

    bad_call = ToolCall(id="c1", function=Function(name="mkdir", arguments="{dir_name: temp"))
    provider = ScriptedProvider(
        [
            LMOutput(text="", tool_calls=[bad_call]),
            LMOutput(text="Done."),
            LMOutput(text="ok"),
        ]
    )
    result = await run(
        provider,
        request_for(
            [
                [{"role": "user", "content": "Make a temp directory."}],
                [{"role": "user", "content": "Now list it."}],
            ],
            call_source="tool_calls",
        ),
    )
    assert result.final_output.extracted_answer[0] == []  # the malformed step reads as no calls
    history = provider.requests[1].messages
    assistant = [m for m in history if m["role"] == "assistant"]
    assert assistant and "tool_calls" not in assistant[0]
    assert "mkdir({dir_name: temp" in assistant[0]["content"]
