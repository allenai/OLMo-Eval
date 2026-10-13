"""Tests for the OpenAI Agents scaffold (requires the optional agents deps)."""

import contextlib
import logging
from types import SimpleNamespace
from typing import Any

import pytest

from olmo_eval.common.types import LMRequest, RequestType
from olmo_eval.harness.config import HarnessConfig
from olmo_eval.harness.scaffolds.openai_agents import (
    DEFAULT_CHAT_TEMPLATE_KWARGS,
    FORCED_FINAL_ANSWER_INSTRUCTION,
    OpenAIAgentsScaffold,
    _chat_completions_model_class,
    _make_tool_error_formatter,
    _mirror_vllm_reasoning,
)

pytest.importorskip("agents")


def _function_call(name: str, arguments: str = "{}"):
    from openai.types.responses import ResponseFunctionToolCall

    return ResponseFunctionToolCall(
        arguments=arguments,
        call_id=f"call_{name}",
        name=name,
        type="function_call",
    )


def _tool_turn_items(agent, output: str = "Search result text"):
    from agents.items import ToolCallItem, ToolCallOutputItem

    call = _function_call("search", '{"query":"olmo"}')
    return [
        ToolCallItem(agent=agent, raw_item=call),
        ToolCallOutputItem(
            agent=agent,
            raw_item={
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": output,
            },
            output=output,
        ),
    ]


def _reasoning_item(agent, *summary: str, content: tuple[str, ...] = ()):
    from agents.items import ReasoningItem
    from openai.types.responses import ResponseReasoningItem
    from openai.types.responses.response_reasoning_item import Content, Summary

    raw = ResponseReasoningItem(
        id="rs_1",
        type="reasoning",
        summary=[Summary(text=text, type="summary_text") for text in summary],
        content=[Content(text=text, type="reasoning_text") for text in content] or None,
    )
    return ReasoningItem(agent=agent, raw_item=raw)


def _message_item(agent, text: str):
    from agents.items import MessageOutputItem
    from openai.types.responses import ResponseOutputMessage, ResponseOutputText

    raw = ResponseOutputMessage(
        id="msg_1",
        type="message",
        role="assistant",
        status="completed",
        content=[ResponseOutputText(text=text, type="output_text", annotations=[])],
    )
    return MessageOutputItem(agent=agent, raw_item=raw)


def _refusal_message_item(agent, refusal: str):
    from agents.items import MessageOutputItem
    from openai.types.responses import ResponseOutputMessage, ResponseOutputRefusal

    raw = ResponseOutputMessage(
        id="msg_1",
        type="message",
        role="assistant",
        status="completed",
        content=[ResponseOutputRefusal(refusal=refusal, type="refusal")],
    )
    return MessageOutputItem(agent=agent, raw_item=raw)


class _FakeRunData:
    def __init__(self, *, new_items, final_output=None):
        self.new_items = new_items
        self.final_output = final_output


def _max_turns_exceeded_with_run_data(run_data, message: str = "Max turns (1) exceeded"):
    from agents.exceptions import MaxTurnsExceeded

    exc = MaxTurnsExceeded(message)
    exc.run_data = run_data
    return exc


def _agent_request():
    return LMRequest(
        request_type=RequestType.CHAT,
        messages=({"role": "user", "content": "What did you find?"},),
    )


def _patch_scaffold_agent(monkeypatch, agent):
    import agents

    monkeypatch.setattr(
        OpenAIAgentsScaffold,
        "_get_or_create_agent",
        lambda self, provider, config, sandbox_manager=None: agent,
    )
    monkeypatch.setattr(agents, "trace", lambda *args, **kwargs: contextlib.nullcontext())


class TestOpenAIAgentsMaxTurns:
    @pytest.mark.anyio
    async def test_max_turns_preserves_trajectory_and_forces_final_answer(self, monkeypatch):
        from agents import Agent, Runner

        agent = Agent(name="test-agent", instructions="Use tools.")
        partial_run_data = _FakeRunData(
            new_items=_tool_turn_items(agent, output="Tool result before cap"),
        )
        run_calls = []

        def fail_run_streamed(**kwargs):
            raise AssertionError("Runner.run_streamed should not be used")

        async def fake_run(**kwargs):
            run_calls.append(kwargs)
            if len(run_calls) == 1:
                raise _max_turns_exceeded_with_run_data(partial_run_data)
            return SimpleNamespace(final_output="Forced final answer")

        _patch_scaffold_agent(monkeypatch, agent)
        monkeypatch.setattr(Runner, "run_streamed", staticmethod(fail_run_streamed))
        monkeypatch.setattr(Runner, "run", staticmethod(fake_run))

        result = await OpenAIAgentsScaffold().run(
            provider=SimpleNamespace(),
            config=HarnessConfig(name="test", max_turns=1),
            request=_agent_request(),
            enable_compaction=False,
        )

        assert result.max_turns_reached is True
        assert result.error is None
        assert result.final_output.text == "Forced final answer"
        assert result.trajectory is not None
        assert result.trajectory.total_tool_calls == 1
        assert result.trajectory.tool_result_sequence[0].content == "Tool result before cap"
        assert result.trajectory.tool_result_sequence[0].tool_call_id == "call_search"

        assert len(run_calls) == 2
        assert run_calls[0]["max_turns"] == 1
        final_call = run_calls[1]
        assert final_call["max_turns"] == 1
        assert final_call["starting_agent"].tools == []
        assert final_call["starting_agent"].handoffs == []
        assert final_call["starting_agent"].mcp_servers == []
        assert final_call["starting_agent"].model_settings.tool_choice == "none"
        assert final_call["input"][-1] == {
            "role": "user",
            "content": FORCED_FINAL_ANSWER_INSTRUCTION,
        }

    @pytest.mark.anyio
    async def test_forced_final_answer_failure_falls_back_to_old_result(self, monkeypatch):
        from agents import Agent, Runner

        agent = Agent(name="test-agent", instructions="Use tools.")
        partial_run_data = _FakeRunData(
            new_items=_tool_turn_items(agent),
        )
        run_calls = []

        def fail_run_streamed(**kwargs):
            raise AssertionError("Runner.run_streamed should not be used")

        async def fake_run(**kwargs):
            run_calls.append(kwargs)
            if len(run_calls) == 1:
                raise _max_turns_exceeded_with_run_data(partial_run_data)
            raise RuntimeError("connection failed")

        _patch_scaffold_agent(monkeypatch, agent)
        monkeypatch.setattr(Runner, "run_streamed", staticmethod(fail_run_streamed))
        monkeypatch.setattr(Runner, "run", staticmethod(fake_run))

        result = await OpenAIAgentsScaffold().run(
            provider=SimpleNamespace(),
            config=HarnessConfig(name="test", max_turns=1),
            request=_agent_request(),
            enable_compaction=False,
        )

        assert result.max_turns_reached is True
        assert result.final_output.text == "[Max turns exceeded]"
        assert result.trajectory is not None
        assert result.trajectory.turns == ()
        assert result.error == "Max turns (1) exceeded"
        assert len(run_calls) == 2

    @pytest.mark.anyio
    async def test_normal_completion_does_not_force_final_answer(self, monkeypatch):
        from agents import Agent, Runner

        agent = Agent(name="test-agent", instructions="Use tools.")
        run_result = _FakeRunData(
            new_items=_tool_turn_items(agent, output="Normal tool result"),
            final_output="Normal final answer",
        )
        run_calls = []

        def fail_run_streamed(**kwargs):
            raise AssertionError("Runner.run_streamed should not be used")

        async def fake_run(**kwargs):
            run_calls.append(kwargs)
            return run_result

        _patch_scaffold_agent(monkeypatch, agent)
        monkeypatch.setattr(Runner, "run_streamed", staticmethod(fail_run_streamed))
        monkeypatch.setattr(Runner, "run", staticmethod(fake_run))

        result = await OpenAIAgentsScaffold().run(
            provider=SimpleNamespace(),
            config=HarnessConfig(name="test", max_turns=3),
            request=_agent_request(),
            enable_compaction=False,
        )

        assert result.max_turns_reached is False
        assert result.error is None
        assert result.final_output.text == "Normal final answer"
        assert result.trajectory is not None
        assert result.trajectory.total_tool_calls == 1
        assert result.trajectory.tool_result_sequence[0].content == "Normal tool result"
        assert len(run_calls) == 1


class TestReasoningItems:
    """The SDK emits reasoning as a ReasoningItem ahead of the turn it belongs to."""

    def _convert(self, items):
        return OpenAIAgentsScaffold()._convert_trajectory(_FakeRunData(new_items=items))

    def test_reasoning_lands_on_the_following_message_turn(self):
        from agents import Agent

        agent = Agent(name="test-agent")
        trajectory = self._convert(
            [_reasoning_item(agent, "Think first."), _message_item(agent, "Final answer.")]
        )

        assert [turn.role for turn in trajectory.turns] == ["assistant"]
        turn = trajectory.turns[0]
        assert turn.content == "Final answer."
        assert turn.reasoning == "Think first."
        assert "Think first." not in turn.content

    def test_reasoning_lands_on_the_following_tool_call_turn(self):
        from agents import Agent

        agent = Agent(name="test-agent")
        trajectory = self._convert(
            [_reasoning_item(agent, "Plan the search."), *_tool_turn_items(agent)]
        )

        assert [turn.role for turn in trajectory.turns] == ["assistant", "tool"]
        assert trajectory.turns[0].has_tool_calls
        assert trajectory.turns[0].reasoning == "Plan the search."
        assert trajectory.turns[1].reasoning is None

    def test_trailing_reasoning_is_kept_on_an_empty_assistant_turn(self):
        from agents import Agent

        agent = Agent(name="test-agent")
        trajectory = self._convert(
            [*_tool_turn_items(agent), _reasoning_item(agent, "Cut off here.")]
        )

        assert [turn.role for turn in trajectory.turns] == ["assistant", "tool", "assistant"]
        last = trajectory.turns[-1]
        assert last.content == ""
        assert not last.has_tool_calls
        assert last.reasoning == "Cut off here."

    def test_reasoning_before_a_refusal_stays_on_that_response(self):
        from agents import Agent

        agent = Agent(name="test-agent")
        trajectory = self._convert(
            [
                _reasoning_item(agent, "Think first."),
                _refusal_message_item(agent, "I cannot help with that."),
                _message_item(agent, "Final answer."),
            ]
        )

        assert [turn.role for turn in trajectory.turns] == ["assistant", "assistant"]
        assert trajectory.turns[0].content == ""
        assert trajectory.turns[0].reasoning == "Think first."
        assert trajectory.turns[1].content == "Final answer."
        assert trajectory.turns[1].reasoning is None

    def test_pending_reasoning_is_flushed_before_a_tool_output(self):
        from agents import Agent

        agent = Agent(name="test-agent")
        _, tool_output = _tool_turn_items(agent)
        trajectory = self._convert(
            [_reasoning_item(agent, "Orphaned."), tool_output, _message_item(agent, "Final.")]
        )

        assert [turn.role for turn in trajectory.turns] == ["assistant", "tool", "assistant"]
        assert trajectory.turns[0].content == ""
        assert trajectory.turns[0].reasoning == "Orphaned."
        assert trajectory.turns[2].reasoning is None

    def test_content_parts_supersede_the_summary(self):
        from agents import Agent

        agent = Agent(name="test-agent")
        item = _reasoning_item(agent, "summary", content=("block one", "block two"))
        trajectory = self._convert([item, _message_item(agent, "Done.")])

        assert trajectory.turns[0].reasoning == "block one\n\nblock two"

    def test_turns_without_reasoning_serialize_unchanged(self):
        from agents import Agent

        agent = Agent(name="test-agent")
        trajectory = self._convert([*_tool_turn_items(agent), _message_item(agent, "Done.")])

        assert len(trajectory.turns) == 3
        for turn in trajectory.turns:
            assert turn.reasoning is None
            assert "reasoning" not in turn.to_dict()

    @pytest.mark.anyio
    async def test_run_keeps_reasoning_in_the_harness_result(self, monkeypatch):
        from agents import Agent, Runner

        agent = Agent(name="test-agent", instructions="Use tools.")
        run_result = _FakeRunData(
            new_items=[_reasoning_item(agent, "Think first."), _message_item(agent, "Final.")],
            final_output="Final.",
        )

        async def fake_run(**kwargs):
            return run_result

        _patch_scaffold_agent(monkeypatch, agent)
        monkeypatch.setattr(Runner, "run", staticmethod(fake_run))

        result = await OpenAIAgentsScaffold().run(
            provider=SimpleNamespace(),
            config=HarnessConfig(name="test", max_turns=3),
            request=_agent_request(),
            enable_compaction=False,
        )

        assert result.final_output.text == "Final."
        assert result.trajectory is not None
        assert result.trajectory.turns[0].reasoning == "Think first."
        assert result.trajectory.turns[0].content == "Final."


def _stub_openai_client():
    return SimpleNamespace(base_url="http://localhost:8000/v1")


def _self_hosted_provider(chat_template_kwargs=None, client=None, model_name="test-model"):
    """Provider shaped like VLLMServerProvider: exposes ``chat_template_kwargs``."""
    openai_client = client if client is not None else _stub_openai_client()
    return SimpleNamespace(
        model_name=model_name,
        chat_template_kwargs=chat_template_kwargs,
        get_openai_client=lambda: openai_client,
    )


def _managed_api_provider():
    """Provider shaped like an OpenAI-compatible managed API: no such attribute."""
    openai_client = SimpleNamespace(base_url="https://api.openai.com/v1")
    return SimpleNamespace(
        model_name="gpt-4o",
        get_openai_client=lambda: openai_client,
    )


_SCAFFOLD_LOGGER = "olmo_eval.harness.scaffolds.openai_agents"


def _build_agent(provider):
    return OpenAIAgentsScaffold()._create_agent(provider, HarnessConfig(name="test"))


def _scaffold_records(caplog):
    return [record for record in caplog.records if record.name == _SCAFFOLD_LOGGER]


def _sent_chat_template_kwargs(agent):
    extra_body = agent.model_settings.extra_body
    return None if not extra_body else extra_body.get("chat_template_kwargs")


class _CapturingChatCompletions:
    def __init__(self):
        self.create_kwargs: dict[str, Any] | None = None

    async def create(self, **kwargs):
        from openai.types.chat import ChatCompletion, ChatCompletionMessage
        from openai.types.chat.chat_completion import Choice

        self.create_kwargs = kwargs
        return ChatCompletion(
            id="chatcmpl-test",
            created=0,
            model="test-model",
            object="chat.completion",
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    message=ChatCompletionMessage(role="assistant", content="done"),
                )
            ],
        )


class _CapturingClient:
    """Minimal AsyncOpenAI stand-in that records the request body the SDK builds."""

    def __init__(self):
        self.base_url = "http://localhost:8000/v1"
        self.completions = _CapturingChatCompletions()
        self.chat = SimpleNamespace(completions=self.completions)


class TestChatTemplateKwargs:
    def test_self_hosted_provider_disables_thinking_by_default(self):
        agent = _build_agent(_self_hosted_provider())

        assert _sent_chat_template_kwargs(agent) == {"enable_thinking": False}

    def test_explicit_enable_thinking_wins_over_the_default(self):
        agent = _build_agent(_self_hosted_provider({"enable_thinking": True}))

        assert _sent_chat_template_kwargs(agent) == {"enable_thinking": True}

    def test_other_configured_kwargs_are_kept_alongside_the_default(self):
        agent = _build_agent(_self_hosted_provider({"custom_flag": "on"}))

        assert _sent_chat_template_kwargs(agent) == {
            "enable_thinking": False,
            "custom_flag": "on",
        }

    def test_managed_api_provider_gets_no_chat_template_kwargs(self):
        agent = _build_agent(_managed_api_provider())

        assert agent.model_settings.extra_body is None

    def test_metrics_wrapped_provider_still_gets_the_default(self):
        from olmo_eval.inference.metrics.core.collector import InstrumentedProvider

        agent = _build_agent(InstrumentedProvider(_self_hosted_provider()))

        assert _sent_chat_template_kwargs(agent) == {"enable_thinking": False}

    def test_default_never_leaks_into_provider_or_module_state(self):
        configured: dict[str, Any] = {}
        provider = _self_hosted_provider(configured)

        first = _sent_chat_template_kwargs(_build_agent(provider))
        assert first == {"enable_thinking": False}

        # Mutating what we sent must not reach the provider config, the module
        # default, or any later agent built from the same provider.
        assert first is not configured
        assert first is not DEFAULT_CHAT_TEMPLATE_KWARGS
        first["enable_thinking"] = True
        first["injected"] = "leak"

        assert configured == {}
        assert provider.chat_template_kwargs == {}
        assert DEFAULT_CHAT_TEMPLATE_KWARGS == {"enable_thinking": False}
        assert _sent_chat_template_kwargs(_build_agent(provider)) == {"enable_thinking": False}

    def test_applying_the_default_is_logged_at_info(self, caplog):
        with caplog.at_level(logging.DEBUG, logger=_SCAFFOLD_LOGGER):
            _build_agent(_self_hosted_provider())

        info = [r.message for r in _scaffold_records(caplog) if r.levelno == logging.INFO]
        assert len(info) == 1
        assert "Defaulted chat_template_kwargs" in info[0]
        assert "enable_thinking" in info[0]

    def test_explicit_config_is_not_logged_at_info(self, caplog):
        with caplog.at_level(logging.DEBUG, logger=_SCAFFOLD_LOGGER):
            _build_agent(_self_hosted_provider({"enable_thinking": True}))

        records = _scaffold_records(caplog)
        assert not [r for r in records if r.levelno >= logging.INFO]
        assert any("explicitly configured chat_template_kwargs" in r.message for r in records)

    @pytest.mark.anyio
    async def test_default_reaches_the_chat_completions_request_body(self):
        from agents import ModelTracing

        client = _CapturingClient()
        agent = _build_agent(_self_hosted_provider(client=client))

        await agent.model.get_response(
            system_instructions=None,
            input="hello",
            model_settings=agent.model_settings,
            tools=[],
            output_schema=None,
            handoffs=[],
            tracing=ModelTracing.DISABLED,
        )

        assert client.completions.create_kwargs is not None
        assert client.completions.create_kwargs["extra_body"] == {
            "chat_template_kwargs": {"enable_thinking": False}
        }


def _named_tool(name: str):
    from olmo_eval.harness.tools import Tool

    async def _execute(query: str = "") -> str:
        return "ok"

    return Tool(
        name=name,
        description=f"{name} description",
        execute=_execute,
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
        },
    )


class TestToolErrorFormatter:
    def test_message_names_the_bad_tool_and_lists_the_valid_ones(self):
        formatter = _make_tool_error_formatter(["paper_search", "snippet_search"])

        message = formatter(
            SimpleNamespace(kind="tool_not_found", tool_name="web_search", call_id="c1")
        )

        assert message == (
            "Error: tool 'web_search' does not exist. "
            "Available tools: paper_search, snippet_search. Call one of these exact names."
        )

    def test_defers_to_sdk_default_for_other_error_kinds(self):
        formatter = _make_tool_error_formatter(["paper_search"])

        rejected = formatter(
            SimpleNamespace(kind="approval_rejected", tool_name="paper_search", call_id="c1")
        )

        assert rejected is None

    def test_handles_an_empty_tool_inventory(self):
        formatter = _make_tool_error_formatter([])

        message = formatter(
            SimpleNamespace(kind="tool_not_found", tool_name="web_search", call_id="c1")
        )

        assert message == "Error: tool 'web_search' does not exist. No tools are available."


class TestNativeUnknownToolRecovery:
    @pytest.mark.anyio
    async def test_run_opts_into_native_recovery_with_the_real_tool_names(self, monkeypatch):
        from agents import Agent, Runner

        agent = Agent(name="test-agent", instructions="Use tools.")
        run_calls = []

        async def fake_run(**kwargs):
            run_calls.append(kwargs)
            return SimpleNamespace(final_output="done", new_items=[])

        _patch_scaffold_agent(monkeypatch, agent)
        monkeypatch.setattr(Runner, "run", staticmethod(fake_run))

        await OpenAIAgentsScaffold().run(
            provider=SimpleNamespace(),
            config=HarnessConfig(
                name="test",
                max_turns=3,
                tools=(_named_tool("paper_search"), _named_tool("snippet_search")),
            ),
            request=_agent_request(),
            enable_compaction=False,
        )

        run_config = run_calls[0]["run_config"]
        assert run_config.tool_not_found_behavior == "return_error_to_model"

        message = run_config.tool_error_formatter(
            SimpleNamespace(kind="tool_not_found", tool_name="web_search", call_id="c1")
        )
        assert "paper_search" in message
        assert "snippet_search" in message

    @pytest.mark.anyio
    async def test_unknown_tool_recovers_against_the_real_sdk_dispatch(self):
        """Pin the opt-in against the SDK's own dispatch, not a mock of it.

        Scripts a chat-completions endpoint that calls a tool that was never registered,
        then answers normally, and asserts the SDK fed our message back as a tool result
        instead of raising ModelBehaviorError.
        """
        import json

        import httpx
        from agents import (
            Agent,
            OpenAIChatCompletionsModel,
            RunConfig,
            Runner,
            function_tool,
        )
        from openai import AsyncOpenAI

        requests: list[dict] = []

        def completion(message: dict, finish: str) -> dict:
            return {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "test-model",
                "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(json.loads(request.content.decode()))
            if len(requests) == 1:
                return httpx.Response(
                    200,
                    json=completion(
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_bogus",
                                    "type": "function",
                                    "function": {
                                        "name": "web_search",
                                        "arguments": '{"query":"olmo"}',
                                    },
                                }
                            ],
                        },
                        "tool_calls",
                    ),
                )
            return httpx.Response(
                200, json=completion({"role": "assistant", "content": "recovered"}, "stop")
            )

        @function_tool(strict_mode=False)
        async def paper_search(query: str) -> str:
            """Search papers."""
            return "results"

        client = AsyncOpenAI(
            api_key="test-key-not-real",
            base_url="http://test.invalid/v1",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        agent = Agent(
            name="test-agent",
            instructions="Use tools.",
            model=OpenAIChatCompletionsModel(openai_client=client, model="test-model"),
            tools=[paper_search],
        )

        result = await Runner.run(
            starting_agent=agent,
            input="find something",
            max_turns=5,
            run_config=RunConfig(
                tool_not_found_behavior="return_error_to_model",
                tool_error_formatter=_make_tool_error_formatter(["paper_search"]),
            ),
        )

        assert result.final_output == "recovered"
        tool_messages = [m for m in requests[1]["messages"] if m.get("role") == "tool"]
        assert len(tool_messages) == 1
        assert tool_messages[0]["content"] == (
            "Error: tool 'web_search' does not exist. "
            "Available tools: paper_search. Call one of these exact names."
        )


class TestModelRefusal:
    @pytest.mark.anyio
    async def test_refusal_is_reported_instead_of_taking_down_the_run(self, monkeypatch):
        from agents import Agent, Runner
        from agents.exceptions import ModelRefusalError

        agent = Agent(name="test-agent", instructions="Use tools.")

        async def fake_run(**kwargs):
            raise ModelRefusalError("I cannot help with that.")

        _patch_scaffold_agent(monkeypatch, agent)
        monkeypatch.setattr(Runner, "run", staticmethod(fake_run))

        result = await OpenAIAgentsScaffold().run(
            provider=SimpleNamespace(),
            config=HarnessConfig(name="test", max_turns=3),
            request=_agent_request(),
            enable_compaction=False,
        )

        assert "I cannot help with that." in result.final_output.text
        assert result.error is not None


def _scripted_client(*messages: dict[str, Any]):
    """Real AsyncOpenAI over a mock transport that answers with ``messages`` in order.

    Returns the client and the request bodies the SDK sent, so a test can check both
    what came back and what was replayed into the next request. Running out of scripted
    messages fails the test.
    """
    import json

    import httpx
    from openai import AsyncOpenAI

    queue = list(messages)
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content.decode()))
        if not queue:
            # pytest.fail raises a BaseException, which the OpenAI client's retry loop
            # cannot swallow the way it would an AssertionError from the transport.
            pytest.fail(
                f"scripted client got request {len(requests)} but only "
                f"{len(requests) - 1} message(s) were scripted"
            )
        message = queue.pop(0)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": "tool_calls" if message.get("tool_calls") else "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    client = AsyncOpenAI(
        api_key="test-key-not-real",
        base_url="http://test.invalid/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    return client, requests


def _vllm_message(content: str | None, reasoning: str, tool_calls=None) -> dict[str, Any]:
    """Assistant message shaped like vLLM 0.19: thinking under ``reasoning`` only."""
    message: dict[str, Any] = {"role": "assistant", "content": content, "reasoning": reasoning}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def _reasoning_summaries(items) -> list[list[str]]:
    return [
        [part.text for part in item.summary]
        for item in items
        if type(item).__name__ == "ResponseReasoningItem"
    ]


class TestVllmReasoningField:
    """vLLM returns thinking as ``message.reasoning``, which openai-agents 0.20 ignores.

    The scaffold's model subclass mirrors it onto ``reasoning_content`` as soon as the raw
    ChatCompletion comes back, so the SDK's own converter emits the ReasoningItem.
    """

    async def _output_items(self, message: dict[str, Any]):
        from agents import ModelSettings, ModelTracing

        client, _ = _scripted_client(message)
        async with client:
            model = _chat_completions_model_class()(openai_client=client, model="test-model")
            response = await model.get_response(
                system_instructions=None,
                input="hello",
                model_settings=ModelSettings(),
                tools=[],
                output_schema=None,
                handoffs=[],
                tracing=ModelTracing.DISABLED,
            )
        return response.output

    @pytest.mark.anyio
    async def test_vllm_reasoning_becomes_a_reasoning_item(self):
        items = await self._output_items(_vllm_message("answer", "thought"))

        assert [type(i).__name__ for i in items] == [
            "ResponseReasoningItem",
            "ResponseOutputMessage",
        ]
        assert _reasoning_summaries(items) == [["thought"]]

    @pytest.mark.anyio
    async def test_vllm_reasoning_on_a_tool_call_message(self):
        call = {
            "id": "call_1",
            "type": "function",
            "function": {"name": "paper_search", "arguments": '{"query":"olmo"}'},
        }
        items = await self._output_items(_vllm_message(None, "tool thought", [call]))

        assert [type(i).__name__ for i in items] == [
            "ResponseReasoningItem",
            "ResponseFunctionToolCall",
        ]
        assert _reasoning_summaries(items) == [["tool thought"]]

    @pytest.mark.anyio
    async def test_reasoning_content_is_left_alone(self):
        """DeepSeek already uses the field the SDK reads; the shim must not touch it."""
        items = await self._output_items(
            {"role": "assistant", "content": "answer", "reasoning_content": "rc"}
        )

        assert _reasoning_summaries(items) == [["rc"]]

    @pytest.mark.anyio
    async def test_reasoning_content_wins_over_a_stray_reasoning_field(self):
        items = await self._output_items(
            {
                "role": "assistant",
                "content": "answer",
                "reasoning_content": "rc",
                "reasoning": "raw",
            }
        )

        assert _reasoning_summaries(items) == [["rc"]]

    @pytest.mark.anyio
    async def test_a_message_without_reasoning_is_untouched(self):
        items = await self._output_items({"role": "assistant", "content": "answer"})

        assert [type(i).__name__ for i in items] == ["ResponseOutputMessage"]

    def test_mirroring_only_adds_the_field_for_the_vllm_shape(self):
        from openai.types.chat import ChatCompletionMessage

        vllm = ChatCompletionMessage.model_validate(
            {"role": "assistant", "content": "a", "reasoning": "t"}
        )
        deepseek = ChatCompletionMessage.model_validate(
            {"role": "assistant", "content": "a", "reasoning_content": "rc"}
        )
        plain = ChatCompletionMessage(role="assistant", content="a")
        plain_before = plain.model_dump()

        for message in (vllm, deepseek, plain):
            _mirror_vllm_reasoning(message)

        assert vllm.model_extra == {"reasoning": "t", "reasoning_content": "t"}
        assert deepseek.model_extra == {"reasoning_content": "rc"}
        assert not plain.model_extra
        assert plain.model_dump() == plain_before

    @pytest.mark.parametrize(
        ("extra", "reasoning_content_after"),
        [
            pytest.param({"reasoning": ""}, None, id="empty_reasoning"),
            pytest.param({"reasoning": ["x"]}, None, id="non_string_reasoning"),
            pytest.param(
                {"reasoning": "t", "thinking_blocks": [{"type": "thinking", "thinking": "b"}]},
                None,
                id="thinking_blocks_present",
            ),
            pytest.param(
                {"reasoning": "t", "reasoning_content": ""}, "t", id="empty_reasoning_content"
            ),
        ],
    )
    def test_mirroring_edge_shapes(self, extra, reasoning_content_after):
        from openai.types.chat import ChatCompletionMessage

        message = ChatCompletionMessage.model_validate(
            {"role": "assistant", "content": "a", **extra}
        )
        before = message.model_dump()

        _mirror_vllm_reasoning(message)

        assert getattr(message, "reasoning_content", None) == reasoning_content_after
        if reasoning_content_after is None:
            assert message.model_dump() == before

    def test_the_sdk_hook_point_is_still_there(self):
        """Fail loudly if the SDK renames the private method the scaffold overrides.

        ``get_response`` must still await ``self._fetch_response`` and hand its result to
        the converter; otherwise the override is dead code and reasoning is silently lost.
        """
        import inspect

        from agents import OpenAIChatCompletionsModel

        hook = getattr(OpenAIChatCompletionsModel, "_fetch_response", None)
        assert inspect.iscoroutinefunction(hook), (
            "OpenAIChatCompletionsModel._fetch_response is gone; the reasoning shim no longer runs"
        )
        assert "self._fetch_response(" in inspect.getsource(OpenAIChatCompletionsModel.get_response)
        assert "message_to_output_items(" in inspect.getsource(
            OpenAIChatCompletionsModel.get_response
        )
        assert issubclass(_chat_completions_model_class(), OpenAIChatCompletionsModel)

    @pytest.mark.anyio
    async def test_vllm_reasoning_reaches_the_saved_trajectory(self):
        """Scaffold -> SDK Runner -> converter -> AgentTurn.reasoning, with no replay."""
        call = {
            "id": "call_1",
            "type": "function",
            "function": {"name": "paper_search", "arguments": '{"query":"olmo"}'},
        }
        client, requests = _scripted_client(
            _vllm_message(None, "Search first.", [call]),
            _vllm_message("Final.", "Now answer."),
        )

        async with client:
            result = await OpenAIAgentsScaffold().run(
                provider=_self_hosted_provider(client=client),
                config=HarnessConfig(
                    name="test", max_turns=3, tools=(_named_tool("paper_search"),)
                ),
                request=_agent_request(),
                enable_compaction=False,
            )

        assert result.final_output.text == "Final."
        assert result.trajectory is not None
        turns = result.trajectory.turns
        assert [t.role for t in turns] == ["assistant", "tool", "assistant"]
        assert turns[0].reasoning == "Search first."
        assert turns[0].tool_calls[0].function.name == "paper_search"
        assert turns[2].content == "Final."
        assert turns[2].reasoning == "Now answer."
        assert turns[2].to_dict()["reasoning"] == "Now answer."

        # The mirrored field is not replayed to a non-DeepSeek model, so what the server
        # sees on later turns is exactly what it saw before the shim.
        assert len(requests) == 2
        assistant_messages = [m for m in requests[1]["messages"] if m.get("role") == "assistant"]
        assert assistant_messages
        assert all(
            "reasoning_content" not in m and "reasoning" not in m for m in assistant_messages
        )

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("model_name", "replayed"), [("deepseek-r1", True), ("qwen3-8b", False)]
    )
    async def test_mirrored_reasoning_is_replayed_only_to_deepseek_names(
        self, model_name, replayed
    ):
        """The SDK replays ``reasoning_content`` to models named deepseek, mirrored or not.

        vLLM 0.19.1 ignores the key in assistant history, so for a DeepSeek model served
        by vLLM this is extra payload rather than a behavior change.
        """
        call = {
            "id": "call_1",
            "type": "function",
            "function": {"name": "paper_search", "arguments": '{"query":"olmo"}'},
        }
        client, requests = _scripted_client(
            _vllm_message(None, "Search first.", [call]),
            _vllm_message("Final.", "Now answer."),
        )

        async with client:
            result = await OpenAIAgentsScaffold().run(
                provider=_self_hosted_provider(client=client, model_name=model_name),
                config=HarnessConfig(
                    name="test", max_turns=3, tools=(_named_tool("paper_search"),)
                ),
                request=_agent_request(),
                enable_compaction=False,
            )

        assert result.trajectory is not None
        assert result.trajectory.turns[0].reasoning == "Search first."
        assert len(requests) == 2
        assert requests[1]["model"] == model_name
        assistant_messages = [m for m in requests[1]["messages"] if m.get("role") == "assistant"]
        assert [m.get("reasoning_content") for m in assistant_messages] == (
            ["Search first."] if replayed else [None]
        )
        assert all("reasoning" not in m for m in assistant_messages)
