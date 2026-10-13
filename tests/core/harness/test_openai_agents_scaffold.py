"""Tests for the OpenAI Agents scaffold (requires the optional agents/openai deps)."""

import pytest

pytest.importorskip("agents")
pytest.importorskip("openai")

import contextlib  # noqa: E402
import logging  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from typing import Any  # noqa: E402

from olmo_eval.common.types import LMRequest, RequestType  # noqa: E402
from olmo_eval.harness.config import HarnessConfig  # noqa: E402
from olmo_eval.harness.scaffolds.openai_agents import (  # noqa: E402
    DEFAULT_CHAT_TEMPLATE_KWARGS,
    FORCED_FINAL_ANSWER_INSTRUCTION,
    OpenAIAgentsScaffold,
    _make_tool_error_formatter,
)
from olmo_eval.harness.tools import Tool  # noqa: E402


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


def _stub_openai_client():
    return SimpleNamespace(base_url="http://localhost:8000/v1")


def _self_hosted_provider(chat_template_kwargs=None, client=None):
    """Provider shaped like VLLMServerProvider: exposes ``chat_template_kwargs``."""
    openai_client = client if client is not None else _stub_openai_client()
    return SimpleNamespace(
        model_name="test-model",
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

    @pytest.mark.anyio
    async def test_the_forced_final_answer_is_told_it_has_no_tools(self, monkeypatch):
        """The forced final call drops every tool, so it must not name the ones it dropped."""
        from agents import Agent, Runner

        agent = Agent(name="test-agent", instructions="Use tools.")
        run_calls = []

        async def fake_run(**kwargs):
            run_calls.append(kwargs)
            if len(run_calls) == 1:
                raise _max_turns_exceeded_with_run_data(_FakeRunData(new_items=[]))
            return SimpleNamespace(final_output="Forced final answer")

        _patch_scaffold_agent(monkeypatch, agent)
        monkeypatch.setattr(Runner, "run", staticmethod(fake_run))

        await OpenAIAgentsScaffold().run(
            provider=SimpleNamespace(),
            config=HarnessConfig(
                name="test",
                max_turns=1,
                tools=(_named_tool("paper_search"),),
            ),
            request=_agent_request(),
            enable_compaction=False,
        )

        bad_call = SimpleNamespace(kind="tool_not_found", tool_name="web_search", call_id="c1")
        loop_config = run_calls[0]["run_config"]
        final_config = run_calls[1]["run_config"]

        assert "paper_search" in loop_config.tool_error_formatter(bad_call)
        assert final_config.tool_not_found_behavior == "return_error_to_model"
        assert final_config.tool_error_formatter(bad_call) == (
            "Error: tool 'web_search' does not exist. No tools are available."
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


def _chat_message(*, content=None, tool_calls=None):
    from openai.types.chat import ChatCompletionMessage

    return ChatCompletionMessage(role="assistant", content=content, tool_calls=tool_calls)


def _chat_tool_call(call_id: str, name: str, arguments: str = "{}"):
    from openai.types.chat import ChatCompletionMessageFunctionToolCall
    from openai.types.chat.chat_completion_message_function_tool_call import Function

    return ChatCompletionMessageFunctionToolCall(
        id=call_id,
        type="function",
        function=Function(name=name, arguments=arguments),
    )


def _chat_completion(message):
    from openai.types.chat import ChatCompletion
    from openai.types.chat.chat_completion import Choice

    finish_reason = "tool_calls" if message.tool_calls else "stop"
    return ChatCompletion(
        id="chatcmpl-test",
        choices=[Choice(finish_reason=finish_reason, index=0, message=message)],
        created=0,
        model="test-model",
        object="chat.completion",
    )


def _calls_tool(name: str, arguments: str = "{}", call_id: str | None = None):
    return _chat_message(tool_calls=[_chat_tool_call(call_id or f"call_{name}", name, arguments)])


class _ScriptedClient:
    """A real AsyncOpenAI whose completions are scripted, recording what the model was sent."""

    def __init__(self, script):
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(api_key="test", base_url="http://example.com/v1")
        self.script = list(script)
        self.sent_messages = []
        self.sent_tool_names = []

        async def create(**kwargs):
            self.sent_messages.append(kwargs.get("messages"))
            tools = kwargs.get("tools") or []
            self.sent_tool_names.append(
                [t.get("function", {}).get("name") for t in tools if isinstance(t, dict)]
            )
            message = self.script.pop(0) if self.script else _chat_message(content="fallback")
            return _chat_completion(message)

        self.client.chat.completions.create = create  # type: ignore[method-assign]

    def tool_results_seen(self):
        """Every tool result the model was handed, read off the last request's history."""
        results = []
        for message in (self.sent_messages[-1] if self.sent_messages else []) or []:
            if isinstance(message, dict) and message.get("role") == "tool":
                results.append(str(message.get("content", "")))
        return results


def _search_config(**kwargs):
    async def search(query: str) -> str:
        return f"results for {query}"

    return HarnessConfig(
        name="test",
        tools=(Tool.from_function(search, name="search", description="Search for information."),),
        **kwargs,
    )


async def _run_scaffold(scripted, config, *, agent=None):
    class ProviderStub:
        model_name = "test-model"

        def get_openai_client(self):
            return scripted.client

    scaffold = OpenAIAgentsScaffold()
    if agent is not None:
        scaffold._get_or_create_agent = (  # type: ignore[method-assign]
            lambda provider, config, sandbox_manager=None: agent
        )
    return await scaffold.run(
        provider=ProviderStub(),
        config=config,
        request=_agent_request(),
        enable_compaction=False,
    )


class TestUnknownToolCallThroughTheScaffold:
    """Drive the SDK's real dispatch through scaffold.run, not through Runner.run directly.

    An unknown tool name used to abort the instance and replace the answer with an error
    string. These pin that the opt-in in run() is what keeps the loop alive, and that the
    fumble is still legible afterwards.
    """

    @pytest.mark.anyio
    async def test_the_raw_failure_this_opt_in_replaces(self):
        """Without the opt-in the SDK still raises, so the opt-in is load-bearing."""
        from agents import Agent, OpenAIChatCompletionsModel, Runner
        from agents.exceptions import ModelBehaviorError

        scripted = _ScriptedClient(
            [
                _calls_tool("thought", '{"text":"let me think"}'),
                _chat_message(content="Never reached."),
            ]
        )
        agent = Agent(
            name="openai_agents",
            instructions="",
            model=OpenAIChatCompletionsModel(openai_client=scripted.client, model="test-model"),
            tools=[],
        )

        with pytest.raises(ModelBehaviorError) as excinfo:
            await Runner.run(starting_agent=agent, input="What did you find?", max_turns=6)

        assert str(excinfo.value) == "Tool thought not found in agent openai_agents"

    @pytest.mark.anyio
    async def test_unknown_tool_call_recovers_and_answer_survives(self):
        scripted = _ScriptedClient(
            [
                _calls_tool("thought", '{"text":"let me think"}'),
                _calls_tool("search", '{"query":"olmo"}'),
                _chat_message(content="The answer survived."),
            ]
        )

        result = await _run_scaffold(scripted, _search_config(max_turns=6))

        assert result.final_output.text == "The answer survived."
        assert result.error is None
        assert result.max_turns_reached is False

        tool_results = scripted.tool_results_seen()
        assert any("thought" in r and "does not exist" in r for r in tool_results)
        assert any("Available tools: search" in r for r in tool_results)
        assert "results for olmo" in tool_results

        for advertised in scripted.sent_tool_names:
            assert advertised == ["search"]

        assert result.trajectory.total_tool_calls == 2

    @pytest.mark.anyio
    async def test_repeated_unknown_tool_calls_still_finish(self):
        scripted = _ScriptedClient(
            [
                _calls_tool("thought", call_id="call_1"),
                _calls_tool("thought", call_id="call_2"),
                _calls_tool("ponder", call_id="call_3"),
                _chat_message(content="Answer after three bad calls."),
            ]
        )

        result = await _run_scaffold(scripted, _search_config(max_turns=8))

        assert result.final_output.text == "Answer after three bad calls."
        assert result.error is None
        tool_results = scripted.tool_results_seen()
        assert len([r for r in tool_results if "does not exist" in r]) == 3
        assert any("ponder" in r for r in tool_results)

    @pytest.mark.anyio
    async def test_unknown_tool_calls_consume_the_turn_budget(self):
        """Endless unknown calls must hit the existing max-turns path, not spin forever."""
        scripted = _ScriptedClient([_calls_tool("thought", call_id=f"call_{i}") for i in range(10)])

        result = await _run_scaffold(scripted, _search_config(max_turns=3))

        assert result.max_turns_reached is True
        assert len(scripted.sent_messages) <= 10

    @pytest.mark.anyio
    async def test_the_fumbled_name_stays_in_the_trajectory(self):
        """Scoring and inspection need the name the model actually called."""
        scripted = _ScriptedClient(
            [
                _calls_tool("thought", '{"text":"let me think"}', call_id="call_1"),
                _calls_tool("search", '{"query":"olmo"}', call_id="call_2"),
                _chat_message(content="done"),
            ]
        )

        result = await _run_scaffold(scripted, _search_config(max_turns=6))

        called = [
            call.function.name for turn in result.trajectory.turns for call in turn.tool_calls
        ]
        assert called == ["thought", "search"]

        errored = [
            r.content
            for turn in result.trajectory.turns
            for r in turn.tool_results
            if "does not exist" in r.content
        ]
        assert len(errored) == 1
        assert "thought" in errored[0]

    @pytest.mark.anyio
    async def test_an_unknown_call_on_the_forced_final_turn_gives_up_softly(self):
        """The forced final answer runs with no tools and one turn, so it cannot recover.

        What matters is that the instance still comes back as a scored result with the
        fallback answer instead of raising out of the scaffold.
        """
        scripted = _ScriptedClient(
            [
                _calls_tool("search", '{"query":"olmo"}', call_id="call_1"),
                _calls_tool("search", '{"query":"olmo2"}', call_id="call_2"),
                _calls_tool("thought", '{"text":"still thinking"}', call_id="call_3"),
            ]
        )

        result = await _run_scaffold(scripted, _search_config(max_turns=2))

        assert result.max_turns_reached is True
        assert result.final_output.text == "[Max turns exceeded]"
        assert result.error == "Max turns (2) exceeded"
        assert scripted.sent_tool_names[-1] == []

    @pytest.mark.anyio
    async def test_normal_run_is_unaffected(self):
        scripted = _ScriptedClient(
            [
                _calls_tool("search", '{"query":"olmo"}'),
                _chat_message(content="Normal answer."),
            ]
        )

        result = await _run_scaffold(scripted, _search_config(max_turns=6))

        assert result.final_output.text == "Normal answer."
        assert result.error is None
        assert scripted.tool_results_seen() == ["results for olmo"]
        assert result.trajectory.total_tool_calls == 1
