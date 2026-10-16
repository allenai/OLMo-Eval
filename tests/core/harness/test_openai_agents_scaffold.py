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
    _make_tool_error_formatter,
    truncate_tool_output,
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
        lambda self, provider, config, sandbox_manager=None, **kwargs: agent,
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
        openai_client_model_name="test-model",
        chat_template_kwargs=chat_template_kwargs,
        get_openai_client=lambda: openai_client,
    )


def _managed_api_provider():
    """Provider shaped like an OpenAI-compatible managed API: no such attribute."""
    openai_client = SimpleNamespace(base_url="https://api.openai.com/v1")
    return SimpleNamespace(
        model_name="gpt-4o",
        openai_client_model_name="gpt-4o",
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

    def test_agent_sends_the_client_model_name(self):
        provider = SimpleNamespace(
            model_name="gpt-4-0613",
            openai_client_model_name="openai/gpt-4",
            get_openai_client=lambda: SimpleNamespace(base_url="https://openrouter.ai/api/v1"),
        )

        assert _build_agent(provider).model.model == "openai/gpt-4"

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


class TestToolOutputTruncation:
    def test_short_output_and_no_limit_are_unchanged(self):
        assert truncate_tool_output("abc", 10) == "abc"
        assert truncate_tool_output("x" * 100, None) == "x" * 100

    def test_long_output_keeps_head_and_tail(self):
        output = "a" * 50 + "b" * 50
        truncated = truncate_tool_output(output, 20)
        assert truncated.startswith("a" * 10)
        assert truncated.endswith("b" * 10)
        assert "[... 80 characters omitted ...]" in truncated

    @pytest.mark.anyio
    async def test_session_tool_output_is_truncated_during_a_run(self, monkeypatch):
        from agents import Agent, Runner

        from olmo_eval.harness.sandbox.executor import ExecutionResult
        from olmo_eval.harness.scaffolds import openai_agents
        from olmo_eval.harness.tools import get_tool

        class FakeBinding:
            async def execute_in_session(self, command):
                return ExecutionResult(success=False, output="x" * 1000, exit_code=2)

            async def release(self):
                pass

        class FakeManager:
            async def acquire_binding(self, capabilities):
                return FakeBinding()

        tool = get_tool("execute_bash_session")
        execute = OpenAIAgentsScaffold()._wrap_sandboxed_tool(tool, FakeManager())  # type: ignore[arg-type]
        outputs = []

        async def fake_run(**kwargs):
            outputs.append(await execute("cat big_file"))
            return SimpleNamespace(new_items=[], final_output="done")

        agent = Agent(name="test-agent", instructions="Use tools.")
        _patch_scaffold_agent(monkeypatch, agent)
        monkeypatch.setattr(Runner, "run", staticmethod(fake_run))
        scaffold = OpenAIAgentsScaffold()
        scaffold.set_sandbox_manager(FakeManager())

        await scaffold.run(
            provider=SimpleNamespace(),
            config=HarnessConfig(name="test", max_turns=3, tools=(tool,)),
            request=_agent_request(),
            enable_compaction=False,
            max_tool_output_chars=100,
        )

        assert len(outputs[0]) < 200
        assert "[... 900 characters omitted ...]" in outputs[0]
        assert outputs[0].endswith("[Exit code: 2]")
        assert openai_agents._max_tool_output_chars.get() is None


class TestStopAtTools:
    def _scaffold_with_stub_agent(self, monkeypatch):
        created = []

        def fake_create(self, provider, config, sandbox_manager=None, stop_at_tools=()):
            created.append(stop_at_tools)
            return object()

        monkeypatch.setattr(OpenAIAgentsScaffold, "_create_agent", fake_create)
        return OpenAIAgentsScaffold(), created

    def test_stop_setting_is_part_of_the_agent_cache_key(self, monkeypatch):
        scaffold, created = self._scaffold_with_stub_agent(monkeypatch)
        provider = SimpleNamespace()
        config = HarnessConfig(name="test", max_turns=3)

        first = scaffold._get_or_create_agent(provider, config, stop_at_tools=("submit",))
        again = scaffold._get_or_create_agent(provider, config, stop_at_tools=("submit",))
        plain = scaffold._get_or_create_agent(provider, config)

        assert first is again
        assert plain is not first
        assert created == [("submit",), ()]

    def test_agent_stops_at_named_tool(self):
        from olmo_eval.harness.tools import get_tool

        provider = SimpleNamespace(
            get_openai_client=lambda: SimpleNamespace(base_url="http://x"),
            model_name="m",
            openai_client_model_name="m",
        )
        config = HarnessConfig(name="test", max_turns=3, tools=(get_tool("submit"),))
        agent = OpenAIAgentsScaffold()._create_agent(provider, config, stop_at_tools=("submit",))
        assert agent.tool_use_behavior == {"stop_at_tool_names": ["submit"]}

        default = OpenAIAgentsScaffold()._create_agent(provider, config)
        assert default.tool_use_behavior == "run_llm_again"

    @pytest.mark.anyio
    @pytest.mark.parametrize("stop", [True, False])
    async def test_calling_submit_ends_the_run(self, stop):
        """Scripts a chat endpoint that calls submit on every turn."""
        import json

        import httpx
        from openai import AsyncOpenAI

        from olmo_eval.harness.tools import get_tool

        requests: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(json.loads(request.content.decode()))
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"call_{len(requests)}",
                        "type": "function",
                        "function": {"name": "submit", "arguments": "{}"},
                    }
                ],
            }
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "test-model",
                    "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                },
            )

        client = AsyncOpenAI(
            api_key="test-key-not-real",
            base_url="http://test.invalid/v1",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        provider = SimpleNamespace(
            get_openai_client=lambda: client,
            model_name="test-model",
            openai_client_model_name="test-model",
        )
        config = HarnessConfig(name="test", max_turns=3, tools=(get_tool("submit"),))

        result = await OpenAIAgentsScaffold().run(
            provider=provider,
            config=config,
            request=_agent_request(),
            enable_compaction=False,
            stop_at_tools=("submit",) if stop else None,
        )

        if stop:
            assert len(requests) == 1
            assert not result.max_turns_reached
        else:
            assert len(requests) >= 3
            assert result.max_turns_reached


class TestCompaction:
    async def _run_kwargs(self, monkeypatch) -> dict[str, Any]:
        from agents import Agent, Runner

        agent = Agent(name="test-agent", instructions="Answer.")
        run_calls: list[dict[str, Any]] = []

        async def fake_run(**kwargs):
            run_calls.append(kwargs)
            return _FakeRunData(new_items=[], final_output="done")

        _patch_scaffold_agent(monkeypatch, agent)
        monkeypatch.setattr(Runner, "run", staticmethod(fake_run))
        await OpenAIAgentsScaffold().run(
            provider=SimpleNamespace(),
            config=HarnessConfig(name="test", max_turns=3),
            request=_agent_request(),
        )
        return run_calls[0]

    @pytest.mark.anyio
    async def test_openai_key_enables_compaction(self, monkeypatch):
        from agents.memory import OpenAIResponsesCompactionSession

        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")

        kwargs = await self._run_kwargs(monkeypatch)

        assert isinstance(kwargs["session"], OpenAIResponsesCompactionSession)

    @pytest.mark.anyio
    async def test_openrouter_key_alone_skips_compaction(self, monkeypatch, caplog):
        from olmo_eval.harness.scaffolds import openai_agents

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
        openai_agents._warn_compaction_unavailable.cache_clear()

        with caplog.at_level(logging.WARNING, logger=_SCAFFOLD_LOGGER):
            kwargs = await self._run_kwargs(monkeypatch)

        assert "session" not in kwargs
        assert "responses.compact" in caplog.text
