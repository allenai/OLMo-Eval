"""Pin the openai client surface this repo actually consumes.

These build real SDK response objects rather than mocks, so a client upgrade that
changes a response shape fails here instead of silently zeroing token accounting.
"""

import pytest

from olmo_eval.inference.retry import (
    ALWAYS_RETRY_EXCEPTION_TYPES,
    NEVER_RETRY_EXCEPTION_TYPES,
)

# The openai client ships with the `clients`/`litellm` extras; CI installs neither.
pytest.importorskip("openai")

# Names in the retry tables that belong to litellm, not the openai SDK.
_LITELLM_ONLY_EXCEPTION_NAMES = frozenset({"Timeout", "ServiceUnavailableError"})


def _chat_completion(*, prompt_tokens=11, completion_tokens=7, tool_calls=None, content="hi"):
    """Build a real ChatCompletion, the way the server would return one."""
    from openai.types.chat import ChatCompletion, ChatCompletionMessage
    from openai.types.chat.chat_completion import Choice
    from openai.types.completion_usage import CompletionUsage

    return ChatCompletion(
        id="chatcmpl-test",
        object="chat.completion",
        created=0,
        model="test-model",
        choices=[
            Choice(
                index=0,
                finish_reason="tool_calls" if tool_calls else "stop",
                message=ChatCompletionMessage(
                    role="assistant",
                    content=None if tool_calls else content,
                    tool_calls=tool_calls,
                ),
            )
        ],
        usage=CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


class TestRetryExceptionSurface:
    """retry.py resolves openai exception classes reflectively, by name.

    A rename in the SDK would not raise; it would silently downgrade classification
    to status-code and string matching. Assert the names still resolve.
    """

    @pytest.mark.parametrize(
        "name",
        [n for n in NEVER_RETRY_EXCEPTION_TYPES + ALWAYS_RETRY_EXCEPTION_TYPES],
    )
    def test_exception_name_resolves_or_is_known_litellm_only(self, name):
        import openai

        resolved = getattr(openai, name, None)
        if name in _LITELLM_ONLY_EXCEPTION_NAMES:
            pytest.skip(f"{name} is a litellm-only name")
        assert resolved is not None, (
            f"openai no longer exports {name!r}; retry classification would silently "
            "fall back to string matching"
        )
        assert issubclass(resolved, Exception)

    def test_status_code_classification_still_works_on_a_real_sdk_error(self):
        import httpx
        import openai

        from olmo_eval.inference.retry import is_retryable_error

        request = httpx.Request("POST", "http://test.invalid/v1/chat/completions")
        rate_limited = openai.RateLimitError(
            "slow down",
            response=httpx.Response(429, request=request),
            body=None,
        )
        bad_request = openai.BadRequestError(
            "nope",
            response=httpx.Response(400, request=request),
            body=None,
        )

        assert is_retryable_error(rate_limited, openai) is True
        assert is_retryable_error(bad_request, openai) is False

    def test_out_of_credits_is_not_retried(self):
        import httpx
        import openai

        from olmo_eval.inference.retry import is_retryable_error

        request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
        out_of_credits = openai.APIStatusError(
            "Insufficient credits",
            response=httpx.Response(402, request=request),
            body=None,
        )

        assert is_retryable_error(out_of_credits, openai) is False


class TestInstrumentedClientUsageAccounting:
    """The metrics wrapper reads usage with getattr, so a shape change zeroes it silently."""

    @pytest.mark.anyio
    async def test_usage_is_recorded_from_a_real_chat_completion(self):
        from olmo_eval.inference.metrics.core.collector import InstrumentedChatCompletions

        response = _chat_completion(prompt_tokens=11, completion_tokens=7)

        class _Completions:
            async def create(self, **kwargs):
                return response

        class _Collector:
            model_name = "test-model"

            def __init__(self):
                self._request_metrics = []

            def _start_gpu_monitor_if_needed(self):
                return None

        collector = _Collector()
        wrapped = InstrumentedChatCompletions(_Completions(), collector)

        returned = await wrapped.create(model="test-model", messages=[])

        assert returned is response
        assert len(collector._request_metrics) == 1
        recorded = collector._request_metrics[0]
        assert recorded.prompt_tokens == 11
        assert recorded.completion_tokens == 7
        assert recorded.finish_reason == "stop"
        assert recorded.model == "test-model"


class TestVLLMServerUsageParsing:
    def test_usage_metadata_from_a_real_completion_usage(self):
        from openai.types.completion_usage import CompletionUsage

        from olmo_eval.inference.providers.vllm_server import VLLMServerProvider

        usage = CompletionUsage(prompt_tokens=5, completion_tokens=3, total_tokens=8)
        metadata = VLLMServerProvider._completion_usage_metadata(None, usage)

        assert metadata == {"prompt_tokens": 5, "completion_tokens": 3}

    def test_usage_metadata_tolerates_the_new_cache_write_tokens_detail(self):
        """openai 2.45+ added cache_write_tokens to PromptTokensDetails (optional)."""
        from openai.types.completion_usage import CompletionUsage, PromptTokensDetails

        from olmo_eval.inference.providers.vllm_server import VLLMServerProvider

        usage = CompletionUsage(
            prompt_tokens=5,
            completion_tokens=3,
            total_tokens=8,
            prompt_tokens_details=PromptTokensDetails(cached_tokens=2),
        )
        metadata = VLLMServerProvider._completion_usage_metadata(None, usage)

        assert metadata == {"prompt_tokens": 5, "completion_tokens": 3}

    def test_chat_response_tool_calls_and_usage_are_read_directly(self):
        """_generate_chat reads usage.prompt_tokens as a direct attribute, not getattr."""
        from openai.types.chat.chat_completion_message_function_tool_call import (
            ChatCompletionMessageFunctionToolCall,
            Function,
        )

        from olmo_eval.common.types.tools import ToolCall

        tool_call = ChatCompletionMessageFunctionToolCall(
            id="call_1",
            type="function",
            function=Function(name="paper_search", arguments='{"query":"olmo"}'),
        )
        response = _chat_completion(tool_calls=[tool_call])

        choice = response.choices[0]
        usage = response.usage

        # Mirror the exact reads _generate_chat performs on the SDK objects.
        assert usage.prompt_tokens == 11
        assert usage.completion_tokens == 7
        assert choice.message.content is None
        assert choice.message.tool_calls is not None
        converted = [
            ToolCall.create(call_id=tc.id, name=tc.function.name, arguments=tc.function.arguments)
            for tc in choice.message.tool_calls
        ]
        assert converted[0].id == "call_1"
        assert converted[0].function.name == "paper_search"
        assert converted[0].function.arguments == '{"query":"olmo"}'


class TestLiteLLMOpenAIClient:
    """The litellm provider hands this client to the agents scaffold."""

    def test_raises_without_a_base_url(self):
        from olmo_eval.inference.providers.litellm import LiteLLMProvider

        provider = LiteLLMProvider.__new__(LiteLLMProvider)
        provider.base_url = None
        provider._client = None
        provider._request_overrides = {}

        with pytest.raises(NotImplementedError, match="base_url"):
            provider.get_openai_client()

    def test_builds_and_caches_a_client_for_a_base_url(self, monkeypatch):
        from olmo_eval.inference.providers.litellm import LiteLLMProvider

        monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")

        provider = LiteLLMProvider.__new__(LiteLLMProvider)
        provider.base_url = "https://api.openai.com/v1"
        provider._client = None
        provider._request_overrides = {}

        client = provider.get_openai_client()

        assert client is not None
        assert str(client.base_url).rstrip("/") == "https://api.openai.com/v1"
        assert provider.get_openai_client() is client


class TestAgentsConverterPatch:
    """The vLLM strict patch monkeypatches the agents SDK converter."""

    def test_patch_drops_strict_for_non_strict_tools(self):
        pytest.importorskip("agents")
        from agents import FunctionTool
        from agents.models.chatcmpl_converter import Converter

        from olmo_eval.inference.utils import patch_openai_agents_for_vllm

        patch_openai_agents_for_vllm()

        async def _invoke(ctx, args):
            return "ok"

        tool = FunctionTool(
            name="paper_search",
            description="search",
            params_json_schema={"type": "object", "properties": {}},
            on_invoke_tool=_invoke,
            strict_json_schema=False,
        )

        payload = Converter.tool_to_openai(tool)

        assert payload["function"]["name"] == "paper_search"
        assert "strict" not in payload["function"]
