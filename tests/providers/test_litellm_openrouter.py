"""Tests for routing OpenAI models through OpenRouter in the LiteLLM provider."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

from olmo_eval.common.types import LMRequest, RequestType, SamplingParams
from olmo_eval.inference.providers.config import ProviderConfig
from olmo_eval.inference.providers.litellm import LiteLLMProvider

_ALL_PARAMS = {"logprobs", "top_logprobs", "max_completion_tokens", "n", "stop", "temperature"}


@pytest.fixture
def fake_litellm(monkeypatch) -> list[dict]:
    """Install a stub ``litellm`` module and return the recorded request kwargs."""
    calls: list[dict] = []

    async def acompletion(**kwargs):
        calls.append(kwargs)
        message = SimpleNamespace(content="ok")
        return SimpleNamespace(choices=[SimpleNamespace(message=message, logprobs=None)])

    def get_supported_openai_params(model, custom_llm_provider):
        # Mirrors LiteLLM's model data: reasoning models reject logprobs and top_p.
        if model.startswith("o3"):
            return ["max_completion_tokens", "n", "stop", "temperature", "reasoning_effort"]
        return sorted(_ALL_PARAMS)

    module = types.ModuleType("litellm")
    module.acompletion = acompletion  # type: ignore[ty:unresolved-attribute]
    module.get_supported_openai_params = get_supported_openai_params  # type: ignore[ty:unresolved-attribute]
    monkeypatch.setitem(sys.modules, "litellm", module)
    return calls


@pytest.fixture
def openrouter_only(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")


async def _generate(provider: LiteLLMProvider) -> None:
    request = LMRequest(request_type=RequestType.CHAT, prompt="hi")
    await provider._generate_single_impl(request, SamplingParams())


@pytest.mark.anyio
@pytest.mark.usefixtures("openrouter_only")
@pytest.mark.parametrize(
    ("model", "api_base", "expected"),
    [
        ("gpt-4o", "https://api.openai.com/v1", "openrouter/openai/gpt-4o"),
        ("openai/gpt-4o-mini", "https://api.openai.com/v1", "openrouter/openai/gpt-4o-mini"),
        ("openai/o3-mini-2025-01-31", "https://api.openai.com/v1", "openrouter/openai/o3-mini"),
        ("gpt-4o", None, "openrouter/openai/gpt-4o"),
    ],
)
async def test_openai_models_route_to_openrouter(
    fake_litellm, model: str, api_base: str | None, expected: str
) -> None:
    provider = LiteLLMProvider(model, api_base=api_base)
    await _generate(provider)

    assert fake_litellm[0]["model"] == expected
    assert fake_litellm[0]["api_base"] is None
    assert fake_litellm[0]["api_key"] == "or-key"
    assert provider.model_name == model


@pytest.mark.anyio
async def test_open_router_key_alias_is_passed_explicitly(fake_litellm, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPEN_ROUTER_KEY", "alias-key")
    await _generate(LiteLLMProvider("gpt-4o", api_base="https://api.openai.com/v1"))

    assert fake_litellm[0]["model"] == "openrouter/openai/gpt-4o"
    assert fake_litellm[0]["api_key"] == "alias-key"


@pytest.mark.anyio
@pytest.mark.usefixtures("openrouter_only")
@pytest.mark.parametrize(
    ("model", "api_base"),
    [
        ("anthropic/claude-3-opus-20240229", "https://api.anthropic.com"),
        ("litellm_proxy/openai/Olmo-3-7B-Instruct", "https://ai2-model-hub.allen.ai"),
        ("openai/Olmo-3-7B", "http://localhost:8000/v1"),
    ],
)
async def test_other_models_are_unchanged(fake_litellm, model: str, api_base: str) -> None:
    await _generate(LiteLLMProvider(model, api_base=api_base))

    assert fake_litellm[0]["model"] == model
    assert fake_litellm[0]["api_base"] == api_base
    assert "api_key" not in fake_litellm[0]


@pytest.mark.anyio
async def test_openai_key_keeps_openai_endpoint(fake_litellm, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    await _generate(LiteLLMProvider("gpt-4o", api_base="https://api.openai.com/v1"))

    assert fake_litellm[0]["model"] == "gpt-4o"
    assert fake_litellm[0]["api_base"] == "https://api.openai.com/v1"


@pytest.mark.anyio
@pytest.mark.usefixtures("openrouter_only")
async def test_logprobs_are_dropped_for_models_openai_rejects_them_on(fake_litellm) -> None:
    await _generate(LiteLLMProvider("openai/o3-mini-2025-01-31"))
    await _generate(LiteLLMProvider("gpt-4o"))

    o3_request, gpt4o_request = fake_litellm
    assert "logprobs" not in o3_request
    assert "top_logprobs" not in o3_request
    assert o3_request["temperature"] == SamplingParams().temperature
    assert gpt4o_request["logprobs"] is True


@pytest.mark.anyio
@pytest.mark.usefixtures("openrouter_only")
async def test_drop_params_drops_other_unsupported_params(fake_litellm) -> None:
    await _generate(LiteLLMProvider("openai/o3-mini-2025-01-31", drop_params=True, top_p=0.9))

    assert "top_p" not in fake_litellm[0]
    assert "logprobs" not in fake_litellm[0]


@pytest.mark.anyio
@pytest.mark.usefixtures("openrouter_only")
async def test_unsupported_params_are_kept_without_drop_params(fake_litellm) -> None:
    await _generate(LiteLLMProvider("openai/o3-mini-2025-01-31", top_p=0.9))

    assert fake_litellm[0]["top_p"] == 0.9


@pytest.mark.anyio
async def test_logprobs_are_kept_without_openrouter_routing(fake_litellm, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    await _generate(
        LiteLLMProvider("openai/o3-mini-2025-01-31", api_base="https://api.openai.com/v1")
    )

    assert fake_litellm[0]["logprobs"] is True


async def _logprobs(provider: LiteLLMProvider) -> None:
    request = LMRequest(request_type=RequestType.CHAT, prompt="hi", continuations=("a", "b"))
    await provider._logprobs_single_impl(request)


@pytest.mark.anyio
@pytest.mark.usefixtures("openrouter_only")
async def test_logprob_requests_route_to_openrouter(fake_litellm) -> None:
    await _logprobs(LiteLLMProvider("gpt-4o", api_base="https://api.openai.com/v1"))

    assert len(fake_litellm) == 2
    for call in fake_litellm:
        assert call["model"] == "openrouter/openai/gpt-4o"
        assert call["api_base"] is None
        assert call["api_key"] == "or-key"
        assert call["logprobs"] is True


@pytest.mark.anyio
@pytest.mark.usefixtures("openrouter_only")
async def test_logprob_requests_drop_params_openai_rejects(fake_litellm) -> None:
    await _logprobs(LiteLLMProvider("openai/o3-mini-2025-01-31"))

    assert fake_litellm[0]["model"] == "openrouter/openai/o3-mini"
    assert "logprobs" not in fake_litellm[0]
    assert "top_logprobs" not in fake_litellm[0]


@pytest.mark.anyio
async def test_logprob_requests_keep_openai_endpoint_with_openai_key(
    fake_litellm, monkeypatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    await _logprobs(LiteLLMProvider("gpt-4o", api_base="https://api.openai.com/v1"))

    assert fake_litellm[0]["model"] == "gpt-4o"
    assert fake_litellm[0]["api_base"] == "https://api.openai.com/v1"
    assert "api_key" not in fake_litellm[0]


@pytest.mark.anyio
@pytest.mark.usefixtures("openrouter_only")
async def test_loglikelihood_fails_for_models_without_logprobs(fake_litellm) -> None:
    provider = LiteLLMProvider("openai/o3-mini-2025-01-31")
    request = LMRequest(request_type=RequestType.CHAT, prompt="hi", continuations=("a",))

    with pytest.raises(ValueError, match="does not return logprobs"):
        await provider.alogprobs([request])
    assert fake_litellm == []


@pytest.mark.usefixtures("openrouter_only", "fake_litellm")
def test_openai_client_targets_openrouter() -> None:
    pytest.importorskip("openai")
    provider = LiteLLMProvider("gpt-4-0613", api_base="https://api.openai.com/v1")

    client = provider.get_openai_client()

    assert client is not None
    assert str(client.base_url).rstrip("/") == "https://openrouter.ai/api/v1"
    assert client.api_key == "or-key"
    assert provider.get_openai_client() is client
    assert provider.openai_client_model_name == "openai/gpt-4"
    assert provider.request_model_name == "openrouter/openai/gpt-4"
    assert provider.model_name == "gpt-4-0613"


@pytest.mark.usefixtures("fake_litellm")
def test_model_names_are_unchanged_with_openai_key(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    provider = LiteLLMProvider("gpt-4-0613", api_base="https://api.openai.com/v1")

    with pytest.raises(NotImplementedError):
        provider.get_openai_client()
    assert provider.openai_client_model_name == "gpt-4-0613"
    assert provider.request_model_name == "gpt-4-0613"


class TestProviderConfigSecrets:
    def _config(self) -> ProviderConfig:
        return ProviderConfig(model="gpt-4o", required_secrets=("OPENAI_API_KEY",))

    @pytest.mark.usefixtures("openrouter_only")
    def test_openrouter_key_satisfies_openai_requirement(self) -> None:
        assert self._config().validate_secrets() == []

    def test_missing_keys_are_reported(self, monkeypatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        assert self._config().validate_secrets() == ["OPENAI_API_KEY"]

    def test_missing_key_error_names_both_options(self, monkeypatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(ValueError, match=r"OPENAI_API_KEY \(or OPENROUTER_API_KEY\)"):
            self._config().create_provider()
