"""Tests for routing OpenAI models through OpenRouter in the LiteLLM provider."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

from olmo_eval.common.types import LMRequest, RequestType, SamplingParams
from olmo_eval.inference.providers.config import ProviderConfig
from olmo_eval.inference.providers.litellm import LiteLLMProvider


@pytest.fixture
def fake_litellm(monkeypatch) -> list[dict]:
    """Install a stub ``litellm`` module and return the recorded request kwargs."""
    calls: list[dict] = []

    async def acompletion(**kwargs):
        calls.append(kwargs)
        message = SimpleNamespace(content="ok")
        return SimpleNamespace(choices=[SimpleNamespace(message=message, logprobs=None)])

    module = types.ModuleType("litellm")
    module.acompletion = acompletion  # type: ignore[ty:unresolved-attribute]
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
    assert provider.model_name == model


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


@pytest.mark.anyio
async def test_openai_key_keeps_openai_endpoint(fake_litellm, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    await _generate(LiteLLMProvider("gpt-4o", api_base="https://api.openai.com/v1"))

    assert fake_litellm[0]["model"] == "gpt-4o"
    assert fake_litellm[0]["api_base"] == "https://api.openai.com/v1"


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
