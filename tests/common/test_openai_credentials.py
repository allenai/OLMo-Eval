"""Tests for OpenAI credential resolution with the OpenRouter fallback."""

from __future__ import annotations

import sys
import types

import pytest

from olmo_eval.common.openai_credentials import (
    OPENROUTER_BASE_URL,
    OpenAICredentials,
    create_async_openai_client,
    describe_secret,
    is_openai_model,
    litellm_model_name,
    openai_endpoint_model_name,
    openai_sandbox_env,
    openrouter_model_slug,
    require_openai_credentials,
    resolve_openai_credentials,
    secret_available,
)

OPENAI_ONLY = {"OPENAI_API_KEY": "sk-openai"}
OPENROUTER_ONLY = {"OPENROUTER_API_KEY": "sk-or"}
BOTH = {**OPENAI_ONLY, **OPENROUTER_ONLY}


class TestResolveOpenAICredentials:
    def test_openai_key(self) -> None:
        credentials = resolve_openai_credentials(OPENAI_ONLY)
        assert credentials == OpenAICredentials(api_key="sk-openai")
        assert not credentials.via_openrouter

    def test_openrouter_key(self) -> None:
        credentials = resolve_openai_credentials(OPENROUTER_ONLY)
        assert credentials == OpenAICredentials(api_key="sk-or", base_url=OPENROUTER_BASE_URL)
        assert credentials.via_openrouter

    def test_openai_key_wins_when_both_are_set(self) -> None:
        credentials = resolve_openai_credentials(BOTH)
        assert credentials is not None
        assert credentials.api_key == "sk-openai"

    def test_no_key(self) -> None:
        assert resolve_openai_credentials({}) is None

    def test_open_router_key_alias(self) -> None:
        assert resolve_openai_credentials({"OPEN_ROUTER_KEY": "sk-alias"}) == OpenAICredentials(
            api_key="sk-alias", base_url=OPENROUTER_BASE_URL
        )

    def test_canonical_openrouter_name_wins_over_alias(self) -> None:
        environ = {**OPENROUTER_ONLY, "OPEN_ROUTER_KEY": "sk-alias"}
        credentials = resolve_openai_credentials(environ)
        assert credentials is not None
        assert credentials.api_key == "sk-or"

    def test_empty_values_are_ignored(self) -> None:
        assert resolve_openai_credentials({"OPENAI_API_KEY": "", **OPENROUTER_ONLY}) == (
            OpenAICredentials(api_key="sk-or", base_url=OPENROUTER_BASE_URL)
        )

    def test_require_reads_the_process_environment(self, monkeypatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or")
        assert require_openai_credentials("a judge").via_openrouter

    def test_require_names_both_keys_when_missing(self, monkeypatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OPENAI_API_KEY or OPENROUTER_API_KEY .* a judge"):
            require_openai_credentials("a judge")


class TestModelNames:
    @pytest.mark.parametrize(
        ("model", "slug"),
        [
            ("gpt-4o", "openai/gpt-4o"),
            ("openai/gpt-4o-mini", "openai/gpt-4o-mini"),
            ("gpt-4o-2024-05-13", "openai/gpt-4o-2024-05-13"),
            ("gpt-4-0613", "openai/gpt-4"),
            ("o3-mini-2025-01-31", "openai/o3-mini"),
            ("openai/o3-mini-2025-01-31", "openai/o3-mini"),
            ("gpt-5.5-2026-04-23", "openai/gpt-5.5"),
            ("anthropic/claude-3-opus", "anthropic/claude-3-opus"),
        ],
    )
    def test_openrouter_model_slug(self, model: str, slug: str) -> None:
        assert openrouter_model_slug(model) == slug

    def test_model_name_is_unchanged_with_openai_key(self) -> None:
        assert OpenAICredentials(api_key="k").model_name("gpt-4-0613") == "gpt-4-0613"

    def test_model_name_uses_slug_with_openrouter(self) -> None:
        credentials = OpenAICredentials(api_key="k", base_url=OPENROUTER_BASE_URL)
        assert credentials.model_name("gpt-4-0613") == "openai/gpt-4"

    @pytest.mark.parametrize(
        ("model", "api_base", "expected"),
        [
            ("gpt-4o", None, True),
            ("o3-mini", None, True),
            ("openai/gpt-4o-mini", None, True),
            ("gpt-4o", "https://api.openai.com/v1/", True),
            ("openai/Olmo-3-7B", "http://localhost:8000/v1", False),
            ("claude-3-opus-20240229", None, False),
            ("anthropic/claude-3-opus-20240229", None, False),
            ("litellm_proxy/openai/Olmo-3-7B-Instruct", None, False),
        ],
    )
    def test_is_openai_model(self, model: str, api_base: str | None, expected: bool) -> None:
        assert is_openai_model(model, api_base) is expected

    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("gpt-4o", "openrouter/openai/gpt-4o"),
            ("openai/gpt-4o-mini", "openrouter/openai/gpt-4o-mini"),
            ("openrouter/openai/gpt-4o", "openrouter/openai/gpt-4o"),
            ("anthropic/claude-3-opus-20240229", "anthropic/claude-3-opus-20240229"),
        ],
    )
    def test_litellm_model_name_with_openrouter(self, model: str, expected: str) -> None:
        assert litellm_model_name(model, OPENROUTER_ONLY) == expected

    @pytest.mark.parametrize("environ", [OPENAI_ONLY, BOTH, {}])
    def test_litellm_model_name_without_openrouter_fallback(self, environ) -> None:
        assert litellm_model_name("gpt-4o", environ) == "gpt-4o"

    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("gpt-4o", "openai/gpt-4o"),
            ("gpt-4-0613", "openai/gpt-4"),
            ("openai/gpt-5.5-2026-04-23", "openai/gpt-5.5"),
            ("anthropic/claude-3-opus-20240229", "anthropic/claude-3-opus-20240229"),
        ],
    )
    def test_openai_endpoint_model_name_with_openrouter(self, model: str, expected: str) -> None:
        assert openai_endpoint_model_name(model, OPENROUTER_ONLY) == expected

    @pytest.mark.parametrize("environ", [OPENAI_ONLY, BOTH, {}])
    def test_openai_endpoint_model_name_without_openrouter_fallback(self, environ) -> None:
        assert openai_endpoint_model_name("gpt-4-0613", environ) == "gpt-4-0613"


class TestSandboxEnv:
    def test_openai_key(self) -> None:
        assert openai_sandbox_env(OPENAI_ONLY) == {"OPENAI_API_KEY": "sk-openai"}

    def test_openrouter_key(self) -> None:
        assert openai_sandbox_env(OPENROUTER_ONLY) == {
            "OPENROUTER_API_KEY": "sk-or",
            "OPENAI_API_KEY": "sk-or",
            "OPENAI_BASE_URL": OPENROUTER_BASE_URL,
        }

    def test_no_key(self) -> None:
        assert openai_sandbox_env({}) == {}


class TestSecretAvailable:
    @pytest.mark.parametrize("environ", [OPENAI_ONLY, OPENROUTER_ONLY])
    def test_openai_requirement(self, environ) -> None:
        assert secret_available("OPENAI_API_KEY", environ)

    def test_openai_requirement_with_alias(self) -> None:
        assert secret_available("OPENAI_API_KEY", {"OPEN_ROUTER_KEY": "k"})

    def test_openai_requirement_missing(self) -> None:
        assert not secret_available("OPENAI_API_KEY", {})

    def test_secret_without_alternatives(self) -> None:
        assert not secret_available("ANTHROPIC_API_KEY", OPENROUTER_ONLY)
        assert secret_available("ANTHROPIC_API_KEY", {"ANTHROPIC_API_KEY": "k"})


def test_create_async_openai_client_passes_base_url(monkeypatch) -> None:
    created: list[dict] = []

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            created.append(kwargs)

    module = types.ModuleType("openai")
    module.AsyncOpenAI = FakeAsyncOpenAI  # type: ignore[ty:unresolved-attribute]
    monkeypatch.setitem(sys.modules, "openai", module)

    create_async_openai_client(OpenAICredentials(api_key="sk-openai"))
    create_async_openai_client(OpenAICredentials(api_key="sk-or", base_url=OPENROUTER_BASE_URL))
    assert created == [
        {"api_key": "sk-openai"},
        {"api_key": "sk-or", "base_url": OPENROUTER_BASE_URL},
    ]


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("OPENAI_API_KEY", "OPENAI_API_KEY (or OPENROUTER_API_KEY)"),
        ("S2_API_KEY", "S2_API_KEY"),
    ],
)
def test_describe_secret(name: str, expected: str) -> None:
    assert describe_secret(name) == expected
