"""Credentials for OpenAI models, with OpenRouter as a fallback.

Anything that calls an OpenAI model (LLM judges, user simulators, LiteLLM
registry models) accepts either ``OPENAI_API_KEY`` or ``OPENROUTER_API_KEY``.
An OpenAI key wins when both are set. With only an OpenRouter key, requests go
to OpenRouter's OpenAI-compatible endpoint and bare OpenAI model names are
rewritten to OpenRouter's ``openai/<model>`` slugs.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai import AsyncOpenAI

__all__ = [
    "OPENAI_API_KEY_ENV",
    "OPENAI_BASE_URL",
    "OPENROUTER_API_KEY_ALIASES",
    "OPENROUTER_API_KEY_ENV",
    "OPENROUTER_BASE_URL",
    "SECRET_ALTERNATIVES",
    "OpenAICredentials",
    "create_async_openai_client",
    "describe_secret",
    "is_openai_model",
    "litellm_model_name",
    "openai_endpoint_model_name",
    "openai_sandbox_env",
    "openrouter_model_slug",
    "require_openai_credentials",
    "resolve_openai_credentials",
    "secret_available",
]

OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
# Other names an OpenRouter key may be stored under, checked after OPENROUTER_API_KEY.
OPENROUTER_API_KEY_ALIASES: tuple[str, ...] = ("OPEN_ROUTER_KEY",)
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENAI_BASE_URL = "https://api.openai.com/v1"

# Secrets that may stand in for a required secret, in order of preference.
SECRET_ALTERNATIVES: Mapping[str, tuple[str, ...]] = {
    OPENAI_API_KEY_ENV: (OPENROUTER_API_KEY_ENV, *OPENROUTER_API_KEY_ALIASES),
}


_OPENROUTER_OPENAI_PREFIX = "openai/"
_LITELLM_OPENROUTER_PREFIX = "openrouter/"
_BARE_OPENAI_MODEL_PREFIXES = ("gpt-", "chatgpt-", "o1", "o3", "o4")

# Dated OpenAI snapshots that OpenRouter serves only under the undated slug.
# Each target's canonical snapshot on OpenRouter is the same model.
_OPENROUTER_SNAPSHOT_ALIASES: Mapping[str, str] = {
    "gpt-4-0613": "gpt-4",
    "gpt-5.5-2026-04-23": "gpt-5.5",
    "o3-mini-2025-01-31": "o3-mini",
}


def secret_available(name: str, environ: Mapping[str, str] | None = None) -> bool:
    """Whether a required secret, or one that can stand in for it, is set."""
    env = os.environ if environ is None else environ
    return any(env.get(candidate) for candidate in (name, *SECRET_ALTERNATIVES.get(name, ())))


def describe_secret(name: str) -> str:
    """A secret's name plus the secrets that can stand in for it, for messages to users."""
    alternatives = [
        alt for alt in SECRET_ALTERNATIVES.get(name, ()) if alt not in OPENROUTER_API_KEY_ALIASES
    ]
    if not alternatives:
        return name
    return f"{name} (or {' or '.join(alternatives)})"


def openrouter_model_slug(model: str) -> str:
    """OpenRouter slug for an OpenAI model name, such as ``openai/gpt-4o``.

    Accepts bare names (``gpt-4o``) and ``openai/``-prefixed names. Names with
    another vendor prefix are returned unchanged.
    """
    name = model.removeprefix(_OPENROUTER_OPENAI_PREFIX)
    if "/" in name:
        return model
    return f"{_OPENROUTER_OPENAI_PREFIX}{_OPENROUTER_SNAPSHOT_ALIASES.get(name, name)}"


@dataclass(frozen=True)
class OpenAICredentials:
    """API key and endpoint for calling OpenAI models."""

    api_key: str
    base_url: str | None = None

    @property
    def via_openrouter(self) -> bool:
        """Whether requests are routed through OpenRouter."""
        return self.base_url == OPENROUTER_BASE_URL

    def model_name(self, model: str) -> str:
        """Model name to send to the OpenAI-compatible endpoint."""
        return openrouter_model_slug(model) if self.via_openrouter else model


def resolve_openai_credentials(
    environ: Mapping[str, str] | None = None,
) -> OpenAICredentials | None:
    """Find credentials for OpenAI models in the environment.

    Returns None when neither ``OPENAI_API_KEY`` nor an OpenRouter key
    (``OPENROUTER_API_KEY`` or one of its aliases) is set.
    """
    env = os.environ if environ is None else environ
    if api_key := env.get(OPENAI_API_KEY_ENV):
        return OpenAICredentials(api_key=api_key)
    for name in (OPENROUTER_API_KEY_ENV, *OPENROUTER_API_KEY_ALIASES):
        if api_key := env.get(name):
            return OpenAICredentials(api_key=api_key, base_url=OPENROUTER_BASE_URL)
    return None


def require_openai_credentials(purpose: str) -> OpenAICredentials:
    """Like ``resolve_openai_credentials`` but raise when no key is set.

    Args:
        purpose: What needs the key, used in the error message.
    """
    credentials = resolve_openai_credentials()
    if credentials is None:
        raise ValueError(
            f"{OPENAI_API_KEY_ENV} or {OPENROUTER_API_KEY_ENV} is required for {purpose}."
        )
    return credentials


def create_async_openai_client(credentials: OpenAICredentials) -> AsyncOpenAI:
    """Create an AsyncOpenAI client for the given credentials."""
    try:
        from openai import AsyncOpenAI
    except ImportError:
        raise ImportError("openai package required: pip install openai") from None
    if credentials.base_url is None:
        return AsyncOpenAI(api_key=credentials.api_key)
    return AsyncOpenAI(api_key=credentials.api_key, base_url=credentials.base_url)


def is_openai_model(model: str, api_base: str | None = None) -> bool:
    """Whether a LiteLLM model name and API base target OpenAI's own API.

    Args:
        model: LiteLLM model name, such as ``gpt-4o`` or ``openai/gpt-4o-mini``.
        api_base: API base the model is called with, if any.
    """
    if api_base is not None:
        return api_base.rstrip("/") == OPENAI_BASE_URL
    if model.startswith(_OPENROUTER_OPENAI_PREFIX):
        return True
    return "/" not in model and model.startswith(_BARE_OPENAI_MODEL_PREFIXES)


def litellm_model_name(model: str, environ: Mapping[str, str] | None = None) -> str:
    """LiteLLM model name for an OpenAI model, routed by the available key.

    With only an OpenRouter key, ``gpt-4o`` and ``openai/gpt-4o`` both become
    ``openrouter/openai/gpt-4o``. Other models, and all models when an OpenAI
    key is available, are returned unchanged.
    """
    credentials = resolve_openai_credentials(environ)
    if credentials is None or not credentials.via_openrouter or not is_openai_model(model):
        return model
    return f"{_LITELLM_OPENROUTER_PREFIX}{openrouter_model_slug(model)}"


def openai_endpoint_model_name(model: str, environ: Mapping[str, str] | None = None) -> str:
    """Name to send to the OpenAI-compatible endpoint that ``openai_sandbox_env`` sets up.

    With only an OpenRouter key, OpenAI models become OpenRouter slugs such as
    ``openai/gpt-4o``, which both the OpenAI SDK and LiteLLM's ``openai/`` provider
    accept. Other models, and all models when an OpenAI key is available, are
    returned unchanged.
    """
    credentials = resolve_openai_credentials(environ)
    if credentials is None or not credentials.via_openrouter or not is_openai_model(model):
        return model
    return openrouter_model_slug(model)


def openai_sandbox_env(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """Environment variables that give third-party code access to OpenAI models.

    With an OpenAI key this is just ``OPENAI_API_KEY``. With only an OpenRouter
    key, the OpenRouter key is exported under both names and ``OPENAI_BASE_URL``
    points the OpenAI SDK at OpenRouter. Returns an empty dict when no key is set.
    """
    credentials = resolve_openai_credentials(environ)
    if credentials is None:
        return {}
    if not credentials.via_openrouter:
        return {OPENAI_API_KEY_ENV: credentials.api_key}
    return {
        OPENROUTER_API_KEY_ENV: credentials.api_key,
        OPENAI_API_KEY_ENV: credentials.api_key,
        "OPENAI_BASE_URL": OPENROUTER_BASE_URL,
    }
