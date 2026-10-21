"""Inference provider implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .config import ProviderConfig

if TYPE_CHECKING:
    from .huggingface import HuggingFaceProvider
    from .litellm import LiteLLMProvider
    from .mock import MockProvider
    from .olmo_core import OlmoCoreProvider
    from .olmo_core_vlm import OlmoCoreVLMProvider
    from .vllm import VLLMProvider
    from .vllm_server import VLLMServerProvider

__all__ = [
    "HuggingFaceProvider",
    "LiteLLMProvider",
    "MockProvider",
    "OlmoCoreProvider",
    "OlmoCoreVLMProvider",
    "ProviderConfig",
    "VLLMProvider",
    "VLLMServerProvider",
]


def __getattr__(name: str):
    if name == "HuggingFaceProvider":
        from .huggingface import HuggingFaceProvider

        return HuggingFaceProvider
    if name == "LiteLLMProvider":
        from .litellm import LiteLLMProvider

        return LiteLLMProvider
    if name == "MockProvider":
        from .mock import MockProvider

        return MockProvider
    if name == "OlmoCoreProvider":
        from .olmo_core import OlmoCoreProvider

        return OlmoCoreProvider
    if name == "OlmoCoreVLMProvider":
        from .olmo_core_vlm import OlmoCoreVLMProvider

        return OlmoCoreVLMProvider
    if name == "VLLMProvider":
        from .vllm import VLLMProvider

        return VLLMProvider
    if name == "VLLMServerProvider":
        from .vllm_server import VLLMServerProvider

        return VLLMServerProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
