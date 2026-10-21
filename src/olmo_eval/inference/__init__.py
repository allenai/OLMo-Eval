"""Language model inference providers."""

from olmo_eval.common.types import ProviderKind

from .base import InferenceProvider
from .tokenizer_utils import (
    encode_context_and_continuation,
    get_bos_token_ids,
    get_context_token_ids,
    has_bos_token,
)

__all__ = [
    "InferenceProvider",
    "ProviderKind",
    "MockProvider",
    "HuggingFaceProvider",
    "VLLMProvider",
    "VLLMServerProvider",
    "OlmoCoreProvider",
    "OlmoCoreVLMProvider",
    "LiteLLMProvider",
    "create_provider",
    # Tokenizer utilities
    "encode_context_and_continuation",
    "get_bos_token_ids",
    "get_context_token_ids",
    "has_bos_token",
    # Metrics (lazy import via __getattr__)
    "metrics",
]


def _print_model_config(
    model_name: str,
    *,
    revision: str | None = None,
    trust_remote_code: bool = True,
    force_download: bool = False,
) -> None:
    """Print the model architecture/config when loading a model."""
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.pretty import Pretty
        from transformers import AutoConfig

        console = Console()
        config = AutoConfig.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=trust_remote_code,
            force_download=force_download,
        )
        architectures = getattr(config, "architectures", ["Unknown"])
        config_dict = config.to_dict()

        console.print(
            Panel(
                Pretty(config_dict),
                title=f"[bold blue]Model Config: {model_name}[/bold blue]",
                subtitle=f"[dim]Architecture: {architectures}[/dim]",
            )
        )
    except Exception as e:
        from rich import print as rprint

        rprint(f"[yellow]Could not load model config for {model_name}: {e}[/yellow]")


def create_provider(
    provider_kind: ProviderKind | str,
    model_name: str,
    worker_id: str | None = None,
    **kwargs,
) -> InferenceProvider:
    """Create a provider instance.

    Args:
        provider_kind: Kind of provider to create (e.g., "vllm", "vllm_server", "litellm").
        model_name: Model identifier or path.
        worker_id: Optional worker identifier for logging (only used by vLLM).
        **kwargs: Additional arguments passed to provider constructor.

    Returns:
        Initialized provider instance.

    Raises:
        ValueError: If provider kind is unknown.
    """
    # Normalize to string for comparison (StrEnum compares equal to its value)
    kind_str = str(provider_kind)
    revision = kwargs.get("revision")
    trust_remote_code = kwargs.get("trust_remote_code", True)
    force_download = bool(kwargs.get("force_download", False))

    match kind_str:
        case "mock":
            from .providers.mock import MockProvider

            return MockProvider(model_name)
        case "hf":
            from .providers.huggingface import HuggingFaceProvider

            _print_model_config(
                model_name,
                revision=revision,
                trust_remote_code=trust_remote_code,
                force_download=force_download,
            )
            return HuggingFaceProvider(model_name, **kwargs)
        case "vllm":
            from .providers.vllm import VLLMProvider

            _print_model_config(
                model_name,
                revision=revision,
                trust_remote_code=trust_remote_code,
                force_download=force_download,
            )
            return VLLMProvider(model_name, worker_id=worker_id, **kwargs)
        case "vllm_server":
            from .providers.vllm_server import VLLMServerProvider

            _print_model_config(
                model_name,
                revision=revision,
                trust_remote_code=trust_remote_code,
                force_download=force_download,
            )
            return VLLMServerProvider(model_name, **kwargs)
        case "olmo_core":
            from .providers.olmo_core_vlm import is_multimodal_checkpoint

            if is_multimodal_checkpoint(model_name):
                import logging

                logging.getLogger(__name__).warning(
                    "provider.kind=olmo_core detected a multimodal checkpoint and is "
                    "using OlmoCoreVLMProvider; pass provider.kind=olmo_core_vlm "
                    "explicitly so remote jobs install the matching olmo_core_vlm extra."
                )
                from .providers.olmo_core_vlm import OlmoCoreVLMProvider

                return OlmoCoreVLMProvider(model_name, **kwargs)

            from .providers.olmo_core import OlmoCoreProvider

            return OlmoCoreProvider(model_name, **kwargs)
        case "olmo_core_vlm":
            from .providers.olmo_core_vlm import OlmoCoreVLMProvider

            return OlmoCoreVLMProvider(model_name, **kwargs)
        case "litellm":
            from .providers.litellm import LiteLLMProvider

            return LiteLLMProvider(model_name, **kwargs)
        case _:
            raise ValueError(f"Unknown provider kind: {provider_kind}")


# Lazy imports for optional dependencies
def __getattr__(name: str):
    if name == "MockProvider":
        from .providers.mock import MockProvider

        return MockProvider
    if name == "HuggingFaceProvider":
        from .providers.huggingface import HuggingFaceProvider

        return HuggingFaceProvider
    if name == "VLLMProvider":
        from .providers.vllm import VLLMProvider

        return VLLMProvider
    if name == "VLLMServerProvider":
        from .providers.vllm_server import VLLMServerProvider

        return VLLMServerProvider
    if name == "OlmoCoreProvider":
        from .providers.olmo_core import OlmoCoreProvider

        return OlmoCoreProvider
    if name == "OlmoCoreVLMProvider":
        from .providers.olmo_core_vlm import OlmoCoreVLMProvider

        return OlmoCoreVLMProvider
    if name == "LiteLLMProvider":
        from .providers.litellm import LiteLLMProvider

        return LiteLLMProvider
    if name == "metrics":
        from . import metrics

        return metrics
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
