"""Configuration for inference providers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

from olmo_eval.common.openai_credentials import secret_available
from olmo_eval.common.repr import hide_unset
from olmo_eval.common.types import ProviderKind

if TYPE_CHECKING:
    from olmo_eval.inference.base import InferenceProvider


@hide_unset()
@dataclass(frozen=True)
class ProviderConfig:
    """Immutable configuration for creating an InferenceProvider.

    This configuration contains all the information needed to instantiate
    a provider via the create_provider factory function.

    Attributes:
        kind: Provider type (vllm, vllm_server, hf, olmo_core, litellm, mock).
        model: Model name or path (HuggingFace ID, API model name, or local path).
        alias: Short display name for the model (used in DB and S3 paths).
        base_url: Base URL for API-based providers (vllm_server, litellm).
        tokenizer: Tokenizer path/identifier (defaults to model if None).
        revision: Model revision/commit hash for HuggingFace models.
        force_download: Force re-download of HuggingFace model/tokenizer files and refresh
            the local cache before loading.
        trust_remote_code: Whether to trust remote code for HuggingFace models.
        dtype: Data type for model weights (auto, float16, bfloat16, float32).
        max_model_len: Maximum sequence length (overrides model default).
        max_concurrency: Maximum concurrent requests.
        num_instances: Number of server instances for horizontal scaling.
        required_secrets: Environment variable names that must be set.
        package: Package specifier that overrides the default provider extra (e.g., vllm fork).
        dependencies: Additional package specifiers for runtime installation.
        kwargs: Additional arguments passed to the provider constructor.
    """

    kind: str = ProviderKind.VLLM_SERVER
    model: str = ""
    alias: str | None = None
    base_url: str | None = None
    api_base: str | None = None
    tokenizer: str | None = None
    revision: str | None = None
    force_download: bool = False
    trust_remote_code: bool = False
    dtype: str = "auto"
    max_model_len: int | None = None
    max_concurrency: int | None = None
    num_instances: int = 1
    required_secrets: tuple[str, ...] = ()
    package: str | None = None
    dependencies: tuple[str, ...] = ()
    kwargs: Mapping[str, Any] = field(default_factory=dict)

    # Providers that require GPU resources for local inference
    _GPU_PROVIDERS: ClassVar[frozenset[str]] = frozenset(
        {"vllm", "vllm_server", "hf", "olmo_core", "olmo_core_vlm"}
    )

    @property
    def requires_gpu(self) -> bool:
        """Whether this provider requires GPU resources.

        Returns True for local inference providers (vllm, vllm_server, hf, olmo_core).
        Returns False for API-based providers (litellm, mock).
        """
        return str(self.kind) in self._GPU_PROVIDERS

    @property
    def requires_local_gpu(self) -> bool:
        """Whether this provider needs local GPU resources.

        Returns True for local inference (vllm, vllm_server, hf, olmo_core)
        without external base_url.
        Returns False for API-backed providers or configs pointing to external servers.
        """
        if not self.requires_gpu:
            return False
        # vllm_server with base_url points to external server
        return not self.base_url

    # Config fields accepted by each provider kind (fields with non-default values are passed)
    _PROVIDER_FIELDS: ClassVar[dict[str, tuple[str, ...]]] = {
        "vllm": (
            "tokenizer",
            "revision",
            "force_download",
            "trust_remote_code",
            "dtype",
            "max_model_len",
        ),
        "vllm_server": (
            "base_url",
            "tokenizer",
            "revision",
            "force_download",
            "trust_remote_code",
            "dtype",
            "max_concurrency",
            "max_model_len",
        ),
        "litellm": ("base_url", "api_base", "max_concurrency"),
        "hf": ("tokenizer", "revision", "force_download", "trust_remote_code", "dtype"),
        "olmo_core": (
            "tokenizer",
            "revision",
            "force_download",
            "trust_remote_code",
            "dtype",
            "max_model_len",
        ),
        "olmo_core_vlm": (
            "tokenizer",
            "revision",
            "force_download",
            "trust_remote_code",
            "dtype",
            "max_model_len",
        ),
        "mock": (),
    }

    def create_provider(self) -> InferenceProvider:
        """Create an InferenceProvider from this configuration."""
        from olmo_eval.inference import create_provider

        missing = self.validate_secrets()
        if missing:
            raise ValueError(f"Missing required secrets: {', '.join(missing)}")

        kind_str = str(self.kind)
        provider_kwargs: dict[str, Any] = dict(self.kwargs)

        # Add config fields this provider accepts (skip None/default values)
        for field_name in self._PROVIDER_FIELDS.get(kind_str, ()):
            value = getattr(self, field_name)
            if value is not None and value is not False and value != "auto":
                provider_kwargs[field_name] = value

        return create_provider(self.kind, self.model, **provider_kwargs)

    def get_provider_name(self, override: str | None = None) -> str:
        """Get the effective provider name as a string.

        Args:
            override: Optional provider name override.

        Returns:
            Provider name string (e.g., "vllm", "litellm", "hf").
        """
        if override:
            return override
        kind = self.kind
        return str(kind.value) if hasattr(kind, "value") else str(kind)

    def validate_secrets(self) -> list[str]:
        """Check that all required secrets are available.

        Returns:
            List of missing secret names (empty if all present).
        """
        return [secret for secret in self.required_secrets if not secret_available(secret)]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation suitable for JSON serialization.
        """
        d: dict[str, Any] = {
            "kind": self.kind,
            "model": self.model,
        }
        if self.base_url is not None:
            d["base_url"] = self.base_url
        if self.api_base is not None:
            d["api_base"] = self.api_base
        if self.tokenizer is not None:
            d["tokenizer"] = self.tokenizer
        if self.revision is not None:
            d["revision"] = self.revision
        if self.force_download:
            d["force_download"] = self.force_download
        if self.trust_remote_code:
            d["trust_remote_code"] = self.trust_remote_code
        if self.dtype != "auto":
            d["dtype"] = self.dtype
        if self.max_model_len is not None:
            d["max_model_len"] = self.max_model_len
        if self.max_concurrency is not None:
            d["max_concurrency"] = self.max_concurrency
        if self.num_instances != 1:
            d["num_instances"] = self.num_instances
        if self.required_secrets:
            d["required_secrets"] = list(self.required_secrets)
        if self.package:
            d["package"] = self.package
        if self.dependencies:
            d["dependencies"] = list(self.dependencies)
        if self.kwargs:
            d["kwargs"] = dict(self.kwargs)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProviderConfig:
        """Create from dictionary.

        Args:
            data: Dictionary with ProviderConfig data.

        Returns:
            A new ProviderConfig instance.

        Raises:
            ValueError: If dependencies is a string instead of a list.
        """
        deps = data.get("dependencies", [])
        if isinstance(deps, str):
            raise ValueError(
                f"provider.dependencies must be a list, not a string: {deps!r}. "
                "Use provider.dependencies=[url1,url2] syntax."
            )
        # Unknown keys flow into kwargs (same semantics as with_overrides), so
        # dotted CLI overrides like `-o provider.batch_size=8` reach the provider
        # constructor instead of being silently dropped.
        from dataclasses import fields as dc_fields

        known = {f.name for f in dc_fields(cls)} | {"model_name"}
        extra_kwargs = {k: v for k, v in data.items() if k not in known}
        merged_kwargs = {**data.get("kwargs", {}), **extra_kwargs}
        return cls(
            kind=data.get("kind", ProviderKind.VLLM),
            model=data.get("model", data.get("model_name", "")),
            base_url=data.get("base_url"),
            api_base=data.get("api_base"),
            tokenizer=data.get("tokenizer"),
            revision=data.get("revision"),
            force_download=data.get("force_download", False),
            trust_remote_code=data.get("trust_remote_code", False),
            dtype=data.get("dtype", "auto"),
            max_model_len=data.get("max_model_len"),
            max_concurrency=data.get("max_concurrency"),
            num_instances=data.get("num_instances", 1),
            required_secrets=tuple(data.get("required_secrets", [])),
            package=data.get("package"),
            dependencies=tuple(deps),
            kwargs=merged_kwargs,
        )

    def with_overrides(self, **overrides: Any) -> ProviderConfig:
        """Create a new config with overrides applied.

        Known field names are set directly on the config. Unknown names
        are merged into kwargs. None values are ignored.

        Args:
            **overrides: Field overrides (e.g., max_model_len=4096).

        Returns:
            New ProviderConfig with overrides applied.

        Example:
            config = provider_config.with_overrides(
                tensor_parallel_size=4,
                max_model_len=8192,
                tokenizer="/path/to/tokenizer",
            )
        """
        from dataclasses import fields, replace

        # Separate known fields from kwargs
        field_names = {f.name for f in fields(self)}
        field_overrides = {}
        extra_kwargs = dict(self.kwargs)

        for key, value in overrides.items():
            if value is None:
                continue
            if key in field_names and key != "kwargs":
                field_overrides[key] = value
            else:
                extra_kwargs[key] = value

        if extra_kwargs != dict(self.kwargs):
            field_overrides["kwargs"] = extra_kwargs

        return replace(self, **field_overrides) if field_overrides else self
