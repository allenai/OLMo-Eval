"""Batch processing configuration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, ClassVar


class BatchStrategy(StrEnum):
    """Available batching strategies."""

    BATCHED = "batched"  # Process in chunks with continuous dispatch
    STREAMING = "streaming"  # Stream items directly to provider


# Default values
DEFAULT_CHUNK_SIZE = 64
DEFAULT_CHUNK_TIMEOUT = 5.0


@dataclass(frozen=True)
class BatchConfig:
    """Configuration for batch processing.

    Attributes:
        strategy: Batching strategy to use.
        chunk_size: Maximum items per batch.
        chunk_timeout: Seconds to wait for batch to fill before processing partial.
    """

    # Providers that only support sequential (LLM() is not thread-safe)
    _SEQUENTIAL_ONLY: ClassVar[frozenset[str]] = frozenset(
        {"vllm", "hf", "olmo_core", "olmo_core_vlm"}
    )

    strategy: BatchStrategy = BatchStrategy.BATCHED
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_timeout: float = DEFAULT_CHUNK_TIMEOUT

    def validate_for_provider(self, provider_kind: str) -> None:
        """Validate that this batching config is compatible with the provider.

        Args:
            provider_kind: The provider type (e.g., "vllm", "vllm_server").

        Raises:
            ValueError: If the batching strategy is not supported by the provider.
        """
        if provider_kind in self._SEQUENTIAL_ONLY and self.strategy != BatchStrategy.BATCHED:
            raise ValueError(
                f"Provider '{provider_kind}' only supports batched processing. "
                f"Use vllm_server for streaming."
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "strategy": str(self.strategy),
            "chunk_size": self.chunk_size,
            "chunk_timeout": self.chunk_timeout,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BatchConfig:
        """Create from dictionary."""
        strategy = data.get("strategy", BatchStrategy.BATCHED)
        if isinstance(strategy, str):
            strategy = BatchStrategy(strategy)
        return cls(
            strategy=strategy,
            chunk_size=data.get("chunk_size", DEFAULT_CHUNK_SIZE),
            chunk_timeout=data.get("chunk_timeout", DEFAULT_CHUNK_TIMEOUT),
        )

    @classmethod
    def batched(cls, chunk_size: int = DEFAULT_CHUNK_SIZE) -> BatchConfig:
        """Create batched config with continuous dispatch within each chunk."""
        return cls(strategy=BatchStrategy.BATCHED, chunk_size=chunk_size)

    @classmethod
    def streaming(cls) -> BatchConfig:
        """Create streaming config (items sent directly to provider)."""
        return cls(strategy=BatchStrategy.STREAMING)
