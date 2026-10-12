"""Pluggable execution scaffolds for Harness.

Scaffolds are registered using the @register_scaffold decorator and define
how the harness executes requests.
"""

from __future__ import annotations

import logging
from typing import Any, TypeVar

from olmo_eval.common.types import LMRequest, SamplingParams
from olmo_eval.harness.config import HarnessConfig
from olmo_eval.harness.result import HarnessResult
from olmo_eval.inference.base import InferenceProvider

logger = logging.getLogger(__name__)


class Scaffold:
    """Base class for harness execution scaffolds.

    Scaffolds define how the harness executes requests. Scaffolds may optionally
    implement `run` for multi-turn execution support.

    Class Attributes:
        name: Human-readable name for this scaffold.
        required_extras: Tuple of pyproject.toml extra names required by this scaffold.

    Instance Attributes:
        _sandbox_manager: Optional sandbox manager for scaffolds that need sandbox access.
    """

    name: str = "base"
    required_extras: tuple[str, ...] = ()
    _sandbox_manager: Any = None

    def set_sandbox_manager(self, sandbox_manager: Any) -> None:
        """Set an external sandbox manager for this scaffold.

        Use this to inject a pre-configured sandbox manager instead of
        letting the scaffold create its own during initialize().

        Args:
            sandbox_manager: SandboxManager instance to use.
        """
        self._sandbox_manager = sandbox_manager

    async def initialize(self, config: HarnessConfig) -> None:
        """Initialize scaffold resources like sandbox managers.

        Called during worker startup before processing begins.
        Subclasses should override to set up resources that need
        to be ready before the first request.

        Args:
            config: Harness configuration.
        """
        pass

    async def run(
        self,
        provider: InferenceProvider,
        config: HarnessConfig,
        request: LMRequest,
        sampling_params: SamplingParams | None = None,
        trace_metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> HarnessResult:
        """Execute the request and return the result.

        Args:
            provider: The inference provider for model calls.
            config: Harness configuration (tools, system prompt, etc.).
            request: The initial request to process.
            sampling_params: Optional sampling parameters override.
            trace_metadata: Optional metadata for tracing (e.g., instance_id, task_id).
            **kwargs: Scaffold-specific keyword arguments (e.g., enable_compaction).

        Returns:
            HarnessResult with trajectory and final output.

        Raises:
            NotImplementedError: If this scaffold doesn't support run().
        """
        raise NotImplementedError(f"Scaffold '{self.name}' does not support run()")

    async def cleanup(self) -> None:
        """Clean up resources held by this scaffold.

        Subclasses should override this to clean up any resources like
        sandbox managers, connections, etc.
        """
        pass


# -----------------------------------------------------------------------------
# Scaffold Registry
# -----------------------------------------------------------------------------

SCAFFOLD_REGISTRY: dict[str, type[Scaffold]] = {}

T = TypeVar("T", bound=Scaffold)


def register_scaffold(name: str):
    """Decorator to register a Scaffold class.

    Usage:
        @register_scaffold("my_scaffold")
        class MyScaffold(Scaffold):
            ...

    Args:
        name: Name to register the scaffold under.

    Returns:
        Decorator function that registers the class.
    """

    def decorator(cls: type[T]) -> type[T]:
        if name in SCAFFOLD_REGISTRY:
            logger.warning(f"Overwriting existing scaffold: {name}")
        SCAFFOLD_REGISTRY[name] = cls
        return cls

    return decorator


def get_scaffold(name: str) -> Scaffold:
    """Get a scaffold instance by name.

    Args:
        name: Scaffold name (e.g., "openai_agents", "openhands").

    Returns:
        Scaffold instance.

    Raises:
        ValueError: If scaffold name is unknown.
    """
    if name not in SCAFFOLD_REGISTRY:
        available = ", ".join(sorted(SCAFFOLD_REGISTRY.keys()))
        raise ValueError(f"Unknown scaffold: '{name}'. Available: {available}")
    return SCAFFOLD_REGISTRY[name]()


def list_scaffolds() -> list[str]:
    """List all registered scaffold names.

    Returns:
        Sorted list of scaffold names.
    """
    return sorted(SCAFFOLD_REGISTRY.keys())


def get_scaffold_extras(name: str) -> tuple[str, ...]:
    """Get the required extras for a scaffold by name.

    Args:
        name: Scaffold name (e.g., "openai_agents", "openhands").

    Returns:
        Tuple of pyproject.toml extra names required by the scaffold.

    Raises:
        ValueError: If scaffold name is unknown.
    """
    if name not in SCAFFOLD_REGISTRY:
        available = ", ".join(sorted(SCAFFOLD_REGISTRY.keys()))
        raise ValueError(f"Unknown scaffold: '{name}'. Available: {available}")
    return SCAFFOLD_REGISTRY[name].required_extras


def validate_scaffold(name: str) -> None:
    """Validate that a scaffold's requirements are satisfied.

    This should be called early (e.g., during worker initialization) to
    fail fast if required dependencies are missing.

    Args:
        name: Scaffold name to validate.

    Raises:
        ImportError: If the scaffold's required dependencies are not installed.
        ValueError: If scaffold name is unknown.
    """
    if name not in SCAFFOLD_REGISTRY:
        available = ", ".join(sorted(SCAFFOLD_REGISTRY.keys()))
        raise ValueError(f"Unknown scaffold: '{name}'. Available: {available}")

    # Maps scaffold name -> (import_module, display_name, extra_name)
    scaffold_checks = {
        "openai_agents": ("agents", "OpenAI Agents SDK", "agents"),
        "openhands": ("openhands", "OpenHands SDK", "openhands"),
    }

    if name in scaffold_checks:
        module, display_name, extra = scaffold_checks[name]
        try:
            import importlib

            importlib.import_module(module)
        except ImportError as e:
            raise ImportError(
                f"Scaffold '{name}' requires {display_name}. "
                f"Install with: uv pip install -e '.[{extra}]'"
            ) from e


# Import scaffolds to trigger registration
from .bfcl_multi_turn import BFCLMultiTurnScaffold  # noqa: E402
from .openai_agents import OpenAIAgentsScaffold  # noqa: E402
from .openhands import OpenHandsScaffold  # noqa: E402
from .vanillux import VanilluxScaffold  # noqa: E402

__all__ = [
    "Scaffold",
    "SCAFFOLD_REGISTRY",
    "BFCLMultiTurnScaffold",
    "OpenAIAgentsScaffold",
    "OpenHandsScaffold",
    "VanilluxScaffold",
    "get_scaffold",
    "get_scaffold_extras",
    "list_scaffolds",
    "register_scaffold",
    "validate_scaffold",
]
