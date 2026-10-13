"""Infrastructure configuration for different deployment environments.

All settings are read from environment variables with sensible defaults
that work for local development. Deployment-specific launchers set the
appropriate env vars for their environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class InfrastructureConfig:
    """Centralized infrastructure configuration.

    All settings come from environment variables with generic defaults.
    Deployment-specific code sets env vars before evaluation runs.
    """

    # Container settings
    container_runtime: str
    swerex_registry: str

    # Network settings
    pasta_host_ip: str

    # Sandbox images
    sandbox_base_image: str

    # Storage paths (None = use tool defaults)
    hf_cache_dir: str | None
    uv_cache_dir: str | None
    inspect_cache_dir: str | None
    result_dir: str

    @classmethod
    def from_environment(cls) -> InfrastructureConfig:
        """Create config from environment variables."""
        return cls(
            container_runtime=os.environ.get("OLMO_CONTAINER_RUNTIME", "docker"),
            swerex_registry=os.environ.get("SWEREX_REGISTRY", ""),
            pasta_host_ip=os.environ.get("OLMO_PASTA_HOST_IP", "169.254.1.2"),
            sandbox_base_image=os.environ.get(
                "OLMO_SANDBOX_IMAGE", "ghcr.io/astral-sh/uv:python3.12-bookworm-slim"
            ),
            hf_cache_dir=os.environ.get("HF_HOME"),
            uv_cache_dir=os.environ.get("UV_CACHE_DIR"),
            inspect_cache_dir=os.environ.get("INSPECT_CACHE_DIR"),
            result_dir=os.environ.get("OLMO_RESULT_DIR", "/tmp/results"),
        )


# Cached config instance
_config: InfrastructureConfig | None = None


def get_infra_config() -> InfrastructureConfig:
    """Get the current infrastructure config (cached, reads env vars once)."""
    global _config
    if _config is None:
        _config = InfrastructureConfig.from_environment()
    return _config


def reset_infra_config() -> None:
    """Reset cached config (for testing or after env var changes)."""
    global _config
    _config = None
