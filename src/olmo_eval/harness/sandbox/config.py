"""Configuration for sandboxed tool execution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

from olmo_eval.common.repr import hide_unset

ContainerRuntime = Literal["docker", "podman"]
ImagePullPolicy = Literal["never", "missing", "always"]
RegistryProvider = Literal["gcp", "ecr", "docker"]


@dataclass(frozen=True)
class RegistryAuth:
    """Authentication for private container registries.

    For Modal deployments, credentials must be pre-stored as Modal Secrets.
    For Docker deployments, uses local credential helpers (gcloud, aws ecr).

    Attributes:
        provider: Registry provider ("gcp", "ecr", "docker").
        secret_name: Modal secret name containing credentials (Modal only).
            - GCP: Must contain SERVICE_ACCOUNT_JSON
            - ECR: Must contain AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
            - Docker: Must contain DOCKER_USERNAME, DOCKER_PASSWORD
    """

    provider: RegistryProvider
    secret_name: str | None = None  # Required for Modal, ignored for Docker


class SandboxMode(StrEnum):
    """Sandbox deployment modes."""

    LOCAL = "local"
    DOCKER = "docker"
    MODAL = "modal"


class Capability:
    """Standard sandbox capability identifiers.

    Example:
        @registered_tool(sandbox=Capability.BASH)
        async def execute_bash(command: str) -> str:
            ...
    """

    BASH: frozenset[str] = frozenset({"bash"})
    PYTHON: frozenset[str] = frozenset({"python"})
    DEFAULT: frozenset[str] = BASH


DEFAULT_MAX_CONCURRENCY = 4
DEFAULT_DEPLOYMENT_TIMEOUT = 6 * 60 * 60


@hide_unset()
@dataclass(frozen=True)
class SandboxConfig:
    """Configuration for sandboxed tool execution via SWE-ReX.

    Attributes:
        image: Container image for the sandbox environment.
        mode: How to run the sandbox.
        capabilities: Capabilities this sandbox provides (e.g., {"bash"}).
        instances: Explicit number of executor instances to create from this config.
            When None, the caller can auto-allocate capacity; consumers that do
            not perform allocation fall back to one executor.
            Multiple instances enable higher throughput via round-robin.
        max_concurrency: Maximum concurrent operations per executor instance.
            Total concurrent sandbox operations for a capability set is
            max_concurrency * (number of running instances).
        min_instances: Minimum instances that must start successfully.
            None (default) means all instances are required. Set to a lower
            value to allow partial failures during startup.
        startup_timeout: Timeout for container startup in seconds.
        command_timeout: Default timeout for command execution in seconds.
        remove_container: Whether to remove container after use.
        working_dir: Working directory inside the container.
        environment: Environment variables as tuple of (name, value) pairs.
        volumes: Volume mounts as tuple of (host_path, container_path) pairs.
        modal_sandbox_kwargs: Additional kwargs for Modal sandbox configuration.
        runtime_timeout: Timeout for individual Modal runtime requests in seconds.
        deployment_timeout: Maximum lifetime of a Modal sandbox in seconds.
        required_secrets: Environment variable names that must be set.
        enable_diagnostics: Whether to run background diagnostics monitor.
        inject_swerex: Whether to build a derived image with swe-rex pre-installed.
        pristine_image: When building a derived image, leave the base image's
            packages, PATH and environment untouched so commands run in the
            task's own environment rather than the swe-rex virtualenv.
        dockerfile_extra: Additional Dockerfile commands to inject when building derived images.
        image_pull: Image pull policy for swerex ("never", "missing", "always").
            Use "never" when inject_swerex=True to skip redundant image checks.
        registry_auth: Authentication for private container registries (Modal only).
    """

    image: str
    mode: SandboxMode
    capabilities: frozenset[str] = Capability.DEFAULT
    instances: int | None = None
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY
    min_instances: int | None = None
    container_runtime: ContainerRuntime = "podman"
    startup_timeout: float = 60.0
    command_timeout: float = 30.0
    remove_container: bool = True
    working_dir: str = "/workspace"
    environment: tuple[tuple[str, str], ...] = ()
    volumes: tuple[tuple[str, str], ...] = ()
    modal_sandbox_kwargs: dict[str, Any] | None = None
    runtime_timeout: float = 3600.0
    deployment_timeout: float = DEFAULT_DEPLOYMENT_TIMEOUT
    required_secrets: tuple[str, ...] = ()
    docker_args: tuple[str, ...] = ()
    log_dir: str | None = None
    exec_shell: tuple[str, ...] | None = None
    enable_diagnostics: bool = True
    inject_swerex: bool = False
    pristine_image: bool = False
    dockerfile_extra: tuple[str, ...] = ()
    image_pull: ImagePullPolicy | None = None
    registry_auth: RegistryAuth | None = None

    @property
    def is_local(self) -> bool:
        """True if sandbox runs locally (docker/local), False if remote (modal)."""
        return self.mode in (SandboxMode.LOCAL, SandboxMode.DOCKER)

    @property
    def resolved_instances(self) -> int:
        """Executor count with the default auto-allocation fallback applied."""
        return self.instances if self.instances is not None else 1

    @property
    def resolved_min_instances(self) -> int:
        """Minimum required instances, clamped to the resolved executor count."""
        if self.min_instances is None:
            return self.resolved_instances
        return min(self.min_instances, self.resolved_instances)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        from dataclasses import asdict

        result = asdict(self)
        result["mode"] = self.mode.value
        result["capabilities"] = sorted(self.capabilities)
        if self.registry_auth is not None:
            result["registry_auth"] = asdict(self.registry_auth)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SandboxConfig:
        """Create from dictionary."""
        if "image" not in data:
            raise ValueError("SandboxConfig requires 'image' to be specified")
        if "mode" not in data:
            raise ValueError("SandboxConfig requires 'mode' to be specified")
        capabilities = data.get("capabilities")
        registry_auth_data = data.get("registry_auth")
        registry_auth = RegistryAuth(**registry_auth_data) if registry_auth_data else None
        return cls(
            image=data["image"],
            mode=SandboxMode(data["mode"]),
            capabilities=frozenset(capabilities) if capabilities else Capability.DEFAULT,
            instances=data.get("instances"),
            max_concurrency=data.get("max_concurrency", DEFAULT_MAX_CONCURRENCY),
            min_instances=data.get("min_instances"),
            container_runtime=data.get("container_runtime", "podman"),
            startup_timeout=data.get("startup_timeout", 60.0),
            command_timeout=data.get("command_timeout", 30.0),
            remove_container=data.get("remove_container", True),
            working_dir=data.get("working_dir", "/workspace"),
            environment=tuple(tuple(e) for e in data.get("environment", [])),
            volumes=tuple(tuple(v) for v in data.get("volumes", [])),
            modal_sandbox_kwargs=data.get("modal_sandbox_kwargs"),
            runtime_timeout=data.get("runtime_timeout", 3600.0),
            deployment_timeout=data.get("deployment_timeout", DEFAULT_DEPLOYMENT_TIMEOUT),
            required_secrets=tuple(data.get("required_secrets", [])),
            docker_args=tuple(data.get("docker_args", [])),
            log_dir=data.get("log_dir"),
            exec_shell=tuple(data["exec_shell"]) if data.get("exec_shell") else None,
            enable_diagnostics=data.get("enable_diagnostics", True),
            inject_swerex=data.get("inject_swerex", False),
            pristine_image=data.get("pristine_image", False),
            dockerfile_extra=tuple(data.get("dockerfile_extra", [])),
            image_pull=data.get("image_pull"),
            registry_auth=registry_auth,
        )
