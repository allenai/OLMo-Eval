"""Base class for external evaluations."""

from __future__ import annotations

import json
import logging
import os
import shlex
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from olmo_eval.evals.external.result import ExternalEvalResult

if TYPE_CHECKING:
    from olmo_eval.inference.base import InferenceProvider
    from olmo_eval.inference.providers.config import ProviderConfig

logger = logging.getLogger(__name__)


class ExternalEval(ABC):
    """Abstract base class for external evaluations."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this evaluation."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Short description of what this evaluation measures."""
        ...

    @property
    @abstractmethod
    def timeout_seconds(self) -> float:
        """Maximum execution time in seconds."""
        ...

    @property
    def run_command(self) -> str:
        """Command template for display. Override to show the run command structure."""
        return ""

    @property
    def required_secrets(self) -> tuple[str, ...]:
        """Environment variable names that must be forwarded to the sandbox."""
        return ()

    @property
    def extras(self) -> tuple[str, ...]:
        """Package extras that must be installed for this evaluation to run."""
        return ()

    @property
    def arguments(self) -> dict[str, tuple[str, Any | None]]:
        """Arguments that can be passed to this evaluation.

        Returns a dict of arg_name -> (description, default_value).
        Use None for default_value if the argument is optional with no default.
        """
        return {}

    @property
    def scaffold(self) -> str | None:
        """Scaffold name used by this evaluation, if any."""
        return None

    @abstractmethod
    async def execute(
        self,
        provider: InferenceProvider,
        args: dict[str, Any],
        output_dir: str | None = None,
        container_runtime: str = "podman",
    ) -> ExternalEvalResult:
        """Execute the external evaluation.

        Args:
            provider: Inference provider for LLM calls.
            args: Evaluation-specific arguments (e.g., domain, num_trials).
            output_dir: Optional directory to write results.
            container_runtime: Container runtime to use (docker or podman).

        Returns:
            Result of the evaluation.
        """
        ...

    async def execute_with_provider(
        self,
        provider: InferenceProvider | None = None,
        provider_config: ProviderConfig | None = None,
        args: dict[str, Any] | None = None,
        output_dir: str | None = None,
        container_runtime: str = "podman",
    ) -> ExternalEvalResult:
        """Execute the evaluation using a provider or provider configuration.

        Args:
            provider: Inference provider instance (preferred).
            provider_config: Provider configuration to create a provider from.
            args: Evaluation-specific arguments.
            output_dir: Optional directory to write results.
            container_runtime: Container runtime to use.

        Returns:
            Result of the evaluation.
        """
        if provider is None:
            if provider_config is None:
                raise ValueError("Either provider or provider_config must be provided")
            provider = provider_config.create_provider()

        return await self.execute(
            provider=provider,
            args=args or {},
            output_dir=output_dir,
            container_runtime=container_runtime,
        )

    def _build_env_vars(self, secrets: tuple[str, ...] | None = None) -> dict[str, str]:
        """Build environment variables for the sandbox, validating required secrets."""
        secrets = secrets or self.required_secrets
        env_vars: dict[str, str] = {}
        missing = []
        for secret in secrets:
            value = os.environ.get(secret)
            if value:
                env_vars[secret] = value
            else:
                missing.append(secret)

        if missing:
            raise ValueError(f"Missing required secrets: {', '.join(missing)}")

        return env_vars

    def _is_local_provider(self, provider: Any, provider_url: str) -> bool:
        """Check if the provider is a locally-deployed server.

        Returns True for vLLM servers or other local providers.
        Returns False for external API providers (OpenAI, Anthropic, etc.).

        Args:
            provider: The inference provider instance.
            provider_url: The provider's base URL.

        Returns:
            True if this is a local provider requiring health checks.
        """
        from urllib.parse import urlparse

        # Check if provider is managing its own server process
        # (VLLMServerProvider sets _server when starting a local server)
        server = getattr(provider, "_server", None)
        if server is not None:
            return True

        # Check if URL points to a local address
        parsed = urlparse(provider_url)
        hostname = parsed.hostname or ""

        # Local hostnames
        if hostname in ("localhost", "127.0.0.1", "::1"):
            return True

        # Private IP ranges (common in container networking)
        return hostname.startswith(("10.", "172.", "192.168."))

    def _get_provider_url_for_sandbox(self, provider_url: str) -> str:
        """Get the provider URL that's accessible from within the sandbox.

        Rewrites localhost URLs to use the pasta host IP.

        Args:
            provider_url: Original provider URL.

        Returns:
            URL accessible from within the sandbox.
        """
        from urllib.parse import urlparse, urlunparse

        from olmo_eval.common.config import get_infra_config

        parsed = urlparse(provider_url)

        # Only rewrite localhost URLs
        if parsed.hostname not in ("localhost", "127.0.0.1"):
            return provider_url

        # Reconstruct URL with pasta host IP
        pasta_host_ip = get_infra_config().pasta_host_ip
        new_netloc = pasta_host_ip
        if parsed.port:
            new_netloc = f"{pasta_host_ip}:{parsed.port}"

        new_url = urlunparse(
            (
                parsed.scheme,
                new_netloc,
                parsed.path,
                parsed.params,
                parsed.query,
                parsed.fragment,
            )
        )

        logger.info(f"[{self.name}] Rewrote provider URL: {provider_url} -> {new_url}")
        return new_url

    async def _check_provider_health(
        self,
        executor: Any,
        provider_url: str,
        max_attempts: int = 5,
        retry_delay: float = 2.0,
    ) -> bool:
        """Check if the provider is reachable from within the sandbox.

        This should only be called for locally-deployed providers (e.g., vLLM server).
        External API providers (OpenAI, Anthropic, etc.) don't expose /health endpoints.

        Args:
            executor: Sandbox executor instance.
            provider_url: URL of the provider to check.
            max_attempts: Maximum number of connection attempts.
            retry_delay: Seconds to wait between attempts.

        Returns:
            True if provider is reachable, False otherwise.
        """
        import asyncio

        # Build health check URL (try /health endpoint first, fall back to base)
        health_url = provider_url.rstrip("/")
        if "/v1" in health_url:
            health_url = health_url.replace("/v1", "/health")
        else:
            health_url = health_url + "/health"

        for attempt in range(1, max_attempts + 1):
            logger.info(
                f"[{self.name}] Checking provider health at {health_url} "
                f"(attempt {attempt}/{max_attempts})"
            )

            # Use curl with verbose output to diagnose connection issues
            # -s: silent, -S: show errors, -v: verbose (to stderr)
            # --max-time: total timeout, --connect-timeout: connection phase timeout
            check_cmd = (
                f"curl -sS --max-time 5 --connect-timeout 3 "
                f"-o /dev/null -w 'HTTP_CODE:%{{http_code}}' {shlex.quote(health_url)} 2>&1"
            )
            result = await executor.execute_command(check_cmd, timeout=10.0)
            output = result.output.strip()

            # Extract HTTP code from output
            http_code = "000"
            if "HTTP_CODE:" in output:
                http_code = output.split("HTTP_CODE:")[-1].strip()

            if http_code == "200":
                logger.info(f"[{self.name}] Provider is reachable at {provider_url}")
                return True

            # Log full curl output for debugging (includes error messages)
            logger.warning(
                f"[{self.name}] Provider not reachable (attempt {attempt}/{max_attempts}): "
                f"http_code={http_code}, curl_output={output}"
            )

            if attempt < max_attempts:
                await asyncio.sleep(retry_delay)

        logger.error(f"[{self.name}] Provider unreachable after {max_attempts} attempts")
        return False

    def _error_result(
        self, error: str, start_time: float, raw_output: str = ""
    ) -> ExternalEvalResult:
        """Create an error result."""
        return ExternalEvalResult(
            name=self.name,
            success=False,
            error=error,
            raw_output=raw_output,
            duration_seconds=time.time() - start_time,
        )

    def _save_results(self, result: ExternalEvalResult, output_dir: str) -> None:
        """Save results to the output directory."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        results_file = output_path / f"{self.name}_results.json"

        with open(results_file, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        logger.info(f"[{self.name}] Results saved to {results_file}")


class SandboxedExternalEval(ExternalEval):
    """External eval that runs in a single sandbox container.

    Subclass this for evals that use one container for the entire evaluation.
    """

    @property
    def sandbox_image(self) -> str | None:
        """Docker image to use for the sandbox container.

        Return None if the image should be built dynamically via build_sandbox_image().
        """
        return None

    def build_sandbox_image(self, container_runtime: str) -> str:
        """Build or fetch the sandbox image dynamically.

        Override this method to implement custom image building logic.
        Called when sandbox_image returns None.

        Args:
            container_runtime: Container runtime to use (docker or podman).

        Returns:
            The image name to use.

        Raises:
            NotImplementedError: If sandbox_image is None and this method is not overridden.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must set sandbox_image or override build_sandbox_image()"
        )

    def get_sandbox_image(self, container_runtime: str) -> str:
        """Get the sandbox image, building if necessary.

        Args:
            container_runtime: Container runtime to use (docker or podman).

        Returns:
            The image name to use.
        """
        if self.sandbox_image is not None:
            return self.sandbox_image
        return self.build_sandbox_image(container_runtime)

    @property
    @abstractmethod
    def working_dir(self) -> str:
        """Working directory inside the sandbox container."""
        ...

    @property
    @abstractmethod
    def setup_command(self) -> tuple[str, ...]:
        """Commands to run during setup."""
        ...

    @property
    def results_dir(self) -> str:
        """Directory where evaluation results are saved."""
        return f"{self.working_dir}/results"

    def _create_sandbox_config(
        self,
        container_runtime: str,
        output_dir: str | None = None,
    ) -> Any:
        """Create sandbox configuration for this evaluation."""
        from olmo_eval.evals.external.network import get_docker_network_args
        from olmo_eval.harness.sandbox.config import (
            ContainerRuntime,
            SandboxConfig,
            SandboxMode,
        )

        # Set log_dir for sandbox container logs
        log_dir = None
        if output_dir:
            log_dir = os.path.join(output_dir, "logs")

        runtime = cast(ContainerRuntime, container_runtime)
        image = self.get_sandbox_image(container_runtime)
        return SandboxConfig(
            image=image,
            mode=SandboxMode.DOCKER,
            container_runtime=runtime,
            command_timeout=self.timeout_seconds,
            working_dir=self.working_dir,
            environment=tuple(self._build_env_vars().items()),
            docker_args=tuple(get_docker_network_args(runtime=container_runtime)),
            log_dir=log_dir,
        )

    async def _run_setup(
        self, executor: Any, all_output: list[str], start_time: float
    ) -> ExternalEvalResult | None:
        """Run setup commands. Returns error result if any fail, None on success."""
        for cmd in self.setup_command:
            logger.info(f"[{self.name}] Setup: {cmd}")
            result = await executor.execute_command(
                cmd, timeout=self.timeout_seconds, stream=True, log_prefix=self.name
            )
            all_output.append(f"$ {cmd}\n{result.output}")
            logger.info(f"[{self.name}] Exit code: {result.exit_code}")

            if not result.success:
                return ExternalEvalResult(
                    name=self.name,
                    success=False,
                    error=f"Setup failed: {cmd}",
                    raw_output="\n".join(all_output),
                    duration_seconds=time.time() - start_time,
                )
        return None
