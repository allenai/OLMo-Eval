"""Manager for multiple sandbox executors with capability-based routing."""

from __future__ import annotations

import asyncio
import atexit
import concurrent.futures
import contextlib
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TypeVar

from olmo_eval.common.execution.environment import ExecutionResult
from olmo_eval.common.logging import configure_worker_logging

from .config import Capability, SandboxConfig, SandboxMode
from .errors import SandboxInfrastructureError
from .executor import SandboxExecutor

_MAX_SANDBOX_START_WORKERS = 16
_ResultT = TypeVar("_ResultT")


def _sandbox_start_worker_count(executor_count: int) -> int:
    """Return the bounded number of concurrent sandbox startup workers."""
    return min(executor_count, _MAX_SANDBOX_START_WORKERS)


class _NoSandboxAvailableError(ValueError):
    """No healthy executor can satisfy a lease request."""


@dataclass
class ExecutorBinding:
    """Pins a caller to a specific executor for session state continuity."""

    id: str
    executor: SandboxExecutor
    capabilities: frozenset[str]
    _manager: SandboxManager
    _released: bool = field(default=False, repr=False)

    async def execute_in_session(
        self, command: str, timeout: float | None = None
    ) -> ExecutionResult:
        """Execute in the bound executor's bash session."""
        if self._released:
            raise RuntimeError("Binding has been released")
        return await self.executor.execute_in_session(command, timeout)

    async def interrupt_session(self) -> None:
        """Interrupt the command running in the bound executor's bash session."""
        if self._released:
            raise RuntimeError("Binding has been released")
        await self.executor.interrupt_session()

    async def release(self) -> None:
        if not self._released:
            self._released = True
            await self._manager._release_binding(self)

    async def __aenter__(self) -> ExecutorBinding:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.release()


@dataclass(frozen=True)
class CapabilityExecutionEnvironment:
    """Execution environment that routes each operation to a capability pool."""

    manager: SandboxManager
    capabilities: frozenset[str]

    @property
    def is_running(self) -> bool:
        return self.manager.has_executor(self.capabilities)

    async def execute(self, command: str, timeout: float | None = None) -> str:
        return await self.manager.execute(
            command,
            timeout,
            capabilities=self.capabilities,
            failover_on_quarantine=True,
        )

    async def execute_command(self, command: str, timeout: float | None = None) -> ExecutionResult:
        return await self.manager.execute_command(
            command,
            timeout,
            capabilities=self.capabilities,
            failover_on_quarantine=True,
        )

    async def execute_code(
        self,
        code: str,
        language: str = "python",
        timeout: float | None = None,
    ) -> ExecutionResult:
        return await self.manager.execute_code(
            code,
            language,
            timeout,
            capabilities=self.capabilities,
            failover_on_quarantine=True,
        )

    async def execute_with_files(
        self,
        command: str,
        files: Mapping[str, str],
        timeout: float | None = None,
    ) -> ExecutionResult:
        return await self.manager.execute_with_files(
            command,
            files,
            timeout,
            capabilities=self.capabilities,
            failover_on_quarantine=True,
        )


class SandboxManager:
    """Manages multiple sandbox executors with capability-based routing.

    Executors are selected using round-robin among those that support the
    required capabilities. For session-based execution requiring state
    continuity, use acquire_binding() or the binding() context manager.

    Usage:
        from olmo_eval.harness.sandbox import Capability

        configs = [SandboxConfig(...), SandboxConfig(...)]
        manager = SandboxManager(configs, owner="scorer")
        await manager.start()
        try:
            result = await manager.execute("echo hello", capabilities=Capability.BASH)
        finally:
            await manager.stop()
    """

    def __init__(self, configs: Sequence[SandboxConfig], owner: str = "default") -> None:
        """Initialize the sandbox manager.

        Args:
            configs: Sequence of sandbox configurations to manage.
            owner: Identifier for the owner of these sandboxes (e.g., "agent", "scorer").
                Used in log messages to distinguish sandbox instances.
        """
        self._configs = list(configs)
        self._owner = owner
        self._logger = configure_worker_logging("sb-manager")
        self._executors: list[SandboxExecutor] = []
        self._round_robin_indices: dict[frozenset[str], int] = {}
        self._execution_semaphores: dict[frozenset[str], asyncio.Semaphore] = {}
        self._active_operations: dict[int, int] = {}
        self._capacity_condition = asyncio.Condition()
        self._bindings: dict[str, ExecutorBinding] = {}
        self._bound_executors: set[int] = set()
        self._binding_counter: int = 0
        self._modal_app_name: str | None = None

        # Generate shared Modal app name if any config uses Modal
        if any(c.mode == SandboxMode.MODAL for c in self._configs):
            self._modal_app_name = f"swerex-{uuid.uuid4().hex[:12]}"
            self._logger.info(f"Using Modal app: {self._modal_app_name}")

    async def start(self) -> None:
        """Start all sandbox executors.

        Uses thread pool to avoid event loop blocking from swe-rex subprocess calls.
        Allows partial failures if min_instances is configured on the sandbox config.
        """
        # Track per-type instance indices for naming
        type_indices: dict[str, int] = {}
        # Track which config each executor belongs to: executor index -> config index
        executor_to_config: dict[int, int] = {}

        # Create all executors first
        executor_idx = 0
        for config_idx, config in enumerate(self._configs):
            # Derive type name from capabilities, replacing ':' to avoid
            # breaking podman volume mount paths (host:container separator)
            type_name = "+".join(sorted(config.capabilities)) or str(config_idx)
            safe_type_name = type_name.replace(":", "_")

            for _ in range(config.resolved_instances):
                idx = type_indices.get(type_name, 0)
                name = f"sb-{safe_type_name}-{self._owner}-{idx}"
                type_indices[type_name] = idx + 1

                executor = SandboxExecutor(config, name=name, modal_app_name=self._modal_app_name)
                self._executors.append(executor)
                executor_to_config[executor_idx] = config_idx
                executor_idx += 1

        # Start all executors in thread pool to avoid blocking event loop
        # swe-rex's DockerDeployment.start() has blocking subprocess calls
        start_time = time.time()
        start_workers = _sandbox_start_worker_count(len(self._executors))
        self._logger.info(
            f"Starting {len(self._executors)} sandbox executors "
            f"with up to {start_workers} concurrent starts..."
        )

        def start_in_thread(executor: SandboxExecutor) -> None:
            """Run executor.start() in a dedicated thread with its own event loop."""
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(executor.start())
            finally:
                loop.close()

        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=start_workers) as pool:
            futures = [loop.run_in_executor(pool, start_in_thread, e) for e in self._executors]
            results = await asyncio.gather(*futures, return_exceptions=True)

        # Track successes and failures per config
        num_configs = len(self._configs)
        config_successes: dict[int, list[SandboxExecutor]] = {i: [] for i in range(num_configs)}
        config_failures: dict[int, list[tuple[SandboxExecutor, BaseException]]] = {
            i: [] for i in range(num_configs)
        }

        for exec_idx, (executor, result) in enumerate(zip(self._executors, results, strict=True)):
            config_idx = executor_to_config[exec_idx]
            if isinstance(result, BaseException):
                config_failures[config_idx].append((executor, result))
                self._logger.warning(f"Executor {executor.name} failed to start: {result}")
            else:
                config_successes[config_idx].append(executor)

        startup_error: RuntimeError | None = None

        # Check minimum requirements per config
        for config_idx, config in enumerate(self._configs):
            if (
                config.min_instances is not None
                and config.min_instances > config.resolved_instances
            ):
                self._logger.warning(
                    f"Sandbox config {config_idx} ({config.image}) requested "
                    f"min_instances={config.min_instances} with only "
                    f"{config.resolved_instances} executor(s); clamping to "
                    f"{config.resolved_instances}"
                )
            min_required = config.resolved_min_instances
            started_count = len(config_successes[config_idx])
            failed_count = len(config_failures[config_idx])

            if started_count < min_required:
                startup_error = RuntimeError(
                    f"Sandbox config {config_idx} ({config.image}): "
                    f"only {started_count}/{min_required} required instances started "
                    f"({failed_count} failed)"
                )
                break

            if failed_count > 0:
                self._logger.warning(
                    f"Sandbox config {config_idx} ({config.image}): "
                    f"{started_count}/{config.resolved_instances} instances started "
                    f"({failed_count} failed, min_required={min_required})"
                )

        if startup_error is not None:
            await self.stop()
            raise startup_error

        failed_executors = [
            executor for failures in config_failures.values() for executor, _ in failures
        ]
        await self._stop_executors(failed_executors)

        # Keep only successfully started executors
        self._executors = [e for successes in config_successes.values() for e in successes]
        self._active_operations = {id(e): 0 for e in self._executors}

        # Build per-capability execution semaphores from running executors
        cap_counts: dict[frozenset[str], int] = {}
        cap_mc: dict[frozenset[str], int] = {}
        for e in self._executors:
            cap = e.config.capabilities
            cap_counts[cap] = cap_counts.get(cap, 0) + 1
            cap_mc[cap] = e.config.max_concurrency
        for cap, count in cap_counts.items():
            limit = cap_mc[cap] * count
            self._execution_semaphores[cap] = asyncio.Semaphore(limit)
            self._logger.info(
                f"Execution semaphore for {sorted(cap)}: "
                f"{limit} ({cap_mc[cap]} x {count} instances)"
            )

        elapsed = time.time() - start_time
        total_attempted = sum(c.resolved_instances for c in self._configs)
        self._logger.info(
            f"Started {len(self._executors)}/{total_attempted} sandbox executors in {elapsed:.1f}s"
        )

        atexit.register(self._atexit_cleanup)

    async def stop(self) -> None:
        """Stop all sandbox executors."""
        async with self._capacity_condition:
            for binding in self._bindings.values():
                binding._released = True
            self._bindings.clear()
            self._bound_executors.clear()
            self._capacity_condition.notify_all()

        await self._stop_executors(self._executors)
        self._executors.clear()
        self._round_robin_indices.clear()
        self._active_operations.clear()
        self._logger.info("All sandboxes stopped")

        if self._modal_app_name is not None:
            from .modal_deployment import stop_modal_app

            try:
                await stop_modal_app(self._modal_app_name)
            except Exception as e:
                self._logger.warning(f"Failed to stop Modal app {self._modal_app_name}: {e}")

        with contextlib.suppress(Exception):
            atexit.unregister(self._atexit_cleanup)

    async def _stop_executors(self, executors: Sequence[SandboxExecutor]) -> None:
        """Best-effort stop every executor without abandoning later cleanups."""
        results = await asyncio.gather(
            *(executor.stop() for executor in executors),
            return_exceptions=True,
        )
        for executor, result in zip(executors, results, strict=True):
            if isinstance(result, BaseException):
                self._logger.warning(f"Executor {executor.name} failed to stop: {result}")

    def _atexit_cleanup(self) -> None:
        """Synchronous cleanup for atexit. Runs stop() if executors are still active."""
        if not self._executors:
            return
        self._logger.info("Cleaning up sandboxes on exit")
        try:
            asyncio.run(self.stop())
        except Exception as e:
            self._logger.error(f"Sandbox cleanup failed: {e}")

    def get_executor(self, required_capabilities: frozenset[str]) -> SandboxExecutor:
        """Get an executor that supports the required capabilities.

        Uses round-robin selection among matching executors.

        Args:
            required_capabilities: Set of capabilities the executor must support.

        Returns:
            A SandboxExecutor that supports all required capabilities.

        Raises:
            ValueError: If no executor supports the required capabilities.
        """
        matching = [
            (i, e)
            for i, e in enumerate(self._executors)
            if (
                required_capabilities <= e.config.capabilities
                and i not in self._bound_executors
                and e.is_running
            )
        ]

        if not matching:
            available = [e.config.capabilities for e in self._executors if e.is_running]
            raise ValueError(
                f"No sandbox supports capabilities {required_capabilities}. Available: {available}"
            )

        # Round-robin selection
        key = required_capabilities
        idx = self._round_robin_indices.get(key, 0)
        selected_idx = idx % len(matching)
        self._round_robin_indices[key] = idx + 1

        return matching[selected_idx][1]

    def has_executor(self, required_capabilities: frozenset[str]) -> bool:
        """Return whether a healthy, unbound executor supports the capabilities."""
        return any(
            required_capabilities <= executor.config.capabilities
            and idx not in self._bound_executors
            and executor.is_running
            for idx, executor in enumerate(self._executors)
        )

    def for_capabilities(
        self, required_capabilities: frozenset[str]
    ) -> CapabilityExecutionEnvironment:
        """Return an environment that leases capacity for every operation."""
        return CapabilityExecutionEnvironment(self, required_capabilities)

    @asynccontextmanager
    async def lease_executor(
        self,
        required_capabilities: frozenset[str],
        *,
        excluded: frozenset[int] = frozenset(),
    ) -> AsyncIterator[SandboxExecutor]:
        """Lease one capacity slot on the least-loaded compatible executor."""
        async with self._capacity_condition:
            while True:
                compatible = [
                    (idx, executor)
                    for idx, executor in enumerate(self._executors)
                    if required_capabilities <= executor.config.capabilities
                    and idx not in self._bound_executors
                    and id(executor) not in excluded
                    and executor.is_running
                ]
                if not compatible:
                    available = [
                        executor.config.capabilities
                        for executor in self._executors
                        if executor.is_running
                    ]
                    raise _NoSandboxAvailableError(
                        f"No sandbox supports capabilities {required_capabilities}. "
                        f"Available: {available}"
                    )

                candidates = [
                    (idx, executor)
                    for idx, executor in compatible
                    if self._active_operations.get(id(executor), 0)
                    < executor.config.max_concurrency
                ]
                if candidates:
                    key = required_capabilities
                    start = self._round_robin_indices.get(key, 0) % len(candidates)
                    ordered = candidates[start:] + candidates[:start]
                    _, selected = min(
                        ordered,
                        key=lambda item: self._active_operations.get(id(item[1]), 0),
                    )
                    self._round_robin_indices[key] = start + 1
                    executor_id = id(selected)
                    self._active_operations[executor_id] = (
                        self._active_operations.get(executor_id, 0) + 1
                    )
                    break

                await self._capacity_condition.wait()

        try:
            yield selected
        finally:
            async with self._capacity_condition:
                executor_id = id(selected)
                active = self._active_operations.get(executor_id, 0)
                self._active_operations[executor_id] = max(0, active - 1)
                self._capacity_condition.notify_all()

    def get_execution_semaphore(
        self, required_capabilities: frozenset[str]
    ) -> asyncio.Semaphore | None:
        """Get the execution semaphore for the given capabilities.

        Returns a shared semaphore sized to max_concurrency * running_instances
        for the matching capability set. Returns None if no match.
        """
        for cap, sem in self._execution_semaphores.items():
            if required_capabilities <= cap:
                return sem
        return None

    async def execute(
        self,
        command: str,
        timeout: float | None = None,
        capabilities: frozenset[str] | None = None,
        *,
        failover_on_quarantine: bool = False,
    ) -> str:
        """Execute a command on a sandbox.

        Args:
            command: The command to execute.
            timeout: Optional timeout override in seconds.
            capabilities: Optional required capabilities. If None, uses default.

        Returns:
            The command output.
        """
        required = capabilities or Capability.DEFAULT
        return await self._execute_with_lease(
            required,
            lambda executor: executor.execute(command, timeout),
            failover_on_quarantine=failover_on_quarantine,
        )

    async def execute_command(
        self,
        command: str,
        timeout: float | None = None,
        capabilities: frozenset[str] | None = None,
        *,
        failover_on_quarantine: bool = False,
    ) -> ExecutionResult:
        """Execute a command and return structured result.

        Args:
            command: The command to execute.
            timeout: Optional timeout override in seconds.
            capabilities: Optional required capabilities. If None, uses default.

        Returns:
            ExecutionResult with success status, output, and exit code.
        """
        required = capabilities or Capability.DEFAULT
        return await self._execute_with_lease(
            required,
            lambda executor: executor.execute_command(command, timeout),
            failover_on_quarantine=failover_on_quarantine,
        )

    async def execute_code(
        self,
        code: str,
        language: str = "python",
        timeout: float | None = None,
        capabilities: frozenset[str] | None = None,
        *,
        failover_on_quarantine: bool = False,
    ) -> ExecutionResult:
        """Execute code in the specified language.

        Implements the ExecutionEnvironment protocol by delegating to the
        first available executor.

        Args:
            code: Source code to execute.
            language: Programming language (default: "python").
            timeout: Optional timeout in seconds.
            capabilities: Optional required capabilities. If None, uses default.

        Returns:
            ExecutionResult with success status and output.
        """
        required = capabilities or Capability.DEFAULT
        return await self._execute_with_lease(
            required,
            lambda executor: executor.execute_code(code, language, timeout),
            failover_on_quarantine=failover_on_quarantine,
        )

    async def execute_with_files(
        self,
        command: str,
        files: Mapping[str, str],
        timeout: float | None = None,
        capabilities: frozenset[str] | None = None,
        *,
        failover_on_quarantine: bool = False,
    ) -> ExecutionResult:
        """Stage files and run a command against them on one executor.

        Both steps hold the same lease, so the command sees the files that
        were written for it rather than the contents of another executor.

        Args:
            command: The command to execute once the files are in place.
            files: Mapping of absolute sandbox path to file content.
            timeout: Seconds allowed for each step, applied to every file
                write and to the command separately rather than to the
                operation as a whole. Defaults to the executor's command
                timeout.
            capabilities: Optional required capabilities. If None, uses default.

        Returns:
            ExecutionResult with success status, output, and exit code.
        """
        required = capabilities or Capability.DEFAULT

        async def stage_and_execute(executor: SandboxExecutor) -> ExecutionResult:
            await executor.write_files(files, timeout)
            return await executor.execute_command(command, timeout)

        return await self._execute_with_lease(
            required,
            stage_and_execute,
            failover_on_quarantine=failover_on_quarantine,
        )

    async def _execute_with_lease(
        self,
        required_capabilities: frozenset[str],
        operation: Callable[[SandboxExecutor], Awaitable[_ResultT]],
        *,
        failover_on_quarantine: bool,
    ) -> _ResultT:
        """Execute with strict per-executor capacity and optional one-hop failover."""
        excluded: frozenset[int] = frozenset()
        last_infrastructure_error: Exception | None = None
        while True:
            try:
                async with self.lease_executor(
                    required_capabilities,
                    excluded=excluded,
                ) as executor:
                    try:
                        return await operation(executor)
                    except Exception as exc:
                        if not failover_on_quarantine or executor.is_running:
                            raise
                        last_infrastructure_error = exc
                        excluded = excluded | {id(executor)}
                        self._logger.warning(
                            f"Retrying operation on a different sandbox after "
                            f"an infrastructure failure from {executor.name}: {exc}"
                        )
            except _NoSandboxAvailableError:
                if last_infrastructure_error is None:
                    raise
                raise SandboxInfrastructureError(
                    f"No healthy sandbox remained for {sorted(required_capabilities)} "
                    f"after infrastructure failures: {last_infrastructure_error}"
                ) from last_infrastructure_error

    async def acquire_binding(
        self,
        capabilities: frozenset[str] | None = None,
    ) -> ExecutorBinding:
        """Acquire exclusive binding to an executor for session execution."""
        required = capabilities or frozenset()
        async with self._capacity_condition:
            available = [
                (i, e)
                for i, e in enumerate(self._executors)
                if (
                    required <= e.config.capabilities
                    and i not in self._bound_executors
                    and e.is_running
                    and self._active_operations.get(id(e), 0) == 0
                )
            ]
            if not available:
                raise ValueError(f"No available executor for {required}")

            executor_idx, executor = available[0]
            self._binding_counter += 1
            binding = ExecutorBinding(
                id=f"binding-{self._binding_counter}",
                executor=executor,
                capabilities=required,
                _manager=self,
            )
            self._bindings[binding.id] = binding
            self._bound_executors.add(executor_idx)
            return binding

    async def _release_binding(self, binding: ExecutorBinding) -> None:
        async with self._capacity_condition:
            if binding.id not in self._bindings:
                return
            for idx, e in enumerate(self._executors):
                if e is binding.executor:
                    self._bound_executors.discard(idx)
                    break
            del self._bindings[binding.id]
            self._capacity_condition.notify_all()

    @asynccontextmanager
    async def binding(
        self,
        capabilities: frozenset[str] | None = None,
    ) -> AsyncIterator[ExecutorBinding]:
        """Context manager for executor binding."""
        b = await self.acquire_binding(capabilities)
        try:
            yield b
        finally:
            await b.release()

    @property
    def is_running(self) -> bool:
        """Check if any executors are running."""
        return any(e.is_running for e in self._executors)

    @property
    def executor_count(self) -> int:
        """Number of active executors."""
        return len(self._executors)
