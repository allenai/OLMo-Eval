"""Stand-ins for sandbox executors, shared by the manager tests.

A `TrackingExecutor` records what was asked of it and how much of it ran at
once, so a test can assert routing, capacity and staging behavior without a
container. `make_manager` wires executors into a `SandboxManager` the way
`start()` would, without starting anything.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping

from olmo_eval.common.execution import ExecutionResult
from olmo_eval.harness.sandbox import Capability, SandboxConfig, SandboxManager, SandboxMode
from olmo_eval.harness.sandbox.errors import SandboxTransportError


class TrackingExecutor:
    """Executor stub that counts operations in flight and records requests."""

    def __init__(
        self,
        name: str,
        *,
        capabilities: frozenset[str] = Capability.DEFAULT,
        max_concurrency: int = 2,
        fail_and_quarantine: bool = False,
        fail_transport: bool = False,
        output: str = "",
    ) -> None:
        self.name = name
        self.config = SandboxConfig(
            image="test",
            mode=SandboxMode.MODAL,
            capabilities=capabilities,
            max_concurrency=max_concurrency,
        )
        self.running = True
        self.fail_and_quarantine = fail_and_quarantine
        self.fail_transport = fail_transport
        self.output = output

        #: Operations of any kind this executor has completed.
        self.calls = 0
        self.active = 0
        #: Highest number of operations in flight at once. Every operation
        #: counts, so a lease that bounds writing as well as executing shows
        #: up here.
        self.peak = 0

        self.files: dict[str, str] = {}
        self.commands: list[str] = []
        #: Files present when each command ran, to catch a command that
        #: executes somewhere its files were never written.
        self.files_at_command: list[set[str]] = []
        #: Timeout passed to each write, to check the caller's bound reaches
        #: staging and not just the command.
        self.write_timeouts: list[float | None] = []

    @property
    def is_running(self) -> bool:
        return self.running

    async def _run(self) -> None:
        """Simulate one operation, honoring the configured failure mode."""
        self.calls += 1
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(0.01)
            if self.fail_and_quarantine:
                self.running = False
                raise ConnectionResetError("sandbox disconnected")
            if self.fail_transport:
                raise SandboxTransportError("request retries exhausted")
        finally:
            self.active -= 1

    async def execute_code(
        self,
        code: str,
        language: str = "python",
        timeout: float | None = None,
    ) -> ExecutionResult:
        del code, language, timeout
        await self._run()
        return ExecutionResult(success=True, output=self.output)

    async def execute_command(self, command: str, timeout: float | None = None) -> ExecutionResult:
        del timeout
        await self._run()
        self.commands.append(command)
        self.files_at_command.append(set(self.files))
        return ExecutionResult(success=True, output=self.output)

    async def write_files(self, files: Mapping[str, str], timeout: float | None = None) -> None:
        self.write_timeouts.append(timeout)
        await self._run()
        self.files.update(files)


def make_manager(*executors: TrackingExecutor) -> SandboxManager:
    """Return a manager over `executors` without starting any sandbox."""
    manager = SandboxManager([])
    manager._executors = list(executors)  # type: ignore[assignment]
    manager._active_operations = {id(executor): 0 for executor in executors}
    return manager
