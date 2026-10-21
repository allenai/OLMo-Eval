"""Tests for staging files into a sandbox before running a command."""

from __future__ import annotations

import asyncio
import time

import pytest

from olmo_eval.harness.sandbox import SandboxConfig, SandboxExecutor, SandboxMode
from olmo_eval.harness.sandbox.errors import SandboxTransportError
from tests.core.harness.sandbox_stubs import TrackingExecutor, make_manager

# ---------------------------------------------------------------------------
# Staging through the manager
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_files_are_written_before_the_command_runs() -> None:
    executor = TrackingExecutor("sb-0")
    manager = make_manager(executor)

    result = await manager.execute_with_files(
        "python3 grade.py",
        {"/tmp/work/grade.py": "print(1)", "/tmp/work/problem.json": "{}"},
    )

    assert result.success
    assert executor.commands == ["python3 grade.py"]
    assert executor.files_at_command == [{"/tmp/work/grade.py", "/tmp/work/problem.json"}]


@pytest.mark.anyio
async def test_command_runs_on_the_executor_that_received_the_files() -> None:
    first = TrackingExecutor("sb-0")
    second = TrackingExecutor("sb-1")
    manager = make_manager(first, second)

    # Enough concurrent calls to spread across both executors.
    await asyncio.gather(
        *(
            manager.execute_with_files(
                f"python3 grade.py {index}",
                {f"/tmp/work-{index}/solution.py": f"# {index}"},
            )
            for index in range(6)
        )
    )

    assert first.commands and second.commands, "expected work on both executors"
    for executor in (first, second):
        for command, staged in zip(executor.commands, executor.files_at_command, strict=True):
            index = command.rsplit(" ", 1)[-1]
            assert f"/tmp/work-{index}/solution.py" in staged


@pytest.mark.anyio
async def test_staging_respects_executor_capacity() -> None:
    executor = TrackingExecutor("sb-0", max_concurrency=1)
    manager = make_manager(executor)

    await asyncio.gather(
        *(manager.execute_with_files(f"cmd {index}", {f"/tmp/{index}": "x"}) for index in range(4))
    )

    assert len(executor.commands) == 4
    # One slot means writing and running never overlap, so a concurrent
    # caller cannot replace a file between another's write and its command.
    assert executor.peak == 1


@pytest.mark.anyio
async def test_staging_uses_every_slot_an_executor_allows() -> None:
    executor = TrackingExecutor("sb-0", max_concurrency=3)
    manager = make_manager(executor)

    await asyncio.gather(
        *(manager.execute_with_files(f"cmd {index}", {f"/tmp/{index}": "x"}) for index in range(6))
    )

    assert len(executor.commands) == 6
    assert 1 < executor.peak <= 3


@pytest.mark.anyio
async def test_timeout_reaches_the_file_writes() -> None:
    # The bound has to cover staging, not just the command that follows it.
    executor = TrackingExecutor("sb-0")
    manager = make_manager(executor)

    await manager.execute_with_files("cmd", {"/tmp/a": "x"}, timeout=900)

    assert executor.write_timeouts == [900]


# ---------------------------------------------------------------------------
# The write itself is bounded
# ---------------------------------------------------------------------------


class _HangingRuntime:
    """Runtime whose file writes never complete on their own."""

    def __init__(self) -> None:
        self.calls = 0

    async def write_file(self, request: object) -> None:
        del request
        self.calls += 1
        await asyncio.sleep(3600)


class _HealthyDeployment:
    async def is_alive(self, *, timeout: float) -> bool:
        del timeout
        return True


def _executor_with_hanging_writes(**config: object) -> tuple[SandboxExecutor, _HangingRuntime]:
    executor = SandboxExecutor(
        SandboxConfig(image="test", mode=SandboxMode.MODAL, **config),  # type: ignore[arg-type]
        name="test-sandbox",
    )
    runtime = _HangingRuntime()
    executor._runtime = runtime
    executor._deployment = _HealthyDeployment()
    return executor, runtime


@pytest.fixture(autouse=True)
def _no_retry_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retry immediately so a timed-out write is re-attempted without waiting."""
    monkeypatch.setattr(
        "olmo_eval.harness.sandbox.executor._exponential_backoff",
        lambda initial_delay, retry: 0.0,
    )


@pytest.mark.anyio
async def test_a_stalled_write_is_given_up_on_within_the_timeout() -> None:
    executor, runtime = _executor_with_hanging_writes()
    started = time.monotonic()

    with pytest.raises(SandboxTransportError, match="file write to /tmp/fixture.json"):
        await executor.write_files({"/tmp/fixture.json": "x" * 1024}, timeout=0.05)

    # Timed out on every attempt, then stopped, rather than waiting on the
    # HTTP client's own default.
    assert runtime.calls == 4
    assert time.monotonic() - started < 2.0


@pytest.mark.anyio
async def test_writes_default_to_the_command_timeout() -> None:
    # A caller that gives no timeout still gets a bound, the same one commands use.
    executor, runtime = _executor_with_hanging_writes(command_timeout=0.05)
    started = time.monotonic()

    with pytest.raises(SandboxTransportError):
        await executor.write_files({"/tmp/fixture.json": "x"})

    assert runtime.calls == 4
    assert time.monotonic() - started < 2.0
