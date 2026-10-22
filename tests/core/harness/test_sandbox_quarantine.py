"""Tests for retrying and quarantining unresponsive sandbox executors."""

from __future__ import annotations

import aiohttp
import pytest

from olmo_eval.harness.sandbox.config import Capability, SandboxConfig, SandboxMode
from olmo_eval.harness.sandbox.errors import SandboxTransportError
from olmo_eval.harness.sandbox.executor import SandboxExecutor
from olmo_eval.harness.sandbox.manager import SandboxManager


class _Runtime:
    def __init__(
        self,
        failures: int,
        transport_error: type[Exception] = aiohttp.ConnectionTimeoutError,
    ) -> None:
        self.failures = failures
        self.transport_error = transport_error
        self.calls = 0

    async def write_file(self, request: object) -> object:
        return await self.execute(request)

    async def execute(self, command: object) -> object:
        self.calls += 1
        if self.calls <= self.failures:
            raise self.transport_error("transport failure")
        return type("Response", (), {"stdout": "ok", "stderr": "", "exit_code": 0})()


class _Deployment:
    def __init__(self, health: list[bool]) -> None:
        self.health = health
        self.calls = 0

    async def is_alive(self, *, timeout: float) -> bool:
        result = self.health[min(self.calls, len(self.health) - 1)]
        self.calls += 1
        return result


class _RemoteExecutionConnectionError(ConnectionResetError):
    """Connection-shaped error returned by command execution via SWE-ReX."""

    extra_info: dict[str, object] = {}


def _executor(
    *,
    runtime_failures: int,
    health: list[bool],
    transport_error: type[Exception] = aiohttp.ConnectionTimeoutError,
) -> SandboxExecutor:
    executor = SandboxExecutor(
        SandboxConfig(image="test", mode=SandboxMode.MODAL),
        name="test-sandbox",
    )
    executor._runtime = _Runtime(runtime_failures, transport_error)
    executor._deployment = _Deployment(health)
    return executor


@pytest.fixture(autouse=True)
def no_retry_delays(monkeypatch: pytest.MonkeyPatch) -> None:
    async def no_sleep(_delay: float) -> None:
        pass

    monkeypatch.setattr("olmo_eval.harness.sandbox.executor.asyncio.sleep", no_sleep)
    monkeypatch.setattr("olmo_eval.harness.sandbox.executor.random.uniform", lambda _low, _high: 0)


@pytest.mark.anyio
@pytest.mark.parametrize(
    "transport_error",
    [aiohttp.ConnectionTimeoutError, BrokenPipeError, ConnectionResetError],
)
async def test_single_transport_failure_recovers_without_quarantine(
    transport_error: type[Exception],
) -> None:
    executor = _executor(
        runtime_failures=1,
        health=[True],
        transport_error=transport_error,
    )

    await executor.write_files({"/tmp/value": "value"})

    assert executor._runtime.calls == 2
    assert executor._deployment.calls == 0
    assert executor.is_running is True


@pytest.mark.anyio
async def test_healthy_probe_prevents_quarantine_after_retries() -> None:
    executor = _executor(runtime_failures=4, health=[True])

    with pytest.raises(SandboxTransportError):
        await executor.write_files({"/tmp/value": "value"})

    assert executor._runtime.calls == 4
    assert executor._deployment.calls == 1
    assert executor.is_running is True


@pytest.mark.anyio
async def test_repeated_transport_and_health_failures_quarantine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delays: list[float] = []

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr("olmo_eval.harness.sandbox.executor.asyncio.sleep", record_sleep)
    executor = _executor(runtime_failures=4, health=[False, False, False, False])

    with pytest.raises(SandboxTransportError):
        await executor.write_files({"/tmp/value": "value"})

    assert executor._runtime.calls == 4
    assert executor._deployment.calls == 4
    assert executor.is_running is False
    assert delays == [0.25, 0.5, 1.0, 0.5, 1.0, 2.0]


@pytest.mark.anyio
async def test_transport_retry_uses_backoff_with_jitter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delays: list[float] = []
    jitter_ranges: list[tuple[float, float]] = []

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    def midpoint_jitter(low: float, high: float) -> float:
        jitter_ranges.append((low, high))
        return high / 2

    monkeypatch.setattr("olmo_eval.harness.sandbox.executor.asyncio.sleep", record_sleep)
    monkeypatch.setattr(
        "olmo_eval.harness.sandbox.executor.random.uniform",
        midpoint_jitter,
    )
    executor = _executor(runtime_failures=1, health=[True])

    await executor.write_files({"/tmp/value": "value"})

    assert executor._runtime.calls == 2
    assert delays == [0.375]
    assert jitter_ranges == [(0.0, 0.25)]


@pytest.mark.anyio
async def test_remote_execution_error_is_not_retried() -> None:
    executor = _executor(
        runtime_failures=1,
        health=[True],
        transport_error=_RemoteExecutionConnectionError,
    )

    with pytest.raises(_RemoteExecutionConnectionError):
        await executor.execute_command("true")

    assert executor._runtime.calls == 1
    assert executor._deployment.calls == 0


def test_manager_skips_only_confirmed_unresponsive_executor() -> None:
    manager = SandboxManager([])
    broken = _executor(runtime_failures=0, health=[True])
    healthy = _executor(runtime_failures=0, health=[True])
    broken._quarantined_reason = "confirmed unhealthy"
    manager._executors = [broken, healthy]

    assert manager.is_running is True
    assert manager.get_executor(Capability.DEFAULT) is healthy


def test_modal_deployment_receives_configured_lifetime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _ModalDeployment:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(
        "olmo_eval.harness.sandbox.modal_deployment.ManagedModalDeployment",
        _ModalDeployment,
    )
    executor = SandboxExecutor(
        SandboxConfig(
            image="python:3.12",
            mode=SandboxMode.MODAL,
            runtime_timeout=120.0,
            deployment_timeout=14_400.0,
        )
    )

    executor.get_deployment()

    assert captured["runtime_timeout"] == 120.0
    assert captured["deployment_timeout"] == 14_400.0


@pytest.mark.anyio
@pytest.mark.parametrize("transport_error", [TimeoutError, ConnectionResetError])
async def test_command_with_lost_response_is_not_replayed(transport_error) -> None:
    executor = _executor(runtime_failures=1, health=[True], transport_error=transport_error)
    with pytest.raises(SandboxTransportError):
        await executor.execute_command("expensive-command")
    assert executor._runtime.calls == 1
    assert executor.is_running


@pytest.mark.anyio
async def test_command_can_retry_a_connection_that_was_never_established() -> None:
    from types import SimpleNamespace

    class ConnectorError(aiohttp.ClientConnectorError):
        def __init__(self, message):
            key = SimpleNamespace(host="sandbox", port=80, ssl=False)
            super().__init__(key, OSError(111, message))

    executor = _executor(runtime_failures=1, health=[True], transport_error=ConnectorError)
    assert (await executor.execute_command("true")).success
    assert executor._runtime.calls == 2
    assert executor._deployment.calls == 0
