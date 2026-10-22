"""Tests for strict per-executor sandbox capacity and routing."""

from __future__ import annotations

import asyncio
import threading

import pytest

from olmo_eval.harness.sandbox import Capability
from olmo_eval.harness.sandbox.errors import SandboxTransportError
from tests.core.harness.sandbox_stubs import TrackingExecutor, make_manager


@pytest.mark.anyio
async def test_each_executor_enforces_its_own_capacity() -> None:
    executors = [TrackingExecutor(f"sandbox-{idx}") for idx in range(3)]
    manager = make_manager(*executors)
    environment = manager.for_capabilities(Capability.DEFAULT)

    await asyncio.gather(*(environment.execute_code("pass") for _ in range(18)))

    assert sum(executor.calls for executor in executors) == 18
    assert all(executor.calls > 0 for executor in executors)
    assert all(executor.peak <= 2 for executor in executors)


@pytest.mark.anyio
async def test_capability_environment_routes_each_operation_across_pool() -> None:
    capability = frozenset({"sandbox:bigcodebench"})
    executors = [
        TrackingExecutor(f"bcb-{idx}", capabilities=capability, max_concurrency=1)
        for idx in range(5)
    ]
    manager = make_manager(*executors)
    environment = manager.for_capabilities(capability)

    await asyncio.gather(*(environment.execute_code("pass") for _ in range(5)))

    assert [executor.calls for executor in executors] == [1, 1, 1, 1, 1]


@pytest.mark.anyio
async def test_scoring_environment_fails_over_after_quarantine() -> None:
    broken = TrackingExecutor("broken", fail_and_quarantine=True)
    healthy = TrackingExecutor("healthy")
    manager = make_manager(broken, healthy)
    environment = manager.for_capabilities(Capability.DEFAULT)

    result = await environment.execute_code("pass")

    assert result.success is True
    assert broken.calls == 1
    assert healthy.calls == 1


@pytest.mark.anyio
async def test_scoring_environment_does_not_fail_over_from_healthy_sandbox() -> None:
    broken = TrackingExecutor("broken", fail_transport=True)
    healthy = TrackingExecutor("healthy")
    manager = make_manager(broken, healthy)
    environment = manager.for_capabilities(Capability.DEFAULT)

    with pytest.raises(SandboxTransportError):
        await environment.execute_code("pass")

    assert broken.is_running is True
    assert broken.calls == 1
    assert healthy.calls == 0


class RecoverableExecutor(TrackingExecutor):
    async def stop(self) -> None:
        self.running = False

    async def start(self) -> None:
        self.running = True


@pytest.mark.anyio
async def test_shutdown_rejects_new_session_bindings(monkeypatch) -> None:
    executor = RecoverableExecutor("healthy")
    manager = make_manager(executor)
    binding = await manager.acquire_binding(Capability.DEFAULT)
    stopping = asyncio.Event()
    finish = asyncio.Event()

    async def stop_executors(executors):
        stopping.set()
        await finish.wait()
        for executor in executors:
            await executor.stop()

    monkeypatch.setattr(manager, "_stop_executors", stop_executors)
    stop_task = asyncio.create_task(manager.stop())
    try:
        await asyncio.wait_for(stopping.wait(), 2)
        assert executor.is_running
        assert binding._released
        with pytest.raises(ValueError, match="Sandbox manager is stopping"):
            await manager.acquire_binding(Capability.DEFAULT)
        assert not manager._bindings
        assert not manager._bound_executors
    finally:
        finish.set()
        await stop_task


@pytest.fixture
def replacements(monkeypatch):
    created = []

    class Replacement(RecoverableExecutor):
        def __init__(self, config, name, modal_app_name):
            super().__init__(
                name, capabilities=config.capabilities, max_concurrency=config.max_concurrency
            )
            self.running = False
            created.append(self)

    monkeypatch.setattr("olmo_eval.harness.sandbox.manager.SandboxExecutor", Replacement)
    return created


@pytest.mark.anyio
async def test_pool_recovers_once_for_concurrent_scoring_calls(replacements) -> None:
    broken = RecoverableExecutor("broken", max_concurrency=1, fail_and_quarantine=True)
    manager = make_manager(broken)
    environment = manager.for_capabilities(Capability.DEFAULT)
    try:
        results = await asyncio.wait_for(
            asyncio.gather(
                *(environment.execute_code("pass") for _ in range(8)), return_exceptions=True
            ),
            2,
        )
        assert sum(isinstance(result, Exception) for result in results) == 1
        assert all(result.success for result in results if not isinstance(result, Exception))
        assert broken.calls == 1
        assert len(replacements) == 1
        assert replacements[0].calls == 7
        assert replacements[0].peak == 1
    finally:
        await manager.stop()


@pytest.mark.anyio
async def test_exhausted_pool_is_recoverable_before_scoring_starts(replacements) -> None:
    broken = RecoverableExecutor("broken")
    broken.running = False
    manager = make_manager(broken)
    try:
        assert manager.is_running
        environment = manager.for_capabilities(Capability.DEFAULT)
        assert environment.is_running
        assert (await environment.execute_code("pass")).success
        assert len(replacements) == 1
    finally:
        await manager.stop()


@pytest.mark.anyio
async def test_bound_session_is_not_replaced(replacements) -> None:
    from olmo_eval.harness.sandbox.errors import SandboxInfrastructureError

    broken = RecoverableExecutor("bound")
    manager = make_manager(broken)
    binding = await manager.acquire_binding(Capability.DEFAULT)
    broken.running = False
    environment = manager.for_capabilities(Capability.DEFAULT)
    try:
        with pytest.raises(SandboxInfrastructureError):
            await environment.execute_code("pass")
        assert not replacements
        await binding.release()
        assert (await environment.execute_code("pass")).success
        assert len(replacements) == 1
    finally:
        await manager.stop()


@pytest.mark.anyio
async def test_failed_restart_does_not_repeat_for_later_samples(monkeypatch, replacements) -> None:
    from olmo_eval.harness.sandbox.errors import SandboxInfrastructureError

    async def fail_start(self):
        raise RuntimeError("container startup failed")

    broken = RecoverableExecutor("broken")
    broken.running = False
    manager = make_manager(broken)
    environment = manager.for_capabilities(Capability.DEFAULT)
    monkeypatch.setattr("olmo_eval.harness.sandbox.manager.SandboxExecutor.start", fail_start)
    try:
        for _ in range(8):
            with pytest.raises(SandboxInfrastructureError):
                await asyncio.wait_for(environment.execute_code("pass"), 2)
        assert len(replacements) == 1
        assert not environment.is_running
    finally:
        await manager.stop()


@pytest.mark.anyio
async def test_one_operation_cannot_consume_the_pool_restart_budget(replacements) -> None:
    from olmo_eval.harness.sandbox.errors import SandboxInfrastructureError

    broken = [RecoverableExecutor(str(i), fail_and_quarantine=True) for i in range(4)]
    manager = make_manager(*broken)
    try:
        with pytest.raises(SandboxInfrastructureError, match="after 2 attempts"):
            await manager.for_capabilities(Capability.DEFAULT).execute_code("poison")
        assert sum(executor.calls for executor in broken) == 2
        assert len(manager._restart_counts) == 2
    finally:
        await manager.stop()


@pytest.mark.anyio
async def test_successful_replacements_have_a_bounded_budget(replacements) -> None:
    from olmo_eval.harness.sandbox.errors import SandboxInfrastructureError

    manager = make_manager(RecoverableExecutor("broken"))
    environment = manager.for_capabilities(Capability.DEFAULT)
    try:
        for _ in range(2):
            manager._executors[0].running = False
            assert (await environment.execute_code("pass")).success
        manager._executors[0].running = False
        with pytest.raises(SandboxInfrastructureError):
            await environment.execute_code("pass")
        assert len(replacements) == 2
        assert not environment.is_running
    finally:
        await manager.stop()


@pytest.mark.anyio
async def test_lease_does_not_wait_for_an_excluded_replacement() -> None:
    broken = RecoverableExecutor("excluded")
    broken.running = False
    manager = make_manager(broken)
    recovery = asyncio.create_task(asyncio.Event().wait())
    manager._recoveries[0] = recovery

    async def lease():
        async with manager.lease_executor(Capability.DEFAULT, excluded=frozenset({0})):
            pytest.fail("An excluded slot was leased")

    try:
        with pytest.raises(ValueError, match="No sandbox supports"):
            await asyncio.wait_for(lease(), 0.1)
    finally:
        recovery.cancel()
        await asyncio.gather(recovery, return_exceptions=True)
        await manager.stop()


@pytest.mark.anyio
@pytest.mark.parametrize("restart", [1, 2])
@pytest.mark.parametrize("operation", ["scoring", "binding"])
@pytest.mark.parametrize("fail_start", [False, True])
async def test_requests_wait_for_inflight_session_recovery(
    monkeypatch, replacements, restart: int, operation: str, fail_start: bool
) -> None:
    from olmo_eval.harness.sandbox.errors import SandboxInfrastructureError

    started = threading.Event()
    finish = threading.Event()

    async def gated_start(self):
        started.set()
        assert finish.wait(5)
        if fail_start:
            raise RuntimeError("startup failed")
        self.running = True

    monkeypatch.setattr("olmo_eval.harness.sandbox.manager.SandboxExecutor.start", gated_start)
    broken = RecoverableExecutor("bound")
    manager = make_manager(broken)
    manager._restart_counts[0] = restart - 1
    binding = await manager.acquire_binding(Capability.DEFAULT)
    broken.running = False
    await binding.release()
    environment = manager.for_capabilities(Capability.DEFAULT)
    pending = asyncio.create_task(
        environment.execute_code("pass")
        if operation == "scoring"
        else manager.acquire_binding(Capability.DEFAULT)
    )
    try:
        assert await asyncio.to_thread(started.wait, 2)
        assert manager._restart_counts[0] == restart
        assert manager.is_running
        assert environment.is_running
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(asyncio.shield(pending), 0.05)
        finish.set()
        if fail_start:
            error = SandboxInfrastructureError if operation == "scoring" else ValueError
            with pytest.raises(error):
                await asyncio.wait_for(pending, 2)
            assert not environment.is_running
        else:
            result = await asyncio.wait_for(pending, 2)
            if operation == "scoring":
                assert result.success
            else:
                assert result.executor is replacements[0]
                await result.release()
        assert len(replacements) == 1
    finally:
        finish.set()
        await asyncio.gather(pending, return_exceptions=True)
        await manager.stop()


@pytest.mark.anyio
async def test_exhausted_budget_warning_is_emitted_once(monkeypatch):
    from unittest.mock import Mock

    executor = RecoverableExecutor("spent")
    executor.running = False
    manager = make_manager(executor)
    manager._restart_counts[0] = 2
    warning = Mock()
    monkeypatch.setattr(manager._logger, "warning", warning)
    manager._schedule_recovery(0, executor)
    manager._schedule_recovery(0, executor)
    warning.assert_called_once()
    assert "after 2 restarts" in warning.call_args.args[0]
    await manager.stop()


@pytest.mark.anyio
async def test_shutdown_waits_for_recovery_and_stops_the_replacement(monkeypatch, replacements):
    from olmo_eval.harness.sandbox.errors import SandboxInfrastructureError

    old_stopped, started, finish = threading.Event(), threading.Event(), threading.Event()
    broken = RecoverableExecutor("broken")
    broken.running = False
    manager = make_manager(broken)

    async def stop(self):
        self.running = False
        if self is broken:
            old_stopped.set()

    async def start(self):
        assert old_stopped.is_set()
        started.set()
        assert finish.wait(5)
        self.running = True

    monkeypatch.setattr(RecoverableExecutor, "stop", stop)
    monkeypatch.setattr("olmo_eval.harness.sandbox.manager.SandboxExecutor.start", start)
    operation = asyncio.create_task(
        manager.for_capabilities(Capability.DEFAULT).execute_code("pass")
    )
    shutdown = None
    try:
        assert await asyncio.to_thread(started.wait, 2)
        shutdown = asyncio.create_task(manager.stop())
        await asyncio.sleep(0)
        assert manager._stopping
        assert not shutdown.done()
        with pytest.raises(SandboxInfrastructureError):
            await asyncio.wait_for(operation, 2)
        finish.set()
        await asyncio.wait_for(shutdown, 2)
        assert len(replacements) == 1 and not replacements[0].is_running
        assert not manager._executors
    finally:
        finish.set()
        await asyncio.gather(operation, return_exceptions=True)
        if shutdown is not None:
            await shutdown
        else:
            await manager.stop()


@pytest.mark.anyio
async def test_operation_failure_after_shutdown_keeps_infrastructure_classification():
    from olmo_eval.harness.sandbox.errors import SandboxInfrastructureError

    entered, finish = asyncio.Event(), asyncio.Event()

    class LeasedExecutor(RecoverableExecutor):
        async def execute_code(self, *args, **kwargs):
            entered.set()
            await finish.wait()
            self.running = False
            raise ConnectionResetError("stopped during operation")

    manager = make_manager(LeasedExecutor("active"))
    operation = asyncio.create_task(
        manager.for_capabilities(Capability.DEFAULT).execute_code("pass")
    )
    await entered.wait()
    await manager.stop()
    finish.set()
    with pytest.raises(SandboxInfrastructureError, match="manager is stopping"):
        await operation


@pytest.mark.anyio
@pytest.mark.parametrize("stage", ["create", "execute"])
async def test_real_session_transport_failure_recovers_on_binding_release(
    monkeypatch, replacements, stage
):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from olmo_eval.harness.sandbox import SandboxConfig, SandboxExecutor, SandboxMode

    executor = SandboxExecutor(
        SandboxConfig(image="test", mode=SandboxMode.MODAL), name="broken-session"
    )
    error = ConnectionResetError("session disconnected")
    executor._runtime = SimpleNamespace(
        create_session=AsyncMock(side_effect=error if stage == "create" else None),
        run_in_session=AsyncMock(side_effect=error),
    )
    executor._deployment = SimpleNamespace(is_alive=AsyncMock(return_value=False))
    executor._session_created = stage == "execute"
    monkeypatch.setattr("olmo_eval.harness.sandbox.executor._HEALTH_CHECK_RETRIES", 0)
    manager = make_manager(executor)
    binding = await manager.acquire_binding(Capability.DEFAULT)
    try:
        with pytest.raises(SandboxTransportError, match="session disconnected"):
            await binding.execute_in_session("true")
        assert not executor.is_running
        await binding.release()
        recovered = await asyncio.wait_for(manager.acquire_binding(Capability.DEFAULT), 2)
        assert recovered.executor is replacements[0]
        await recovered.release()
    finally:
        await binding.release()
        await manager.stop()
