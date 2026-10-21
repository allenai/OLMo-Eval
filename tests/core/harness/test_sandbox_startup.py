"""Tests for sandbox startup rollback."""

from __future__ import annotations

from typing import Any

import pytest

from olmo_eval.harness.sandbox import SandboxConfig, SandboxManager, SandboxMode
from olmo_eval.harness.sandbox.executor import SandboxExecutor


@pytest.mark.anyio
async def test_executor_stops_deployment_when_start_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Deployment:
        def __init__(self) -> None:
            self.stop_calls = 0

        async def start(self) -> None:
            raise RuntimeError("startup failed")

        async def stop(self) -> None:
            self.stop_calls += 1

    deployment = _Deployment()
    executor = SandboxExecutor(
        SandboxConfig(image="test", mode=SandboxMode.DOCKER, enable_diagnostics=False)
    )
    monkeypatch.setattr(executor, "get_deployment", lambda: deployment)

    with pytest.raises(RuntimeError, match="startup failed"):
        await executor.start()

    assert deployment.stop_calls == 1
    assert executor._deployment is None
    assert executor._runtime is None


@pytest.mark.anyio
async def test_manager_stops_started_siblings_when_minimum_is_not_met(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executors: list[Any] = []

    class _Executor:
        def __init__(self, config: SandboxConfig, **kwargs: object) -> None:
            self.config = config
            self.name = kwargs["name"]
            self.stop_calls = 0
            executors.append(self)

        async def start(self) -> None:
            if str(self.name).endswith("-1"):
                raise RuntimeError("startup failed")

        async def stop(self) -> None:
            self.stop_calls += 1
            if str(self.name).endswith("-0"):
                raise RuntimeError("cleanup failed")

    monkeypatch.setattr("olmo_eval.harness.sandbox.manager.SandboxExecutor", _Executor)
    manager = SandboxManager(
        [
            SandboxConfig(
                image="test",
                mode=SandboxMode.DOCKER,
                instances=2,
                min_instances=2,
            )
        ]
    )

    with pytest.raises(RuntimeError, match="only 1/2 required instances started"):
        await manager.start()

    assert [executor.stop_calls for executor in executors] == [1, 1]
    assert manager._executors == []


@pytest.mark.anyio
async def test_executor_stops_the_modal_app_it_created(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stopped: list[str] = []

    async def stop_modal_app(app_name: str) -> None:
        stopped.append(app_name)

    class _Deployment:
        runtime = object()

        async def start(self) -> None:
            pass

        async def stop(self) -> None:
            pass

    monkeypatch.setattr("olmo_eval.harness.sandbox.modal_deployment.stop_modal_app", stop_modal_app)
    executor = SandboxExecutor(SandboxConfig(image="test", mode=SandboxMode.MODAL))
    monkeypatch.setattr(executor, "get_deployment", lambda: _Deployment())

    await executor.start()
    app_name = executor._owned_modal_app
    assert app_name is not None and app_name.startswith("swerex-")
    await executor.stop()
    await executor.stop()

    assert stopped == [app_name]


@pytest.mark.anyio
async def test_manager_stops_its_modal_app_and_executors_leave_it_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stopped: list[str] = []

    async def stop_modal_app(app_name: str) -> None:
        stopped.append(app_name)

    class _Deployment:
        runtime = object()

        async def start(self) -> None:
            pass

        async def stop(self) -> None:
            pass

    monkeypatch.setattr("olmo_eval.harness.sandbox.modal_deployment.stop_modal_app", stop_modal_app)
    monkeypatch.setattr(SandboxExecutor, "get_deployment", lambda _self: _Deployment())
    manager = SandboxManager([SandboxConfig(image="test", mode=SandboxMode.MODAL, instances=2)])

    await manager.start()
    assert all(executor._owned_modal_app is None for executor in manager._executors)
    await manager.stop()

    assert stopped == [manager._modal_app_name]
