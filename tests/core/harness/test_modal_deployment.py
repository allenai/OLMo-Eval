"""Tests for the managed Modal deployment adapter."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import modal
import pytest

from olmo_eval.harness.sandbox.modal_deployment import ManagedModalDeployment, stop_modal_app


@pytest.mark.anyio
async def test_modal_deployment_uses_standard_transport_and_safe_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    tunnel = SimpleNamespace(url="https://sandbox.example")
    sandbox = SimpleNamespace(
        object_id="sb-1",
        tunnels=SimpleNamespace(aio=_return_value({8880: tunnel})),
        poll=SimpleNamespace(aio=_return_value(None)),
        terminate=SimpleNamespace(aio=_record_call(captured, "terminate_calls")),
    )

    async def create(*args: object, **kwargs: Any) -> object:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return sandbox

    class _Runtime:
        def __init__(self, **kwargs: Any) -> None:
            captured["runtime"] = kwargs

        async def close(self) -> None:
            captured["close_calls"] = captured.get("close_calls", 0) + 1

    deployment = object.__new__(ManagedModalDeployment)
    deployment._runtime = None
    deployment._sandbox = None
    deployment._port = 8880
    deployment._image = object()
    deployment._deployment_timeout = 600.0
    deployment._runtime_timeout = 60.0
    deployment._startup_timeout = 30.0
    deployment._modal_kwargs = {"unencrypted_ports": [9999], "cpu": 2}
    deployment._app = object()
    deployment._hooks = SimpleNamespace(on_custom_step=lambda _step: None)
    deployment.logger = logging.getLogger(__name__)
    monkeypatch.setattr(deployment, "_get_token", lambda: "token")
    monkeypatch.setattr(deployment, "_start_swerex_cmd", lambda _token: "start")
    monkeypatch.setattr(deployment, "get_modal_log_url", _return_value("https://logs"))
    monkeypatch.setattr(deployment, "_wait_until_alive", _return_value(None))
    monkeypatch.setattr(
        "olmo_eval.harness.sandbox.modal_deployment.modal.Sandbox",
        SimpleNamespace(create=SimpleNamespace(aio=create)),
    )
    monkeypatch.setattr(
        "olmo_eval.harness.sandbox.modal_deployment.RemoteRuntime",
        _Runtime,
    )
    monkeypatch.setattr(
        "olmo_eval.harness.sandbox.modal_deployment.asyncio.sleep",
        _return_value(None),
    )

    await deployment.start()

    kwargs = captured["kwargs"]
    assert kwargs["encrypted_ports"] == [8880]
    assert "unencrypted_ports" not in kwargs
    assert kwargs["cpu"] == 2
    assert captured["runtime"]["host"] == "https://sandbox.example"
    await deployment.stop()
    assert captured["terminate_calls"] == 1
    assert captured.get("close_calls", 0) == 0


@pytest.mark.anyio
@pytest.mark.parametrize("failure_stage", ["tunnel", "health_check"])
async def test_modal_deployment_terminates_sandbox_when_startup_fails(
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    captured: dict[str, Any] = {}
    tunnel = SimpleNamespace(url="https://sandbox.example")

    async def tunnels() -> dict[int, object]:
        if failure_stage == "tunnel":
            raise RuntimeError("tunnel lookup failed")
        return {8880: tunnel}

    sandbox = SimpleNamespace(
        object_id="sb-1",
        tunnels=SimpleNamespace(aio=tunnels),
        terminate=SimpleNamespace(aio=_record_call(captured, "terminate_calls")),
    )

    class _Runtime:
        def __init__(self, **_kwargs: Any) -> None:
            pass

    deployment = object.__new__(ManagedModalDeployment)
    deployment._runtime = None
    deployment._sandbox = None
    deployment._port = 8880
    deployment._image = object()
    deployment._deployment_timeout = 600.0
    deployment._runtime_timeout = 60.0
    deployment._startup_timeout = 30.0
    deployment._modal_kwargs = {}
    deployment._app = object()
    deployment._hooks = SimpleNamespace(on_custom_step=lambda _step: None)
    deployment.logger = logging.getLogger(__name__)
    monkeypatch.setattr(deployment, "_get_token", lambda: "token")
    monkeypatch.setattr(deployment, "_start_swerex_cmd", lambda _token: "start")
    monkeypatch.setattr(deployment, "get_modal_log_url", _return_value("https://logs"))

    async def wait_until_alive(**_kwargs: object) -> None:
        raise RuntimeError("health check failed")

    monkeypatch.setattr(deployment, "_wait_until_alive", wait_until_alive)
    monkeypatch.setattr(
        "olmo_eval.harness.sandbox.modal_deployment.modal.Sandbox",
        SimpleNamespace(create=SimpleNamespace(aio=_return_value(sandbox))),
    )
    monkeypatch.setattr(
        "olmo_eval.harness.sandbox.modal_deployment.RemoteRuntime",
        _Runtime,
    )
    monkeypatch.setattr(
        "olmo_eval.harness.sandbox.modal_deployment.asyncio.sleep",
        _return_value(None),
    )

    with pytest.raises(RuntimeError, match="failed"):
        await deployment.start()

    assert captured["terminate_calls"] == 1
    assert deployment._sandbox is None
    assert deployment._runtime is None


@pytest.mark.anyio
async def test_stop_modal_app_stops_by_name_and_ignores_missing_apps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stopped: list[str] = []

    async def stop_app(name: str) -> None:
        if stopped:
            raise modal.exception.NotFoundError(f"App '{name}' not found")
        stopped.append(name)

    monkeypatch.setattr(
        "olmo_eval.harness.sandbox.modal_deployment.modal.experimental.stop_app",
        SimpleNamespace(aio=stop_app),
    )

    await stop_modal_app("swerex-abc")
    await stop_modal_app("swerex-abc")

    assert stopped == ["swerex-abc"]


def _return_value(value: Any):
    async def inner(*_args: object, **_kwargs: object) -> Any:
        return value

    return inner


def _record_call(captured: dict[str, Any], key: str):
    async def inner(*_args: object, **_kwargs: object) -> None:
        captured[key] = captured.get(key, 0) + 1

    return inner
