import asyncio
import sys
import types
import unittest
from dataclasses import replace
from unittest import mock

import pytest

from olmo_eval.harness.sandbox.config import SandboxConfig, SandboxMode
from olmo_eval.harness.sandbox.executor import SandboxExecutor


class _FakeProcess:
    def __init__(self, *, block: bool = False) -> None:
        self.block = block
        self.returncode: int | None = None
        self.killed = False

    async def communicate(self) -> tuple[bytes, bytes]:
        if self.block:
            await asyncio.Event().wait()
        self.returncode = 0
        return b"control output\n", b""

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    async def wait(self) -> int:
        return self.returncode or 0


class TestStreamingControlCommand(unittest.IsolatedAsyncioTestCase):
    def _executor(self) -> SandboxExecutor:
        executor = SandboxExecutor(
            SandboxConfig(image="test", mode=SandboxMode.DOCKER, container_runtime="podman")
        )
        executor._deployment = mock.Mock(container_name="sandbox-name")
        executor._runtime = mock.Mock()
        return executor

    async def test_docker_control_bypasses_swerex(self) -> None:
        process = _FakeProcess()
        executor = self._executor()

        with mock.patch("asyncio.create_subprocess_exec", return_value=process) as create:
            result = await executor._execute_stream_control("echo ok", timeout=2.0)

        create.assert_awaited_once_with(
            "podman",
            "exec",
            "sandbox-name",
            "bash",
            "-c",
            "echo ok",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        executor._runtime.execute.assert_not_called()
        self.assertEqual(result.stdout, "control output\n")
        self.assertEqual(result.exit_code, 0)

    async def test_docker_control_timeout_kills_client(self) -> None:
        process = _FakeProcess(block=True)
        executor = self._executor()

        with (
            mock.patch("asyncio.create_subprocess_exec", return_value=process),
            self.assertRaises(TimeoutError),
        ):
            await executor._execute_stream_control("blocked", timeout=0.01)

        self.assertTrue(process.killed)

    async def test_streaming_aborts_when_control_command_cannot_enter_container(self) -> None:
        executor = self._executor()
        executor._runtime.execute = mock.AsyncMock(
            return_value=mock.Mock(stdout="", stderr="", exit_code=0)
        )
        in_progress = mock.Mock(
            stdout="---EXIT_CODE---\n",
            stderr="",
            exit_code=1,
        )
        container_failure = mock.Mock(
            stdout="",
            stderr="container is not running",
            exit_code=125,
        )
        control = mock.AsyncMock(side_effect=[in_progress, *([container_failure] * 5)])
        abstract = types.ModuleType("swerex.runtime.abstract")
        abstract.Command = mock.Mock()  # type: ignore[ty:unresolved-attribute]
        runtime = types.ModuleType("swerex.runtime")
        runtime.abstract = abstract  # type: ignore[ty:unresolved-attribute]
        swerex = types.ModuleType("swerex")
        swerex.runtime = runtime  # type: ignore[ty:unresolved-attribute]

        with (
            mock.patch.dict(
                sys.modules,
                {
                    "swerex": swerex,
                    "swerex.runtime": runtime,
                    "swerex.runtime.abstract": abstract,
                },
            ),
            mock.patch("asyncio.sleep", new=mock.AsyncMock()),
            mock.patch.object(executor, "_execute_stream_control", new=control),
        ):
            result = await executor._execute_streaming("long-running", 60.0, "test")

        self.assertFalse(result.success)
        self.assertEqual(result.exit_code, -1)
        self.assertIn("Sandbox unresponsive after 3 polls", result.output)
        # The initial live poll exits 1 while the command is still running. It is
        # followed by three failed polls, then best-effort kill and cleanup commands.
        self.assertEqual(control.await_count, 6)

    async def test_cancelled_control_command_kills_client(self) -> None:
        process = _FakeProcess(block=True)
        executor = self._executor()
        with mock.patch("asyncio.create_subprocess_exec", return_value=process):
            task = asyncio.create_task(executor._execute_stream_control("blocked", timeout=60))
            await asyncio.sleep(0)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertTrue(process.killed)

    async def test_local_deadline_kills_the_process_group(self) -> None:
        executor = self._executor()
        response = mock.Mock(stdout="", stderr="", exit_code=0)
        control = mock.AsyncMock(side_effect=[TimeoutError(), response])
        from olmo_eval.harness.sandbox.errors import SandboxTransportError

        with (
            mock.patch.object(executor, "_execute_stream_control", new=control),
            mock.patch.object(
                executor, "_quarantine_if_unresponsive", new=mock.AsyncMock()
            ) as health,
            self.assertRaises(SandboxTransportError),
        ):
            await executor.execute_command("blocked", timeout=0.01)
        health.assert_awaited_once()
        commands = [call.args[0] for call in control.await_args_list]
        self.assertEqual(commands[0][:3], ["setsid", "--wait", "bash"])
        self.assertIn("--kill-after=5s", commands[0][4])
        self.assertIn("kill -KILL", commands[1][2])
        executor._runtime.execute.assert_not_called()

    async def test_local_cancellation_kills_the_process_group(self) -> None:
        executor = self._executor()
        started = asyncio.Event()
        cleanup = []

        async def control(command, timeout):
            if isinstance(command, list) and command[0] == "setsid":
                started.set()
                await asyncio.Event().wait()
            cleanup.append(command)
            return mock.Mock(stdout="", stderr="", exit_code=0)

        with mock.patch.object(executor, "_execute_stream_control", new=control):
            task = asyncio.create_task(executor.execute_command("blocked", timeout=60))
            await started.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertEqual(len(cleanup), 1)
        self.assertIn("kill -KILL", cleanup[0][2])

    async def test_local_health_probe_bypasses_blocked_http_server(self) -> None:
        executor = self._executor()
        executor._deployment.is_alive = mock.AsyncMock(return_value=True)
        control = mock.AsyncMock(return_value=mock.Mock(exit_code=0))
        with mock.patch.object(executor, "_execute_stream_control", new=control):
            await executor._quarantine_if_unresponsive(TimeoutError())
        self.assertTrue(executor.is_running)
        executor._deployment.is_alive.assert_awaited_once_with(timeout=5.0)
        control.assert_awaited_once_with("true", timeout=5.0)

    async def test_dead_container_is_an_infrastructure_failure(self) -> None:
        from olmo_eval.harness.sandbox.errors import SandboxTransportError

        executor = self._executor()
        control = mock.AsyncMock(
            return_value=mock.Mock(stdout="", stderr="container stopped", exit_code=125)
        )
        with (
            mock.patch.object(executor, "_execute_stream_control", new=control),
            mock.patch("olmo_eval.harness.sandbox.executor._HEALTH_CHECK_RETRIES", 0),
            self.assertRaises(SandboxTransportError),
        ):
            await executor.execute_command("true")
        self.assertFalse(executor.is_running)

    async def test_live_container_with_dead_server_is_quarantined(self) -> None:
        executor = self._executor()
        executor._deployment.is_alive = mock.AsyncMock(return_value=False)
        control = mock.AsyncMock(return_value=mock.Mock(exit_code=0))
        with (
            mock.patch.object(executor, "_execute_stream_control", new=control),
            mock.patch("olmo_eval.harness.sandbox.executor._HEALTH_CHECK_RETRIES", 0),
        ):
            await executor._quarantine_if_unresponsive(TimeoutError())
        self.assertFalse(executor.is_running)

    async def test_slow_cleanup_uses_health_checks_before_quarantine(self) -> None:
        executor = self._executor()
        executor._deployment.is_alive = mock.AsyncMock(return_value=True)
        from olmo_eval.harness.sandbox.errors import SandboxTransportError

        control = mock.AsyncMock(
            side_effect=[
                TimeoutError(),
                mock.Mock(exit_code=0),
                TimeoutError(),
                mock.Mock(exit_code=0),
            ]
        )
        with (
            mock.patch.object(executor, "_execute_stream_control", new=control),
            self.assertRaises(SandboxTransportError),
        ):
            await executor.execute_command("blocked", timeout=0.01)
        self.assertTrue(executor.is_running)
        self.assertEqual(executor._deployment.is_alive.await_count, 2)

    async def test_local_execute_code_uses_the_bounded_command_path(self) -> None:
        from olmo_eval.common.execution import ExecutionResult

        executor = self._executor()
        expected = ExecutionResult(True, "1")
        execute = mock.AsyncMock(return_value=expected)
        with mock.patch.object(executor, "_execute_local_command", new=execute):
            result = await executor.execute_code("print(1)", timeout=2)
        self.assertIs(result, expected)
        execute.assert_awaited_once_with(["python3", "-c", "print(1)"], 2)
        executor._runtime.execute.assert_not_called()

    async def test_custom_exec_shell_preserves_the_initialized_http_environment(self) -> None:
        from olmo_eval.common.execution import ExecutionResult

        executor = self._executor()
        executor.config = replace(executor.config, exec_shell=("bash", "-l", "-c"))
        deployment = executor._deployment
        deployment.start = mock.AsyncMock()
        expected = ExecutionResult(True, "activated")
        remote = mock.AsyncMock(return_value=expected)
        with (
            mock.patch.object(executor, "get_deployment", return_value=deployment),
            mock.patch.object(executor, "_execute_stream_control") as control,
            mock.patch.object(executor, "_execute_local_command") as local,
            mock.patch.object(executor, "_execute_remote_command", new=remote),
        ):
            await executor.start()
            self.assertIs(await executor.execute_command("tool", timeout=2), expected)
            self.assertIs(await executor.execute_code("print(1)", timeout=2), expected)
        self.assertFalse(executor._local_command_supported)
        control.assert_not_called()
        local.assert_not_called()
        self.assertEqual(
            remote.await_args_list,
            [mock.call(["bash", "-c", "tool"], 2), mock.call(["python3", "-c", "print(1)"], 2)],
        )

    async def test_failed_capability_probe_does_not_silently_enable_http(self) -> None:
        executor = self._executor()
        deployment = executor._deployment
        deployment.start = mock.AsyncMock()
        stopped = mock.AsyncMock()
        with (
            mock.patch.object(executor, "get_deployment", return_value=deployment),
            mock.patch.object(executor, "stop", new=stopped),
            mock.patch.object(executor, "_execute_stream_control", side_effect=TimeoutError()),
            self.assertRaises(TimeoutError),
        ):
            await executor.start()
        self.assertIsNone(executor._local_command_supported)
        stopped.assert_awaited_once()

    async def test_runtime_probe_failure_does_not_enable_http(self) -> None:
        from olmo_eval.harness.sandbox.errors import SandboxTransportError

        executor = self._executor()
        deployment = executor._deployment
        deployment.start = mock.AsyncMock()
        stopped = mock.AsyncMock()
        with (
            mock.patch.object(executor, "get_deployment", return_value=deployment),
            mock.patch.object(executor, "stop", new=stopped),
            mock.patch.object(
                executor,
                "_execute_stream_control",
                return_value=mock.Mock(exit_code=125, stderr="OCI failed"),
            ),
            self.assertRaises(SandboxTransportError),
        ):
            await executor.start()
        self.assertIsNone(executor._local_command_supported)
        stopped.assert_awaited_once()

    async def test_unsupported_tools_keep_the_healthy_http_route(self) -> None:
        executor = self._executor()
        deployment = executor._deployment
        deployment.start = mock.AsyncMock()
        deployment.is_alive = mock.AsyncMock(return_value=True)
        with (
            mock.patch.object(executor, "get_deployment", return_value=deployment),
            mock.patch.object(
                executor, "_execute_stream_control", return_value=mock.Mock(exit_code=1)
            ),
            self.assertLogs("olmo_eval.harness.sandbox.executor", level="WARNING"),
        ):
            await executor.start()
        self.assertFalse(executor._local_command_supported)
        deployment.is_alive.assert_awaited_once()

    async def test_modal_control_uses_swerex(self) -> None:
        executor = SandboxExecutor(SandboxConfig(image="test", mode=SandboxMode.MODAL))
        executor._deployment = mock.Mock()
        executor._runtime = mock.Mock()
        executor._runtime.execute = mock.AsyncMock(
            return_value=mock.Mock(stdout="remote output\n", stderr="", exit_code=0)
        )
        abstract = types.ModuleType("swerex.runtime.abstract")
        abstract.Command = mock.Mock()  # type: ignore[ty:unresolved-attribute]
        runtime = types.ModuleType("swerex.runtime")
        runtime.abstract = abstract  # type: ignore[ty:unresolved-attribute]
        swerex = types.ModuleType("swerex")
        swerex.runtime = runtime  # type: ignore[ty:unresolved-attribute]

        with (
            mock.patch.dict(
                sys.modules,
                {
                    "swerex": swerex,
                    "swerex.runtime": runtime,
                    "swerex.runtime.abstract": abstract,
                },
            ),
            mock.patch("asyncio.create_subprocess_exec") as create,
        ):
            result = await executor._execute_stream_control("echo ok", timeout=2.0)

        create.assert_not_called()
        executor._runtime.execute.assert_awaited_once()
        self.assertEqual(result.stdout, "remote output\n")
        self.assertEqual(result.exit_code, 0)


if __name__ == "__main__":
    unittest.main()


class _CommandTimeoutError(Exception):
    """Named like swe-rex's timeout so the executor recognizes it."""


class TestSessionTimeout(unittest.IsolatedAsyncioTestCase):
    def _executor(self) -> SandboxExecutor:
        executor = SandboxExecutor(
            SandboxConfig(image="test", mode=SandboxMode.DOCKER, container_runtime="docker")
        )
        executor._deployment = mock.Mock()
        executor._runtime = mock.Mock()
        executor._session_created = True
        return executor

    async def test_a_timed_out_command_is_interrupted_and_reported(self) -> None:
        executor = self._executor()
        interrupted = mock.Mock(output="^C", exit_code=0)
        executor._runtime.run_in_session = mock.AsyncMock(
            side_effect=[_CommandTimeoutError("timed out after 5s"), interrupted]
        )

        result = await executor.execute_in_session("sleep 100", timeout=5.0)

        self.assertFalse(result.success)
        self.assertEqual(result.error, "timeout")
        self.assertEqual(result.exit_code, -1)
        self.assertIn("timed out after 5.0s", result.output)
        actions = [c.args[0] for c in executor._runtime.run_in_session.await_args_list]
        self.assertEqual(actions[0].command, "sleep 100")
        self.assertEqual(type(actions[1]).__name__, "BashInterruptAction")

    async def test_a_syntax_error_comes_back_as_a_result(self) -> None:
        executor = self._executor()

        class BashIncorrectSyntaxError(RuntimeError):
            pass

        executor._runtime.run_in_session = mock.AsyncMock(
            side_effect=BashIncorrectSyntaxError("bash: line 1: syntax error near `('")
        )

        result = await executor.execute_in_session("echo (")

        self.assertFalse(result.success)
        self.assertEqual(result.exit_code, 2)
        self.assertEqual(result.error, "syntax")
        self.assertIn("syntax error", result.output)
        self.assertEqual(executor._runtime.run_in_session.await_count, 1)

    async def test_other_failures_still_raise(self) -> None:
        executor = self._executor()
        executor._runtime.run_in_session = mock.AsyncMock(side_effect=RuntimeError("gone"))

        with self.assertRaises(RuntimeError):
            await executor.execute_in_session("ls")

    async def test_interrupt_without_a_session_is_a_no_op(self) -> None:
        executor = self._executor()
        executor._session_created = False
        executor._runtime.run_in_session = mock.AsyncMock()

        await executor.interrupt_session()

        executor._runtime.run_in_session.assert_not_awaited()


@pytest.mark.anyio
@pytest.mark.parametrize("cancel_before_record", [False, True])
async def test_session_cleanup_reaches_nested_process_groups(
    monkeypatch, tmp_path, cancel_before_record
):
    import shlex
    import subprocess
    import threading
    from types import SimpleNamespace

    proc = tmp_path / "proc"
    proc.mkdir()
    record = tmp_path / "record"
    kills = tmp_path / "kills"

    def stat(pid, group, session, stamp):
        fields = ["0"] * 20
        fields[0], fields[2], fields[3], fields[19] = "S", str(group), str(session), str(stamp)
        return f"{pid} (process (name)) " + " ".join(fields) + "\n"

    if not cancel_before_record:
        record.write_text(stat(100, 100, 100, 7))
    for pid, group, session in [(100, 100, 100), (101, 100, 100), (102, 102, 100), (200, 200, 200)]:
        path = proc / str(pid)
        path.mkdir()
        (path / "stat").write_text(stat(pid, group, session, 7))

    monkeypatch.setattr(
        "olmo_eval.harness.sandbox.executor.uuid.uuid4", lambda: SimpleNamespace(hex="test")
    )
    executor = SandboxExecutor(SandboxConfig(image="test", mode=SandboxMode.DOCKER))
    executor._deployment = SimpleNamespace(container_name="test")
    executor._runtime = SimpleNamespace()

    async def control(command, timeout):
        if command[0] == "setsid":
            if cancel_before_record:
                raise asyncio.CancelledError
            return SimpleNamespace(stdout="__OLMO_STARTED_test__\n", stderr="", exit_code=124)
        script = command[2].replace("/proc", shlex.quote(str(proc)))
        # Intercept signals so the real cleanup script operates only on the fixture's process table.
        script = 'kill() { printf "%s\\n" "$3" >> ' + shlex.quote(str(kills)) + "; };\n" + script
        delayed_record = None
        if cancel_before_record:
            delayed_record = threading.Timer(0.1, record.write_text, args=(stat(100, 100, 100, 7),))
            delayed_record.start()
        result = subprocess.run(
            ["bash", "-c", script, "_", str(record)], capture_output=True, text=True, check=True
        )
        if delayed_record is not None:
            delayed_record.join()
        return SimpleNamespace(
            stdout=result.stdout, stderr=result.stderr, exit_code=result.returncode
        )

    monkeypatch.setattr(executor, "_execute_stream_control", control)
    if cancel_before_record:
        with pytest.raises(asyncio.CancelledError):
            await executor.execute_command("true")
    else:
        result = await executor.execute_command("true", timeout=0)
        assert result.error == "timeout"
    assert {int(pid) for pid in kills.read_text().splitlines()} == {100, 101, 102}
    assert not record.exists()


@pytest.mark.anyio
@pytest.mark.parametrize(
    "exit_code,elapsed,label",
    [
        (124, 0.1, None),
        (137, 0.1, None),
        (124, 4, "timeout"),
        (137, 4, "timeout"),
        (9, 4, "timeout"),
        (0, 4, None),
    ],
)
async def test_command_timeout_label_respects_exit_and_elapsed_time(
    monkeypatch, exit_code, elapsed, label
):
    from types import SimpleNamespace

    executor = SandboxExecutor(SandboxConfig(image="test", mode=SandboxMode.DOCKER))
    executor._deployment = SimpleNamespace(container_name="test")
    executor._runtime = SimpleNamespace()
    monkeypatch.setattr(
        "olmo_eval.harness.sandbox.executor.uuid.uuid4", lambda: SimpleNamespace(hex="test")
    )
    times = iter([0, elapsed])
    monkeypatch.setattr(
        "olmo_eval.harness.sandbox.executor.time", SimpleNamespace(monotonic=lambda: next(times))
    )
    commands = []

    async def control(command, timeout):
        commands.append(command)
        return SimpleNamespace(
            stdout="__OLMO_STARTED_test__\n",
            stderr="",
            exit_code=exit_code if command[0] == "setsid" else 0,
        )

    monkeypatch.setattr(executor, "_execute_stream_control", control)
    result = await executor.execute_command("true", timeout=3)
    assert result.error == label
    assert (commands[1][0] == "bash") == (label == "timeout")
