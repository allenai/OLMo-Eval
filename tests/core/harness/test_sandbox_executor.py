import asyncio
import sys
import types
import unittest
from unittest import mock

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


class TestRunDetached(unittest.IsolatedAsyncioTestCase):
    """``run_detached`` starts a script in its own session and polls it to the end."""

    def _executor(self, polls: list[str], start_output: str = "4242\n") -> SandboxExecutor:
        from olmo_eval.common.execution import ExecutionResult
        from olmo_eval.harness.sandbox.executor import _ControlCommandResult

        executor = SandboxExecutor(
            SandboxConfig(image="test", mode=SandboxMode.DOCKER, container_runtime="podman")
        )
        executor._runtime = mock.Mock()
        executor.write_files = mock.AsyncMock()  # type: ignore[method-assign]
        executor.execute_command = mock.AsyncMock(  # type: ignore[method-assign]
            return_value=ExecutionResult(success=True, output=start_output, exit_code=0)
        )
        results = [_ControlCommandResult(stdout=p, stderr="noise", exit_code=0) for p in polls]
        last = results[-1]
        executor._execute_stream_control = mock.AsyncMock(  # type: ignore[method-assign]
            side_effect=lambda *a, **k: results.pop(0) if results else last
        )
        return executor

    async def test_a_finished_script_reports_its_exit_code(self) -> None:
        executor = self._executor(["running\n", "3\n"])
        with mock.patch("asyncio.sleep", new=mock.AsyncMock()):
            run = await executor.run_detached("make all", timeout=60.0)

        self.assertEqual(run.exit_code, 3)
        self.assertFalse(run.timed_out)
        written = executor.write_files.await_args.args[0]  # type: ignore[attr-defined]
        self.assertEqual(written, {f"{run.run_dir}/script.sh": "make all"})
        start = executor.execute_command.await_args.args[0]  # type: ignore[attr-defined]
        self.assertIn(f"mkdir {run.run_dir}/started", start)
        self.assertIn("setsid nohup bash -c", start)
        self.assertEqual(run.stdout_path, f"{run.run_dir}/stdout")

    async def test_a_runner_that_disappears_ends_the_wait(self) -> None:
        executor = self._executor(["running\n", "gone\n"])
        with mock.patch("asyncio.sleep", new=mock.AsyncMock()):
            run = await executor.run_detached("sleep 100", timeout=60.0)

        self.assertIsNone(run.exit_code)
        self.assertFalse(run.timed_out)

    async def test_a_script_past_its_timeout_is_killed_with_its_group(self) -> None:
        executor = self._executor(["running\n"])
        with mock.patch("asyncio.sleep", new=mock.AsyncMock()):
            run = await executor.run_detached("sleep 100", timeout=0.05)

        self.assertTrue(run.timed_out)
        self.assertIsNone(run.exit_code)
        kill = executor._execute_stream_control.await_args.args[0]  # type: ignore[attr-defined]
        self.assertIn("kill -KILL -- -4242", kill)

    async def test_a_script_that_cannot_start_raises(self) -> None:
        executor = self._executor(["0\n"], start_output="bash: permission denied")
        with self.assertRaisesRegex(RuntimeError, "Could not start"):
            await executor.run_detached("true", timeout=60.0)

    async def test_an_unresponsive_sandbox_raises(self) -> None:
        executor = self._executor(["0\n"])
        executor._execute_stream_control = mock.AsyncMock(  # type: ignore[method-assign]
            side_effect=TimeoutError("poll timed out")
        )
        with (
            mock.patch("asyncio.sleep", new=mock.AsyncMock()),
            self.assertRaisesRegex(RuntimeError, "stopped answering"),
        ):
            await executor.run_detached("true", timeout=60.0)
