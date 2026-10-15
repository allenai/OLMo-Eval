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
