"""Terminal-Bench task verification."""

from __future__ import annotations

import base64
import logging
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from olmo_eval.harness.sandbox.executor import SandboxExecutor

logger = logging.getLogger(__name__)

TESTS_DIR = "/tests"
SOLUTION_DIR = "/solution"
VERIFIER_LOGS_DIR = "/logs/verifier"
REWARD_FILE = f"{VERIFIER_LOGS_DIR}/reward.txt"

#: Verification failures that are the harness's fault rather than the agent's.
#: A trial with one of these is reported as an error, separately from a 0 reward.
VERIFIER_TIMEOUT = "verifier_timeout"
REWARD_FILE_MISSING = "reward_file_missing"
REWARD_PARSE_ERROR = "reward_parse_error"


@dataclass
class VerificationResult:
    """Result of task verification.

    Attributes:
        reward: 0.0 for failure, 1.0 for success.
        test_output: Output from running the test script.
        test_exit_code: Exit code from the test script.
        error: Why no reward could be read, if verification itself failed.
    """

    reward: float
    test_output: str
    test_exit_code: int
    error: str | None = None


class TerminalBenchVerifier:
    """Verifies Terminal-Bench task completion."""

    async def inject_files(
        self,
        executor: SandboxExecutor,
        files: dict[str, bytes],
        target_dir: str,
    ) -> None:
        """Write files into a directory in the container.

        Args:
            executor: The sandbox executor.
            files: Mapping of paths relative to the target directory to content.
            target_dir: Absolute directory in the container to write under.
        """
        result = await executor.execute_command(f"mkdir -p {shlex.quote(target_dir)}", timeout=30.0)
        if not result.success:
            logger.warning(f"Failed to create {target_dir}: {result.output}")

        for rel_path, content in files.items():
            target = f"{target_dir}/{rel_path}"
            parent_dir = str(Path(rel_path).parent)
            if parent_dir != ".":
                await executor.execute_command(
                    f"mkdir -p {shlex.quote(f'{target_dir}/{parent_dir}')}",
                    timeout=30.0,
                )

            # Write file via base64 (safe for both text and binary)
            b64 = base64.b64encode(content).decode()

            # Split large base64 strings into chunks to avoid command line limits
            if len(b64) > 50000:
                chunk_size = 50000
                for i in range(0, len(b64), chunk_size):
                    chunk = b64[i : i + chunk_size]
                    redirect = ">" if i == 0 else ">>"
                    await executor.execute_command(
                        f"echo -n '{chunk}' {redirect} /tmp/_tb_chunk",
                        timeout=60.0,
                    )
                await executor.execute_command(
                    f"base64 -d /tmp/_tb_chunk > {shlex.quote(target)} && rm /tmp/_tb_chunk",
                    timeout=60.0,
                )
            else:
                await executor.execute_command(
                    f"echo '{b64}' | base64 -d > {shlex.quote(target)}",
                    timeout=60.0,
                )

        logger.info(f"Injected {len(files)} files into {target_dir}")

    async def inject_tests(
        self,
        executor: SandboxExecutor,
        test_files: dict[str, bytes],
    ) -> None:
        """Inject test files into the container.

        Creates the tests and verifier log directories, then writes all test
        files under the tests directory.

        Args:
            executor: The sandbox executor.
            test_files: Mapping of relative paths to file content.
        """
        result = await executor.execute_command(f"mkdir -p {VERIFIER_LOGS_DIR}", timeout=30.0)
        if not result.success:
            logger.warning(f"Failed to create verifier log directory: {result.output}")
        await self.inject_files(executor, test_files, TESTS_DIR)

    async def run_verification(
        self,
        executor: SandboxExecutor,
        timeout: float,
        working_dir: str = "/app",
        task_id: str | None = None,
    ) -> VerificationResult:
        """Run verification tests.

        Executes the task's test script from the working directory and reads
        the reward it writes. A test script that times out, or that leaves no
        readable reward behind, yields a result with ``error`` set.

        Args:
            executor: The sandbox executor.
            timeout: Timeout for test execution in seconds.
            working_dir: Working directory for running tests.
            task_id: Task identifier for log prefix.

        Returns:
            VerificationResult with reward, output, and exit code.
        """
        log_prefix = f"{task_id}_verifier" if task_id else "terminal_bench_verifier"

        test_result = await executor.execute_command(
            f"cd {shlex.quote(working_dir)} && bash {TESTS_DIR}/test.sh",
            timeout=timeout,
            stream=True,
            log_prefix=log_prefix,
        )

        logger.info(f"Test script exit code: {test_result.exit_code}")

        if test_result.error == "timeout":
            logger.warning(f"Verifier timed out after {timeout}s")
            return VerificationResult(
                reward=0.0,
                test_output=test_result.output,
                test_exit_code=test_result.exit_code,
                error=VERIFIER_TIMEOUT,
            )

        reward_result = await executor.execute_command(f"cat {REWARD_FILE}", timeout=30.0)

        if not reward_result.success:
            logger.warning(f"Failed to read reward file: {reward_result.output[:100]}")
            return VerificationResult(
                reward=0.0,
                test_output=test_result.output,
                test_exit_code=test_result.exit_code,
                error=REWARD_FILE_MISSING,
            )

        try:
            reward = float(reward_result.output.strip())
        except ValueError:
            logger.warning(f"Failed to parse reward: {reward_result.output[:100]}")
            return VerificationResult(
                reward=0.0,
                test_output=test_result.output,
                test_exit_code=test_result.exit_code,
                error=REWARD_PARSE_ERROR,
            )

        logger.info(f"Reward: {reward}")
        return VerificationResult(
            reward=reward,
            test_output=test_result.output,
            test_exit_code=test_result.exit_code,
        )
