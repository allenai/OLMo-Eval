"""Tests for the Terminal-Bench external evaluation."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from olmo_eval.common.execution import ExecutionResult
from olmo_eval.common.types.trajectory import AgentTrajectory, AgentTurn
from olmo_eval.evals.external.benchmarks.terminal_bench import eval as tb_eval
from olmo_eval.evals.external.benchmarks.terminal_bench.loader import (
    TerminalBenchLoader,
    parse_size_mb,
)
from olmo_eval.evals.external.benchmarks.terminal_bench.task import TerminalBenchTask
from olmo_eval.evals.external.benchmarks.terminal_bench.verifier import (
    REWARD_FILE_MISSING,
    REWARD_PARSE_ERROR,
    SOLUTION_DIR,
    VERIFIER_TIMEOUT,
    TerminalBenchVerifier,
)
from olmo_eval.harness.sandbox.config import SandboxMode

TASK_TOML = """\
schema_version = "1.1"

[metadata]
difficulty = "hard"
category = "software engineering"

[verifier]
timeout_sec = 600.0

[agent]
timeout_sec = 1800.0

[environment]
docker_image = "example/task:1"
cpus = 2
memory_mb = 4096
storage_mb = 10240
allow_internet = true
"""


def write_task(root: Path, name: str, toml: str = TASK_TOML, workdir: str = "/app/src") -> Path:
    task_dir = root / name
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "environment" / "Dockerfile").write_text(
        f"FROM ubuntu\nWORKDIR {workdir}\nRUN echo hi\n"
    )
    (task_dir / "task.toml").write_text(toml)
    (task_dir / "instruction.md").write_text("Make it work.")
    (task_dir / "tests").mkdir()
    (task_dir / "tests" / "test.sh").write_text("#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n")
    (task_dir / "tests" / "data").mkdir()
    (task_dir / "tests" / "data" / "fixture.bin").write_bytes(b"\x00\x01")
    (task_dir / "solution").mkdir()
    (task_dir / "solution" / "solve.sh").write_text("#!/bin/bash\npython helper.py\n")
    (task_dir / "solution" / "helper.py").write_text("print('solved')\n")
    return task_dir


def make_task(**overrides: Any) -> TerminalBenchTask:
    fields: dict[str, Any] = {
        "task_id": "demo",
        "image": "example/demo:1",
        "working_dir": "/app",
        "instruction": "Do it.",
        "agent_timeout": 900.0,
        "verifier_timeout": 600.0,
        "test_files": {"test.sh": b"echo 1 > /logs/verifier/reward.txt"},
        "solution_script": "echo done",
        "difficulty": "medium",
        "category": "scientific computing",
    }
    fields.update(overrides)
    return TerminalBenchTask(**fields)


def make_result(
    task_id: str = "t",
    reward: float = 1.0,
    attempt: int = 0,
    error: str | None = None,
    completion_reason: str = "submitted",
    difficulty: str = "easy",
    category: str = "cat",
) -> tb_eval.TaskResult:
    return tb_eval.TaskResult(
        task_id=task_id,
        reward=reward,
        trajectory=AgentTrajectory(turns=()),
        completion_reason=completion_reason,
        agent_duration=1.0,
        verification_output="",
        verification_exit_code=0,
        error=error,
        difficulty=difficulty,
        category=category,
        attempt=attempt,
    )


class TestLoader:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [(None, 7), (2048, 2048), (2048.0, 2048), ("2G", 2048), ("512M", 512), ("1.5gb", 1536)],
    )
    def test_parse_size_mb(self, value, expected) -> None:
        assert parse_size_mb(value, default=7) == expected

    def test_unreadable_sizes_fall_back_to_the_default(self) -> None:
        assert parse_size_mb("lots", default=3) == 3
        assert parse_size_mb(True, default=3) == 3

    def test_pinned_to_the_2_1_release(self) -> None:
        assert "terminal-bench-2-1" in TerminalBenchLoader.REPO_URL
        assert TerminalBenchLoader.DATASET_VERSION == "2.1"
        assert len(TerminalBenchLoader.DEFAULT_REF) == 40

    def test_tasks_are_read_from_the_tasks_directory(self, tmp_path: Path) -> None:
        write_task(tmp_path / "tasks", "alpha")
        (tmp_path / "tasks" / "dataset.toml").write_text("[dataset]\nname = 'x'\n")
        write_task(tmp_path / "tasks", "beta")

        tasks = TerminalBenchLoader().load_tasks(tmp_path)

        assert [t.task_id for t in tasks] == ["alpha", "beta"]

    def test_tasks_at_the_repository_root_still_load(self, tmp_path: Path) -> None:
        write_task(tmp_path, "alpha")

        tasks = TerminalBenchLoader().load_tasks(tmp_path, task_ids=["alpha"])

        assert [t.task_id for t in tasks] == ["alpha"]

    def test_a_task_carries_its_resources_and_files(self, tmp_path: Path) -> None:
        write_task(tmp_path / "tasks", "alpha")

        (task,) = TerminalBenchLoader().load_tasks(tmp_path)

        assert task.image == "example/task:1"
        assert task.working_dir == "/app/src"
        assert task.instruction == "Make it work."
        assert task.agent_timeout == 1800.0
        assert task.verifier_timeout == 600.0
        assert task.cpus == 2
        assert task.memory_mb == 4096
        assert task.storage_mb == 10240
        assert task.allow_internet is True
        assert task.difficulty == "hard"
        assert task.category == "software-engineering"
        assert set(task.test_files) == {"test.sh", "data/fixture.bin"}
        assert task.test_files["data/fixture.bin"] == b"\x00\x01"
        assert set(task.solution_files) == {"solve.sh", "helper.py"}
        assert task.solution_script == "#!/bin/bash\npython helper.py\n"

    def test_a_task_without_a_prebuilt_image_is_built_from_its_dockerfile(
        self, tmp_path: Path
    ) -> None:
        toml = TASK_TOML.replace('docker_image = "example/task:1"\n', "").replace(
            "cpus = 2\nmemory_mb = 4096\nstorage_mb = 10240\n", ""
        )
        write_task(tmp_path, "lite", toml=toml)

        (task,) = TerminalBenchLoader().load_tasks(tmp_path)

        assert task.image == ""
        assert task.needs_build
        assert task.build_context == str(tmp_path / "lite" / "environment")
        assert task.build_timeout == 600.0
        assert task.cpus == 1
        assert task.memory_mb == 2048

    def test_a_task_with_neither_image_nor_dockerfile_is_skipped(self, tmp_path: Path) -> None:
        toml = TASK_TOML.replace('docker_image = "example/task:1"\n', "")
        task_dir = write_task(tmp_path, "broken", toml=toml)
        (task_dir / "environment" / "Dockerfile").unlink()
        write_task(tmp_path, "fine")

        tasks = TerminalBenchLoader().load_tasks(tmp_path)

        assert [t.task_id for t in tasks] == ["fine"]

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Hard", "hard"),
            ("data_science", "data-science"),
            (" Very Hard ", "very-hard"),
            ("", "unknown"),
        ],
    )
    def test_labels_are_normalized(self, raw, expected) -> None:
        from olmo_eval.evals.external.benchmarks.terminal_bench.loader import normalize_label

        assert normalize_label(raw) == expected

    def test_loader_takes_another_dataset(self) -> None:
        loader = TerminalBenchLoader(
            repo_url="https://example/x.git", default_ref="r1", dataset_version="9"
        )
        assert loader.repo_url == "https://example/x.git"
        assert loader.default_ref == "r1"
        assert loader.dataset_version == "9"
        assert TerminalBenchLoader().repo_url == TerminalBenchLoader.REPO_URL

    def test_legacy_size_strings_are_accepted(self, tmp_path: Path) -> None:
        toml = TASK_TOML.replace("memory_mb = 4096", 'memory = "8G"').replace(
            "storage_mb = 10240", 'storage = "20G"'
        )
        write_task(tmp_path / "tasks", "alpha", toml=toml)

        (task,) = TerminalBenchLoader().load_tasks(tmp_path)

        assert task.memory_mb == 8192
        assert task.storage_mb == 20480


class TestArgs:
    def test_defaults_match_the_reference_agent(self) -> None:
        args = tb_eval.TerminalBenchArgs.from_dict({})
        assert args.scaffold == "vanillux"
        assert args.max_turns == 64
        assert args.n_attempts == 1
        assert args.command_timeout == 120.0
        assert args.max_format_errors == 64
        assert args.resource_limits is True
        assert args.sampling_params.temperature == 0.7
        assert args.sampling_params.top_p == 0.95
        assert args.sampling_params.max_tokens == 16384
        assert args.repo_ref == TerminalBenchLoader.DEFAULT_REF

    def test_values_are_read_from_strings(self) -> None:
        args = tb_eval.TerminalBenchArgs.from_dict(
            {
                "task_ids": "a, b",
                "n_attempts": "5",
                "resource_limits": "false",
                "oracle": "true",
                "temperature": "0",
                "max_turns": "10",
            }
        )
        assert args.task_ids == ["a", "b"]
        assert args.n_attempts == 5
        assert args.resource_limits is False
        assert args.oracle is True
        assert args.temperature == 0.0
        assert args.max_turns == 10

    def test_attempts_never_drop_below_one(self) -> None:
        assert tb_eval.TerminalBenchArgs.from_dict({"n_attempts": 0}).n_attempts == 1


class TestMetrics:
    @pytest.mark.parametrize(
        ("n", "c", "k", "expected"),
        [(5, 0, 1, 0.0), (5, 5, 1, 1.0), (4, 2, 1, 0.5), (4, 1, 4, 1.0), (3, 1, 2, 2 / 3)],
    )
    def test_pass_at_k(self, n, c, k, expected) -> None:
        assert tb_eval.pass_at_k(n, c, k) == pytest.approx(expected)

    def test_single_attempt_metrics(self) -> None:
        results = [
            make_result("a", 1.0, difficulty="easy", category="x"),
            make_result("b", 0.0, difficulty="hard", category="y z"),
            make_result("c", 0.0, error="reward_file_missing", completion_reason="error"),
            make_result("d", 1.0, completion_reason="timeout"),
        ]

        metrics = tb_eval.compute_metrics(results, n_attempts=1)

        assert metrics["pass_rate"] == pytest.approx(0.5)
        assert metrics["pass_rate_adjusted"] == pytest.approx(2 / 3)
        assert metrics["error_rate"] == pytest.approx(0.25)
        assert metrics["num_tasks"] == 4
        assert metrics["num_trials"] == 4
        assert metrics["num_passed"] == 2
        assert metrics["num_errors"] == 1
        assert metrics["num_agent_timeouts"] == 1
        assert metrics["pass_rate_hard"] == 0.0
        assert metrics["pass_rate_y_z"] == 0.0
        assert "pass@1" not in metrics
        assert "pass_rate_std" not in metrics

    def test_multi_attempt_metrics(self) -> None:
        results = [
            make_result("a", 1.0, attempt=0),
            make_result("a", 0.0, attempt=1),
            make_result("b", 0.0, attempt=0),
            make_result("b", 0.0, attempt=1),
        ]

        metrics = tb_eval.compute_metrics(results, n_attempts=2)

        assert metrics["pass_rate"] == pytest.approx(0.25)
        assert metrics["pass@1"] == pytest.approx(0.25)
        assert metrics["pass@2"] == pytest.approx(0.5)
        assert metrics["num_tasks"] == 2
        assert metrics["num_trials"] == 4
        # attempt means are 0.5 and 0.0
        assert metrics["pass_rate_std"] == pytest.approx(0.3535533906)
        assert metrics["pass_rate_sem"] == pytest.approx(0.25)

    def test_no_results(self) -> None:
        metrics = tb_eval.compute_metrics([], n_attempts=3)
        assert metrics["pass_rate"] == 0.0
        assert metrics["error_rate"] == 0.0
        assert metrics["num_trials"] == 0


class FakeExecutor:
    """Records commands and answers them from a table of canned results."""

    def __init__(self, responses: dict[str, ExecutionResult] | None = None) -> None:
        self.responses = responses or {}
        self.commands: list[str] = []
        self.timeouts: list[float | None] = []
        self.session_commands: list[tuple[str, float | None]] = []

    async def execute_command(self, command: str, timeout: float | None = None, **_: Any):
        self.commands.append(command)
        self.timeouts.append(timeout)
        for key, result in self.responses.items():
            if key in command:
                return result
        return ExecutionResult(success=True, output="", exit_code=0)

    async def execute_in_session(self, command: str, timeout: float | None = None, **_: Any):
        self.session_commands.append((command, timeout))
        return ExecutionResult(success=True, output="", exit_code=0)


class TestVerifier:
    def test_files_land_under_the_target_directory(self) -> None:
        executor = FakeExecutor()
        files = {"test.sh": b"echo", "data/big.bin": b"x" * 60_000}

        asyncio.run(TerminalBenchVerifier().inject_files(executor, files, "/tests"))

        assert executor.commands[0] == "mkdir -p /tests"
        assert any("> /tests/test.sh" in c for c in executor.commands)
        assert "mkdir -p /tests/data" in executor.commands
        assert any("base64 -d /tmp/_tb_chunk > /tests/data/big.bin" in c for c in executor.commands)

    def test_inject_tests_prepares_the_verifier_log_directory(self) -> None:
        executor = FakeExecutor()

        asyncio.run(TerminalBenchVerifier().inject_tests(executor, {"test.sh": b"echo"}))

        assert executor.commands[0] == "mkdir -p /logs/verifier"
        assert executor.commands[1] == "mkdir -p /tests"

    def test_a_written_reward_is_read_back(self) -> None:
        executor = FakeExecutor(
            {"reward.txt": ExecutionResult(success=True, output="1\n", exit_code=0)}
        )

        result = asyncio.run(
            TerminalBenchVerifier().run_verification(executor, 60.0, "/app/src", "demo")
        )

        assert result.reward == 1.0
        assert result.error is None
        assert executor.commands[0] == "cd /app/src && bash /tests/test.sh"

    def test_a_missing_reward_file_is_an_error_not_a_zero(self) -> None:
        executor = FakeExecutor(
            {"reward.txt": ExecutionResult(success=False, output="No such file", exit_code=1)}
        )

        result = asyncio.run(TerminalBenchVerifier().run_verification(executor, 60.0))

        assert result.reward == 0.0
        assert result.error == REWARD_FILE_MISSING

    def test_an_unreadable_reward_is_an_error(self) -> None:
        executor = FakeExecutor(
            {"reward.txt": ExecutionResult(success=True, output="yes", exit_code=0)}
        )

        result = asyncio.run(TerminalBenchVerifier().run_verification(executor, 60.0))

        assert result.error == REWARD_PARSE_ERROR

    def test_a_timed_out_test_script_is_an_error(self) -> None:
        executor = FakeExecutor(
            {
                "test.sh": ExecutionResult(
                    success=False, output="Command timed out", exit_code=-1, error="timeout"
                )
            }
        )

        result = asyncio.run(TerminalBenchVerifier().run_verification(executor, 60.0))

        assert result.error == VERIFIER_TIMEOUT
        assert not any("reward.txt" in c for c in executor.commands)


class TestSandboxConfig:
    def test_resource_flags_come_from_the_task(self) -> None:
        task = make_task(cpus=4, memory_mb=8192)
        assert tb_eval.resource_docker_args(task) == ("--cpus=4", "--memory=8192m")

    @mock.patch.object(tb_eval, "get_docker_network_args", return_value=("--net",))
    def test_docker_config_is_pristine_with_limits(self, _net) -> None:
        task = make_task(cpus=2, memory_mb=4096, working_dir="/work", agent_timeout=1200.0)
        args = tb_eval.TerminalBenchArgs.from_dict({"command_timeout": 45})

        config = tb_eval.TerminalBenchExternalEval()._sandbox_config(
            task, "podman", args, "swerex-abc:latest"
        )

        assert config.image == "swerex-abc:latest"
        assert config.mode == SandboxMode.DOCKER
        assert config.container_runtime == "podman"
        assert config.pristine_image is True
        assert config.working_dir == "/work"
        assert config.command_timeout == 45.0
        assert config.docker_args == (
            "--net",
            "--userns=auto:size=65536",
            "--cpus=2",
            "--memory=4096m",
        )

    @mock.patch.object(tb_eval, "get_docker_network_args", return_value=("--net",))
    def test_docker_runtime_gets_no_podman_flags(self, _net) -> None:
        args = tb_eval.TerminalBenchArgs.from_dict({})

        config = tb_eval.TerminalBenchExternalEval()._sandbox_config(
            make_task(), "docker", args, "img"
        )

        assert config.docker_args == ("--net", "--cpus=1", "--memory=2048m")

    @mock.patch.object(tb_eval, "get_docker_network_args", return_value=("--net",))
    def test_limits_can_be_switched_off(self, _net) -> None:
        args = tb_eval.TerminalBenchArgs.from_dict({"resource_limits": False})

        config = tb_eval.TerminalBenchExternalEval()._sandbox_config(
            make_task(), "docker", args, "img"
        )

        assert config.docker_args == ("--net",)

    def test_modal_config_has_no_docker_flags(self) -> None:
        args = tb_eval.TerminalBenchArgs.from_dict({"sandbox_mode": "modal"})

        config = tb_eval.TerminalBenchExternalEval()._sandbox_config(
            make_task(), "podman", args, "img"
        )

        assert config.mode == SandboxMode.MODAL
        assert config.docker_args == ()

    def test_unknown_sandbox_mode_is_rejected(self) -> None:
        args = tb_eval.TerminalBenchArgs.from_dict({"sandbox_mode": "cloud"})
        with pytest.raises(ValueError, match="sandbox_mode"):
            tb_eval.TerminalBenchExternalEval()._sandbox_config(make_task(), "docker", args, "img")


class TestTaskImage:
    def test_a_prebuilt_image_only_gets_the_swerex_layer(self) -> None:
        eval_obj = tb_eval.TerminalBenchExternalEval()
        with (
            mock.patch.object(tb_eval, "get_swerex_image", return_value="swerex-1") as swerex,
            mock.patch.object(tb_eval, "build_task_image") as build,
        ):
            image = asyncio.run(eval_obj._task_image(make_task(), "podman"))

        assert image == "swerex-1"
        swerex.assert_called_once_with("example/demo:1", "podman", pristine=True)
        build.assert_not_called()

    def test_a_dockerfile_task_is_built_first(self, tmp_path: Path) -> None:
        eval_obj = tb_eval.TerminalBenchExternalEval()
        task = make_task(image="", build_context=str(tmp_path), build_timeout=42.0)
        assert task.needs_build
        with (
            mock.patch.object(tb_eval, "get_swerex_image", return_value="swerex-2") as swerex,
            mock.patch.object(
                tb_eval, "build_task_image", return_value="tb-task-demo:abc"
            ) as build,
        ):
            image = asyncio.run(eval_obj._task_image(task, "docker"))

        assert image == "swerex-2"
        build.assert_called_once_with(tmp_path, "docker", "tb-task-demo", 42.0)
        swerex.assert_called_once_with("tb-task-demo:abc", "docker", pristine=True)

    def test_attempts_of_one_task_build_one_at_a_time(self, tmp_path: Path) -> None:
        eval_obj = tb_eval.TerminalBenchExternalEval()
        task = make_task(image="", build_context=str(tmp_path))
        active = 0
        peak = 0

        def slow_build(*args, **kwargs):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            import time as _time

            _time.sleep(0.05)
            active -= 1
            return "tb-task-demo:abc"

        async def run_all():
            lock = asyncio.Lock()
            return await asyncio.gather(
                *(eval_obj._task_image(task, "docker", lock) for _ in range(3))
            )

        with (
            mock.patch.object(tb_eval, "get_swerex_image", return_value="swerex-3"),
            mock.patch.object(tb_eval, "build_task_image", side_effect=slow_build) as build,
        ):
            images = asyncio.run(run_all())

        assert images == ["swerex-3"] * 3
        assert build.call_count == 3
        assert peak == 1

    def test_a_failed_build_is_an_errored_trial(self, tmp_path: Path) -> None:
        eval_obj = tb_eval.TerminalBenchExternalEval()
        task = make_task(image="", build_context=str(tmp_path))
        args = tb_eval.TerminalBenchArgs.from_dict({})

        with (
            mock.patch.object(tb_eval, "build_task_image", side_effect=RuntimeError("no base")),
            mock.patch("olmo_eval.harness.sandbox.SandboxManager") as manager,
        ):
            result = asyncio.run(
                eval_obj._execute_task(
                    task=task,
                    attempt=1,
                    provider=FakeProvider(),
                    container_runtime="docker",
                    tb_args=args,
                )
            )

        assert result.error == "no base"
        assert result.completion_reason == "error"
        assert result.attempt == 1
        manager.assert_not_called()


class TestDatasets:
    def test_tblite_is_registered_beside_terminal_bench(self) -> None:
        from olmo_eval.evals.external import get_external_eval
        from olmo_eval.evals.external.benchmarks.terminal_bench.tblite import (
            TBLITE_DATASET,
            OpenThoughtsTBLiteExternalEval,
        )

        tb = get_external_eval("terminal_bench_2")
        lite = get_external_eval("openthoughts_tblite")
        assert isinstance(lite, OpenThoughtsTBLiteExternalEval)
        assert isinstance(lite, tb_eval.TerminalBenchExternalEval)
        assert lite.name == "openthoughts_tblite"
        assert lite.dataset is TBLITE_DATASET
        assert "OpenThoughts-TBLite" in lite.description
        assert lite.dataset.repo_url.endswith("OpenThoughts-TBLite.git")
        assert len(lite.dataset.repo_ref) == 40
        assert lite.arguments["repo_ref"][1] == TBLITE_DATASET.repo_ref
        assert tb.arguments["repo_ref"][1] == TerminalBenchLoader.DEFAULT_REF
        assert lite.dataset.cache_dir != tb.dataset.cache_dir
        assert lite.dataset.image_prefix != tb.dataset.image_prefix

    def test_the_default_ref_follows_the_dataset(self) -> None:
        args = tb_eval.TerminalBenchArgs.from_dict({}, default_ref="abc123")
        assert args.repo_ref == "abc123"
        assert (
            tb_eval.TerminalBenchArgs.from_dict({"repo_ref": "x"}, default_ref="abc").repo_ref
            == "x"
        )

    def test_results_record_the_dataset(self, tmp_path: Path) -> None:
        write_task(tmp_path / "tasks", "alpha")
        from olmo_eval.evals.external.benchmarks.terminal_bench.tblite import (
            OpenThoughtsTBLiteExternalEval,
        )

        eval_obj = OpenThoughtsTBLiteExternalEval()

        async def fake_trial(task, attempt, provider, container_runtime, tb_args, image_lock=None):
            return make_result(task.task_id, 1.0, attempt=attempt)

        with mock.patch.object(eval_obj, "_execute_task", side_effect=fake_trial):
            result = asyncio.run(eval_obj.execute(FakeProvider(), {"repo_path": str(tmp_path)}))

        assert result.name == "openthoughts_tblite"
        assert result.metadata["dataset"] == "openthoughts_tblite"
        assert result.metadata["dataset_version"] == "2.0"


class FakeSandboxManager:
    instances: list[FakeSandboxManager] = []

    def __init__(self, configs, owner="default") -> None:
        self.configs = configs
        self.owner = owner
        self.executor = FakeExecutor(
            {"reward.txt": ExecutionResult(success=True, output="1", exit_code=0)}
        )
        self.started = False
        self.stopped = False
        FakeSandboxManager.instances.append(self)

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    def get_executor(self, capabilities):
        return self.executor


class FakeProvider:
    model_name = "fake-model"


def trajectory_with_steps(n: int) -> AgentTrajectory:
    return AgentTrajectory(turns=tuple(AgentTurn.assistant(content=f"step {i}") for i in range(n)))


class TestExecuteTask:
    def setup_method(self) -> None:
        FakeSandboxManager.instances.clear()

    def _run(self, eval_obj, task, args, **kwargs):
        with (
            mock.patch("olmo_eval.harness.sandbox.SandboxManager", FakeSandboxManager),
            mock.patch.object(tb_eval, "get_swerex_image", return_value="img"),
            mock.patch.object(tb_eval, "get_docker_network_args", return_value=()),
        ):
            return asyncio.run(
                eval_obj._execute_task(
                    task=task,
                    attempt=kwargs.get("attempt", 0),
                    provider=FakeProvider(),
                    container_runtime="docker",
                    tb_args=args,
                )
            )

    def test_agent_runs_then_tests_are_injected_and_verified(self) -> None:
        eval_obj = tb_eval.TerminalBenchExternalEval()
        args = tb_eval.TerminalBenchArgs.from_dict({})
        task = make_task(difficulty="hard", category="c")

        async def fake_agent(manager, task, provider, tb_args):
            return trajectory_with_steps(3), "submitted", 3, None

        with mock.patch.object(eval_obj, "_run_agent", side_effect=fake_agent):
            result = self._run(eval_obj, task, args, attempt=2)

        manager = FakeSandboxManager.instances[0]
        assert manager.owner == "tb-demo-2"
        assert manager.started and manager.stopped
        assert "cd /app && bash /tests/test.sh" in manager.executor.commands
        assert result.reward == 1.0
        assert result.error is None
        assert result.completion_reason == "submitted"
        assert result.agent_steps == 3
        assert result.attempt == 2
        assert result.native_id == "demo__2"
        assert result.difficulty == "hard"

    def test_a_slow_agent_is_cut_off_and_still_verified(self) -> None:
        eval_obj = tb_eval.TerminalBenchExternalEval()
        args = tb_eval.TerminalBenchArgs.from_dict({})
        task = make_task(agent_timeout=0.01)

        async def slow_agent(manager, task, provider, tb_args):
            await asyncio.sleep(5)
            return trajectory_with_steps(1), "submitted", 1, None

        with (
            mock.patch.object(eval_obj, "_run_agent", side_effect=slow_agent),
            mock.patch.object(tb_eval, "AGENT_TIMEOUT_GRACE", 0.0),
        ):
            result = self._run(eval_obj, task, args)

        assert result.completion_reason == "timeout"
        assert result.reward == 1.0
        assert result.error is None
        assert FakeSandboxManager.instances[0].stopped

    def test_a_model_serving_failure_is_an_error(self) -> None:
        eval_obj = tb_eval.TerminalBenchExternalEval()
        args = tb_eval.TerminalBenchArgs.from_dict({})

        async def failed_agent(manager, task, provider, tb_args):
            return AgentTrajectory(turns=()), "error", 0, "Connection error."

        with mock.patch.object(eval_obj, "_run_agent", side_effect=failed_agent):
            result = self._run(eval_obj, make_task(), args)

        assert result.completion_reason == "error"
        assert result.error == "Connection error."
        assert result.errored

    def test_an_infrastructure_failure_is_an_error(self) -> None:
        eval_obj = tb_eval.TerminalBenchExternalEval()
        args = tb_eval.TerminalBenchArgs.from_dict({})

        with mock.patch.object(eval_obj, "_run_agent", side_effect=RuntimeError("no container")):
            result = self._run(eval_obj, make_task(), args)

        assert result.reward == 0.0
        assert result.error == "no container"
        assert result.completion_reason == "error"
        assert FakeSandboxManager.instances[0].stopped

    def test_oracle_runs_the_solution_directory(self) -> None:
        eval_obj = tb_eval.TerminalBenchExternalEval()
        args = tb_eval.TerminalBenchArgs.from_dict({"oracle": True})
        task = make_task(
            working_dir="/work",
            solution_files={"solve.sh": b"python helper.py", "helper.py": b"print(1)"},
        )

        result = self._run(eval_obj, task, args)

        executor = FakeSandboxManager.instances[0].executor
        assert any(f"> {SOLUTION_DIR}/helper.py" in c for c in executor.commands)
        solve = f"cd /work && bash {SOLUTION_DIR}/solve.sh"
        assert solve in executor.commands
        assert executor.timeouts[executor.commands.index(solve)] == 900.0
        assert executor.session_commands == []
        assert result.completion_reason == "oracle"
        assert result.reward == 1.0

    def test_a_timed_out_oracle_is_still_verified(self) -> None:
        eval_obj = tb_eval.TerminalBenchExternalEval()
        args = tb_eval.TerminalBenchArgs.from_dict({"oracle": True})
        timed_out = ExecutionResult(success=False, output="", exit_code=-1, error="timeout")
        original_init = FakeSandboxManager.__init__

        def init_with_slow_solution(self, configs, owner="default"):
            original_init(self, configs, owner)
            self.executor.responses["solve.sh"] = timed_out

        with mock.patch.object(FakeSandboxManager, "__init__", init_with_slow_solution):
            result = self._run(eval_obj, make_task(), args)

        assert result.completion_reason == "timeout"
        assert result.reward == 1.0
        assert result.error is None


class TestRunAgent:
    def test_the_scaffold_gets_the_task_budget_and_sampling(self) -> None:
        eval_obj = tb_eval.TerminalBenchExternalEval()
        args = tb_eval.TerminalBenchArgs.from_dict(
            {"max_turns": 7, "command_timeout": 30, "temperature": 0.1}
        )
        task = make_task(agent_timeout=1234.0)
        scaffold = mock.Mock()
        scaffold.run = mock.AsyncMock(
            return_value=mock.Mock(
                trajectory=trajectory_with_steps(2),
                metadata={"completion_reason": "submitted", "steps": 2},
                max_turns_reached=False,
                error=None,
            )
        )

        with mock.patch("olmo_eval.harness.scaffolds.get_scaffold", return_value=scaffold):
            trajectory, reason, steps, error = asyncio.run(
                eval_obj._run_agent(mock.Mock(), task, FakeProvider(), args)
            )

        scaffold.set_sandbox_manager.assert_called_once()
        call = scaffold.run.await_args
        config = call.args[1]
        assert config.scaffold == "vanillux"
        assert config.tool_names == ("bash",)
        assert config.system_prompt is None
        assert config.max_turns == 7
        assert config.scaffold_kwargs == {
            "command_timeout": 30.0,
            "max_format_errors": 64,
            "agent_timeout": 1234.0,
        }
        assert call.args[2].messages[0]["content"] == "Do it."
        assert call.kwargs["sampling_params"].temperature == 0.1
        assert call.kwargs["trace_metadata"] == {"task_id": "demo"}
        assert "enable_compaction" not in call.kwargs
        assert reason == "submitted"
        assert steps == 2
        assert error is None
        assert trajectory.num_turns == 2

    def test_the_openai_agents_scaffold_keeps_its_own_tools_and_prompt(self) -> None:
        eval_obj = tb_eval.TerminalBenchExternalEval()
        args = tb_eval.TerminalBenchArgs.from_dict(
            {"scaffold": "openai_agents", "enable_compaction": False}
        )
        scaffold = mock.Mock()
        scaffold.run = mock.AsyncMock(
            return_value=mock.Mock(
                trajectory=trajectory_with_steps(4),
                metadata={},
                max_turns_reached=True,
                error="Max turns exceeded",
            )
        )

        with mock.patch("olmo_eval.harness.scaffolds.get_scaffold", return_value=scaffold):
            _, reason, steps, error = asyncio.run(
                eval_obj._run_agent(mock.Mock(), make_task(), FakeProvider(), args)
            )

        call = scaffold.run.await_args
        config = call.args[1]
        assert config.tool_names == ("execute_bash_session", "submit")
        assert config.system_prompt == tb_eval.OPENAI_AGENTS_SYSTEM_PROMPT
        assert call.kwargs["enable_compaction"] is False
        assert reason == "max_turns"
        assert steps == 4
        assert error is None


class TestExecute:
    def test_unsupported_scaffolds_are_refused_before_any_work(self) -> None:
        eval_obj = tb_eval.TerminalBenchExternalEval()
        with pytest.raises(ValueError, match="Unsupported scaffold"):
            asyncio.run(eval_obj.execute(FakeProvider(), {"scaffold": "openhands"}))

    def test_trials_are_run_per_attempt_and_summarized(self, tmp_path: Path) -> None:
        write_task(tmp_path / "tasks", "alpha")
        write_task(tmp_path / "tasks", "beta")
        eval_obj = tb_eval.TerminalBenchExternalEval()
        rewards = {("alpha", 0): 1.0, ("alpha", 1): 0.0, ("beta", 0): 0.0, ("beta", 1): 0.0}
        seen: list[tuple[str, int]] = []

        async def fake_trial(task, attempt, provider, container_runtime, tb_args, image_lock=None):
            seen.append((task.task_id, attempt))
            return make_result(task.task_id, rewards[(task.task_id, attempt)], attempt=attempt)

        with mock.patch.object(eval_obj, "_execute_task", side_effect=fake_trial):
            result = asyncio.run(
                eval_obj.execute(
                    FakeProvider(),
                    {"repo_path": str(tmp_path), "n_attempts": 2, "max_concurrency": 2},
                    output_dir=str(tmp_path / "out"),
                )
            )

        assert sorted(seen) == [("alpha", 0), ("alpha", 1), ("beta", 0), ("beta", 1)]
        assert result.success
        assert result.metrics["pass_rate"] == pytest.approx(0.25)
        assert result.metrics["pass@2"] == pytest.approx(0.5)
        assert result.metadata["dataset_version"] == "2.1"
        assert result.metadata["n_attempts"] == 2
        assert result.metadata["scaffold"] == "vanillux"
        assert result.predictions is not None
        assert sorted(p["native_id"] for p in result.predictions) == [
            "alpha",
            "alpha__1",
            "beta",
            "beta__1",
        ]
        saved = json.loads((tmp_path / "out" / "terminal_bench_2_tasks.json").read_text())
        assert {(row["task_id"], row["attempt"]) for row in saved} == set(rewards)
        assert len(list((tmp_path / "out" / "traces").glob("*.jsonl"))) == 4

    def test_a_trial_that_raises_becomes_an_errored_result(self, tmp_path: Path) -> None:
        write_task(tmp_path / "tasks", "alpha")
        eval_obj = tb_eval.TerminalBenchExternalEval()

        with mock.patch.object(eval_obj, "_execute_task", side_effect=RuntimeError("boom")):
            result = asyncio.run(eval_obj.execute(FakeProvider(), {"repo_path": str(tmp_path)}))

        assert result.metrics["num_errors"] == 1
        assert result.metrics["error_rate"] == 1.0
        assert result.predictions is not None
        assert result.predictions[0]["error"] == "boom"

    def test_no_tasks_is_a_failed_result(self, tmp_path: Path) -> None:
        eval_obj = tb_eval.TerminalBenchExternalEval()

        result = asyncio.run(eval_obj.execute(FakeProvider(), {"repo_path": str(tmp_path)}))

        assert result.success is False
        assert result.error == "No tasks found"
