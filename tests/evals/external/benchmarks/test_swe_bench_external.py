"""Tests for the SWE-bench Verified external evaluation."""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from parameterized import parameterized

from olmo_eval.evals.external.benchmarks.swe_bench import eval as swe_eval
from olmo_eval.evals.external.benchmarks.swe_bench import grader as swe_grader
from olmo_eval.evals.external.benchmarks.swe_bench.loader import select_instances
from olmo_eval.evals.external.benchmarks.swe_bench.task import SWEBenchInstance
from olmo_eval.harness.sandbox.executor import ExecutionResult

EVAL_SCRIPT = """#!/bin/bash
set -uxo pipefail
cd /testbed
: '>>>>> Start Test Output'
pytest -rA tests/test_mod.py
: '>>>>> End Test Output'
"""

GOLD_PATCH = """diff --git a/mod.py b/mod.py
--- a/mod.py
+++ b/mod.py
@@ -1 +1 @@
-x = 1
+x = 2
"""


def _row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "instance_id": "owner__repo-1",
        "repo": "owner/repo",
        "base_commit": "abc123",
        "problem_statement": "Fix the bug in mod.",
        "hints_text": "Look at mod.py.",
        "image": "swebench/sweb.eval.x86_64.owner_1776_repo-1:latest",
        "eval_script": EVAL_SCRIPT,
        "log_parser": "parse_log_pytest",
        "eval_type": "pass_and_fail",
        "version": "1.0",
        "FAIL_TO_PASS": json.dumps(["tests/test_mod.py::test_fixed"]),
        "PASS_TO_PASS": json.dumps(["tests/test_mod.py::test_kept"]),
        "patch": GOLD_PATCH,
        "difficulty": "<15 min fix",
    }
    row.update(overrides)
    return row


def _instance(**overrides: Any) -> SWEBenchInstance:
    return SWEBenchInstance.from_row(_row(**overrides))


def _log(fixed: str, kept: str = "PASSED") -> str:
    return (
        "+ : '>>>>> Start Test Output'\n"
        "+ pytest -rA tests/test_mod.py\n"
        "=== short test summary info ===\n"
        f"{fixed} tests/test_mod.py::test_fixed\n"
        f"{kept} tests/test_mod.py::test_kept\n"
        "+ SWEBENCH_TEST_EXIT_CODE=0\n"
        "+ : '>>>>> End Test Output'\n"
    )


class FakeExecutor:
    """Records commands and answers them from a list of (substring, result) rules."""

    def __init__(self, rules: list[tuple[str, ExecutionResult]]) -> None:
        self.rules = rules
        self.commands: list[str] = []
        self.files: dict[str, str] = {}

    async def execute_command(self, command: str, timeout: float | None = None, **_: Any):
        self.commands.append(command)
        for needle, result in self.rules:
            if needle in command:
                return result
        return ExecutionResult(success=True, output="", exit_code=0)

    async def write_files(self, files: dict[str, str], timeout: float | None = None) -> None:
        self.files.update(files)


def _ok(output: str = "") -> ExecutionResult:
    return ExecutionResult(success=True, output=output, exit_code=0)


def _fail(output: str = "") -> ExecutionResult:
    return ExecutionResult(success=False, output=output, exit_code=1)


class TestInstance(unittest.TestCase):
    def test_from_row_parses_json_test_lists(self) -> None:
        inst = _instance()
        self.assertEqual(inst.fail_to_pass, ("tests/test_mod.py::test_fixed",))
        self.assertEqual(inst.pass_to_pass, ("tests/test_mod.py::test_kept",))
        self.assertEqual(inst.gold_patch, GOLD_PATCH)

    def test_from_row_accepts_list_test_lists(self) -> None:
        inst = _instance(FAIL_TO_PASS=["a", "b"], PASS_TO_PASS=[])
        self.assertEqual(inst.fail_to_pass, ("a", "b"))
        self.assertEqual(inst.pass_to_pass, ())

    def test_to_swebench_dict_builds_upstream_test_spec(self) -> None:
        from swebench.harness.utils import make_test_spec

        spec = make_test_spec(_instance().to_swebench_dict())
        self.assertEqual(spec.instance_id, "owner__repo-1")
        self.assertEqual(spec.FAIL_TO_PASS, ["tests/test_mod.py::test_fixed"])
        self.assertIn("pytest -rA", spec.eval_script)


class TestImageSource(unittest.TestCase):
    def test_ghcr_and_dataset_sources(self) -> None:
        inst = _instance()
        self.assertEqual(
            inst.image_for("ghcr"),
            "ghcr.io/epoch-research/swe-bench.eval.x86_64.owner__repo-1:latest",
        )
        self.assertEqual(inst.image_for("dataset"), inst.image)
        with self.assertRaisesRegex(ValueError, "Unknown image source"):
            inst.image_for("quay")

    def test_args_default_to_ghcr_with_cleanup(self) -> None:
        args = swe_eval.SWEBenchArgs.from_dict({})
        self.assertEqual(args.image_source, "ghcr")
        self.assertTrue(args.cleanup_images)
        self.assertFalse(
            swe_eval.SWEBenchArgs.from_dict({"cleanup_images": "false"}).cleanup_images
        )


class TestSelectInstances(unittest.TestCase):
    def setUp(self) -> None:
        self.instances = [
            _instance(instance_id="a__a-1", repo="a/a"),
            _instance(instance_id="b__b-1", repo="b/b"),
            _instance(instance_id="b__b-2", repo="b/b"),
        ]

    def test_filters_by_id_in_dataset_order(self) -> None:
        selected = select_instances(self.instances, instance_ids=["b__b-2", "a__a-1"])
        self.assertEqual([i.instance_id for i in selected], ["a__a-1", "b__b-2"])

    def test_filters_by_repo_and_limit(self) -> None:
        selected = select_instances(self.instances, repos=["b/b"], limit=1)
        self.assertEqual([i.instance_id for i in selected], ["b__b-1"])

    def test_unknown_id_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing__id"):
            select_instances(self.instances, instance_ids=["missing__id"])


class TestArgsAndPrompt(unittest.TestCase):
    def test_from_dict_parses_strings(self) -> None:
        args = swe_eval.SWEBenchArgs.from_dict(
            {"instance_ids": "a, b", "limit": "3", "oracle": "true", "max_concurrency": "8"}
        )
        self.assertEqual(args.instance_ids, ["a", "b"])
        self.assertEqual(args.limit, 3)
        self.assertTrue(args.oracle)
        self.assertEqual(args.max_concurrency, 8)
        self.assertFalse(args.include_hints)

    def test_other_dataset_does_not_inherit_pinned_revision(self) -> None:
        args = swe_eval.SWEBenchArgs.from_dict({"dataset": "SWE-bench/SWE-bench_Lite"})
        self.assertIsNone(args.revision)
        args = swe_eval.SWEBenchArgs.from_dict({"dataset": "x/y", "revision": "abc"})
        self.assertEqual(args.revision, "abc")

    def test_from_dict_defaults(self) -> None:
        args = swe_eval.SWEBenchArgs.from_dict({})
        self.assertIsNone(args.instance_ids)
        self.assertIsNone(args.limit)
        self.assertEqual(args.revision, swe_eval.DATASET_REVISION)

    def test_from_dict_max_tool_output_chars(self) -> None:
        self.assertEqual(swe_eval.SWEBenchArgs.from_dict({}).max_tool_output_chars, 10000)
        args = swe_eval.SWEBenchArgs.from_dict({"max_tool_output_chars": "none"})
        self.assertIsNone(args.max_tool_output_chars)

    @parameterized.expand([("vanillux", 64), ("openai_agents", 100)])
    def test_max_turns_default_depends_on_scaffold(self, scaffold: str, expected: int) -> None:
        args = swe_eval.SWEBenchArgs.from_dict({"scaffold": scaffold})
        self.assertEqual(args.max_turns, expected)
        args = swe_eval.SWEBenchArgs.from_dict({"scaffold": scaffold, "max_turns": "7"})
        self.assertEqual(args.max_turns, 7)

    def test_default_scaffold_and_sampling(self) -> None:
        args = swe_eval.SWEBenchArgs.from_dict({})
        self.assertEqual(args.scaffold, "vanillux")
        params = args.sampling_params
        self.assertEqual((params.temperature, params.top_p, params.max_tokens), (0.7, 0.95, 16384))
        args = swe_eval.SWEBenchArgs.from_dict({"temperature": "0", "max_tokens": "512"})
        self.assertEqual(args.sampling_params.temperature, 0.0)
        self.assertEqual(args.sampling_params.max_tokens, 512)

    @parameterized.expand(
        [
            ("vllm", "This model's maximum context length is 40960 tokens.", True),
            ("openai", "Error code: 400 - context_length_exceeded", True),
            ("other", "Connection reset by peer", False),
        ]
    )
    def test_is_context_length_error(self, _name: str, message: str, expected: bool) -> None:
        self.assertEqual(swe_eval.is_context_length_error(RuntimeError(message)), expected)

    @parameterized.expand([(True,), (False,)])
    def test_build_prompt_hints(self, include_hints: bool) -> None:
        prompt = swe_eval.build_prompt(_instance(), include_hints=include_hints)
        self.assertIn("Fix the bug in mod.", prompt)
        self.assertIn("/testbed", prompt)
        self.assertEqual("Look at mod.py." in prompt, include_hints)

    @parameterized.expand([(True,), (False,)])
    def test_build_task_text_is_the_bare_issue(self, include_hints: bool) -> None:
        text = swe_eval.build_task_text(_instance(), include_hints=include_hints)
        self.assertTrue(text.startswith("Fix the bug in mod."))
        self.assertNotIn("submit()", text)
        self.assertEqual("Look at mod.py." in text, include_hints)


class TestMetrics(unittest.TestCase):
    def test_compute_metrics(self) -> None:
        results = [
            swe_eval.InstanceResult(
                instance_id="a",
                repo="a/a",
                difficulty="<15 min fix",
                grade=swe_grader.GradeResult(resolved=True, patch_applied=True, tests_ran=True),
            ),
            swe_eval.InstanceResult(
                instance_id="b",
                repo="b/b",
                difficulty="<15 min fix",
                grade=swe_grader.GradeResult(empty_patch=True),
                error="agent_error: boom",
            ),
        ]
        metrics = swe_eval.compute_metrics(results)
        self.assertEqual(next(iter(metrics)), "resolve_rate")
        self.assertEqual(metrics["resolve_rate"], 0.5)
        self.assertEqual(metrics["num_resolved"], 1.0)
        self.assertEqual(metrics["patch_apply_rate"], 0.5)
        self.assertEqual(metrics["tests_ran_rate"], 0.5)
        self.assertEqual(metrics["empty_patch_rate"], 0.5)
        self.assertEqual(metrics["error_rate"], 0.5)
        self.assertEqual(metrics["resolve_rate_repo_a_a"], 1.0)
        self.assertEqual(metrics["resolve_rate_difficulty__15_min_fix"], 0.5)

    def test_compute_metrics_empty(self) -> None:
        self.assertEqual(swe_eval.compute_metrics([])["resolve_rate"], 0.0)


class TestGradeLog(unittest.TestCase):
    @parameterized.expand(
        [
            ("resolved", "PASSED", "PASSED", True),
            ("fail_to_pass_still_fails", "FAILED", "PASSED", False),
            ("pass_to_pass_regressed", "PASSED", "FAILED", False),
        ]
    )
    def test_grade_log(self, _name: str, fixed: str, kept: str, expected: bool) -> None:
        resolved, parsed, status = swe_grader.grade_log(_instance(), GOLD_PATCH, _log(fixed, kept))
        self.assertTrue(parsed)
        self.assertEqual(resolved, expected)
        self.assertIn("FAIL_TO_PASS", status)

    def test_passing_log_from_failed_test_command_is_rejected(self) -> None:
        log = _log("PASSED") + ">>>>> Test Exit Code: 1\n"
        resolved, tests_ran, _ = swe_grader.grade_log(_instance(), GOLD_PATCH, log)
        self.assertFalse(resolved)
        self.assertFalse(tests_ran)

    def test_build_eval_script_records_test_exit_code(self) -> None:
        script = swe_grader.build_eval_script(_instance())
        lines = script.splitlines()
        end = next(i for i, line in enumerate(lines) if "End Test Output" in line)
        self.assertIn("SWEBENCH_TEST_EXIT_CODE=$?", lines[end - 1])
        self.assertIn(">>>>> Test Exit Code", lines[end + 1])

    def test_grade_log_without_markers_is_unparsed(self) -> None:
        resolved, parsed, _ = swe_grader.grade_log(_instance(), GOLD_PATCH, "no output")
        self.assertFalse(resolved)
        self.assertFalse(parsed)


class TestGradeInSandbox(unittest.TestCase):
    def setUp(self) -> None:
        patcher = mock.patch.object(swe_grader, "EVAL_POLL_INTERVAL", 0.01)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_status_check_errors_do_not_abandon_the_run(self) -> None:
        checks = iter([RuntimeError("poll timed out"), _fail(""), _ok("0\n")])

        class FlakyExecutor(FakeExecutor):
            async def execute_command(self, command: str, timeout: float | None = None, **_: Any):
                if "cat /tmp/eval_exit_code" in command:
                    self.commands.append(command)
                    result = next(checks)
                    if isinstance(result, Exception):
                        raise result
                    return result
                return await super().execute_command(command, timeout)

        executor = FlakyExecutor([("cat /tmp/eval_output.log", _ok(_log("PASSED")))])
        grade = asyncio.run(
            swe_grader.grade_in_sandbox(executor, _instance(), GOLD_PATCH, timeout=60)  # type: ignore[arg-type]
        )
        self.assertTrue(grade.resolved)
        self.assertEqual(sum("cat /tmp/eval_exit_code" in c for c in executor.commands), 3)
        start = next(c for c in executor.commands if "setsid nohup" in c)
        self.assertIn("/bin/bash /eval.sh > /tmp/eval_output.log", start)

    def test_resolved_patch(self) -> None:
        executor = FakeExecutor(
            [
                ("git apply --verbose /tmp", _ok("Applied")),
                ("cat /tmp/eval_exit_code", _ok("0\n")),
                ("cat /tmp/eval_output.log", _ok(_log("PASSED"))),
            ]
        )
        grade = asyncio.run(
            swe_grader.grade_in_sandbox(executor, _instance(), GOLD_PATCH, timeout=60)  # type: ignore[arg-type]
        )
        self.assertTrue(grade.resolved)
        self.assertTrue(grade.patch_applied)
        self.assertTrue(grade.tests_ran)
        self.assertIsNone(grade.error)
        self.assertEqual(executor.files["/tmp/patch.diff"], GOLD_PATCH)
        self.assertIn("pytest -rA tests/test_mod.py", executor.files["/eval.sh"])
        self.assertIn(">>>>> Test Exit Code", executor.files["/eval.sh"])

    def test_falls_back_to_lenient_apply(self) -> None:
        executor = FakeExecutor(
            [
                ("--3way", _ok("Applied")),
                ("git apply --verbose /tmp", _fail("conflict")),
                ("cat /tmp/eval_exit_code", _ok("0\n")),
                ("cat /tmp/eval_output.log", _ok(_log("PASSED"))),
            ]
        )
        grade = asyncio.run(
            swe_grader.grade_in_sandbox(executor, _instance(), GOLD_PATCH, timeout=60)  # type: ignore[arg-type]
        )
        self.assertTrue(grade.resolved)
        self.assertTrue(any("git checkout -- ." in c for c in executor.commands))

    def test_unappliable_patch(self) -> None:
        executor = FakeExecutor(
            [
                ("git apply --check --reverse", _fail()),
                ("git apply", _fail("does not apply")),
                ("patch --batch", _fail("does not apply")),
            ]
        )
        grade = asyncio.run(
            swe_grader.grade_in_sandbox(executor, _instance(), GOLD_PATCH, timeout=60)  # type: ignore[arg-type]
        )
        self.assertFalse(grade.resolved)
        self.assertEqual(grade.error, "patch_apply_failed")
        self.assertFalse(any("/eval.sh" in c for c in executor.commands))

    def test_eval_timeout(self) -> None:
        executor = FakeExecutor([("cat /tmp/eval_exit_code", _fail("No such file"))])
        grade = asyncio.run(
            swe_grader.grade_in_sandbox(executor, _instance(), GOLD_PATCH, timeout=0.05)  # type: ignore[arg-type]
        )
        self.assertEqual(grade.error, "eval_timeout")
        self.assertTrue(any("pkill -f /eval.sh" in c for c in executor.commands))
        self.assertTrue(grade.patch_applied)
        self.assertFalse(grade.resolved)

    def test_unreadable_test_log_is_applied_but_not_run(self) -> None:
        executor = FakeExecutor(
            [
                ("cat /tmp/eval_exit_code", _ok("2\n")),
                ("cat /tmp/eval_output.log", _ok("ImportError: cannot import name")),
            ]
        )
        grade = asyncio.run(
            swe_grader.grade_in_sandbox(executor, _instance(), GOLD_PATCH, timeout=60)  # type: ignore[arg-type]
        )
        self.assertTrue(grade.patch_applied)
        self.assertFalse(grade.tests_ran)
        self.assertEqual(grade.error, "test_log_unparsed")

    def test_empty_patch_is_not_run(self) -> None:
        executor = FakeExecutor([])
        grade = asyncio.run(
            swe_grader.grade_in_sandbox(executor, _instance(), "  \n", timeout=60)  # type: ignore[arg-type]
        )
        self.assertTrue(grade.empty_patch)
        self.assertEqual(executor.commands, [])


class TestExtractPatch(unittest.TestCase):
    def test_extract_patch_diffs_against_baseline(self) -> None:
        executor = FakeExecutor([("cat /tmp/_model.patch", _ok(GOLD_PATCH.rstrip("\n")))])
        patch = asyncio.run(swe_grader.extract_patch(executor, "tree123"))  # type: ignore[arg-type]
        self.assertEqual(patch, GOLD_PATCH)
        self.assertIn("GIT_INDEX_FILE=/tmp/_extract.index", executor.commands[0])
        self.assertIn("git diff --cached --binary tree123", executor.commands[0])

    def test_snapshot_returns_tree_hash(self) -> None:
        executor = FakeExecutor([("git write-tree", _ok("warning: CRLF\n8ef914fe\n"))])
        tree = asyncio.run(swe_grader.snapshot_worktree(executor))  # type: ignore[arg-type]
        self.assertEqual(tree, "8ef914fe")
        self.assertIn("GIT_INDEX_FILE=/tmp/_snapshot.index", executor.commands[0])

    def test_snapshot_failure_raises(self) -> None:
        executor = FakeExecutor([("git write-tree", _fail("fatal: not a git repository"))])
        with self.assertRaisesRegex(RuntimeError, "not a git repository"):
            asyncio.run(swe_grader.snapshot_worktree(executor))  # type: ignore[arg-type]

    def test_extract_patch_raises_when_patch_cannot_be_read(self) -> None:
        executor = FakeExecutor([("cat /tmp/_model.patch", _fail("No such file"))])
        with self.assertRaisesRegex(RuntimeError, "No such file"):
            asyncio.run(swe_grader.extract_patch(executor, "tree123"))  # type: ignore[arg-type]

    def test_extract_patch_raises_on_git_failure(self) -> None:
        executor = FakeExecutor([("git add -A", _fail("not a git repository"))])
        with self.assertRaisesRegex(RuntimeError, "not a git repository"):
            asyncio.run(swe_grader.extract_patch(executor, "abc123"))  # type: ignore[arg-type]


class TestExecuteOracle(unittest.TestCase):
    def test_oracle_grades_gold_patch_and_writes_outputs(self) -> None:
        instances = [_instance(), _instance(instance_id="owner__repo-2", patch="")]
        graded: list[str] = []

        async def fake_grade(executor: Any, instance: SWEBenchInstance, patch: str, timeout: float):
            graded.append(instance.instance_id)
            return swe_grader.GradeResult(resolved=True, patch_applied=True)

        manager = mock.MagicMock()
        manager.start = mock.AsyncMock()
        manager.stop = mock.AsyncMock()
        provider = mock.MagicMock(model_name="test-model")

        with (
            mock.patch.object(swe_eval, "load_instances", return_value=instances),
            mock.patch.object(swe_eval, "grade_in_sandbox", side_effect=fake_grade),
            mock.patch("olmo_eval.harness.sandbox.image.remove_swerex_image") as remove,
            mock.patch(
                "olmo_eval.harness.sandbox.image.get_swerex_image", return_value="derived:latest"
            ),
            mock.patch("olmo_eval.harness.sandbox.SandboxManager", return_value=manager),
            tempfile.TemporaryDirectory() as tmp,
        ):
            result = asyncio.run(
                swe_eval.SWEBenchVerifiedExternalEval().execute(
                    provider, {"oracle": True}, output_dir=tmp, container_runtime="docker"
                )
            )
            details = json.loads((Path(tmp) / "swe_bench_verified_instances.json").read_text())
            predictions = (Path(tmp) / "swe_bench_verified_predictions.jsonl").read_text()

        self.assertTrue(result.success)
        self.assertEqual(graded, ["owner__repo-1"])
        self.assertEqual(
            sorted(c.args[0] for c in remove.call_args_list),
            [
                "ghcr.io/epoch-research/swe-bench.eval.x86_64.owner__repo-1:latest",
                "ghcr.io/epoch-research/swe-bench.eval.x86_64.owner__repo-2:latest",
            ],
        )
        self.assertEqual(result.metrics["resolve_rate"], 0.5)
        self.assertEqual(result.metrics["empty_patch_rate"], 0.5)
        self.assertEqual(len(details), 2)
        self.assertEqual(len(predictions.splitlines()), 2)
        self.assertEqual(result.predictions[0]["model_patch"], GOLD_PATCH)  # type: ignore[index]
        self.assertNotIn("agent_duration", result.predictions[0]["instance_metrics"])  # type: ignore[index]
        self.assertEqual(result.metadata["swebench_version"], "5.0.2")
        self.assertTrue(result.metadata["oracle"])

    def test_run_where_every_instance_errors_is_not_a_success(self) -> None:
        provider = mock.MagicMock(model_name="test-model")
        with (
            mock.patch.object(swe_eval, "load_instances", return_value=[_instance()]),
            mock.patch("olmo_eval.harness.sandbox.image.remove_swerex_image"),
            mock.patch(
                "olmo_eval.harness.sandbox.image.get_swerex_image",
                side_effect=RuntimeError("toomanyrequests"),
            ),
        ):
            result = asyncio.run(
                swe_eval.SWEBenchVerifiedExternalEval().execute(provider, {"oracle": True})
            )
        self.assertFalse(result.success)
        self.assertEqual(result.error, "All 1 instances failed")
        self.assertEqual(result.metrics["error_rate"], 1.0)

    def test_unknown_image_source_is_error(self) -> None:
        provider = mock.MagicMock(model_name="test-model")
        with mock.patch.object(swe_eval, "load_instances", return_value=[_instance()]):
            result = asyncio.run(
                swe_eval.SWEBenchVerifiedExternalEval().execute(
                    provider, {"oracle": True, "image_source": "quay"}
                )
            )
        self.assertFalse(result.success)
        self.assertIn("quay", result.error or "")

    def test_no_instances_is_error(self) -> None:
        provider = mock.MagicMock(model_name="test-model")
        with mock.patch.object(swe_eval, "load_instances", return_value=[]):
            result = asyncio.run(
                swe_eval.SWEBenchVerifiedExternalEval().execute(provider, {"oracle": True})
            )
        self.assertFalse(result.success)


class TestRunInstanceAgent(unittest.TestCase):
    def _run(self, agent_error: Exception) -> swe_eval.InstanceResult:
        manager = mock.MagicMock()
        manager.start = mock.AsyncMock()
        manager.stop = mock.AsyncMock()
        grade = swe_grader.GradeResult(resolved=False, patch_applied=True)
        with (
            mock.patch("olmo_eval.harness.sandbox.image.remove_swerex_image"),
            mock.patch(
                "olmo_eval.harness.sandbox.image.get_swerex_image", return_value="derived:latest"
            ),
            mock.patch("olmo_eval.harness.sandbox.SandboxManager", return_value=manager),
            mock.patch.object(
                swe_eval.SWEBenchVerifiedExternalEval,
                "_run_agent",
                mock.AsyncMock(side_effect=agent_error),
            ),
            mock.patch.object(swe_eval, "snapshot_worktree", mock.AsyncMock(return_value="tree")),
            mock.patch.object(swe_eval, "extract_patch", mock.AsyncMock(return_value=GOLD_PATCH)),
            mock.patch.object(swe_eval, "grade_in_sandbox", mock.AsyncMock(return_value=grade)),
        ):
            return asyncio.run(
                swe_eval.SWEBenchVerifiedExternalEval()._run_instance(
                    _instance(),
                    mock.MagicMock(),
                    swe_eval.SWEBenchArgs(),
                    "docker",
                )
            )

    def test_agent_timeout_still_grades_partial_patch(self) -> None:
        result = self._run(TimeoutError())
        self.assertEqual(result.completion_reason, "agent_timeout")
        self.assertIsNone(result.error)
        self.assertEqual(result.patch, GOLD_PATCH)

    def test_context_overflow_still_grades_partial_patch(self) -> None:
        result = self._run(RuntimeError("maximum context length is 40960 tokens"))
        self.assertEqual(result.completion_reason, "context_exceeded")
        self.assertIsNone(result.error)
        self.assertEqual(result.patch, GOLD_PATCH)
        self.assertTrue(result.grade.patch_applied)

    def test_other_agent_failure_is_recorded_as_error(self) -> None:
        result = self._run(RuntimeError("connection reset"))
        self.assertEqual(result.error, "agent_error: connection reset")
        self.assertTrue(result.grade.patch_applied)


def _oracle_patches(graded: list[str]) -> list[Any]:
    """Patches that run oracle-mode instances without containers, recording graded IDs."""

    async def fake_grade(executor: Any, instance: SWEBenchInstance, patch: str, timeout: float):
        graded.append(instance.instance_id)
        return swe_grader.GradeResult(resolved=True, patch_applied=True, tests_ran=True)

    manager = mock.MagicMock()
    manager.start = mock.AsyncMock()
    manager.stop = mock.AsyncMock()
    return [
        mock.patch.object(swe_eval, "grade_in_sandbox", side_effect=fake_grade),
        mock.patch("olmo_eval.harness.sandbox.image.remove_swerex_image"),
        mock.patch(
            "olmo_eval.harness.sandbox.image.get_swerex_image", return_value="derived:latest"
        ),
        mock.patch("olmo_eval.harness.sandbox.SandboxManager", return_value=manager),
    ]


class TestProgressAndResume(unittest.TestCase):
    def setUp(self) -> None:
        self.instances = [_instance(instance_id=f"owner__repo-{i}") for i in range(3)]
        self.provider = mock.MagicMock(model_name="test-model")

    def _execute(self, args: dict[str, Any], output_dir: str, graded: list[str]):
        patches = [
            mock.patch.object(swe_eval, "load_instances", return_value=self.instances),
            *_oracle_patches(graded),
        ]
        for p in patches:
            p.start()
        try:
            return asyncio.run(
                swe_eval.SWEBenchVerifiedExternalEval().execute(
                    self.provider, {"oracle": True, **args}, output_dir=output_dir
                )
            )
        finally:
            for p in patches:
                p.stop()

    def test_record_round_trip(self) -> None:
        result = swe_eval.InstanceResult(
            instance_id="a",
            repo="a/a",
            difficulty="<15 min fix",
            patch=GOLD_PATCH,
            grade=swe_grader.GradeResult(resolved=True, patch_applied=True, tests_ran=True),
            completion_reason="submitted",
            agent_duration=1.5,
            error=None,
        )
        restored = swe_eval.InstanceResult.from_record(json.loads(json.dumps(result.to_record())))
        self.assertEqual(restored, result)

    def test_each_instance_is_recorded_as_it_finishes(self) -> None:
        graded: list[str] = []
        with tempfile.TemporaryDirectory() as tmp:
            self._execute({}, tmp, graded)
            lines = (Path(tmp) / "swe_bench_verified_progress.jsonl").read_text().splitlines()
        self.assertEqual(
            sorted(json.loads(line)["instance_id"] for line in lines),
            ["owner__repo-0", "owner__repo-1", "owner__repo-2"],
        )

    def test_resume_skips_finished_instances_and_keeps_their_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            earlier = Path(tmp) / "earlier.jsonl"
            done = swe_eval.InstanceResult(
                instance_id="owner__repo-1",
                repo="owner/repo",
                difficulty="<15 min fix",
                patch=GOLD_PATCH,
                grade=swe_grader.GradeResult(resolved=False, patch_applied=True, tests_ran=True),
                completion_reason="submitted",
            )
            # The truncated second line mimics a job killed mid-write.
            earlier.write_text(json.dumps(done.to_record()) + '\n{"instance_id": "owner__re')
            graded: list[str] = []
            out = Path(tmp) / "out"
            result = self._execute({"resume_from": str(earlier)}, str(out), graded)
            progress = (out / "swe_bench_verified_progress.jsonl").read_text().splitlines()

        self.assertEqual(sorted(graded), ["owner__repo-0", "owner__repo-2"])
        self.assertEqual(result.metrics["num_instances"], 3.0)
        self.assertEqual(result.metrics["num_resolved"], 2.0)
        self.assertEqual(len(progress), 3)


class TestImageTimeout(unittest.TestCase):
    def test_image_timeout_is_an_unresolved_error_and_skips_the_agent(self) -> None:
        run_agent = mock.AsyncMock()
        with (
            mock.patch("olmo_eval.harness.sandbox.image.remove_swerex_image") as remove,
            mock.patch(
                "olmo_eval.harness.sandbox.image.get_swerex_image",
                side_effect=TimeoutError("Resolving swerex image exceeded 900s"),
            ) as get_image,
            mock.patch.object(swe_eval.SWEBenchVerifiedExternalEval, "_run_agent", run_agent),
        ):
            result = asyncio.run(
                swe_eval.SWEBenchVerifiedExternalEval()._run_instance(
                    _instance(), mock.MagicMock(), swe_eval.SWEBenchArgs(), "docker"
                )
            )
        self.assertEqual(result.completion_reason, "image_timeout")
        self.assertTrue((result.error or "").startswith("image_timeout"))
        self.assertFalse(result.resolved)
        run_agent.assert_not_awaited()
        self.assertEqual(get_image.call_args.kwargs["timeout"], 900.0)
        remove.assert_called_once()


class TestSubmit(unittest.TestCase):
    def _trajectory(self, *names: str) -> Any:
        from olmo_eval.common.types.tools import Function, ToolCall
        from olmo_eval.common.types.trajectory import AgentTrajectory, AgentTurn

        turns = tuple(
            AgentTurn(
                role="assistant",
                tool_calls=(ToolCall(id=n, function=Function(name=n, arguments="{}")),),
            )
            for n in names
        )
        return AgentTrajectory(turns=turns)

    def test_called_tool(self) -> None:
        self.assertTrue(
            swe_eval.called_tool(self._trajectory("execute_bash_session", "submit"), "submit")
        )
        self.assertFalse(swe_eval.called_tool(self._trajectory("execute_bash_session"), "submit"))

    @parameterized.expand([(True, ("submit",)), (False, None)])
    def test_stop_on_submit_reaches_the_scaffold(self, stop: bool, expected: Any) -> None:
        from olmo_eval.common.types import LMOutput
        from olmo_eval.harness.result import HarnessResult

        scaffold = mock.MagicMock()
        scaffold.run = mock.AsyncMock(
            return_value=HarnessResult(
                final_output=LMOutput(text="Task submitted successfully."),
                trajectory=self._trajectory("submit"),
                max_turns_reached=False,
            )
        )
        executor = mock.MagicMock()
        executor.execute_in_session = mock.AsyncMock(return_value=ExecutionResult(success=True))
        manager = mock.MagicMock()
        manager.get_executor.return_value = executor
        result = swe_eval.InstanceResult(instance_id="a", repo="a/a", difficulty="x")
        with mock.patch("olmo_eval.harness.scaffolds.get_scaffold", return_value=scaffold):
            asyncio.run(
                swe_eval.SWEBenchVerifiedExternalEval()._run_agent(
                    manager,
                    _instance(),
                    mock.MagicMock(),
                    swe_eval.SWEBenchArgs(scaffold="openai_agents", stop_on_submit=stop),
                    result,
                )
            )
        self.assertEqual(scaffold.run.call_args.kwargs["stop_at_tools"], expected)
        self.assertEqual(result.completion_reason, "submitted")


class TestVanilluxAgent(unittest.TestCase):
    def _run_agent(
        self, metadata: dict[str, Any], error: str | None = None, **args: Any
    ) -> tuple[mock.MagicMock, swe_eval.InstanceResult]:
        from olmo_eval.common.types import LMOutput
        from olmo_eval.harness.result import HarnessResult

        scaffold = mock.MagicMock()
        scaffold.run = mock.AsyncMock(
            return_value=HarnessResult(
                final_output=LMOutput(text=""),
                max_turns_reached=metadata.get("completion_reason") == "max_steps",
                error=error,
                metadata=metadata,
            )
        )
        executor = mock.MagicMock()
        executor.execute_in_session = mock.AsyncMock(return_value=ExecutionResult(success=True))
        manager = mock.MagicMock()
        manager.get_executor.return_value = executor
        result = swe_eval.InstanceResult(instance_id="a", repo="a/a", difficulty="x")
        with mock.patch("olmo_eval.harness.scaffolds.get_scaffold", return_value=scaffold):
            asyncio.run(
                swe_eval.SWEBenchVerifiedExternalEval()._run_agent(
                    manager, _instance(), mock.MagicMock(), swe_eval.SWEBenchArgs(**args), result
                )
            )
        return scaffold, result

    def test_passes_bash_tool_task_text_and_sampling(self) -> None:
        scaffold, _ = self._run_agent(
            {"completion_reason": "submitted"}, agent_timeout=900.0, temperature=0.5
        )
        provider, config, request = scaffold.run.call_args.args
        kwargs = scaffold.run.call_args.kwargs
        self.assertEqual([t.name for t in config.tools], ["bash"])
        self.assertIsNone(config.system_prompt)
        self.assertEqual(config.max_turns, 64)
        self.assertEqual(config.scaffold_kwargs["agent_timeout"], 900.0)
        self.assertEqual(config.scaffold_kwargs["max_format_errors"], 64)
        self.assertEqual(request.messages[0]["content"], "Fix the bug in mod.")
        self.assertEqual(kwargs["sampling_params"].temperature, 0.5)
        self.assertNotIn("stop_at_tools", kwargs)
        self.assertNotIn("enable_compaction", kwargs)

    def test_openai_agents_keeps_its_prompt_and_server_sampling(self) -> None:
        scaffold, _ = self._run_agent({}, scaffold="openai_agents")
        _, config, request = scaffold.run.call_args.args
        self.assertEqual([t.name for t in config.tools], ["execute_bash_session", "submit"])
        self.assertEqual(config.system_prompt, swe_eval.SYSTEM_PROMPT)
        self.assertIn("<issue_description>", request.messages[0]["content"])
        self.assertIsNone(scaffold.run.call_args.kwargs["sampling_params"])

    @parameterized.expand(
        [
            ("submitted", "submitted"),
            ("max_steps", "max_turns"),
            ("timeout", "agent_timeout"),
            ("context_overflow", "context_exceeded"),
            ("format_errors", "format_errors"),
        ]
    )
    def test_completion_reason(self, reported: str, expected: str) -> None:
        _, result = self._run_agent({"completion_reason": reported})
        self.assertEqual(result.completion_reason, expected)
        self.assertIsNone(result.error)

    def test_provider_failure_is_an_error(self) -> None:
        _, result = self._run_agent({"completion_reason": "error"}, error="server down")
        self.assertEqual(result.error, "agent_error: server down")

    def test_unsupported_scaffold_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported scaffold"):
            asyncio.run(
                swe_eval.SWEBenchVerifiedExternalEval().execute(
                    mock.MagicMock(), {"scaffold": "react"}
                )
            )

    @parameterized.expand([("vanillux", "submitted"), ("openai_agents", "agent_timeout")])
    def test_vanillux_gets_a_grace_period_past_the_agent_timeout(
        self, scaffold: str, expected: str
    ) -> None:
        async def slow_agent(*_args: Any) -> None:
            await asyncio.sleep(0.3)
            _args[-1].completion_reason = "submitted"

        manager = mock.MagicMock()
        manager.start = mock.AsyncMock()
        manager.stop = mock.AsyncMock()
        with (
            mock.patch.object(swe_eval, "AGENT_TIMEOUT_GRACE", 5.0),
            mock.patch("olmo_eval.harness.sandbox.image.remove_swerex_image"),
            mock.patch(
                "olmo_eval.harness.sandbox.image.get_swerex_image", return_value="derived:latest"
            ),
            mock.patch("olmo_eval.harness.sandbox.SandboxManager", return_value=manager),
            mock.patch.object(swe_eval.SWEBenchVerifiedExternalEval, "_run_agent", slow_agent),
            mock.patch.object(swe_eval, "snapshot_worktree", mock.AsyncMock(return_value="tree")),
            mock.patch.object(swe_eval, "extract_patch", mock.AsyncMock(return_value="")),
        ):
            result = asyncio.run(
                swe_eval.SWEBenchVerifiedExternalEval()._run_instance(
                    _instance(),
                    mock.MagicMock(),
                    swe_eval.SWEBenchArgs(scaffold=scaffold, agent_timeout=0.05),
                    "docker",
                )
            )
        self.assertEqual(result.completion_reason, expected)


class TestRegistration(unittest.TestCase):
    def test_registered_with_swebench_extra(self) -> None:
        from olmo_eval.evals.external import get_external_eval

        ev = get_external_eval("swe_bench_verified")
        self.assertEqual(ev.extras, ("swebench",))
        self.assertEqual(ev.scaffold, "vanillux")
        self.assertIn("instance_ids", ev.arguments)


if __name__ == "__main__":
    unittest.main()
