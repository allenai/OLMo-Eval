"""Tests for the LiveCodeBench task, its prompts, and its grader."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from typing import Any
from unittest import mock

import pytest

from olmo_eval.common.execution import ExecutionResult
from olmo_eval.common.scorers import SandboxRequiredError, ScoringIncompleteError
from olmo_eval.common.scorers.code_execution.scripts import get_script
from olmo_eval.common.types import Instance, LMOutput, RequestType
from olmo_eval.data import DataSource
from olmo_eval.evals.tasks.common import get_task
from olmo_eval.evals.tasks.livecodebench import (
    LIVECODEBENCH_REPO,
    LIVECODEBENCH_REVISION,
    RELEASE_V3_FILES,
    RELEASE_V4_V6_FILES,
    SYSTEM_PROMPT,
    VERDICT_PREFIX,
    LiveCodeBenchScorer,
    _open_test_case_rows,
)

STDIN_DOC = {
    "question_id": "abc123_a",
    "question_content": "Print the sum of two integers.",
    "starter_code": "",
    "metadata": "{}",
    "platform": "atcoder",
    "difficulty": "easy",
    "contest_date": "2024-01-01T00:00:00",
    "public_test_cases": "[]",
    "private_test_cases": "[]",
}

STARTER_DOC = {
    **STDIN_DOC,
    "question_id": "2727",
    "starter_code": (
        "class Solution:\n    def countSeniors(self, details: List[str]) -> int:\n        "
    ),
    "metadata": json.dumps({"func_name": "countSeniors"}),
    "platform": "leetcode",
}


# ---------------------------------------------------------------------------
# Instances and prompts
# ---------------------------------------------------------------------------


def test_stdin_problem_asks_for_a_whole_program() -> None:
    task = get_task("livecodebench")
    instance = task.process_doc(STDIN_DOC, index=7)

    assert instance.question == "Print the sum of two integers."
    assert instance.metadata["id"] == "abc123_a"
    assert instance.metadata["fn_name"] is None
    assert "reads the inputs" in instance.metadata["format_instruction"]


def test_starter_code_problem_carries_its_function_name() -> None:
    task = get_task("livecodebench")
    instance = task.process_doc(STARTER_DOC, index=0)

    assert instance.metadata["fn_name"] == "countSeniors"
    assert "starter code" in instance.metadata["format_instruction"]
    assert "def countSeniors" in instance.metadata["format_instruction"]


def test_test_cases_are_referenced_not_carried() -> None:
    # Payloads reach tens of megabytes per problem, and instance metadata is
    # written into the run's request records.
    task = get_task("livecodebench")
    instance = task.process_doc(STDIN_DOC, index=7)

    assert instance.metadata["row"] == 7
    assert instance.metadata["test_files"] == RELEASE_V3_FILES
    assert "public_test_cases" not in instance.metadata
    assert "private_test_cases" not in instance.metadata


def test_default_prompt_asks_for_concise_reasoning_without_naming_tags() -> None:
    # A template that opens a thinking block and one that does not both get
    # the same instruction; literal tags would suit only one of them.
    task = get_task("livecodebench")
    request = task.format_request(task.process_doc(STDIN_DOC))

    assert request.request_type == RequestType.CHAT
    assert request.messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    user = request.messages[1]["content"]
    assert user.startswith(
        "### Question:\nPrint the sum of two integers.\n\n"
        "### Format:\nProvide CONCISE reasoning on how to arrive at the answer.\n"
    )
    assert "<think>" not in user
    assert user.endswith("### Answer: (use the provided format with backticks)\n\n")


def test_think_tags_variant_asks_for_reasoning_in_tags() -> None:
    task = get_task("livecodebench:think_tags")
    user = task.format_request(task.process_doc(STDIN_DOC)).messages[1]["content"]

    assert (
        "Provide CONCISE reasoning on how to arrive at the answer in the <think> </think> tag."
        in user
    )


def test_paper_variant_sends_no_system_message() -> None:
    task = get_task("livecodebench:paper")
    request = task.format_request(task.process_doc(STDIN_DOC))

    assert request.system_prompt is None
    assert [message["role"] for message in request.messages] == ["user"]
    user = request.messages[0]["content"]
    assert "Provide CONCISE reasoning on how to arrive at the answer.\n" in user
    assert "<think>" not in user


def test_tulu_variant_drops_the_reasoning_line() -> None:
    task = get_task("livecodebench:tulu")
    user = task.format_request(task.process_doc(STDIN_DOC)).messages[1]["content"]

    assert "<think>" not in user
    assert user.startswith(
        "### Question:\nPrint the sum of two integers.\n\n### Format:\nRead the inputs from stdin"
    )


def test_answer_extraction_takes_the_last_code_block() -> None:
    # A reasoning model drafts code as it thinks; the answer is the last block.
    task = get_task("livecodebench")
    output = LMOutput(
        text=(
            "Let me try:\n```python\nprint(0)  # draft\n```\n"
            "That is wrong. Actually:\n```python\nprint(1)\n```"
        )
    )

    assert task.extract_answer(output) == "print(1)"


def test_answer_extraction_survives_an_unopened_think_tag() -> None:
    # A chat template may supply the opening tag, so the reply closes one it
    # never opened. Extraction must not depend on seeing a matched pair.
    task = get_task("livecodebench")
    output = LMOutput(text="reasoning...</think>\n```python\nprint(1)\n```")

    assert task.extract_answer(output) == "print(1)"


def test_answer_extraction_without_a_complete_block_yields_nothing() -> None:
    task = get_task("livecodebench")
    assert task.extract_answer(LMOutput(text="no code here at all")) is None
    assert task.extract_answer(LMOutput(text="```python\nprint(1)")) is None


# ---------------------------------------------------------------------------
# Releases and variants
# ---------------------------------------------------------------------------


def test_releases_select_different_contest_windows() -> None:
    assert get_task("livecodebench").config.data_source.data_files == RELEASE_V3_FILES
    assert get_task("livecodebench_hidden").config.data_source.data_files == RELEASE_V4_V6_FILES
    assert not set(RELEASE_V3_FILES) & set(RELEASE_V4_V6_FILES)


def test_default_regime_samples_ten_completions() -> None:
    config = get_task("livecodebench").config

    assert config.sampling_params is not None
    assert config.sampling_params.num_samples == 10
    assert config.sampling_params.temperature == pytest.approx(0.6)
    assert config.sampling_params.top_p == pytest.approx(0.95)
    # Generation is bounded by the model's context, not a fixed budget.
    assert config.sampling_params.max_tokens is None
    assert [metric.name for metric in config.metrics] == ["pass_at_1", "pass_at_5", "pass_at_10"]
    assert config.primary_metric is not None
    assert config.primary_metric.name == "pass_at_1"


def test_lite_variant_scores_a_single_sample() -> None:
    config = get_task("livecodebench:lite").config

    assert config.sampling_params is not None
    assert config.sampling_params.num_samples == 1
    assert config.sampling_params.temperature == pytest.approx(0.6)
    assert [metric.name for metric in config.metrics] == ["pass_at_1"]


def test_grpo_variant_reports_pass_at_10() -> None:
    config = get_task("livecodebench:grpo").config

    assert config.sampling_params is not None
    assert config.sampling_params.temperature == pytest.approx(1.0)
    assert config.sampling_params.top_p == pytest.approx(1.0)
    assert config.sampling_params.max_tokens == 16384
    assert config.primary_metric is not None
    assert config.primary_metric.name == "pass_at_10"


def test_variants_are_registered_for_both_releases() -> None:
    for variant in ("think_tags", "paper", "tulu", "lite", "grpo"):
        assert get_task(f"livecodebench_hidden:{variant}") is not None


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


class _StubStagingEnv:
    """Execution environment that records staged files and returns a verdict."""

    def __init__(
        self, output: str = '{"passed": true, "num_tests": 3}', success: bool = True
    ) -> None:
        self.output = VERDICT_PREFIX + output if output.startswith("{") else output
        self.success = success
        self.staged: dict[str, str] = {}
        self.commands: list[str] = []

    @property
    def is_running(self) -> bool:
        return True

    async def execute(self, command: str, timeout: float | None = None) -> str:
        return ""

    async def execute_command(self, command: str, timeout: float | None = None) -> ExecutionResult:
        return ExecutionResult(success=self.success, output=self.output)

    async def execute_code(
        self, code: str, language: str = "python", timeout: float | None = None
    ) -> ExecutionResult:
        return ExecutionResult(success=self.success, output=self.output)

    async def execute_with_files(
        self, command: str, files: Mapping[str, str], timeout: float | None = None
    ) -> ExecutionResult:
        self.commands.append(command)
        self.staged.update(files)
        return ExecutionResult(success=self.success, output=self.output)


class _NonStagingEnv:
    """Execution environment that cannot stage files."""

    @property
    def is_running(self) -> bool:
        return True

    async def execute(self, command: str, timeout: float | None = None) -> str:
        return ""

    async def execute_command(self, command: str, timeout: float | None = None) -> ExecutionResult:
        return ExecutionResult(success=True)

    async def execute_code(
        self, code: str, language: str = "python", timeout: float | None = None
    ) -> ExecutionResult:
        return ExecutionResult(success=True)


def _instance() -> Instance:
    return Instance(
        question="Print the sum.",
        metadata={
            "id": "abc123_a",
            "row": 3,
            "test_repo": "org/repo",
            "test_files": RELEASE_V3_FILES,
            "fn_name": None,
        },
    )


@pytest.fixture
def stub_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        {"question_id": f"q{index}", "public_test_cases": "[]", "private_test_cases": ""}
        for index in range(4)
    ]
    rows[3] = {
        "question_id": "abc123_a",
        "public_test_cases": '[{"input": "1", "output": "1"}]',
        "private_test_cases": "",
    }

    def fake_rows(repo: str, files: tuple[str, ...], revision: str | None = None) -> Any:
        del repo, files, revision
        return rows

    monkeypatch.setattr(
        "olmo_eval.evals.tasks.livecodebench._test_case_rows",
        fake_rows,
    )


@pytest.mark.anyio
class TestLiveCodeBenchScorer:
    async def test_stages_grader_problem_and_solution(self, stub_rows: None) -> None:
        scorer = LiveCodeBenchScorer()
        env = _StubStagingEnv()
        output = LMOutput(text="unused")
        output.extracted_answer = "print(1)"

        score = await scorer.ascore(_instance(), output, env)

        assert score == 1.0
        staged = sorted(os.path.basename(path) for path in env.staged)
        assert staged == ["grade.py", "problem.json", "solution.py"]

        work_dirs = {os.path.dirname(path) for path in env.staged}
        assert len(work_dirs) == 1, "all files belong to one working directory"
        work_dir = work_dirs.pop()
        assert work_dir in env.commands[0]

        problem = json.loads(env.staged[f"{work_dir}/problem.json"])
        assert problem["public_test_cases"] == '[{"input": "1", "output": "1"}]'
        assert problem["timeout"] == scorer.timeout
        assert env.staged[f"{work_dir}/solution.py"] == "print(1)"

    async def test_working_directory_is_cleaned_up(self, stub_rows: None) -> None:
        scorer = LiveCodeBenchScorer()
        env = _StubStagingEnv()
        output = LMOutput(text="unused")
        output.extracted_answer = "print(1)"

        await scorer.ascore(_instance(), output, env)

        assert "rm -rf" in env.commands[0]

    async def test_failing_verdict_scores_zero(self, stub_rows: None) -> None:
        scorer = LiveCodeBenchScorer()
        env = _StubStagingEnv(
            output='{"passed": false, "num_tests": 3, "error_code": -2, '
            '"error_message": "Wrong Answer"}'
        )
        output = LMOutput(text="unused")
        output.extracted_answer = "print(2)"

        score = await scorer.ascore(_instance(), output, env)

        assert score == 0.0
        assert output.metadata["execution_result"]["error"] == "Wrong Answer"

    async def test_missing_verdict_is_a_grading_failure(self, stub_rows: None) -> None:
        # A grader that never reported (no python3, a killed container) must
        # not turn into a plausible score of zero.
        scorer = LiveCodeBenchScorer()
        env = _StubStagingEnv(output="bash: python3: command not found", success=False)
        output = LMOutput(text="unused")
        output.extracted_answer = "print(1)"

        with pytest.raises(ScoringIncompleteError, match="No grader verdict"):
            await scorer.ascore(_instance(), output, env)
        assert output.metadata["execution_result"]["success"] is False

    async def test_verdict_is_found_among_solution_noise(self, stub_rows: None) -> None:
        # The sandbox appends stderr after stdout, so a solution that prints a
        # JSON object to stderr lands after the real verdict.
        scorer = LiveCodeBenchScorer()
        verdict = VERDICT_PREFIX + '{"passed": true, "num_tests": 3}'
        env = _StubStagingEnv(output=f'debug\n{verdict}\n{{"passed": false}}\n')
        output = LMOutput(text="unused")
        output.extracted_answer = "print(1)"

        assert await scorer.ascore(_instance(), output, env) == 1.0
        assert output.metadata["execution_result"]["num_tests"] == 3

    async def test_verdict_with_no_tests_is_a_grading_failure(self, stub_rows: None) -> None:
        # Passing every test in an empty set is what an undecodable payload
        # looks like; it must not count as a pass.
        scorer = LiveCodeBenchScorer()
        env = _StubStagingEnv(output='{"passed": true, "num_tests": 0}')
        output = LMOutput(text="unused")
        output.extracted_answer = "print(1)"

        with pytest.raises(ScoringIncompleteError, match="No test cases were graded"):
            await scorer.ascore(_instance(), output, env)

    async def test_grader_error_is_a_grading_failure(self, stub_rows: None) -> None:
        scorer = LiveCodeBenchScorer()
        env = _StubStagingEnv(
            output='{"passed": false, "num_tests": 3, "error_code": -5, '
            '"error_message": "TestRunnerError: KeyError(\'input\')"}'
        )
        output = LMOutput(text="unused")
        output.extracted_answer = "print(1)"

        with pytest.raises(ScoringIncompleteError, match="KeyError"):
            await scorer.ascore(_instance(), output, env)

    async def test_grader_runs_under_a_time_limit_that_leaves_cleanup_to_the_shell(
        self, stub_rows: None
    ) -> None:
        scorer = LiveCodeBenchScorer(overall_timeout=120.0)
        env = _StubStagingEnv()
        output = LMOutput(text="unused")
        output.extracted_answer = "print(1)"

        await scorer.ascore(_instance(), output, env)

        command = env.commands[0]
        assert "timeout 120 python3 grade.py" in command
        # Directories left by runs the sandbox had to kill are swept, but
        # only ones too old to belong to a run still in flight.
        assert command.index("find /tmp") < command.index("python3 grade.py")
        assert "-name 'lcb-*' -mmin +4 " in command

    async def test_missing_answer_scores_zero_without_executing(self) -> None:
        scorer = LiveCodeBenchScorer()
        env = _StubStagingEnv()
        output = LMOutput(text="no code here")

        assert await scorer.ascore(_instance(), output, env) == 0.0
        assert env.commands == []

    async def test_environment_without_staging_is_rejected(self, stub_rows: None) -> None:
        scorer = LiveCodeBenchScorer()
        output = LMOutput(text="unused")
        output.extracted_answer = "print(1)"

        with pytest.raises(SandboxRequiredError):
            await scorer.ascore(_instance(), output, _NonStagingEnv())

    async def test_row_pointing_at_another_problem_is_rejected(self, stub_rows: None) -> None:
        # Silent misalignment would grade every solution against the wrong tests.
        scorer = LiveCodeBenchScorer()
        instance = _instance()
        instance.metadata["row"] = 1
        output = LMOutput(text="unused")
        output.extracted_answer = "print(1)"

        with pytest.raises(ScoringIncompleteError, match="row order changed"):
            await scorer.ascore(instance, output, _StubStagingEnv())


# ---------------------------------------------------------------------------
# Grader
# ---------------------------------------------------------------------------


def _grade(
    solution: str,
    tests: list[dict[str, str]] | str,
    fn_name: str | None = None,
    timeout: float = 5,
) -> dict[str, Any]:
    """Run the container-side grader the way the sandbox would."""
    with tempfile.TemporaryDirectory() as work_dir:
        with open(os.path.join(work_dir, "grade.py"), "w") as handle:
            handle.write(get_script("livecodebench_grader"))
        with open(os.path.join(work_dir, "problem.json"), "w") as handle:
            json.dump(
                {
                    "public_test_cases": tests if isinstance(tests, str) else json.dumps(tests),
                    "private_test_cases": "",
                    "fn_name": fn_name,
                    "timeout": timeout,
                },
                handle,
            )
        with open(os.path.join(work_dir, "solution.py"), "w") as handle:
            handle.write(solution)

        process = subprocess.run(
            [sys.executable, "grade.py"],
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=120,
        )

    verdicts = [
        line for line in process.stdout.strip().splitlines() if line.startswith(VERDICT_PREFIX)
    ]
    return json.loads(verdicts[-1][len(VERDICT_PREFIX) :])


SUM_TESTS = [
    {"input": "1 2\n", "output": "3\n"},
    {"input": "10 20\n", "output": "30\n"},
]


class TestGraderStdinMode:
    def test_correct_program_passes(self) -> None:
        solution = "a, b = map(int, input().split())\nprint(a + b)\n"
        assert _grade(solution, SUM_TESTS)["passed"] is True

    def test_wrong_program_fails(self) -> None:
        verdict = _grade("a, b = map(int, input().split())\nprint(a - b)\n", SUM_TESTS)
        assert verdict["passed"] is False
        assert verdict["error_code"] == -2

    def test_main_guard_is_unwrapped(self) -> None:
        solution = (
            "def main():\n"
            "    a, b = map(int, input().split())\n"
            "    print(a + b)\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        )
        assert _grade(solution, SUM_TESTS)["passed"] is True

    def test_extra_output_line_fails(self) -> None:
        solution = "a, b = map(int, input().split())\nprint(a + b)\nprint('done')\n"
        verdict = _grade(solution, SUM_TESTS)
        assert verdict["passed"] is False
        assert "length" in verdict["error_message"]

    def test_numerically_equal_output_passes(self) -> None:
        # Formatting differences must not fail an otherwise correct answer.
        tests = [{"input": "1\n", "output": "1.50\n"}]
        assert _grade("input()\nprint(1.5)\n", tests)["passed"] is True

    def test_runtime_error_fails(self) -> None:
        verdict = _grade("raise ValueError('boom')\n", SUM_TESTS)
        assert verdict["passed"] is False
        assert verdict["error_code"] == -4

    def test_endless_loop_times_out(self) -> None:
        verdict = _grade("while True:\n    pass\n", SUM_TESTS, timeout=2)
        assert verdict["passed"] is False
        assert verdict["error_code"] == -3

    def test_sub_second_timeout_is_enforced(self) -> None:
        # A whole-second alarm would round this down to no alarm at all.
        verdict = _grade("while True:\n    pass\n", SUM_TESTS, timeout=0.5)
        assert verdict["passed"] is False
        assert verdict["error_code"] == -3

    def test_solution_that_disarms_the_alarm_hits_the_overall_budget(self) -> None:
        # The per-case alarm can be silenced from inside the solution; the
        # budget for the whole run cannot, because another process keeps it.
        solution = (
            "import signal\nsignal.signal(signal.SIGALRM, signal.SIG_IGN)\nwhile True:\n    pass\n"
        )
        verdict = _grade(solution, SUM_TESTS[:1], timeout=0.1)
        assert verdict["passed"] is False
        assert verdict["error_code"] == -3
        assert "budget" in verdict["error_message"]

    def test_closing_stdout_keeps_the_output(self) -> None:
        solution = (
            "a, b = map(int, input().split())\nprint(a + b)\nimport sys\nsys.stdout.close()\n"
        )
        assert _grade(solution, SUM_TESTS)["passed"] is True

    def test_large_integers_convert_to_strings(self) -> None:
        # The interpreter's default limit on int/str conversion is far below
        # what contest answers reach; the reference harness raises it.
        tests = [{"input": "5000\n", "output": "1" + "0" * 5000 + "\n"}]
        assert _grade("n = int(input())\nprint(10 ** n)\n", tests)["passed"] is True

    def test_crashed_interpreter_is_a_failed_solution(self) -> None:
        # A crash takes down the process running the solution, not the grader.
        solution = "import ctypes\nctypes.string_at(0)\n"
        verdict = _grade(solution, SUM_TESTS)
        assert verdict["passed"] is False
        assert verdict["error_code"] == -4
        assert "signal" in verdict["error_message"]
        assert verdict["num_tests"] == len(SUM_TESTS)

    def test_syntax_error_is_reported_as_a_compilation_failure(self) -> None:
        verdict = _grade("  this is not python\n", SUM_TESTS)
        assert verdict["passed"] is False
        assert verdict["error_code"] == -4
        assert "Compilation error" in verdict["error_message"]


DOUBLE_TESTS = [
    {"input": "[1, 2, 3]", "output": "[2, 4, 6]"},
    {"input": "[]", "output": "[]"},
]


class TestGraderCallMode:
    def test_correct_solution_class_passes(self) -> None:
        solution = (
            "class Solution:\n"
            "    def double(self, nums: List[int]) -> List[int]:\n"
            "        return [n * 2 for n in nums]\n"
        )
        assert _grade(solution, DOUBLE_TESTS, fn_name="double")["passed"] is True

    def test_wrong_solution_fails(self) -> None:
        solution = (
            "class Solution:\n"
            "    def double(self, nums: List[int]) -> List[int]:\n"
            "        return nums\n"
        )
        verdict = _grade(solution, DOUBLE_TESTS, fn_name="double")
        assert verdict["passed"] is False
        assert verdict["error_code"] == -2

    def test_tuple_result_is_accepted(self) -> None:
        # Ground truth is never a tuple, so a tuple answer is not a mismatch.
        solution = (
            "class Solution:\n"
            "    def double(self, nums: List[int]) -> List[int]:\n"
            "        return tuple(n * 2 for n in nums)\n"
        )
        assert _grade(solution, DOUBLE_TESTS, fn_name="double")["passed"] is True

    def test_missing_function_fails(self) -> None:
        verdict = _grade("class Solution:\n    pass\n", DOUBLE_TESTS, fn_name="double")
        assert verdict["passed"] is False
        assert verdict["error_code"] == -4

    def test_module_level_function_is_found(self) -> None:
        solution = "def double(nums):\n    return [n * 2 for n in nums]\n"
        assert _grade(solution, DOUBLE_TESTS, fn_name="double")["passed"] is True

    def test_endless_module_body_is_a_timeout_not_a_compile_error(self) -> None:
        solution = "while True:\n    pass\ndef double(nums):\n    return nums\n"
        verdict = _grade(solution, DOUBLE_TESTS, fn_name="double", timeout=1)
        assert verdict["passed"] is False
        assert verdict["error_code"] == -3

    def test_exiting_from_the_function_is_a_runtime_error(self) -> None:
        solution = "def double(nums):\n    import sys\n    sys.exit(0)\n"
        verdict = _grade(solution, DOUBLE_TESTS, fn_name="double")
        assert verdict["passed"] is False
        assert verdict["error_code"] == -4


class TestGraderPayloads:
    def test_undecodable_test_cases_yield_no_tests_and_no_pass(self) -> None:
        # An empty test set is passed by every solution, so it must not be one.
        verdict = _grade("print(1)\n", "not json and not base64 pickle")
        assert verdict["passed"] is False
        assert verdict["num_tests"] == 0
        assert verdict["error_code"] == -5
        assert "neither JSON nor compressed pickle" in verdict["error_message"]

    def test_case_count_is_reported(self) -> None:
        solution = "a, b = map(int, input().split())\nprint(a + b)\n"
        assert _grade(solution, SUM_TESTS)["num_tests"] == len(SUM_TESTS)


def test_prompt_and_grader_use_same_dataset_revision():
    for name in ("livecodebench", "livecodebench_hidden"):
        task = get_task(name)
        assert task.config.data_source.revision == LIVECODEBENCH_REVISION
        instance = task.process_doc(STDIN_DOC)
        assert instance.metadata["test_revision"] == LIVECODEBENCH_REVISION


def test_grader_revision_is_part_of_download_url():
    _open_test_case_rows.cache_clear()
    try:
        with mock.patch("datasets.load_dataset", return_value=[]) as load:
            _open_test_case_rows("org/repo", ("test.jsonl",), "revision-a")
            _open_test_case_rows("org/repo", ("test.jsonl",), "revision-b")
        assert load.call_count == 2
        assert load.call_args_list[0].kwargs["data_files"] == {
            "train": ["hf://datasets/org/repo@revision-a/test.jsonl"]
        }
        assert load.call_args_list[1].kwargs["data_files"] == {
            "train": ["hf://datasets/org/repo@revision-b/test.jsonl"]
        }
    finally:
        _open_test_case_rows.cache_clear()


def test_process_doc_accepts_uri_data_source():
    task = get_task("livecodebench", {"data_source": "hf://livecodebench/code_generation_lite"})
    instance = task.process_doc(STDIN_DOC)
    assert instance.metadata["test_revision"] == task.config.get_data_source().revision


def test_grader_reads_tests_from_the_configured_repository():
    mirror = DataSource(
        path="hf://org/lcb-mirror",
        revision="mirror-rev",
        data_files=RELEASE_V3_FILES,
        split="train",
    )
    instance = get_task("livecodebench", {"data_source": mirror}).process_doc(STDIN_DOC)
    assert instance.metadata["test_repo"] == "org/lcb-mirror"
    assert instance.metadata["test_revision"] == "mirror-rev"

    uri_task = get_task("livecodebench", {"data_source": "hf://org/lcb-mirror"})
    assert uri_task.process_doc(STDIN_DOC).metadata["test_repo"] == "org/lcb-mirror"

    instance = get_task("livecodebench").process_doc(STDIN_DOC)
    assert instance.metadata["test_repo"] == LIVECODEBENCH_REPO
