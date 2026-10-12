"""Integration test for LiveCodeBench grading in a real sandbox.

Requires Docker and the LiveCodeBench dataset. Test cases are staged into the
container as files, which is the part that cannot be exercised with a stub:
the payload is far too large to pass in a command, so nothing about this path
is covered by the unit tests.

    pytest tests/integration/test_livecodebench_sandbox.py -v
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import AsyncIterator

import pytest

from olmo_eval.common.types import Instance, LMOutput
from olmo_eval.evals.tasks.common import get_task
from olmo_eval.evals.tasks.livecodebench import (
    LIVECODEBENCH_REPO,
    LIVECODEBENCH_REVISION,
    RELEASE_V3_FILES,
    LiveCodeBenchScorer,
    _test_case_rows,
)
from olmo_eval.harness.sandbox import Capability, SandboxConfig, SandboxManager, SandboxMode

pytestmark = pytest.mark.integration

SANDBOX_IMAGE = "ghcr.io/astral-sh/uv:python3.12-bookworm-slim"

# A whole-program problem: stdin in, stdout compared line by line.
STDIN_PROBLEM_ID = "1873_B"
STDIN_SOLUTION = """
import sys

def main():
    data = sys.stdin.read().split()
    t = int(data[0])
    idx = 1
    out = []
    for _ in range(t):
        n = int(data[idx])
        idx += 1
        digits = [int(x) for x in data[idx : idx + n]]
        idx += n
        best = 0
        for i in range(n):
            product = 1
            for j in range(n):
                product *= digits[j] + (1 if i == j else 0)
            best = max(best, product)
        out.append(str(best))
    print("\\n".join(out))

main()
"""

# A call-based problem: the named function is invoked with parsed arguments.
CALL_PROBLEM_FN = "countSeniors"
CALL_SOLUTION = """
class Solution:
    def countSeniors(self, details: List[str]) -> int:
        return sum(1 for d in details if int(d[11:13]) > 60)
"""


def _docker_available() -> bool:
    try:
        result = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


@pytest.fixture(scope="module")
def livecodebench_instances() -> dict[str, Instance]:
    """Real instances, keyed by problem id."""
    task = get_task("livecodebench")
    return {instance.metadata["id"]: instance for instance in task.instances}


@pytest.fixture
async def sandbox() -> AsyncIterator[object]:
    """A single running sandbox with swe-rex injected into the image."""
    if not _docker_available():
        pytest.skip("Docker is not available")

    manager = SandboxManager(
        [
            SandboxConfig(
                image=SANDBOX_IMAGE,
                mode=SandboxMode.DOCKER,
                container_runtime="docker",
                instances=1,
                startup_timeout=900.0,
                command_timeout=900.0,
                inject_swerex=True,
            )
        ],
        owner="livecodebench-test",
    )
    await manager.start()
    try:
        yield manager.for_capabilities(Capability.DEFAULT)
    finally:
        await manager.stop()


def _call_based_instance(instances: dict[str, Instance]) -> Instance:
    for instance in instances.values():
        if instance.metadata.get("fn_name") == CALL_PROBLEM_FN:
            return instance
    pytest.skip(f"No problem with function {CALL_PROBLEM_FN} in this release")


async def _score(scorer: LiveCodeBenchScorer, instance: Instance, code: str, sandbox) -> float:
    output = LMOutput(text="unused")
    output.extracted_answer = code
    return await scorer.ascore(instance, output, sandbox)


@pytest.mark.anyio
async def test_correct_stdin_solution_passes(
    sandbox, livecodebench_instances: dict[str, Instance]
) -> None:
    instance = livecodebench_instances[STDIN_PROBLEM_ID]
    assert instance.metadata["fn_name"] is None

    score = await _score(LiveCodeBenchScorer(), instance, STDIN_SOLUTION, sandbox)

    assert score == 1.0


@pytest.mark.anyio
async def test_wrong_stdin_solution_fails(
    sandbox, livecodebench_instances: dict[str, Instance]
) -> None:
    instance = livecodebench_instances[STDIN_PROBLEM_ID]
    wrong = STDIN_SOLUTION.replace("best = max(best, product)", "best = max(best, product + 1)")

    score = await _score(LiveCodeBenchScorer(), instance, wrong, sandbox)

    assert score == 0.0


@pytest.mark.anyio
async def test_correct_call_based_solution_passes(
    sandbox, livecodebench_instances: dict[str, Instance]
) -> None:
    instance = _call_based_instance(livecodebench_instances)

    score = await _score(LiveCodeBenchScorer(), instance, CALL_SOLUTION, sandbox)

    assert score == 1.0


@pytest.mark.anyio
async def test_private_test_cases_reach_the_container(
    sandbox, livecodebench_instances: dict[str, Instance]
) -> None:
    """The staged payload must be the problem's full case list, not just public ones."""
    instance = livecodebench_instances[STDIN_PROBLEM_ID]
    output = LMOutput(text="unused")
    output.extracted_answer = STDIN_SOLUTION

    await LiveCodeBenchScorer().ascore(instance, output, sandbox)

    result = output.metadata["execution_result"]
    assert result["success"] is True

    # Public cases are plain JSON on the row; the private ones are what the
    # grader had to decode in the container.
    row = _test_case_rows(LIVECODEBENCH_REPO, RELEASE_V3_FILES, LIVECODEBENCH_REVISION)[
        instance.metadata["row"]
    ]
    public_count = len(json.loads(row["public_test_cases"]))
    assert public_count > 0
    assert result["num_tests"] > public_count


@pytest.mark.anyio
async def test_release_files_are_the_v3_window(
    livecodebench_instances: dict[str, Instance],
) -> None:
    assert len(livecodebench_instances) == 612
    instance = next(iter(livecodebench_instances.values()))
    assert tuple(instance.metadata["test_files"]) == RELEASE_V3_FILES
