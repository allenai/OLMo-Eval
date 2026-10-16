"""SWE-bench instance dataclass."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

# Mirror of the SWE-bench evaluation images that can be pulled without logging in,
# unlike Docker Hub, whose anonymous pull limit is shared across a cluster.
GHCR_IMAGE_TEMPLATE = "ghcr.io/epoch-research/swe-bench.eval.x86_64.{instance_id}:latest"
IMAGE_SOURCES = ("ghcr", "dataset")


def _as_test_list(value: Any) -> tuple[str, ...]:
    """Normalize a test list that the dataset may store as a JSON string."""
    if isinstance(value, str):
        value = json.loads(value) if value.strip() else []
    return tuple(str(v) for v in value or ())


@dataclass(frozen=True)
class SWEBenchInstance:
    """A single SWE-bench task instance.

    Attributes:
        instance_id: Unique identifier, e.g. ``astropy__astropy-12907``.
        repo: GitHub repository in ``owner/name`` form.
        base_commit: Commit the repository is checked out at in the image.
        problem_statement: Issue text shown to the agent.
        hints_text: Issue comments posted before the fix, if any.
        image: Prebuilt evaluation image with the repository and its environment.
        eval_script: Shell script that applies the test patch and runs the tests.
        log_parser: Name of the upstream parser that reads the test log.
        eval_type: Upstream grading mode for the test lists.
        version: Repository version used to select the environment.
        fail_to_pass: Tests that must go from failing to passing.
        pass_to_pass: Tests that must keep passing.
        gold_patch: Reference fix, used by oracle mode.
        difficulty: Annotated time-to-fix bucket.
    """

    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    hints_text: str
    image: str
    eval_script: str
    log_parser: str
    eval_type: str
    version: str
    fail_to_pass: tuple[str, ...]
    pass_to_pass: tuple[str, ...]
    gold_patch: str
    difficulty: str = "unknown"

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> SWEBenchInstance:
        """Build an instance from a dataset row."""
        return cls(
            instance_id=str(row["instance_id"]),
            repo=str(row["repo"]),
            base_commit=str(row["base_commit"]),
            problem_statement=str(row["problem_statement"]),
            hints_text=str(row.get("hints_text") or ""),
            image=str(row["image"]),
            eval_script=str(row["eval_script"]),
            log_parser=str(row["log_parser"]),
            eval_type=str(row["eval_type"]),
            version=str(row["version"]),
            fail_to_pass=_as_test_list(row["FAIL_TO_PASS"]),
            pass_to_pass=_as_test_list(row["PASS_TO_PASS"]),
            gold_patch=str(row.get("patch") or ""),
            difficulty=str(row.get("difficulty") or "unknown"),
        )

    def image_for(self, source: str) -> str:
        """Return the instance's evaluation image from the given source.

        Args:
            source: ``"ghcr"`` for the GitHub Container Registry mirror, or
                ``"dataset"`` for the image named in the dataset (Docker Hub).

        Raises:
            ValueError: If the source is unknown.
        """
        if source == "ghcr":
            return GHCR_IMAGE_TEMPLATE.format(instance_id=self.instance_id)
        if source == "dataset":
            return self.image
        raise ValueError(f"Unknown image source {source!r}; expected one of {IMAGE_SOURCES}")

    def to_swebench_dict(self) -> dict[str, Any]:
        """Return the fields the upstream ``swebench`` grader expects."""
        return {
            "instance_id": self.instance_id,
            "repo": self.repo,
            "version": self.version,
            "image": self.image,
            "eval_script": self.eval_script,
            "log_parser": self.log_parser,
            "eval_type": self.eval_type,
            "FAIL_TO_PASS": list(self.fail_to_pass),
            "PASS_TO_PASS": list(self.pass_to_pass),
        }
