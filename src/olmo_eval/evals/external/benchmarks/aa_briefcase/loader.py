"""Loader for the AA-Briefcase-Lite scenario from Hugging Face."""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

DATASET_REPO = "ArtificialAnalysis/AA-Briefcase-Lite"
DATASET_REVISION = "4dec557b47d43867a1648c0974db1d8208c8b677"

#: Everything but the example submissions, which hold about 75 MB of video.
DATASET_PATTERNS = (
    "checks.jsonl",
    "tasks.jsonl",
    "tasks/*",
    "summary_docs/**",
    "prompts/*",
    "source_files/**",
)

#: Rubric checks graded on one submission. The ``AQ`` and ``P`` checks compare two
#: models' submissions and have no published prompt.
BINARY_SCORING = "binary"


@dataclass(frozen=True)
class RubricCheck:
    """One binary rubric check of a task."""

    check_id: str
    task_id: str
    check_type: str
    description: str
    pass_criteria: str
    fail_criteria: str
    #: Deliverable file names the check grades.
    files: tuple[str, ...]

    @classmethod
    def from_row(cls, row: dict) -> RubricCheck:
        return cls(
            check_id=row["check_id"],
            task_id=row["task_id"],
            check_type=row["check_type"],
            description=row["check_description"],
            pass_criteria=row["score_1_criteria"],
            fail_criteria=row["score_0_criteria"],
            files=tuple(f.strip() for f in row["taskdoer_output_file"].split(",") if f.strip()),
        )


@dataclass(frozen=True)
class BriefcaseTask:
    """One deliverable task of the scenario, with its sources and rubric."""

    task_id: str
    brief: str
    deliverables: tuple[str, ...]
    #: Source entries (files, or directories ending in ``/``) relative to the
    #: dataset root, mounted under ``shared/`` and ``week/``.
    shared_files: tuple[str, ...]
    week_files: tuple[str, ...]
    checks: tuple[RubricCheck, ...]


@dataclass(frozen=True)
class BriefcaseScenario:
    """The scenario's tasks, its context documents and the harness prompts."""

    root: Path
    tasks: tuple[BriefcaseTask, ...]
    scenario_overview: str
    week_overview: str
    agent_system_prompt: str
    agent_task_prompt: str
    judge_system_prompt: str
    judge_user_prompt: str
    #: Pairwise checks left out of grading.
    pairwise_checks: int


def download_scenario(revision: str = DATASET_REVISION) -> Path:
    """Download the scenario at a revision and return its local root."""
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=DATASET_REPO,
            repo_type="dataset",
            revision=revision,
            allow_patterns=list(DATASET_PATTERNS),
        )
    )


def load_scenario(root: str | Path) -> BriefcaseScenario:
    """Read the tasks, rubric checks and prompts of a downloaded scenario.

    Raises:
        ValueError: If the tasks name different context documents, which this
            loader does not support.
    """
    root = Path(root)
    task_rows = _read_jsonl(root / "tasks.jsonl")
    check_rows = _read_jsonl(root / "checks.jsonl")
    binary = [RubricCheck.from_row(r) for r in check_rows if r["scoring_type"] == BINARY_SCORING]

    overview_paths = {(r["scenario_overview_path"], r["week_overview_path"]) for r in task_rows}
    if len(overview_paths) != 1:
        raise ValueError(f"Tasks name different overview documents: {sorted(overview_paths)}")
    scenario_overview_path, week_overview_path = overview_paths.pop()

    tasks = tuple(
        BriefcaseTask(
            task_id=row["task_id"],
            brief=(root / row["task_md_path"]).read_text(),
            deliverables=tuple(row["deliverable_filenames"]),
            shared_files=tuple(row["shared_files"]),
            week_files=tuple(row["week_files"]),
            checks=tuple(c for c in binary if c.task_id == row["task_id"]),
        )
        for row in task_rows
    )
    prompts = root / "prompts"
    return BriefcaseScenario(
        root=root,
        tasks=tasks,
        scenario_overview=(root / scenario_overview_path).read_text(),
        week_overview=(root / week_overview_path).read_text(),
        agent_system_prompt=(prompts / "eval_system.txt").read_text(),
        agent_task_prompt=(prompts / "eval_submission.txt").read_text(),
        judge_system_prompt=(prompts / "judge_system.txt").read_text(),
        judge_user_prompt=(prompts / "judge_user.txt").read_text(),
        pairwise_checks=len(check_rows) - len(binary),
    )


def select_tasks(
    tasks: tuple[BriefcaseTask, ...], task_ids: list[str] | None
) -> list[BriefcaseTask]:
    """Keep the named tasks, in scenario order.

    Raises:
        ValueError: If a requested task is not in the scenario.
    """
    if not task_ids:
        return list(tasks)
    known = {task.task_id for task in tasks}
    missing = [t for t in task_ids if t not in known]
    if missing:
        raise ValueError(f"Unknown AA-Briefcase-Lite tasks: {', '.join(missing)}")
    return [task for task in tasks if task.task_id in task_ids]


def stage_sources(root: Path, task: BriefcaseTask, target: Path) -> None:
    """Copy a task's source pool into ``target/shared`` and ``target/week``.

    Copies follow the Hugging Face cache's symlinks, so the directories can be
    bind-mounted into a container.
    """
    for group, entries in (("shared", task.shared_files), ("week", task.week_files)):
        prefix = Path("source_files") / group
        for entry in entries:
            source = root / entry
            destination = target / group / Path(entry).relative_to(prefix)
            if source.is_dir():
                shutil.copytree(source, destination, dirs_exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(source, destination)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
