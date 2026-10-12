"""Tests that README reference tables and examples match the code."""

import dataclasses
import re
from pathlib import Path

import pytest

from olmo_eval.common.types import RequestType
from olmo_eval.evals.tasks.common.base import TaskConfig
from olmo_eval.evals.tasks.common.registry import _configs, _tasks, get_task
from olmo_eval.harness.sandbox import SandboxConfig

README = (Path(__file__).parents[1] / "README.md").read_text()


def _section(heading: str) -> str:
    """Return README text from a heading up to the next heading of any level."""
    start = README.index(f"{heading}\n")
    next_heading = re.search(r"^#{1,6} ", README[start + len(heading) :], re.MULTILINE)
    end = start + len(heading) + next_heading.start() if next_heading else len(README)
    return README[start:end]


def _table_fields(section: str) -> set[str]:
    return set(re.findall(r"^\| `([a-z_]+)` \|", section, re.MULTILINE))


@pytest.mark.parametrize(
    ("heading", "config_class"),
    [
        ("### TaskConfig Reference", TaskConfig),
        ("### SandboxConfig Fields", SandboxConfig),
    ],
)
def test_config_table_documents_existing_fields(heading: str, config_class: type):
    documented = _table_fields(_section(heading))
    actual = {field.name for field in dataclasses.fields(config_class)}

    assert documented
    assert documented <= actual, f"README documents missing fields: {documented - actual}"


def test_minimal_task_example_formats_a_document():
    section = _section("### Quick Start: Minimal Task Example")
    match = re.search(r"```python\n(.*?)```", section, re.DOTALL)
    assert match is not None

    namespace: dict[str, object] = {"__name__": "readme_minimal_task"}
    try:
        exec(compile(match.group(1), "README minimal task", "exec"), namespace)
        task = get_task("my_task")
        instance = task.process_doc(
            {"question": "What is 2 + 2?", "choices": ["3", "4", "5", "6"], "answer": 1}, 0
        )
        assert instance is not None
        request = task.format_request(instance)
    finally:
        _tasks.pop("my_task", None)
        _configs.pop("my_task", None)

    assert task.config.metrics
    assert task.config.data_source is not None
    assert request.request_type == RequestType.LOGLIKELIHOOD
    assert "B. 4" in request.prompt
    assert instance.metadata["gold_idx"] == 1
