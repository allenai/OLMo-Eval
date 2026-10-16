"""Tests that the checked-in Beaker example configs stay launchable."""

import importlib
from pathlib import Path

import pytest

from olmo_eval.evals.suites import suite_exists
from olmo_eval.evals.tasks.common.registry import get_base_task_name, task_exists
from olmo_eval.launch import EvalConfig
from olmo_eval.launch.beaker.launcher import validate_min_runtime

# Importing the evals package registers every task and suite.
importlib.import_module("olmo_eval.evals")

EXAMPLES_DIR = Path(__file__).parents[3] / "examples" / "beaker" / "configs"
EXAMPLE_CONFIGS = sorted(EXAMPLES_DIR.glob("*.yaml"))


def test_example_configs_exist():
    assert EXAMPLE_CONFIGS


@pytest.mark.parametrize("path", EXAMPLE_CONFIGS, ids=lambda path: path.name)
def test_example_config_references_registered_tasks(path: Path):
    config = EvalConfig.from_yaml(str(path))

    unknown = [
        spec
        for spec in config.tasks
        if not (task_exists(get_base_task_name(spec)) or suite_exists(get_base_task_name(spec)))
    ]
    assert not unknown, f"{path.name} references unregistered tasks: {unknown}"


@pytest.mark.parametrize("path", EXAMPLE_CONFIGS, ids=lambda path: path.name)
def test_example_config_uses_supported_fields(path: Path):
    config = EvalConfig.from_yaml(str(path))

    # The launch CLI treats each model entry as a model name or preset.
    assert all(isinstance(model, str) for model in config.models)
    assert config.cluster
    if config.min_runtime is not None:
        validate_min_runtime(config.min_runtime)


@pytest.mark.parametrize("path", EXAMPLE_CONFIGS, ids=lambda path: path.name)
def test_example_config_usage_comment_points_at_itself(path: Path):
    usage_lines = [line for line in path.read_text().splitlines() if "beaker launch -f" in line]

    assert usage_lines
    for line in usage_lines:
        assert f"examples/beaker/configs/{path.name}" in line
