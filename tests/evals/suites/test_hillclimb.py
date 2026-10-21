"""Tests for the hill-climb dev pass suite."""

import pytest

from olmo_eval.common.configs import expand_tasks
from olmo_eval.evals.suites import AggregationStrategy, get_suite
from olmo_eval.evals.tasks.common import get_task
from olmo_eval.runners.processing.aggregation import compute_suite_aggregations
from olmo_eval.runners.processing.metrics import build_single_model_metrics

SUITE = "hillclimb:dev"
TASKS = (
    "livecodebench:lite",
    "omega_500:hillclimb",
    "omega_500_out",
    "ifeval",
    "ifeval_ood",
    "gpqa_main:cot",
)
# The OMEGA cap. No task in the pass may stop a reasoning trace earlier.
MIN_GENERATION_BUDGET = 32768


def test_one_launch_runs_the_dev_tiers_and_guards():
    suite = get_suite(SUITE)

    assert suite.aggregation == AggregationStrategy.NONE
    assert suite.expand() == TASKS
    assert tuple(expand_tasks([SUITE])) == TASKS


def test_omega_pair_is_the_gap_suite():
    suite = get_suite(SUITE)

    assert get_suite("omega:dev") in suite.tasks


def test_livecodebench_dev_tier_is_one_sample_per_problem():
    config = get_task("livecodebench:lite").config

    assert config.sampling_params is not None
    assert config.sampling_params.num_samples == 1
    assert config.primary_metric.name == "pass_at_1"


@pytest.mark.parametrize("task", TASKS)
def test_no_task_truncates_reasoning_before_the_omega_cap(task):
    params = get_task(task).config.sampling_params

    assert params is not None
    assert params.max_tokens is None or params.max_tokens >= MIN_GENERATION_BUDGET


def _task(score: float) -> dict:
    return {
        "metrics": {"exact_match": {"exact_match": score}},
        "primary_metric": "exact_match:exact_match",
        "num_instances": 500,
    }


def test_reports_the_omega_gap_and_no_pass_average():
    task_results = {task: _task(0.5) for task in TASKS}
    task_results["omega_500:hillclimb"] = _task(0.6)
    task_results["omega_500_out"] = _task(0.45)

    suites = compute_suite_aggregations([SUITE], task_results)

    assert set(suites) == {"omega:dev"}
    assert suites["omega:dev"]["metrics"]["primary_score"]["gap"] == pytest.approx(-0.15)
    assert suites["omega:dev"]["container_suite"] == SUITE
    assert suites["omega:dev"]["primary_metric"] == "primary_score:companion"
    summary = build_single_model_metrics({"tasks": task_results, "suites": suites}).to_dict()[
        "summary"
    ]
    assert SUITE not in summary
    assert summary["omega:dev"]["score"] == 0.45
