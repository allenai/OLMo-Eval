"""Dotted ``-o sampling_params.<field>=...`` overrides merge onto the task's own params."""

from __future__ import annotations

import pytest

from olmo_eval.common.types import SamplingParams
from olmo_eval.evals.tasks.common.base import TaskConfig
from olmo_eval.runners.asynq.runner import AsyncEvalRunner


def _overrides(per_task: dict) -> tuple[dict, dict]:
    runner = AsyncEvalRunner.__new__(AsyncEvalRunner)
    runner.task_overrides = {"t": per_task}
    return runner._build_task_overrides("t")


def test_dotted_sampling_params_route_to_sampling_overrides() -> None:
    # As a task override the dict replaced the task's SamplingParams wholesale, dropping
    # every field it omitted -- gsm8k's stop_sequences among them.
    task_overrides, sampling_overrides = _overrides({"sampling_params": {"max_tokens": 64}})

    assert "sampling_params" not in task_overrides
    assert sampling_overrides == {"max_tokens": 64}


def test_unknown_dotted_sampling_field_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown sampling_params field"):
        _overrides({"sampling_params": {"max_tokenz": 64}})


def test_flat_overrides_are_unchanged() -> None:
    task_overrides, sampling_overrides = _overrides({"limit": 5, "max_tokens": 32})

    assert task_overrides == {"limit": 5}
    assert sampling_overrides == {"max_tokens": 32}


def test_task_config_refuses_a_raw_sampling_params_dict() -> None:
    # A dict reaching TaskConfig means a caller bypassed the runner's merge.
    with pytest.raises(TypeError, match="must be a SamplingParams instance"):
        TaskConfig(name="t", sampling_params={"max_tokens": 64})  # type: ignore[arg-type]


def test_task_config_names_an_unknown_sampling_field() -> None:
    with pytest.raises(ValueError, match="unknown sampling_params field"):
        TaskConfig(name="t", sampling_params={"max_tokenz": 64})  # type: ignore[arg-type]


def test_task_config_accepts_sampling_params() -> None:
    params = SamplingParams(max_tokens=8)
    assert TaskConfig(name="t", sampling_params=params).sampling_params is params
