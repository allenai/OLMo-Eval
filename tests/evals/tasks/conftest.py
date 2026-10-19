"""Fixtures shared by the task tests."""

import pytest

from olmo_eval.common.scorers import llm_judge
from olmo_eval.evals.tasks.common import judge as judge_module


@pytest.fixture
def judge_test_env(monkeypatch):
    """Judge with each task's own default judge, retrying without waiting."""
    monkeypatch.delenv("OLMO_EVAL_JUDGE", raising=False)

    async def sleep(_delay):
        return None

    monkeypatch.setattr(judge_module.asyncio, "sleep", sleep)
    monkeypatch.setattr(llm_judge, "retry_with_backoff", lambda call, **_: call())
