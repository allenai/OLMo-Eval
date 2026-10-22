"""Stand-in judges and helpers shared by the judged-task tests.

A scripted judge returns prepared replies and records what it was asked, so a test
can assert prompts and scores without calling a model. ``use_judge`` installs one
for every judged task built afterwards, and ``score_like_the_runner`` scores an
instance the way the async runner does, failure path included. Modules that use
them also apply the ``judge_test_env`` fixture from this package's conftest.
"""

from __future__ import annotations

from olmo_eval.common.scorers.execution import format_scoring_error
from olmo_eval.common.scorers.llm_judge import JudgeFn
from olmo_eval.common.types import Response
from olmo_eval.evals.tasks.common import judge as judge_module
from olmo_eval.runners.asynq.results import _record_scoring_failure


def scripted_judge(*replies: str) -> tuple[JudgeFn, list[str]]:
    """A judge that returns ``replies`` in order, and the list of prompts it is sent."""
    remaining = iter(replies)
    prompts: list[str] = []

    async def judge(prompt: str, *, system_prompt: str | None = None) -> str:
        prompts.append(prompt)
        return next(remaining)

    return judge, prompts


def use_judge(monkeypatch, judge: JudgeFn) -> None:
    """Make judged tasks call ``judge`` instead of building an API client."""
    monkeypatch.setattr(judge_module, "build_openai_judge_fn", lambda **_: judge)


async def score_like_the_runner(task, response: Response) -> Response:
    """Score one instance the way the async runner does, including its failure path."""
    try:
        return (await task.score_responses([response]))[0]
    except Exception as exc:
        return _record_scoring_failure(response, error=format_scoring_error(exc, phase="response"))
