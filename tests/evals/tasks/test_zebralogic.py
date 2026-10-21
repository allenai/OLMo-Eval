"""Tests for ZebraLogic request formatting."""

from olmo_eval.common.types import Instance, RequestType
from olmo_eval.evals.tasks.common import get_task


def test_completion_request() -> None:
    task = get_task("zebralogic")
    instance = Instance(question="Solve this puzzle.")

    request = task.format_request(instance)

    assert task.request_type == RequestType.COMPLETION
    assert request.request_type == RequestType.COMPLETION
    assert request.prompt == instance.question
    assert request.messages == ()


def test_chat_request() -> None:
    task = get_task("zebralogic:chat")
    instance = Instance(question="Solve this puzzle.")

    request = task.format_request(instance)

    assert task.request_type == RequestType.CHAT
    assert request.request_type == RequestType.CHAT
    assert request.prompt == ""
    assert request.messages == ({"role": "user", "content": instance.question},)
