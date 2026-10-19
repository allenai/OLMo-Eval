"""Tests for the scripted multi-turn conversation helpers."""

import pytest

from olmo_eval.common.multi_turn import (
    missing_replies_error,
    scripted_chat_request,
    turn_count,
    turn_replies,
)
from olmo_eval.common.types import (
    FOLLOW_UP_TURNS_KEY,
    TURNS_KEY,
    LMOutput,
    LMRequest,
    RequestType,
)


def test_a_scripted_request_asks_the_first_turn_and_follows_up_with_the_rest():
    request = scripted_chat_request(["one", "two", "three"])

    assert request.request_type == RequestType.CHAT
    assert request.messages == ({"role": "user", "content": "one"},)
    assert request.metadata == {
        FOLLOW_UP_TURNS_KEY: [
            {"role": "user", "content": "two"},
            {"role": "user", "content": "three"},
        ]
    }
    assert turn_count(request) == 3


def test_a_single_turn_request_has_no_follow_ups():
    request = scripted_chat_request(["only"])

    assert request.metadata == {FOLLOW_UP_TURNS_KEY: []}
    assert turn_count(request) == 1
    assert turn_count(LMRequest(request_type=RequestType.CHAT)) == 1


def test_turn_replies_reads_the_scaffolds_replies():
    assert turn_replies(
        LMOutput(text="b", metadata={TURNS_KEY: [{"text": "a"}, {"text": "b"}]})
    ) == [
        "a",
        "b",
    ]
    assert turn_replies(LMOutput(text="b")) is None


def test_missing_replies_error_names_the_benchmark_and_the_scaffold():
    with pytest.raises(ValueError, match=r"A Multi-IF output .*--harness multi_turn_chat"):
        raise missing_replies_error("Multi-IF")
