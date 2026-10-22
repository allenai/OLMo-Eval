"""End-to-end wiring test: the deepsearchqa task through the dr_tulu harness preset.

Scripts a fake OpenAI-compatible chat-completions endpoint via httpx.MockTransport so
the real `agents` SDK loop, the real dr_tulu tool set, and the real deepsearchqa
prompt/answer-extraction are all exercised together -- with no GPU, no network, and
no API keys.
"""

import json
from types import SimpleNamespace

import httpx
import pytest

pytest.importorskip("agents")

from olmo_eval.common.types import LMOutput, Response
from olmo_eval.evals.tasks import deepsearchqa
from olmo_eval.evals.tasks.common import get_task
from olmo_eval.harness.presets import get_harness_preset
from olmo_eval.harness.scaffolds.openai_agents import OpenAIAgentsScaffold


def _doc():
    return {
        "problem": "Which countries border Chad?",
        "problem_category": "Geography",
        "answer": "Libya, Sudan, Niger",
        "answer_type": "Set Answer",
    }


def _completion(message: dict, finish: str) -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


@pytest.mark.anyio
async def test_deepsearchqa_through_dr_tulu_harness(monkeypatch):
    from openai import AsyncOpenAI

    # Guarantee the real search tools take their "no key configured" short-circuit
    # instead of ever attempting real network I/O.
    monkeypatch.delenv("SERPER_API_KEY", raising=False)
    monkeypatch.delenv("S2_API_KEY", raising=False)

    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content.decode()))
        if len(requests) == 1:
            return httpx.Response(
                200,
                json=_completion(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "serper_google_webpage_search",
                                    "arguments": '{"query": "countries bordering Chad"}',
                                },
                            }
                        ],
                    },
                    "tool_calls",
                ),
            )
        return httpx.Response(
            200,
            json=_completion(
                {
                    "role": "assistant",
                    "content": "Based on my search.\n\nFINAL ANSWER: Libya, Sudan, Niger",
                },
                "stop",
            ),
        )

    client = AsyncOpenAI(
        api_key="test-key-not-real",
        base_url="http://test.invalid/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    provider = SimpleNamespace(model_name="test-model", get_openai_client=lambda: client)

    task = get_task("deepsearchqa:mini")
    instance = task.process_doc(_doc(), index=0)
    request = task.format_request(instance)

    result = await OpenAIAgentsScaffold().run(
        provider=provider,
        config=get_harness_preset("dr_tulu"),
        request=request,
        enable_compaction=False,
    )

    assert result.error is None
    assert "FINAL ANSWER: Libya, Sudan, Niger" in result.final_output.text

    # The real dr_tulu tool set was reachable under its registered name.
    called_tools = [tc.function.name for turn in result.trajectory.turns for tc in turn.tool_calls]
    assert "serper_google_webpage_search" in called_tools

    assert task.extract_answer(result.final_output) == "Libya, Sudan, Niger"

    response = Response(
        instance=instance,
        request=request,
        outputs=[LMOutput(text=result.final_output.text)],
        scores={},
    )

    async def fake_judge(prompt):
        return '{"matched_gold_indices": [0, 1, 2], "matched_submitted_indices": [0, 1, 2]}'

    monkeypatch.setattr(deepsearchqa, "build_deepsearchqa_judge_fn", lambda: fake_judge)
    await task.score_responses([response])

    assert response.scores["deepsearchqa_f1"] == 1.0
