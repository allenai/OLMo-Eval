"""Request failures during auxiliary judge scoring."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from olmo_eval.common.execution import ScoringContext
from olmo_eval.common.scorers.llm_judge import SafetyScorer
from olmo_eval.common.types import Instance, LMOutput
from olmo_eval.inference.errors import REQUEST_ERROR_KEY


@pytest.mark.parametrize("empty", [False, True])
def test_strongreject_request_failure_raises_from_scorer(empty: bool) -> None:
    tokenizer = SimpleNamespace(
        encode=lambda text, **kwargs: list(text), decode=lambda tokens: "".join(tokens)
    )
    provider = SimpleNamespace(
        alogprobs=AsyncMock(
            return_value=[
                [
                    LMOutput(
                        text="",
                        metadata={REQUEST_ERROR_KEY: "RuntimeError: judge service unavailable"},
                    )
                ]
            ]
        ),
        get_tokenizer=lambda: tokenizer,
    )
    if empty:
        provider.alogprobs.return_value = [[]]
    context = ScoringContext(
        inference_pool=SimpleNamespace(get=lambda name: provider, names=["judge"])
    )
    scorer = SafetyScorer(provider_name="judge", judge_format="strongreject")
    instance = Instance(question="q", gold_answer="response_harmfulness=yes")

    with pytest.raises(
        RuntimeError, match="returned no output" if empty else "judge service unavailable"
    ):
        asyncio.run(scorer.ascore_with_context(instance, LMOutput(text="answer"), context))

    provider.alogprobs.assert_awaited_once()
    assert "judge_result" not in instance.metadata
