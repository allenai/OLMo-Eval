"""Tests for DenseCaptionJudgeScorer and dense_caption task.

Two levels:
1. Unit tests for parse helpers — no I/O, no GPU, no API key needed.
2. Metric aggregation tests — synthetic scored outputs, verify ×100 scaling
   and valid-filter semantics.
3. Scorer integration tests with mocked GPT calls — verify end-to-end
   metadata plumbing and error handling.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any
from unittest.mock import patch

import pytest

from olmo_eval.common.execution import ScoringContext
from olmo_eval.common.types import Instance, LMOutput, Response
from olmo_eval.evals.vision.benchmarks.dense_caption import (
    _DEFAULT_METRICS,
    DenseCaptionAvgMetric,
)
from olmo_eval.evals.vision.scoring import judges
from olmo_eval.evals.vision.scoring.judges import (
    DenseCaptionJudgeScorer,
    parse_consistency_output,
    parse_recall_output,
)

# ---------------------------------------------------------------------------
# 1. Unit tests — parse_recall_output
# ---------------------------------------------------------------------------


class TestParseRecallOutput:
    def test_basic_stated_not_stated(self):
        text = "1. Stated\n2. Not Stated\n3. Stated"
        covered, total = parse_recall_output(text)
        assert total == 3
        assert covered == 2

    def test_gpt_misspelling_not_stted(self):
        # Legacy regex allows any suffix after "not st"
        text = "1. Not stated\n2. Not stted\n3. Stated"
        covered, total = parse_recall_output(text)
        assert total == 3
        assert covered == 1

    def test_numbered_with_prefix(self):
        # "Not Stated (missing)" does NOT match the end-anchored regex but
        # does contain " stated", so the legacy logic counts it as Stated.
        text = "1. The fact is Stated.\n2. Not Stated (missing)"
        covered, total = parse_recall_output(text)
        assert total == 2
        assert covered == 2

    def test_clean_not_stated_at_end(self):
        text = "1. Stated\n2. Not Stated"
        covered, total = parse_recall_output(text)
        assert total == 2
        assert covered == 1

    def test_ambiguous_lines_skipped(self):
        text = "1. Stated\n2. Unclear\n3. Not Stated"
        covered, total = parse_recall_output(text)
        assert total == 2  # "Unclear" is skipped
        assert covered == 1

    def test_empty_input(self):
        covered, total = parse_recall_output("")
        assert covered == 0
        assert total == 0

    def test_case_insensitive(self):
        text = "1. STATED\n2. NOT STATED"
        covered, total = parse_recall_output(text)
        assert total == 2
        assert covered == 1


class TestLegacyCacheKey:
    def test_key_matches_mm_olmo_gpt4_with_cache(self):
        # The shared gpt4-cache/ is only reusable while the key stays byte-identical to
        # mm_olmo's Gpt4WithCache: model, JSON-encoded prompt, sorted kwargs.
        prompt = 'Here is the caption: "a cat"'
        expected = hashlib.sha256(
            ("gpt-4o-2024-05-13::::" + json.dumps(prompt) + '::::{"temperature": 0}').encode()
        ).hexdigest()
        assert judges._cache_key("gpt-4o-2024-05-13", prompt) == expected


# ---------------------------------------------------------------------------
# 2. Unit tests — parse_consistency_output
# ---------------------------------------------------------------------------


class TestParseConsistencyOutput:
    def test_basic_consistent_inconsistent(self):
        text = "1. Consistent\n2. Inconsistent\n3. Consistent"
        consistent, total = parse_consistency_output(text)
        assert total == 3
        assert consistent == 2

    def test_fuzzy_misspellings(self):
        text = "1. inconsisent\n2. constistent\n3. Inconsistent"
        consistent, total = parse_consistency_output(text)
        assert total == 3
        assert consistent == 1  # only constistent is consistent

    def test_unknown_labels_skipped(self):
        text = "1. Consistent\n2. Not specified\n3. Inconsistent\n4. ambiguous"
        consistent, total = parse_consistency_output(text)
        assert total == 2  # unknown labels skipped
        assert consistent == 1

    def test_empty_input(self):
        consistent, total = parse_consistency_output("")
        assert consistent == 0
        assert total == 0

    def test_case_insensitive(self):
        text = "1. CONSISTENT\n2. INCONSISTENT"
        consistent, total = parse_consistency_output(text)
        assert total == 2
        assert consistent == 1


# ---------------------------------------------------------------------------
# 3. Metric aggregation tests
# ---------------------------------------------------------------------------


def _make_response(result: dict) -> Response:
    """Build a synthetic Response with a pre-filled dense_caption_result."""
    instance = Instance(question="Describe this image.", gold_answer=None, metadata={})
    output = LMOutput(text="A caption.")
    output.metadata = {"dense_caption_result": result}
    return Response(instance=instance, request=None, outputs=[output], scores={})  # type: ignore[arg-type]


def _metric(name: str):
    return next(m for m in _DEFAULT_METRICS if m.name == name)


class TestDenseCaptionMetrics:
    def _valid_result(
        self,
        recall: float = 0.5,
        consistency: float = 0.8,
        num_covered: int = 5,
        num_statements: int = 10,
        num_consistent: int = 8,
        consistency_num_statements: int = 10,
        consistency_valid: bool = True,
    ) -> dict:
        return dict(
            recall=recall,
            recall_at_10=min(num_covered, 10) / min(num_statements, 10),
            num_statements=num_statements,
            num_covered=num_covered,
            recall_valid=True,
            consistency=consistency,
            num_consistent=num_consistent,
            consistency_num_statements=consistency_num_statements,
            consistency_valid=consistency_valid,
        )

    def test_recall_metric_valid(self):
        responses = [
            _make_response(self._valid_result(recall=0.4)),
            _make_response(self._valid_result(recall=0.6)),
        ]
        score = _metric("recall").compute(responses)
        assert abs(score - 50.0) < 1e-6

    def test_recall_metric_filters_invalid(self):
        invalid = dict(
            recall=0.0,
            recall_valid=False,
            consistency=0.0,
            consistency_valid=False,
            num_statements=0,
            num_covered=0,
            recall_at_10=0.0,
            num_consistent=0,
        )
        responses = [_make_response(self._valid_result(recall=0.6)), _make_response(invalid)]
        score = _metric("recall").compute(responses)
        assert abs(score - 60.0) < 1e-6

    def test_consistency_metric(self):
        responses = [
            _make_response(self._valid_result(consistency=0.7)),
            _make_response(self._valid_result(consistency=0.9)),
        ]
        score = _metric("consistency").compute(responses)
        assert abs(score - 80.0) < 1e-6

    def test_recall_at_10(self):
        result = self._valid_result(num_covered=8, num_statements=20)
        # recall_at_10 = min(8,10)/min(20,10) = 8/10 = 0.8
        responses = [_make_response(result)]
        score = _metric("recall_at_10").compute(responses)
        assert abs(score - 80.0) < 1e-6

    def test_num_statements_not_scaled(self):
        # num_statements reports the consistency-side canonical-statement count
        # (mm_olmo's ConsistencyEval.num_statements), not the recall-side count.
        responses = [
            _make_response(self._valid_result(consistency_num_statements=20)),
            _make_response(self._valid_result(consistency_num_statements=10)),
        ]
        score = _metric("num_statements").compute(responses)
        assert abs(score - 15.0) < 1e-6

    def test_avg_metric(self):
        responses = [
            _make_response(self._valid_result(recall=0.4, consistency=0.6)),
            _make_response(self._valid_result(recall=0.6, consistency=0.8)),
        ]
        # mean_recall=0.5, mean_cons=0.7, avg=(0.5+0.7)/2*100=60
        score = DenseCaptionAvgMetric().compute(responses)
        assert abs(score - 60.0) < 1e-6

    def test_empty_responses(self):
        for metric in [
            _metric("recall"),
            _metric("consistency"),
            _metric("recall_at_10"),
            _metric("num_statements"),
            DenseCaptionAvgMetric(),
        ]:
            assert metric.compute([]) == 0.0


# ---------------------------------------------------------------------------
# 4. Scorer unit test with mocked GPT calls
# ---------------------------------------------------------------------------


class TestDenseCaptionJudgeScorerMocked:
    @pytest.mark.anyio
    async def test_scorer_sets_metadata_and_returns_recall(self, tmp_path):
        scorer = DenseCaptionJudgeScorer(
            cache_dir=str(tmp_path),
            cache_only=False,
        )
        instance = Instance(
            question="Describe this image.",
            gold_answer=None,
            metadata={
                "url": "http://example.com/img.jpg",
                "mturk_statements": "1. The sky is blue.\n2. There is a tree.",
                "transcripts": [{"whisperTranscript": "A sunny day with a tree."}],
            },
        )
        output = LMOutput(text="The sky is blue and there is a tall tree.")
        output.metadata = {}

        # GPT recall response: 2 Stated
        # GPT canonical response: "1. Sky is blue.\n2. Tree is present."
        # GPT consistency response: "1. Consistent\n2. Consistent"
        call_responses = [
            "1. Stated\n2. Stated",  # recall check
            "1. Sky is blue.\n2. Tree is present.",  # canonical statements
            "1. Consistent\n2. Consistent",  # consistency check
        ]
        call_iter = iter(call_responses)

        async def fake_gpt(*args: Any, **kwargs: Any) -> str:
            return next(call_iter)

        with patch(
            "olmo_eval.evals.vision.scoring.judges._cached_gpt_call",
            side_effect=fake_gpt,
        ):
            score = await scorer.ascore_with_context(instance, output, ScoringContext())

        assert output.metadata is not None
        result = output.metadata["dense_caption_result"]
        assert result["recall_valid"] is True
        assert result["num_statements"] == 2
        assert result["num_covered"] == 2
        assert abs(result["recall"] - 1.0) < 1e-6
        assert abs(result["consistency"] - 1.0) < 1e-6
        assert abs(score - 1.0) < 1e-6  # primary return is recall

    @pytest.mark.anyio
    async def test_scorer_propagates_gpt_errors(self, tmp_path):
        """GPT/cache failures must reach the runner's structured scoring_errors,
        not silently shrink the valid-example denominator."""
        scorer = DenseCaptionJudgeScorer(cache_dir=str(tmp_path), cache_only=False)
        instance = Instance(
            question="Describe this image.",
            gold_answer=None,
            metadata={
                "url": "http://example.com/img.jpg",
                "mturk_statements": "1. Something.",
                "transcripts": [{"whisperTranscript": "Something."}],
            },
        )
        output = LMOutput(text="A caption.")
        output.metadata = {}

        async def failing_gpt(*args: Any, **kwargs: Any) -> str:
            raise RuntimeError("API error")

        with (
            patch(
                "olmo_eval.evals.vision.scoring.judges._cached_gpt_call",
                side_effect=failing_gpt,
            ),
            pytest.raises(RuntimeError, match="API error"),
        ):
            await scorer.ascore_with_context(instance, output, ScoringContext())

    @pytest.mark.anyio
    async def test_unparseable_judge_response_marks_example_invalid(self, tmp_path):
        """A malformed judge response is a per-example invalid result, not an error."""
        scorer = DenseCaptionJudgeScorer(cache_dir=str(tmp_path), cache_only=False)
        instance = Instance(
            question="Describe this image.",
            gold_answer=None,
            metadata={
                "url": "http://example.com/img.jpg",
                "mturk_statements": "1. Something.",
                "transcripts": [{"whisperTranscript": "Something."}],
            },
        )
        output = LMOutput(text="A caption.")
        output.metadata = {}

        async def garbled_gpt(*args: Any, **kwargs: Any) -> str:
            return "no recognizable verdict lines here"

        with patch(
            "olmo_eval.evals.vision.scoring.judges._cached_gpt_call",
            side_effect=garbled_gpt,
        ):
            score = await scorer.ascore_with_context(instance, output, ScoringContext())

        assert score == 0.0
        result = output.metadata["dense_caption_result"]
        assert result["recall_valid"] is False
        assert result["consistency_valid"] is False
