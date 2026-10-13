"""Tests for olmo_eval.core.metrics module."""

import math

import pytest

from olmo_eval.common.metrics import (
    AccuracyMetric,
    BPBMetricByteAvg,
    BPBMetricInstanceAvg,
    LogprobMCAccuracyMetric,
    LogprobPerCharMCAccuracyMetric,
    NGramCopyingBPBMetricByteAvg,
    SafetyErrorMetric,
)
from olmo_eval.common.scorers import MultipleChoiceScorer
from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, Response


class TestAccuracyMetric:
    """Tests for AccuracyMetric."""

    def _make_response(self, score: float, scorer_name: str = "exact_match") -> Response:
        """Helper to create a response with a score."""
        return Response(
            instance=Instance(question="Q", gold_answer="A"),
            request=LMRequest(request_type=RequestType.COMPLETION, prompt="Q"),
            outputs=[LMOutput(text="A")],
            scores={scorer_name: score},
        )

    def test_accuracy_all_correct(self):
        """Test accuracy with all correct answers."""
        metric = AccuracyMetric()
        responses = [
            self._make_response(1.0),
            self._make_response(1.0),
            self._make_response(1.0),
        ]

        accuracy = metric.compute(responses)

        assert accuracy == 1.0

    def test_accuracy_all_incorrect(self):
        """Test accuracy with all incorrect answers."""
        metric = AccuracyMetric()
        responses = [
            self._make_response(0.0),
            self._make_response(0.0),
            self._make_response(0.0),
        ]

        accuracy = metric.compute(responses)

        assert accuracy == 0.0

    def test_accuracy_mixed(self):
        """Test accuracy with mixed results."""
        metric = AccuracyMetric()
        responses = [
            self._make_response(1.0),
            self._make_response(0.0),
            self._make_response(1.0),
        ]

        accuracy = metric.compute(responses)

        assert accuracy == pytest.approx(2 / 3)

    def test_accuracy_empty_responses(self):
        """Test accuracy with empty response list."""
        metric = AccuracyMetric()

        accuracy = metric.compute([])

        assert accuracy == 0.0

    def test_accuracy_single_response(self):
        """Test accuracy with single response."""
        metric = AccuracyMetric()
        responses = [self._make_response(1.0)]

        accuracy = metric.compute(responses)

        assert accuracy == 1.0

    def test_accuracy_custom_scorer(self):
        """Test accuracy with custom scorer type."""
        metric = AccuracyMetric(scorer=MultipleChoiceScorer)
        responses = [
            self._make_response(1.0, "multiple_choice"),
            self._make_response(0.0, "multiple_choice"),
        ]

        accuracy = metric.compute(responses)

        assert accuracy == 0.5

    def test_accuracy_missing_scorer(self):
        """Test accuracy when scorer name not in scores dict."""
        metric = AccuracyMetric(scorer=MultipleChoiceScorer)
        responses = [
            self._make_response(1.0, "exact_match"),
            self._make_response(1.0, "exact_match"),
        ]

        accuracy = metric.compute(responses)

        # Missing scorer defaults to 0.0
        assert accuracy == 0.0

    def test_accuracy_name(self):
        """Test metric name."""
        metric = AccuracyMetric()
        assert metric.name == "accuracy"

        custom = AccuracyMetric(name="custom_accuracy")
        assert custom.name == "custom_accuracy"

    def test_accuracy_partial_scores(self):
        """Test accuracy with partial scores (not just 0 or 1)."""
        metric = AccuracyMetric()
        responses = [
            self._make_response(0.5),
            self._make_response(0.5),
        ]

        accuracy = metric.compute(responses)

        assert accuracy == 0.5


class TestBPBMetricInstanceAvg:
    """Tests for BPBMetricInstanceAvg with gold selection."""

    def _make_output_with_logprobs(self, text: str, logprobs: list[float]) -> LMOutput:
        """Helper to create an LMOutput with logprobs.

        Each logprob entry corresponds to one character of the text for simplicity.
        """
        # Create logprob entries with bytes (one char per token for test simplicity)
        entries = []
        for i, lp in enumerate(logprobs):
            char = text[i] if i < len(text) else ""
            entries.append(
                {
                    "token": char,
                    "logprob": lp,
                    "bytes": list(char.encode("utf-8")),
                }
            )
        return LMOutput(text=text, logprobs=entries)

    def _make_response(
        self,
        outputs: list[LMOutput],
        gold_idx: int | None = None,
    ) -> Response:
        """Helper to create a response with multiple outputs."""
        metadata = {"gold_idx": gold_idx} if gold_idx is not None else {}
        return Response(
            instance=Instance(question="Q", gold_answer="A", metadata=metadata),
            request=LMRequest(request_type=RequestType.COMPLETION, prompt="Q"),
            outputs=outputs,
        )

    def test_bpb_single_output(self):
        """Test BPB with single output."""
        metric = BPBMetricInstanceAvg()
        # Text "ab" = 2 bytes, logprobs sum to -2.0
        # BPB = -(-2.0) / (2 * log(2)) = 2.0 / 1.386 ≈ 1.443
        output = self._make_output_with_logprobs("ab", [-1.0, -1.0])
        responses = [self._make_response([output])]

        bpb = metric.compute(responses)

        expected = 2.0 / (2 * math.log(2))
        assert bpb == pytest.approx(expected)

    def test_bpb_multiple_outputs_selects_gold(self):
        """Test that BPB selects the gold/correct output."""
        metric = BPBMetricInstanceAvg()

        # Create 3 outputs with different BPB values
        # Output 0: "a" (1 byte), logprob -1.0 -> BPB = 1/(1*log2) ≈ 1.443
        # Output 1: "ab" (2 bytes), logprob -4.0 -> BPB = 4/(2*log2) ≈ 2.885 (gold)
        # Output 2: "abc" (3 bytes), logprob -3.0 -> BPB = 3/(3*log2) ≈ 1.443
        outputs = [
            self._make_output_with_logprobs("a", [-1.0]),
            self._make_output_with_logprobs("ab", [-2.0, -2.0]),
            self._make_output_with_logprobs("abc", [-1.0, -1.0, -1.0]),
        ]
        responses = [self._make_response(outputs, gold_idx=1)]

        bpb = metric.compute(responses)

        # Should use output 1 (gold_idx=1): 4.0 / (2 * log(2))
        expected = 4.0 / (2 * math.log(2))
        assert bpb == pytest.approx(expected)

    def test_bpb_multiple_outputs_without_gold_idx_uses_first(self):
        """Test that BPB falls back to first output without gold_idx."""
        metric = BPBMetricInstanceAvg()

        outputs = [
            self._make_output_with_logprobs("a", [-1.0]),
            self._make_output_with_logprobs("ab", [-2.0, -2.0]),
        ]
        # No gold_idx specified
        responses = [self._make_response(outputs, gold_idx=None)]

        bpb = metric.compute(responses)

        # Should use output 0 (first): 1.0 / (1 * log(2))
        expected = 1.0 / (1 * math.log(2))
        assert bpb == pytest.approx(expected)

    def test_bpb_empty_responses(self):
        """Test BPB with empty response list."""
        metric = BPBMetricInstanceAvg()

        bpb = metric.compute([])

        assert bpb == 0.0

    def test_bpb_aggregates_across_responses(self):
        """Test that BPB uses simple arithmetic mean across responses."""
        metric = BPBMetricInstanceAvg()

        # Response 1: "a" (1 byte), logprob -1.0 → BPB = 1.0 / (1 * log(2))
        # Response 2: "ab" (2 bytes), logprob -4.0 (sum of -2.0, -2.0) → BPB = 4.0 / (2 * log(2))
        # Simple mean: (bpb1 + bpb2) / 2
        responses = [
            self._make_response([self._make_output_with_logprobs("a", [-1.0])]),
            self._make_response([self._make_output_with_logprobs("ab", [-2.0, -2.0])]),
        ]

        bpb = metric.compute(responses)

        bpb1 = 1.0 / (1 * math.log(2))
        bpb2 = 4.0 / (2 * math.log(2))
        expected = (bpb1 + bpb2) / 2
        assert bpb == pytest.approx(expected)

    def test_bpb_no_logprobs_returns_zero(self):
        """Test that output without logprobs returns 0."""
        metric = BPBMetricInstanceAvg()

        output = LMOutput(text="test", logprobs=None)
        responses = [self._make_response([output])]

        bpb = metric.compute(responses)

        assert bpb == 0.0


class TestBPBMetricByteAvg:
    """Tests for BPBMetricByteAvg (byte-weighted aggregation)."""

    def _make_output_with_logprobs(self, text: str, logprobs: list[float]) -> LMOutput:
        entries = []
        for i, lp in enumerate(logprobs):
            char = text[i] if i < len(text) else ""
            entries.append(
                {
                    "token": char,
                    "logprob": lp,
                    "bytes": list(char.encode("utf-8")),
                }
            )
        return LMOutput(text=text, logprobs=entries)

    def _make_response(
        self,
        outputs: list[LMOutput],
        gold_idx: int | None = None,
    ) -> Response:
        metadata = {"gold_idx": gold_idx} if gold_idx is not None else {}
        return Response(
            instance=Instance(question="Q", gold_answer="A", metadata=metadata),
            request=LMRequest(request_type=RequestType.COMPLETION, prompt="Q"),
            outputs=outputs,
        )

    def test_byte_avg_single_output(self):
        """Single output should match instance-avg (only one instance)."""
        metric = BPBMetricByteAvg()
        output = self._make_output_with_logprobs("ab", [-1.0, -1.0])
        responses = [self._make_response([output])]

        bpb = metric.compute(responses)

        expected = 2.0 / (2 * math.log(2))
        assert bpb == pytest.approx(expected)

    def test_byte_avg_weights_by_byte_count(self):
        """BPBMetricByteAvg weights each instance's BPB by its byte count."""
        metric = BPBMetricByteAvg()

        # Response 1: "a" (1 byte), logprob -1.0 → BPB = 1.0 / (1 * log2) = 1/log2
        # Response 2: "ab" (2 bytes), logprobs [-2.0, -2.0] → BPB = 4.0 / (2 * log2) = 2/log2
        # Byte-weighted: (1/log2 * 1 + 2/log2 * 2) / (1 + 2) = 5 / (3 * log2)
        responses = [
            self._make_response([self._make_output_with_logprobs("a", [-1.0])]),
            self._make_response([self._make_output_with_logprobs("ab", [-2.0, -2.0])]),
        ]

        bpb = metric.compute(responses)

        expected = 5.0 / (3 * math.log(2))
        assert bpb == pytest.approx(expected)

    def test_byte_avg_empty_responses(self):
        metric = BPBMetricByteAvg()
        assert metric.compute([]) == 0.0

    def test_byte_avg_no_logprobs_returns_zero(self):
        metric = BPBMetricByteAvg()
        output = LMOutput(text="test", logprobs=None)
        responses = [self._make_response([output])]
        assert metric.compute(responses) == 0.0


class TestLogprobMCAccuracyMetric:
    """Tests for LogprobMCAccuracyMetric."""

    def _make_output(self, text: str, logprobs: list[float] | None) -> LMOutput:
        if logprobs is None:
            return LMOutput(text=text, logprobs=None)
        entries = [{"token": f"t{i}", "logprob": lp} for i, lp in enumerate(logprobs)]
        return LMOutput(text=text, logprobs=entries)

    def _make_response(self, outputs: list[LMOutput], gold_idx: int) -> Response:
        return Response(
            instance=Instance(
                question="Q",
                gold_answer="A",
                choices=("a", "b", "c"),
                metadata={"gold_idx": gold_idx},
            ),
            request=LMRequest(request_type=RequestType.LOGLIKELIHOOD, prompt="Q"),
            outputs=outputs,
        )

    def test_selects_highest_logprob(self):
        metric = LogprobMCAccuracyMetric()
        outputs = [
            self._make_output("a", [-5.0, -5.0]),
            self._make_output("b", [-1.0, -1.0]),
            self._make_output("c", [-3.0, -3.0]),
        ]
        responses = [self._make_response(outputs, gold_idx=1)]
        assert metric.compute(responses) == 1.0

    def test_incorrect_prediction(self):
        metric = LogprobMCAccuracyMetric()
        outputs = [
            self._make_output("a", [-1.0]),
            self._make_output("b", [-5.0]),
            self._make_output("c", [-3.0]),
        ]
        responses = [self._make_response(outputs, gold_idx=1)]
        assert metric.compute(responses) == 0.0

    def test_missing_logprobs_treated_as_neg_inf(self):
        """Outputs with missing logprobs get -inf, losing to any finite sum."""
        metric = LogprobMCAccuracyMetric()
        outputs = [
            self._make_output("a", None),
            self._make_output("b", [-1.0]),
            self._make_output("c", None),
        ]
        responses = [self._make_response(outputs, gold_idx=1)]
        assert metric.compute(responses) == 1.0

    def test_empty_logprobs_treated_as_neg_inf(self):
        """Outputs with empty logprobs list get -inf, losing to any finite sum."""
        metric = LogprobMCAccuracyMetric()
        outputs = [
            self._make_output("a", []),
            self._make_output("b", [-2.0]),
            self._make_output("c", []),
        ]
        responses = [self._make_response(outputs, gold_idx=1)]
        assert metric.compute(responses) == 1.0

    def test_empty_responses(self):
        metric = LogprobMCAccuracyMetric()
        assert metric.compute([]) == 0.0

    def test_missing_gold_idx_skipped(self):
        metric = LogprobMCAccuracyMetric()
        response = Response(
            instance=Instance(question="Q", gold_answer="A", metadata={}),
            request=LMRequest(request_type=RequestType.LOGLIKELIHOOD, prompt="Q"),
            outputs=[self._make_output("a", [-1.0])],
        )
        assert metric.compute([response]) == 0.0


class TestLogprobPerCharMCAccuracyMetric:
    """Tests for LogprobPerCharMCAccuracyMetric."""

    def _make_output(self, text: str, logprobs: list[float] | None) -> LMOutput:
        if logprobs is None:
            return LMOutput(text=text, logprobs=None)
        entries = [{"token": f"t{i}", "logprob": lp} for i, lp in enumerate(logprobs)]
        return LMOutput(text=text, logprobs=entries)

    def _make_response(self, outputs: list[LMOutput], gold_idx: int) -> Response:
        return Response(
            instance=Instance(
                question="Q",
                gold_answer="A",
                choices=("a", "bb", "ccc"),
                metadata={"gold_idx": gold_idx},
            ),
            request=LMRequest(request_type=RequestType.LOGLIKELIHOOD, prompt="Q"),
            outputs=outputs,
        )

    def test_normalizes_by_char_length(self):
        """Shorter text with same total logprob should have higher per-char score."""
        metric = LogprobPerCharMCAccuracyMetric()
        # "a" (1 char): total=-2, per_char=-2/1=-2
        # "bb" (2 chars): total=-2, per_char=-2/2=-1 (highest)
        # "ccc" (3 chars): total=-6, per_char=-6/3=-2
        outputs = [
            self._make_output("a", [-2.0]),
            self._make_output("bb", [-1.0, -1.0]),
            self._make_output("ccc", [-2.0, -2.0, -2.0]),
        ]
        responses = [self._make_response(outputs, gold_idx=1)]
        assert metric.compute(responses) == 1.0

    def test_missing_logprobs_treated_as_neg_inf(self):
        """Outputs with missing logprobs get -inf per-char score, losing to any finite score."""
        metric = LogprobPerCharMCAccuracyMetric()
        outputs = [
            self._make_output("a", None),
            self._make_output("bb", [-1.0]),
            self._make_output("ccc", None),
        ]
        responses = [self._make_response(outputs, gold_idx=1)]
        assert metric.compute(responses) == 1.0

    def test_empty_responses(self):
        metric = LogprobPerCharMCAccuracyMetric()
        assert metric.compute([]) == 0.0


class TestSafetyErrorMetric:
    """Tests for SafetyErrorMetric."""

    def _make_response(
        self, score: float, is_parsing_error: bool | None, scorer_name: str = "safety_judge"
    ) -> Response:
        """Helper to create a response with a score."""
        if is_parsing_error is not None:
            return Response(
                instance=Instance(
                    question="Q", gold_answer="A", metadata={"is_parsing_error": is_parsing_error}
                ),
                request=LMRequest(request_type=RequestType.CHAT, prompt="Q"),
                outputs=[LMOutput(text="A")],
                scores={scorer_name: score},
            )
        else:
            return Response(
                instance=Instance(question="Q", gold_answer="A", metadata={}),
                request=LMRequest(request_type=RequestType.CHAT, prompt="Q"),
                outputs=[LMOutput(text="A")],
                scores={scorer_name: score},
            )

    def test_error_all_correct_no_errors(self):
        """Test error metric with all correct answers and no errors."""
        metric = SafetyErrorMetric()
        responses = [
            self._make_response(1.0, False),
            self._make_response(1.0, False),
            self._make_response(1.0, False),
        ]
        errors = metric.compute(responses)

        assert errors == 0

    def test_error_all_correct_all_errors(self):
        """Test error metric with all correct answers and all errors."""
        metric = SafetyErrorMetric()
        responses = [
            self._make_response(1.0, True),
            self._make_response(1.0, True),
            self._make_response(1.0, True),
        ]
        errors = metric.compute(responses)

        assert errors == 3

    def test_error_mixed(self):
        """Test errors with mixed results."""
        metric = SafetyErrorMetric()
        responses = [
            self._make_response(1.0, True),
            self._make_response(0.0, True),
            self._make_response(1.0, False),
        ]
        errors = metric.compute(responses)

        assert errors == 2

    def test_errors_empty_responses(self):
        """Test errors with empty response list."""
        metric = SafetyErrorMetric()
        errors = metric.compute([])

        assert errors == 0.0

    def test_error_single_response(self):
        """Test errors with single response."""
        metric = SafetyErrorMetric()
        responses = [self._make_response(0.0, True)]
        errors = metric.compute(responses)

        assert errors == 1

    def test_error_name(self):
        """Test metric name."""
        metric = SafetyErrorMetric()
        assert metric.name == "parsing_error"

    def test_fewer_errors_is_better(self):
        """Test that fewer parsing errors count as better."""
        assert SafetyErrorMetric().higher_is_better() is False

    def test_missing_errors_instance_level(self):
        """Test management of errors when is_parsing_error is missing."""
        metric = SafetyErrorMetric()
        response = self._make_response(1.0, None)
        errors = metric.compute_instance(response)

        assert errors == 0

    def test_error_instance_level_error(self):
        """Test management of errors when is_parsing_error is missing."""
        metric = SafetyErrorMetric()
        response = self._make_response(1.0, True)
        errors = metric.compute_instance(response)

        assert errors == 1

    def test_error_instance_level_noerror(self):
        """Test management of errors when is_parsing_error is missing."""
        metric = SafetyErrorMetric()
        response = self._make_response(1.0, False)
        errors = metric.compute_instance(response)

        assert errors == 0

    def test_missing_errors_summary(self):
        """Test compute for errors when is_parsing_error is missing."""
        metric = SafetyErrorMetric()
        responses = [
            self._make_response(1.0, False),
            self._make_response(0.0, False),
            self._make_response(0.0, None),
            self._make_response(1.0, None),
        ]
        errors = metric.compute(responses)

        assert errors == 0


class TestNGramCopyingBPBMetric:
    """Tests for NGramCopyingBPBMetricByteAvg."""

    def _make_response(self, tokens: list[str], logprobs: list[float]) -> Response:
        """Build a single-output, teacher-forced (loglikelihood) Response from tokens."""
        entries = [
            {"token": tok, "logprob": lp, "bytes": list(tok.encode("utf-8"))}
            for tok, lp in zip(tokens, logprobs, strict=True)
        ]
        text = "".join(tokens)
        output = LMOutput(text=text, logprobs=entries)
        return Response(
            instance=Instance(question="", gold_answer=text),
            request=LMRequest(request_type=RequestType.LOGLIKELIHOOD, prompt=""),
            outputs=[output],
        )

    def test_masks_repeated_positions_only(self):
        """`a b c a b` at k=1: only positions 3,4 ('a','b') are masked in."""
        metric = NGramCopyingBPBMetricByteAvg(k=1)
        response = self._make_response(["a", "b", "c", "a", "b"], [-1.0, -1.0, -1.0, -1.0, -1.0])

        # Masked positions 3,4: total_logprob=-2.0, total_bytes=2.
        expected = 2.0 / (2 * math.log(2))
        assert metric.compute_instance(response) == pytest.approx(expected)
        assert metric.compute([response]) == pytest.approx(expected)

    def test_documents_without_repeats_contribute_nothing(self):
        """A document with zero positions meeting the threshold doesn't shift the pooled value."""
        metric = NGramCopyingBPBMetricByteAvg(k=1)
        no_repeats = self._make_response(["p", "q", "r", "s"], [-1.0, -1.0, -1.0, -1.0])
        with_repeats = self._make_response(
            ["a", "b", "c", "a", "b"], [-1.0, -1.0, -1.0, -1.0, -1.0]
        )

        assert metric.compute_instance(no_repeats) is None
        # Pooling with a zero-copy document should equal pooling without it.
        expected = 2.0 / (2 * math.log(2))
        assert metric.compute([no_repeats, with_repeats]) == pytest.approx(expected)
        assert metric.compute([no_repeats]) == 0.0

    def test_weights_by_masked_byte_count(self):
        """Pooling weights each document's contribution by its masked byte count.

        Doc A = "a b a": mask (k=1) is [F,F,T]; masked logprob=-2.0, bytes=1.
        Doc B = "p p p p": mask (k=1) is [F,T,T,T]; masked logprob=-3.0, bytes=3.
        """
        metric = NGramCopyingBPBMetricByteAvg(k=1)
        doc_a = self._make_response(["a", "b", "a"], [-1.0, -1.0, -2.0])
        doc_b = self._make_response(["p", "p", "p", "p"], [-1.0, -1.0, -1.0, -1.0])

        bpb_a = 2.0 / (1 * math.log(2))
        bpb_b = 3.0 / (3 * math.log(2))
        assert metric.compute_instance(doc_a) == pytest.approx(bpb_a)
        assert metric.compute_instance(doc_b) == pytest.approx(bpb_b)

        # Pools logprobs/bytes across documents before dividing once, rather than
        # averaging the two per-document values unweighted -- doc B's larger masked
        # span (3 bytes vs. 1) pulls the pooled result further toward its own bpb.
        assert metric.compute([doc_a, doc_b]) == pytest.approx(5.0 / (4 * math.log(2)))
