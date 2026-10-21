"""Tests for external evaluation results."""

from olmo_eval.evals.external.result import ExternalEvalResult


class TestNumInstances:
    def test_counts_predictions(self):
        result = ExternalEvalResult(
            name="tau2",
            metadata={"num_tasks": 50},
            predictions=[{"id": 1}, {"id": 2}, {"id": 3}],
        )

        assert result.num_instances == 3

    def test_falls_back_to_num_tasks(self):
        result = ExternalEvalResult(name="terminal_bench", metadata={"num_tasks": 80})

        assert result.num_instances == 80

    def test_defaults_to_zero(self):
        assert ExternalEvalResult(name="scicode").num_instances == 0
        assert ExternalEvalResult.from_error("asta", "boom").num_instances == 0
