"""Diagnostics must not turn absent telemetry into a clean evaluation."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/internal/screen_diagnostics.py"
spec = importlib.util.spec_from_file_location("screen_diagnostics", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_counts_caps_empty_errors_and_missing():
    rows = [
        {"doc_id": 0, "model_output": [{"text": "", "finish_reason": "length"}]},
        {"doc_id": 1, "model_output": [{"text": "answer", "num_tokens_all": 32}]},
        {
            "doc_id": 2,
            "model_output": [
                {
                    "text": "bad code",
                    "finish_reason": "stop",
                    "scoring_errors": {"grader": "unavailable"},
                }
            ],
        },
    ]
    result = module.summarize(rows, expected=4, cap=32)
    assert result["cap_hit"] == 2
    assert result["empty"] == result["errored"] == result["missing_items"] == 1
    assert not result["diagnostics_complete"]


def test_unknown_telemetry_is_not_zero_caps():
    result = module.summarize([{"doc_id": 0, "model_output": [{"text": "answer"}]}], 1, 32)
    assert result["cap_unknown"] == 1
    assert not result["diagnostics_complete"]


def test_wrong_candidate_is_not_infrastructure_error():
    output = {"text": "wrong", "finish_reason": "stop", "execution_result": {"success": False}}
    result = module.summarize([{"doc_id": 0, "model_output": [output]}], 1, 32)
    assert result["errored"] == 0
    assert result["diagnostics_complete"]


def test_refuses_duplicate_items_and_multiple_samples():
    row = {"doc_id": 0, "model_output": [{"text": "x"}]}
    with pytest.raises(ValueError, match="duplicate"):
        module.summarize([row, row], 2, 32)
    with pytest.raises(ValueError, match="K=1"):
        module.summarize([{"doc_id": 0, "model_output": [{}, {}]}], 1, 32)
