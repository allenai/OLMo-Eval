from __future__ import annotations

import math

from olmo_eval.common.metrics import LogprobPerTokenMCAccuracyMetric
from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, Response
from olmo_eval.runners.io.builders import build_predictions


def test_build_predictions_includes_scoring_errors() -> None:
    response = Response(
        instance=Instance(question="Q", gold_answer="A"),
        request=LMRequest(request_type=RequestType.COMPLETION, prompt="Q"),
        outputs=[
            LMOutput(
                text="out",
                extracted_answer="out",
                metadata={
                    "scoring_errors": {
                        "code_exec": {
                            "phase": "execution",
                            "type": "RuntimeError",
                            "message": "boom",
                        }
                    },
                    "score:code_exec": 0.0,
                },
            )
        ],
        scores={"code_exec": 0.0},
    )

    predictions = build_predictions([response])

    assert predictions[0]["instance_metrics"] == {"code_exec": {"code_exec": 0.0}}
    assert predictions[0]["model_output"][0]["scoring_errors"] == {
        "code_exec": {
            "phase": "execution",
            "type": "RuntimeError",
            "message": "boom",
        }
    }
    assert predictions[0]["model_output"][0]["sample_metrics"] == {"code_exec": {"code_exec": 0.0}}
    assert "is_greedy" not in predictions[0]["model_output"][0]


def test_build_predictions_includes_per_output_judge_result() -> None:
    judge_result = {
        "parsed_outcome": None,
        "parse_error": True,
        "raw_judge_response": "unparseable",
    }
    response = Response(
        instance=Instance(question="Q", gold_answer="A"),
        request=LMRequest(request_type=RequestType.COMPLETION, prompt="Q"),
        outputs=[LMOutput(text="out", metadata={"judge_result": judge_result})],
        scores={},
    )

    output = build_predictions([response])[0]["model_output"][0]

    assert output["judge_result"] == judge_result


def test_build_predictions_includes_sample_metrics_for_multiple_outputs() -> None:
    response = Response(
        instance=Instance(question="Q", gold_answer="A"),
        request=LMRequest(request_type=RequestType.COMPLETION, prompt="Q"),
        outputs=[
            LMOutput(
                text="wrong",
                extracted_answer="B",
                metadata={"score:exact_match": 0.0},
            ),
            LMOutput(
                text="right",
                extracted_answer="A",
                metadata={"score:exact_match": 1.0},
            ),
        ],
        scores={"exact_match": 0.0},
    )

    predictions = build_predictions([response])

    assert predictions[0]["model_output"][0]["sample_metrics"] == {
        "exact_match": {"exact_match": 0.0}
    }
    assert predictions[0]["model_output"][1]["sample_metrics"] == {
        "exact_match": {"exact_match": 1.0}
    }


def test_build_predictions_omits_logprob_fields_when_unavailable() -> None:
    response = Response(
        instance=Instance(question="Q", gold_answer="A"),
        request=LMRequest(request_type=RequestType.CHAT, prompt="Q"),
        outputs=[LMOutput(text="hello", extracted_answer="hello")],
        scores={},
    )

    output = build_predictions([response])[0]["model_output"][0]

    for field in (
        "sum_logits",
        "num_tokens",
        "num_tokens_all",
        "is_greedy",
        "logits_per_token",
        "logits_per_char",
        "bits_per_byte",
    ):
        assert field not in output, f"{field} should be omitted when logprobs unavailable"
    assert output["num_chars"] == len("hello")


def test_build_predictions_includes_known_is_greedy() -> None:
    response = Response(
        instance=Instance(question="Q", gold_answer="A"),
        request=LMRequest(request_type=RequestType.COMPLETION, prompt="Q"),
        outputs=[
            LMOutput(
                text="out",
                metadata={"is_greedy": True},
            )
        ],
        scores={},
    )

    predictions = build_predictions([response])

    assert predictions[0]["model_output"][0]["is_greedy"] is True


def test_build_predictions_materializes_exact_mc_accuracy_metric_keys() -> None:
    response = Response(
        instance=Instance(
            question="Q",
            choices=("A", "B"),
            metadata={"gold_idx": 1},
        ),
        request=LMRequest(request_type=RequestType.LOGLIKELIHOOD, prompt="Q"),
        outputs=[
            LMOutput(
                text="A",
                logprobs=[{"token": "A", "logprob": -4.0}],
            ),
            LMOutput(
                text="B",
                logprobs=[{"token": "B", "logprob": -1.0}],
            ),
        ],
        scores={"logprob": -1.0},
    )

    predictions = build_predictions([response], metrics=(LogprobPerTokenMCAccuracyMetric(),))

    assert predictions[0]["instance_metrics"]["logprob"]["logprob"] == -1.0
    assert predictions[0]["instance_metrics"]["accuracy"]["logprob"] == 1.0


def test_build_predictions_includes_original_text_when_present() -> None:
    response = Response(
        instance=Instance(question="Q", gold_answer="A"),
        request=LMRequest(request_type=RequestType.COMPLETION, prompt="Q"),
        outputs=[
            LMOutput(
                text="A",
                extracted_answer="A",
                metadata={"original_text": "<think>reasoning</think>A"},
            ),
            LMOutput(text="B", extracted_answer="B"),
        ],
    )

    predictions = build_predictions([response])

    outputs = predictions[0]["model_output"]
    assert outputs[0]["original_text"] == "<think>reasoning</think>A"
    assert "original_text" not in outputs[1]


def test_build_predictions_normalizes_logprobs_by_original_text() -> None:
    original = "<think>reasoning</think>4"  # 25 chars, 25 bytes
    response = Response(
        instance=Instance(question="Q", gold_answer="4"),
        request=LMRequest(request_type=RequestType.COMPLETION, prompt="Q"),
        outputs=[
            LMOutput(
                text="4",
                extracted_answer="4",
                logprobs=[{"token": "x", "logprob": -1.0} for _ in range(5)],
                metadata={"original_text": original},
            )
        ],
    )

    output = build_predictions([response])[0]["model_output"][0]

    assert output["num_chars"] == 1
    assert output["num_chars_all"] == len(original)
    assert output["logits_per_char"] == -5.0 / len(original)
    assert output["bits_per_byte"] == 5.0 / (len(original.encode("utf-8")) * math.log(2))


def test_build_predictions_preserves_termination_without_logprobs() -> None:
    response = Response(
        instance=Instance(question="Q", gold_answer="A"),
        request=LMRequest(request_type=RequestType.COMPLETION, prompt="Q"),
        outputs=[
            LMOutput(text="", metadata={"finish_reason": "length", "completion_tokens": 32768})
        ],
    )
    output = build_predictions([response])[0]["model_output"][0]
    assert output["finish_reason"] == "length"
    assert output["completion_tokens"] == 32768
    assert output["text"] == ""
    assert "sum_logits" not in output


def test_build_predictions_includes_answer_format_correct() -> None:
    response = Response(
        instance=Instance(question="Q", gold_answer="B"),
        request=LMRequest(
            request_type=RequestType.CHAT, messages=({"role": "user", "content": "Q"},)
        ),
        outputs=[LMOutput(text="the answer is B", metadata={"answer_format_correct": 0.5})],
        scores={"exact_match": 1.0},
    )

    output = build_predictions([response])[0]["model_output"][0]

    assert output["answer_format_correct"] == 0.5
