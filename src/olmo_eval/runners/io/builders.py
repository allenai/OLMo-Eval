"""Functions for building predictions and requests from evaluation responses."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from olmo_eval.common.metrics import Metric
from olmo_eval.common.metrics.predictions import augment_prediction_instance_metrics
from olmo_eval.common.scorers.base import SCORER_RESULTS_KEY
from olmo_eval.common.types import Response, SamplingParams


def build_predictions(scored: Sequence[Any], metrics: Sequence[Metric] = ()) -> list[dict]:
    """Build per-instance predictions from scored responses.

    Args:
        scored: Sequence of scored Response objects
        metrics: Task metrics used to materialize exact per-instance metric keys

    Returns:
        List of prediction dicts suitable for JSONL output
    """
    predictions = []
    for idx, resp in enumerate(scored):
        # Build model_output from LMOutput objects
        model_output = []
        for out in resp.outputs:
            # Get values from metadata (set by backend) or compute from logprobs
            meta = out.metadata or {}
            num_chars = len(out.text) if out.text else 0
            # Logprobs describe the whole generation. If strip_thinking removed a
            # reasoning trace, normalize by the original text, not the stripped answer.
            full_text = meta.get("original_text") or out.text or ""
            num_bytes = len(full_text.encode("utf-8"))
            num_chars_all = len(full_text)

            # Use metadata values if available (from vLLM provider), else compute
            sum_logits: float | None
            num_tokens: int | None
            if "sum_logits" in meta:
                sum_logits = meta["sum_logits"]
                num_tokens = meta.get("num_tokens", len(out.logprobs) if out.logprobs else 0)
            elif out.logprobs:
                sum_logits = sum(t.get("logprob", 0) for t in out.logprobs)
                num_tokens = len(out.logprobs)
            else:
                sum_logits = None
                num_tokens = None

            out_data: dict[str, Any] = {
                "text": out.text,
                "extracted_answer": out.extracted_answer,
            }
            if "is_greedy" in meta:
                out_data["is_greedy"] = meta["is_greedy"]

            # Logprob-derived fields are only meaningful when the backend returned
            # logprobs; omit them otherwise so downstream code can distinguish
            # "unavailable" from a genuine zero.
            if sum_logits is not None:
                out_data["sum_logits"] = sum_logits
                out_data["num_tokens"] = num_tokens
                out_data["num_tokens_all"] = meta.get("num_tokens_all", num_tokens)
                out_data["is_greedy"] = meta.get("is_greedy", False)

                if num_tokens:
                    out_data["logits_per_token"] = sum_logits / num_tokens
                if num_chars_all > 0:
                    out_data["logits_per_char"] = sum_logits / num_chars_all
                if num_bytes > 0:
                    out_data["bits_per_byte"] = -sum_logits / (num_bytes * math.log(2))

            for key in ("finish_reason", "completion_tokens"):
                if key in meta:
                    out_data[key] = meta[key]

            out_data["num_chars"] = num_chars
            if "original_text" in meta:
                out_data["num_chars_all"] = num_chars_all

            sample_metrics: dict[str, dict[str, float]] = {}
            for key, value in meta.items():
                if key.startswith("score:") and isinstance(value, (int, float)):
                    scorer_name = key[6:]
                    sample_metrics.setdefault(scorer_name, {})[scorer_name] = float(value)
            if sample_metrics:
                out_data["sample_metrics"] = sample_metrics

            # Full text before strip_thinking removed the reasoning trace, so the
            # trace length and termination can be audited from the predictions file.
            if "original_text" in meta:
                out_data["original_text"] = meta["original_text"]

            # Include execution result if present
            if "execution_result" in meta:
                out_data["execution_result"] = meta["execution_result"]
            if "scoring_errors" in meta:
                out_data["scoring_errors"] = meta["scoring_errors"]
            if "judge_result" in meta:
                out_data["judge_result"] = meta["judge_result"]
            # Whether the answer was stated in the requested format (scorer-set).
            if "answer_format_correct" in meta:
                out_data["answer_format_correct"] = meta["answer_format_correct"]
            # Per-instance results scorers keep, keyed by scorer name.
            if SCORER_RESULTS_KEY in meta:
                out_data[SCORER_RESULTS_KEY] = meta[SCORER_RESULTS_KEY]

            model_output.append(out_data)

        # Get label from metadata or gold_answer
        label = resp.instance.metadata.get("gold_idx", resp.instance.gold_answer)

        # Build instance_metrics in nested format {metric: {scorer: value}}
        # resp.scores is {scorer_name: value}, we need to convert to nested
        # Since instance-level scores are keyed by scorer, we use scorer as inner key
        instance_metrics: dict[str, dict[str, float]] = {}
        for scorer_name, value in resp.scores.items():
            # Each scorer produces scores under its own name
            # We use scorer_name as both outer and inner key for simplicity
            if scorer_name not in instance_metrics:
                instance_metrics[scorer_name] = {}
            instance_metrics[scorer_name][scorer_name] = value

        # Build prediction (doc and context available in requests.jsonl)
        prediction: dict[str, Any] = {
            "doc_id": idx,
            "native_id": resp.instance.metadata.get("id", f"doc_{idx}"),
            "model_output": model_output,
            "instance_metrics": instance_metrics,
            "label": label,
        }

        # Add judge result if present (LLM-as-judge tasks)
        if "judge_result" in resp.instance.metadata:
            prediction["judge_result"] = resp.instance.metadata["judge_result"]

        # Add final_output text for chat/agent tasks
        if resp.outputs and resp.outputs[0].text:
            prediction["final_output"] = resp.outputs[0].text

        # Add trajectory if present (multi-turn/agent tasks)
        if resp.trajectory is not None:
            prediction["trajectory"] = resp.trajectory.to_dict()

        if metrics:
            augment_prediction_instance_metrics(prediction, metrics, response=resp)

        predictions.append(prediction)

    return predictions


def build_requests(
    instances: Sequence[Any],
    requests: Sequence[Any],
    task_name: str,
    sampling_params: SamplingParams | None = None,
) -> list[dict]:
    """Build per-instance request objects in oe-eval compatible format.

    This produces the same format as oe-eval's *-requests.jsonl files, which
    shows exactly what the model saw during evaluation.

    Args:
        instances: Sequence of Instance objects
        requests: Sequence of LMRequest objects (parallel to instances)
        task_name: Name of the task
        sampling_params: Optional sampling parameters

    Returns:
        List of request dicts suitable for JSONL output with oe-eval compatible schema:
        {
            "request_type": "loglikelihood" | "generate_until" | ...,
            "doc": {
                "query": "...",  # The instance question
                "choices": [...],  # For BPB/MC tasks
                # Plus original metadata
            },
            "request": {
                "context": "...",  # Full prompt (few-shot + current)
                "continuation": "...",  # For loglikelihood: the text being scored
                "perplexity_context": "...",  # Usually same as context
                "stop_sequences": [...],
                "generation_kwargs": {...}
            },
            "idx": 0,
            "task_name": "...",
            "doc_id": 0,
            "native_id": "...",
            "label": ...
        }
    """
    from olmo_eval.common.types import RequestType

    request_list = []

    for idx, (instance, request) in enumerate(zip(instances, requests, strict=True)):
        # Build doc from instance
        doc: dict[str, Any] = {
            "query": instance.question,
            **instance.metadata,
        }

        # Add choices for BPB and MC tasks
        if instance.choices:
            doc["choices"] = list(instance.choices)
        elif request.continuations:
            # For BPB tasks without explicit choices, use continuations
            doc["choices"] = list(request.continuations)

        # Build request object based on request type
        request_dict: dict[str, Any] = {}

        if request.request_type == RequestType.LOGLIKELIHOOD:
            # oe-eval's GenerateUntilAndLoglikelihoodRequest format
            request_dict["context"] = request.prompt
            request_dict["perplexity_context"] = request.prompt
            if request.continuations:
                request_dict["continuation"] = request.continuations[0]
        elif request.request_type == RequestType.COMPLETION:
            # oe-eval's GenerateUntilRequest format
            request_dict["context"] = request.prompt
            if request.continuations:
                request_dict["continuation"] = request.continuations[0]
        elif request.request_type == RequestType.CHAT:
            # Chat format - context is the messages
            request_dict["context"] = list(request.messages)

        # Add generation kwargs
        if sampling_params:
            request_dict["stop_sequences"] = (
                list(sampling_params.stop_sequences) if sampling_params.stop_sequences else []
            )
            request_dict["generation_kwargs"] = {
                "max_gen_toks": sampling_params.max_tokens,
                "do_sample": sampling_params.temperature > 0,
                "temperature": sampling_params.temperature,
            }
            if sampling_params.top_p is not None:
                request_dict["generation_kwargs"]["top_p"] = sampling_params.top_p
        else:
            request_dict["stop_sequences"] = []
            request_dict["generation_kwargs"] = {}

        # Determine request type string (oe-eval naming)
        if request.request_type == RequestType.LOGLIKELIHOOD:
            request_type_str = "loglikelihood"
        elif request.request_type == RequestType.COMPLETION:
            if request.continuations:
                request_type_str = "generate_until_and_loglikelihood"
            else:
                request_type_str = "generate_until"
        elif request.request_type == RequestType.CHAT:
            request_type_str = "generate_until"
        else:
            request_type_str = "unknown"

        # Determine label (ground truth)
        label = instance.metadata.get("gold_idx")
        if label is None and instance.gold_answer is not None:
            label = instance.gold_answer

        request_list.append(
            {
                "request_type": request_type_str,
                "doc": doc,
                "request": request_dict,
                "idx": 0,  # For multi-sample, this would vary
                "task_name": task_name,
                "doc_id": idx,
                "native_id": instance.metadata.get("id", f"doc_{idx}"),
                "label": label,
            }
        )

    return request_list


def build_requests_from_responses(
    responses: Sequence[Response],
    task_name: str,
) -> list[dict]:
    """Build request objects from executed responses.

    Unlike ``build_requests()``, this uses the transformed request that actually
    reached the provider, plus the provider-authored request trace captured at
    execution time. That makes the saved requests JSONL a replayable source of
    truth rather than a prep-time approximation.
    """
    from olmo_eval.common.types import RequestType

    request_list = []

    for idx, response in enumerate(responses):
        instance = response.instance
        request = response.request

        doc: dict[str, Any] = {
            "query": instance.question,
            **instance.metadata,
        }

        if instance.choices:
            doc["choices"] = list(instance.choices)
        elif request.continuations:
            doc["choices"] = list(request.continuations)

        request_dict: dict[str, Any] = {}
        if request.request_type == RequestType.LOGLIKELIHOOD:
            request_dict["context"] = request.prompt
            request_dict["perplexity_context"] = request.prompt
            if request.continuations:
                request_dict["continuation"] = request.continuations[0]
                request_dict["continuations"] = list(request.continuations)
        elif request.request_type == RequestType.COMPLETION:
            request_dict["context"] = request.prompt
            if request.continuations:
                request_dict["continuation"] = request.continuations[0]
                request_dict["continuations"] = list(request.continuations)
        elif request.request_type == RequestType.CHAT:
            request_dict["context"] = list(request.messages)

        if request.continuation_prompts:
            request_dict["continuation_prompts"] = list(request.continuation_prompts)
        if request.system_prompt is not None:
            request_dict["system_prompt"] = request.system_prompt
        if request.max_length is not None:
            request_dict["max_length"] = request.max_length
        if request.tools:
            request_dict["tools"] = [tool.to_openai() for tool in request.tools]

        request_trace = response.request_trace or {}
        if request_trace:
            request_dict["provider_request"] = request_trace

        generation_kwargs = request_trace.get("generation_kwargs")
        if generation_kwargs:
            request_dict["generation_kwargs"] = dict(generation_kwargs)
        else:
            request_dict["generation_kwargs"] = {}

        stop_sequences = request_trace.get("stop_sequences")
        request_dict["stop_sequences"] = list(stop_sequences or [])

        if request.request_type == RequestType.LOGLIKELIHOOD:
            request_type_str = "loglikelihood"
        elif request.request_type == RequestType.COMPLETION:
            if request.continuations:
                request_type_str = "generate_until_and_loglikelihood"
            else:
                request_type_str = "generate_until"
        elif request.request_type == RequestType.CHAT:
            request_type_str = "generate_until"
        else:
            request_type_str = "unknown"

        label = instance.metadata.get("gold_idx")
        if label is None and instance.gold_answer is not None:
            label = instance.gold_answer

        request_list.append(
            {
                "request_type": request_type_str,
                "doc": doc,
                "request": request_dict,
                "idx": 0,
                "task_name": task_name,
                "doc_id": idx,
                "native_id": instance.metadata.get("id", f"doc_{idx}"),
                "label": label,
            }
        )

    return request_list
