"""Token counts that inference outputs report, shared by the runner and the upload client.

Both callers pass one mapping per output: LMOutput.metadata in the runner, or a
predictions model_output entry in the upload client. They use the same key names.

Rules:
- Log-likelihood outputs (one per continuation) report ``num_tokens_all``, the
  context plus continuation tokens sent to the model, and ``num_tokens``, the
  scored continuation tokens. Some providers set ``num_tokens_all`` equal to
  ``num_tokens`` (vllm_server, huggingface), which leaves out the context. A
  value that is not larger than ``num_tokens`` therefore counts as unreported.
  Log-likelihood requests generate nothing, so their completion count is 0.
- Generation outputs report ``prompt_tokens`` and ``completion_tokens`` when the
  provider returns usage. Every sample of a request shares one prompt, so the
  prompt count is read from the first output and completion counts are summed.
- A count that is missing makes the result None (unknown), never 0, so totals
  built from these values are not silently low.
- Agent results with more than one turn report only the last model call, so
  both counts are None for them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def nonneg_int(value: Any) -> int | None:
    """Return value as a non-negative int, or None when it is not one."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, float) and value.is_integer() and value >= 0:
        return int(value)
    return None


def _sum_or_none(counts: Sequence[int | None]) -> int | None:
    total = 0
    for count in counts:
        if count is None:
            return None
        total += count
    return total


def loglikelihood_prompt_tokens(output: Mapping[str, Any]) -> int | None:
    """Context plus continuation tokens one log-likelihood output reports, or None."""
    total = nonneg_int(output.get("num_tokens_all"))
    scored = nonneg_int(output.get("num_tokens"))
    if total is None or scored is None or total <= scored:
        return None
    return total


def request_token_usage(
    outputs: Sequence[Mapping[str, Any]], *, loglikelihood: bool
) -> tuple[int | None, int | None]:
    """(prompt_tokens, completion_tokens) for one request, each None when unknown.

    A request with no outputs reports nothing and returns (0, 0).
    """
    if not outputs:
        return 0, 0
    if loglikelihood:
        return _sum_or_none([loglikelihood_prompt_tokens(output) for output in outputs]), 0
    turns = nonneg_int(outputs[0].get("num_turns"))
    if turns is not None and turns > 1:
        return None, None
    prompt = nonneg_int(outputs[0].get("prompt_tokens"))
    completion = _sum_or_none([nonneg_int(output.get("completion_tokens")) for output in outputs])
    return prompt, completion


def instance_prompt_tokens(outputs: Sequence[Mapping[str, Any]]) -> int | None:
    """Prompt tokens for one predictions record, read from its first output.

    Generation outputs carry ``prompt_tokens``. Log-likelihood outputs are
    recognized by having ``num_tokens_all`` without the generation keys
    ``finish_reason`` and ``completion_tokens``; for multiple choice the first
    answer choice's request stands for the instance.
    """
    if not outputs:
        return None
    first = outputs[0]
    prompt = nonneg_int(first.get("prompt_tokens"))
    if prompt is not None:
        return prompt
    if "finish_reason" in first or "completion_tokens" in first:
        return None
    return loglikelihood_prompt_tokens(first)


__all__ = [
    "instance_prompt_tokens",
    "loglikelihood_prompt_tokens",
    "nonneg_int",
    "request_token_usage",
]
