"""Inference provider utilities."""

from __future__ import annotations

import asyncio
import concurrent.futures
from collections.abc import Awaitable, Callable, Coroutine
from typing import Any, cast

from olmo_eval.common.types import LMOutput, LMRequest
from olmo_eval.inference.dispatch import dispatch_concurrent_results
from olmo_eval.inference.errors import REQUEST_ERROR_KEY


async def dispatch_requests(
    requests: list[LMRequest],
    process_fn: Callable[[LMRequest], Awaitable[list[LMOutput]]],
    max_in_flight: int = 256,
    max_retries: int = 0,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[list[LMOutput]]:
    """Dispatch inference requests, recording failed requests as marked outputs."""
    results = await dispatch_concurrent_results(
        requests, process_fn, max_in_flight, max_retries, on_progress
    )
    outputs = []
    for result in results:
        if result.error is not None:
            error = result.error
            message = str(error)
            if len(message) > 500:
                message = message[:500] + "…"
            outputs.append(
                [
                    LMOutput(
                        text="", metadata={REQUEST_ERROR_KEY: f"{type(error).__name__}: {message}"}
                    )
                ]
            )
        else:
            outputs.append(result.result if result.result is not None else [])
    return outputs


def run_async[T](coro: Coroutine[Any, Any, T]) -> T:
    """Run async code from a sync context, handling nested event loops.

    Unlike asyncio.run(), this helper detects when there's already a running
    event loop (e.g., in Jupyter notebooks or async applications) and runs
    the coroutine in a dedicated thread to avoid RuntimeError.

    Args:
        coro: The coroutine to run.

    Returns:
        The result of the coroutine.
    """
    try:
        asyncio.get_running_loop()
        # Already in an async context - run in a thread
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            return cast(T, executor.submit(asyncio.run, coro).result())
    except RuntimeError:
        # No running loop - use asyncio.run directly
        return asyncio.run(coro)


def patch_openai_agents_for_vllm() -> None:
    """Patch openai-agents SDK to omit 'strict' field for vLLM compatibility.

    vLLM doesn't support the 'strict' field in tool schemas.
    The openai-agents SDK always includes it, even when strict_mode=False.
    This patch makes the SDK omit the field when strict_json_schema is False.

    See: https://github.com/vllm-project/vllm/issues/27746

    Call this once before creating any agents that will talk to vLLM.
    Safe to call multiple times (idempotent).
    """
    from agents import FunctionTool  # type: ignore[ty:unresolved-import]
    from agents.models.chatcmpl_converter import Converter  # type: ignore[ty:unresolved-import]

    # Check if already patched
    if getattr(Converter, "_vllm_patched", False):
        return

    _original_tool_to_openai = Converter.tool_to_openai

    @classmethod
    def _patched_tool_to_openai(cls, tool):
        result = _original_tool_to_openai(tool)
        # Remove 'strict' field if False (vLLM doesn't support it)
        if isinstance(tool, FunctionTool) and not tool.strict_json_schema:
            result.get("function", {}).pop("strict", None)
        return result

    Converter.tool_to_openai = _patched_tool_to_openai
    Converter._vllm_patched = True
