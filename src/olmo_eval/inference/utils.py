"""Inference provider utilities."""

from __future__ import annotations

import asyncio
import concurrent.futures
from collections.abc import Coroutine
from typing import Any, cast


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
    """Patch the openai-agents SDK for vLLM compatibility.

    - vLLM doesn't support the 'strict' field in tool schemas, which the SDK always
      includes, even when strict_mode=False. The patch omits the field when
      strict_json_schema is False. See https://github.com/vllm-project/vllm/issues/27746
    - vLLM returns a reasoning parser's output in a ``reasoning`` field, while the SDK
      only reads ``reasoning_content``. The patch reads either, so the model's
      reasoning is kept as a reasoning item instead of being dropped.

    Call this once before creating any agents that will talk to vLLM.
    Safe to call multiple times (idempotent).
    """
    from agents import FunctionTool  # type: ignore[ty:unresolved-import]
    from agents.models.chatcmpl_converter import Converter  # type: ignore[ty:unresolved-import]

    _patch_reasoning_field(Converter)

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


def _patch_reasoning_field(converter: Any) -> None:
    """Make the SDK's converter read vLLM's ``reasoning`` field as ``reasoning_content``."""
    if getattr(converter, "_vllm_reasoning_patched", False):
        return

    original = converter.message_to_output_items

    @classmethod
    def _patched_message_to_output_items(cls, message, *args, **kwargs):
        reasoning = getattr(message, "reasoning", None)
        if (
            isinstance(reasoning, str)
            and reasoning
            and not getattr(message, "reasoning_content", None)
        ):
            message = message.model_copy(update={"reasoning_content": reasoning})
        return original(message, *args, **kwargs)

    converter.message_to_output_items = _patched_message_to_output_items
    converter._vllm_reasoning_patched = True
