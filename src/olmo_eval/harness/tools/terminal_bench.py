"""Terminal-Bench specific tools."""

from __future__ import annotations

from olmo_eval.harness.sandbox import Capability

from .registry import registered_tool


@registered_tool(
    name="bash",
    description="Execute a bash command.",
    sandbox=Capability.BASH,
    session=True,
)
async def bash(command: str) -> str:
    """Execute a bash command in a persistent shell session.

    The single tool of the mini-swe-agent style scaffolds: the model issues one
    command per step and reads back its output and exit code. Working directory
    changes and exported variables persist between calls.

    Args:
        command: The bash command to execute.

    Returns:
        The command output (stdout + stderr combined).
    """
    raise NotImplementedError(
        "bash requires sandbox execution. Ensure sandbox is enabled in HarnessConfig."
    )


@registered_tool(
    name="submit",
    description="Call this tool when you have completed the task.",
)
async def submit() -> str:
    """Signal that the task is complete.

    Call this tool when you have finished working on the task and are
    ready to submit your solution for verification.

    Returns:
        Confirmation message.
    """
    return "Task submitted successfully. Your work will now be verified."
