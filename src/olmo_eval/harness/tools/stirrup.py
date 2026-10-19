"""Tools of the ``stirrup`` scaffold.

Names, descriptions and parameter schemas follow Stirrup's ``code_exec`` and
``finish`` tools. Stirrup ships no abandon tool; ``abandon_task_finish`` is the
give-up tool Artificial Analysis's agentic benchmarks add, with the one
``reason`` parameter they document. The scaffold runs all three itself.
"""

from __future__ import annotations

from olmo_eval.harness.sandbox import Capability

from .registry import register_tool
from .tool import Tool


def _run_by_scaffold(name: str):
    def execute(**_: object) -> str:
        raise NotImplementedError(f"{name} is run by the stirrup scaffold.")

    return execute


code_exec = register_tool(
    Tool(
        name="code_exec",
        description=(
            "Execute a shell command in the execution environment. Returns exit code, "
            "stdout, and stderr as XML."
        ),
        parameters={
            "type": "object",
            "description": "Shell command to execute in the execution environment.",
            "properties": {
                "cmd": {
                    "type": "string",
                    "description": (
                        "Shell command to execute (bash syntax). IMPORTANT: Use only relative "
                        "paths. Do not use absolute paths (starting with / or ~) or reference "
                        "directories outside the working directory."
                    ),
                }
            },
            "required": ["cmd"],
        },
        execute=_run_by_scaffold("code_exec"),
        sandbox=Capability.BASH,
    )
)

finish = register_tool(
    Tool(
        name="finish",
        description=(
            "Signal task completion with a reason. Use when the task is finished or cannot "
            "proceed further. Note that you will need a separate turn to finish."
        ),
        parameters={
            "type": "object",
            "description": "Explanation for why the task is complete or cannot proceed.",
            "properties": {
                "reason": {"type": "string", "description": "Reason for finishing."},
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "List of file paths created or modified. Do not include directories, "
                        "only files."
                    ),
                },
            },
            "required": ["reason", "paths"],
        },
        execute=_run_by_scaffold("finish"),
        sandbox=Capability.BASH,
    )
)

abandon_task_finish = register_tool(
    Tool(
        name="abandon_task_finish",
        description=(
            "Give up on the task. Call this only when you have concluded the task cannot be "
            "completed, not to escape a difficult task."
        ),
        parameters={
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Why the task cannot be completed.",
                }
            },
            "required": ["reason"],
        },
        execute=_run_by_scaffold("abandon_task_finish"),
    )
)
