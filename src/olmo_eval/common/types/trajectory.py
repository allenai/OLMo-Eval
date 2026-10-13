"""Multi-turn agent trajectory types for evaluation.

This module provides types for representing agent interactions over multiple
turns, including tool calls, tool results, and state tracking.
"""

from dataclasses import dataclass, field
from typing import Any, Literal

from .tools import ToolCall, ToolResult


@dataclass(frozen=True, slots=True)
class AgentTurn:
    """A single turn in an agent trajectory.

    Represents either an assistant message (with optional tool calls)
    or a tool response. Assistant turns may also carry the model's
    reasoning text when the provider exposes it.
    """

    role: Literal["assistant", "tool", "user", "system"]
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    tool_results: tuple[ToolResult, ...] = ()
    timestamp_ms: int | None = None
    token_count: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    reasoning: str | None = None

    @property
    def has_tool_calls(self) -> bool:
        """Check if this turn contains tool calls."""
        return len(self.tool_calls) > 0

    @property
    def has_tool_results(self) -> bool:
        """Check if this turn contains tool results."""
        return len(self.tool_results) > 0

    @classmethod
    def assistant(
        cls,
        content: str = "",
        tool_calls: list[ToolCall] | None = None,
        timestamp_ms: int | None = None,
        token_count: int | None = None,
        reasoning: str | None = None,
    ) -> "AgentTurn":
        """Create an assistant turn.

        Args:
            content: The text content of the assistant message.
            tool_calls: Optional list of tool calls made by the assistant.
            timestamp_ms: Optional timestamp in milliseconds.
            token_count: Optional token count for this turn.
            reasoning: Optional reasoning text the model produced for this turn.

        Returns:
            A new AgentTurn with role="assistant".
        """
        return cls(
            role="assistant",
            content=content,
            tool_calls=tuple(tool_calls) if tool_calls else (),
            timestamp_ms=timestamp_ms,
            token_count=token_count,
            reasoning=reasoning,
        )

    @classmethod
    def tool(
        cls,
        results: list[ToolResult],
        timestamp_ms: int | None = None,
    ) -> "AgentTurn":
        """Create a tool result turn.

        Args:
            results: List of tool results.
            timestamp_ms: Optional timestamp in milliseconds.

        Returns:
            A new AgentTurn with role="tool".
        """
        return cls(
            role="tool",
            tool_results=tuple(results),
            timestamp_ms=timestamp_ms,
        )

    @classmethod
    def user(
        cls,
        content: str,
        timestamp_ms: int | None = None,
    ) -> "AgentTurn":
        """Create a user turn.

        Args:
            content: The user's message content.
            timestamp_ms: Optional timestamp in milliseconds.

        Returns:
            A new AgentTurn with role="user".
        """
        return cls(
            role="user",
            content=content,
            timestamp_ms=timestamp_ms,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Only includes fields with non-empty/non-null values for cleaner output.

        Returns:
            Dictionary representation of the AgentTurn.
        """
        result: dict[str, Any] = {"role": self.role}
        if self.content:
            result["content"] = self.content
        if self.tool_calls:
            result["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        if self.tool_results:
            result["tool_results"] = [tr.to_dict() for tr in self.tool_results]
        if self.timestamp_ms is not None:
            result["timestamp_ms"] = self.timestamp_ms
        if self.token_count is not None:
            result["token_count"] = self.token_count
        if self.metadata:
            result["metadata"] = self.metadata
        if self.reasoning is not None:
            result["reasoning"] = self.reasoning
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentTurn":
        """Create from dictionary.

        Args:
            data: Dictionary with AgentTurn data.

        Returns:
            A new AgentTurn instance.
        """
        return cls(
            role=data.get("role", "assistant"),
            content=data.get("content", ""),
            tool_calls=tuple(ToolCall.from_dict(tc) for tc in data.get("tool_calls", [])),
            tool_results=tuple(ToolResult.from_dict(tr) for tr in data.get("tool_results", [])),
            timestamp_ms=data.get("timestamp_ms"),
            token_count=data.get("token_count"),
            metadata=data.get("metadata", {}),
            reasoning=data.get("reasoning"),
        )


@dataclass(frozen=True, slots=True)
class AgentTrajectory:
    """Complete trajectory of an agent interaction.

    Tracks all turns, the final answer, and state information.
    """

    turns: tuple[AgentTurn, ...] = ()
    final_answer: str | None = None
    state_snapshot: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        """Calculate total tokens across all turns."""
        return sum(t.token_count or 0 for t in self.turns)

    @property
    def total_tool_calls(self) -> int:
        """Count total tool calls across all turns."""
        return sum(len(t.tool_calls) for t in self.turns)

    @property
    def tool_call_sequence(self) -> list[ToolCall]:
        """Get flattened list of all tool calls in order."""
        calls: list[ToolCall] = []
        for turn in self.turns:
            calls.extend(turn.tool_calls)
        return calls

    @property
    def tool_result_sequence(self) -> list[ToolResult]:
        """Get flattened list of all tool results in order."""
        results: list[ToolResult] = []
        for turn in self.turns:
            results.extend(turn.tool_results)
        return results

    @property
    def unique_tools_used(self) -> set[str]:
        """Get set of unique tool names used in this trajectory."""
        return {call.function.name for call in self.tool_call_sequence}

    @property
    def num_turns(self) -> int:
        """Get the number of turns in the trajectory."""
        return len(self.turns)

    @property
    def assistant_turns(self) -> list[AgentTurn]:
        """Get all assistant turns."""
        return [t for t in self.turns if t.role == "assistant"]

    @property
    def tool_turns(self) -> list[AgentTurn]:
        """Get all tool result turns."""
        return [t for t in self.turns if t.role == "tool"]

    def tool_calls_by_name(self, name: str) -> list[ToolCall]:
        """Get all tool calls with the given function name.

        Args:
            name: The function name to filter by.

        Returns:
            List of ToolCall objects with matching function name.
        """
        return [call for call in self.tool_call_sequence if call.function.name == name]

    def tool_call_names(self) -> list[str]:
        """Get list of tool names in call order.

        Returns:
            List of function names in the order they were called.
        """
        return [call.function.name for call in self.tool_call_sequence]

    def get_turn(self, index: int) -> AgentTurn | None:
        """Get a turn by index.

        Args:
            index: The turn index.

        Returns:
            The AgentTurn at the given index, or None if out of bounds.
        """
        if 0 <= index < len(self.turns):
            return self.turns[index]
        return None

    def with_final_answer(self, answer: str) -> "AgentTrajectory":
        """Create a new trajectory with the given final answer.

        Args:
            answer: The final answer to set.

        Returns:
            A new AgentTrajectory with the final answer set.
        """
        return AgentTrajectory(
            turns=self.turns,
            final_answer=answer,
            state_snapshot=self.state_snapshot,
            metadata=self.metadata,
        )

    def with_state(self, state: dict[str, Any]) -> "AgentTrajectory":
        """Create a new trajectory with the given state snapshot.

        Args:
            state: The state snapshot to set.

        Returns:
            A new AgentTrajectory with the state snapshot set.
        """
        return AgentTrajectory(
            turns=self.turns,
            final_answer=self.final_answer,
            state_snapshot=state,
            metadata=self.metadata,
        )

    @classmethod
    def from_turns(cls, turns: list[AgentTurn]) -> "AgentTrajectory":
        """Create a trajectory from a list of turns.

        Args:
            turns: List of AgentTurn objects.

        Returns:
            A new AgentTrajectory.
        """
        return cls(turns=tuple(turns))

    def to_messages(self) -> list[dict[str, Any]]:
        """Convert trajectory to OpenAI message format.

        Returns:
            List of message dictionaries.
        """
        messages: list[dict[str, Any]] = []
        for turn in self.turns:
            if turn.role == "assistant":
                msg: dict[str, Any] = {"role": "assistant", "content": turn.content}
                if turn.tool_calls:
                    msg["tool_calls"] = [tc.to_openai() for tc in turn.tool_calls]
                messages.append(msg)
            elif turn.role == "tool":
                for result in turn.tool_results:
                    messages.append(result.to_openai())
            elif turn.role == "user":
                messages.append({"role": "user", "content": turn.content})
            elif turn.role == "system":
                messages.append({"role": "system", "content": turn.content})
        return messages

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Only includes fields with non-empty/non-null values for cleaner output.

        Returns:
            Dictionary representation of the AgentTrajectory.
        """
        result: dict[str, Any] = {
            "turns": [t.to_dict() for t in self.turns],
        }
        if self.final_answer is not None:
            result["final_answer"] = self.final_answer
        if self.state_snapshot:
            result["state_snapshot"] = self.state_snapshot
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentTrajectory":
        """Create from dictionary.

        Args:
            data: Dictionary with AgentTrajectory data.

        Returns:
            A new AgentTrajectory instance.
        """
        return cls(
            turns=tuple(AgentTurn.from_dict(t) for t in data.get("turns", [])),
            final_answer=data.get("final_answer"),
            state_snapshot=data.get("state_snapshot", {}),
            metadata=data.get("metadata", {}),
        )
