"""Tests for olmo_eval.core.trajectory_types module."""

from olmo_eval.common.types import AgentTrajectory, AgentTurn, ToolCall, ToolResult


class TestAgentTurn:
    """Tests for AgentTurn dataclass."""

    def test_create_assistant_with_tool_calls(self):
        """Test assistant turn with tool calls."""
        calls = [ToolCall.create("1", "search", {"query": "test"})]
        turn = AgentTurn.assistant(content="Let me search", tool_calls=calls)
        assert turn.has_tool_calls
        assert len(turn.tool_calls) == 1
        assert turn.tool_calls[0].function.name == "search"


class TestAgentTrajectory:
    """Tests for AgentTrajectory dataclass."""

    def test_from_turns(self):
        """Test creating from list of turns."""
        turns = [
            AgentTurn.user(content="Hello"),
            AgentTurn.assistant(content="Hi there"),
        ]
        traj = AgentTrajectory.from_turns(turns)
        assert traj.num_turns == 2

    def test_total_tokens(self):
        """Test token counting."""
        turns = [
            AgentTurn.assistant(content="Hello", token_count=5),
            AgentTurn.assistant(content="World", token_count=10),
        ]
        traj = AgentTrajectory.from_turns(turns)
        assert traj.total_tokens == 15

    def test_total_tool_calls(self):
        """Test tool call counting."""
        turns = [
            AgentTurn.assistant(
                tool_calls=[
                    ToolCall.create("1", "search", {}),
                    ToolCall.create("2", "calc", {}),
                ]
            ),
            AgentTurn.assistant(tool_calls=[ToolCall.create("3", "search", {})]),
        ]
        traj = AgentTrajectory.from_turns(turns)
        assert traj.total_tool_calls == 3

    def test_tool_call_sequence(self):
        """Test flattened tool call sequence."""
        turns = [
            AgentTurn.assistant(tool_calls=[ToolCall.create("1", "first", {})]),
            AgentTurn.assistant(tool_calls=[ToolCall.create("2", "second", {})]),
        ]
        traj = AgentTrajectory.from_turns(turns)
        sequence = traj.tool_call_sequence
        assert len(sequence) == 2
        assert sequence[0].function.name == "first"
        assert sequence[1].function.name == "second"

    def test_unique_tools_used(self):
        """Test getting unique tools."""
        turns = [
            AgentTurn.assistant(
                tool_calls=[
                    ToolCall.create("1", "search", {}),
                    ToolCall.create("2", "search", {}),
                    ToolCall.create("3", "calc", {}),
                ]
            ),
        ]
        traj = AgentTrajectory.from_turns(turns)
        assert traj.unique_tools_used == {"search", "calc"}

    def test_tool_calls_by_name(self):
        """Test filtering tool calls by name."""
        turns = [
            AgentTurn.assistant(
                tool_calls=[
                    ToolCall.create("1", "search", {"q": "a"}),
                    ToolCall.create("2", "calc", {}),
                    ToolCall.create("3", "search", {"q": "b"}),
                ]
            ),
        ]
        traj = AgentTrajectory.from_turns(turns)
        search_calls = traj.tool_calls_by_name("search")
        assert len(search_calls) == 2

    def test_tool_call_names(self):
        """Test getting tool names in order."""
        turns = [
            AgentTurn.assistant(
                tool_calls=[
                    ToolCall.create("1", "a", {}),
                    ToolCall.create("2", "b", {}),
                ]
            ),
            AgentTurn.assistant(tool_calls=[ToolCall.create("3", "c", {})]),
        ]
        traj = AgentTrajectory.from_turns(turns)
        assert traj.tool_call_names() == ["a", "b", "c"]

    def test_assistant_turns(self):
        """Test getting assistant turns only."""
        turns = [
            AgentTurn.user(content="Hello"),
            AgentTurn.assistant(content="Hi"),
            AgentTurn.user(content="Bye"),
            AgentTurn.assistant(content="Goodbye"),
        ]
        traj = AgentTrajectory.from_turns(turns)
        assistant = traj.assistant_turns
        assert len(assistant) == 2
        assert all(t.role == "assistant" for t in assistant)

    def test_tool_turns(self):
        """Test getting tool turns only."""
        turns = [
            AgentTurn.assistant(tool_calls=[ToolCall.create("1", "test", {})]),
            AgentTurn.tool([ToolResult(tool_call_id="1", content="result")]),
        ]
        traj = AgentTrajectory.from_turns(turns)
        tool_turns = traj.tool_turns
        assert len(tool_turns) == 1
        assert tool_turns[0].role == "tool"

    def test_get_turn(self):
        """Test getting turn by index."""
        turns = [
            AgentTurn.user(content="First"),
            AgentTurn.assistant(content="Second"),
        ]
        traj = AgentTrajectory.from_turns(turns)
        assert traj.get_turn(0).content == "First"
        assert traj.get_turn(1).content == "Second"
        assert traj.get_turn(99) is None
        assert traj.get_turn(-1) is None

    def test_with_final_answer(self):
        """Test setting final answer."""
        traj = AgentTrajectory()
        new_traj = traj.with_final_answer("The answer is 42")
        assert new_traj.final_answer == "The answer is 42"
        assert traj.final_answer is None  # Original unchanged

    def test_with_state(self):
        """Test setting state snapshot."""
        traj = AgentTrajectory()
        new_traj = traj.with_state({"count": 10})
        assert new_traj.state_snapshot == {"count": 10}
        assert traj.state_snapshot == {}  # Original unchanged

    def test_to_messages(self):
        """Test converting to OpenAI message format."""
        call = ToolCall.create("1", "search", {"q": "test"})
        result = ToolResult(tool_call_id="1", content="results")
        turns = [
            AgentTurn.user(content="Search for test"),
            AgentTurn.assistant(content="Let me search", tool_calls=[call]),
            AgentTurn.tool([result]),
            AgentTurn.assistant(content="I found results"),
        ]
        traj = AgentTrajectory.from_turns(turns)
        messages = traj.to_messages()

        assert len(messages) == 4
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert "tool_calls" in messages[1]
        assert messages[2]["role"] == "tool"
        assert messages[3]["role"] == "assistant"


class TestAgentTrajectoryToolResults:
    """Tests for tool result handling in AgentTrajectory."""

    def test_tool_result_sequence(self):
        """Test getting tool results in order."""
        turns = [
            AgentTurn.tool([ToolResult(tool_call_id="1", content="first")]),
            AgentTurn.tool([ToolResult(tool_call_id="2", content="second")]),
        ]
        traj = AgentTrajectory.from_turns(turns)
        results = traj.tool_result_sequence
        assert len(results) == 2
        assert results[0].content == "first"
        assert results[1].content == "second"


class TestAgentTurnSerialization:
    """Tests for AgentTurn serialization."""

    def test_to_dict_simple(self):
        """Test converting simple AgentTurn to dict."""
        turn = AgentTurn.assistant(content="Hello")
        data = turn.to_dict()
        assert data["role"] == "assistant"
        assert data["content"] == "Hello"
        # Empty fields should be omitted
        assert "tool_calls" not in data
        assert "tool_results" not in data
        assert "timestamp_ms" not in data
        assert "token_count" not in data
        assert "metadata" not in data

    def test_to_dict_with_tool_calls(self):
        """Test converting AgentTurn with tool calls to dict."""
        calls = [ToolCall.create("1", "search", {"query": "test"})]
        turn = AgentTurn.assistant(content="Searching", tool_calls=calls)
        data = turn.to_dict()
        assert len(data["tool_calls"]) == 1
        assert data["tool_calls"][0]["function"]["name"] == "search"

    def test_to_dict_with_tool_results(self):
        """Test converting AgentTurn with tool results to dict."""
        results = [ToolResult(tool_call_id="1", content="Result")]
        turn = AgentTurn.tool(results)
        data = turn.to_dict()
        assert data["role"] == "tool"
        assert len(data["tool_results"]) == 1
        assert data["tool_results"][0]["content"] == "Result"
        # Empty tool_calls should be omitted
        assert "tool_calls" not in data

    def test_to_dict_with_metadata(self):
        """Test converting AgentTurn with optional fields to dict."""
        turn = AgentTurn.assistant(content="Test", timestamp_ms=12345, token_count=10)
        data = turn.to_dict()
        assert data["timestamp_ms"] == 12345
        assert data["token_count"] == 10

    def test_from_dict_simple(self):
        """Test creating AgentTurn from dict."""
        data = {
            "role": "user",
            "content": "Hello there",
            "tool_calls": [],
            "tool_results": [],
        }
        turn = AgentTurn.from_dict(data)
        assert turn.role == "user"
        assert turn.content == "Hello there"

    def test_from_dict_with_tool_calls(self):
        """Test creating AgentTurn with tool calls from dict."""
        data = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "calc", "arguments": '{"x": 1}'},
                }
            ],
            "tool_results": [],
        }
        turn = AgentTurn.from_dict(data)
        assert turn.has_tool_calls
        assert turn.tool_calls[0].function.name == "calc"

    def test_roundtrip(self):
        """Test to_dict/from_dict roundtrip."""
        calls = [ToolCall.create("1", "search", {"q": "test"})]
        original = AgentTurn.assistant(
            content="Let me search",
            tool_calls=calls,
            timestamp_ms=99999,
            token_count=25,
        )
        restored = AgentTurn.from_dict(original.to_dict())
        assert restored.role == original.role
        assert restored.content == original.content
        assert len(restored.tool_calls) == len(original.tool_calls)
        assert restored.tool_calls[0].function.name == original.tool_calls[0].function.name
        assert restored.timestamp_ms == original.timestamp_ms
        assert restored.token_count == original.token_count

    def test_to_dict_omits_reasoning_when_absent(self):
        """Turns without reasoning must serialize exactly as before the field existed."""
        turn = AgentTurn.assistant(content="Hello")
        assert turn.reasoning is None
        assert "reasoning" not in turn.to_dict()

    def test_roundtrip_with_reasoning(self):
        """Reasoning survives to_dict/from_dict and stays separate from content."""
        original = AgentTurn.assistant(content="Answer", reasoning="Think it through first.")
        data = original.to_dict()
        assert data["content"] == "Answer"
        assert data["reasoning"] == "Think it through first."
        restored = AgentTurn.from_dict(data)
        assert restored.reasoning == original.reasoning
        assert restored.content == original.content

    def test_from_dict_without_reasoning_key_reserializes_identically(self):
        """A dict written before the reasoning field loads and re-serializes unchanged."""
        data = {
            "role": "assistant",
            "content": "Searching",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "search", "arguments": '{"q": "test"}'},
                }
            ],
            "timestamp_ms": 99999,
            "token_count": 25,
        }
        turn = AgentTurn.from_dict(data)
        assert turn.reasoning is None
        assert turn.to_dict() == data


class TestAgentTrajectorySerialization:
    """Tests for AgentTrajectory serialization."""

    def test_to_dict_empty(self):
        """Test converting empty AgentTrajectory to dict."""
        traj = AgentTrajectory()
        data = traj.to_dict()
        assert data["turns"] == []
        # Empty/null fields should be omitted
        assert "final_answer" not in data
        assert "state_snapshot" not in data
        assert "metadata" not in data

    def test_to_dict_with_turns(self):
        """Test converting AgentTrajectory with turns to dict."""
        turns = [
            AgentTurn.user(content="Hi"),
            AgentTurn.assistant(content="Hello"),
        ]
        traj = AgentTrajectory.from_turns(turns)
        data = traj.to_dict()
        assert len(data["turns"]) == 2
        assert data["turns"][0]["role"] == "user"
        assert data["turns"][1]["role"] == "assistant"
        # Empty fields should be omitted
        assert "final_answer" not in data
        assert "state_snapshot" not in data
        assert "metadata" not in data

    def test_to_dict_with_final_answer(self):
        """Test converting AgentTrajectory with final answer to dict."""
        traj = AgentTrajectory().with_final_answer("42")
        data = traj.to_dict()
        assert data["final_answer"] == "42"

    def test_to_dict_with_state(self):
        """Test converting AgentTrajectory with state to dict."""
        traj = AgentTrajectory().with_state({"count": 5, "done": True})
        data = traj.to_dict()
        assert data["state_snapshot"] == {"count": 5, "done": True}

    def test_from_dict_empty(self):
        """Test creating empty AgentTrajectory from dict."""
        data = {"turns": [], "final_answer": None, "state_snapshot": {}, "metadata": {}}
        traj = AgentTrajectory.from_dict(data)
        assert traj.num_turns == 0
        assert traj.final_answer is None

    def test_from_dict_with_turns(self):
        """Test creating AgentTrajectory with turns from dict."""
        data = {
            "turns": [
                {"role": "user", "content": "Hello", "tool_calls": [], "tool_results": []},
                {"role": "assistant", "content": "Hi", "tool_calls": [], "tool_results": []},
            ],
            "final_answer": "Done",
            "state_snapshot": {"key": "value"},
            "metadata": {},
        }
        traj = AgentTrajectory.from_dict(data)
        assert traj.num_turns == 2
        assert traj.final_answer == "Done"
        assert traj.state_snapshot == {"key": "value"}

    def test_roundtrip(self):
        """Test to_dict/from_dict roundtrip with complex trajectory."""
        call = ToolCall.create("1", "search", {"query": "test"})
        result = ToolResult(tool_call_id="1", content="found it")
        turns = [
            AgentTurn.user(content="Find test"),
            AgentTurn.assistant(content="Searching", tool_calls=[call]),
            AgentTurn.tool([result]),
            AgentTurn.assistant(content="Found it!"),
        ]
        original = AgentTrajectory(
            turns=tuple(turns),
            final_answer="Found it!",
            state_snapshot={"searches": 1},
            metadata={"task": "search"},
        )
        restored = AgentTrajectory.from_dict(original.to_dict())
        assert restored.num_turns == original.num_turns
        assert restored.final_answer == original.final_answer
        assert restored.state_snapshot == original.state_snapshot
        assert restored.metadata == original.metadata
        assert restored.tool_call_names() == original.tool_call_names()

    def test_from_dict_without_reasoning_reserializes_identically(self):
        """A trajectory saved before the reasoning field round-trips unchanged."""
        data = {
            "turns": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ],
            "final_answer": "Hi there",
        }
        traj = AgentTrajectory.from_dict(data)
        assert all(turn.reasoning is None for turn in traj.turns)
        assert traj.to_dict() == data
