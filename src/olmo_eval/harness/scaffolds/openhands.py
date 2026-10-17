"""OpenHands SDK scaffold."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from olmo_eval.common.types import LMOutput, LMRequest, SamplingParams
from olmo_eval.common.types.tools import ToolCall, ToolResult
from olmo_eval.common.types.trajectory import AgentTrajectory, AgentTurn
from olmo_eval.harness.config import HarnessConfig
from olmo_eval.harness.result import HarnessResult
from olmo_eval.harness.scaffolds import Scaffold, register_scaffold
from olmo_eval.inference.base import InferenceProvider

if TYPE_CHECKING:
    from olmo_eval.harness.sandbox import SandboxManager
    from olmo_eval.harness.sandbox.executor import SandboxExecutor

logger = logging.getLogger(__name__)


@register_scaffold("openhands")
class OpenHandsScaffold(Scaffold):
    """Scaffold that delegates execution to OpenHands SDK.

    This scaffold uses the openhands-ai SDK to run agents with built-in
    tools like bash execution and file editing.
    """

    name = "openhands"
    required_extras = ("openhands",)

    def __init__(self) -> None:
        self._sandbox_manager: SandboxManager | None = None

    async def initialize(self, config: HarnessConfig) -> None:
        """Initialize sandbox manager if needed."""
        needs_sandbox = config.sandboxes and config.has_sandbox_tools

        if needs_sandbox and self._sandbox_manager is None:
            from olmo_eval.harness.sandbox import SandboxManager

            self._sandbox_manager = SandboxManager(config.sandboxes, owner=config.name)
            await self._sandbox_manager.start()
            logger.info(
                f"Sandbox manager started with {self._sandbox_manager.executor_count} executor(s)"
            )

    async def cleanup(self) -> None:
        """Clean up resources including sandbox manager."""
        if self._sandbox_manager is not None:
            await self._sandbox_manager.stop()
            self._sandbox_manager = None

    def _get_sandbox_executor(self) -> SandboxExecutor | None:
        """Get the first available sandbox executor.

        Returns:
            SandboxExecutor instance or None if not available.
        """
        if self._sandbox_manager is None:
            return None
        try:
            return self._sandbox_manager.get_executor(frozenset())
        except ValueError:
            return None

    async def run(
        self,
        provider: InferenceProvider,
        config: HarnessConfig,
        request: LMRequest,
        sampling_params: SamplingParams | None = None,
        trace_metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> HarnessResult:
        """Execute using OpenHands SDK.

        Args:
            provider: The inference provider for model calls.
            config: Harness configuration (tools, system prompt, etc.).
            request: The initial request.
            sampling_params: Optional sampling parameters.
            trace_metadata: Optional metadata for tracing (e.g., instance_id, task_id).
            **kwargs: Scaffold-specific options:
                - enable_compaction: Enable context compaction (default: True).

        Returns:
            HarnessResult with trajectory from SDK execution.
        """
        try:
            from openhands.sdk import (  # type: ignore[ty:unresolved-import]
                LLM,
                Conversation,
            )
            from openhands.sdk.conversation.state import (  # type: ignore[ty:unresolved-import]
                ConversationExecutionStatus,
            )
            from openhands.sdk.event import (  # type: ignore[ty:unresolved-import]
                ActionEvent,
                MessageEvent,
                ObservationEvent,
            )
            from openhands.sdk.event.conversation_error import (  # type: ignore[ty:unresolved-import]
                ConversationErrorEvent,
            )
        except ImportError as e:
            raise ImportError(
                "OpenHands SDK not installed. Install with: pip install openhands-ai"
            ) from e

        max_turns = config.max_turns or 10

        # Get the input message
        input_text = ""
        if request.messages:
            for msg in reversed(request.messages):
                if msg.get("role") == "user":
                    input_text = msg.get("content", "")
                    break

        # Configure LLM using the provider
        # OpenHands uses LiteLLM internally, which requires provider prefix
        from olmo_eval.inference.providers.litellm import LiteLLMProvider

        client = provider.get_openai_client()

        if isinstance(provider, LiteLLMProvider):
            model_name = provider.request_model_name
            api_key = client.api_key
            base_url = str(client.base_url) if client.base_url else None
            is_self_hosted = False
        else:
            model_name = f"hosted_vllm/{provider.model_name}"
            api_key = client.api_key or "dummy"  # vLLM doesn't require auth
            base_url = str(client.base_url) if client.base_url else None
            is_self_hosted = True

        llm_kwargs: dict[str, Any] = {
            "model": model_name,
            "api_key": api_key,
            "base_url": base_url,
        }
        if is_self_hosted:
            # Disable cost tracking for self-hosted models (not in LiteLLM's pricing DB)
            llm_kwargs["input_cost_per_token"] = 0.0
            llm_kwargs["output_cost_per_token"] = 0.0

        llm = LLM(**llm_kwargs)

        from olmo_eval.harness.adapters.openhands import create_sandbox_agent

        # Get sandbox runtime if available and needed
        runtime = None
        working_dir = "/workspace"
        command_timeout = 120.0

        if config.has_sandbox_tools:
            if self._sandbox_manager is None:
                raise RuntimeError(
                    "Sandbox tools configured but no sandbox available. "
                    "Add 'sandboxes' to your HarnessConfig."
                )
            executor = self._get_sandbox_executor()
            if executor is None:
                raise RuntimeError("Sandbox configured but no executor available")
            if executor._runtime is None:
                raise RuntimeError("Sandbox executor not started")
            runtime = executor._runtime
            working_dir = executor.config.working_dir
            command_timeout = executor.config.command_timeout

        agent = create_sandbox_agent(
            llm=llm,
            runtime=runtime,
            working_dir=working_dir,
            timeout=command_timeout,
            system_prompt=config.system_prompt,
            harness_tools=config.resolved_tools,
        )
        logger.info(f"OpenHands agent ready with {len(config.resolved_tools)} tool(s)")

        # Create conversation with visualizer=None to disable console output
        conversation = Conversation(
            agent=agent,
            workspace=working_dir,
            max_iteration_per_run=max_turns,
            visualizer=None,  # Disable console output
        )

        # Send message and run (run() takes no arguments)
        conversation.send_message(input_text)
        conversation.run()

        # Check if max iterations was reached
        max_turns_reached = False
        error_msg = None
        if conversation.state.execution_status == ConversationExecutionStatus.ERROR:
            # Check events for MaxIterationsReached error
            for event in reversed(conversation.state.events):
                if isinstance(event, ConversationErrorEvent):
                    if event.code == "MaxIterationsReached":
                        max_turns_reached = True
                        error_msg = "Max turns exceeded"
                    else:
                        error_msg = f"{event.code}: {event.detail}"
                    break

        # Convert conversation events to trajectory
        trajectory = self._convert_events_to_trajectory(
            conversation.state.events,
            MessageEvent,
            ActionEvent,
            ObservationEvent,
        )

        # Get final output from last agent message
        final_text = ""
        for event in reversed(conversation.state.events):
            if isinstance(event, MessageEvent) and event.source == "agent":
                # Extract text content from message
                if event.llm_message and event.llm_message.content:
                    for content in event.llm_message.content:
                        if hasattr(content, "text"):
                            final_text = content.text
                            break
                break

        return HarnessResult(
            trajectory=trajectory,
            final_output=LMOutput(text=final_text),
            max_turns_reached=max_turns_reached,
            error=error_msg,
        )

    def _convert_events_to_trajectory(
        self,
        events: list[Any],
        message_event_cls: type,
        action_event_cls: type,
        observation_event_cls: type,
    ) -> AgentTrajectory:
        """Convert OpenHands events to AgentTrajectory.

        Args:
            events: List of OpenHands events from conversation.state.events.
            message_event_cls: The MessageEvent class.
            action_event_cls: The ActionEvent class.
            observation_event_cls: The ObservationEvent class.

        Returns:
            AgentTrajectory with converted turns.
        """
        turns: list[AgentTurn] = []

        for event in events:
            if isinstance(event, message_event_cls):
                # MessageEvent contains user or agent messages
                content = ""
                if event.llm_message and event.llm_message.content:
                    for c in event.llm_message.content:
                        if hasattr(c, "text"):
                            content = c.text
                            break

                if event.source == "user":
                    turns.append(AgentTurn.user(content=content))
                elif event.source == "agent":
                    turns.append(AgentTurn.assistant(content=content))

            elif isinstance(event, action_event_cls):
                # ActionEvent contains agent tool calls
                content = ""
                if event.thought:
                    # Extract thought text
                    for t in event.thought:
                        if hasattr(t, "text"):
                            content = t.text
                            break

                # Convert tool call
                tool_calls = []
                if event.tool_call:
                    # Get arguments from the action object
                    args = "{}"
                    if event.action:
                        # Serialize the action to get arguments
                        try:
                            args = json.dumps(event.action.model_dump())
                        except Exception:
                            args = "{}"

                    call = ToolCall.create(
                        call_id=event.tool_call_id or event.id,
                        name=event.tool_name,
                        arguments=args,
                    )
                    tool_calls.append(call)

                turns.append(AgentTurn.assistant(content=content, tool_calls=tool_calls))

            elif isinstance(event, observation_event_cls):
                # ObservationEvent contains tool results
                content = ""
                if event.observation:
                    # Serialize observation to get content
                    try:
                        content = json.dumps(event.observation.model_dump())
                    except Exception:
                        content = str(event.observation)

                tool_result = ToolResult(
                    tool_call_id=event.tool_call_id or event.action_id,
                    content=content,
                )
                turns.append(AgentTurn.tool([tool_result]))

        return AgentTrajectory(turns=tuple(turns))
