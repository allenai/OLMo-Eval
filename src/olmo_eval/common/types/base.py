"""Core data types and enums for evaluation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal, NotRequired, TypedDict

from olmo_eval.common.repr import hide_unset

if TYPE_CHECKING:
    from .tools import ToolCall, ToolSchema
    from .trajectory import AgentTrajectory


_MODEL_HASH_IGNORED_PROVIDER_KEYS = frozenset(
    {
        # Transport and auth metadata should not split a logical model identity.
        "api_base",
        "base_url",
        "force_download",
        "max_concurrency",
        "num_instances",
        "required_secrets",
    }
)

_MODEL_HASH_IGNORED_PROVIDER_KWARGS = frozenset(
    {
        # Backend/runtime tuning knobs that affect throughput or scaling, not model identity.
        # Keep request/template kwargs like default_chat_template_kwargs because they can
        # change the model's actual outputs.
        "attention_backend",
        "enable_expert_parallel",
        "enable_prefix_caching",
        "force_download",
        "gpu_memory_utilization",
        "max_num_batched_tokens",
        "max_num_seqs",
        "pipeline_parallel_size",
        "tensor_parallel_size",
    }
)


def _normalize_model_config_for_hash(
    config: Any,
    *,
    root: bool = False,
    provider_kwargs: bool = False,
) -> Any:
    """Drop operational provider fields so model hashes track model behavior.

    The stored model_config also contains execution details like endpoint selection,
    concurrency, and parallelism. Those should not create distinct model hashes.
    """
    if isinstance(config, dict):
        ignored_keys = (
            _MODEL_HASH_IGNORED_PROVIDER_KEYS
            if root
            else _MODEL_HASH_IGNORED_PROVIDER_KWARGS
            if provider_kwargs
            else frozenset()
        )

        canonical_attention_backend = None
        if root:
            root_attention_backend = config.get("attention_backend")
            kwargs_attention_backend = None
            kwargs_value = config.get("kwargs")
            if isinstance(kwargs_value, dict):
                kwargs_attention_backend = kwargs_value.get("attention_backend")
            canonical_attention_backend = (
                root_attention_backend
                if root_attention_backend is not None
                else kwargs_attention_backend
            )

        normalized: dict[str, Any] = {}
        for key, value in config.items():
            if key in ignored_keys:
                continue
            if root and key == "attention_backend":
                continue
            if root and key == "kwargs" and isinstance(value, dict):
                normalized_kwargs = _normalize_model_config_for_hash(value, provider_kwargs=True)
                if normalized_kwargs:
                    normalized[key] = normalized_kwargs
                continue
            normalized[key] = _normalize_model_config_for_hash(value)
        if root and canonical_attention_backend is not None:
            normalized["attention_backend"] = canonical_attention_backend
        return normalized

    if isinstance(config, list):
        return [_normalize_model_config_for_hash(item) for item in config]

    return config


def compute_model_hash(config: dict[str, Any] | None) -> str | None:
    """Compute a deterministic hash from a model configuration dict.

    The model hash is used to identify unique model configurations
    across experiments. The same config always produces the same hash,
    allowing multiple experiments (from different users or runs) to be
    associated with the same model configuration.

    Args:
        config: Model configuration dictionary. Operational provider fields such as
            replica count, endpoint URLs, concurrency limits, and auth metadata are
            ignored.

    Returns:
        16-character hex string hash of the config, or None if config is None.

    Example:
        >>> config = {"model": "llama3.1-8b", "temperature": 0.7}
        >>> model_hash = compute_model_hash(config)
        >>> len(model_hash)
        16
    """
    if not config:
        return None

    normalized_config = _normalize_model_config_for_hash(config, root=True)
    config_str = json.dumps(normalized_config, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def compute_task_hash(config: dict[str, Any] | None) -> str | None:
    """Compute a deterministic hash from a task configuration dict.

    Args:
        config: Task configuration dictionary.

    Returns:
        16-character hex string hash of the config, or None if config is None.
    """
    if not config:
        return None

    config_str = json.dumps(config, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


class Split(StrEnum):
    """Dataset split identifiers."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    ALL = "all"


class MetricName(StrEnum):
    """Standard metric identifiers."""

    ACCURACY = "accuracy"
    ACC_PER_CHAR = "acc_per_char"
    ACC_PER_TOKEN = "acc_per_token"
    EXACT_MATCH = "exact_match"
    PASS_AT_1 = "pass_at_1"
    PASS_AT_K = "pass_at_k"
    F1 = "f1"


class RequestType(StrEnum):
    """Type of request to send to the LM."""

    CHAT = "chat"
    COMPLETION = "completion"
    LOGLIKELIHOOD = "loglikelihood"


class TopLogProb(TypedDict):
    """A single top logprob alternative."""

    token: str
    logprob: float
    bytes: NotRequired[list[int]]


class LogProbEntry(TypedDict):
    """A single logprob entry for a token.

    Compatible with OpenAI's ChatCompletionTokenLogprob format.
    The bytes and top_logprobs fields are optional for backward compatibility.
    """

    token: str
    logprob: float
    token_id: NotRequired[int]
    bytes: NotRequired[list[int]]
    top_logprobs: NotRequired[list[TopLogProb]]


@dataclass(frozen=True, slots=True)
class Instance:
    """A single evaluation instance.

    The base fields (question, gold_answer, choices, metadata) support
    traditional evaluation. The tool-related fields support agent and
    tool calling evaluation.
    """

    question: str
    gold_answer: str | None = None
    choices: tuple[str, ...] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    tools: tuple[ToolSchema, ...] | None = None
    expected_tool_calls: tuple[dict[str, Any], ...] | None = None
    should_abstain: bool | None = None
    required_trajectory: tuple[dict[str, Any], ...] | None = None
    initial_state: dict[str, Any] | None = None
    expected_final_state: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class LMRequest:
    """Request to send to a language model."""

    request_type: RequestType
    messages: tuple[dict[str, Any], ...] = ()
    prompt: str = ""
    continuations: tuple[str, ...] | None = None
    continuation_prompts: tuple[str, ...] | None = None
    tools: tuple[ToolSchema, ...] | None = None
    system_prompt: str | None = None
    max_length: int | None = None
    images: tuple[Any, ...] | None = None
    #: Per-instance payload for a scaffold that needs more than the messages,
    #: such as the state a multi-turn rollout acts on. Not sent to the model.
    metadata: dict[str, Any] | None = None


@hide_unset()
@dataclass(frozen=True, slots=True)
class SamplingParams:
    """Parameters for language model sampling."""

    max_tokens: int | None = 512  # None means generate to the model's context limit
    temperature: float = 0.0
    top_p: float | None = None
    top_k: int | None = None
    stop_sequences: tuple[str, ...] | None = None
    num_samples: int = 1
    logprobs: int | None = None
    do_sample: bool = True
    truncate_prompt_tokens: int | None = None
    truncation_side: Literal["left", "right"] | None = None
    #: Shrink ``max_tokens`` so the prompt and the generation together fit the
    #: model's context window. Without it a prompt that leaves less room than
    #: ``max_tokens`` makes the server reject the whole request.
    fit_max_tokens_to_context: bool = False


@dataclass(slots=True)
class LMOutput:
    """Output from a language model.

    Supports both text generation and tool calling outputs.
    """

    text: str
    logprobs: list[LogProbEntry] | None = None
    extracted_answer: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    tool_calls: list[ToolCall] | None = None

    @property
    def has_tool_calls(self) -> bool:
        """Check if this output contains tool calls."""
        return self.tool_calls is not None and len(self.tool_calls) > 0


@dataclass(slots=True)
class Response:
    """Complete response pairing instance, request, and outputs.

    For multi-turn agent evaluations, the trajectory field contains
    the complete interaction history.
    """

    instance: Instance
    request: LMRequest
    outputs: list[LMOutput] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    trajectory: AgentTrajectory | None = None
    request_trace: dict[str, Any] | None = None
