"""Utilities for the OLMo-core inference provider."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TextIO, cast

if TYPE_CHECKING:
    import torch


class CachedPathResult(Protocol):
    def open(self) -> TextIO: ...


CachedPath = Callable[[str], CachedPathResult]


class TokenizerProtocol(Protocol):
    pad_token_id: int | None
    eos_token_id: int | None
    model_max_length: int
    add_bos_token: bool

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...

    def decode(self, token_ids: int | list[int], skip_special_tokens: bool = True) -> str: ...

    def apply_chat_template(
        self,
        messages: Sequence[Mapping[str, object]],
        **kwargs: object,
    ) -> str: ...


class TokenizerConfigProtocol(Protocol):
    identifier: str | None
    pad_token_id: int | None
    eos_token_id: int | None


class AutoTokenizerFactory(Protocol):
    def from_pretrained(self, tokenizer_path: str, **kwargs: object) -> TokenizerProtocol: ...


class TokenizerConfigFactory(Protocol):
    def from_dict(self, data: object) -> TokenizerConfigProtocol: ...

    def dolma2(self) -> TokenizerConfigProtocol: ...


class GenerationConfigFactory(Protocol):
    def __call__(
        self,
        *,
        pad_token_id: int,
        eos_token_id: int,
        use_cache: bool,
    ) -> object: ...


class AttentionBackendFactory(Protocol):
    def __call__(self, backend: str) -> object: ...


class TensorRows(Protocol):
    shape: tuple[int, ...]

    def tolist(self) -> list[list[int]] | list[list[float]]: ...


class GenerationModuleProtocol(Protocol):
    def free_inference_cache(self) -> None: ...

    def generate_batch(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_logprobs: bool,
        completions_only: bool,
        log_timing: bool,
        **generation_kwargs: object,
    ) -> tuple[TensorRows, object | None, TensorRows | None]: ...

    def model_forward(self, *, input_ids: torch.Tensor) -> torch.Tensor: ...


class TransformerGenerationModuleFactory(Protocol):
    def from_checkpoint(self, **kwargs: object) -> GenerationModuleProtocol: ...


class CheckpointMetadataProtocol(Protocol):
    state_dict_metadata: Mapping[str, object]


class CudaModuleProtocol(Protocol):
    def is_available(self) -> bool: ...

    def get_device_capability(self) -> tuple[int, int]: ...


class TorchModuleProtocol(Protocol):
    cuda: CudaModuleProtocol

    def device(self, device: str) -> torch.device: ...


# Transformers uses int(1e30) as a sentinel when tokenizer.model_max_length is unset.
_TRANSFORMERS_UNSET_MODEL_MAX_LENGTH = int(1e30)
_EXPECTED_CHECKPOINT_FORMAT = (
    "expected a raw OLMo-core checkpoint with config.json containing 'model' and "
    "'dataset.tokenizer', plus distributed checkpoint metadata at either "
    "'model_and_optim/.metadata' or '.metadata'. HF-format checkpoints should use "
    "the 'hf', 'vllm', or 'vllm_server' provider instead"
)
_MAX_LENGTH_CONFIG_KEYS = ("max_sequence_length", "max_seq_len", "max_position_embeddings")
_MODULE_KWARG_NAMES = ("float8_config", "state_dict_load_opts", "load_key_mapping")


@dataclass(frozen=True)
class OlmoCoreImports:
    AutoTokenizer: AutoTokenizerFactory
    AttentionBackendName: AttentionBackendFactory
    GenerationConfig: GenerationConfigFactory
    TokenizerConfig: TokenizerConfigFactory
    TransformerGenerationModule: TransformerGenerationModuleFactory
    cached_path: CachedPath
    get_checkpoint_metadata: Callable[[str], CheckpointMetadataProtocol]
    torch: TorchModuleProtocol


@dataclass(frozen=True)
class CheckpointInfo:
    config: dict[str, object]
    tokenizer_config: TokenizerConfigProtocol
    metadata_dir: str | None = None


@dataclass(frozen=True)
class LogprobInput:
    input_ids: list[int]
    input_length: int
    num_tokens_all: int
    continuation_token_ids: list[int]
    continuation: str
    prompt_truncated_tokens: int = 0


def _import_olmo_core() -> OlmoCoreImports:
    try:
        import torch
        from cached_path import cached_path
        from olmo_core.data import TokenizerConfig
        from olmo_core.distributed.checkpoint import get_checkpoint_metadata
        from olmo_core.generate.generation_module import (
            GenerationConfig,
            TransformerGenerationModule,
        )
        from olmo_core.nn.attention import AttentionBackendName
        from transformers import AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "ai2-olmo-core and transformers are required for OlmoCoreProvider. "
            "Install with: pip install 'olmo-eval[olmo_core]'"
        ) from e

    return OlmoCoreImports(
        AutoTokenizer=cast(AutoTokenizerFactory, AutoTokenizer),
        AttentionBackendName=cast(AttentionBackendFactory, AttentionBackendName),
        GenerationConfig=cast(GenerationConfigFactory, GenerationConfig),
        TokenizerConfig=cast(TokenizerConfigFactory, TokenizerConfig),
        TransformerGenerationModule=cast(
            TransformerGenerationModuleFactory,
            TransformerGenerationModule,
        ),
        cached_path=cast(CachedPath, cached_path),
        get_checkpoint_metadata=cast(
            Callable[[str], CheckpointMetadataProtocol],
            get_checkpoint_metadata,
        ),
        torch=cast(TorchModuleProtocol, torch),
    )


def _is_remote_path(path: str) -> bool:
    return "://" in path


def _join_checkpoint_path(checkpoint_dir: str, *parts: str) -> str:
    if _is_remote_path(checkpoint_dir):
        return "/".join([checkpoint_dir.rstrip("/"), *parts])
    return str(Path(checkpoint_dir, *parts))


def _checkpoint_value_error(checkpoint_dir: str, reason: str) -> ValueError:
    return ValueError(
        f"Invalid OLMo-core checkpoint {checkpoint_dir!r}: {reason}. {_EXPECTED_CHECKPOINT_FORMAT}."
    )


def _read_checkpoint_config(
    checkpoint_dir: str,
    *,
    cached_path: CachedPath | None = None,
) -> dict[str, object]:
    config_path = _join_checkpoint_path(checkpoint_dir, "config.json")
    try:
        if _is_remote_path(checkpoint_dir):
            if cached_path is None:
                raise FileNotFoundError(config_path)
            with cached_path(config_path).open() as f:
                return cast(dict[str, object], json.load(f))
        with Path(config_path).open() as f:
            return cast(dict[str, object], json.load(f))
    except FileNotFoundError as e:
        raise _checkpoint_value_error(checkpoint_dir, "missing config.json") from e
    except json.JSONDecodeError as e:
        raise _checkpoint_value_error(checkpoint_dir, f"config.json is not valid JSON: {e}") from e


def _validate_token_ids(
    *,
    checkpoint_dir: str,
    pad_token_id: int | None,
    eos_token_id: int | None,
) -> tuple[int, int]:
    if pad_token_id is None:
        raise _checkpoint_value_error(checkpoint_dir, "missing pad_token_id")
    if eos_token_id is None:
        raise _checkpoint_value_error(checkpoint_dir, "missing eos_token_id")
    if pad_token_id < 0:
        raise _checkpoint_value_error(
            checkpoint_dir,
            f"pad_token_id must be >= 0, got {pad_token_id}",
        )
    if eos_token_id < 0:
        raise _checkpoint_value_error(
            checkpoint_dir,
            f"eos_token_id must be >= 0, got {eos_token_id}",
        )
    if pad_token_id == eos_token_id:
        raise _checkpoint_value_error(
            checkpoint_dir,
            f"pad_token_id and eos_token_id must be different, got {pad_token_id}",
        )
    return pad_token_id, eos_token_id


def _tokenizer_config_from_checkpoint_config(
    checkpoint_dir: str,
    config: Mapping[str, object],
    *,
    TokenizerConfig: TokenizerConfigFactory,
) -> TokenizerConfigProtocol:
    try:
        dataset = config["dataset"]
        if not isinstance(dataset, Mapping):
            raise TypeError("dataset is not an object")
        dataset = cast(Mapping[str, object], dataset)
        return TokenizerConfig.from_dict(dataset["tokenizer"])
    except KeyError as e:
        raise _checkpoint_value_error(
            checkpoint_dir,
            f"config.json missing required field {e}",
        ) from e
    except Exception as e:
        raise _checkpoint_value_error(
            checkpoint_dir,
            f"config.json field 'dataset.tokenizer' is not a valid OLMo-core TokenizerConfig: {e}",
        ) from e


def _validate_olmo_core_checkpoint(
    checkpoint_dir: str,
    *,
    TokenizerConfig: TokenizerConfigFactory,
    get_checkpoint_metadata: Callable[[str], CheckpointMetadataProtocol],
    cached_path: CachedPath | None = None,
) -> CheckpointInfo:
    config = _read_checkpoint_config(checkpoint_dir, cached_path=cached_path)
    try:
        model_config = config["model"]
    except KeyError as e:
        raise _checkpoint_value_error(
            checkpoint_dir,
            f"config.json missing required field {e}",
        ) from e

    if not isinstance(model_config, Mapping):
        raise _checkpoint_value_error(
            checkpoint_dir,
            "config.json field 'model' must be an object",
        )

    tokenizer_config = _tokenizer_config_from_checkpoint_config(
        checkpoint_dir,
        config,
        TokenizerConfig=TokenizerConfig,
    )

    _validate_token_ids(
        checkpoint_dir=checkpoint_dir,
        pad_token_id=getattr(tokenizer_config, "pad_token_id", None),
        eos_token_id=getattr(tokenizer_config, "eos_token_id", None),
    )

    metadata = None
    metadata_dir = None
    metadata_errors: list[str] = []
    for candidate in (
        _join_checkpoint_path(checkpoint_dir, "model_and_optim"),
        checkpoint_dir,
    ):
        try:
            metadata = get_checkpoint_metadata(candidate)
            metadata_dir = candidate
            break
        except FileNotFoundError as e:
            metadata_errors.append(str(e))
        except Exception as e:
            raise _checkpoint_value_error(
                checkpoint_dir,
                f"could not read distributed checkpoint metadata at {candidate!r}: {e}",
            ) from e

    if metadata is None:
        detail = "; ".join(error for error in metadata_errors if error)
        reason = "missing distributed checkpoint metadata"
        if detail:
            reason = f"{reason}: {detail}"
        raise _checkpoint_value_error(checkpoint_dir, reason)

    state_metadata = getattr(metadata, "state_dict_metadata", {}) or {}
    if not any(str(key).startswith("model") for key in state_metadata):
        raise _checkpoint_value_error(
            checkpoint_dir,
            "distributed checkpoint metadata does not contain model state keys",
        )

    return CheckpointInfo(
        config=config,
        tokenizer_config=tokenizer_config,
        metadata_dir=metadata_dir,
    )


def _load_checkpoint_config_and_tokenizer_config(
    checkpoint_dir: str,
    *,
    TokenizerConfig: TokenizerConfigFactory,
    cached_path: CachedPath | None = None,
) -> tuple[dict[str, object], TokenizerConfigProtocol]:
    config = _read_checkpoint_config(checkpoint_dir, cached_path=cached_path)
    return config, _tokenizer_config_from_checkpoint_config(
        checkpoint_dir,
        config,
        TokenizerConfig=TokenizerConfig,
    )


def _resolve_checkpoint(
    checkpoint_dir: str,
    *,
    imports: OlmoCoreImports,
    validate_checkpoint: bool,
    allow_tokenizer_fallback: bool,
) -> tuple[dict[str, object] | None, TokenizerConfigProtocol]:
    if validate_checkpoint:
        checkpoint_info = _validate_olmo_core_checkpoint(
            checkpoint_dir,
            TokenizerConfig=imports.TokenizerConfig,
            get_checkpoint_metadata=imports.get_checkpoint_metadata,
            cached_path=imports.cached_path,
        )
        return checkpoint_info.config, checkpoint_info.tokenizer_config

    try:
        config, tokenizer_config = _load_checkpoint_config_and_tokenizer_config(
            checkpoint_dir,
            TokenizerConfig=imports.TokenizerConfig,
            cached_path=imports.cached_path,
        )
    except ValueError:
        if not allow_tokenizer_fallback:
            raise
        return None, imports.TokenizerConfig.dolma2()
    return config, tokenizer_config


def _resolve_max_model_len_alias(
    max_model_len: int | None,
    kwargs: dict[str, object],
    provider: str = "OlmoCoreProvider",
) -> int | None:
    max_length = kwargs.pop("max_length", None)
    if max_length is None:
        return max_model_len
    if not isinstance(max_length, int):
        raise ValueError(f"{provider} max_length must be an integer when set")
    if max_model_len is not None and max_model_len != max_length:
        raise ValueError(
            f"{provider} received both max_model_len and max_length with different values"
        )
    return max_length


def _validate_max_model_len(max_model_len: int | None) -> None:
    if max_model_len is not None and max_model_len <= 0:
        raise ValueError("OlmoCoreProvider max_model_len must be positive when set")


def _validate_batch_size(batch_size: int | None) -> None:
    if batch_size is not None and (not isinstance(batch_size, int) or batch_size <= 0):
        raise ValueError("OlmoCoreProvider batch_size must be a positive integer or None")


def _validate_tensor_parallel_size(tensor_parallel_size: object) -> None:
    if tensor_parallel_size in (None, 1):
        return
    raise ValueError(
        "OlmoCoreProvider only supports tensor_parallel_size of 1 or None. "
        "ai2-olmo-core 2.4.0 generation exposes data-parallel process groups, "
        "but not tensor-parallel generation config; run multiple provider "
        "processes instead."
    )


def _pop_module_kwargs(kwargs: dict[str, object]) -> dict[str, object]:
    return {key: kwargs.pop(key) for key in _MODULE_KWARG_NAMES if key in kwargs}


def _raise_for_unsupported_kwargs(kwargs: Mapping[str, object]) -> None:
    if kwargs:
        unsupported = ", ".join(sorted(kwargs))
        raise ValueError(f"Unsupported OlmoCoreProvider kwargs: {unsupported}")


def _resolve_tokenizer_path(
    checkpoint_dir: str,
    *,
    explicit_tokenizer: str | None,
    tokenizer_config: TokenizerConfigProtocol,
    TokenizerConfig: TokenizerConfigFactory,
    allow_tokenizer_fallback: bool,
) -> tuple[str, TokenizerConfigProtocol]:
    if explicit_tokenizer is not None:
        return explicit_tokenizer, tokenizer_config

    tokenizer_path = getattr(tokenizer_config, "identifier", None)
    if tokenizer_path is not None:
        return tokenizer_path, tokenizer_config

    if not allow_tokenizer_fallback:
        raise _checkpoint_value_error(
            checkpoint_dir,
            "checkpoint tokenizer config does not include an identifier",
        )

    fallback_config = TokenizerConfig.dolma2()
    if fallback_config.identifier is None:
        raise _checkpoint_value_error(
            checkpoint_dir,
            "fallback tokenizer config does not include an identifier",
        )
    return fallback_config.identifier, fallback_config


def _preferred_token_id(
    explicit_token_id: int | None,
    *,
    tokenizer: TokenizerProtocol,
    tokenizer_config: TokenizerConfigProtocol,
    attr: str,
) -> int | None:
    if explicit_token_id is not None:
        return explicit_token_id
    tokenizer_token_id = getattr(tokenizer, attr, None)
    if isinstance(tokenizer_token_id, int):
        return tokenizer_token_id
    config_token_id = getattr(tokenizer_config, attr, None)
    return config_token_id if isinstance(config_token_id, int) else None


def _valid_tokenizer_model_max_length(tokenizer: TokenizerProtocol) -> int | None:
    model_max_length = getattr(tokenizer, "model_max_length", None)
    if not isinstance(model_max_length, int) or model_max_length <= 0:
        return None
    if model_max_length >= _TRANSFORMERS_UNSET_MODEL_MAX_LENGTH:
        return None
    return model_max_length


def _resolve_attention_backend(
    attention_backend: str | None,
    *,
    AttentionBackendName: AttentionBackendFactory,
) -> object | None:
    if attention_backend is None:
        return None
    return AttentionBackendName(attention_backend)


def _resolve_max_length(
    *,
    explicit_max_length: int | None,
    tokenizer: TokenizerProtocol,
    checkpoint_config: Mapping[str, object] | None,
    checkpoint_dir: str,
) -> int:
    if explicit_max_length is not None:
        return explicit_max_length

    model_config_value = checkpoint_config.get("model", {}) if checkpoint_config else {}
    model_config = (
        cast(Mapping[str, object], model_config_value)
        if isinstance(model_config_value, Mapping)
        else {}
    )
    for key in _MAX_LENGTH_CONFIG_KEYS:
        value = model_config.get(key)
        if isinstance(value, int) and value > 0:
            return value

    tokenizer_model_max_length = _valid_tokenizer_model_max_length(tokenizer)
    if tokenizer_model_max_length is not None:
        return tokenizer_model_max_length

    raise ValueError(
        "Could not determine OLMo-core max_model_len for checkpoint "
        f"{checkpoint_dir!r}: pass max_model_len explicitly, set one of "
        "model.max_sequence_length, model.max_seq_len, or model.max_position_embeddings "
        "in config.json, or use a tokenizer with a real model_max_length."
    )
