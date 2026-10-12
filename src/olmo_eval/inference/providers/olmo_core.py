"""OLMo-core inference provider."""

from __future__ import annotations

import asyncio
import gc
import logging
from contextlib import suppress
from dataclasses import replace
from typing import TYPE_CHECKING, Literal, cast

import olmo_eval.inference.providers.olmo_core_utils as core_utils
from olmo_eval.common.debug import is_debug_requests
from olmo_eval.common.types import LMOutput, LMRequest, LogProbEntry, RequestType, SamplingParams
from olmo_eval.inference.base import InferenceProvider
from olmo_eval.inference.tokenizer_utils import (
    encode_context_and_continuation,
    get_bos_token_ids,
)

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

_OLMO_CORE_GENERATION_LOGGER = "olmo_core.generate.generation_module.transformer.generation_module"
_OLMO_CORE_CONFIG_LOG_PREFIXES = ("TransformerConfig(", "GenerationConfig(")


class _OlmoCoreConfigLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not record.getMessage().startswith(_OLMO_CORE_CONFIG_LOG_PREFIXES)


class _TokenizerBosOverride:
    """Tokenizer proxy that gives logprob encoding the provider's BOS setting."""

    def __init__(self, tokenizer: core_utils.TokenizerProtocol, add_bos_token: bool) -> None:
        self._tokenizer = tokenizer
        self.add_bos_token = add_bos_token

    def __getattr__(self, name: str) -> object:
        return getattr(self._tokenizer, name)


class OlmoCoreProvider(InferenceProvider):
    """Provider using OLMo-core's in-process generation module."""

    # Experimental for now; keep this path close to OLMo-core's public behavior.
    def __init__(
        self,
        model_name: str,
        tokenizer: str | None = None,
        *,
        dtype: str = "bfloat16",
        max_model_len: int | None = None,
        attention_backend: str | None = None,
        use_cache: bool = True,
        compile_model: bool = False,
        batch_size: int | None = None,
        validate_checkpoint: bool = True,
        allow_tokenizer_fallback: bool = False,
        chat_template: str | None = None,
        pad_token_id: int | None = None,
        eos_token_id: int | None = None,
        device: str | None = None,
        load_thread_count: int | None = None,
        pre_download: bool = True,
        work_dir: str | None = None,
        revision: str | None = None,
        force_download: bool = False,
        trust_remote_code: bool = False,
        token: str | None = None,
        cache_dir: str | None = None,
        local_files_only: bool = False,
        add_bos_token: bool = False,
        **kwargs: object,
    ) -> None:
        max_model_len = core_utils._resolve_max_model_len_alias(max_model_len, kwargs)
        core_utils._validate_max_model_len(max_model_len)
        core_utils._validate_batch_size(batch_size)
        core_utils._validate_tensor_parallel_size(kwargs.pop("tensor_parallel_size", 1))

        module_kwargs = core_utils._pop_module_kwargs(kwargs)
        core_utils._raise_for_unsupported_kwargs(kwargs)

        imports = core_utils._import_olmo_core()
        super().__init__(model_name)

        checkpoint_config, tokenizer_config = core_utils._resolve_checkpoint(
            model_name,
            imports=imports,
            validate_checkpoint=validate_checkpoint,
            allow_tokenizer_fallback=allow_tokenizer_fallback,
        )
        tokenizer_path, tokenizer_config = core_utils._resolve_tokenizer_path(
            model_name,
            explicit_tokenizer=tokenizer,
            tokenizer_config=tokenizer_config,
            TokenizerConfig=imports.TokenizerConfig,
            allow_tokenizer_fallback=allow_tokenizer_fallback,
        )

        tokenizer_kwargs = {
            key: value
            for key, value in {
                "revision": revision,
                "force_download": force_download,
                "trust_remote_code": trust_remote_code,
                "token": token,
                "cache_dir": cache_dir,
                "local_files_only": local_files_only,
            }.items()
            if value
        }
        self.tokenizer: core_utils.TokenizerProtocol = imports.AutoTokenizer.from_pretrained(
            tokenizer_path,
            **tokenizer_kwargs,
        )
        self.add_bos_token = add_bos_token

        resolved_pad_token_id, resolved_eos_token_id = core_utils._validate_token_ids(
            checkpoint_dir=model_name,
            pad_token_id=core_utils._preferred_token_id(
                pad_token_id,
                tokenizer=self.tokenizer,
                tokenizer_config=tokenizer_config,
                attr="pad_token_id",
            ),
            eos_token_id=core_utils._preferred_token_id(
                eos_token_id,
                tokenizer=self.tokenizer,
                tokenizer_config=tokenizer_config,
                attr="eos_token_id",
            ),
        )
        self.pad_token_id = resolved_pad_token_id
        self.eos_token_id = resolved_eos_token_id
        if getattr(self.tokenizer, "pad_token_id", None) is None:
            self.tokenizer.pad_token_id = resolved_pad_token_id
        if getattr(self.tokenizer, "eos_token_id", None) is None:
            self.tokenizer.eos_token_id = resolved_eos_token_id
        self.use_cache = use_cache
        self.batch_size = batch_size
        self.chat_template = chat_template
        self.max_length = core_utils._resolve_max_length(
            explicit_max_length=max_model_len,
            tokenizer=self.tokenizer,
            checkpoint_config=checkpoint_config,
            checkpoint_dir=model_name,
        )

        self.device = imports.torch.device(
            device or ("cuda" if imports.torch.cuda.is_available() else "cpu")
        )
        attention_backend_value = core_utils._resolve_attention_backend(
            attention_backend,
            AttentionBackendName=imports.AttentionBackendName,
        )
        transformer_config = self._build_transformer_config_for_display(
            checkpoint_config=checkpoint_config,
            dtype=dtype,
            attention_backend=attention_backend_value,
        )

        self.generation_config = imports.GenerationConfig(
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
            use_cache=use_cache,
        )
        self._print_olmo_core_config(transformer_config, self.generation_config)

        load_kwargs = {
            "checkpoint_dir": model_name,
            "generation_config": self.generation_config,
            "device": self.device,
            "pre_download": pre_download,
            "load_thread_count": load_thread_count,
            "work_dir": work_dir,
            "attention_backend": attention_backend_value,
            "compile_model": compile_model,
            **module_kwargs,
        }
        if dtype != "auto":
            load_kwargs["dtype"] = dtype

        config_log_filter = _OlmoCoreConfigLogFilter()
        olmo_core_logger = logging.getLogger(_OLMO_CORE_GENERATION_LOGGER)
        olmo_core_logger.addFilter(config_log_filter)
        try:
            self.generation_module: core_utils.GenerationModuleProtocol = (
                imports.TransformerGenerationModule.from_checkpoint(
                    **{key: value for key, value in load_kwargs.items() if value is not None}
                )
            )
        finally:
            olmo_core_logger.removeFilter(config_log_filter)

    def get_tokenizer(self) -> core_utils.TokenizerProtocol:
        return self.tokenizer

    @staticmethod
    def _build_transformer_config_for_display(
        *,
        checkpoint_config: dict[str, object] | None,
        dtype: str,
        attention_backend: object | None,
    ) -> object | None:
        if checkpoint_config is None:
            return None
        model_config = checkpoint_config.get("model")
        if not isinstance(model_config, dict):
            return None

        try:
            from olmo_core.config import DType
            from olmo_core.nn.transformer import TransformerConfig

            transformer_config = TransformerConfig.from_dict(model_config)
            if dtype != "auto":
                dtype_value = DType(dtype)
                transformer_config.apply(
                    lambda config: (
                        setattr(config, "dtype", dtype_value) if hasattr(config, "dtype") else None
                    )
                )
            if attention_backend is not None:

                def set_attention_backend(config: object) -> None:
                    mixer = getattr(config, "sequence_mixer", None) or getattr(
                        config, "attention", None
                    )
                    if mixer is None and hasattr(config, "backend"):
                        mixer = config
                    if mixer is not None and hasattr(mixer, "backend"):
                        setattr(mixer, "backend", attention_backend)  # noqa: B010

                transformer_config.apply(set_attention_backend)
            return transformer_config
        except Exception:
            logger.debug("Could not build OLMo-core TransformerConfig for rich display.")
            return model_config

    @staticmethod
    def _config_display_value(config: object) -> object:
        as_dict = getattr(config, "as_dict", None)
        if callable(as_dict):
            with suppress(Exception):
                return as_dict()
        return config

    @classmethod
    def _print_olmo_core_config(
        cls,
        transformer_config: object | None,
        generation_config: object,
    ) -> None:
        if not logger.isEnabledFor(logging.INFO):
            return

        from rich.panel import Panel
        from rich.pretty import Pretty

        from olmo_eval.common.console import console

        payload = {
            "generation": cls._config_display_value(generation_config),
        }
        if transformer_config is not None:
            payload = {
                "transformer": cls._config_display_value(transformer_config),
                **payload,
            }

        console.print(
            Panel(
                Pretty(payload, expand_all=True),
                title="[bold]OLMo-core Configuration[/bold]",
                border_style="cyan",
            )
        )

    def _free_inference_cache(self) -> None:
        self.generation_module.free_inference_cache()

    def _iter_chunks(self, requests: list[LMRequest]) -> list[list[LMRequest]]:
        if self.batch_size is None:
            return [requests]
        return [
            requests[start : start + self.batch_size]
            for start in range(0, len(requests), self.batch_size)
        ]

    def _encode_prompt(self, prompt: str) -> list[int]:
        token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if self.add_bos_token:
            return get_bos_token_ids(self.tokenizer, fallback_to_eos=True) + token_ids
        return token_ids

    def _format_prompt(self, request: LMRequest) -> str:
        if request.request_type == RequestType.CHAT and request.messages:
            if not hasattr(self.tokenizer, "apply_chat_template"):
                raise ValueError("CHAT requests require a tokenizer with apply_chat_template")
            kwargs: dict[str, object] = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            if self.chat_template is not None:
                kwargs["chat_template"] = self.chat_template
            return self.tokenizer.apply_chat_template(list(request.messages), **kwargs)
        return request.prompt

    def _logprob_tokenizer(self) -> core_utils.TokenizerProtocol:
        return cast(
            core_utils.TokenizerProtocol,
            _TokenizerBosOverride(self.tokenizer, self.add_bos_token),
        )

    def _encode_logprob_context_and_continuation(
        self,
        prompt: str,
        continuation: str,
    ) -> tuple[list[int], list[int]]:
        context_ids, continuation_ids = encode_context_and_continuation(
            self._logprob_tokenizer(),
            prompt,
            continuation,
        )
        if continuation_ids and not context_ids:
            context_ids = get_bos_token_ids(self.tokenizer, fallback_to_eos=True) or [
                self.eos_token_id
            ]
        return context_ids, continuation_ids

    def _pad_sequences(
        self,
        sequences: list[list[int]],
        *,
        side: Literal["left", "right"],
        return_attention_mask: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        import torch

        max_len = max(max((len(seq) for seq in sequences), default=0), 1)
        input_ids = torch.full(
            (len(sequences), max_len),
            self.pad_token_id,
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = (
            torch.zeros(
                (len(sequences), max_len),
                dtype=torch.long,
                device=self.device,
            )
            if return_attention_mask
            else None
        )
        for idx, seq in enumerate(sequences):
            if not seq:
                continue
            seq_tensor = torch.tensor(seq, dtype=torch.long, device=self.device)
            target = slice(-len(seq), None) if side == "left" else slice(0, len(seq))
            input_ids[idx, target] = seq_tensor
            if attention_mask is not None:
                attention_mask[idx, target] = 1
        return input_ids, attention_mask

    def _left_pad(self, sequences: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids, attention_mask = self._pad_sequences(
            sequences,
            side="left",
            return_attention_mask=True,
        )
        assert attention_mask is not None
        return input_ids, attention_mask

    def _right_pad(self, sequences: list[list[int]]) -> torch.Tensor:
        input_ids, _ = self._pad_sequences(sequences, side="right")
        return input_ids

    def _build_generation_kwargs(self, params: SamplingParams) -> dict[str, object]:
        # ``max_tokens=None`` means "generate to the context limit"; the
        # generation path resolves it per prompt via ``_resolve_max_tokens``,
        # and the request-trace path never resolves it at all.
        if params.max_tokens is not None and params.max_tokens <= 0:
            raise ValueError("OlmoCoreProvider requires sampling max_tokens > 0")
        if params.num_samples <= 0:
            raise ValueError("OlmoCoreProvider requires sampling num_samples > 0")

        do_sample = params.do_sample and params.temperature > 0
        return {
            "do_sample": do_sample,
            "temperature": params.temperature if do_sample else 0.0,
            "top_k": params.top_k if do_sample and params.top_k is not None else -1,
            "top_p": params.top_p if do_sample and params.top_p is not None else 1.0,
            "use_cache": self.use_cache,
        }

    def _decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def _text_before_stop(self, text: str, stop_sequences: tuple[str, ...]) -> str | None:
        stop_idx: int | None = None
        for stop in stop_sequences:
            if stop in text:
                idx = text.find(stop)
                if stop_idx is None or idx < stop_idx:
                    stop_idx = idx
        if stop_idx is None:
            return None
        return text[:stop_idx]

    def _stop_sequences_with_eos(self, stop_sequences: tuple[str, ...] | None) -> tuple[str, ...]:
        stops = [stop for stop in stop_sequences or () if stop]
        eos = self._decode([self.eos_token_id], skip_special_tokens=False)
        if eos and eos not in stops:
            stops.append(eos)
        return tuple(stops)

    def _resolve_max_tokens(
        self,
        params: SamplingParams,
        prompt_token_ids: list[list[int]],
    ) -> SamplingParams:
        """Resolve an uncapped (``max_tokens=None``) request to a concrete budget.

        "Uncapped" means generate to the model's context limit, so reserve the
        room left after the longest prompt. Downstream length validation and
        truncation then operate on a concrete int.
        """
        if params.max_tokens is not None:
            return params
        longest_prompt = max((len(ids) for ids in prompt_token_ids), default=0)
        budget = self.max_length - longest_prompt
        if budget <= 0:
            raise ValueError(
                f"OlmoCoreProvider cannot generate uncapped: longest prompt "
                f"({longest_prompt}) leaves no room within max_length ({self.max_length})."
            )
        return replace(params, max_tokens=budget)

    def _validate_generation_lengths(
        self,
        prompt_token_ids: list[list[int]],
        params: SamplingParams,
    ) -> None:
        for request_idx, token_ids in enumerate(prompt_token_ids):
            requested_length = len(token_ids) + params.max_tokens
            if requested_length > self.max_length:
                raise ValueError(
                    "OLMo-core generation request exceeds max_length for request "
                    f"{request_idx}: prompt length ({len(token_ids)}) + max_tokens "
                    f"({params.max_tokens}) = {requested_length}, but max_length is "
                    f"{self.max_length}."
                )

    def _left_truncate_prompts_for_generation(
        self,
        prompt_token_ids: list[list[int]],
        params: SamplingParams,
    ) -> list[list[int]]:
        max_context_len = self.max_length - params.max_tokens
        if max_context_len <= 0:
            raise ValueError(
                f"max_tokens ({params.max_tokens}) is greater than or equal to max_length "
                f"({self.max_length}) for OlmoCoreProvider generation."
            )
        return [token_ids[-max_context_len:] for token_ids in prompt_token_ids]

    def _normalize_generation_output(
        self,
        token_ids: list[int],
        token_logprobs: list[float] | None,
        stop_sequences: tuple[str, ...] | None,
    ) -> tuple[list[int], list[float] | None, str]:
        end = len(token_ids)
        for idx, token_id in enumerate(token_ids):
            if token_id in {self.eos_token_id, self.pad_token_id}:
                end = idx if token_id == self.pad_token_id else idx + 1
                break

        token_ids = token_ids[:end]
        if token_logprobs is not None:
            token_logprobs = token_logprobs[:end]

        stop_sequences = tuple(stop for stop in stop_sequences or () if stop)
        decoded = self._decode(token_ids, skip_special_tokens=True)
        text = self._text_before_stop(decoded, stop_sequences)
        return token_ids, token_logprobs, text if text is not None else decoded

    def _logprob_entries(
        self,
        token_ids: list[int],
        token_logprobs: list[float],
    ) -> list[LogProbEntry]:
        entries: list[LogProbEntry] = []
        for token_id, logprob in zip(token_ids, token_logprobs, strict=True):
            token_str = self._decode([token_id], skip_special_tokens=False)
            entries.append(
                {
                    "token": token_str,
                    "logprob": float(logprob),
                    "token_id": int(token_id),
                    "bytes": list(token_str.encode("utf-8")),
                }
            )
        return entries

    def _generation_output(
        self,
        token_ids: list[int],
        token_logprobs: list[float] | None,
        text: str,
    ) -> LMOutput:
        entries = (
            self._logprob_entries(token_ids, token_logprobs) if token_logprobs is not None else None
        )
        metadata: dict[str, object] = {}
        if entries:
            total = sum(entry["logprob"] for entry in entries)
            metadata = {
                "sum_logits": total,
                "num_tokens": len(entries),
                "num_tokens_all": len(entries),
            }
        return LMOutput(
            text=text,
            logprobs=entries,
            metadata=metadata,
        )

    def generate(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        if not requests:
            return []

        params = self._default_sampling_params(sampling_params)
        if params.truncate_prompt_tokens is not None or params.truncation_side is not None:
            logger.warning(
                "truncate_prompt_tokens or truncation_side has been set in the params, "
                "but is not supported for the OlmoCoreProvider and will not be used."
            )
        results: list[list[LMOutput]] = []
        try:
            for chunk in self._iter_chunks(requests):
                results.extend(self._generate_chunk(chunk, params))
        except Exception:
            self._free_inference_cache()
            raise

        self._free_inference_cache()
        return results

    def _generate_chunk(
        self,
        requests: list[LMRequest],
        params: SamplingParams,
    ) -> list[list[LMOutput]]:
        prompt_strs = [self._format_prompt(request) for request in requests]
        if is_debug_requests():
            for i, prompt in enumerate(prompt_strs):
                logger.info(f"Prompt {i}:\n{prompt}")

        encoded_prompts = [self._encode_prompt(prompt) for prompt in prompt_strs]
        params = self._resolve_max_tokens(params, encoded_prompts)
        generation_kwargs = self._build_generation_kwargs(params)
        prompt_token_ids = self._left_truncate_prompts_for_generation(encoded_prompts, params)
        self._validate_generation_lengths(prompt_token_ids, params)
        expanded_token_ids = [
            token_ids for token_ids in prompt_token_ids for _ in range(params.num_samples)
        ]
        input_ids, attention_mask = self._left_pad(expanded_token_ids)
        prompt_len = input_ids.shape[1]
        generation_kwargs["max_length"] = prompt_len + params.max_tokens

        generated_ids, _, logprobs = self.generation_module.generate_batch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_logprobs=True,
            completions_only=False,
            log_timing=False,
            **generation_kwargs,
        )

        generated_rows = [row[prompt_len:] for row in cast(list[list[int]], generated_ids.tolist())]
        logprob_rows = (
            cast(list[list[float]], logprobs.tolist())
            if logprobs is not None
            else [None] * len(generated_rows)
        )

        results = [[] for _ in requests]
        for row_idx, (row_ids, row_logprobs) in enumerate(
            zip(generated_rows, logprob_rows, strict=True)
        ):
            request_idx = row_idx // params.num_samples
            token_ids, token_logprobs, text = self._normalize_generation_output(
                row_ids,
                row_logprobs,
                self._stop_sequences_with_eos(params.stop_sequences),
            )
            results[request_idx].append(self._generation_output(token_ids, token_logprobs, text))
        return results

    def describe_request(
        self,
        request: LMRequest,
        sampling_params: SamplingParams | None = None,
    ) -> dict[str, object] | None:
        params = self._default_sampling_params(sampling_params)
        trace = super().describe_request(request, sampling_params)
        if trace is None:
            return None

        trace["provider"] = "OlmoCoreProvider"
        if request.request_type == RequestType.LOGLIKELIHOOD:
            trace["endpoint"] = "generation_module.model_forward"
            trace["input_mode"] = "input_ids"
            trace["generation_kwargs"] = {
                "temperature": params.temperature,
            }
            trace["stop_sequences"] = []
            return trace

        trace["endpoint"] = "generation_module.generate_batch"
        trace["input_mode"] = "input_ids"
        trace["generation_kwargs"] = {
            **self._build_generation_kwargs(params),
            "num_samples": params.num_samples,
            "return_logprobs": True,
            "completions_only": False,
            "max_length": "<padded_prompt_length + max_tokens>",
        }
        trace["stop_sequences"] = list(params.stop_sequences or ())
        return trace

    def logprobs(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        params = self._default_sampling_params(sampling_params)
        if params.truncate_prompt_tokens is not None or params.truncation_side is not None:
            logger.warning(
                "truncate_prompt_tokens or truncation_side has been set in the params, "
                "but is not supported for the OlmoCoreProvider and will not be used."
            )
        del sampling_params
        del params
        if not requests:
            return []

        self._free_inference_cache()
        results: list[list[LMOutput]] = []
        for chunk in self._iter_chunks(requests):
            results.extend(self._logprobs_chunk(chunk))
        return results

    def _logprob_inputs_for_request(self, request: LMRequest) -> list[core_utils.LogprobInput]:
        max_len = request.max_length or self.max_length
        if max_len <= 0:
            raise ValueError("OlmoCoreProvider requires max_length > 0 for logprobs")

        rows: list[core_utils.LogprobInput] = []
        cont_prompts = request.continuation_prompts
        for i, continuation in enumerate(request.continuations or ()):
            prompt = cont_prompts[i] if cont_prompts else request.prompt
            context_ids, continuation_ids = self._encode_logprob_context_and_continuation(
                prompt,
                continuation,
            )
            if not continuation_ids:
                rows.append(
                    core_utils.LogprobInput(
                        input_ids=context_ids or [self.eos_token_id],
                        input_length=0,
                        num_tokens_all=len(context_ids),
                        continuation_token_ids=[],
                        continuation=continuation,
                    )
                )
                continue
            if len(continuation_ids) > max_len:
                raise ValueError(
                    "Continuation is longer than the OLMo-core provider max_length "
                    f"({len(continuation_ids)} > {max_len})"
                )

            full_ids = context_ids + continuation_ids
            truncated = full_ids[-(max_len + 1) :]
            model_input = truncated[:-1]
            rows.append(
                core_utils.LogprobInput(
                    input_ids=model_input,
                    input_length=len(model_input),
                    num_tokens_all=len(truncated),
                    continuation_token_ids=continuation_ids,
                    continuation=continuation,
                    prompt_truncated_tokens=len(full_ids) - len(truncated),
                )
            )
        return rows

    def _logprob_output_from_logits(
        self,
        row: core_utils.LogprobInput,
        logits: torch.Tensor,
    ) -> LMOutput:
        if not row.continuation_token_ids:
            return LMOutput(
                text=row.continuation,
                logprobs=[],
                metadata={
                    "total_logprob": 0.0,
                    "sum_logits": 0.0,
                    "num_tokens": 0,
                    "num_tokens_all": row.num_tokens_all,
                    "is_greedy": True,
                    "prompt_truncated_tokens": row.prompt_truncated_tokens,
                },
            )

        import torch
        import torch.nn.functional as F

        continuation_length = len(row.continuation_token_ids)
        continuation_logits = logits[
            row.input_length - continuation_length : row.input_length
        ].float()
        log_probs = F.log_softmax(continuation_logits, dim=-1)
        continuation_tensor = torch.tensor(
            row.continuation_token_ids,
            dtype=torch.long,
            device=log_probs.device,
        )
        token_log_probs = torch.gather(
            log_probs,
            1,
            continuation_tensor.unsqueeze(-1),
        ).squeeze(-1)
        greedy_tokens = log_probs.argmax(dim=-1)
        is_greedy = bool((greedy_tokens == continuation_tensor).all().item())

        token_logprob_values = cast(list[float], token_log_probs.tolist())
        entries = self._logprob_entries(row.continuation_token_ids, token_logprob_values)
        total = float(sum(token_logprob_values))
        return LMOutput(
            text=row.continuation,
            logprobs=entries,
            metadata={
                "total_logprob": total,
                "sum_logits": total,
                "num_tokens": len(entries),
                "num_tokens_all": row.num_tokens_all,
                "is_greedy": is_greedy,
                "prompt_truncated_tokens": row.prompt_truncated_tokens,
            },
        )

    def _logprobs_chunk(self, requests: list[LMRequest]) -> list[list[LMOutput]]:
        import torch

        rows_by_request = [self._logprob_inputs_for_request(request) for request in requests]
        token_inputs = [row.input_ids for rows in rows_by_request for row in rows]

        if token_inputs:
            batched_inputs = self._right_pad(token_inputs)
            with torch.no_grad():
                batch_logits = self.generation_module.model_forward(input_ids=batched_inputs)
        else:
            batch_logits = []

        output_iter = iter(batch_logits)
        results: list[list[LMOutput]] = []

        for rows in rows_by_request:
            results.append(
                [self._logprob_output_from_logits(row, next(output_iter)) for row in rows]
            )

        return results

    async def agenerate(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        return await asyncio.to_thread(self.generate, requests, sampling_params)

    async def alogprobs(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        return await asyncio.to_thread(self.logprobs, requests, sampling_params)

    def close(self) -> None:
        if hasattr(self, "generation_module"):
            try:
                self._free_inference_cache()
            except Exception:
                logger.debug("Failed to free OLMo-core inference cache", exc_info=True)
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            logger.debug("Failed to clear CUDA cache", exc_info=True)

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()
