"""vLLM provider."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING, Any

from olmo_eval.common.debug import is_debug_provider, is_debug_requests
from olmo_eval.common.images import resolve_images
from olmo_eval.common.types import LMOutput, LMRequest, LogProbEntry, RequestType, SamplingParams
from olmo_eval.inference.base import InferenceProvider
from olmo_eval.inference.hf_cache import refresh_hf_cache
from olmo_eval.inference.request_utils import chat_messages_for_request
from olmo_eval.inference.tokenizer_utils import encode_context_and_continuation

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from vllm import LLM
    from vllm.outputs import RequestOutput


def _configure_vllm_logger(worker_id: str | None) -> None:
    """Configure vLLM's logger to include worker_id in output.

    Args:
        worker_id: Worker identifier to include in log format, or None to use default format.
    """
    vllm_logger = logging.getLogger("vllm")

    # Remove existing handlers to avoid duplicates
    for handler in vllm_logger.handlers[:]:
        vllm_logger.removeHandler(handler)

    handler = logging.StreamHandler()
    if worker_id:
        handler.setFormatter(
            logging.Formatter(
                f"%(asctime)s [{worker_id}] [%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
    vllm_logger.addHandler(handler)
    vllm_logger.setLevel(logging.INFO)
    vllm_logger.propagate = False


def _get_token_string(logprob_obj: Any, token_id: int, tokenizer: Any = None) -> str:
    """Extract token string from vLLM logprob object."""
    if hasattr(logprob_obj, "decoded_token"):
        return logprob_obj.decoded_token
    if tokenizer is not None:
        return tokenizer.decode([token_id])
    return str(token_id)


def _coerce_logprob_to_num(logprob: Any) -> float:
    """Handle both old (float) and new (Logprob object) vLLM versions."""
    return getattr(logprob, "logprob", logprob)


def _convert_logprobs(
    vllm_logprobs: list[dict[int, Any]] | None,
    tokenizer: Any = None,
) -> list[LogProbEntry] | None:
    """Convert vLLM logprobs format to standard format.

    Works with both old (float) and new (Logprob object) vLLM versions.
    """
    if vllm_logprobs is None:
        return None

    result: list[LogProbEntry] = []
    for token_logprobs in vllm_logprobs:
        if not token_logprobs:
            continue
        # vLLM returns dict of {token_id: LogprobInfo}, take first (chosen) token
        token_id, logprob_obj = next(iter(token_logprobs.items()))
        token_str = _get_token_string(logprob_obj, token_id, tokenizer)
        logprob_val = _coerce_logprob_to_num(logprob_obj)
        result.append(
            {
                "token": token_str,
                "logprob": logprob_val,
                "token_id": int(token_id),
                "bytes": list(token_str.encode("utf-8")),
            }
        )

    return result


class VLLMProvider(InferenceProvider):
    """Provider using vLLM for high-throughput inference.

    Image requests are rendered through the tokenizer's chat template, which emits the
    model's own image placeholder tokens, and handed to vLLM with the images as
    ``multi_modal_data``; vLLM expands the placeholders against them internally.
    """

    supports_images = True

    def __init__(
        self,
        model_name: str,
        tokenizer: str | None = None,
        attention_backend: str | None = None,
        worker_id: str | None = None,
        force_download: bool = False,
        max_images: int = 8,
        chat_template_kwargs: dict[str, Any] | None = None,
        **engine_kwargs,
    ) -> None:
        """Initialize the provider.

        Args:
            model_name: HuggingFace model identifier or local path.
            tokenizer: Tokenizer path/identifier. If not specified, uses the model path.
            attention_backend: Attention backend to use (e.g., "FLASHINFER", "FLASH_ATTN").
                If not specified, vLLM will auto-select based on available backends.
            worker_id: Optional worker identifier for logging. If provided, vLLM logs
                will include this identifier.
            force_download: Force-refresh Hugging Face model/tokenizer cache entries
                before initializing vLLM.
            max_images: Maximum images accepted in a single prompt, forwarded as
                ``limit_mm_per_prompt={"image": max_images}``. vLLM's own default is 1,
                which is too low for interleaved-image tasks: MMMU-Pro's standard
                settings carry up to 7 (``image_1``..``image_7``).
            chat_template_kwargs: Extra keyword arguments forwarded to the tokenizer's
                ``apply_chat_template``. Needed for models whose template exposes a mode
                switch -- e.g. the hybrid Qwen3 releases reason by default and require
                ``{"enable_thinking": false}`` to be compared against a non-thinking model.
            **engine_kwargs: Additional arguments passed to vLLM LLM engine.
        """
        # Set vLLM logging level - DEBUG if OLMO_EVAL_DEBUG_PROVIDER=1, otherwise WARNING
        if is_debug_provider():
            os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"
        else:
            os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

        # Configure vLLM logger with worker_id if provided
        if worker_id:
            _configure_vllm_logger(worker_id)

        try:
            from vllm import LLM
        except ImportError as e:
            import traceback

            logger.error(f"Failed to import vllm: {e}")
            logger.error(traceback.format_exc())
            raise ImportError("vllm is required for VLLMProvider") from e

        super().__init__(model_name)
        self._worker_id = worker_id
        if force_download:
            model_revision = engine_kwargs.get("revision")
            cache_dir = engine_kwargs.get("download_dir") or engine_kwargs.get("cache_dir")
            token = engine_kwargs.get("token")
            refresh_hf_cache(
                model_name,
                revision=model_revision,
                cache_dir=cache_dir,
                token=token,
                force_download=True,
            )
            if tokenizer and tokenizer != model_name:
                tokenizer_revision = engine_kwargs.get("tokenizer_revision") or model_revision
                refresh_hf_cache(
                    tokenizer,
                    revision=tokenizer_revision,
                    cache_dir=cache_dir,
                    token=token,
                    force_download=True,
                )

        engine_kwargs.setdefault("gpu_memory_utilization", 0.8)

        # Configure attention backend if specified (e.g., FLASHINFER, FLASH_ATTN)
        if attention_backend:
            engine_kwargs.setdefault("attention_backend", attention_backend)

        # Use separate tokenizer if specified
        if tokenizer:
            engine_kwargs.setdefault("tokenizer", tokenizer)

        # Disable tqdm loading bar by default, enable with --debug-provider
        engine_kwargs.setdefault("use_tqdm_on_load", is_debug_provider())

        # Extract add_bos_token before passing to LLM (not a valid vLLM EngineArgs parameter).
        # When False, prompts will be pre-tokenized without special tokens and passed as token IDs,
        # matching the old framework's behavior (tokenizer(text, add_special_tokens=False)).
        self._add_bos_token: bool | None = engine_kwargs.pop("add_bos_token", None)

        self._chat_template_kwargs = dict(chat_template_kwargs or {})
        # Raise the per-prompt image cap (vLLM defaults to 1). setdefault so an explicit
        # limit_mm_per_prompt in engine_kwargs still wins -- and then read the effective
        # value back, because the guard in _format_prompt has to track the limit the engine
        # will actually enforce. Leaving it at max_images would make the guard permissive
        # in exactly the case it exists for: an explicit limit *below* max_images, where
        # vLLM fails the entire chunk rather than the one oversized request.
        if int(max_images) > 0:
            engine_kwargs.setdefault("limit_mm_per_prompt", {"image": int(max_images)})
        self._max_images = int(
            (engine_kwargs.get("limit_mm_per_prompt") or {}).get("image", max_images)
        )

        self.llm: LLM = LLM(model=model_name, **engine_kwargs)

    @property
    def max_length(self) -> int:
        """Get the maximum model context length."""
        if not hasattr(self, "_max_length"):
            self._max_length = self.llm.llm_engine.model_config.max_model_len
        return self._max_length

    def get_tokenizer(self) -> Any:
        """Get the tokenizer for this provider."""
        return self.llm.get_tokenizer()

    def _encode_pair(self, context: str, continuation: str) -> tuple[list[int], list[int]]:
        """Encode context and continuation separately (robust to non-additive tokenization).

        Matches lm_eval behavior: trailing spaces from context are moved to continuation
        before tokenization to ensure consistent token boundaries.
        """
        tokenizer = self.llm.get_tokenizer()

        # Match lm_eval behavior: move trailing spaces from context to continuation
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = tokenizer.encode(context + continuation, add_special_tokens=False)
        context_enc = tokenizer.encode(context, add_special_tokens=False)
        continuation_enc = whole_enc[len(context_enc) :]

        return context_enc, continuation_enc

    def _build_sampling_params(self, params: SamplingParams) -> Any:
        """Convert SamplingParams to vLLM SamplingParams."""
        from vllm import SamplingParams as VLLMSamplingParams

        # Handle do_sample=False (greedy decoding)
        temperature = 0.0 if not params.do_sample else params.temperature
        top_p = None if not params.do_sample else params.top_p
        top_k = None if not params.do_sample else params.top_k

        # vLLM natively accepts max_tokens=None as "generate to the context limit",
        # matching our uncapped contract, so pass it through unchanged.
        kwargs: dict[str, Any] = {
            "max_tokens": params.max_tokens,
            "n": params.num_samples,
        }

        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p
        if top_k is not None:
            kwargs["top_k"] = top_k
        if params.stop_sequences:
            kwargs["stop"] = list(params.stop_sequences)
        # Always request logprobs (default to 1) for metrics computation
        kwargs["logprobs"] = params.logprobs if params.logprobs is not None else 1

        return VLLMSamplingParams(**kwargs)

    @staticmethod
    def _image_chat_messages(request: LMRequest, num_images: int) -> list[dict[str, Any]]:
        """Build chat messages with image placeholders ahead of the user's text.

        Same convention as ``HuggingFaceProvider._build_chat_messages`` and
        ``OlmoCoreVLMProvider._chat_text``: one user turn whose content is the image(s)
        followed by the question, with images attached to the *first* user turn. Keeping
        the providers on one convention is what makes their scores comparable.
        """
        image_parts: list[dict[str, Any]] = [{"type": "image"} for _ in range(num_images)]

        chat: list[dict[str, Any]] = []
        attached = False
        for msg in chat_messages_for_request(request):
            role = msg.get("role", "user")
            text = msg.get("content", "") or ""
            if role == "user" and not attached and image_parts:
                chat.append(
                    {"role": role, "content": [*image_parts, {"type": "text", "text": text}]}
                )
                attached = True
            else:
                chat.append({"role": role, "content": [{"type": "text", "text": text}]})
        if not attached and image_parts:
            chat.insert(0, {"role": "user", "content": image_parts})
        return chat

    def _format_prompt(self, request: LMRequest) -> str | dict[str, Any]:
        """Format an LMRequest into the prompt expected by inline vLLM.

        Returns a plain string for text-only requests, or vLLM's dict form
        ``{"prompt": ..., "multi_modal_data": {"image": [...]}}`` when the request carries
        images. The rendered prompt holds the model's own image placeholder tokens (e.g.
        Qwen3-VL's ``<|vision_start|><|image_pad|><|vision_end|>``), which vLLM expands
        against the images internally -- no ``AutoProcessor`` needed here, because the
        tokenizer-level chat template emits the same placeholders as the processor-level one.
        """
        images = resolve_images(request.images)
        if images:
            if self._add_bos_token is False:
                # The add_bos_token=False path pre-tokenizes to prompt_token_ids, which
                # cannot carry multi_modal_data. Failing here beats dropping the images.
                raise ValueError(
                    "add_bos_token=False is incompatible with image requests: the "
                    "pre-tokenized prompt_token_ids path cannot carry multi_modal_data."
                )
            if len(images) > self._max_images:
                # vLLM raises for the whole `llm.generate` call when any one prompt exceeds
                # limit_mm_per_prompt, so a single oversized instance fails its entire
                # chunk -- 2 instances with 35 images cost 128 of MMMU-Pro's 5190. Check
                # here so the message names the knob instead of vLLM's generic
                # "At most N image(s) may be provided in one prompt."
                raise ValueError(
                    f"Request has {len(images)} images but max_images={self._max_images}. "
                    f"Raise it via `-o provider.kwargs.max_images=N`; note vLLM profiles "
                    f"worst-case multimodal memory against this cap, so a large value "
                    f"reduces the KV cache."
                )
            tokenizer = self.llm.get_tokenizer()
            if not hasattr(tokenizer, "apply_chat_template"):
                raise ValueError(
                    "Image requests require a tokenizer with apply_chat_template so the "
                    "model's image placeholder tokens can be rendered."
                )
            text = tokenizer.apply_chat_template(
                self._image_chat_messages(request, len(images)),
                tokenize=False,
                add_generation_prompt=True,
                **self._chat_template_kwargs,
            )
            pil_images = [
                img.convert("RGB") if getattr(img, "mode", "RGB") != "RGB" else img
                for img in images
            ]
            return {"prompt": text, "multi_modal_data": {"image": pil_images}}

        if request.request_type == RequestType.CHAT and request.messages:
            tokenizer = self.llm.get_tokenizer()
            if not hasattr(tokenizer, "apply_chat_template"):
                raise ValueError("CHAT requests require a tokenizer with apply_chat_template")
            return tokenizer.apply_chat_template(
                list(request.messages),
                tokenize=False,
                add_generation_prompt=True,
                **self._chat_template_kwargs,
            )

        return request.prompt

    def generate(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        params = self._default_sampling_params(sampling_params)
        if params.truncate_prompt_tokens is not None or params.truncation_side is not None:
            logger.warning(
                "truncate_prompt_tokens or truncation_side has been set in the params, "
                "but is not supported for the VLLMProvider and will not be used."
            )
        vllm_params = self._build_sampling_params(params)

        # Image requests come back as {"prompt": ..., "multi_modal_data": ...} dicts;
        # text-only requests as plain strings.
        formatted = [self._format_prompt(req) for req in requests]

        if is_debug_requests():
            for i, prompt in enumerate(formatted):
                if isinstance(prompt, dict):
                    n_images = len(prompt["multi_modal_data"]["image"])
                    logger.info(f"Prompt {i} ({n_images} image(s)):\n{prompt['prompt']}")
                else:
                    logger.info(f"Prompt {i}:\n{prompt}")

        # When add_bos_token=False, pre-tokenize without special tokens and pass token IDs.
        # This bypasses vLLM's internal tokenization, matching the old framework behavior of
        # calling tokenizer(text, add_special_tokens=False) before passing to vLLM.
        # _format_prompt already rejects add_bos_token=False + images, so everything here
        # is a string.
        if self._add_bos_token is False:
            tokenizer = self.llm.get_tokenizer()
            vllm_prompts: list = [
                {"prompt_token_ids": tokenizer.encode(p, add_special_tokens=False)}
                for p in formatted
            ]
        else:
            vllm_prompts = formatted

        # Disable tqdm progress bar - we use our own worker-scoped logging
        outputs: list[RequestOutput] = self.llm.generate(vllm_prompts, vllm_params, use_tqdm=False)

        results: list[list[LMOutput]] = []
        for output in outputs:
            request_outputs: list[LMOutput] = []
            for completion in output.outputs:
                logprobs = _convert_logprobs(completion.logprobs)

                # Compute metadata from logprobs
                metadata: dict[str, Any] = {}
                if logprobs:
                    sum_logits = sum(entry.get("logprob", 0.0) for entry in logprobs)
                    num_tokens = len(logprobs)
                    metadata = {
                        "sum_logits": sum_logits,
                        "num_tokens": num_tokens,
                        "num_tokens_all": num_tokens,
                    }

                metadata["finish_reason"] = completion.finish_reason
                metadata["completion_tokens"] = len(completion.token_ids)

                request_outputs.append(
                    LMOutput(
                        text=completion.text,
                        logprobs=logprobs,
                        metadata=metadata,
                    )
                )
            results.append(request_outputs)

        return results

    def describe_request(
        self,
        request: LMRequest,
        sampling_params: SamplingParams | None = None,
    ) -> dict[str, Any] | None:
        params = self._default_sampling_params(sampling_params)
        trace = super().describe_request(request, sampling_params)
        if trace is None:
            return None

        if request.request_type == RequestType.LOGLIKELIHOOD:
            trace["provider"] = "VLLMProvider"
            trace["endpoint"] = "llm.generate"
            trace["generation_kwargs"] = {
                "max_gen_toks": 1,
                "do_sample": False,
                "temperature": params.temperature,
                "prompt_logprobs": 1,
            }
            trace["stop_sequences"] = []
            trace["input_mode"] = "prompt_token_ids"
            return trace

        vllm_params = self._build_sampling_params(params)
        trace["provider"] = "VLLMProvider"
        trace["endpoint"] = "llm.generate"
        trace["generation_kwargs"] = {
            "max_gen_toks": vllm_params.max_tokens,
            "do_sample": params.do_sample and params.temperature > 0,
            "temperature": getattr(vllm_params, "temperature", params.temperature),
            "logprobs": getattr(vllm_params, "logprobs", None),
            "num_samples": vllm_params.n,
        }
        if getattr(vllm_params, "top_p", None) is not None:
            trace["generation_kwargs"]["top_p"] = vllm_params.top_p
        if getattr(vllm_params, "top_k", None) is not None:
            trace["generation_kwargs"]["top_k"] = vllm_params.top_k
        trace["stop_sequences"] = list(params.stop_sequences or ())
        n_images = len(request.images or ())
        if n_images:
            trace["input_mode"] = "multi_modal_data"
            trace["num_images"] = n_images
        else:
            trace["input_mode"] = "prompt_token_ids" if self._add_bos_token is False else "text"
        return trace

    def logprobs(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        from vllm import SamplingParams as VLLMSamplingParams

        params = self._default_sampling_params(sampling_params)
        if params.truncate_prompt_tokens is not None or params.truncation_side is not None:
            logger.warning(
                "truncate_prompt_tokens or truncation_side has been set in the params, "
                "but is not supported for the VLLMProvider and will not be used."
            )
        # This path scores raw token sequences and has nowhere to attach multi_modal_data,
        # so images would be silently dropped and every continuation scored against text
        # alone. None of the vision tasks use LOGLIKELIHOOD.
        if any(request.images for request in requests):
            raise ValueError(
                "VLLMProvider.logprobs() does not support images. Use generate() for "
                "multimodal tasks, or the HuggingFace/olmo_core_vlm provider if you need "
                "image-conditioned loglikelihood scoring."
            )
        vllm_params = VLLMSamplingParams(
            prompt_logprobs=1,
            max_tokens=1,
            temperature=params.temperature,
        )

        tokenizer = self.llm.get_tokenizer()
        default_max_len = self.max_length

        # Build token sequences for all continuations
        token_inputs: list[list[int]] = []
        request_meta: list[tuple[int, int, int]] = []  # (ctxlen, num_tokens_all, overflow)

        for request in requests:
            # Use per-request max_length if set (e.g., from task config), else provider default.
            max_len = request.max_length or default_max_len
            continuations = request.continuations or ()
            cont_prompts = request.continuation_prompts
            for i, continuation in enumerate(continuations):
                prompt = cont_prompts[i] if cont_prompts else request.prompt
                context_enc, continuation_enc = encode_context_and_continuation(
                    tokenizer, prompt, continuation
                )

                # Calculate overflow and left-truncate to max_length - 1
                full_len = len(context_enc) + len(continuation_enc)
                overflow = full_len - (max_len - 1)
                inp = (context_enc + continuation_enc)[-(max_len - 1) :]

                # Adjust ctxlen based on overflow
                ctxlen = len(context_enc) - max(0, overflow)
                ctxlen = max(0, ctxlen)  # Ensure non-negative

                token_inputs.append(inp)
                request_meta.append((ctxlen, len(inp), overflow))

        # Call vLLM with token IDs instead of strings
        # Pass as list of dicts with prompt_token_ids key
        # Disable tqdm progress bar - we use our own worker-scoped logging
        prompts = [{"prompt_token_ids": tokens} for tokens in token_inputs]

        if is_debug_requests():
            logger.info(f"vLLM logprobs: {len(prompts)} continuations")
            logger.info(f"Sampling params: {vllm_params}")

        outputs: list[RequestOutput] = self.llm.generate(prompts, vllm_params, use_tqdm=False)

        # Parse results back to per-request structure
        output_iter = iter(outputs)
        meta_iter = iter(request_meta)
        tokens_iter = iter(token_inputs)
        results = []

        for request in requests:
            continuations = request.continuations or ()
            request_outputs = []

            for continuation in continuations:
                output = next(output_iter)
                ctxlen, num_tokens_all, overflow = next(meta_iter)
                inp = next(tokens_iter)

                logprob_entries: list[LogProbEntry] = []
                total = 0.0
                is_greedy = True

                prompt_logprobs = output.prompt_logprobs or []
                # Skip the first ctxlen positions (context tokens)
                cont_logprobs = prompt_logprobs[ctxlen:] if ctxlen < len(prompt_logprobs) else []
                # Get continuation token IDs from the actual input
                cont_tokens = inp[ctxlen:]

                for token_id, token_probs in zip(cont_tokens, cont_logprobs, strict=True):
                    if not token_probs:
                        continue

                    # Check if this token is the argmax (greedy choice).
                    # Must check BEFORE the lp_obj gate so we catch non-greedy tokens
                    # even when they aren't in the top-k returned by prompt_logprobs.
                    if is_greedy:
                        max_token_id = max(
                            token_probs.keys(),
                            key=lambda tid: _coerce_logprob_to_num(token_probs[tid]),
                        )
                        if max_token_id != token_id:
                            is_greedy = False

                    # Look up logprob for the actual continuation token (not first key in dict)
                    lp_obj = token_probs.get(token_id)
                    if lp_obj is None:
                        continue
                    logprob_val = _coerce_logprob_to_num(lp_obj)

                    token_str = _get_token_string(lp_obj, token_id, tokenizer)
                    logprob_entries.append(
                        {
                            "token": token_str,
                            "logprob": logprob_val,
                            "token_id": int(token_id),
                            "bytes": list(token_str.encode("utf-8")),
                        }
                    )
                    total += logprob_val

                num_tokens = len(logprob_entries)
                request_outputs.append(
                    LMOutput(
                        text=continuation,
                        logprobs=logprob_entries,
                        metadata={
                            "total_logprob": total,
                            "sum_logits": total,  # Alias for compatibility
                            "num_tokens": num_tokens,
                            "num_tokens_all": num_tokens_all,
                            "is_greedy": is_greedy,
                            "prompt_truncated_tokens": max(0, overflow),
                        },
                    )
                )

            results.append(request_outputs)

        return results

    async def agenerate(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        """Async generate completions.

        Runs the synchronous vLLM generate in a thread pool to avoid blocking.

        Args:
            requests: Batch of requests to process.
            sampling_params: Sampling configuration.

        Returns:
            List of output lists, one per request.
        """
        return await asyncio.to_thread(self.generate, requests, sampling_params)

    async def alogprobs(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        """Async compute logprobs for continuations.

        Runs the synchronous vLLM logprobs in a thread pool to avoid blocking.

        Args:
            requests: Batch of requests with continuations to score.

        Returns:
            List of output lists with logprobs populated.
        """
        return await asyncio.to_thread(self.logprobs, requests, sampling_params)
