"""Hotspot tests for the OLMo-core provider."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import olmo_eval.inference.providers.olmo_core_utils as olmo_core_utils
from olmo_eval.common.types import LMOutput, LMRequest, RequestType, SamplingParams
from olmo_eval.inference.providers.olmo_core import OlmoCoreProvider
from olmo_eval.inference.providers.olmo_core_utils import _TRANSFORMERS_UNSET_MODEL_MAX_LENGTH


class FakeTokenizerConfig:
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SimpleNamespace:
        return SimpleNamespace(
            identifier=data.get("identifier"),
            pad_token_id=data.get("pad_token_id"),
            eos_token_id=data.get("eos_token_id"),
        )


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    bos_token_id = 1
    model_max_length = 32

    def __init__(self) -> None:
        self.add_bos_token = False
        self.encode_calls: list[dict[str, Any]] = []
        self.decode_calls: list[list[int]] = []
        self.vocab = {
            "": [],
            "Prompt": [10, 11],
            "Prompt!": [10, 11, 6],
            "Other": [12],
            "!": [6],
            " !": [13, 6],
            " STOP": [4],
            "STOP": [8, 9],
        }
        self.id_to_text = {
            0: "<pad>",
            1: "<bos>",
            2: "<eos>",
            4: " STOP",
            5: "hello",
            6: "!",
            7: "x",
            8: "ST",
            9: "OP",
            10: "P",
            11: "rompt",
            12: "Other",
            13: " ",
        }

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        self.encode_calls.append(
            {
                "text": text,
                "add_special_tokens": add_special_tokens,
            }
        )
        if text in self.vocab:
            token_ids = self.vocab[text]
        else:
            token_ids = [ord(char) % 13 + 3 for char in text]
        if add_special_tokens:
            return [self.bos_token_id, *token_ids]
        return token_ids

    def decode(self, token_ids: int | list[int], skip_special_tokens: bool = True) -> str:
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        self.decode_calls.append(list(token_ids))
        pieces = []
        for token_id in token_ids:
            if skip_special_tokens and token_id in {self.pad_token_id, self.eos_token_id}:
                continue
            pieces.append(self.id_to_text.get(token_id, str(token_id)))
        return "".join(pieces)


class FakeAutoTokenizer:
    @classmethod
    def from_pretrained(cls, tokenizer_path: str, **kwargs: Any) -> FakeTokenizer:
        del cls, kwargs
        assert tokenizer_path == "fake-tokenizer"
        tokenizer = FakeTokenizer()
        tokenizer.model_max_length = _TRANSFORMERS_UNSET_MODEL_MAX_LENGTH
        return tokenizer


class MissingSpecialTokenAutoTokenizer:
    @classmethod
    def from_pretrained(cls, tokenizer_path: str, **kwargs: Any) -> FakeTokenizer:
        del cls
        tokenizer = FakeAutoTokenizer.from_pretrained(tokenizer_path, **kwargs)
        tokenizer.pad_token_id = None
        tokenizer.eos_token_id = None
        return tokenizer


class FakeTensorRows:
    def __init__(self, rows: list[list[int]] | list[list[float]]) -> None:
        self._rows = rows
        self.shape = (len(rows), len(rows[0]) if rows else 0)

    def __getitem__(self, idx: int) -> Any:
        return SimpleNamespace(tolist=lambda: self._rows[idx])

    def tolist(self) -> list[list[int]] | list[list[float]]:
        return self._rows


class FakeGenerationModule:
    def __init__(self) -> None:
        self.generate_calls: list[dict[str, Any]] = []
        self.checkpoint_kwargs: dict[str, Any] = {}
        self.prepare_calls: list[tuple[int, int]] = []
        self.cache_allocated = False
        self.free_calls = 0

    @classmethod
    def from_checkpoint(cls, **kwargs: Any) -> FakeGenerationModule:
        module = cls()
        module.checkpoint_kwargs = kwargs
        return module

    def generate_batch(self, **kwargs: Any):
        self.generate_calls.append(kwargs)
        batch_size = kwargs["input_ids"].shape[0]
        if kwargs["use_cache"]:
            self.prepare_inference_cache(batch_size, kwargs["max_length"])

        completion_rows = [[5, 4, 0] if idx % 2 == 0 else [5, 6, 0] for idx in range(batch_size)]
        generated_rows = (
            completion_rows
            if kwargs["completions_only"]
            else [
                [*kwargs["input_ids"][idx].tolist(), *completion_rows[idx]]
                for idx in range(batch_size)
            ]
        )
        logprob_rows = [
            [-0.1, -0.2, -9.0] if idx % 2 == 0 else [-0.3, -0.4, -9.0] for idx in range(batch_size)
        ]
        return FakeTensorRows(generated_rows), None, FakeTensorRows(logprob_rows)

    def prepare_inference_cache(self, batch_size: int, max_seq_len: int) -> None:
        self.prepare_calls.append((batch_size, max_seq_len))
        self.cache_allocated = True

    def free_inference_cache(self) -> None:
        self.cache_allocated = False
        self.free_calls += 1


def _write_raw_checkpoint(
    checkpoint_dir: Path,
    *,
    model: dict[str, Any] | None = None,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "config.json").write_text(
        json.dumps(
            {
                "model": model if model is not None else {"d_model": 8},
                "dataset": {
                    "tokenizer": {
                        "identifier": "fake-tokenizer",
                        "vocab_size": 16,
                        "pad_token_id": 0,
                        "eos_token_id": 2,
                    }
                },
            }
        )
    )
    (checkpoint_dir / ".metadata").write_text("fake")


def _metadata_reader(path: str | Path) -> SimpleNamespace:
    path_obj = path if isinstance(path, Path) else Path(path)
    if (path_obj / ".metadata").exists():
        return SimpleNamespace(state_dict_metadata={"model.transformer.wte.weight": object()})
    raise FileNotFoundError(path)


def _fake_olmo_core_imports(
    *,
    cuda_available: bool = False,
    auto_tokenizer: type[Any] = FakeAutoTokenizer,
) -> SimpleNamespace:
    return SimpleNamespace(
        AutoTokenizer=auto_tokenizer,
        AttentionBackendName=str,
        GenerationConfig=lambda **kwargs: SimpleNamespace(**kwargs),
        TokenizerConfig=FakeTokenizerConfig,
        TransformerGenerationModule=FakeGenerationModule,
        cached_path=None,
        get_checkpoint_metadata=_metadata_reader,
        torch=SimpleNamespace(
            cuda=SimpleNamespace(
                get_device_capability=lambda: (9, 0),
                is_available=lambda: cuda_available,
            ),
            device=lambda device: device,
        ),
    )


@pytest.fixture
def fake_provider() -> tuple[OlmoCoreProvider, FakeGenerationModule, FakeTokenizer]:
    provider = OlmoCoreProvider.__new__(OlmoCoreProvider)
    tokenizer = FakeTokenizer()
    module = FakeGenerationModule()
    provider.model_name = "fake-model"
    provider.tokenizer = tokenizer
    provider.generation_module = module
    provider.pad_token_id = tokenizer.pad_token_id
    provider.eos_token_id = tokenizer.eos_token_id
    provider.use_cache = True
    provider.add_bos_token = False
    provider.batch_size = None
    provider.chat_template = None
    provider.max_length = 32

    def left_pad(sequences: list[list[int]]) -> tuple[FakeTensorRows, FakeTensorRows]:
        max_len = max(max((len(seq) for seq in sequences), default=0), 1)
        rows = []
        masks = []
        for seq in sequences:
            pad_len = max_len - len(seq)
            rows.append([tokenizer.pad_token_id] * pad_len + seq)
            masks.append([0] * pad_len + [1] * len(seq))
        return FakeTensorRows(rows), FakeTensorRows(masks)

    provider._left_pad = left_pad
    return provider, module, tokenizer


def test_provider_loads_checkpoint_with_olmes_defaults(tmp_path, monkeypatch) -> None:
    checkpoint_dir = tmp_path / "step1000"
    _write_raw_checkpoint(checkpoint_dir, model={"d_model": 8, "max_sequence_length": 4096})
    monkeypatch.setattr(
        olmo_core_utils,
        "_import_olmo_core",
        lambda: _fake_olmo_core_imports(auto_tokenizer=MissingSpecialTokenAutoTokenizer),
    )

    provider = OlmoCoreProvider(str(checkpoint_dir))

    checkpoint_kwargs = provider.generation_module.checkpoint_kwargs
    generation_config = checkpoint_kwargs["generation_config"]
    assert provider.max_length == 4096
    assert provider.add_bos_token is False
    assert provider.pad_token_id == 0
    assert provider.eos_token_id == 2
    assert provider.tokenizer.pad_token_id == 0
    assert provider.tokenizer.eos_token_id == 2
    assert checkpoint_kwargs["dtype"] == "bfloat16"
    assert "attention_backend" not in checkpoint_kwargs
    assert generation_config.pad_token_id == 0
    assert generation_config.eos_token_id == 2
    assert generation_config.use_cache is True


def test_provider_passes_explicit_attention_backend(tmp_path, monkeypatch) -> None:
    checkpoint_dir = tmp_path / "step1000"
    _write_raw_checkpoint(checkpoint_dir, model={"d_model": 8, "max_sequence_length": 4096})
    monkeypatch.setattr(
        olmo_core_utils,
        "_import_olmo_core",
        lambda: _fake_olmo_core_imports(cuda_available=True),
    )

    provider = OlmoCoreProvider(
        str(checkpoint_dir),
        attention_backend="torch",
    )

    assert provider.generation_module.checkpoint_kwargs["attention_backend"] == "torch"


def test_generate_uses_olmes_batch_contract(
    fake_provider: tuple[OlmoCoreProvider, FakeGenerationModule, FakeTokenizer],
) -> None:
    provider, module, _ = fake_provider

    outputs = provider.generate(
        [
            LMRequest(request_type=RequestType.COMPLETION, prompt="Prompt"),
            LMRequest(request_type=RequestType.COMPLETION, prompt="Other"),
        ],
        SamplingParams(
            max_tokens=3,
            num_samples=2,
            temperature=0.7,
            top_p=None,
            top_k=None,
            stop_sequences=(" STOP", "!"),
        ),
    )

    call = module.generate_calls[0]
    assert call["input_ids"].tolist() == [
        [10, 11],
        [10, 11],
        [0, 12],
        [0, 12],
    ]
    assert call["attention_mask"].tolist() == [
        [1, 1],
        [1, 1],
        [0, 1],
        [0, 1],
    ]
    assert call["return_logprobs"] is True
    assert call["completions_only"] is False
    assert call["max_length"] == 5
    assert "max_new_tokens" not in call
    assert "stop_token_ids" not in call
    assert module.prepare_calls == [(4, 5)]
    assert [output.text for output in outputs[0]] == ["hello", "hello"]
    assert outputs[0][0].metadata["sum_logits"] == pytest.approx(-0.3)
    assert outputs[1][1].metadata["num_tokens"] == 2
    assert module.cache_allocated is False
    assert module.free_calls == 1


def test_generate_uncapped_fills_context(
    fake_provider: tuple[OlmoCoreProvider, FakeGenerationModule, FakeTokenizer],
) -> None:
    provider, module, _ = fake_provider
    provider.max_length = 32

    provider.generate(
        [LMRequest(request_type=RequestType.COMPLETION, prompt="Prompt")],
        SamplingParams(max_tokens=None),
    )

    call = module.generate_calls[0]
    # The 2-token prompt isn't truncated, and generation fills the rest of the budget.
    assert call["input_ids"].tolist() == [[10, 11]]
    assert call["max_length"] == 32


def test_describe_request_handles_uncapped_max_tokens(
    fake_provider: tuple[OlmoCoreProvider, FakeGenerationModule, FakeTokenizer],
) -> None:
    """The trace path never resolves ``max_tokens``, so it must tolerate None.

    ``describe_request`` is called for every request before generation, so
    raising here would fail an uncapped task before inference starts.
    """
    provider, _, _ = fake_provider

    trace = provider.describe_request(
        LMRequest(request_type=RequestType.COMPLETION, prompt="Prompt"),
        SamplingParams(max_tokens=None),
    )

    assert trace is not None
    assert trace["provider"] == "OlmoCoreProvider"
    assert trace["generation_kwargs"]["use_cache"] is True


def test_generate_left_truncates_to_leave_completion_room(
    fake_provider: tuple[OlmoCoreProvider, FakeGenerationModule, FakeTokenizer],
) -> None:
    provider, module, _ = fake_provider
    provider.max_length = 4

    provider.generate(
        [LMRequest(request_type=RequestType.COMPLETION, prompt="Prompt")],
        SamplingParams(max_tokens=3),
    )

    call = module.generate_calls[0]
    assert call["input_ids"].tolist() == [[11]]
    assert call["attention_mask"].tolist() == [[1]]
    assert call["max_length"] == 4

    module.generate_calls.clear()
    provider.max_length = 3
    with pytest.raises(ValueError, match=r"max_tokens \(3\) is greater than or equal"):
        provider.generate(
            [LMRequest(request_type=RequestType.COMPLETION, prompt="Prompt")],
            SamplingParams(max_tokens=3),
        )
    assert module.generate_calls == []


def test_generation_encoding_uses_provider_bos_flag(
    fake_provider: tuple[OlmoCoreProvider, FakeGenerationModule, FakeTokenizer],
) -> None:
    provider, _, tokenizer = fake_provider
    tokenizer.add_bos_token = True

    assert provider._encode_prompt("Prompt") == [10, 11]
    assert tokenizer.encode_calls[-1] == {
        "text": "Prompt",
        "add_special_tokens": False,
    }

    provider.add_bos_token = True
    assert provider._encode_prompt("Prompt") == [1, 10, 11]
    assert tokenizer.encode_calls[-1] == {
        "text": "Prompt",
        "add_special_tokens": False,
    }


def test_logprob_encoding_uses_provider_bos_flag(
    fake_provider: tuple[OlmoCoreProvider, FakeGenerationModule, FakeTokenizer],
) -> None:
    provider, _, tokenizer = fake_provider
    tokenizer.add_bos_token = True

    rows = provider._logprob_inputs_for_request(
        LMRequest(
            request_type=RequestType.LOGLIKELIHOOD,
            prompt="Prompt",
            continuations=("!",),
        )
    )
    assert rows[0].input_ids == [10, 11]
    assert rows[0].continuation_token_ids == [6]

    provider.add_bos_token = True
    rows = provider._logprob_inputs_for_request(
        LMRequest(
            request_type=RequestType.LOGLIKELIHOOD,
            prompt="Prompt",
            continuations=("!",),
        )
    )
    assert rows[0].input_ids == [1, 10, 11]
    assert rows[0].continuation_token_ids == [6]


def test_logprob_encoding_adds_prefix_when_context_tokenizes_empty(
    fake_provider: tuple[OlmoCoreProvider, FakeGenerationModule, FakeTokenizer],
) -> None:
    provider, _, _ = fake_provider

    rows = provider._logprob_inputs_for_request(
        LMRequest(
            request_type=RequestType.LOGLIKELIHOOD,
            prompt=" ",
            continuations=("!",),
        )
    )

    assert rows[0].input_ids == [1, 13]
    assert rows[0].continuation_token_ids == [13, 6]
    assert rows[0].input_length == 2
    assert rows[0].num_tokens_all == 3


def test_logprob_inputs_record_truncated_prompt_tokens(
    fake_provider: tuple[OlmoCoreProvider, FakeGenerationModule, FakeTokenizer],
) -> None:
    provider, _, _ = fake_provider

    def rows_for(max_length: int | None) -> list:
        return provider._logprob_inputs_for_request(
            LMRequest(
                request_type=RequestType.LOGLIKELIHOOD,
                prompt="Prompt",
                continuations=("!",),
                max_length=max_length,
            )
        )

    fits = rows_for(None)
    assert fits[0].prompt_truncated_tokens == 0

    over = rows_for(1)
    assert over[0].input_ids == [11]
    assert over[0].continuation_token_ids == [6]
    assert over[0].prompt_truncated_tokens == 1


def test_stop_text_postprocessing_matches_olmes(
    fake_provider: tuple[OlmoCoreProvider, FakeGenerationModule, FakeTokenizer],
) -> None:
    provider, _, tokenizer = fake_provider

    token_ids = [7, 7, 8, 9, 7]
    token_logprobs = [-0.01] * len(token_ids)
    normalized_ids, normalized_logprobs, text = provider._normalize_generation_output(
        token_ids,
        token_logprobs,
        ("STOP",),
    )
    assert normalized_ids == token_ids
    assert normalized_logprobs == token_logprobs
    assert text == "xx"

    tokenizer.id_to_text[13] = tokenizer.decode(
        [tokenizer.eos_token_id],
        skip_special_tokens=False,
    )
    normalized_ids, normalized_logprobs, text = provider._normalize_generation_output(
        [5, 13, 7],
        [-0.1, -0.2, -0.3],
        provider._stop_sequences_with_eos(None),
    )
    assert normalized_ids == [5, 13, 7]
    assert normalized_logprobs == [-0.1, -0.2, -0.3]
    assert text == "hello"


def test_logprobs_clears_generation_cache_before_forward(
    fake_provider: tuple[OlmoCoreProvider, FakeGenerationModule, FakeTokenizer],
) -> None:
    provider, module, _ = fake_provider
    module.cache_allocated = True
    cache_states: list[bool] = []

    def logprobs_chunk(requests: list[LMRequest]) -> list[list[LMOutput]]:
        cache_states.append(module.cache_allocated)
        return [[] for _ in requests]

    provider._logprobs_chunk = logprobs_chunk
    provider.logprobs(
        [
            LMRequest(
                request_type=RequestType.LOGLIKELIHOOD,
                prompt="Prompt",
                continuations=("!",),
            )
        ]
    )

    assert cache_states == [False]
    assert module.cache_allocated is False
    assert module.free_calls == 1


class TestMmOlmoKeyRemap:
    """mm_olmo trainer tensor names -> released-Molmo2 HF names."""

    def test_untied_lm_head_becomes_hf_lm_head(self) -> None:
        """The transformer's own ``ff_out`` is the LM head, not an MLP output.

        Leaving it unrenamed makes the loader fall back to the weight-tied path and
        substitute the embedding table for a head the model does not tie, which
        produces fluent garbage instead of an error.
        """
        from olmo_eval.inference.providers.olmo_core_vlm.checkpoint import _mm_olmo_key_to_hf_key

        assert _mm_olmo_key_to_hf_key("model.transformer.ff_out.weight") == "lm_head.weight"

    def test_block_ff_out_still_nests_under_mlp(self) -> None:
        from olmo_eval.inference.providers.olmo_core_vlm.checkpoint import _mm_olmo_key_to_hf_key

        assert (
            _mm_olmo_key_to_hf_key("model.transformer.blocks.3.ff_out.weight")
            == "model.transformer.blocks.3.mlp.ff_out.weight"
        )

    @pytest.mark.parametrize(
        ("key", "expected"),
        [
            ("model.transformer.wte.embedding", "model.transformer.wte.embedding"),
            ("model.transformer.ln_f.weight", "model.transformer.ln_f.weight"),
            (
                "model.transformer.blocks.0.att_proj.weight",
                "model.transformer.blocks.0.self_attn.att_proj.weight",
            ),
        ],
    )
    def test_other_keys(self, key: str, expected: str) -> None:
        from olmo_eval.inference.providers.olmo_core_vlm.checkpoint import _mm_olmo_key_to_hf_key

        assert _mm_olmo_key_to_hf_key(key) == expected


class TestMmOlmoUnpickleShim:
    """The shim registers a module with ``__spec__ = None``; repeat calls must not crash."""

    def _clear(self) -> None:
        for name in ("olmo.train.remote_filesystem", "olmo.train", "olmo"):
            sys.modules.pop(name, None)

    def test_shim_survives_repeat_calls(self) -> None:
        from olmo_eval.inference.providers.olmo_core_vlm.checkpoint import (
            _ensure_mm_olmo_unpickle_shim,
        )

        self._clear()
        try:
            _ensure_mm_olmo_unpickle_shim()
            first = sys.modules["olmo.train.remote_filesystem"]
            _ensure_mm_olmo_unpickle_shim()  # crashed with ValueError before the fix
            assert sys.modules["olmo.train.remote_filesystem"] is first
            assert hasattr(first, "_StorageInfo")
        finally:
            self._clear()


def test_cached_backend_accepts_every_parent_forward_argument() -> None:
    """OLMo-core's ``Attention`` passes every backend keyword (``sinks`` since #872), so a
    parameter missing from the cached backend's ``forward`` fails every multimodal request."""
    import inspect

    pytest.importorskip("torch")
    backend = pytest.importorskip("olmo_core.nn.attention.backend")
    from olmo_eval.inference.providers.olmo_core_vlm.cache import _cached_torch_backend_class

    parent = inspect.signature(backend.TorchAttentionBackend.forward).parameters
    cached = inspect.signature(_cached_torch_backend_class().forward).parameters
    assert set(parent) <= set(cached)


class TestBuildDecodeAttentionMask:
    """Direct coverage for the cached-decode SDPA mask (the correctness-critical path)."""

    @pytest.fixture(autouse=True)
    def _torch(self):
        self.torch = pytest.importorskip("torch")

    def _build(self, **kwargs):
        from olmo_eval.inference.providers.olmo_core_vlm.cache import (
            build_decode_attention_mask,
        )

        return build_decode_attention_mask(device="cpu", **kwargs)

    def test_causal_prefill(self) -> None:
        mask = self._build(seq_len=4, pos=0, total=4)
        expected = self.torch.ones(4, 4, dtype=self.torch.bool).tril()
        assert self.torch.equal(mask, expected)

    def test_plain_decode_step_needs_no_mask(self) -> None:
        assert self._build(seq_len=1, pos=3, total=4) is None

    def test_or_mask_reopens_image_block(self) -> None:
        # bidirectional block over keys 1..2: every query may see them
        or_mask = self.torch.zeros(4, 4, dtype=self.torch.bool)
        or_mask[:, 1:3] = True
        mask = self._build(seq_len=4, pos=0, total=4, or_mask=or_mask)
        assert bool(mask[0, 2])  # non-causal position opened by or_mask
        assert not bool(mask[0, 3])  # untouched future key stays closed

    def test_or_mask_is_left_padded_to_cached_keys(self) -> None:
        # a decode-step or_mask covering only the current key must not shift onto
        # the cached prefix
        or_mask = self.torch.ones(1, 1, dtype=self.torch.bool)
        mask = self._build(seq_len=1, pos=3, total=4, or_mask=or_mask)
        expected = self.torch.tensor([[True, True, True, True]])
        assert self.torch.equal(mask, expected)

    def test_left_padding_hides_pad_keys_from_real_queries(self) -> None:
        leftpad = self.torch.tensor([2])
        mask = self._build(seq_len=4, pos=0, total=4, cache_leftpad=leftpad)
        assert mask.shape == (1, 1, 4, 4)
        row = mask[0, 0]
        # real query (abs pos 2) sees no pad keys but keeps its causal window
        assert row[2].tolist() == [False, False, True, False]
        # pad-slot query (abs pos 0) keeps its causal row so softmax has support
        assert row[0].tolist() == [True, False, False, False]


class TestVLMEmbedSplitVocab:
    """The split base/extra lookup must equal a lookup over the concatenated table."""

    @pytest.fixture(autouse=True)
    def _torch(self):
        self.torch = pytest.importorskip("torch")

    def _provider(self, extra: bool):
        from olmo_eval.inference.providers.olmo_core_vlm.provider import OlmoCoreVLMProvider

        torch = self.torch
        gen = torch.Generator().manual_seed(0)
        weight = torch.randn(10, 4, generator=gen)
        emb = SimpleNamespace(weight=weight, padding_idx=None)
        if extra:
            emb.extra_weight = torch.randn(3, 4, generator=gen)
        provider = OlmoCoreVLMProvider.__new__(OlmoCoreVLMProvider)
        provider.model = SimpleNamespace(
            lm=SimpleNamespace(embeddings=emb, embed_scale=None, embedding_norm=None)
        )
        return provider, emb

    def test_matches_concatenated_lookup(self) -> None:
        import torch.nn.functional as F

        provider, emb = self._provider(extra=True)
        ids = self.torch.tensor([[0, 9, 10, 12, 3, 11]])
        expected = F.embedding(ids, self.torch.cat([emb.weight, emb.extra_weight], dim=0))
        assert self.torch.allclose(provider._embed(ids), expected)

    def test_base_only_table(self) -> None:
        import torch.nn.functional as F

        provider, emb = self._provider(extra=False)
        ids = self.torch.tensor([[1, 2, 3]])
        assert self.torch.allclose(provider._embed(ids), F.embedding(ids, emb.weight))


class _FakeVLMLm:
    """Callable LM returning zero logits; embeddings sized for FakeTokenizer's vocab."""

    def __init__(self, torch, vocab_size: int = 32) -> None:
        self._torch = torch
        self.embeddings = SimpleNamespace(weight=torch.zeros(vocab_size, 4), padding_idx=None)
        self.embed_scale = None
        self.embedding_norm = None
        self.vocab_size = vocab_size

    def __call__(self, ids, input_embeddings=None):
        batch, seq = ids.shape
        return self._torch.zeros(batch, seq, self.vocab_size)


class TestVLMLogprobsBoundaries:
    """Continuation boundary handling mirrors OlmoCoreProvider."""

    @pytest.fixture(autouse=True)
    def _torch(self):
        self.torch = pytest.importorskip("torch")

    def _provider(self, max_length: int = 8):
        import threading

        from olmo_eval.inference.providers.olmo_core_vlm.provider import OlmoCoreVLMProvider

        provider = OlmoCoreVLMProvider.__new__(OlmoCoreVLMProvider)
        tokenizer = FakeTokenizer()
        # joined context+continuation form, so the continuation splits to [8, 9]
        tokenizer.vocab["PromptSTOP"] = [10, 11, 8, 9]
        provider.tokenizer = tokenizer
        provider.max_length = max_length
        provider.device = self.torch.device("cpu")
        provider.autocast_dtype = None
        provider._model_lock = threading.Lock()
        provider.model = SimpleNamespace(lm=_FakeVLMLm(self.torch))
        return provider

    def _request(self, continuation: str, max_length: int | None = None) -> LMRequest:
        return LMRequest(
            request_type=RequestType.LOGLIKELIHOOD,
            prompt="Prompt",
            continuations=(continuation,),
            max_length=max_length,
        )

    def test_empty_continuation_scores_zero(self) -> None:
        [[output]] = self._provider().logprobs([self._request("")])
        assert output.logprobs == []
        assert output.metadata["total_logprob"] == 0.0
        assert output.metadata["num_tokens"] == 0
        assert output.metadata["is_greedy"] is True

    def test_continuation_at_limit_scores(self) -> None:
        # "STOP" encodes to 2 tokens; a limit of exactly 2 must work
        [[output]] = self._provider(max_length=2).logprobs([self._request("STOP")])
        assert output.metadata["num_tokens"] == 2
        assert len(output.logprobs) == 2

    def test_continuation_over_limit_raises(self) -> None:
        with pytest.raises(ValueError, match="longer than"):
            self._provider(max_length=1).logprobs([self._request("STOP")])

    def test_request_max_length_overrides_provider(self) -> None:
        provider = self._provider(max_length=8)
        with pytest.raises(ValueError, match="longer than"):
            provider.logprobs([self._request("STOP", max_length=1)])
        [[output]] = provider.logprobs([self._request("STOP", max_length=2)])
        assert output.metadata["num_tokens"] == 2

    def test_nonpositive_limit_raises(self) -> None:
        with pytest.raises(ValueError, match="max_length > 0"):
            self._provider(max_length=0).logprobs([self._request("!")])


class TestVLMOverlongPrompt:
    """A prompt longer than the window fails its own request, not the whole batch."""

    @pytest.fixture(autouse=True)
    def _torch(self):
        self.torch = pytest.importorskip("torch")

    def test_only_the_overlong_request_fails(self) -> None:
        import contextlib
        import threading

        from olmo_eval.inference.errors import REQUEST_ERROR_KEY, request_error
        from olmo_eval.inference.providers.olmo_core_vlm.provider import OlmoCoreVLMProvider

        provider = OlmoCoreVLMProvider.__new__(OlmoCoreVLMProvider)
        provider.max_length = 5
        provider.use_cache = True
        provider._model_lock = threading.Lock()
        provider._autocast = contextlib.nullcontext
        provider._encode_request = lambda request: ([1] * len(request.prompt), None, None)
        batched: list = []
        provider._build_batch = lambda entries: batched.extend(entries) or entries
        provider._decode_batch_cached = lambda batch, params: [
            len(token_ids) for token_ids, *_ in batch
        ]
        provider._finalize_output = lambda generated, params: LMOutput(text=f"len {generated}")

        requests = [
            LMRequest(request_type=RequestType.CHAT, prompt=prompt)
            for prompt in ("ab", "abcdefgh", "abc")
        ]
        outputs = provider._generate_chunk(requests, SamplingParams(max_tokens=2))

        assert [o.text for o in outputs[0]] == ["len 2"]
        assert [o.text for o in outputs[2]] == ["len 3"]
        assert "prompt length (8) >= max_length (5)" in outputs[1][0].metadata[REQUEST_ERROR_KEY]
        assert request_error(outputs[1]) is not None
        assert request_error(outputs[0]) is None
        assert len(batched) == 2


class TestVLMContextAndCrops:
    """Defaults follow mm_olmo's eval: a 64k window and 8 crops per multi-image image."""

    @staticmethod
    def _info(fmt: str = "mm_olmo_dcp", *, llm=None, image=None, dataset=None):
        from olmo_eval.inference.providers.olmo_core_vlm.checkpoint import (
            MultimodalCheckpointInfo,
        )

        model_config = {"llm": llm or {}, "mm_preprocessor": {"image": image or {}}}
        config = {"dataset": dataset} if dataset is not None else {}
        return MultimodalCheckpointInfo(format=fmt, config=config, model_config=model_config)

    def test_max_length_floors_at_the_eval_length(self) -> None:
        from olmo_eval.inference.providers.olmo_core_vlm import preprocessing

        short = self._info(llm={"max_sequence_length": 16384})
        long = self._info(llm={"max_sequence_length": 131072})
        assert preprocessing.resolve_max_length(short, None) == 64_000
        assert preprocessing.resolve_max_length(long, None) == 131072
        assert preprocessing.resolve_max_length(short, 2048) == 2048
        assert preprocessing.resolve_max_length(self._info("olmo_core_dcp"), None) == 64_000

    def test_multi_image_crops(self) -> None:
        from olmo_eval.inference.providers.olmo_core_vlm import preprocessing

        resolve = preprocessing.resolve_max_multi_image_crops
        assert resolve(self._info(image={"max_multi_image_crops": 6}), None) == 6
        assert resolve(self._info(image={"max_crops": 24}), None) == 8
        assert resolve(self._info(image={"max_multi_image_crops": 6}), 3) == 3
        core = self._info("olmo_core_dcp", dataset={"max_multi_image_crops": 5})
        assert resolve(core, None) == 5

    def test_multi_image_requests_use_the_multi_image_budget(self, monkeypatch) -> None:
        torch = pytest.importorskip("torch")
        from olmo_eval.inference.providers.olmo_core_vlm.provider import OlmoCoreVLMProvider

        seen: list[int] = []

        def preprocess(image, dtype, device, *, image_size, patch_size, max_crops):
            seen.append(max_crops)
            return torch.zeros(1, 2, 3, 4), torch.zeros(1, 2, 3, dtype=torch.long), (1, 1, 1, 1)

        monkeypatch.setitem(
            sys.modules,
            "olmo_core.nn.vision.molmo2_image_processor",
            SimpleNamespace(preprocess_image_molmo2=preprocess),
        )
        monkeypatch.setitem(
            sys.modules,
            "olmo_core.nn.vision.molmo2_tokens",
            SimpleNamespace(build_image_token_ids=lambda *grid: [0]),
        )
        provider = OlmoCoreVLMProvider.__new__(OlmoCoreVLMProvider)
        provider.model_config = SimpleNamespace(
            vision=SimpleNamespace(image_default_input_size=(378, 378), image_patch_size=14)
        )
        provider.param_dtype = torch.float32
        provider.device = torch.device("cpu")
        provider.max_crops = 24
        provider.max_multi_image_crops = 8

        image = SimpleNamespace(mode="RGB")
        provider._preprocess_images((image,))
        provider._preprocess_images((image, image))
        assert seen == [24, 8, 8]


class TestUntiedLmHeadGuard:
    """A wrong checkpoint read that hands an untied model its embedding table as the
    LM head must fail loudly instead of scoring at chance."""

    @pytest.fixture(autouse=True)
    def _torch(self):
        self.torch = pytest.importorskip("torch")

    def _cfg(self, tied: bool):
        return SimpleNamespace(lm=SimpleNamespace(tie_word_embeddings=tied))

    def test_duplicated_head_raises(self) -> None:
        from olmo_eval.inference.providers.olmo_core_vlm.checkpoint import (
            _verify_lm_head_is_untied,
        )

        emb = self.torch.randn(8, 4)
        state = {"lm.embeddings.weight": emb, "lm.lm_head.w_out.weight": emb.clone()}
        with pytest.raises(ValueError, match="byte-identical"):
            _verify_lm_head_is_untied(state, self._cfg(tied=False), "/ckpt")

    def test_distinct_head_passes(self) -> None:
        from olmo_eval.inference.providers.olmo_core_vlm.checkpoint import (
            _verify_lm_head_is_untied,
        )

        state = {
            "lm.embeddings.weight": self.torch.randn(8, 4),
            "lm.lm_head.w_out.weight": self.torch.randn(8, 4),
        }
        _verify_lm_head_is_untied(state, self._cfg(tied=False), "/ckpt")

    def test_tied_model_is_exempt(self) -> None:
        from olmo_eval.inference.providers.olmo_core_vlm.checkpoint import (
            _verify_lm_head_is_untied,
        )

        emb = self.torch.randn(8, 4)
        state = {"lm.embeddings.weight": emb, "lm.lm_head.w_out.weight": emb}
        _verify_lm_head_is_untied(state, self._cfg(tied=True), "/ckpt")


class TestChatMessagesForRequest:
    """A completion prompt must survive harness-injected system messages."""

    def _request(self, prompt=None, messages=None):
        return LMRequest(
            request_type=RequestType.CHAT,
            prompt=prompt,
            messages=tuple(messages) if messages else None,
        )

    def test_bare_prompt_becomes_user_turn(self):
        from olmo_eval.inference.request_utils import chat_messages_for_request

        msgs = chat_messages_for_request(self._request(prompt="How many people?"))
        assert msgs == ({"role": "user", "content": "How many people?"},)

    def test_prompt_survives_injected_system_message(self):
        from olmo_eval.inference.request_utils import chat_messages_for_request

        msgs = chat_messages_for_request(
            self._request(
                prompt="How many people?", messages=[{"role": "system", "content": "Be terse."}]
            )
        )
        assert msgs == (
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "How many people?"},
        )

    def test_existing_user_turn_is_untouched(self):
        from olmo_eval.inference.request_utils import chat_messages_for_request

        original = (
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "Original question"},
        )
        msgs = chat_messages_for_request(self._request(prompt="ignored", messages=original))
        assert msgs == original

    def test_no_prompt_no_messages(self):
        from olmo_eval.inference.request_utils import chat_messages_for_request

        assert chat_messages_for_request(self._request()) == ()
