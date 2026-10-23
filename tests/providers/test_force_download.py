"""Tests for Hugging Face force-download wiring."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch


def test_huggingface_provider_passes_force_download_to_tokenizer_and_model():
    """The HF provider should refresh both tokenizer and model loads."""
    from olmo_eval.inference.providers.huggingface import HuggingFaceProvider

    tokenizer = MagicMock()
    model = MagicMock()
    auto_tokenizer = SimpleNamespace(from_pretrained=MagicMock(return_value=tokenizer))
    auto_model = SimpleNamespace(from_pretrained=MagicMock(return_value=model))
    # the provider inspects the config to decide between the causal and
    # seq2seq auto-classes; a causal model reports is_encoder_decoder=False
    auto_config = SimpleNamespace(
        from_pretrained=MagicMock(return_value=SimpleNamespace(is_encoder_decoder=False))
    )
    auto_seq2seq = SimpleNamespace(from_pretrained=MagicMock())
    fake_transformers = SimpleNamespace(
        AutoTokenizer=auto_tokenizer,
        AutoModelForCausalLM=auto_model,
        AutoModelForSeq2SeqLM=auto_seq2seq,
        AutoConfig=auto_config,
    )
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
        device=lambda name: name,
    )

    with patch.dict("sys.modules", {"transformers": fake_transformers, "torch": fake_torch}):
        HuggingFaceProvider(
            "org/model",
            tokenizer="org/tokenizer",
            revision="main",
            trust_remote_code=True,
            force_download=True,
            dtype="float16",
        )

    auto_tokenizer.from_pretrained.assert_called_once_with(
        "org/tokenizer",
        revision="main",
        trust_remote_code=True,
        force_download=True,
    )
    auto_model.from_pretrained.assert_called_once_with(
        "org/model",
        revision="main",
        trust_remote_code=True,
        force_download=True,
        dtype="float16",
    )
    model.to.assert_called_once_with("cpu")
    # a causal model must not be routed through the seq2seq class
    auto_seq2seq.from_pretrained.assert_not_called()
    model.eval.assert_called_once_with()


def test_vllm_provider_force_download_refreshes_cache_without_forwarding_to_engine():
    """vLLM has no force-download engine arg, so refresh before engine startup."""
    from olmo_eval.inference.providers.vllm import VLLMProvider

    llm_cls = MagicMock()
    fake_vllm = SimpleNamespace(LLM=llm_cls)

    with (
        patch.dict("sys.modules", {"vllm": fake_vllm}),
        patch("olmo_eval.inference.providers.vllm.refresh_hf_cache") as refresh,
    ):
        VLLMProvider(
            "org/model",
            tokenizer="org/tokenizer",
            force_download=True,
            revision="main",
            download_dir="/tmp/hf-cache",
            token=True,
        )

    assert refresh.call_args_list == [
        call(
            "org/model",
            revision="main",
            cache_dir="/tmp/hf-cache",
            token=True,
            force_download=True,
        ),
        call(
            "org/tokenizer",
            revision="main",
            cache_dir="/tmp/hf-cache",
            token=True,
            force_download=True,
        ),
    ]
    engine_kwargs = llm_cls.call_args.kwargs
    assert engine_kwargs["model"] == "org/model"
    assert "force_download" not in engine_kwargs


def test_huggingface_provider_uses_seq2seq_class_for_encoder_decoder():
    """Encoder-decoder models must load via AutoModelForSeq2SeqLM.

    olmo-eval needs this for AutoAIS (google/t5_xxl_true_nli_mixture), and
    vLLM is not an alternative -- it has no T5 support.
    """
    from olmo_eval.inference.providers.huggingface import HuggingFaceProvider

    tokenizer = MagicMock()
    seq2seq_model = MagicMock()
    auto_tokenizer = SimpleNamespace(from_pretrained=MagicMock(return_value=tokenizer))
    auto_causal = SimpleNamespace(from_pretrained=MagicMock())
    auto_seq2seq = SimpleNamespace(from_pretrained=MagicMock(return_value=seq2seq_model))
    auto_config = SimpleNamespace(
        from_pretrained=MagicMock(return_value=SimpleNamespace(is_encoder_decoder=True))
    )
    fake_transformers = SimpleNamespace(
        AutoTokenizer=auto_tokenizer,
        AutoModelForCausalLM=auto_causal,
        AutoModelForSeq2SeqLM=auto_seq2seq,
        AutoConfig=auto_config,
    )
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
        device=lambda name: name,
    )

    with patch.dict("sys.modules", {"transformers": fake_transformers, "torch": fake_torch}):
        provider = HuggingFaceProvider("google/t5_xxl_true_nli_mixture")

    assert provider.is_encoder_decoder is True
    auto_seq2seq.from_pretrained.assert_called_once()
    # the causal class must not be used for an encoder-decoder model
    auto_causal.from_pretrained.assert_not_called()
