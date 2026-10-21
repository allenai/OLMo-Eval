"""CI-safe tests for the multimodal HuggingFaceProvider path (mocked transformers).

No GPU, no real model, no network: ``transformers`` / ``torch`` are mocked and
token tensors are plain numpy arrays. The real ``allenai/Molmo2-4B`` smoke is
env-gated and runs separately. These guard the multimodal wiring: loading via
``AutoProcessor`` + ``AutoModelForImageTextToText`` (not the text-only
``AutoModelForCausalLM``) and the image-bearing ``generate`` path.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from olmo_eval.common.types import LMRequest, RequestType, SamplingParams


def _fake_torch():
    """A torch stand-in: cpu device + a no-op ``no_grad`` context manager."""
    return SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
        device=lambda name: name,
        no_grad=contextlib.nullcontext,
    )


def _fake_multimodal_transformers(processor, model):
    """A transformers stand-in exposing the multimodal Auto classes."""
    return SimpleNamespace(
        AutoProcessor=SimpleNamespace(from_pretrained=MagicMock(return_value=processor)),
        AutoModelForImageTextToText=SimpleNamespace(from_pretrained=MagicMock(return_value=model)),
        # Present so a stray text-path import would be detectable, but it must
        # not be used when multimodal=True.
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=MagicMock()),
        AutoTokenizer=SimpleNamespace(from_pretrained=MagicMock()),
    )


def _build_provider(processor, model):
    """Construct a multimodal HuggingFaceProvider with everything mocked."""
    from olmo_eval.inference.providers.huggingface import HuggingFaceProvider

    fake_transformers = _fake_multimodal_transformers(processor, model)
    # The RoPE shim re-registers the "default" init fn; pre-seed it so the shim
    # is a no-op and never touches torch.arange.
    fake_rope = SimpleNamespace(ROPE_INIT_FUNCTIONS={"default": object()})

    with patch.dict(
        "sys.modules",
        {
            "transformers": fake_transformers,
            "transformers.modeling_rope_utils": fake_rope,
            "torch": _fake_torch(),
        },
    ):
        provider = HuggingFaceProvider(
            "allenai/Molmo2-4B",
            multimodal=True,
            max_crops=24,
            trust_remote_code=True,
            dtype="bfloat16",
        )
    return provider, fake_transformers


def test_multimodal_multi_image_request_uses_multi_image_crops():
    processor = MagicMock()
    processor.tokenizer = MagicMock()
    processor.tokenizer.decode = MagicMock(return_value="A")
    processor.apply_chat_template = MagicMock(return_value="<formatted-chat>")
    processor.return_value = {"input_ids": np.array([[1, 2, 3]])}
    model = MagicMock()
    model.named_modules.return_value = []
    model.generate.return_value = np.array([[1, 2, 3, 7]])

    provider, _ = _build_provider(processor, model)
    images = (SimpleNamespace(mode="RGB"), SimpleNamespace(mode="RGB"))
    request = LMRequest(
        request_type=RequestType.CHAT,
        messages=({"role": "user", "content": "Which image?"},),
        images=images,
    )
    with patch.dict("sys.modules", {"torch": _fake_torch()}):
        provider.generate([request], SamplingParams(temperature=0.0, max_tokens=4))

    # mm_olmo crops each image of a multi-image prompt to max_multi_image_crops (8),
    # not the single-image budget the preset raises to 24.
    _, proc_kwargs = processor.call_args
    assert proc_kwargs["max_crops"] == 8


def test_multimodal_init_uses_processor_and_image_text_model():
    processor = MagicMock()
    processor.tokenizer = MagicMock()
    model = MagicMock()
    model.named_modules.return_value = []  # _reinit_rope_buffers iterates this

    provider, fake_transformers = _build_provider(processor, model)

    assert provider.is_multimodal is True
    assert provider.max_crops == 24
    # Loaded the image-text-to-text model + processor, not the text-only CausalLM.
    fake_transformers.AutoModelForImageTextToText.from_pretrained.assert_called_once()
    fake_transformers.AutoProcessor.from_pretrained.assert_called_once()
    fake_transformers.AutoModelForCausalLM.from_pretrained.assert_not_called()
    # dtype/trust_remote_code flow through to the model load.
    _, model_kwargs = fake_transformers.AutoModelForImageTextToText.from_pretrained.call_args
    assert model_kwargs["dtype"] == "bfloat16"
    assert model_kwargs["trust_remote_code"] is True
    # The processor's tokenizer is reused so shared decode helpers work.
    assert provider.tokenizer is processor.tokenizer
    model.eval.assert_called_once_with()


def test_multimodal_generate_with_image_builds_chat_and_decodes():
    processor = MagicMock()
    processor.tokenizer = MagicMock()
    processor.tokenizer.decode = MagicMock(return_value="  8 people  ")
    processor.apply_chat_template = MagicMock(return_value="<formatted-chat>")
    processor.return_value = {"input_ids": np.array([[1, 2, 3]])}  # prompt_len = 3
    model = MagicMock()
    model.named_modules.return_value = []
    model.generate.return_value = np.array([[1, 2, 3, 7, 8]])  # 2 new tokens after prompt

    provider, _ = _build_provider(processor, model)

    image = SimpleNamespace(mode="RGB")  # stand-in PIL image; no conversion needed
    request = LMRequest(
        request_type=RequestType.CHAT,
        messages=({"role": "user", "content": "How many people?"},),
        images=(image,),
    )

    with patch.dict("sys.modules", {"torch": _fake_torch()}):
        outputs = provider.generate([request], SamplingParams(temperature=0.0, max_tokens=16))

    assert len(outputs) == 1 and len(outputs[0]) == 1
    assert outputs[0][0].text == "8 people"  # decoded + stripped

    # Chat template built with image + text content in a single user turn.
    chat_arg = processor.apply_chat_template.call_args.args[0]
    content = chat_arg[0]["content"]
    assert {"type": "image", "image": image} in content
    assert {"type": "text", "text": "How many people?"} in content

    # Processor invoked with the image(s), formatted text, and max_crops.
    _, proc_kwargs = processor.call_args
    assert proc_kwargs["images"] == [image]
    assert proc_kwargs["text"] == "<formatted-chat>"
    assert proc_kwargs["max_crops"] == 24

    # Greedy decoding at temperature 0.
    _, gen_kwargs = model.generate.call_args
    assert gen_kwargs["do_sample"] is False
    assert gen_kwargs["max_new_tokens"] == 16


def test_multimodal_logprobs_not_supported():
    import pytest

    processor = MagicMock()
    processor.tokenizer = MagicMock()
    model = MagicMock()
    model.named_modules.return_value = []
    provider, _ = _build_provider(processor, model)

    request = LMRequest(
        request_type=RequestType.LOGLIKELIHOOD,
        prompt="ctx",
        continuations=("a",),
    )
    with pytest.raises(NotImplementedError):
        provider.logprobs([request])


def test_masking_kwargs_shim_matches_the_installed_signature():
    """The Molmo2 remote code's ``input_embeds``/``cache_position`` kwargs reach transformers
    in whatever form the installed ``masking_utils`` accepts."""
    from olmo_eval.inference.providers.huggingface import _patch_masking_kwargs

    calls: dict[str, dict] = {}

    def create_causal_mask(
        config, inputs_embeds, attention_mask, past_key_values, position_ids=None
    ):
        calls["causal"] = {"inputs_embeds": inputs_embeds, "position_ids": position_ids}

    def create_masks_for_generate(config, inputs_embeds, attention_mask, past_key_values, **kwargs):
        calls["generate"] = {"inputs_embeds": inputs_embeds, **kwargs}

    fake = SimpleNamespace(
        create_causal_mask=create_causal_mask, create_masks_for_generate=create_masks_for_generate
    )
    with patch.dict("sys.modules", {"transformers.masking_utils": fake}):
        _patch_masking_kwargs()
        _patch_masking_kwargs()  # idempotent: the second call must not wrap the wrapper
        assert fake.create_causal_mask.__wrapped__ is create_causal_mask
        fake.create_causal_mask(
            config=None,
            input_embeds="x",
            attention_mask=None,
            cache_position=3,
            past_key_values=None,
            position_ids="p",
        )
        fake.create_masks_for_generate(
            config=None,
            input_embeds="y",
            attention_mask=None,
            cache_position=3,
            past_key_values=None,
        )

    # create_causal_mask lost cache_position in 5.9, so it is dropped there ...
    assert calls["causal"] == {"inputs_embeds": "x", "position_ids": "p"}
    # ... while create_masks_for_generate still takes **kwargs, so it keeps it.
    assert calls["generate"] == {"inputs_embeds": "y", "cache_position": 3}
