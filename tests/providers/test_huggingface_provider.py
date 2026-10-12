"""Unit tests for HuggingFaceProvider generate-kwargs construction."""

from types import SimpleNamespace

import pytest

from olmo_eval.common.types import LMRequest, RequestType, SamplingParams
from olmo_eval.inference.providers.huggingface import HuggingFaceProvider


@pytest.fixture
def provider() -> HuggingFaceProvider:
    instance = HuggingFaceProvider.__new__(HuggingFaceProvider)
    instance.model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=2048))
    return instance


def test_finite_max_tokens_passes_through(provider: HuggingFaceProvider) -> None:
    kwargs = provider._build_generate_kwargs(SamplingParams(max_tokens=512), prompt_len=100)
    assert kwargs["max_new_tokens"] == 512


def test_uncapped_reserves_room_after_prompt(provider: HuggingFaceProvider) -> None:
    kwargs = provider._build_generate_kwargs(SamplingParams(max_tokens=None), prompt_len=2000)
    assert kwargs["max_new_tokens"] == 2048 - 2000


def test_uncapped_with_no_prompt_uses_full_context(provider: HuggingFaceProvider) -> None:
    kwargs = provider._build_generate_kwargs(SamplingParams(max_tokens=None))
    assert kwargs["max_new_tokens"] == 2048


def test_uncapped_floors_at_one_when_prompt_exceeds_context(provider: HuggingFaceProvider) -> None:
    kwargs = provider._build_generate_kwargs(SamplingParams(max_tokens=None), prompt_len=5000)
    assert kwargs["max_new_tokens"] == 1


class _FakeChatTokenizer:
    def apply_chat_template(
        self,
        messages: list[dict[str, object]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str:
        rendered = "|".join(f"{m['role']}:{m['content']}" for m in messages)
        return f"<chat>{rendered}<assistant>"


def test_text_path_renders_chat_messages(provider: HuggingFaceProvider) -> None:
    # A CHAT request keeps its content on `messages` and leaves `prompt` empty. Reading
    # `prompt` alone produced "" -> zero tokens, which surfaced from inside attention as
    # "cannot reshape tensor of 0 elements into shape [1, 0, -1, 128]" and failed all 900
    # MMMU instances on a text-only backbone run.
    provider.tokenizer = _FakeChatTokenizer()
    request = LMRequest(
        request_type=RequestType.CHAT,
        messages=({"role": "user", "content": "vqa2: What is 2+2?"},),
    )

    assert provider._format_text_prompt(request) == "<chat>user:vqa2: What is 2+2?<assistant>"


def test_text_path_leaves_completion_prompts_alone(provider: HuggingFaceProvider) -> None:
    provider.tokenizer = _FakeChatTokenizer()
    request = LMRequest(request_type=RequestType.COMPLETION, prompt="raw text")

    assert provider._format_text_prompt(request) == "raw text"


class _FakeAutoConfig:
    """Stand-in for a loaded HF config."""

    def __init__(self, *, vision_config=None, vit_config=None, architectures=None, auto_map=None):
        if vision_config is not None:
            self.vision_config = vision_config
        if vit_config is not None:
            self.vit_config = vit_config
        if auto_map is not None:
            self.auto_map = auto_map
        self.architectures = architectures or []


def _patch_autoconfig(monkeypatch, config):
    import transformers

    class _Auto:
        @staticmethod
        def from_pretrained(name, **kwargs):
            if config is None:
                raise OSError("no config")
            return config

    monkeypatch.setattr(transformers, "AutoConfig", _Auto)


def test_looks_multimodal_detects_a_vision_config(monkeypatch) -> None:
    # A multimodal HF directory (e.g. one written by tools/olmo_core_to_hf) used to fall
    # through to the text path and die on AutoModelForCausalLM.
    from olmo_eval.inference.providers.huggingface import looks_multimodal_hf

    _patch_autoconfig(monkeypatch, _FakeAutoConfig(vision_config=object()))
    assert looks_multimodal_hf("some/model") is True


def test_looks_multimodal_detects_a_known_architecture(monkeypatch) -> None:
    from olmo_eval.inference.providers.huggingface import looks_multimodal_hf

    _patch_autoconfig(
        monkeypatch, _FakeAutoConfig(architectures=["Qwen3VLForConditionalGeneration"])
    )
    assert looks_multimodal_hf("some/model") is True


def test_looks_multimodal_is_false_for_a_text_model(monkeypatch) -> None:
    from olmo_eval.inference.providers.huggingface import looks_multimodal_hf

    _patch_autoconfig(monkeypatch, _FakeAutoConfig(architectures=["Qwen3ForCausalLM"]))
    assert looks_multimodal_hf("some/model") is False


def test_looks_multimodal_is_false_when_the_config_cannot_be_read(monkeypatch) -> None:
    # Falling back to the text path is no worse than the previous behaviour; raising here
    # would break every text run whose config needs kwargs we do not forward.
    from olmo_eval.inference.providers.huggingface import looks_multimodal_hf

    _patch_autoconfig(monkeypatch, None)
    assert looks_multimodal_hf("some/model") is False


def test_looks_multimodal_detects_molmo2_via_auto_map(monkeypatch) -> None:
    # The real shape, which the first attempt missed: Molmo2 names its vision tower
    # `vit_config` (not `vision_config`) and its architecture is remote code, so it is
    # absent from transformers' built-in mapping. Only auto_map identifies it.
    from olmo_eval.inference.providers.huggingface import looks_multimodal_hf

    _patch_autoconfig(
        monkeypatch,
        _FakeAutoConfig(
            vit_config=object(),
            architectures=["Molmo2ForConditionalGeneration"],
            auto_map={
                "AutoConfig": "configuration_molmo2.Molmo2Config",
                "AutoModelForImageTextToText": "modeling_molmo2.Molmo2ForConditionalGeneration",
            },
        ),
    )
    assert looks_multimodal_hf("some/export") is True


def test_looks_multimodal_detects_a_vit_config_alone(monkeypatch) -> None:
    from olmo_eval.inference.providers.huggingface import looks_multimodal_hf

    _patch_autoconfig(monkeypatch, _FakeAutoConfig(vit_config=object()))
    assert looks_multimodal_hf("some/model") is True


def test_looks_multimodal_ignores_a_text_only_auto_map(monkeypatch) -> None:
    from olmo_eval.inference.providers.huggingface import looks_multimodal_hf

    _patch_autoconfig(
        monkeypatch,
        _FakeAutoConfig(
            architectures=["Qwen3ForCausalLM"],
            auto_map={"AutoModelForCausalLM": "modeling_x.XForCausalLM"},
        ),
    )
    assert looks_multimodal_hf("some/model") is False


def test_looks_multimodal_names_the_flag_for_a_remote_code_repo(monkeypatch) -> None:
    # Returning False here would send Molmo2 down the text path, where it dies with an
    # error that names neither the cause nor the fix.
    import transformers

    from olmo_eval.inference.providers.huggingface import looks_multimodal_hf

    class _Auto:
        @staticmethod
        def from_pretrained(name, **kwargs):
            assert kwargs["trust_remote_code"] is False
            raise ValueError("The repository contains custom code ... trust_remote_code=True")

    monkeypatch.setattr(transformers, "AutoConfig", _Auto)
    with pytest.raises(ValueError, match="provider.trust_remote_code=true"):
        looks_multimodal_hf("allenai/Molmo2-4B")
