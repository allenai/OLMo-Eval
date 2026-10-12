"""Unit tests for the inline VLLMProvider."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from olmo_eval.common.types import LMRequest, RequestType, SamplingParams
from olmo_eval.inference.providers.vllm import VLLMProvider


class FakeVllmModule(ModuleType):
    SamplingParams: object


def _render_content(content: object) -> str:
    """Render a chat message's content, mirroring a real vision chat template.

    String content renders as-is (the text-only path). A content *list* renders each part,
    turning ``{"type": "image"}`` into a ``<image>`` placeholder, so tests can assert the
    placeholder actually lands in the prompt vLLM receives.
    """
    if isinstance(content, str):
        return content
    parts = []
    for part in content:
        if part.get("type") == "image":
            parts.append("<image>")
        else:
            parts.append(part.get("text", ""))
    return "".join(parts)


class FakeImage:
    """Minimal stand-in for a PIL image (only ``mode`` and ``convert`` are used)."""

    def __init__(self, name: str, mode: str = "RGB") -> None:
        self.name = name
        self.mode = mode
        self.converted_to: str | None = None

    def convert(self, mode: str) -> FakeImage:
        converted = FakeImage(self.name, mode)
        converted.converted_to = mode
        return converted


def test_build_sampling_params_passes_none_max_tokens_through(monkeypatch) -> None:
    # vLLM treats max_tokens=None as "generate to the context limit", so the
    # provider must forward None unchanged rather than crashing or coercing it.
    fake_vllm = FakeVllmModule("vllm")
    fake_vllm.SamplingParams = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    provider = VLLMProvider.__new__(VLLMProvider)
    built = provider._build_sampling_params(SamplingParams(max_tokens=None, do_sample=False))

    assert built["max_tokens"] is None


class FakeTokenizer:
    def __init__(self) -> None:
        self.template_calls: list[dict[str, object]] = []
        self.encode_calls: list[dict[str, object]] = []

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        **kwargs: object,
    ) -> str:
        self.template_calls.append(
            {
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
                **kwargs,
            }
        )
        rendered_messages = "|".join(
            f"{m['role']}:{_render_content(m['content'])}" for m in messages
        )
        return f"<chat>{rendered_messages}<assistant>"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        self.encode_calls.append(
            {
                "text": text,
                "add_special_tokens": add_special_tokens,
            }
        )
        return [ord(char) for char in text]


class FakeLLM:
    def __init__(self, tokenizer: FakeTokenizer) -> None:
        self.tokenizer = tokenizer
        self.generate_calls: list[dict[str, object]] = []

    def get_tokenizer(self) -> FakeTokenizer:
        return self.tokenizer

    def generate(
        self,
        prompts: list[str] | list[dict[str, list[int]]],
        sampling_params: object,
        use_tqdm: bool = False,
    ) -> list[object]:
        self.generate_calls.append(
            {
                "prompts": prompts,
                "sampling_params": sampling_params,
                "use_tqdm": use_tqdm,
            }
        )
        completion = SimpleNamespace(
            text="ok", logprobs=None, finish_reason="length", token_ids=[7]
        )
        return [SimpleNamespace(outputs=[completion]) for _ in prompts]


@pytest.fixture
def fake_provider() -> tuple[VLLMProvider, FakeLLM, FakeTokenizer]:
    tokenizer = FakeTokenizer()
    llm = FakeLLM(tokenizer)
    provider = VLLMProvider.__new__(VLLMProvider)
    provider.model_name = "test-model"
    provider.llm = llm
    provider._add_bos_token = None
    provider._max_images = 8
    provider._chat_template_kwargs = {}
    provider._build_sampling_params = lambda params: "fake-sampling-params"
    return provider, llm, tokenizer


def test_generate_formats_chat_messages_with_template(
    fake_provider: tuple[VLLMProvider, FakeLLM, FakeTokenizer],
) -> None:
    provider, llm, tokenizer = fake_provider
    request = LMRequest(
        request_type=RequestType.CHAT,
        messages=({"role": "user", "content": "Hello"},),
    )

    outputs = provider.generate([request], SamplingParams(max_tokens=1))

    assert outputs[0][0].text == "ok"
    assert tokenizer.template_calls == [
        {
            "messages": [{"role": "user", "content": "Hello"}],
            "tokenize": False,
            "add_generation_prompt": True,
        }
    ]
    assert llm.generate_calls[0]["prompts"] == ["<chat>user:Hello<assistant>"]


def test_generate_tokenizes_formatted_chat_prompt_when_bos_disabled(
    fake_provider: tuple[VLLMProvider, FakeLLM, FakeTokenizer],
) -> None:
    provider, llm, tokenizer = fake_provider
    provider._add_bos_token = False
    request = LMRequest(
        request_type=RequestType.CHAT,
        messages=({"role": "user", "content": "Hello"},),
    )

    provider.generate([request], SamplingParams(max_tokens=1))

    assert tokenizer.encode_calls == [
        {
            "text": "<chat>user:Hello<assistant>",
            "add_special_tokens": False,
        }
    ]
    assert llm.generate_calls[0]["prompts"] == [
        {"prompt_token_ids": [ord(char) for char in "<chat>user:Hello<assistant>"]}
    ]


def test_generate_keeps_completion_prompt_unchanged(
    fake_provider: tuple[VLLMProvider, FakeLLM, FakeTokenizer],
) -> None:
    provider, llm, tokenizer = fake_provider
    request = LMRequest(request_type=RequestType.COMPLETION, prompt="Complete me")

    provider.generate([request], SamplingParams(max_tokens=1))

    assert tokenizer.template_calls == []
    assert llm.generate_calls[0]["prompts"] == ["Complete me"]


def test_generate_preserves_termination_without_logprobs(fake_provider) -> None:
    provider, _, _ = fake_provider
    request = LMRequest(request_type=RequestType.COMPLETION, prompt="Q")
    output = provider.generate([request], SamplingParams(max_tokens=1))[0][0]
    assert output.metadata["finish_reason"] == "length"
    assert output.metadata["completion_tokens"] == 1


# ---------------------------------------------------------------------------
# Image support
# ---------------------------------------------------------------------------


def test_generate_passes_images_as_multi_modal_data(
    fake_provider: tuple[VLLMProvider, FakeLLM, FakeTokenizer],
) -> None:
    # vLLM takes images via {"prompt": ..., "multi_modal_data": {"image": [...]}}, and the
    # prompt must carry the model's image placeholders for vLLM to expand against them.
    provider, llm, tokenizer = fake_provider
    image = FakeImage("chart")
    request = LMRequest(
        request_type=RequestType.CHAT,
        messages=({"role": "user", "content": "What is shown?"},),
        images=(image,),
    )

    outputs = provider.generate([request], SamplingParams(max_tokens=8))

    assert outputs[0][0].text == "ok"
    prompt = llm.generate_calls[0]["prompts"][0]
    assert isinstance(prompt, dict), "image requests must use vLLM's dict prompt form"
    assert prompt["multi_modal_data"] == {"image": [image]}
    # Placeholder precedes the question text, matching the HF/olmo_core_vlm convention.
    assert prompt["prompt"] == "<chat>user:<image>What is shown?<assistant>"


def test_generate_places_one_placeholder_per_image(
    fake_provider: tuple[VLLMProvider, FakeLLM, FakeTokenizer],
) -> None:
    # MMMU-Pro's standard settings interleave up to 7 images; a placeholder count that
    # disagrees with the image count is exactly the failure that yields plausible-but-wrong
    # scores rather than an error.
    provider, llm, _ = fake_provider
    images = tuple(FakeImage(f"img{i}") for i in range(3))
    request = LMRequest(
        request_type=RequestType.CHAT,
        messages=({"role": "user", "content": "Compare."},),
        images=images,
    )

    provider.generate([request], SamplingParams(max_tokens=8))

    prompt = llm.generate_calls[0]["prompts"][0]
    assert prompt["prompt"].count("<image>") == 3
    assert prompt["multi_modal_data"]["image"] == list(images)


def test_generate_converts_non_rgb_images(
    fake_provider: tuple[VLLMProvider, FakeLLM, FakeTokenizer],
) -> None:
    provider, llm, _ = fake_provider
    request = LMRequest(
        request_type=RequestType.CHAT,
        messages=({"role": "user", "content": "Q"},),
        images=(FakeImage("grayscale", mode="L"),),
    )

    provider.generate([request], SamplingParams(max_tokens=8))

    passed = llm.generate_calls[0]["prompts"][0]["multi_modal_data"]["image"][0]
    assert passed.mode == "RGB"
    assert passed.converted_to == "RGB"


def test_generate_text_only_requests_stay_bare_strings(
    fake_provider: tuple[VLLMProvider, FakeLLM, FakeTokenizer],
) -> None:
    # Regression guard: adding image support must not change the text-only wire format.
    provider, llm, _ = fake_provider
    request = LMRequest(
        request_type=RequestType.CHAT,
        messages=({"role": "user", "content": "Hello"},),
    )

    provider.generate([request], SamplingParams(max_tokens=1))

    assert llm.generate_calls[0]["prompts"] == ["<chat>user:Hello<assistant>"]


def test_images_with_bos_disabled_raises(
    fake_provider: tuple[VLLMProvider, FakeLLM, FakeTokenizer],
) -> None:
    # add_bos_token=False pre-tokenizes to prompt_token_ids, which cannot carry
    # multi_modal_data -- the images would be silently dropped.
    provider, _, _ = fake_provider
    provider._add_bos_token = False
    request = LMRequest(
        request_type=RequestType.CHAT,
        messages=({"role": "user", "content": "Q"},),
        images=(FakeImage("x"),),
    )

    with pytest.raises(ValueError, match="add_bos_token=False is incompatible"):
        provider.generate([request], SamplingParams(max_tokens=8))


def test_logprobs_with_images_raises(
    fake_provider: tuple[VLLMProvider, FakeLLM, FakeTokenizer],
    monkeypatch,
) -> None:
    # The loglikelihood path scores raw token sequences with nowhere to attach images, so
    # it must refuse rather than score every continuation against text alone.
    fake_vllm = FakeVllmModule("vllm")
    fake_vllm.SamplingParams = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    provider, _, _ = fake_provider
    request = LMRequest(
        request_type=RequestType.LOGLIKELIHOOD,
        prompt="Q",
        continuations=("A", "B"),
        images=(FakeImage("x"),),
    )

    with pytest.raises(ValueError, match="does not support images"):
        provider.logprobs([request], SamplingParams(max_tokens=1))


def test_init_raises_per_prompt_image_limit(monkeypatch) -> None:
    # vLLM's own limit_mm_per_prompt default is 1 image, which silently breaks every
    # interleaved-image task, so the provider must raise it at engine construction.
    captured: dict[str, object] = {}

    class RecordingLLM:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    fake_vllm = FakeVllmModule("vllm")
    fake_vllm.LLM = RecordingLLM
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    VLLMProvider("test-model", max_images=7)

    assert captured["limit_mm_per_prompt"] == {"image": 7}


def test_init_respects_explicit_limit_mm_per_prompt(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class RecordingLLM:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    fake_vllm = FakeVllmModule("vllm")
    fake_vllm.LLM = RecordingLLM
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    VLLMProvider("test-model", limit_mm_per_prompt={"image": 2, "video": 1})

    assert captured["limit_mm_per_prompt"] == {"image": 2, "video": 1}


def test_describe_request_reports_image_count(
    fake_provider: tuple[VLLMProvider, FakeLLM, FakeTokenizer],
) -> None:
    # --inspect-request should show that images are on the wire, and that the prompt goes
    # out as multi_modal_data rather than plain text.
    provider, _, _ = fake_provider
    # The shared fixture stubs _build_sampling_params to a string; describe_request reads
    # attributes off it, so give it a real-shaped object here.
    provider._build_sampling_params = lambda params: SimpleNamespace(
        max_tokens=params.max_tokens, temperature=params.temperature, logprobs=1, n=1
    )
    request = LMRequest(
        request_type=RequestType.CHAT,
        messages=({"role": "user", "content": "Q"},),
        images=(FakeImage("a"), FakeImage("b")),
    )

    trace = provider.describe_request(request, SamplingParams(max_tokens=8))

    assert trace is not None
    assert trace["input_mode"] == "multi_modal_data"
    assert trace["num_images"] == 2


def test_too_many_images_names_the_knob(
    fake_provider: tuple[VLLMProvider, FakeLLM, FakeTokenizer],
) -> None:
    # vLLM fails the entire llm.generate call when one prompt exceeds
    # limit_mm_per_prompt, so an oversized instance takes its whole chunk with it.
    # Its own message ("At most N image(s)...") names neither the instance nor the knob.
    provider, _, _ = fake_provider
    provider._max_images = 2
    request = LMRequest(
        request_type=RequestType.CHAT,
        messages=({"role": "user", "content": "Q"},),
        images=tuple(FakeImage(f"i{n}") for n in range(3)),
    )

    with pytest.raises(ValueError, match=r"3 images but max_images=2"):
        provider.generate([request], SamplingParams(max_tokens=8))


def test_chat_template_kwargs_are_forwarded(
    fake_provider: tuple[VLLMProvider, FakeLLM, FakeTokenizer],
) -> None:
    # The hybrid Qwen3 releases reason by default; comparing them against a non-thinking
    # model requires enable_thinking=False to reach the template.
    provider, _, tokenizer = fake_provider
    provider._chat_template_kwargs = {"enable_thinking": False}
    request = LMRequest(request_type=RequestType.CHAT, messages=({"role": "user", "content": "Q"},))

    provider.generate([request], SamplingParams(max_tokens=8))

    assert tokenizer.template_calls[0]["enable_thinking"] is False


def test_max_images_guard_tracks_the_effective_engine_limit(monkeypatch) -> None:
    # limit_mm_per_prompt is applied with setdefault, so an explicit engine limit wins. If
    # the guard kept reading max_images it would be permissive in exactly the case it
    # exists for -- an engine limit *below* max_images, where vLLM fails the whole chunk.
    captured: dict[str, object] = {}

    class RecordingLLM:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    fake_vllm = FakeVllmModule("vllm")
    fake_vllm.LLM = RecordingLLM
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    provider = VLLMProvider("test-model", max_images=8, limit_mm_per_prompt={"image": 2})

    assert captured["limit_mm_per_prompt"] == {"image": 2}
    assert provider._max_images == 2


def test_generate_resolves_lazy_image_references(
    fake_provider: tuple[VLLMProvider, FakeLLM, FakeTokenizer],
) -> None:
    # Vision tasks attach lazy references (zero-arg callables or paths) and leave unused
    # MMMU-Pro slots as None; the provider resolves them, and the holes must neither reach
    # vLLM nor be counted as placeholders.
    provider, llm, _ = fake_provider
    first, second = FakeImage("a"), FakeImage("b")
    request = LMRequest(
        request_type=RequestType.CHAT,
        messages=({"role": "user", "content": "Q"},),
        images=(lambda: first, None, lambda: second),
    )

    provider.generate([request], SamplingParams(max_tokens=8))

    prompt = llm.generate_calls[0]["prompts"][0]
    assert prompt["multi_modal_data"]["image"] == [first, second]
    assert prompt["prompt"].count("<image>") == 2


def test_provider_advertises_image_support() -> None:
    # The harness rejects image requests for providers that do not set this.
    assert VLLMProvider.supports_images is True
