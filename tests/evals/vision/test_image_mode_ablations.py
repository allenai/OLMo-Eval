"""CI-safe tests for the ``image_mode`` perception-vs-knowledge ablations.

No data files, model, GPU, or network. These guard the three request shapes the ablations
depend on -- real image, no image, caption-instead-of-image -- and the failure modes that
would otherwise produce a plausible-looking but meaningless score.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import pytest

import olmo_eval.evals  # noqa: F401  (registration side effect)
from olmo_eval.common.types import Instance, RequestType
from olmo_eval.evals.tasks.common.base import TaskConfig
from olmo_eval.evals.tasks.common.registry import get_task
from olmo_eval.evals.vision.benchmarks.mmmu_pro import MmmuProTask
from olmo_eval.evals.vision.tasks.image_qa import ImageQATask
from olmo_eval.evals.vision.tasks.prompting import (
    COT_MC_INSTRUCTION,
    COT_OPEN_INSTRUCTION,
    extract_cot_answer,
    reject_unsupported_prompt_knobs,
)

Image = pytest.importorskip("PIL.Image")


class _DummyImageQATask(ImageQATask):
    def _build_instances(self) -> Iterator[Instance]:
        return iter(())


def _tiny_image():
    return Image.new("RGB", (8, 8))


def _instance(example_id: str = "ex1"):
    return Instance(
        question="What is the value?",
        metadata={"image": _tiny_image(), "example_id": example_id},
    )


def _caption_file(tmp_path, records: dict[str, str]):
    path = tmp_path / "captions.jsonl"
    with open(path, "w") as f:
        for example_id, caption in records.items():
            f.write(json.dumps({"example_id": example_id, "caption": caption}) + "\n")
    return str(path)


# ---------------------------------------------------------------------------
# The three modes
# ---------------------------------------------------------------------------


def test_default_mode_still_attaches_the_image():
    # Regression guard: the ablation machinery must not perturb the published benchmarks.
    task = _DummyImageQATask(TaskConfig(name="dummy"))
    instance = _instance()

    request = task.format_request(instance)

    assert request.request_type == RequestType.CHAT
    assert request.images == (instance.metadata["image"],)
    assert request.messages[0]["content"] == "What is the value?"


def test_text_only_mode_drops_the_image_and_keeps_the_question():
    task = _DummyImageQATask(TaskConfig(name="dummy", image_mode="none"))

    request = task.format_request(_instance())

    assert request.images is None
    assert request.messages[0]["content"] == "What is the value?"


def test_caption_mode_substitutes_the_caption_for_the_image(tmp_path):
    source = _caption_file(tmp_path, {"ex1": "A bar chart peaking at 60."})
    task = _DummyImageQATask(TaskConfig(name="dummy", image_mode="caption", caption_source=source))

    request = task.format_request(_instance())

    assert request.images is None
    content = request.messages[0]["content"]
    assert "A bar chart peaking at 60." in content
    assert content.endswith("What is the value?")


# ---------------------------------------------------------------------------
# Failure modes that would otherwise score silently
# ---------------------------------------------------------------------------


def test_missing_caption_raises_rather_than_falling_back_to_text_only(tmp_path):
    # Silently dropping to text-only would collapse the two conditions the experiment is
    # built to distinguish, and still report a number.
    source = _caption_file(tmp_path, {"other": "irrelevant"})
    task = _DummyImageQATask(TaskConfig(name="dummy", image_mode="caption", caption_source=source))

    with pytest.raises(ValueError, match="No caption for example_id"):
        task.format_request(_instance("ex1"))


def test_caption_mode_without_a_source_raises():
    task = _DummyImageQATask(TaskConfig(name="dummy", image_mode="caption"))

    with pytest.raises(ValueError, match="requires caption_source"):
        task.format_request(_instance())


def test_unknown_image_mode_raises():
    task = _DummyImageQATask(TaskConfig(name="dummy", image_mode="oracle"))

    with pytest.raises(ValueError, match="Unknown image_mode"):
        task.format_request(_instance())


def test_empty_caption_file_raises(tmp_path):
    source = _caption_file(tmp_path, {})
    task = _DummyImageQATask(TaskConfig(name="dummy", image_mode="caption", caption_source=source))

    with pytest.raises(ValueError, match="is empty"):
        task.format_request(_instance())


# ---------------------------------------------------------------------------
# MMMU-Pro: the vision setting cannot be run imageless
# ---------------------------------------------------------------------------


def test_mmmu_pro_vision_setting_rejects_imageless_modes():
    # In the vision setting the question is rendered into the screenshot, so
    # instance.question is only the answer-format instruction. Dropping the image leaves
    # nothing to answer -- and the official parser falls back to a seeded random choice,
    # so it would report ~25% rather than failing.
    task = MmmuProTask(TaskConfig(name="mmmu_pro", image_mode="none"))
    instance = Instance(
        question="Answer with the option letter from the given choices directly.",
        metadata={
            "mmmu_pro_setting": "vision",
            "example_id": "v1",
            "image": _tiny_image(),
        },
    )

    with pytest.raises(ValueError, match="vision setting"):
        task.format_request(instance)


def test_mmmu_pro_standard_setting_drops_all_interleaved_images():
    task = MmmuProTask(TaskConfig(name="mmmu_pro", image_mode="none"))
    instance = Instance(
        question="Which is larger?\nA. x\nB. y",
        metadata={
            "mmmu_pro_setting": "standard10",
            "example_id": "s1",
            "images": [_tiny_image(), _tiny_image(), _tiny_image()],
        },
    )

    request = task.format_request(instance)

    assert request.images is None
    assert request.messages[0]["content"] == "Which is larger?\nA. x\nB. y"


def test_mmmu_pro_standard_setting_keeps_images_by_default():
    task = MmmuProTask(TaskConfig(name="mmmu_pro"))
    instance = Instance(
        question="Which is larger?",
        metadata={
            "mmmu_pro_setting": "standard10",
            "example_id": "s1",
            "images": [_tiny_image(), _tiny_image()],
        },
    )

    request = task.format_request(instance)

    assert request.images is not None
    assert len(request.images) == 2


# ---------------------------------------------------------------------------
# Variant registration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spec,expected",
    [
        ("mmmu", "real"),
        ("mmmu:text_only", "none"),
        ("mmmu:oracle_caption", "caption"),
        ("charxiv_descriptive:text_only", "none"),
        ("charxiv_descriptive:oracle_caption", "caption"),
        ("charxiv_reasoning:text_only", "none"),
        ("charxiv_reasoning:oracle_caption", "caption"),
    ],
)
def test_registered_variants_resolve_to_expected_image_mode(spec: str, expected: str):
    assert get_task(spec).config.image_mode == expected


# ---------------------------------------------------------------------------
# prompt_style: molmo (default) vs neutral
# ---------------------------------------------------------------------------


def _q(text: str):
    return Instance(question=text, metadata={"image": _tiny_image(), "example_id": "e"})


def test_molmo_style_is_the_default_and_changes_nothing():
    # The style tags are mm_olmo's SFT convention and load-bearing for Molmo checkpoints;
    # the default must keep sending them.
    task = _DummyImageQATask(TaskConfig(name="dummy"))
    assert task.format_request(_q("vqa2: what colour?")).messages[0]["content"] == (
        "vqa2: what colour?"
    )


def test_neutral_style_strips_the_tag_and_asks_for_a_short_answer():
    # Without this a chat-tuned model answers "There are three bars shown in the chart."
    # and exact-match scoring against "3" gives zero.
    task = _DummyImageQATask(TaskConfig(name="dummy", prompt_style="neutral"))
    content = task.format_request(_q("chart_qa: how many bars?")).messages[0]["content"]
    assert content == "how many bars?\nAnswer the question using a single word or phrase."


def test_neutral_style_does_not_double_up_answer_instructions():
    # Multiple-choice questions already say how to answer; a second, conflicting
    # instruction would be worse than none. AI2D already matches published numbers.
    task = _DummyImageQATask(TaskConfig(name="dummy", prompt_style="neutral"))
    mc = "What type of leaf?\nOnly return the correct answer option.\nA. Simple\nB. Rachis"
    assert task.format_request(_q(mc)).messages[0]["content"] == mc


def test_neutral_style_removes_mmmu_image_markers():
    # MMMU marks image positions with <image N>. Images are attached via the chat template
    # instead, so the marker is a stray token no model was trained to read.
    task = _DummyImageQATask(TaskConfig(name="dummy", prompt_style="neutral"))
    content = task.format_request(_q("<image 1> What is the value?")).messages[0]["content"]
    assert "<image" not in content
    assert content.startswith("What is the value?")


def test_unknown_prompt_style_raises():
    task = _DummyImageQATask(TaskConfig(name="dummy", prompt_style="official"))
    with pytest.raises(ValueError, match="Unknown prompt_style"):
        task.format_request(_q("vqa2: q"))


@pytest.mark.parametrize(
    "spec",
    [
        "vqa2:neutral",
        "chart_qa:neutral",
        "doc_qa:neutral",
        "info_qa:neutral",
        "text_vqa:neutral",
        "ai2d:neutral",
        "real_world_qa:neutral",
        "mmmu:neutral",
    ],
)
def test_neutral_variants_are_registered(spec: str):
    assert get_task(spec).config.prompt_style == "neutral"


# ---------------------------------------------------------------------------
# prompt_style="cot" (MMMU)
# ---------------------------------------------------------------------------


def test_cot_style_replaces_the_direct_answer_instruction():
    # Leaving "Only return the correct answer option." in place would contradict the
    # request to reason.
    task = _DummyImageQATask(TaskConfig(name="dummy", prompt_style="cot"))
    mc = "What is the value?\nOnly return the correct answer option.\nA. 6\nB. 7"
    content = task.format_request(_q(mc)).messages[0]["content"]
    assert "Only return the correct answer option." not in content
    assert content.endswith(COT_MC_INSTRUCTION)
    assert "A. 6" in content


def test_cot_style_uses_the_open_instruction_without_options():
    task = _DummyImageQATask(TaskConfig(name="dummy", prompt_style="cot"))
    content = task.format_request(_q("vqa2: Compute the deflection.")).messages[0]["content"]
    assert content.endswith(COT_OPEN_INSTRUCTION)
    assert "vqa2:" not in content


def test_cot_extractor_takes_the_last_answer_marker():
    # A reasoning trace routinely writes "Answer:" mid-thought. The shared
    # clean_prediction does pred.split("Answer:")[1] -- the FIRST -- which would score the
    # model's discarded first guess.
    assert extract_cot_answer("Maybe Answer: C.\nNo wait.\n\nAnswer: B") == "B"
    assert extract_cot_answer("reasoning\nAnswer: $42.50.") == "$42.50"
    assert extract_cot_answer("B") == "B"


def test_cot_extractor_keeps_an_unanswered_trace_visible():
    # A trace truncated before it could answer must not collapse to "", which would score
    # like an empty prediction and hide the truncation.
    truncated = "To determine the per unit manufacturing overhead costs when 30,000"
    assert extract_cot_answer(truncated) == truncated
    assert extract_cot_answer("") == ""


def test_mmmu_cot_variant_is_registered():
    cfg = get_task("mmmu:cot").config
    assert cfg.prompt_style == "cot"
    assert cfg.answer_extractor is extract_cot_answer


# ---------------------------------------------------------------------------
# Only ImageQATask (and MmmuProTask, a subclass) implement the knobs
# ---------------------------------------------------------------------------


def test_mmmu_pro_honours_prompt_style():
    # mmmu_pro overrides format_request for its vision-setting guard; that override must
    # still route through ImageQATask's prompt rendering.
    task = MmmuProTask(TaskConfig(name="mmmu_pro", prompt_style="neutral"))
    instance = Instance(
        question="<image 1> What is the value?\nOnly return the correct answer option.\nA. 6",
        metadata={"mmmu_pro_setting": "standard10", "example_id": "s1", "images": [_tiny_image()]},
    )

    content = task.format_request(instance).messages[0]["content"]
    assert "<image" not in content


@pytest.mark.parametrize("knob", [{"image_mode": "none"}, {"prompt_style": "neutral"}])
def test_pointing_and_multi_image_refuse_unimplemented_knobs(knob):
    # These families build prompts by their own rules. Silently ignoring the flags would run
    # the unmodified image benchmark and report it as an ablation.
    with pytest.raises(ValueError, match="does not implement image_mode/prompt_style"):
        reject_unsupported_prompt_knobs(TaskConfig(name="t", **knob), "PointingTask")


def test_guard_allows_the_defaults():
    reject_unsupported_prompt_knobs(TaskConfig(name="t"), "PointingTask")


# ---------------------------------------------------------------------------
# Token budget and serialization
# ---------------------------------------------------------------------------


def test_mmmu_cot_sets_a_usable_token_budget():
    # The base task caps at 12 tokens, which truncates before the required "Answer:" line,
    # so the variant could not work at its own default.
    assert get_task("mmmu:cot").config.sampling_params.max_tokens == 2048


def test_to_dict_records_the_prompt_and_ablation_fields():
    # Without these an ablation run serializes (and hashes) identically to the real run.
    cfg = TaskConfig(
        name="t", image_mode="caption", caption_source="/tmp/c.jsonl", prompt_style="neutral"
    )
    d = cfg.to_dict()
    assert d["image_mode"] == "caption"
    assert d["caption_source"] == "/tmp/c.jsonl"
    assert d["prompt_style"] == "neutral"


def test_to_dict_omits_the_knobs_at_their_defaults():
    # Emitted only when changed, so the published benchmarks keep the task hash they had
    # before these fields existed.
    d = TaskConfig(name="t").to_dict()
    for key in ("prompt_style", "image_mode", "caption_source"):
        assert key not in d


# ---------------------------------------------------------------------------
# Families that build prompts by their own rules refuse the knobs end to end
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spec", ["muir_bench", "pixmo_points_eval", "dense_caption"])
@pytest.mark.parametrize("knob", [{"image_mode": "none"}, {"prompt_style": "neutral"}])
def test_non_image_qa_families_refuse_the_knobs(spec: str, knob: dict):
    task = get_task(spec, knob)
    instance = Instance(question="q", metadata={"image": _tiny_image(), "example_id": "e"})

    with pytest.raises(ValueError, match="does not implement image_mode/prompt_style"):
        task.format_request(instance)
