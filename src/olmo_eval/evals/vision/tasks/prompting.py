"""Prompt-convention and image-ablation knobs for the vision tasks.

``TaskConfig.prompt_style`` picks how a question is rendered (mm_olmo's SFT convention,
the neutral convention published VLM numbers use, or chain-of-thought), and
``TaskConfig.image_mode`` picks what stands in for the image (the real one, nothing, or an
oracle caption). Both are read by :class:`ImageQATask.format_request`; every other vision
family refuses them via :func:`reject_unsupported_prompt_knobs`.
"""

from __future__ import annotations

import re

from olmo_eval.common.types import Instance

PROMPT_STYLES = ("molmo", "neutral", "cot")

#: mm_olmo SFT style tags, prefixed to the question at instance-build time by the
#: individual tasks (``vqa2.py``, ``chart_qa.py``, ...). In-distribution for Molmo
#: checkpoints; meaningless to any other model.
_STYLE_TAG_RE = re.compile(r"^(?:vqa2|chart_qa|doc_qa|info_qa|text_vqa|ai2_diagram\w*): ")

#: The short-answer instruction published VLM numbers are measured under. Without it a
#: chat-tuned model answers "There are three bars shown in the chart." and exact-match
#: scoring against "3" gives zero -- which is what sank VQAv2/ChartQA/TextVQA here.
SHORT_ANSWER_INSTRUCTION = "\nAnswer the question using a single word or phrase."

#: MMMU embeds ``<image 1>``, ``<image 2>`` ... in the question to mark where each image
#: belongs. We attach images via the chat template instead, so under the neutral style the
#: markers are stray tokens that no model was trained to read. (``mmmu_pro`` already
#: resolves them via ``replace_images_tokens``; ``mmmu`` never did.)
_IMAGE_MARKER_RE = re.compile(r"\s*<image\s+\d+>\s*")

#: Chain-of-thought instructions. MMMU is a reasoning benchmark whose published numbers are
#: produced with CoT; "Only return the correct answer option." forbids it, which caps the
#: score at the model's direct-answer ability. Measured on Qwen3-VL-4B-Instruct: 22% of
#: outputs began reasoning anyway and truncated, and among those that did answer cleanly
#: accuracy was only 56.94% -- so eliminating truncation alone reaches ~57 against a
#: reported 67.4. Format follows MMMU-Pro's official prompt, which asks for a final
#: ``Answer:`` line precisely so a trace can be parsed.
COT_MC_INSTRUCTION = (
    "Answer the preceding multiple-choice question. Think step by step, then end your "
    "response with a line of exactly this form: 'Answer: $LETTER' (without quotes) where "
    "$LETTER is one of the option letters."
)
COT_OPEN_INSTRUCTION = (
    "Answer the preceding question. Think step by step, then end your response with a line "
    "of exactly this form: 'Answer: $ANSWER' (without quotes) where $ANSWER is your final "
    "answer, as briefly as possible."
)

#: Markers left over from a multiple-choice template; if the question already tells the
#: model how to answer, do not append a second, conflicting instruction.
_HAS_ANSWER_INSTRUCTION = (
    "Only return the correct answer option.",
    "Please answer directly with only the letter",
    "Answer with the option letter",
)


def render_question(config, instance: Instance) -> str:
    """Apply ``config.prompt_style`` to an instance's question.

    ``molmo`` returns it unchanged. ``neutral`` strips the mm_olmo style tag and, for
    short-answer tasks, appends an explicit brevity instruction. Multiple-choice questions
    already carry their own answer instruction and are left alone apart from the tag.
    """
    style = getattr(config, "prompt_style", "molmo") or "molmo"
    if style not in PROMPT_STYLES:
        raise ValueError(f"Unknown prompt_style {style!r}; expected one of {PROMPT_STYLES}")
    question = instance.question
    if style == "molmo":
        return question

    question = _STYLE_TAG_RE.sub("", question)
    question = _IMAGE_MARKER_RE.sub(" ", question).strip()
    already_instructed = any(marker in question for marker in _HAS_ANSWER_INSTRUCTION)

    if style == "cot":
        # Strip any direct-answer instruction first: leaving it in would contradict the
        # request to reason.
        for marker in _HAS_ANSWER_INSTRUCTION:
            question = question.replace(marker, "")
        question = re.sub(r"\n{2,}", "\n", question).strip()
        instruction = COT_MC_INSTRUCTION if already_instructed else COT_OPEN_INSTRUCTION
        return f"{question}\n\n{instruction}"

    if not already_instructed:
        question = question + SHORT_ANSWER_INSTRUCTION
    return question


#: Matches the final ``Answer:`` line a CoT response is asked to end with. Deliberately the
#: LAST occurrence: a reasoning trace routinely writes "Answer:" mid-thought, and the shared
#: ``clean_prediction`` takes the FIRST (``pred.split("Answer:")[1]``), which would score the
#: model's discarded first guess.
_COT_ANSWER_RE = re.compile(r"answer\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)


def extract_cot_answer(text: str) -> str:
    """Pull the final ``Answer: ...`` payload out of a chain-of-thought response.

    Falls back to the whole text when no marker is present, so a trace truncated before it
    could answer stays visible to the scorer rather than becoming an empty string.
    """
    if not text:
        return ""
    matches = _COT_ANSWER_RE.findall(text)
    if not matches:
        return text
    return matches[-1].strip().rstrip(".").strip()


IMAGE_MODES = ("real", "none", "caption")

#: Cache of loaded caption files, keyed by path. Instances are formatted one at a time, so
#: without this the JSONL would be re-read once per instance.
_CAPTION_CACHE: dict[str, dict[str, str]] = {}


def load_captions(path: str) -> dict[str, str]:
    """Load a ``{"example_id": ..., "caption": ...}`` JSONL into a dict, cached by path."""
    if path not in _CAPTION_CACHE:
        import json

        captions: dict[str, str] = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                captions[str(record["example_id"])] = record["caption"]
        if not captions:
            raise ValueError(f"Caption source {path!r} is empty")
        _CAPTION_CACHE[path] = captions
    return _CAPTION_CACHE[path]


def reject_unsupported_prompt_knobs(config, task_name: str) -> None:
    """Refuse ``image_mode`` / ``prompt_style`` on task families that do not implement them.

    Only :class:`ImageQATask` (including ``MmmuProTask``) routes these through
    ``format_request``.
    The pointing, multi-image and captioning families build prompts by their own rules, so
    silently ignoring the flags would run the unmodified benchmark and label it a text-only or
    caption ablation -- a wrong number with nothing to indicate it.
    """
    mode = getattr(config, "image_mode", "real") or "real"
    style = getattr(config, "prompt_style", "molmo") or "molmo"
    if mode != "real" or style != "molmo":
        raise ValueError(
            f"{task_name} does not implement image_mode/prompt_style (got "
            f"image_mode={mode!r}, prompt_style={style!r}). They are wired through "
            f"ImageQATask only; running here would evaluate the "
            f"unmodified benchmark and report it as an ablation."
        )


def resolve_image_mode(config, instance: Instance) -> tuple[bool, str | None]:
    """Resolve ``config.image_mode`` for one instance.

    :returns: ``(send_image, caption)``. ``caption`` is non-None only in caption mode, and
        should be prepended to the question.

    :raises ValueError: on an unknown mode, on caption mode without a ``caption_source``,
        or when a caption is missing for this instance. Missing captions must fail rather
        than fall back to text-only: the difference between "described the image" and "saw
        nothing" is the entire measurement.
    """
    mode = getattr(config, "image_mode", "real") or "real"
    if mode not in IMAGE_MODES:
        raise ValueError(f"Unknown image_mode {mode!r}; expected one of {', '.join(IMAGE_MODES)}")
    if mode == "real":
        return True, None
    if mode == "none":
        return False, None

    source = getattr(config, "caption_source", None)
    if not source:
        raise ValueError("image_mode='caption' requires caption_source to be set")
    example_id = str(instance.metadata.get("example_id", ""))
    captions = load_captions(source)
    if example_id not in captions:
        raise ValueError(
            f"No caption for example_id {example_id!r} in {source}. Regenerate the caption "
            "file for this task/split rather than running with partial coverage."
        )
    return False, captions[example_id]


def apply_caption(question: str, caption: str) -> str:
    """Prepend an oracle caption to a question, standing in for the image."""
    return f"Figure description:\n{caption}\n\n{question}"
