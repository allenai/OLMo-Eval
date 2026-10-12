"""Stage-1 prompts and English-only variants of the OCR and dense-caption tasks.

A stage-1 checkpoint (``-o prompt_templates=none -o system_prompt_style=style_and_length_v2``)
must see exactly its training prompt, the style tag alone: ``long_caption:`` for captions,
``olmocr:`` for page transcription, ``textocr:`` for scene text (OLMo-core #875). The defaults
stay the instruction-tuned prompts.
"""

from __future__ import annotations

import base64
import io
import json

import pytest

from olmo_eval.evals.tasks.common.registry import get_task
from olmo_eval.evals.vision.benchmarks import cc_ocr, omnidocbench
from olmo_eval.evals.vision.tasks.ocr import ocr_question

STAGE1 = {"prompt_templates": "none", "system_prompt_style": "style_and_length_v2"}
INSTRUCTION = "Convert this page to markdown."


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("style", ["olmocr", "textocr"])
def test_stage1_ocr_prompt_is_the_tag_alone(style: str) -> None:
    assert (
        ocr_question(
            INSTRUCTION,
            style=style,
            prompt_templates="none",
            system_prompt_style="style_and_length_v2",
        )
        == f"{style}:"
    )


def test_instruction_tuned_ocr_prompt_is_the_instruction() -> None:
    assert (
        ocr_question(
            INSTRUCTION,
            style="olmocr",
            prompt_templates="uber_model_v2",
            system_prompt_style="demo_or_style_v2",
        )
        == INSTRUCTION
    )


@pytest.mark.parametrize(
    "task, tag",
    [
        ("olmocr_bench", "olmocr:"),
        ("omnidocbench", "olmocr:"),
        ("omnidocbench_v15_en", "olmocr:"),
        ("cc_ocr_multi_scene", "textocr:"),
        ("cc_ocr_multi_scene_en", "textocr:"),
    ],
)
def test_ocr_tasks_send_their_tag_to_stage1_checkpoints(task: str, tag: str) -> None:
    assert get_task(task, STAGE1)._question(INSTRUCTION) == tag
    assert get_task(task)._question(INSTRUCTION) == INSTRUCTION


def test_captioner_sends_the_bare_caption_tag() -> None:
    assert get_task("dense_caption_captioner")._question(0) == "long_caption:"
    assert get_task("dense_caption", STAGE1)._question(0) == "long_caption:"
    # Instruction-tuned checkpoints keep the seeded natural-language instruction.
    assert "long_caption" not in get_task("dense_caption")._question(0)


# ---------------------------------------------------------------------------
# English-only variants
# ---------------------------------------------------------------------------


def _png_b64() -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _write_cc_ocr(root) -> None:
    track = root / cc_ocr.TRACK
    track.mkdir(parents=True)
    image = _png_b64()
    for subset in cc_ocr.SUBSETS:
        with open(track / f"{subset}.tsv", "w", encoding="utf-8") as f:
            f.write("index\timage_name\timage\tquestion\tanswer\tcategory\tsplit\n")
            f.write(f"0\t{subset}_0.png\t{image}\tRead it.\tHELLO\t{cc_ocr.TRACK}\t{subset}\n")


def test_cc_ocr_english_runs_the_eight_english_subsets(tmp_path, monkeypatch) -> None:
    _write_cc_ocr(tmp_path)
    monkeypatch.setenv("CC_OCR_DIR", str(tmp_path))
    assert len(cc_ocr.ENGLISH_SUBSETS) == 8
    assert not any("zh" in s for s in cc_ocr.ENGLISH_SUBSETS)

    english = get_task("cc_ocr_multi_scene_en", STAGE1)
    got = {i.metadata["dataset"] for i in english._build_instances()}
    assert got == set(cc_ocr.ENGLISH_SUBSETS)
    assert {m.name for m in english.metrics} == {"macro_f1", "micro_f1", *cc_ocr.ENGLISH_SUBSETS}

    full = get_task("cc_ocr_multi_scene")
    assert {i.metadata["dataset"] for i in full._build_instances()} == set(cc_ocr.SUBSETS)


def test_omnidocbench_english_keeps_only_english_pages(tmp_path, monkeypatch) -> None:
    pages = [
        {"page_info": {"image_path": f"{lang}.jpg", "page_attribute": {"language": lang}}}
        for lang in ("english", "simplified_chinese", "en_ch_mixed", "english")
    ]
    pages[3]["page_info"]["image_path"] = "english2.jpg"
    (tmp_path / "OmniDocBench.json").write_text(json.dumps(pages), encoding="utf-8")
    monkeypatch.setenv("OMNIDOCBENCH_V15_DIR", str(tmp_path))
    monkeypatch.setattr(omnidocbench, "ensure_evaluator", lambda version: None)
    monkeypatch.setattr(omnidocbench, "ensure_cdm_toolchain", lambda version: None)

    task = get_task("omnidocbench_v15_en", STAGE1)
    names = [i.metadata["image_name"] for i in task._build_instances()]
    assert names == ["english.jpg", "english2.jpg"]
    assert all(i.question == "olmocr:" for i in task._build_instances())

    full = get_task("omnidocbench_v15")
    assert len(list(full._build_instances())) == 4
