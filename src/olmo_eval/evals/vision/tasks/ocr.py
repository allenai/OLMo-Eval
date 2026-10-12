"""Shared base for the document-OCR benchmarks (olmOCR-bench, CC-OCR, OmniDocBench).

These benchmarks hand the model one page image and a transcription instruction, and grade
the markdown (or JSON) it writes back. :class:`OcrTask` adds the two things they share on
top of :class:`ImageQATask`:

* the prompt follows the checkpoint's prompt family, like the pointing and captioning tasks
  (:func:`ocr_question`): a stage-1 checkpoint gets the task's OCR style tag alone, an
  instruction-tuned one the benchmark's instruction;
* scoring runs off the event loop, since grading a page is CPU work that can take seconds.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import TYPE_CHECKING

from olmo_eval.common.types import Response
from olmo_eval.evals.vision.scoring.prompts import apply_style_prefix
from olmo_eval.evals.vision.tasks.image_qa import ImageQATask

if TYPE_CHECKING:
    from olmo_eval.common.execution import ScoringContext

#: OLMo-core stage-1 OCR styles (OLMo-core #875): the style tag is the whole user turn.
#: ``olmocr`` is olmOCR's page transcription (reading order, HTML tables, LaTeX math);
#: ``textocr`` is TextOCR's scene text (every piece of text in the image, joined by spaces).
OLMOCR_STYLE = "olmocr"
TEXTOCR_STYLE = "textocr"


def ocr_question(
    instruction: str, *, style: str, prompt_templates: str, system_prompt_style: str
) -> str:
    """The user turn for an OCR benchmark, following the checkpoint's prompt family.

    * Instruction-tuned checkpoints (``uber_model_v2`` + ``demo_or_style_v*``, the defaults)
      get the benchmark's instruction verbatim.
    * Stage-1 checkpoints (``-o prompt_templates=none -o system_prompt_style=
      style_and_length_v2``) get the task's OCR style tag alone, e.g. ``"olmocr:"``: under
      ``prompt_templates="none"`` the question is empty, as it is for a caption, and the
      ``style_and_length*`` family prefixes the style. That is exactly OLMo-core's stage-1 OCR
      training prompt, so train and test share one form.

    :param instruction: The benchmark's own instruction.
    :param style: The OCR style the task's answer form was trained under (:data:`OLMOCR_STYLE`
        or :data:`TEXTOCR_STYLE`).
    """
    question = "" if prompt_templates == "none" else instruction
    return apply_style_prefix(question, system_prompt_style, style)


class OcrTask(ImageQATask):
    """Base class for single-page OCR / document-parsing benchmarks."""

    #: Prompt family assumed when the run does not say; matches the instruction-tuned
    #: checkpoints, mirroring the pointing and captioning tasks.
    default_prompt_templates = "uber_model_v2"
    default_system_prompt_style = "demo_or_style_v2"
    #: The stage-1 OCR style whose answer form this benchmark scores (see :func:`ocr_question`).
    ocr_style: str = OLMOCR_STYLE

    def _question(self, instruction: str) -> str:
        return ocr_question(
            instruction,
            style=self.ocr_style,
            prompt_templates=self.config.prompt_templates or self.default_prompt_templates,
            system_prompt_style=self.config.system_prompt_style or self.default_system_prompt_style,
        )

    async def score_responses(
        self,
        responses: Sequence[Response],
        context: ScoringContext | None = None,
    ) -> Sequence[Response]:
        self._extract_answers(responses)
        await asyncio.to_thread(self._apply_scorers, responses)
        return responses
