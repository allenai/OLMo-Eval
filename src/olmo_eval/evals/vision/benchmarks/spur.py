"""SPUR — scientific experimental image perception, understanding and reasoning (ACL 2026).

`SPUR <https://bupt-reasoning-lab.github.io/SPUR>`_ (arXiv 2604.27604) asks 4,264
multiple-choice questions about multi-panel figures from PubMed Central papers (14.3
panels per figure on average). Questions and figures come from the Hugging Face release
``BUPT-Reasoning-Lab/SPUR`` at a pinned revision; its three QA files carry the same
questions, options and answers as the GitHub ``dataset/`` copies. The questions use 958
of the release's 1,083 figures (the paper counts 1,084).

Seven tasks in three stages, read from each row's ``QUESTION_TAG``:

* Panel-level perception (``single_qa_pair.json``, 1,891): Numerical Perception
  (``np``, tag "Single-image - Numerical"), Morphological Perception (``mp``,
  "Single-image - Graphical") and Information Localization (``il``,
  "Single-image - Spatial").
* Cross-panel understanding (``multi_panel_qa_pair.json``, 1,487): Trend Analysis
  (``ta``, the Numerical / Morphological / Spatial multi-image tags) and Heterogeneous
  Integration (``hi``, the Cross-Modality relation tags).
* Expert-level reasoning (``reason_qa_pair.json``, 886): Qualitative (``qual``) and
  Quantitative (``quant``) reasoning.

The per-task counts match the paper's Table 2 (636/634/621, 1,357/130), except that the
release tags 320 qualitative and 566 quantitative questions where Table 2 lists 567 and
319. The release agrees with Table 1 (36.1% qualitative) and with the Table 3 stage
averages, which reproduce only with the release's split, so the release tags are used.

Protocol (the official ``api_test/base.py``):

* The official instruction asks for step-by-step reasoning and the answer letter inside
  ``<ANSWER> ... </ANSWER>``; the question follows as ``QUESTION + "\\n {Options}:" +
  OPTION`` together with the figure.
* Scoring takes the first ``<ANSWER>(.*?)</ANSWER>`` match, strips it and counts it
  correct only when it equals the gold letter exactly. A response without the tag is
  wrong.
* Metrics are accuracies (0-1; the paper reports x100): the seven tasks, a micro
  average per stage, and ``overall``, the micro average over all 4,264 questions
  (the paper's "Overall" column).

Parity: ``gpt-4o-2024-11-20`` at temperature 1 (the API default the paper ran) scores 0.623
with this task's layout and 0.431 with ``spur_paper_layout``, against the paper's 0.540. With
the official system-turn layout, a quarter of GPT-4o's tags hold "C. 75%"-style text, which
the exact-match rule scores as wrong (``overall_lenient`` 0.562 vs 0.633). The same model's
score thus spans the paper's number depending on layout alone; compare models only under one
layout. The official script also sends its text part under a malformed key (``"{question}:"``
instead of ``"text"``) and scores failed requests as wrong without retrying them, so what the
paper's runs received cannot be reconstructed exactly.

Deviations:

* The official script sends the instruction as a system turn. Here it leads the user
  turn, because Molmo2's chat template rejects a system role, and this keeps every model
  on the same prompt. The official instruction's unfilled ``{question} {options}
  {image}`` placeholders are kept verbatim.
* The official script places the image after the text; the providers here place it
  before the text, as they do for every image task ``spur_paper_layout`` restores the
  official layout (system-turn instruction, text then image) for API models.
* Decoding is greedy with a 2,048-token cap. The official runs used each API's default
  sampling with no cap.
* ``overall_lenient`` and ``no_answer_tag`` are diagnostics, not paper metrics: the
  former also accepts an answer letter written without the exact tag format, the latter
  is the share of responses with no ``<ANSWER>`` tag.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from olmo_eval.common.metrics.base import Metric
from olmo_eval.common.scorers.base import Scorer
from olmo_eval.common.types import (
    Instance,
    LMOutput,
    LMRequest,
    RequestType,
    Response,
    SamplingParams,
    Split,
)
from olmo_eval.evals.tasks.common import register
from olmo_eval.evals.vision.scoring.common import response_text
from olmo_eval.evals.vision.tasks.image_qa import ImageQATask

HF_REPO = "BUPT-Reasoning-Lab/SPUR"
HF_REVISION = "5e984a2002c59c27ad8014bc29558d650c8d4ec7"

# Verbatim from the official api_test/base.py (leading space dropped).
SPUR_INSTRUCTION = (
    "You are a helpful assistant specialized in analyzing academic images. Please answer this "
    "question."
    " Input: {question} {options} {image}"
    " Answer with the option’s letter from the given choices.Please solve the problem step by "
    "step."
    " Make sure to follow this output format strictly:"
    " <ANSWER> the correct answer (A or B or C or D or E) of question here] </ANSWER>"
    " <EVIDENCE> the explain of your answer here </EVIDENCE>"
)

_FILES: tuple[tuple[str, str], ...] = (
    ("single_qa_pair.json", "perception"),
    ("multi_panel_qa_pair.json", "understanding"),
    ("reason_qa_pair.json", "reasoning"),
)

STAGE_TASKS: dict[str, tuple[str, ...]] = {
    "perception": ("np", "mp", "il"),
    "understanding": ("ta", "hi"),
    "reasoning": ("qual", "quant"),
}

_ANSWER_TAG = re.compile(r"<ANSWER>(.*?)</ANSWER>", re.DOTALL)
_LOOSE_TAG = re.compile(r"<\s*answer\s*>\s*\[?\s*\(?((?-i:[A-E]))\b", re.IGNORECASE)
_LOOSE_PHRASE = re.compile(
    r"answer\W{0,3}(?:is\W{0,3})?\(?((?-i:[A-E]))\b(?![A-Za-z'])", re.IGNORECASE
)
_ANGLE_LETTER = re.compile(r"<\s*([A-E])\s*>")
_BARE_LETTER = re.compile(r"^\s*\(?\[?([A-E])\]?\)?\s*(?:[.:)]|$)")


def task_for_tag(stage: str, tag: str) -> str:
    """The paper's task abbreviation for a release ``QUESTION_TAG``."""
    t = tag.strip("[] ").lower()
    if stage == "perception":
        for key, task in (("numerical", "np"), ("graphical", "mp"), ("spatial", "il")):
            if key in t:
                return task
    elif stage == "understanding":
        return "hi" if "cross-modality" in t else "ta"
    elif stage == "reasoning":
        if "qualitative" in t:
            return "qual"
        if "quantitative" in t:
            return "quant"
    raise ValueError(f"Unrecognized SPUR {stage} tag {tag!r}")


def build_question(question: str, options: str) -> str:
    """The user turn: the official instruction, then the official question/options text."""
    return f"{SPUR_INSTRUCTION}\n{question}\n {{Options}}:{options}"


def official_user_text(question: str, options: str) -> str:
    """The official user text part, without the instruction (sent as a system turn there)."""
    return f"{question}\n {{Options}}:{options}"


def extract_official_answer(text: str) -> str | None:
    """The official extraction: the first ``<ANSWER>`` match, stripped (``None`` if absent)."""
    matches = _ANSWER_TAG.findall(text)
    return matches[0].strip() if matches else None


def extract_lenient_answer(text: str) -> str | None:
    """An answer letter from a response that may not follow the tag format exactly."""
    official = extract_official_answer(text)
    if official is not None:
        bare = _BARE_LETTER.match(official)
        if bare:
            return bare.group(1).upper()
    for pattern in (_LOOSE_TAG, _ANGLE_LETTER):
        found = pattern.search(text)
        if found:
            return found.group(1).upper()
    phrases = _LOOSE_PHRASE.findall(text)
    if phrases:
        return phrases[-1].upper()
    bare = _BARE_LETTER.match(text)
    return bare.group(1).upper() if bare else None


@dataclass(frozen=True, slots=True)
class SpurScorer(Scorer):
    """Official exact match of the ``<ANSWER>`` content against the gold letter."""

    name: str = "spur"

    def score(self, instance: Instance, output: LMOutput) -> float:
        answer = extract_official_answer(response_text(output))
        return float(answer is not None and answer == instance.metadata["answer"])


@dataclass(frozen=True, slots=True)
class SpurLenientScorer(Scorer):
    """Diagnostic: also credits a correct letter given outside the exact tag format."""

    name: str = "spur_lenient"

    def score(self, instance: Instance, output: LMOutput) -> float:
        return float(extract_lenient_answer(response_text(output)) == instance.metadata["answer"])


@dataclass(frozen=True, slots=True)
class SpurNoTagScorer(Scorer):
    """Diagnostic: 1.0 when the response has no ``<ANSWER>...</ANSWER>`` tag."""

    name: str = "spur_no_answer_tag"

    def score(self, instance: Instance, output: LMOutput) -> float:
        return float(extract_official_answer(response_text(output)) is None)


_SCORER = SpurScorer()


@dataclass(frozen=True)
class SpurAccuracyMetric(Metric):
    """Micro-averaged score over the examples whose task is in ``tasks`` (all when empty)."""

    name: str  # type: ignore[misc]
    scorer: Scorer  # type: ignore[misc]
    tasks: tuple[str, ...] = ()

    def _selected(self, response: Response) -> bool:
        return not self.tasks or response.instance.metadata.get("spur_task") in self.tasks

    def compute(self, responses: Sequence[Response]) -> float:
        name = self.scorer().name
        vals = [r.scores.get(name, 0.0) for r in responses if self._selected(r)]
        return sum(vals) / len(vals) if vals else 0.0

    def compute_instance(self, response: Response) -> float | None:
        """The example's score, or ``None`` when it is outside ``tasks``."""
        if not self._selected(response):
            return None
        value = response.scores.get(self.scorer().name)
        return float(value) if isinstance(value, (int, float)) else None

    def supports_pairwise_scorer_fallback(self) -> bool:
        return False


_OVERALL = SpurAccuracyMetric(name="overall", scorer=_SCORER)
_METRICS: tuple[Metric, ...] = (
    _OVERALL,
    *(
        SpurAccuracyMetric(name=stage, scorer=_SCORER, tasks=tasks)
        for stage, tasks in STAGE_TASKS.items()
    ),
    *(
        SpurAccuracyMetric(name=task, scorer=_SCORER, tasks=(task,))
        for tasks in STAGE_TASKS.values()
        for task in tasks
    ),
    SpurAccuracyMetric(name="overall_lenient", scorer=SpurLenientScorer()),
    SpurAccuracyMetric(name="no_answer_tag", scorer=SpurNoTagScorer()),
)


def _data_dir() -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=HF_REPO,
            repo_type="dataset",
            revision=HF_REVISION,
            allow_patterns=[*(name for name, _ in _FILES), "origin_images/*.png"],
        )
    )


@register("spur")
class SpurTask(ImageQATask):
    dependencies = ["pillow", "huggingface-hub"]
    sampling_params = SamplingParams(temperature=0.0, max_tokens=2048)
    metrics = _METRICS
    primary_metric = _OVERALL
    split = Split.TEST  # the release's only split

    def _build_instances(self) -> Iterator[Instance]:
        root = _data_dir()
        for filename, stage in _FILES:
            rows = json.loads((root / filename).read_text(encoding="utf-8"))
            for idx, row in enumerate(rows):
                # Official image lookup: QUESTION_INDEX minus its "_<n>" question suffix.
                image_path = root / "origin_images" / f"{row['QUESTION_INDEX'][:-2]}.png"
                if not image_path.is_file():
                    raise FileNotFoundError(f"SPUR image missing: {image_path}")
                yield Instance(
                    question=build_question(row["QUESTION"], row["OPTION"]),
                    gold_answer=row["ANSWER"],
                    metadata={
                        "answer": row["ANSWER"],
                        "spur_stage": stage,
                        "spur_task": task_for_tag(stage, row["QUESTION_TAG"]),
                        "question_tag": row["QUESTION_TAG"],
                        "question_index": row["QUESTION_INDEX"],
                        # QUESTION_INDEX repeats within a file, so the row position is the id.
                        "example_id": f"{stage}_{idx}",
                        "image_path": str(image_path),
                        "user_text": official_user_text(row["QUESTION"], row["OPTION"]),
                    },
                )


@register("spur_paper_layout")
class SpurPaperLayoutTask(SpurTask):
    """SPUR with the official script's message layout, for API models.

    The instruction (with its leading space) is a system turn and the user turn is the
    question/options text followed by the figure, as in ``api_test/base.py``. Only the
    LiteLLM provider honors the image position; Molmo2's chat template rejects the
    system turn, so this task is for checking parity with the paper's API runs.
    """

    def format_request(self, instance: Instance) -> LMRequest:
        return LMRequest(
            request_type=RequestType.CHAT,
            messages=(
                {"role": "system", "content": " " + SPUR_INSTRUCTION},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instance.metadata["user_text"]},
                        {"type": "image"},
                    ],
                },
            ),
            images=self._attach_images(instance),
        )
