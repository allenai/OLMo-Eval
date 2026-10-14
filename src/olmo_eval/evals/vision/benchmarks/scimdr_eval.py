"""SciMDR-Eval — expert-annotated multimodal QA over scientific papers (907 questions).

From "SciMDR: Advancing Scientific Multimodal Document Reasoning" (Chen et al., ACL 2026,
arXiv 2603.12249). Data: ``scimdr/SciMDR-Eval`` on the Hugging Face Hub (pinned revision):
``data/test.jsonl`` holds each question's annotated evidence (``related_text`` passages,
``related_images`` figure/table renders) and a three-part answer (visual key points,
textual key points, conclusion); ``papers/<id>.json`` holds the parsed paper.

Two tasks, one per input setting of the paper (Table 7):

* ``scimdr_eval`` — the **standard** setting, the paper's default and the one behind its
  main results: the annotated evidence plus same-paper distractors, at most 8 images and
  6 text passages.
* ``scimdr_eval_oracle`` — the **oracle** setting: only the annotated evidence passages
  and images.

The paper's full-paper setting is not implemented (whole papers do not fit these models'
image and context budgets).

Scoring follows Appendix A.4: GPT-5-mini grades each response against the question and the
annotated answer with the rubric of Figure 4 (text citation 0.3, image citation 0.3, answer
accuracy 0.4). The primary metric ``accuracy`` is the paper's strict binary score: a
response counts only with full points on all three components (rubric total 1.0). Also
reported: the paper's fine-grained rates (``text_correct``, ``visual_correct``: full citation
points; ``partial_credit``: answer points / 0.4), the rate of full answer points alone
``answer_correct``, the mean rubric total ``rubric_score``, per-question-type accuracy, and
``judge_errors``.
Responses the judge never grades count as wrong. Metrics are 0-1 (the paper reports x100).

The authors' evaluation module (a custom lmms-eval task) is not released, so these parts
are reconstructed and may differ from the published numbers:

* **Distractors** (standard setting). The paper says only that the default "simulates
  realistic retrieval by including limited noise (maximum 8 images and 6 paragraphs)".
  Here the annotated evidence is padded with distractors drawn, with a per-question seed,
  from the same paper: figure/table renders for images, and section sentences plus
  captions of the non-evidence figures/tables for text (the passages in ``related_text``
  are sentence-level units of the same parse). Evidence and distractors are shuffled
  together. Questions with more than 8 evidence images keep their first 8 (7 questions);
  questions with more than 6 evidence passages keep all of them and get no text
  distractors (9 questions).
* **Model prompt.** Passages are numbered, images are referred to as ``Image 1`` ..
  ``Image N`` in the order attached, and the model is asked to cite the supporting
  passages and images, reason, and give a final answer (the judge grades citations).
* **Judge input.** The annotated answer is rendered as its visual key points, textual key
  points and conclusion; the judge's reply format (JSON with the three component scores)
  is not given in the paper and is appended to the rubric.
* **Binary score.** The paper defines it as "correctly addresses all key points with
  accurate reasoning". ``accuracy`` reads that as full points on every rubric component;
  for Qwen3-VL-8B it gives 39.1 against the published 34.2, while full answer points alone
  (``answer_correct``) give 58.2.
* **Table images.** The release lists table renders (``<id>-Table<N>-1.png``; the evidence
  of 85 questions) but does not ship them. They are re-rendered from the arXiv PDF of the
  same paper version with the pdffigures2 page, region and DPI stored in the paper JSON,
  for the 48 papers that need one (all of those papers' tables, so tables also appear as
  distractors there), and cached under ``SCIMDR_TABLE_DIR``.
* **Broken images.** 70 release files under image names are not decodable (66 saved
  HTML error pages covering 7 papers' figures, 4 truncated PNGs). None is the evidence of
  any question; they are left out of the distractor pool.
* **Image size.** Images over 2048x32x32 pixels are downscaled (aspect ratio kept) to that
  area: lmms-eval's Qwen3-VL default ``max_pixels``, which the paper's lmms-eval runs used.
  Transparent backgrounds are flattened onto white.

Parity: there is no official evaluation code, and the distractor sampling, model prompt and
judge reply format here are reconstructions. Scores run about 5 points above the paper's
for both models checked: ``gpt-4o`` 0.298 vs 0.247 and Qwen3-VL-8B 0.391 vs 0.342 (standard
setting). Model rankings agree with the paper; absolute values are not comparable to it.
"""

from __future__ import annotations

import asyncio
import functools
import json
import os
import random
import re
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from olmo_eval.common.metrics.base import Metric
from olmo_eval.common.scorers.base import Scorer
from olmo_eval.common.types import Instance, LMOutput, Response, SamplingParams, Split
from olmo_eval.evals.tasks.common import register
from olmo_eval.evals.vision.scoring.scimdr import (
    ANSWER_ACCURACY_MAX,
    CITATION_MAX,
    SCIMDR_JUDGE_MODEL,
    default_scimdr_cache_dir,
    format_ground_truth,
    grade,
)
from olmo_eval.evals.vision.tasks.multi_image import MultiImageQATask

if TYPE_CHECKING:
    from olmo_eval.common.execution import ScoringContext

HF_REPO = "scimdr/SciMDR-Eval"
HF_REVISION = "72ae71a32224008644d8506be64ec5ea23a02ab0"

QUESTION_TYPES = ("EEQ", "CIM", "HVI", "CAC", "ARS")
MAX_IMAGES = 8
MAX_PASSAGES = 6
#: lmms-eval's Qwen3-VL default ``max_pixels``.
MAX_IMAGE_PIXELS = 2048 * 32 * 32
#: Section sentences shorter than this are headings or parse debris, not passages.
_MIN_PASSAGE_CHARS = 40
_MARKER = re.compile(r"\s*###[a-z]+(?:_[a-z0-9]+)*###")


def _data_dir() -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=HF_REPO, repo_type="dataset", revision=HF_REVISION))


def _table_dir() -> Path:
    """Where re-rendered table images (and their source PDFs) are cached."""
    env_dir = os.environ.get("SCIMDR_TABLE_DIR")
    return Path(env_dir) if env_dir else Path.home() / ".cache" / "olmo_eval" / "scimdr_tables"


def _download_pdf(paper_id: str, dest: Path, attempts: int = 5) -> None:
    import urllib.request

    url = f"https://arxiv.org/pdf/{paper_id}"
    for attempt in range(1, attempts + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "olmo-eval"})
            with urllib.request.urlopen(request, timeout=120) as response:
                data = response.read()
            if not data.startswith(b"%PDF"):
                raise ValueError(f"{url} did not return a PDF")
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(".tmp")
            tmp.write_bytes(data)
            tmp.replace(dest)
            return
        except Exception:
            if attempt == attempts:
                raise
            time.sleep(5 * attempt)


def _render_tables(paper_id: str, paper: dict) -> dict[str, Path]:
    """Re-render the paper's table images, which the release lists but does not ship.

    Each table is cropped from the arXiv PDF of the same version at the pdffigures2
    ``page`` (0-based), ``regionBoundary`` (PDF points) and ``renderDpi`` recorded in
    the paper JSON, which is how the listed ``renderURL`` images were produced.
    """
    tables = paper.get("new_table", {})
    out_dir = _table_dir()
    targets = {f"images/{t['renderURL']}": out_dir / t["renderURL"] for t in tables.values()}
    if all(path.exists() for path in targets.values()):
        return targets
    import pymupdf

    pdf_path = out_dir / "pdfs" / f"{paper_id}.pdf"
    if not pdf_path.exists():
        _download_pdf(paper_id, pdf_path)
    with pymupdf.open(pdf_path) as doc:
        for table in tables.values():
            path = targets[f"images/{table['renderURL']}"]
            if path.exists():
                continue
            box = table["regionBoundary"]
            clip = pymupdf.Rect(box["x1"], box["y1"], box["x2"], box["y2"])
            pixmap = doc[table["page"]].get_pixmap(clip=clip, dpi=table["renderDpi"])
            tmp = path.with_suffix(".tmp.png")
            pixmap.save(str(tmp))
            tmp.replace(path)
    return targets


def _load_image(path: str, max_pixels: int):
    """Open a figure render, flatten transparency onto white, cap its pixel count."""
    from PIL import Image

    # Some figure renders exceed PIL's decompression-bomb limit (up to ~145M pixels);
    # the data is a pinned release, so the guard is lifted.
    Image.MAX_IMAGE_PIXELS = None
    image = Image.open(path)
    if image.mode in ("RGBA", "LA", "P"):
        rgba = image.convert("RGBA")
        image = Image.new("RGB", rgba.size, (255, 255, 255))
        image.paste(rgba, mask=rgba.getchannel("A"))
    else:
        image = image.convert("RGB")
    width, height = image.size
    if width * height > max_pixels:
        scale = (max_pixels / (width * height)) ** 0.5
        size = (max(1, int(width * scale)), max(1, int(height * scale)))
        image = image.resize(size, Image.Resampling.BICUBIC)
    return image


def _is_readable(path: Path) -> bool:
    """Whether ``path`` is an intact image (the release has HTML pages and truncated
    files saved under image names)."""
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    try:
        with Image.open(path) as image:
            image.verify()
    except Exception:
        return False
    return True


def _normalize(text: str) -> str:
    return " ".join(text.split()).lower()


def _paper_images(paper: dict, files: dict[str, Path]) -> list[str]:
    """The paper's figure and table images that are available (``files`` maps each
    repo-relative image name to a local file)."""
    names = [entry["figure_path"] for entry in paper.get("image_paths", {}).values()]
    names += [entry["renderURL"] for entry in paper.get("new_table", {}).values()]
    return [p for p in dict.fromkeys(f"images/{n}" for n in names) if p in files]


def _paper_passages(paper: dict, exclude_images: set[str]) -> list[str]:
    """Distractor text: section sentences plus captions of non-evidence figures/tables."""
    passages = []
    for section in paper.get("sections", []):
        for line in (section.get("text") or "").split("\n"):
            line = _MARKER.sub("", line).strip()
            if len(line) >= _MIN_PASSAGE_CHARS:
                passages.append(line)
    for entry in paper.get("image_paths", {}).values():
        if f"images/{entry['figure_path']}" not in exclude_images and entry.get("caption"):
            passages.append(entry["caption"].strip())
    for entry in paper.get("new_table", {}).values():
        if f"images/{entry['renderURL']}" not in exclude_images and entry.get("caption"):
            passages.append(entry["caption"].strip())
    return passages


def build_context(
    row: dict, paper: dict, available_images: list[str], *, seed: str
) -> tuple[list[str], list[str]]:
    """The standard setting's ``(passages, image paths)`` for one question."""
    rng = random.Random(seed)
    gold_text = list(row["related_text"])
    gold_images = list(row["related_images"])[:MAX_IMAGES]

    images = list(gold_images)
    pool = [p for p in available_images if p not in set(row["related_images"])]
    rng.shuffle(pool)
    images += pool[: max(0, MAX_IMAGES - len(images))]
    rng.shuffle(images)

    passages = list(gold_text)
    if len(passages) < MAX_PASSAGES:
        seen = {_normalize(t) for t in gold_text}
        candidates = []
        for p in _paper_passages(paper, set(row["related_images"])):
            key = _normalize(p)
            if key not in seen and not any(key in g or g in key for g in seen):
                seen.add(key)
                candidates.append(p)
        rng.shuffle(candidates)
        passages += candidates[: MAX_PASSAGES - len(passages)]
    rng.shuffle(passages)
    return passages, images


def build_prompt(question: str, passages: Sequence[str], num_images: int) -> str:
    """The model prompt: numbered passages, images referenced by attachment order."""
    if num_images == 1:
        images_desc = "one image of its figures and tables, Image 1"
    else:
        images_desc = (
            f"{num_images} images of its figures and tables, given in order as "
            f"Image 1 to Image {num_images}"
        )
    passage_block = "\n".join(f"[Passage {i}] {p}" for i, p in enumerate(passages, 1))
    return (
        f"The following are excerpts from a scientific paper: {images_desc}, "
        f"and these text passages:\n\n{passage_block}\n\n"
        f"Question: {question}\n\n"
        "Answer the question using the evidence in the passages and images. Identify which "
        "passages and images support your answer, explain your reasoning, and end with "
        "your final answer."
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _result_for(response: Response) -> dict | None:
    for output in response.outputs:
        if output.metadata and "scimdr_result" in output.metadata:
            return output.metadata["scimdr_result"]
    return None


@dataclass(frozen=True)
class ScimdrJudgeScorer(Scorer):
    """Score channel for the SciMDR judge; grading runs in the task, which stores the
    binary score on each output's ``score:scimdr``."""

    name: str = "scimdr"

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.__class__.__name__, "name": self.name, "model": SCIMDR_JUDGE_MODEL}

    def score(self, instance: Instance, output: LMOutput) -> float:
        value = (output.metadata or {}).get("score:scimdr", 0.0)
        return float(value) if isinstance(value, (int, float)) else 0.0


#: Per-response value of each metric from a stored judge result (ungraded -> 0).
_FIELDS = {
    "accuracy": lambda r: float(
        r["text_citation_score"] >= CITATION_MAX
        and r["image_citation_score"] >= CITATION_MAX
        and r["answer_accuracy_score"] >= ANSWER_ACCURACY_MAX
    ),
    "answer_correct": lambda r: float(r["answer_accuracy_score"] >= ANSWER_ACCURACY_MAX),
    "partial_credit": lambda r: r["answer_accuracy_score"] / ANSWER_ACCURACY_MAX,
    "text_correct": lambda r: float(r["text_citation_score"] >= CITATION_MAX),
    "visual_correct": lambda r: float(r["image_citation_score"] >= CITATION_MAX),
    "rubric_score": lambda r: (
        r["text_citation_score"] + r["image_citation_score"] + r["answer_accuracy_score"]
    ),
}


@dataclass(frozen=True)
class ScimdrMetric(Metric):
    """Mean of one judge-derived field, optionally restricted to one question type."""

    name: str  # type: ignore[misc]
    scorer: Scorer  # type: ignore[misc]
    field: str = "accuracy"
    question_type: str | None = None

    def compute_instance(self, response: Response) -> float | None:
        if (
            self.question_type is not None
            and response.instance.metadata.get("question_type") != self.question_type
        ):
            return None
        result = _result_for(response)
        if result is None:
            return None
        if result.get("judge_error"):
            return 0.0
        return float(_FIELDS[self.field](result))

    def compute(self, responses: Sequence[Response]) -> float:
        values = [v for r in responses if (v := self.compute_instance(r)) is not None]
        return sum(values) / len(values) if values else 0.0

    def supports_pairwise_scorer_fallback(self) -> bool:
        return False


@dataclass(frozen=True)
class ScimdrJudgeErrorMetric(Metric):
    """Number of responses the judge never graded."""

    name: str  # type: ignore[misc]
    scorer: Scorer  # type: ignore[misc]

    def compute_instance(self, response: Response) -> float | None:
        result = _result_for(response)
        return None if result is None else float(bool(result.get("judge_error")))

    def compute(self, responses: Sequence[Response]) -> float:
        return float(sum(self.compute_instance(r) or 0.0 for r in responses))

    def supports_pairwise_scorer_fallback(self) -> bool:
        return False


_SCORER = ScimdrJudgeScorer()
_METRICS: tuple[Metric, ...] = (
    *(ScimdrMetric(name=field, scorer=_SCORER, field=field) for field in _FIELDS),
    *(
        ScimdrMetric(name=f"accuracy/{qt}", scorer=_SCORER, field="accuracy", question_type=qt)
        for qt in QUESTION_TYPES
    ),
    ScimdrJudgeErrorMetric(name="judge_errors", scorer=_SCORER),
)


async def _gather_bounded(coros, limit: int):
    semaphore = asyncio.Semaphore(limit)

    async def _run(coro):
        async with semaphore:
            return await coro

    return await asyncio.gather(*(_run(c) for c in coros))


@register("scimdr_eval")
class ScimdrEvalTask(MultiImageQATask):
    """SciMDR-Eval, standard setting (evidence + same-paper distractors)."""

    #: The judge calls OpenAI; ungraded responses are counted by ``judge_errors``.
    dependencies = ["pillow", "openai", "huggingface-hub", "pymupdf"]
    required_secrets = ("OPENAI_API_KEY",)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=2048)
    metrics = _METRICS
    primary_metric = _METRICS[0]  # accuracy
    split = Split.TEST
    max_images = MAX_IMAGES * 2  # the oracle setting keeps every evidence image (up to 12)

    #: Whether to pad the evidence with distractors (standard) or not (oracle).
    distractors: bool = True

    def _build_instances(self) -> Iterator[Instance]:
        root = _data_dir()
        with open(root / "data" / "test.jsonl") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        files = {f"images/{p.name}": p for p in (root / "images").iterdir() if _is_readable(p)}
        papers: dict[str, dict] = {}
        for row in rows:
            pid = row["paper_id"]
            if pid not in papers:
                papers[pid] = json.loads((root / "papers" / f"{pid}.json").read_text())
            if any(p not in files for p in row["related_images"]):
                files.update(_render_tables(pid, papers[pid]))

        for index, row in enumerate(rows):
            pid = row["paper_id"]
            paper = papers[pid]
            if self.distractors:
                passages, images = build_context(
                    row, paper, _paper_images(paper, files), seed=f"scimdr:{index}:{pid}"
                )
            else:
                passages, images = list(row["related_text"]), list(row["related_images"])
            missing = [p for p in images if p not in files]
            if missing:
                raise FileNotFoundError(f"SciMDR-Eval question {index}: missing {missing}")
            yield Instance(
                question=build_prompt(row["question"], passages, len(images)),
                gold_answer=row["answer"]["conclusion"],
                metadata={
                    "example_id": str(index),
                    "paper_id": pid,
                    "question_type": row["question_type"],
                    "raw_question": row["question"],
                    "ground_truth": format_ground_truth(row["answer"]),
                    "num_images": len(images),
                    "image_names": images,
                    "evidence_images": list(row["related_images"]),
                    "images": [
                        functools.partial(_load_image, str(files[p]), MAX_IMAGE_PIXELS)
                        for p in images
                    ],
                },
            )

    async def score_responses(
        self,
        responses: Sequence[Response],
        context: ScoringContext | None = None,
    ) -> Sequence[Response]:
        self._extract_answers(responses)
        cache_dir = default_scimdr_cache_dir()
        limit = context.scoring_concurrency if context is not None else 8
        results = await _gather_bounded(
            (
                grade(
                    r.instance.metadata["raw_question"],
                    r.instance.metadata["ground_truth"],
                    (r.outputs[0].text or "") if r.outputs else "",
                    cache_dir=cache_dir,
                )
                for r in responses
            ),
            limit,
        )
        for response, scores in zip(responses, results, strict=True):
            result: dict[str, Any] = dict(scores) if scores is not None else {"judge_error": True}
            binary = 0.0 if scores is None else _FIELDS["accuracy"](scores)
            response.scores[_SCORER.name] = binary
            for output in response.outputs[:1]:
                if output.metadata is None:
                    output.metadata = {}
                output.metadata["scimdr_result"] = result
                output.metadata["score:scimdr"] = binary
        return responses


@register("scimdr_eval_oracle")
class ScimdrEvalOracleTask(ScimdrEvalTask):
    """SciMDR-Eval, oracle setting (annotated evidence only)."""

    distractors = False
