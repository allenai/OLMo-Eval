"""PixelRAG reader benchmark: Wikipedia QA answered from retrieved screenshots.

Paper: https://arxiv.org/abs/2606.28344; data: https://huggingface.co/datasets/StarTrail-org/pixelrag-bench.

Each question comes with the Wikipedia screenshot tiles the PixelRAG retriever returned for it
(slices of rendered pages, 875 px wide and at most 1024 px tall), so the model is evaluated as
the reader of a visual RAG pipeline: it has to find and read the answer in the tiles, which
usually include distractors and sometimes miss the answer. One task per source benchmark:

* ``pixelrag_simpleqa`` — SimpleQA questions whose sources include a Wikipedia page;
* ``pixelrag_nq`` — Natural Questions validation;
* ``pixelrag_nq_tables`` — NQ-Tables dev, answered from tables;
* ``pixelrag_mmsearch`` — MMSearch end2end, most questions with a query image;
* ``pixelrag_evqa`` — Encyclopedic-VQA's automatic Google Landmarks questions, each with a query
  photo (the card says test; the ids rebuild from the val split).

The reader protocol is the paper's (``build_messages`` in the official ``eval/lib/llm.py``): the
top 3 tiles, the paper's system prompt, and a user turn of the question, the tiles, then the
subset's answer-format instruction. A question with a query image gets the query-image system
prompt and one text block (question, a note on the images, the instruction) followed by the query
image and the tiles. Images are sent as stored; a tile missing from the dataset is left out.
Decoding is greedy with the budgets of the official ``reproduce.sh``: 200 new tokens for
SimpleQA, NQ and NQ-Tables, 16,384 for MMSearch and EVQA (where the paper also turns on Qwen3.5's
thinking for MMSearch; models without a thinking mode answer directly). Grading is the official
GPT-4.1 judge (:mod:`olmo_eval.evals.vision.scoring.pixelrag`): ``accuracy`` is the fraction
graded correct, the paper's number, and NQ and NQ-Tables also report exact match.

How the turn reaches the model depends on its chat template: one without a system role (Molmo2)
gets no system prompt, and Molmo2's places every image before the text.

The questions are instruction-following QA over several images, so the tasks target
instruction-tuned checkpoints and have no stage-1 prompt form. Data is fetched from the Hub at a
pinned revision; set ``PIXELRAG_BENCH_DIR`` to a local copy of the dataset repository to read it
from disk.
"""

from __future__ import annotations

import asyncio
import functools
import io
import json
import os
import threading
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from olmo_eval.common.metrics.base import Metric
from olmo_eval.common.scorers.base import Scorer, get_scorer_result, set_scorer_result
from olmo_eval.common.types import Instance, LMRequest, RequestType, Response, SamplingParams, Split
from olmo_eval.evals.tasks.common import register
from olmo_eval.evals.vision.scoring.pixelrag import (
    SCORER_NAME,
    Grader,
    JudgeError,
    PixelRagJudgeScorer,
    Verdict,
    default_judge_cache_dir,
    is_exact_match,
    judge_verdict,
    strip_think,
)
from olmo_eval.evals.vision.tasks.base import VisionTask

if TYPE_CHECKING:
    from olmo_eval.common.execution import ScoringContext

HF_REPO = "StarTrail-org/pixelrag-bench"
HF_REVISION = "43581cfcdd39f7e1e3734fa3e5e7e13b1a41a032"

#: Tiles the paper's reader sees, best first.
READER_TOP_K = 3

#: ``SYSTEM_PROMPT_EVIDENCE_QA`` of ``eval/lib/llm.py``, verbatim.
SYSTEM_PROMPT = (
    "You are a research assistant who answers questions based on provided evidence.\n"
    "Use <think></think> tags to show your reasoning if needed.\n"
    "Answer the question directly and concisely based ONLY on the provided evidence.\n"
)
#: ``SYSTEM_PROMPT_MULTIMODAL_QUERY`` of ``eval/lib/llm.py``, verbatim.
SYSTEM_PROMPT_QUERY_IMAGE = (
    "You are a research assistant who answers questions based on retrieved visual evidence.\n"
    "You will receive: (1) a text question, (2) a query image, and (3) retrieved Wikipedia "
    "evidence images.\n"
    "Use the query image and evidence images to answer the question.\n"
    "Use <think></think> tags to show your reasoning if needed.\n"
    "Answer the question directly and concisely.\n"
)
#: The note ``build_messages`` adds when a query image comes with retrieved tiles.
QUERY_IMAGE_NOTE = (
    "The first image is the query image. The following images are retrieved Wikipedia "
    "evidence. Answer the question based on the evidence."
)

_IMAGE_PART = {"type": "image"}
_SCORER = PixelRagJudgeScorer()

# The runner can prepare several of these tasks at once, and datasets' loader is not safe to
# run concurrently on one cache; serialize the load.
_LOAD_LOCK = threading.Lock()


def _load_subset(config: str):
    import datasets

    local = os.environ.get("PIXELRAG_BENCH_DIR")
    with _LOAD_LOCK:
        if local:
            ds = datasets.load_dataset(local, config, split="test")
        else:
            ds = datasets.load_dataset(HF_REPO, config, split="test", revision=HF_REVISION)
    # No cast to undecoded images: casting a multi-GB image column concatenates it into one
    # Arrow array, past its 2 GB offset limit. Images are read from Arrow as raw bytes instead.
    return ds


def _has_bytes(images) -> list[bool]:
    """Per entry of an Arrow image-struct array, whether it holds image bytes."""
    valid = images.is_valid().to_pylist()
    with_bytes = images.field("bytes").is_valid().to_pylist()
    return [a and b for a, b in zip(valid, with_bytes, strict=True)]


def _present_tile_ranks(dataset) -> list[tuple[int, ...]]:
    """Per row, the ranks among the top ``READER_TOP_K`` tiles that have image bytes.

    Read through Arrow so that no image bytes are copied.
    """
    ranks: list[tuple[int, ...]] = []
    for chunk in dataset.data.column("retrieved_images").chunks:
        present = _has_bytes(chunk.values)
        offsets = chunk.offsets.to_pylist()
        for start, end in zip(offsets[:-1], offsets[1:], strict=True):
            ranks.append(
                tuple(
                    rank for rank in range(min(end - start, READER_TOP_K)) if present[start + rank]
                )
            )
    return ranks


def _present_query_images(dataset) -> list[bool]:
    """Per row, whether it has a query image (read through Arrow, like the tiles)."""
    present: list[bool] = []
    for chunk in dataset.data.column("query_image").chunks:
        present.extend(_has_bytes(chunk))
    return present


def _decode(record: dict[str, Any]):
    from PIL import Image

    return Image.open(io.BytesIO(record["bytes"]))


def _load_images(dataset, index: int, tile_ranks: tuple[int, ...], with_query: bool) -> list:
    """The query image (if any) then the row's tiles, decoded (module-level so it is picklable).

    The row is read from the Arrow table, so only the images sent are decoded.
    """
    row = dataset.data.slice(index, 1).select(["query_image", "retrieved_images"]).to_pylist()[0]
    images = [_decode(row["query_image"])] if with_query else []
    return images + [_decode(row["retrieved_images"][rank]) for rank in tile_ranks]


def _text(text: str) -> dict[str, str]:
    return {"type": "text", "text": text}


def reader_messages(
    question: str, instructions: str, num_tiles: int, *, query_image: bool
) -> tuple[dict[str, Any], ...]:
    """The official reader conversation, with ``{"type": "image"}`` parts where images go."""
    if query_image:
        text = f"Question: {question}\n\n{QUERY_IMAGE_NOTE}"
        if instructions:
            text += f"\n\n{instructions}"
        system = SYSTEM_PROMPT_QUERY_IMAGE
        content = [_text(text), _IMAGE_PART, *([_IMAGE_PART] * num_tiles)]
    else:
        system = SYSTEM_PROMPT
        content = [_text(question), *([_IMAGE_PART] * num_tiles)]
        if instructions:
            content.append(_text(instructions))
    return ({"role": "system", "content": system}, {"role": "user", "content": content})


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _result(response: Response) -> dict[str, Any] | None:
    return get_scorer_result(response.outputs[0], SCORER_NAME) if response.outputs else None


def _results(responses: Sequence[Response]) -> Iterator[dict[str, Any]]:
    for response in responses:
        result = _result(response)
        if result is not None:
            yield result


@dataclass(frozen=True)
class PixelRagVerdictMetric(Metric):
    """Fraction of the judged responses with one verdict; ``correct`` is the benchmark score."""

    name: str  # type: ignore[misc]
    scorer: Scorer  # type: ignore[misc]
    verdict: Verdict = "correct"

    def compute(self, responses: Sequence[Response]) -> float:
        verdicts = [r["verdict"] for r in _results(responses) if "verdict" in r]
        return sum(v == self.verdict for v in verdicts) / len(verdicts) if verdicts else 0.0

    def compute_instance(self, response: Response) -> float | None:
        result = _result(response)
        if result is None or "verdict" not in result:
            return None
        return float(result["verdict"] == self.verdict)

    def supports_pairwise_scorer_fallback(self) -> bool:
        return False

    def pairwise_higher_is_better(self) -> bool:
        return self.verdict == "correct"


@dataclass(frozen=True)
class PixelRagExactMatchMetric(Metric):
    """The official normalized exact match against any reference answer."""

    name: str  # type: ignore[misc]
    scorer: Scorer  # type: ignore[misc]

    def compute(self, responses: Sequence[Response]) -> float:
        matches = [r["exact_match"] for r in _results(responses) if "exact_match" in r]
        return sum(matches) / len(matches) if matches else 0.0

    def compute_instance(self, response: Response) -> float | None:
        result = _result(response)
        return float(result["exact_match"]) if result and "exact_match" in result else None

    def supports_pairwise_scorer_fallback(self) -> bool:
        return False


@dataclass(frozen=True)
class PixelRagJudgeErrorsMetric(Metric):
    """Responses the judge could not grade; they are left out of the verdict fractions."""

    name: str  # type: ignore[misc]
    scorer: Scorer  # type: ignore[misc]

    def compute(self, responses: Sequence[Response]) -> float:
        return float(sum("judge_error" in r for r in _results(responses)))

    def compute_instance(self, response: Response) -> float | None:
        result = _result(response)
        return float("judge_error" in result) if result is not None else None

    def supports_pairwise_scorer_fallback(self) -> bool:
        return False

    def pairwise_higher_is_better(self) -> bool:
        return False


_ACCURACY = PixelRagVerdictMetric(name="accuracy", scorer=_SCORER, verdict="correct")
_JUDGE_METRICS: tuple[Metric, ...] = (
    _ACCURACY,
    PixelRagVerdictMetric(name="incorrect", scorer=_SCORER, verdict="incorrect"),
    PixelRagVerdictMetric(name="not_attempted", scorer=_SCORER, verdict="unattempted"),
    PixelRagJudgeErrorsMetric(name="n_judge_errors", scorer=_SCORER),
)
_EM_METRIC = PixelRagExactMatchMetric(name="exact_match", scorer=_SCORER)


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


class PixelRagTask(VisionTask):
    """One subset of the PixelRAG reader benchmark."""

    dependencies = ["pillow", "openai"]
    required_secrets = ("OPENAI_API_KEY",)
    #: The official budget for the short-answer subsets; MMSearch and EVQA allow long answers.
    sampling_params = SamplingParams(temperature=0.0, max_tokens=200)
    metrics = _JUDGE_METRICS
    primary_metric = _ACCURACY
    split = Split.TEST
    #: The dataset config.
    subset: str = ""
    #: Which official judge prompt grades the subset.
    grader: Grader = "worldvqa"
    #: Also score the official exact match against ``gold_answers``.
    exact_match: bool = False

    def _build_instances(self) -> Iterator[Instance]:
        ds = _load_subset(self.subset)
        rows = ds.select_columns(["id", "question", "instructions", "answer", "original_data"])
        has_query = _present_query_images(ds)
        tile_ranks = _present_tile_ranks(ds)
        for index, row in enumerate(rows):
            source = json.loads(row["original_data"])["original_data"]
            with_query = bool(has_query[index])
            yield Instance(
                question=source["problem"],
                gold_answer=row["answer"],
                metadata={
                    "id": row["id"],
                    "example_id": row["id"],
                    "subset": self.subset,
                    "problem": source["problem"],
                    # The ground truth exactly as the official judge receives it.
                    "answer": row["answer"],
                    "gold_answers": source.get("gold_answers") or [],
                    "instructions": row["instructions"] or "",
                    "has_query_image": with_query,
                    "tile_ranks": tile_ranks[index],
                    "images": functools.partial(
                        _load_images, ds, index, tile_ranks[index], with_query
                    ),
                },
            )

    def format_request(self, instance: Instance) -> LMRequest:
        meta = instance.metadata
        return LMRequest(
            request_type=RequestType.CHAT,
            messages=reader_messages(
                meta["problem"],
                meta["instructions"],
                len(meta["tile_ranks"]),
                query_image=meta["has_query_image"],
            ),
            images=(meta["images"],),
        )

    async def score_responses(
        self,
        responses: Sequence[Response],
        context: ScoringContext | None = None,
    ) -> Sequence[Response]:
        self._extract_answers(responses)
        cache_dir = default_judge_cache_dir()
        semaphore = asyncio.Semaphore(context.scoring_concurrency if context is not None else 16)
        await asyncio.gather(*(self._grade(r, semaphore, cache_dir) for r in responses))
        return responses

    async def _grade(self, response: Response, semaphore: asyncio.Semaphore, cache_dir: str):
        if not response.outputs:
            return
        output = response.outputs[0]
        if output.metadata is None:
            output.metadata = {}
        meta = response.instance.metadata
        result: dict[str, Any] = {}
        if self.exact_match:
            result["exact_match"] = is_exact_match(strip_think(output.text), meta["gold_answers"])
        try:
            async with semaphore:
                verdict, reply = await judge_verdict(
                    self.grader,
                    question=meta["problem"],
                    ground_truth=meta["answer"],
                    response=output.text,
                    cache_dir=cache_dir,
                )
        except JudgeError as exc:
            # A response the judge could not grade is lost to infrastructure, not wrong.
            result["judge_error"] = str(exc)
            output.metadata.setdefault("scoring_errors", {})[SCORER_NAME] = {
                "phase": "score_responses",
                "type": type(exc).__qualname__,
                "message": str(exc),
                "infrastructure": "true",
            }
        else:
            result.update(verdict=verdict, judge_reply=reply)
            score = float(verdict == "correct")
            output.metadata[f"score:{SCORER_NAME}"] = score
            response.scores[SCORER_NAME] = score
        set_scorer_result(output, SCORER_NAME, result)


@register("pixelrag_simpleqa")
class PixelRagSimpleQaTask(PixelRagTask):
    """SimpleQA questions whose sources include a Wikipedia page, graded by the SimpleQA judge."""

    subset = "simpleqa"
    grader = "simpleqa"


@register("pixelrag_nq")
class PixelRagNqTask(PixelRagTask):
    """Natural Questions validation, graded by the judge as in the paper, plus exact match."""

    metrics = (*_JUDGE_METRICS, _EM_METRIC)
    subset = "nq"
    exact_match = True


@register("pixelrag_nq_tables")
class PixelRagNqTablesTask(PixelRagTask):
    """NQ-Tables dev, graded by the judge as in the paper, plus exact match."""

    metrics = (*_JUDGE_METRICS, _EM_METRIC)
    subset = "nq_tables"
    exact_match = True


@register("pixelrag_mmsearch")
class PixelRagMmSearchTask(PixelRagTask):
    """MMSearch end2end; questions with a query image send it before the tiles."""

    sampling_params = SamplingParams(temperature=0.0, max_tokens=16384)
    subset = "mmsearch"


@register("pixelrag_evqa")
class PixelRagEvqaTask(PixelRagTask):
    """Encyclopedic-VQA's Google Landmarks questions, each with its query photo."""

    sampling_params = SamplingParams(temperature=0.0, max_tokens=16384)
    subset = "encyclopedic_vqa_landmarks"
