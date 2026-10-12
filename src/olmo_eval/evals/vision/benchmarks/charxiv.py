"""CharXiv (validation) — descriptive + reasoning chart-understanding benchmarks.

Two tasks mirror the official CharXiv evaluation (https://charxiv.github.io, ported verbatim in
:mod:`olmo_eval.evals.vision.scoring.charxiv` / :mod:`olmo_eval.evals.vision.scoring.judges`):

* ``charxiv_descriptive`` — 4 template questions per chart (4000 instances), graded by GPT-4o in
  the official batched form: responses grouped by template id, five response/ground-truth triplets
  per grading call, with per-template rubrics.
* ``charxiv_reasoning`` — 1 open question per chart (1000 instances), graded by GPT-4o with the
  official answer-type-specific instructions.

Only the ``validation`` split is supported (the leaderboard split; ``test`` answers are withheld).
Question texts and grading prompts are byte-identical to the official pipeline. Grading needs
``OPENAI_API_KEY``; judge replies are cached under ``CHARXIV_GPT_CACHE_DIR`` (or a temp dir).
Metrics are 0-1 (the leaderboard reports x100); the official protocol counts failed grades
(score -1) as 0 while ``n_invalid`` tracks them.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from olmo_eval.common.metrics.base import Metric
from olmo_eval.common.scorers.base import Scorer
from olmo_eval.common.types import Instance, Response, SamplingParams, Split
from olmo_eval.evals.tasks.common import register
from olmo_eval.evals.vision.data.images import lazy_hf_image
from olmo_eval.evals.vision.scoring.charxiv import (
    IDX2ANSTYPE,
    build_descriptive_grading_batches,
    build_reasoning_grading_prompt,
    build_reasoning_question,
    descriptive_query_helper,
    qnum_to_qtype,
)
from olmo_eval.evals.vision.scoring.judges import (
    CharxivJudgeScorer,
    default_charxiv_cache_dir,
    grade_descriptive_batch,
    grade_reasoning,
)
from olmo_eval.evals.vision.tasks.image_qa import ImageQATask

if TYPE_CHECKING:
    from olmo_eval.common.execution import ScoringContext

_SCORER = CharxivJudgeScorer()


# The runner prepares both charxiv tasks concurrently, and datasets' arrow reader goes through
# tqdm's thread_map whose class-level lock handling is not thread-safe (concurrent loads race on
# `del tqdm._lock`). Serialize the load; everything after it is thread-safe.
_LOAD_LOCK = threading.Lock()


def _load_charxiv_nodecode(split: str):
    import datasets

    with _LOAD_LOCK:
        ds = datasets.load_dataset("princeton-nlp/CharXiv", split=split)
    return ds.cast_column("image", datasets.Image(decode=False))


def _figure_id(figure_path: str) -> int:
    return int(figure_path.split("/")[-1].split(".")[0])


def _subplot_loc(ex: dict) -> list[int] | str:
    loc = ex["subplot_loc"]
    return loc if loc is not None else [ex["subplot_row"], ex["subplot_col"]]


def _response_text(response: Response) -> str:
    if response.outputs:
        return response.outputs[0].text or ""
    return ""


def _store_result(response: Response, extracted, raw_score, **extra) -> None:
    # official get_stats: scores outside {0, 1} are invalid and count as 0
    score = float(raw_score) if raw_score in (0, 1) else 0.0
    response.scores["charxiv"] = score
    if response.outputs:
        output = response.outputs[0]
        if output.metadata is None:
            output.metadata = {}
        output.metadata["charxiv_result"] = {
            "extracted_answer": extracted,
            "score": raw_score,
            **extra,
        }
        output.metadata["score:charxiv"] = score


def _results(responses: Sequence[Response]) -> Iterator[dict]:
    for response in responses:
        for output in response.outputs:
            if output.metadata and "charxiv_result" in output.metadata:
                yield output.metadata["charxiv_result"]


def _result_for(response: Response) -> dict | None:
    for output in response.outputs:
        if output.metadata and "charxiv_result" in output.metadata:
            return output.metadata["charxiv_result"]
    return None


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


@dataclass(frozen=True)
class CharxivScoreMetric(Metric):
    """Mean official score (invalid -1 grades count as 0, per ``get_stats.py``).

    Optional filters restrict to one leaderboard slice: ``category`` (descriptive question
    type via ``QNUM2QTYPE``) or ``inst_category`` (reasoning answer type).
    """

    name: str  # type: ignore[misc]
    scorer: Scorer  # type: ignore[misc]
    category: str | None = None
    inst_category: int | None = None

    def compute(self, responses: Sequence[Response]) -> float:
        values = []
        for result in _results(responses):
            if self.category is not None and qnum_to_qtype(result["qid"]) != self.category:
                continue
            if self.inst_category is not None and result["inst_category"] != self.inst_category:
                continue
            values.append(float(result["score"]) if result["score"] in (0, 1) else 0.0)
        return _mean(values)

    def compute_instance(self, response: Response) -> float | None:
        """The example's own graded score, or ``None`` outside this category."""
        result = _result_for(response)
        if result is None:
            return None
        if self.category is not None and qnum_to_qtype(result["qid"]) != self.category:
            return None
        return float(result["score"]) if result["score"] in (0, 1) else 0.0

    def supports_scorer_fallback(self) -> bool:
        return False


@dataclass(frozen=True)
class CharxivInvalidCountMetric(Metric):
    """Number of grading calls that failed (official dummy score -1)."""

    name: str  # type: ignore[misc]
    scorer: Scorer  # type: ignore[misc]

    def compute(self, responses: Sequence[Response]) -> float:
        return float(sum(1 for r in _results(responses) if r["score"] not in (0, 1)))

    def compute_instance(self, response: Response) -> float | None:
        """1.0 when this example's grading was invalid, 0.0 when it graded."""
        result = _result_for(response)
        if result is None:
            return None
        return 0.0 if result["score"] in (0, 1) else 1.0

    def supports_scorer_fallback(self) -> bool:
        return False


_DESC_METRICS: tuple[Metric, ...] = (
    CharxivScoreMetric(name="descriptive_overall", scorer=_SCORER),
    CharxivScoreMetric(
        name="descriptive_information_extraction", scorer=_SCORER, category="Information Extraction"
    ),
    CharxivScoreMetric(name="descriptive_enumeration", scorer=_SCORER, category="Enumeration"),
    CharxivScoreMetric(
        name="descriptive_pattern_recognition", scorer=_SCORER, category="Pattern Recognition"
    ),
    CharxivScoreMetric(name="descriptive_counting", scorer=_SCORER, category="Counting"),
    CharxivScoreMetric(
        name="descriptive_compositionality", scorer=_SCORER, category="Compositionality"
    ),
    CharxivInvalidCountMetric(name="n_invalid", scorer=_SCORER),
)

_REAS_METRICS: tuple[Metric, ...] = (
    CharxivScoreMetric(name="reasoning_overall", scorer=_SCORER),
    *(
        CharxivScoreMetric(
            name=f"reasoning_{IDX2ANSTYPE[i].lower().replace('-', '_')}",
            scorer=_SCORER,
            inst_category=i,
        )
        for i in (1, 2, 3, 4)
    ),
    CharxivInvalidCountMetric(name="n_invalid", scorer=_SCORER),
)


async def _gather_bounded(coros, context: ScoringContext | None):
    limit = context.scoring_concurrency if context is not None else 8
    semaphore = asyncio.Semaphore(limit)

    async def _run(coro):
        async with semaphore:
            return await coro

    return await asyncio.gather(*(_run(c) for c in coros))


@register("charxiv_descriptive")
class CharxivDescriptiveTask(ImageQATask):
    #: The judge calls OpenAI; a grading failure is recorded as invalid and
    #: counted by `n_invalid` rather than silently scored as a wrong answer.
    dependencies = ["pillow", "openai"]
    required_secrets = ("OPENAI_API_KEY",)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1024)
    metrics = _DESC_METRICS
    primary_metric = _DESC_METRICS[0]  # descriptive_overall
    split = Split.VALIDATION

    def _build_instances(self) -> Iterator[Instance]:
        ds = _load_charxiv_nodecode(self.config.split.value)
        for idx in range(len(ds)):
            ex = ds[idx]
            fid = _figure_id(ex["figure_path"])
            subplot_loc = _subplot_loc(ex)
            for i in range(4):
                qid = ex[f"descriptive_q{i + 1}"]
                answer = ex[f"descriptive_a{i + 1}"]
                resp_key = f"{fid}_{i}"
                yield Instance(
                    question=descriptive_query_helper(qid, subplot_loc),
                    gold_answer=answer,
                    metadata={
                        "figure_id": fid,
                        "subq_idx": i,
                        "qid": qid,
                        "answer": answer,
                        "resp_key": resp_key,
                        "example_id": resp_key,
                        "image": lazy_hf_image(ds, idx, "image"),
                    },
                )

    async def score_responses(
        self,
        responses: Sequence[Response],
        context: ScoringContext | None = None,
    ) -> Sequence[Response]:
        self._extract_answers(responses)
        # Official protocol: group all responses by template id, grade 5 triplets per GPT call.
        items = [
            (
                r.instance.metadata["resp_key"],
                _response_text(r),
                r.instance.metadata["answer"],
                r.instance.metadata["qid"],
            )
            for r in responses
        ]
        batches = build_descriptive_grading_batches(items)
        cache_dir = default_charxiv_cache_dir()
        results = await _gather_bounded(
            (
                grade_descriptive_batch(
                    batch["grading_query"], len(batch["resp_keys"]), cache_dir=cache_dir
                )
                for batch in batches
            ),
            context,
        )
        graded = {}
        for batch, result in zip(batches, results, strict=True):
            for i, resp_key in enumerate(batch["resp_keys"]):
                graded[resp_key] = (result[f"extract_answer_T{i + 1}"], result[f"score_T{i + 1}"])
        for response in responses:
            meta = response.instance.metadata
            extracted, raw_score = graded[meta["resp_key"]]
            _store_result(response, extracted, raw_score, qid=meta["qid"])
        return responses


@register("charxiv_reasoning")
class CharxivReasoningTask(ImageQATask):
    #: The judge calls OpenAI; a grading failure is recorded as invalid and
    #: counted by `n_invalid` rather than silently scored as a wrong answer.
    dependencies = ["pillow", "openai"]
    required_secrets = ("OPENAI_API_KEY",)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1024)
    metrics = _REAS_METRICS
    primary_metric = _REAS_METRICS[0]  # reasoning_overall
    split = Split.VALIDATION

    def _build_instances(self) -> Iterator[Instance]:
        ds = _load_charxiv_nodecode(self.config.split.value)
        for idx in range(len(ds)):
            ex = ds[idx]
            fid = _figure_id(ex["figure_path"])
            inst_category = ex["reasoning_a_type"]
            yield Instance(
                question=build_reasoning_question(
                    ex["reasoning_q"], inst_category, ex["reasoning_a"]
                ),
                gold_answer=ex["reasoning_a"],
                metadata={
                    "figure_id": fid,
                    "raw_question": ex["reasoning_q"],
                    "answer": ex["reasoning_a"],
                    "inst_category": inst_category,
                    "qa_source": ex["reasoning_q_source"],
                    "example_id": str(fid),
                    "image": lazy_hf_image(ds, idx, "image"),
                },
            )

    async def score_responses(
        self,
        responses: Sequence[Response],
        context: ScoringContext | None = None,
    ) -> Sequence[Response]:
        self._extract_answers(responses)
        cache_dir = default_charxiv_cache_dir()
        results = await _gather_bounded(
            (
                grade_reasoning(
                    build_reasoning_grading_prompt(
                        r.instance.metadata["raw_question"],
                        r.instance.metadata["inst_category"],
                        r.instance.metadata["answer"],
                        _response_text(r),
                    ),
                    cache_dir=cache_dir,
                )
                for r in responses
            ),
            context,
        )
        for response, (extracted, raw_score) in zip(responses, results, strict=True):
            _store_result(
                response,
                extracted,
                raw_score,
                inst_category=response.instance.metadata["inst_category"],
            )
        return responses
