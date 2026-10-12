"""Answer scoring for the image-QA benchmarks, vendored from mm_olmo's VQA evaluators."""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from olmo_eval.common.scorers.base import Scorer
from olmo_eval.common.scorers.execution import ContextScorer
from olmo_eval.common.types import Instance, LMOutput
from olmo_eval.evals.vision.scoring.common import response_text
from olmo_eval.evals.vision.scoring.math_vista_offline import (
    create_test_prompt,
    extract_answer_quick,
    math_vista_score_from_extraction,
    math_vista_score_offline,
)
from olmo_eval.evals.vision.scoring.multiple_choice import mmmu_score
from olmo_eval.evals.vision.scoring.vqa_normalization import (
    anls_metric,
    clean_prediction,
    real_world_qa_score,
    relaxed_correctness,
    scifi_relaxed_correctness,
    select_mc_option,
    vqa_score,
)

if TYPE_CHECKING:
    from olmo_eval.common.execution import ScoringContext

logger = logging.getLogger(__name__)


def _answers(instance: Instance) -> list[str]:
    answers = instance.metadata.get("answers")
    if answers is None:
        answer = instance.metadata.get("answer")
        answers = [] if answer is None else [answer]
    if isinstance(answers, str):
        answers = [answers]
    return list(answers)


@dataclass(frozen=True, slots=True)
class VqaScoreScorer(Scorer):
    """Official VQA v2 accuracy against the reference answer list."""

    name: str = "vqa_score"

    def score(self, instance: Instance, output: LMOutput) -> float:
        pred = clean_prediction(response_text(output))
        return float(vqa_score(_answers(instance), pred))


@dataclass(frozen=True, slots=True)
class AnlsScorer(Scorer):
    """ANLS (DocVQA / InfographicVQA), max over reference answers."""

    #: mm_olmo's metric key, misspelled upstream; kept for result-key parity.
    name: str = "ansl"

    def score(self, instance: Instance, output: LMOutput) -> float:
        pred = clean_prediction(response_text(output))
        answers = _answers(instance)
        if not answers:
            return 0.0
        return float(max(anls_metric(ref, pred) for ref in answers))


@dataclass(frozen=True, slots=True)
class EmScorer(Scorer):
    """Case-insensitive exact match against any reference answer."""

    name: str = "em"

    def score(self, instance: Instance, output: LMOutput) -> float:
        pred = clean_prediction(response_text(output))
        return float(pred.lower() in [x.lower() for x in _answers(instance)])


@dataclass(frozen=True, slots=True)
class RelaxedCorrectnessScorer(Scorer):
    """ChartQA relaxed accuracy, max over reference answers."""

    name: str = "relaxed_correctness"

    def score(self, instance: Instance, output: LMOutput) -> float:
        pred = clean_prediction(response_text(output))
        answers = _answers(instance)
        if not answers:
            return 0.0
        return float(max(relaxed_correctness(ans, pred) for ans in answers))


@dataclass(frozen=True, slots=True)
class ScifiRelaxedScorer(Scorer):
    """Lenient ChartQA relaxed accuracy, max over reference answers."""

    name: str = "scifi_relaxed_correctness"

    def score(self, instance: Instance, output: LMOutput) -> float:
        pred = clean_prediction(response_text(output))
        answers = _answers(instance)
        if not answers:
            return 0.0
        return float(max(scifi_relaxed_correctness(ans, pred) for ans in answers))


@dataclass(frozen=True, slots=True)
class MmmuScorer(Scorer):
    """Official MMMU scoring (multiple-choice parsing or open matching)."""

    name: str = "mmmu_score"

    def score(self, instance: Instance, output: LMOutput) -> float:
        pred = clean_prediction(response_text(output))
        meta = instance.metadata
        return mmmu_score(
            _answers(instance),
            pred,
            question_type=meta["question_type"],
            options=meta.get("options") or [],
            stable_id=str(meta.get("example_id", "")),
        )


@dataclass(frozen=True, slots=True)
class RealWorldQaScorer(Scorer):
    """RealWorldQA: A–D letter match for MC, normalized EM otherwise."""

    name: str = "real_world_qa_score"

    def score(self, instance: Instance, output: LMOutput) -> float:
        pred = clean_prediction(response_text(output))
        meta = instance.metadata
        return float(real_world_qa_score(meta["answer"], pred, meta["question_type"]))


def _mark_extraction_failed(output: LMOutput, exc: Exception) -> None:
    """Record that a zero came from a failed extraction, not a wrong answer."""
    if output.metadata is None:
        output.metadata = {}
    output.metadata["extraction_failed"] = f"{type(exc).__name__}: {exc}"


@dataclass(frozen=True, slots=True)
class MathVistaOfflineScorer(Scorer):
    """MathVista scoring with offline (no-GPT) answer extraction."""

    name: str = "score"

    def score(self, instance: Instance, output: LMOutput) -> float:
        pred = response_text(output).strip()
        meta = instance.metadata
        try:
            correct = math_vista_score_offline(
                pred,
                question_type=meta["question_type"],
                answer_type=meta["answer_type"],
                choices=list(meta.get("choices") or []),
                precision=meta.get("precision"),
                target=meta["answer"],
            )
        except (ValueError, TypeError, IndexError, KeyError) as exc:
            # An unparseable prediction is a genuine zero; record that it was a
            # failed extraction rather than a wrong answer. Anything else (a bug
            # in the vendored normalizer) propagates so the runner records a
            # scoring error instead of reporting a run of zeros as Success.
            logger.warning(
                "MathVista offline extraction failed for %s: %s", meta.get("example_id"), exc
            )
            _mark_extraction_failed(output, exc)
            return 0.0
        return float(correct)


@dataclass(frozen=True, slots=True)
class Ai2dScorer(Scorer):
    """AI2D multiple-choice scoring with opaque/transparent routing metadata.

    Stores ``{is_correct, abc_label, has_transparent_box}`` in
    ``output.metadata["ai2d_result"]`` so the two AI2D metrics can route each
    abc-label question to exactly one of the opaque/transparent variants.
    """

    name: str = "mc_ai2d"

    def score(self, instance: Instance, output: LMOutput) -> float:
        pred = clean_prediction(response_text(output))
        meta = instance.metadata
        options = list(meta["option_names"])
        pred_idx = select_mc_option(pred, options)
        is_correct = float(pred_idx == meta["answer_idx"])
        if output.metadata is None:
            output.metadata = {}
        output.metadata["ai2d_result"] = {
            "is_correct": is_correct,
            "abc_label": bool(meta["abc_label"]),
            "has_transparent_box": bool(meta["has_transparent_box"]),
        }
        return is_correct


_PROCESS_GPT_CACHE_DIR: list[str] = []


def _default_gpt_cache_dir() -> str:
    """Per-run GPT cache dir: env override or a fresh process-local temp dir.

    Never defaults to any pre-existing shared cache; the shared mm_olmo
    ``gpt4-cache`` must not be read or written by this scorer.
    """
    env_dir = os.environ.get("MATHVISTA_GPT_CACHE_DIR")
    if env_dir:
        return env_dir
    if not _PROCESS_GPT_CACHE_DIR:
        _PROCESS_GPT_CACHE_DIR.append(tempfile.mkdtemp(prefix="mathvista-gpt-cache-"))
    return _PROCESS_GPT_CACHE_DIR[0]


@dataclass(frozen=True)
class MathVistaGptScorer(ContextScorer):
    """MathVista scoring with the official GPT-4 answer extraction.

    Follows the official protocol: deterministic short-circuits first, then a
    ``gpt-4-0613`` extraction call (requires ``OPENAI_API_KEY``).  Responses
    are cached under ``cache_dir`` (env ``MATHVISTA_GPT_CACHE_DIR`` or a fresh
    per-process temp dir) so a user's own re-runs are cheap; existing shared
    caches are never touched.
    """

    name: str = "score"

    model: str = "gpt-4-0613"
    cache_dir: str | None = field(default_factory=_default_gpt_cache_dir)
    cache_only: bool = False
    recompute: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Only the output-affecting settings; the cache location and mode are
        machine-local and must not enter the task hash."""
        return {"type": self.__class__.__name__, "name": self.name, "model": self.model}

    async def ascore_with_context(
        self,
        instance: Instance,
        output: LMOutput,
        context: ScoringContext,
    ) -> float:
        from olmo_eval.evals.vision.scoring.judges import _cached_gpt_call

        pred = response_text(output).strip()
        meta = instance.metadata
        choices = list(meta.get("choices") or [])
        question_type = meta["question_type"]
        answer_type = meta["answer_type"]

        extraction = extract_answer_quick(pred, question_type, answer_type, choices)
        if extraction is None:
            try:
                extraction = await _cached_gpt_call(
                    create_test_prompt(meta["query"], pred),
                    model=self.model,
                    cache_dir=self.cache_dir or _default_gpt_cache_dir(),
                    cache_only=self.cache_only,
                    recompute=self.recompute,
                )
            except (ValueError, TypeError, KeyError) as exc:
                # A malformed reply is a failed extraction; an auth error, rate
                # limit or outage propagates to the runner's scoring_errors
                # instead of silently zeroing every affected instance.
                logger.warning(
                    "MathVista GPT extraction failed for %s: %s", meta.get("example_id"), exc
                )
                _mark_extraction_failed(output, exc)
                return 0.0

        if output.metadata is None:
            output.metadata = {}
        output.metadata["math_vista_extraction"] = extraction

        try:
            correct = math_vista_score_from_extraction(
                extraction,
                question_type=question_type,
                answer_type=answer_type,
                choices=choices,
                precision=meta.get("precision"),
                target=meta["answer"],
            )
        except (ValueError, TypeError, IndexError, KeyError) as exc:
            logger.warning(
                "MathVista GPT extraction failed for %s: %s", meta.get("example_id"), exc
            )
            _mark_extraction_failed(output, exc)
            return 0.0
        return float(correct)
