"""GPT-judge scorer for pixmo-cap dense-caption evaluation.

Ports the scoring logic from mm_olmo/scripts/gpt_dense_caption_eval.py into
the olmo-eval-internal ContextScorer abstraction.  The cache-key scheme is
byte-identical to the legacy Gpt4WithCache so the existing gpt4-cache/ files
are reused for offline/reproducible runs.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import random
import re
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from olmo_eval.common.execution import ScoringContext
from olmo_eval.common.scorers.base import Scorer
from olmo_eval.common.scorers.execution import ContextScorer
from olmo_eval.common.types import Instance, LMOutput
from olmo_eval.evals.vision.scoring.charxiv import (
    build_dummy_output,
    verify_grading_output,
)

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = "/weka/oe-training-default/mm-olmo/dense_caption_eval/gpt4-cache"


def default_dense_caption_cache_dir() -> str:
    """The judge cache directory: ``$DENSE_CAPTION_EVAL_DIR/gpt4-cache`` when the
    reference-data root is overridden, else the shared weka cache."""
    root = os.environ.get("DENSE_CAPTION_EVAL_DIR")
    if root:
        return str(Path(root) / "gpt4-cache")
    return _DEFAULT_CACHE_DIR


# Labels that GPT returns instead of Consistent/Inconsistent; skip them silently.
_UNKNOWN_CONSISTENCY_LABELS = [
    "not specified",
    "cannot determine",
    "not determinable",
    "no verification",
    "n/a",
    "not confirmed",
    "neither",
    "not stated",
    "no judgement",
    "unable to determine",
    "inconclusive",
    "undetermined",
    "insufficient information",
    "no relevant information",
    "no conclusion",
    "not clear",
    "unknown",
    "uncertain",
    "ambiguous",
    "not addressed",
    "not enough information",
    "not mentioned",
    "not enough info",
    "no information",
    "not verifiable",
    "not applicable",
]
_UNKNOWN_PATTERN = re.compile(
    r".*\b(" + "|".join(re.escape(s) for s in _UNKNOWN_CONSISTENCY_LABELS) + r").*$",
    re.IGNORECASE,
)

# Module-level lazy async clients, keyed by model name.
_ASYNC_CLIENTS: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Cache helpers (identical semantics to legacy Gpt4WithCache)
# ---------------------------------------------------------------------------


def _compute_hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _cache_key(model: str, prompt: str) -> str:
    kwargs = {"temperature": 0}
    return _compute_hash(
        model + "::::" + json.dumps(prompt) + "::::" + json.dumps(kwargs, sort_keys=True)
    )


async def _cached_gpt_call(
    prompt: str,
    *,
    model: str,
    cache_dir: str,
    cache_only: bool,
    recompute: bool = False,
) -> str:
    """Async GPT call with file-based caching compatible with legacy gpt4-cache/.

    When ``recompute=True`` an existing cache entry is ignored and a fresh API
    call is made; the new result overwrites the old cache file.
    """
    key = _cache_key(model, prompt)
    cache_file = Path(cache_dir) / f"{key}-v1.json"

    if not recompute and cache_file.exists():
        with open(cache_file) as f:
            data = json.load(f)
        return data["choices"][0]["message"]["content"]

    if cache_only:
        raise ValueError(f"Cache miss (cache_only=True) for key {key[:16]}…")

    if model not in _ASYNC_CLIENTS:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is required for DenseCaptionJudgeScorer on a cache miss."
            )
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("openai package required: pip install openai") from None
        _ASYNC_CLIENTS[model] = AsyncOpenAI(api_key=api_key)

    client = _ASYNC_CLIENTS[model]
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    completion = response.model_dump()

    # Atomic write: tmp → rename, identical to legacy Gpt4WithCache.
    # Ensure the cache dir exists so a user-supplied (e.g. MATHVISTA_GPT_CACHE_DIR) path that
    # hasn't been created yet doesn't fail every GPT call with "No such file or directory".
    os.makedirs(cache_dir, exist_ok=True)
    fd, tmp = tempfile.mkstemp(".tmp", prefix=f"{key}-v1.json", text=True, dir=cache_dir)
    os.close(fd)
    with open(tmp, "w") as f:
        json.dump(completion, f)
    os.rename(tmp, str(cache_file))

    return completion["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# GPT prompt builders (verbatim from gpt_dense_caption_eval.py)
# ---------------------------------------------------------------------------


def _recall_prompt(mturk_statements: str, caption: str) -> str:
    return (
        "Here are statements that annotators gave for an image.\n\n"
        + mturk_statements.strip()
        + (
            "\n\nNext, consider the following caption of the image. For each statement above,"
            ' state whether the fact is "Stated" or "Not Stated" in the caption.'
            " The output should be in the form\n\n1. Stated\n2. Not Stated\n3. Stated\n\n"
            "Do not output anything other than an ordered list of Stated and Not Stated.\n\n"
            " Here is the caption: "
        )
        + (caption.strip() if caption else "No caption provided.")
    )


def _canonical_prompt(caption: str) -> str:
    return (
        "Based on the description of the image, come up with a list of the MOST canonical"
        " statements that are mentioned in it. Each statement should be broken down as much"
        " as possible. The statements should be an ordered list, where each item is separated"
        " a newline. For instance, the rseponse may look like:\n\n"
        "1. Statement A\n2. Statement B\n3. Statement C\n\n\n"
        f"\n\n\nHere is the image description: {caption}"
    )


def _consistency_prompt(num_transcripts: int, transcripts_str: str, statements_str: str) -> str:
    return (
        f"Here are {num_transcripts} captions people gave for an image using their voice.\n\n"
        + transcripts_str
        + (
            "\n\nHere are statements that a captioning model made about the image."
            ' For each statement, state whether it\'s "Consistent" or "Inconsistent"'
            " with the statements provided above. The output should be in the form\n\n"
            "1. Consistent\n2. Inconsistent\n3. Consistent\n\n"
            "Do not output anything other than an ordered list of Consistent and Inconsistent.\n\n"
        )
        + statements_str
    )


# ---------------------------------------------------------------------------
# Parse helpers (verbatim logic from gpt_dense_caption_eval.py)
# ---------------------------------------------------------------------------


def parse_recall_output(text: str) -> tuple[int, int]:
    """Parse GPT stated/not-stated output.

    Returns (num_covered, num_statements) counting only unambiguous lines.
    Mirrors eval_recall() lines 323–346 in gpt_dense_caption_eval.py.
    """
    lines = [x.strip() for x in text.split("\n") if x.strip()]
    valid_scores: list[bool] = []
    for line in lines:
        # Vendored verbatim from mm_olmo (vixmo_caption_utils.py): the negative match is
        # end-anchored, so "Not Stated." with trailing punctuation falls through to the
        # positive branch and counts as covered, and a judge response with fewer lines
        # than statements is scored over the lines it returned. Both are deliberate —
        # every published Molmo2 dense-caption number was produced this way, and
        # diverging here breaks comparability with the stored results.
        if re.fullmatch(r".*\bnot st[a-z]+$", line, flags=re.IGNORECASE):
            valid_scores.append(False)
        elif " stated" in line.lower():
            valid_scores.append(True)
        # else: ambiguous line — skip (like legacy code)
    return int(sum(valid_scores)), len(valid_scores)


def parse_consistency_output(text: str) -> tuple[int, int]:
    """Parse GPT consistent/inconsistent output.

    Returns (num_consistent, num_valid) counting only unambiguous lines.
    Mirrors eval_consistency() lines 403–461 in gpt_dense_caption_eval.py.
    """
    lines = [x.strip() for x in text.split("\n") if x.strip()]
    valid_scores: list[bool] = []
    for line in lines:
        inconsistent: bool | None = None
        if re.fullmatch(
            r".*[^a-z]((i?inconsis?ten(t|cy)?)|incorrect|inconsistence|iconsistent"
            r"|inconsisent|incomplete|contradictory).*",
            line,
            flags=re.IGNORECASE,
        ):
            inconsistent = True
        if re.fullmatch(
            r".*[^a-z](consistent(ly)?|constistent|correct).*$",
            line,
            flags=re.IGNORECASE,
        ):
            # both matched — treat as ambiguous (None); otherwise consistent (False)
            inconsistent = None if inconsistent else False
        if inconsistent is None:
            if not _UNKNOWN_PATTERN.match(line):
                logger.warning("Unexpected consistency label: %r", line)
            continue
        valid_scores.append(inconsistent)
    num_consistent = sum(not x for x in valid_scores)
    return num_consistent, len(valid_scores)


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DenseCaptionJudgeScorer(ContextScorer):
    """GPT-as-judge scorer for pixmo-cap dense-caption evaluation.

    Runs up to three GPT calls per example (recall stated-check, canonical
    statements, consistency check) and stashes all per-example results in
    ``output.metadata["dense_caption_result"]``.  The primary ``float``
    return value is the raw recall ratio (0–1) for that output, or 0.0 if
    the example is invalid.

    Cache keys are byte-identical to the legacy ``Gpt4WithCache`` in
    ``mm_olmo/scripts/gpt_dense_caption_eval.py``, so existing ``gpt4-cache/``
    entries are reused automatically.

    ``instance.metadata`` must contain:
        - ``mturk_statements`` (str): canonical_statements string from
          ``mturk-eval-statements/{sha256(url)}.json``.
        - ``transcripts`` (list[dict]): dicts with a ``"whisperTranscript"``
          key, from ``final-data.json``.
    """

    name: str = "dense_caption_judge"

    model: str = "gpt-4o-2024-05-13"
    cache_dir: str = field(default_factory=default_dense_caption_cache_dir)
    cache_only: bool = False
    recompute: bool = False
    target_metrics: tuple[str, ...] = ("recall", "consistency")

    def to_dict(self) -> dict[str, Any]:
        """Only the output-affecting settings; cache location and mode must not enter
        the task hash, or the same eval hashes differently per machine."""
        return {"type": self.__class__.__name__, "name": self.name, "model": self.model}

    async def ascore_with_context(
        self,
        instance: Instance,
        output: LMOutput,
        context: ScoringContext,
    ) -> float:
        caption = (output.extracted_answer or output.text or "").strip()
        mturk_statements: str = instance.metadata.get("mturk_statements", "")
        transcripts: list[dict] = instance.metadata.get("transcripts", [])
        transcripts_str = "\n\n".join(
            t["whisperTranscript"] for t in transcripts if "whisperTranscript" in t
        )

        result: dict = {}

        if "recall" in self.target_metrics:
            # GPT/cache failures propagate so the runner records a structured scoring
            # error; a malformed judge response parses to zero counts and is recorded
            # as an invalid example (recall_valid=False), matching mm_olmo.
            raw = await _cached_gpt_call(
                _recall_prompt(mturk_statements, caption),
                model=self.model,
                cache_dir=self.cache_dir,
                cache_only=self.cache_only,
                recompute=self.recompute,
            )
            num_covered, num_statements = parse_recall_output(raw)
            recall_valid = num_statements > 0
            if not recall_valid:
                logger.warning(
                    "Judge returned no parseable recall lines for %s",
                    instance.metadata.get("url", "?"),
                )
            result["recall"] = num_covered / num_statements if recall_valid else 0.0
            result["recall_at_10"] = (
                min(num_covered, 10) / min(num_statements, 10) if recall_valid else 0.0
            )
            result["num_statements"] = num_statements
            result["num_covered"] = num_covered
            result["recall_valid"] = recall_valid

        if "consistency" in self.target_metrics:
            statements_str = await _cached_gpt_call(
                _canonical_prompt(caption),
                model=self.model,
                cache_dir=self.cache_dir,
                cache_only=self.cache_only,
                recompute=self.recompute,
            )
            cons_raw = await _cached_gpt_call(
                _consistency_prompt(len(transcripts), transcripts_str, statements_str),
                model=self.model,
                cache_dir=self.cache_dir,
                cache_only=self.cache_only,
                recompute=self.recompute,
            )
            num_consistent, num_valid = parse_consistency_output(cons_raw)
            consistency_valid = num_valid > 0
            if not consistency_valid:
                logger.warning(
                    "Judge returned no parseable consistency lines for %s",
                    instance.metadata.get("url", "?"),
                )
            result["consistency"] = num_consistent / num_valid if consistency_valid else 0.0
            result["num_consistent"] = num_consistent
            # mm_olmo's reported `num_statements` is this consistency-side count:
            # the canonical statements GPT derived from the *model caption*
            # (ConsistencyEval.num_statements), not the recall-side mturk count.
            result["consistency_num_statements"] = num_valid
            result["consistency_valid"] = consistency_valid

        if output.metadata is None:
            output.metadata = {}
        output.metadata["dense_caption_result"] = result

        return result.get("recall", 0.0) if result.get("recall_valid", False) else 0.0


# --------------------------------------------------------------------------------------
# CharXiv GPT judge (vendored protocol; grading prompts in `vision.scoring.charxiv`)
# --------------------------------------------------------------------------------------


CHARXIV_JUDGE_MODEL = "gpt-4o-2024-05-13"
_RETRY_BASE_DELAY = 1.0
_RETRY_MAX_DELAY = 30.0

_ASYNC_CLIENTS: dict[str, Any] = {}
_PROCESS_CACHE_DIR: list[str] = []


def default_charxiv_cache_dir() -> str:
    """Judge-response cache dir: env override or a fresh process-local temp dir."""
    env_dir = os.environ.get("CHARXIV_GPT_CACHE_DIR")
    if env_dir:
        return env_dir
    if not _PROCESS_CACHE_DIR:
        _PROCESS_CACHE_DIR.append(tempfile.mkdtemp(prefix="charxiv-gpt-cache-"))
    return _PROCESS_CACHE_DIR[0]


def _charxiv_cache_key(model: str, prompt: str) -> str:
    return hashlib.sha256(f"{model}\x00{prompt}".encode()).hexdigest()


def _get_client(model: str):
    if model not in _ASYNC_CLIENTS:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for CharXiv grading on a cache miss.")
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("openai package required: pip install openai") from None
        _ASYNC_CLIENTS[model] = AsyncOpenAI(api_key=api_key)
    return _ASYNC_CLIENTS[model]


def _is_auth_error(exc: Exception) -> bool:
    """Whether the judge call failed because the credentials are unusable."""
    status = getattr(exc, "status_code", None) or getattr(exc, "http_status", None)
    if status in (401, 403):
        return True
    return type(exc).__name__ in ("AuthenticationError", "PermissionDeniedError")


def _retry_delay(attempt: int) -> float:
    """Randomized exponential backoff, in seconds, for grading attempt ``attempt``.

    Grading runs many calls at once, so a transient API condition otherwise burns
    every retry within milliseconds and turns a gradable response into a dummy
    score that downstream stats count as zero.
    """
    return min(_RETRY_MAX_DELAY, _RETRY_BASE_DELAY * 2 ** (attempt - 1)) * (0.5 + random.random())


async def _charxiv_chat_json(
    prompt: str,
    *,
    validate: Callable[[dict], None],
    dummy: dict,
    cache_dir: str,
    cache_only: bool = False,
    recompute: bool = False,
    max_retries: int = 10,
) -> dict:
    """One official grading call: cached, JSON-mode, with the official retry ladder."""
    key = _charxiv_cache_key(CHARXIV_JUDGE_MODEL, prompt)
    cache_file = Path(cache_dir) / f"{key}-v1.json"
    if not recompute and cache_file.exists():
        with open(cache_file) as f:
            content = json.load(f)
        return content
    if cache_only:
        raise ValueError(f"Cache miss (cache_only=True) for key {key[:16]}…")

    client = _get_client(CHARXIV_JUDGE_MODEL)
    curr_retries = 0
    max_tokens = 256
    content: dict | None = None
    response: str | None = None
    while curr_retries < max_retries:
        try:
            response = (
                (
                    await client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model=CHARXIV_JUDGE_MODEL,
                        response_format={"type": "json_object"},
                        n=1,
                        max_tokens=max_tokens,
                        temperature=0,
                        top_p=1,
                        seed=42,
                    )
                )
                .choices[0]
                .message.content
            )
            content = json.loads(response)
            validate(content)
            break
        except Exception as e:  # official: retry on any error; grow tokens on truncation
            if _is_auth_error(e):
                # An unusable key cannot succeed on retry, and burning the ladder
                # would turn the whole run into dummy scores.
                raise
            logger.warning("CharXiv grading error: %s", e)
            content = None
            if "Unterminated string starting at" in str(e):
                if max_tokens >= 1024:
                    logger.warning("CharXiv grading failed (truncated at max tokens)")
                    return dummy
                max_tokens = min(1024, max_tokens * 2)
            else:
                curr_retries += 1
                await asyncio.sleep(_retry_delay(curr_retries))
    if content is None:
        logger.warning(
            "CharXiv grading failed after %d retries; last reply: %.200s",
            max_retries,
            response,
        )
        return dummy

    os.makedirs(cache_dir, exist_ok=True)
    fd, tmp = tempfile.mkstemp(".tmp", prefix=f"{key}-v1.json", text=True, dir=cache_dir)
    os.close(fd)
    with open(tmp, "w") as f:
        json.dump(content, f)
    os.rename(tmp, str(cache_file))
    return content


async def grade_descriptive_batch(
    grading_query: str,
    length: int,
    *,
    cache_dir: str,
    cache_only: bool = False,
    recompute: bool = False,
) -> dict:
    """Grade one batched descriptive query (official ``get_descriptive_result_gpt``)."""

    def _validate(content: dict) -> None:
        verify_grading_output(content, length)

    return await _charxiv_chat_json(
        grading_query,
        validate=_validate,
        dummy=build_dummy_output(length),
        cache_dir=cache_dir,
        cache_only=cache_only,
        recompute=recompute,
    )


async def grade_reasoning(
    grading_query: str,
    *,
    cache_dir: str,
    cache_only: bool = False,
    recompute: bool = False,
) -> tuple[Any, int]:
    """Grade one reasoning query (official ``get_reasoning_result_gpt``)."""

    def _validate(content: dict) -> None:
        # official accesses these keys directly; a KeyError triggers a retry
        content["extracted_answer"]
        content["score"]

    content = await _charxiv_chat_json(
        grading_query,
        validate=_validate,
        dummy={"extracted_answer": "Failed to parse response", "score": -1},
        cache_dir=cache_dir,
        cache_only=cache_only,
        recompute=recompute,
    )
    return content["extracted_answer"], content["score"]


@dataclass(frozen=True)
class CharxivJudgeScorer(Scorer):
    """Score channel for the CharXiv GPT judge.

    Actual grading happens task-level (the official protocol batches descriptive triplets
    across instances), which stores per-output ``score:charxiv``; this scorer exposes that
    stored value so the metric/scorer plumbing stays uniform.
    """

    name: str = "charxiv"
    cache_dir: str = field(default_factory=default_charxiv_cache_dir)
    cache_only: bool = False
    recompute: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Only the output-affecting settings; the cache location and mode are
        machine-local and must not enter the task hash."""
        return {
            "type": self.__class__.__name__,
            "name": self.name,
            "model": CHARXIV_JUDGE_MODEL,
        }

    def score(self, instance: Instance, output: LMOutput) -> float:
        value = (output.metadata or {}).get("score:charxiv", 0.0)
        return float(value) if isinstance(value, (int, float)) else 0.0
