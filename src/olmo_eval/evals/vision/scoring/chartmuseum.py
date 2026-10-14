"""ChartMuseum prompts, answer extraction and the official answer-equivalence judge.

The QA prompt, the judge prompt and the extraction regex are verbatim from the official
repository (https://github.com/Liyan06/ChartMuseum, ``prompt.py`` / ``evaluate.py``). The judge
is ``gpt-4.1-mini-2025-04-14`` at temperature 0, and a reply counts as equivalent when it
contains ``yes`` (case-insensitive), exactly as ``AnswerCompareGenerator.parse`` reads it.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from olmo_eval.common.scorers.execution import ContextScorer
from olmo_eval.common.types import Instance, LMOutput

if TYPE_CHECKING:
    from olmo_eval.common.execution import ScoringContext

logger = logging.getLogger(__name__)

CHARTMUSEUM_JUDGE_MODEL = "gpt-4.1-mini-2025-04-14"

# Several lines of the official prompts end in a space. The prompts are verbatim, so those
# spaces are spelled out (explicit "\n" joins, `_TRAILING_SPACE`) where an editor cannot strip them.
QA_PROMPT = (
    "Please answer the question using the chart image.\n"
    "\n"
    "Question: [QUESTION]\n"
    "\n"
    "Please first generate your reasoning process and then provide the user with the answer. "
    "Use the following format:\n"
    "\n"
    "<think> \n"
    "... your thinking process here ... \n"
    "</think> \n"
    "<answer> \n"
    "... your final answer (entity(s) or number) ...\n"
    "</answer>"
)

_TRAILING_SPACE = " "

COMPARE_ANSWER_PROMPT = f"""You are provided with a question and two answers. Please determine if these answers are equivalent. Follow these guidelines:

1. Numerical Comparison:
   - For decimal numbers, consider them as equivalent if their relative difference is sufficiently small.{_TRAILING_SPACE}
   For example, the following pairs are equivalent:
    - 32.35 and 32.34
    - 90.05 and 90.00
    - 83.3% and 83.2%
    - 31 and 31%
   The following pairs are not equivalent:
   - 32.35 and 35.25
   - 90.05 and 91.05
   - 83.3% and 45.2%

   Note that if the question asks for years or dates, please do the exact match with no error tolerance.

2. Unit Handling:
   - If only one answer includes units (e.g. '$', '%', '-', etc.), ignore the units and compare only the numerical values
   For example, the following pairs are equivalent:
   - 305 million and 305 million square meters
   - 0.75 and 0.75%
   - 0.6 and 60%
   - $80 and 80
   The following pairs are not equivalent:
   - 305 million and 200 million square meters
   - 0.75 and 0.90%

3. Text Comparison:
   - Ignore differences in capitalization
   - Treat mathematical expressions in different but equivalent forms as the same (e.g., "2+3" = "5")

Question: [QUESTION]
Answer 1: [ANSWER1]
Answer 2: [ANSWER2]

Please respond with:
- "Yes" if the answers are equivalent
- "No" if the answers are different"""

_ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

_MAX_RETRIES = 10
_PROCESS_CACHE_DIR: list[str] = []


def build_qa_prompt(question: str) -> str:
    """The official question-answering prompt for one question."""
    return QA_PROMPT.replace("[QUESTION]", question)


def extract_answer(text: str) -> str:
    """The short answer inside ``<answer>`` tags, or ``""`` without an opening tag.

    The official extractor appends a closing tag first, so an answer the model never closed
    still counts.
    """
    match = _ANSWER_PATTERN.search(text + "</answer>")
    return match.group(1).strip() if match else ""


def build_compare_prompt(question: str, gold: str, pred: str) -> str:
    """The official judge prompt (``Answer 1`` is the reference, ``Answer 2`` the model's)."""
    return (
        COMPARE_ANSWER_PROMPT.replace("[QUESTION]", question)
        .replace("[ANSWER1]", gold)
        .replace("[ANSWER2]", pred)
    )


def parse_judge_reply(reply: str) -> int:
    """1 when the judge's reply contains ``yes`` anywhere, else 0 (the official parse)."""
    return 1 if "yes" in reply.lower() else 0


def default_chartmuseum_cache_dir() -> str:
    """Judge-reply cache dir: ``CHARTMUSEUM_JUDGE_CACHE_DIR`` or a fresh process-local temp dir."""
    env_dir = os.environ.get("CHARTMUSEUM_JUDGE_CACHE_DIR")
    if env_dir:
        return env_dir
    if not _PROCESS_CACHE_DIR:
        _PROCESS_CACHE_DIR.append(tempfile.mkdtemp(prefix="chartmuseum-judge-cache-"))
    return _PROCESS_CACHE_DIR[0]


def _cache_file(cache_dir: str, prompt: str) -> Path:
    key = hashlib.sha256(f"{CHARTMUSEUM_JUDGE_MODEL}\x00{prompt}".encode()).hexdigest()
    return Path(cache_dir) / f"{key}-v1.json"


async def judge_equivalence(
    prompt: str,
    *,
    cache_dir: str,
    cache_only: bool = False,
    recompute: bool = False,
) -> str | None:
    """The judge's reply to one comparison prompt, or ``None`` if every attempt failed.

    Unusable credentials raise instead of retrying, so a bad key cannot turn a run into
    silent zeros.
    """
    from olmo_eval.evals.vision.scoring.judges import _get_client, _is_auth_error, _retry_delay

    cache_file = _cache_file(cache_dir, prompt)
    if not recompute and cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)["reply"]
    if cache_only:
        raise ValueError(f"Cache miss (cache_only=True) for {cache_file.name}")

    client = _get_client(CHARTMUSEUM_JUDGE_MODEL)
    reply: str | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            completion = await client.chat.completions.create(
                model=CHARTMUSEUM_JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            reply = completion.choices[0].message.content
            if reply is not None:
                break
        except Exception as e:
            if _is_auth_error(e):
                raise
            logger.warning("ChartMuseum judge error (attempt %d): %s", attempt, e)
        await asyncio.sleep(_retry_delay(attempt))
    if reply is None:
        return None

    os.makedirs(cache_dir, exist_ok=True)
    fd, tmp = tempfile.mkstemp(".tmp", prefix=cache_file.name, text=True, dir=cache_dir)
    os.close(fd)
    with open(tmp, "w") as f:
        json.dump({"model": CHARTMUSEUM_JUDGE_MODEL, "reply": reply}, f)
    os.rename(tmp, str(cache_file))
    return reply


@dataclass(frozen=True)
class ChartMuseumJudgeScorer(ContextScorer):
    """Extract the ``<answer>`` span and ask the official judge whether it matches the reference.

    Stores ``chartmuseum_result`` (``pred_answer``, ``has_answer_tag``, ``judge_reply``,
    ``equal``) on the output; ``equal`` is ``None`` when the judge never replied, which scores 0
    and is counted by ``n_invalid``.
    """

    name: str = "chartmuseum"
    cache_dir: str = field(default_factory=default_chartmuseum_cache_dir)
    cache_only: bool = False
    recompute: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Only the output-affecting settings; the cache location and mode are
        machine-local and must not enter the task hash."""
        return {
            "type": self.__class__.__name__,
            "name": self.name,
            "model": CHARTMUSEUM_JUDGE_MODEL,
        }

    async def ascore_with_context(
        self,
        instance: Instance,
        output: LMOutput,
        context: ScoringContext,
    ) -> float:
        text = output.text or ""
        pred = extract_answer(text)
        meta = instance.metadata
        reply = await judge_equivalence(
            build_compare_prompt(meta["raw_question"], str(meta["answer"]), pred),
            cache_dir=self.cache_dir,
            cache_only=self.cache_only,
            recompute=self.recompute,
        )
        equal = parse_judge_reply(reply) if reply is not None else None
        if output.metadata is None:
            output.metadata = {}
        output.metadata["chartmuseum_result"] = {
            "pred_answer": pred,
            "has_answer_tag": "<answer>" in text,
            "judge_reply": reply,
            "equal": equal,
        }
        return float(equal or 0)
