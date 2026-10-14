"""SciMDR-Eval judge: the paper's GPT-5-mini rubric (Figure 4) and a cached async client.

The rubric text is verbatim from the SciMDR paper (arXiv 2603.12249v2, Figure 4). The paper
does not print the judge's reply format, so a JSON instruction naming the three rubric
components is appended after the rubric.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import random
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: The paper's judge (Section 3.2 / Appendix A.4), pinned to its dated snapshot.
SCIMDR_JUDGE_MODEL = "gpt-5-mini-2025-08-07"

#: Allowed points per rubric component (Figure 4).
TEXT_CITATION_POINTS = (0.0, 0.1, 0.2, 0.3)
IMAGE_CITATION_POINTS = (0.0, 0.1, 0.2, 0.3)
ANSWER_ACCURACY_POINTS = (0.0, 0.2, 0.4)
ANSWER_ACCURACY_MAX = 0.4
CITATION_MAX = 0.3

_SYSTEM_PROMPT = (
    "You are an expert evaluator for multi-modal question-answering tasks. Evaluate the "
    "model's prediction based on the following three-component rubric."
)

_RUBRIC = """Scoring Rubric (Total: 1.0 point)
Evaluate the model's response across three dimensions:
1. Text Citation Score (0.30 points)
Evaluate whether the model accurately found and cited relevant textual content:
• 0.30 points: The model accurately identified and cited all relevant text passages that fully support the answer
• 0.20 points: The model identified and cited most relevant text passages, with minor omissions
• 0.10 points: The model cited some relevant text but missed many key passages or included significant irrelevant text
• 0.0 points: The model failed to identify or cite relevant textual content, or only cited irrelevant text
2. Image Citation Score (0.30 points)
Evaluate whether the model accurately identified and referenced relevant images:
• 0.30 points: The model accurately identified and referenced all relevant images needed to answer the question
• 0.20 points: The model identified and referenced most relevant images, with minor omissions
• 0.10 points: The model referenced some relevant images but missed many or included significant irrelevant images
• 0.0 points: The model failed to identify or reference relevant images, or only referenced irrelevant images
3. Answer Accuracy Score (0.40 points)
Evaluate whether the model correctly answered the key points of the question:
• 0.40 points: The model's answer correctly addresses all key points and matches the ground truth
• 0.20 points: The model's answer partially addresses the question but misses some key points or contains minor errors
• 0.0 points: The model's answer is incorrect or fails to address the key points of the question"""

_REPLY_FORMAT = (
    "Respond with a JSON object with the keys "
    '"text_citation_score", "image_citation_score", "answer_accuracy_score" '
    "(each one of the point values listed above) and "
    '"explanation" (a brief justification).'
)

_SCORE_KEYS = {
    "text_citation_score": TEXT_CITATION_POINTS,
    "image_citation_score": IMAGE_CITATION_POINTS,
    "answer_accuracy_score": ANSWER_ACCURACY_POINTS,
}


def format_ground_truth(answer: dict[str, str]) -> str:
    """Render the annotated answer: its visual and textual key points, then the conclusion."""
    return (
        f"Visual evidence: {answer['detail_v']}\n"
        f"Textual evidence: {answer['detail_t']}\n"
        f"Conclusion: {answer['conclusion']}"
    )


def build_judge_messages(question: str, ground_truth: str, prediction: str) -> list[dict]:
    """The judge conversation for one response (Figure 4, plus the reply-format line)."""
    user = (
        f"Question\n{question}\n"
        f"Ground Truth Answer\n{ground_truth}\n"
        f"Model Prediction\n{prediction}\n"
        f"{_RUBRIC}\n\n{_REPLY_FORMAT}"
    )
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def parse_judge_reply(content: str) -> dict[str, float]:
    """The three component scores from a judge reply.

    :raises ValueError: If the reply is not JSON or a score is missing or off the rubric.
    """
    data = json.loads(content)
    scores: dict[str, float] = {}
    for key, allowed in _SCORE_KEYS.items():
        value = float(data[key])
        match = next((p for p in allowed if abs(p - value) < 1e-6), None)
        if match is None:
            raise ValueError(f"{key}={value} is not one of {allowed}")
        scores[key] = match
    return scores


_RETRY_BASE_DELAY = 1.0
_RETRY_MAX_DELAY = 30.0
_CLIENTS: dict[str, Any] = {}
_PROCESS_CACHE_DIR: list[str] = []


def default_scimdr_cache_dir() -> str:
    """Judge-reply cache dir: ``SCIMDR_JUDGE_CACHE_DIR`` or a process-local temp dir."""
    env_dir = os.environ.get("SCIMDR_JUDGE_CACHE_DIR")
    if env_dir:
        return env_dir
    if not _PROCESS_CACHE_DIR:
        _PROCESS_CACHE_DIR.append(tempfile.mkdtemp(prefix="scimdr-judge-cache-"))
    return _PROCESS_CACHE_DIR[0]


def _client():
    if "client" not in _CLIENTS:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for SciMDR-Eval grading.")
        from openai import AsyncOpenAI

        _CLIENTS["client"] = AsyncOpenAI(api_key=api_key)
    return _CLIENTS["client"]


def _is_auth_error(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status in (401, 403):
        return True
    return type(exc).__name__ in ("AuthenticationError", "PermissionDeniedError")


async def grade(
    question: str,
    ground_truth: str,
    prediction: str,
    *,
    cache_dir: str,
    max_retries: int = 8,
) -> dict[str, float] | None:
    """Grade one response; ``None`` when the judge never returns a usable reply.

    An empty response scores zero without a judge call: shown an empty prediction, the
    judge grades the ground truth as if it were the answer.
    """
    if not prediction.strip():
        return dict.fromkeys(_SCORE_KEYS, 0.0)
    messages = build_judge_messages(question, ground_truth, prediction)
    key = hashlib.sha256(f"{SCIMDR_JUDGE_MODEL}\x00{json.dumps(messages)}".encode()).hexdigest()
    cache_file = Path(cache_dir) / f"{key}.json"
    if cache_file.exists():
        return parse_judge_reply(cache_file.read_text())

    client = _client()
    content = None
    for attempt in range(1, max_retries + 1):
        try:
            # GPT-5-mini is a reasoning model: it accepts only the default temperature,
            # and the completion budget covers its reasoning tokens.
            reply = await client.chat.completions.create(
                model=SCIMDR_JUDGE_MODEL,
                messages=messages,
                response_format={"type": "json_object"},
                max_completion_tokens=8192,
            )
            content = reply.choices[0].message.content or ""
            scores = parse_judge_reply(content)
        except Exception as e:
            if _is_auth_error(e):
                raise
            logger.warning("SciMDR judge attempt %d failed: %s", attempt, e)
            delay = min(_RETRY_MAX_DELAY, _RETRY_BASE_DELAY * 2 ** (attempt - 1))
            await asyncio.sleep(delay * (0.5 + random.random()))
            continue
        os.makedirs(cache_dir, exist_ok=True)
        fd, tmp = tempfile.mkstemp(".tmp", prefix=key, dir=cache_dir)
        with os.fdopen(fd, "w") as f:
            f.write(content)
        os.replace(tmp, cache_file)
        return scores
    logger.warning(
        "SciMDR judge failed after %d attempts; last reply: %.200s", max_retries, content
    )
    return None
