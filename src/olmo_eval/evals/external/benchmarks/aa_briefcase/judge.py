"""Binary rubric grading of AA-Briefcase-Lite submissions.

Each binary check is graded on its own judge call with the dataset's published
prompts: the system prompt, and a user message carrying the task brief, the
check's pass and fail criteria and the parsed deliverables the check names. The
deliverables go in where the user template marks them, as text, page images and
parser notes. The judge replies with a JSON verdict. A check whose deliverables
were never submitted fails without a judge call.

A judge call that keeps failing, or a reply that never parses, leaves the check
ungraded rather than failed.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from olmo_eval.common.scorers.llm_judge import (
    ConfiguredJudgeFn,
    build_openai_judge_fn,
    judge_with_retries,
)

from .loader import BriefcaseScenario, RubricCheck

logger = logging.getLogger(__name__)

#: GPT-5.5 at high reasoning effort is one of the three judges Artificial Analysis
#: samples from for AA-Briefcase rubric checks.
DEFAULT_JUDGE_MODEL = "gpt-5.5-2026-04-23"
DEFAULT_JUDGE_REASONING_EFFORT = "high"
DEFAULT_JUDGE_MAX_TOKENS = 32768
#: Judge calls per check before an unparseable reply is given up on.
JUDGE_ATTEMPTS = 3
#: Retries of a failing judge call, with waits doubling from ``JUDGE_RETRY_DELAY``
#: seconds, so a stretch of rate limiting is waited out rather than left ungraded.
JUDGE_RETRIES = 6
JUDGE_RETRY_DELAY = 2.0

SUBMISSION_MARKER = "<<<SUBMISSION CONTENT MESSAGES>>>"

VERDICT_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "rubric_verdict",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string"},
                "passed": {"type": "boolean"},
            },
            "required": ["reasoning", "passed"],
            "additionalProperties": False,
        },
    },
}


@dataclass(frozen=True)
class CheckVerdict:
    """The outcome of one rubric check.

    ``passed`` is None when the check could not be graded; ``error`` then says why.
    """

    check_id: str
    check_type: str
    passed: bool | None
    reasoning: str = ""
    raw_reply: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "check_type": self.check_type,
            "passed": self.passed,
            "reasoning": self.reasoning,
            "raw_reply": self.raw_reply,
            "error": self.error,
        }


@dataclass(frozen=True)
class ParsedSubmission:
    """A trial's deliverables as the judge sees them.

    Attributes:
        files: Blocks per submitted file name, as ``parse_submission.py`` writes them.
        image_dir: Directory the image blocks' paths are relative to.
    """

    files: dict[str, list[dict[str, Any]]]
    image_dir: Path

    @classmethod
    def load(cls, parsed_dir: Path) -> ParsedSubmission:
        manifest = json.loads((parsed_dir / "manifest.json").read_text())
        return cls(files=manifest["files"], image_dir=parsed_dir)

    def content_parts(self, file_name: str) -> list[dict[str, Any]]:
        """Chat content parts for one deliverable, headed by its name."""
        parts: list[dict[str, Any]] = [{"type": "text", "text": f"File: {file_name}"}]
        blocks = self.files.get(file_name)
        if blocks is None:
            return [*parts, _note(f"{file_name} was not submitted.")]
        for block in blocks:
            if block["type"] == "image":
                path = self.image_dir / block["path"]
                media_type = "image/png" if path.suffix == ".png" else "image/jpeg"
                data = base64.b64encode(path.read_bytes()).decode()
                url = f"data:{media_type};base64,{data}"
                parts.append({"type": "image_url", "image_url": {"url": url}})
            elif block["type"] == "note":
                parts.append(_note(block["text"]))
            else:
                parts.append({"type": "text", "text": block["text"]})
        return parts


def _note(text: str) -> dict[str, Any]:
    return {"type": "text", "text": f"[Parser note: {text}]"}


def build_judge_fn(model: str, reasoning_effort: str | None, max_tokens: int) -> ConfiguredJudgeFn:
    return build_openai_judge_fn(
        model=model,
        scorer_name="AA-Briefcase-Lite judge",
        max_tokens=max_tokens,
        temperature=0.0,
        reasoning_effort=reasoning_effort,
        response_format=VERDICT_FORMAT,
    )


def build_judge_content(
    scenario: BriefcaseScenario,
    brief: str,
    check: RubricCheck,
    submission: ParsedSubmission,
) -> list[dict[str, Any]]:
    """The user message for one check, with the deliverables where the template marks them."""
    user = (
        scenario.judge_user_prompt.replace("{task_markdown}", brief)
        .replace("{check_description}", check.description)
        .replace("{score_1_criteria}", check.pass_criteria)
        .replace("{score_0_criteria}", check.fail_criteria)
    )
    before, _, after = user.partition(SUBMISSION_MARKER)
    parts: list[dict[str, Any]] = [{"type": "text", "text": before}]
    for file_name in check.files:
        parts.extend(submission.content_parts(file_name))
    parts.append({"type": "text", "text": after})
    return parts


def parse_verdict(reply: str) -> tuple[bool, str] | None:
    """Read ``(passed, reasoning)`` from a judge reply, or None if it holds no verdict."""
    try:
        verdict = json.loads(reply)
    except (TypeError, ValueError):
        return None
    if not isinstance(verdict, dict) or not isinstance(verdict.get("passed"), bool):
        return None
    return verdict["passed"], str(verdict.get("reasoning", ""))


async def grade_check(
    judge_fn: ConfiguredJudgeFn,
    scenario: BriefcaseScenario,
    brief: str,
    check: RubricCheck,
    submission: ParsedSubmission | None,
    limiter: asyncio.Semaphore,
) -> CheckVerdict:
    """Grade one check, failing it without a call when none of its files were submitted."""
    if submission is None or not any(name in submission.files for name in check.files):
        return CheckVerdict(
            check_id=check.check_id,
            check_type=check.check_type,
            passed=False,
            reasoning=f"Not submitted: {', '.join(check.files)}",
        )
    content = build_judge_content(scenario, brief, check, submission)
    try:
        verdict, raw = await judge_with_retries(
            judge_fn,
            content,
            parse_verdict,
            limiter=limiter,
            system_prompt=scenario.judge_system_prompt,
            attempts=JUDGE_ATTEMPTS,
            max_retries=JUDGE_RETRIES,
            retry_delay=JUDGE_RETRY_DELAY,
            context=f"aa_briefcase_lite {check.check_id}",
        )
    except Exception as e:
        logger.warning(f"Judge call for {check.check_id} failed: {e}")
        return CheckVerdict(
            check_id=check.check_id,
            check_type=check.check_type,
            passed=None,
            error=f"judge_error: {e}",
        )
    if verdict is None:
        return CheckVerdict(
            check_id=check.check_id,
            check_type=check.check_type,
            passed=None,
            raw_reply=raw,
            error=f"judge_parse_error: no verdict after {JUDGE_ATTEMPTS} attempts",
        )
    passed, reasoning = verdict
    return CheckVerdict(
        check_id=check.check_id,
        check_type=check.check_type,
        passed=passed,
        reasoning=reasoning,
        raw_reply=raw,
    )
