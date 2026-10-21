"""Helpers shared by the vision scorers."""

from __future__ import annotations

from olmo_eval.common.types import LMOutput


def response_text(output: LMOutput) -> str:
    """The model text a scorer should parse.

    Prefers a non-empty ``extracted_answer`` over the raw text, and returns the
    string unchanged otherwise: the vendored parsers do their own normalization,
    so stripping here would silently diverge from the reference implementations.
    Callers that need whitespace trimmed do it explicitly.
    """
    answer = output.extracted_answer
    if isinstance(answer, str) and answer:
        return answer
    return output.text or ""
