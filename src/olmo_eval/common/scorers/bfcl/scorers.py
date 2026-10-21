"""Scorer for the Berkeley Function Calling Leaderboard.

The scorer reads the calls a task decoded onto ``output.extracted_answer`` and
the entry's function documents and possible answers from ``instance.metadata``,
so it is indifferent to whether the calls arrived as native tool calls or as
text. Which check runs is decided per instance by its category, which lets one
task pool several categories.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from olmo_eval.common.scorers.base import Scorer
from olmo_eval.common.types import Instance, LMOutput

from .checker import ast_checker
from .constants import IRRELEVANCE_CATEGORIES, RELEVANCE_CATEGORIES, language_for_category
from .decoding import is_empty_output, is_function_calling_format


def _reject(output: LMOutput, error: Any, error_type: str) -> float:
    """Record why a reply was rejected, and score it zero."""
    output.metadata["bfcl_error"] = {"error_type": error_type, "error": error}
    return 0.0


@dataclass(frozen=True, slots=True)
class BFCLScorer(Scorer):
    """Score one BFCL reply against its category's criterion.

    The AST categories compare the predicted calls against the entry's possible
    answers. The irrelevance categories supply functions that cannot answer the
    question, so calling nothing is correct; the live relevance category
    supplies one that can, so calling something is correct. Neither relevance
    category has answers to compare against.
    """

    name: str = "bfcl"

    def score(self, instance: Instance, output: LMOutput) -> float:
        test_category = instance.metadata["test_category"]
        if test_category in RELEVANCE_CATEGORIES:
            return self._score_relevance(instance, output, test_category)
        return self._score_ast(instance, output, test_category)

    def _score_relevance(self, instance: Instance, output: LMOutput, test_category: str) -> float:
        decoded = output.extracted_answer
        called = decoded is not None and not is_empty_output(decoded)
        expects_call = test_category not in IRRELEVANCE_CATEGORIES

        if called == expects_call:
            return 1.0
        if expects_call:
            return _reject(
                output,
                output.metadata.get(
                    "bfcl_decode_error", "Decoded no function call when one was expected."
                ),
                "relevance_error:decoder_failed",
            )
        return _reject(
            output,
            f"Called {decoded!r} when no function should have been called.",
            "irrelevance_error:decoder_success",
        )

    def _score_ast(self, instance: Instance, output: LMOutput, test_category: str) -> float:
        decoded = output.extracted_answer
        if decoded is None:
            return _reject(
                output,
                output.metadata.get("bfcl_decode_error", "Failed to decode the reply."),
                "ast_decoder:decoder_failed",
            )
        if not is_function_calling_format(decoded):
            return _reject(
                output,
                f"Decoded to {decoded!r}, which is not a list of function calls.",
                "ast_decoder:decoder_wrong_output_format",
            )

        try:
            result = ast_checker(
                instance.metadata["functions"],
                decoded,
                instance.metadata["ground_truth"],
                language_for_category(test_category),
                test_category,
            )
        except Exception as exc:
            # A malformed function document or possible answer should cost this
            # instance, not abort the whole task.
            return _reject(output, f"{type(exc).__name__}: {exc}", "checker:raised")

        if result["valid"]:
            return 1.0
        return _reject(output, result["error"], result.get("error_type", ""))


__all__ = ["BFCLScorer"]
