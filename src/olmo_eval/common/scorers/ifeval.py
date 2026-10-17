"""Scorers for IFBench / IFEval / Multi-IF instruction-following evaluation.

Uses the ``ifbench`` package registry, which covers the original IFEval
(DEFAULT) verifiers, the OOD verifiers used by ``allenai/IFBench_test``,
and the verifiers used by the multi-turn ``VGraf/ifeval_mt`` slices. The
scorer evaluates a response against per-instance instructions (looked up
in ``instance.metadata["instruction_id_list"]`` / ``"kwargs"``) and writes
both strict and loose pass/fail lists for each instruction into
``output.metadata["ifeval"]``. The four IFEval metrics consume that field.

The Multi-IF scorer runs the same checks once per conversation turn, with the
multilingual verifiers from
:mod:`olmo_eval.common.scorers.multilingual_instructions`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from olmo_eval.common.multi_turn import missing_replies_error
from olmo_eval.common.scorers.base import Scorer, set_scorer_result
from olmo_eval.common.types import Instance, LMOutput


def _loose_response_variants(response: str) -> list[str]:
    """Generate the eight response variants used by upstream loose scoring."""
    lines = response.split("\n")
    remove_first = "\n".join(lines[1:]).strip()
    remove_last = "\n".join(lines[:-1]).strip()
    remove_both = "\n".join(lines[1:-1]).strip()
    return [
        response,
        response.replace("*", ""),
        remove_first,
        remove_last,
        remove_both,
        remove_first.replace("*", ""),
        remove_last.replace("*", ""),
        remove_both.replace("*", ""),
    ]


def _build_instruction(
    instruction_cls: Any,
    instruction_id: str,
    kwargs: Mapping[str, Any],
    prompt: str,
) -> Any:
    """Instantiate and configure a single instruction verifier.

    Built once per instruction and reused for the strict check and every loose
    variant: verifiers draw random parameters for unspecified kwargs, so
    rebuilding per variant would score each variant against different criteria.
    """
    instruction = instruction_cls(instruction_id)
    cleaned_kwargs = {k: v for k, v in kwargs.items() if v is not None}
    arg_keys = instruction.get_instruction_args_keys()
    if "prompt_to_repeat" in arg_keys and not cleaned_kwargs.get("prompt_to_repeat"):
        cleaned_kwargs["prompt_to_repeat"] = prompt
    instruction.build_description(**cleaned_kwargs)
    args = instruction.get_instruction_args()
    if args and "prompt" in args:
        instruction.build_description(prompt=prompt)
    return instruction


def _check_one(instruction: Any, response: str) -> bool:
    return bool(response.strip()) and bool(instruction.check_following(response))


def verify_instructions(
    response: str,
    instruction_ids: Sequence[str],
    kwargs_list: Sequence[Mapping[str, Any]],
    prompt: str,
    registry: Mapping[str, Any],
) -> dict[str, Any]:
    """Check a response against each instruction, strictly and loosely.

    Returns ``"strict"`` and ``"loose"`` pass lists with one entry per
    instruction, plus ``"errors"`` naming any instruction that could not be
    built from its kwargs.
    """
    strict_results: list[bool] = []
    loose_results: list[bool] = []
    errors: dict[str, str] = {}
    loose_variants = _loose_response_variants(response)
    for inst_id, inst_kwargs in zip(instruction_ids, kwargs_list, strict=True):
        # An instruction the dataset describes incorrectly — an unrecognized
        # id, or kwargs the verifier does not accept — scores as not followed,
        # matching the reference implementation, rather than failing the whole
        # response. Failures raised while checking a successfully built
        # instruction are not model behavior (missing NLTK corpora on an
        # offline worker, say), so they propagate instead of being recorded as
        # instructions the response did not follow.
        try:
            instruction = _build_instruction(registry[inst_id], inst_id, inst_kwargs, prompt)
        except (TypeError, KeyError) as exc:
            errors[inst_id] = f"{type(exc).__name__}: {exc}"
            strict_results.append(False)
            loose_results.append(False)
            continue
        strict_results.append(_check_one(instruction, response))
        loose_results.append(any(_check_one(instruction, variant) for variant in loose_variants))

    result: dict[str, Any] = {"strict": strict_results, "loose": loose_results}
    if errors:
        result["errors"] = errors
    return result


@dataclass(frozen=True)
class IFEvalScorer(Scorer):
    """Run IFBench/IFEval instruction verifiers against a response.

    The numeric ``score()`` return is the prompt-level loose accuracy (1.0 if
    every instruction passes under at least one loose variant, else 0.0). The
    full strict + loose pass lists are written to ``output.metadata["ifeval"]``
    so the four metric classes can derive prompt/inst × strict/loose figures.
    """

    name: str = "ifeval"

    def score(self, instance: Instance, output: LMOutput) -> float:
        instruction_ids: list[str] = instance.metadata.get("instruction_id_list", [])
        kwargs_list: list[dict[str, Any]] = instance.metadata.get("kwargs", [])
        prompt: str = instance.metadata.get("prompt", instance.question)
        # Verify what the task extracted when it did (reasoning models'
        # ``<think>`` content is removed there); otherwise the raw text.
        extracted = output.extracted_answer
        response: str = extracted if isinstance(extracted, str) else (output.text or "")

        result: dict[str, Any] = {"strict": [], "loose": []}
        if instruction_ids:
            from ifbench import instructions_registry

            result = verify_instructions(
                response,
                instruction_ids,
                kwargs_list,
                prompt,
                instructions_registry.INSTRUCTION_DICT,
            )

        if output.metadata is None:
            output.metadata = {}
        output.metadata["ifeval"] = result

        loose_results = result["loose"]
        if not loose_results:
            return 0.0
        return 1.0 if all(loose_results) else 0.0


@dataclass(frozen=True)
class MultiIFScorer(Scorer):
    """Verify each turn of a Multi-IF conversation against its own instructions.

    Expects ``output.extracted_answer`` to hold the model's reply to each user
    turn, in order; a turn the conversation never reached scores as an empty
    reply. Each turn's strict and loose pass lists are recorded as this
    scorer's result, under ``"turns"``. The numeric score is the fraction of
    turns whose instructions all pass under loose scoring.
    """

    name: str = "multi_if"

    def score(self, instance: Instance, output: LMOutput) -> float:
        replies = output.extracted_answer
        if not isinstance(replies, list):
            raise missing_replies_error("Multi-IF")
        from olmo_eval.common.scorers.multilingual_instructions import MULTI_IF_INSTRUCTIONS

        turns: list[dict[str, Any]] = instance.metadata["turns"]
        turn_results = [
            verify_instructions(
                replies[index] if index < len(replies) else "",
                turn["instruction_id_list"],
                turn["kwargs"],
                turn["prompt"],
                MULTI_IF_INSTRUCTIONS,
            )
            for index, turn in enumerate(turns)
        ]
        set_scorer_result(output, self.name, {"turns": turn_results})
        if not turn_results:
            return 0.0
        followed = sum(1 for result in turn_results if all(result["loose"]))
        return followed / len(turn_results)
