"""Pure logic for the MMMU-Pro benchmark (decoupled from the MMMU task).

Vendored verbatim from the official MMMU-Pro eval repo
(``MMMU-Benchmark/MMMU`` → ``mmmu-pro/{prompts.yaml,evaluate.py,infer/infer_gemini.py}``) so
MMMU-Pro is scored exactly as the benchmark authors intend — independent of the (different) MMMU
parser/prompts in :mod:`olmo_eval.evals.vision.scoring.multiple_choice`.

Contents:

* the official **direct** prompt strings;
* the standard-setting prompt assembly (``parse_options`` / ``construct_standard_prompt``) and the
  interleaved-image handling (``replace_images_tokens`` — options are shuffled, so ``<image i>``
  token order need not match ``image_i`` key order);
* the official answer parser (``parse_multi_choice_response``) and ``mmmu_pro_score``.

MMMU-Pro is multiple-choice only.
"""

from __future__ import annotations

import random
import re
import zlib
from dataclasses import dataclass

from olmo_eval.common.scorers.base import Scorer
from olmo_eval.common.types import Instance, LMOutput

# Direct-mode prompts, verbatim from mmmu-pro/prompts.yaml (CoT mode is not used).
MMMU_PRO_STANDARD_DIRECT = "Answer with the option letter from the given choices directly."
MMMU_PRO_VISION_DIRECT = (
    "Answer with the option letter from the given choices directly. The last line of your "
    "response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER "
    "is one of options."
)

_IMAGE_TOKEN_RE = re.compile(r"<image\s+(\d+)>")


def parse_options(options: list[str]) -> str:
    """Render options as ``"A. <opt>\\nB. <opt>\\n…"`` (official ``parse_options``)."""
    letters = [chr(ord("A") + i) for i in range(len(options))]
    return "\n".join(f"{letter}. {opt}" for letter, opt in zip(letters, options, strict=True))


def construct_standard_prompt(question: str, options: list[str]) -> str:
    """Official ``construct_prompt`` for the standard setting (question + options + direct prompt)."""
    return f"{question}\n{parse_options(options)}\n{MMMU_PRO_STANDARD_DIRECT}"


def replace_images_tokens(text: str) -> tuple[str, list[int]]:
    """Official ``replace_images_tokens``: record ``<image i>`` order, replace with ``<image>``.

    Returns ``(text, image_order)`` where ``image_order`` is the list of image indices in their
    order of appearance across the assembled text (duplicates kept). Each ``<image i>`` maps to the
    dataset's ``image_i`` key; the option shuffle means this order need not be ``1, 2, 3, …``.
    """
    image_order = [int(n) for n in _IMAGE_TOKEN_RE.findall(text)]
    text = _IMAGE_TOKEN_RE.sub("<image>", text)
    return text, image_order


def get_multi_choice_info(options: list[str]) -> tuple[dict[str, str], list[str]]:
    """Official ``get_multi_choice_info``: ``(index2ans, all_choices)`` lettered A, B, C, …."""
    index2ans: dict[str, str] = {}
    all_choices: list[str] = []
    for i, option in enumerate(options):
        letter = chr(ord("A") + i)
        index2ans[letter] = option
        all_choices.append(letter)
    return index2ans, all_choices


def parse_multi_choice_response(
    response: str,
    all_choices: list[str],
    index2ans: dict[str, str],
    *,
    seed: int | None = None,
) -> str:
    """Official MMMU-Pro ``parse_multi_choice_response`` (fed the raw model response).

    Ladder: (1) the last ``"Answer:"`` marker — if exactly one choice letter follows it, take it;
    (2) else strip trailing punctuation + pad, then prefer ``(A)`` → ``"A "`` → ``"A."`` → option
    text (only when >5 tokens); (3) no candidate → random choice; (4) multiple → the last
    occurrence. The only change from the official code is that the random fallback is seeded
    (``random.Random(seed)``) so re-runs are deterministic on otherwise-unparseable outputs.
    """
    last_answer_pos = response.rfind("Answer:")
    if last_answer_pos != -1:
        answer_str = response[last_answer_pos + len("Answer:") :].strip()
        matching_options = [option for option in all_choices if option in answer_str]
        if len(matching_options) == 1:
            return matching_options[0]

    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "  # avoid partial matches

    index_ans = True
    ans_with_brack = False
    candidates: list[str] = []
    for choice in all_choices:  # (A) (B) (C) …
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True
    if len(candidates) == 0:
        for choice in all_choices:  # A B C …
            if f"{choice} " in response:
                candidates.append(choice)
    if len(candidates) == 0:
        for choice in all_choices:  # A. B. C. …
            if f"{choice}." in response:
                candidates.append(choice)
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False  # matched on content, not the letter

    if len(candidates) == 0:
        rng = random.Random(seed) if seed is not None else random
        return rng.choice(all_choices)
    if len(candidates) > 1:
        start_indexes: list[int] = []
        if index_ans:
            for can in candidates:
                pattern = f"({can})" if ans_with_brack else f" {can} "
                start_indexes.append(response.rfind(pattern))
        else:
            for can in candidates:
                start_indexes.append(response.lower().rfind(index2ans[can].lower()))
        # the last occurrence wins (official uses np.argmax over rfind positions)
        return candidates[max(range(len(candidates)), key=lambda i: start_indexes[i])]
    return candidates[0]


def mmmu_pro_score(
    response: str,
    options: list[str],
    answer: str | list[str],
    *,
    example_id: str = "",
) -> float:
    """1.0 if the parsed option letter matches the gold answer, else 0.0."""
    index2ans, all_choices = get_multi_choice_info(options)
    seed = zlib.crc32(str(example_id).encode())
    parsed = parse_multi_choice_response(response or "", all_choices, index2ans, seed=seed)
    if isinstance(answer, list):
        return float(parsed in answer)
    return float(parsed == answer)


@dataclass(frozen=True, slots=True)
class MmmuProScorer(Scorer):
    """Official MMMU-Pro multiple-choice scoring (1.0 if the parsed letter matches the gold)."""

    name: str = "mmmu_pro"

    def score(self, instance: Instance, output: LMOutput) -> float:
        meta = instance.metadata
        return mmmu_pro_score(
            output.text or "",
            list(meta.get("options") or []),
            meta["answer"],
            example_id=str(meta.get("example_id", "")),
        )
