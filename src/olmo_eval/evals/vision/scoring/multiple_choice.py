"""MMMU response parsing and scoring.

Vendored from ``mm_olmo/olmo/eval/mmmu_eval_utils.py`` (itself adapted from
the official MMMU repository) plus the ``mmmu_score`` dispatcher from
``mm_olmo/olmo/eval/vqa.py``.

One intentional change: the original falls back to ``random.choice`` (with a
module-level ``random.seed(42)``) when a multiple-choice response cannot be
parsed, which makes scores depend on scoring order.  Here the fallback uses a
``random.Random`` seeded per call from a stable instance id so results are
deterministic and order-independent.
"""

from __future__ import annotations

import random
import re
import string
import zlib


def _argmax(values: list[int]) -> int:
    """Index of the maximum value, first occurrence (matches np.argmax)."""
    best_ix = 0
    for ix in range(1, len(values)):
        if values[ix] > values[best_ix]:
            best_ix = ix
    return best_ix


def _fallback_rng(stable_id: str | None) -> random.Random:
    seed = 42 if stable_id is None else zlib.crc32(stable_id.encode("utf-8"))
    return random.Random(seed)


# ----------- Process Multi-choice -------------
def parse_multi_choice_response(
    response: str,
    all_choices: list[str],
    index2ans: dict[str, str],
    stable_id: str | None = None,
) -> str:
    """Parse the predicted option letter (e.g. A/B/C/D) from a response."""
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "  # add space to avoid partial match

    index_ans = True
    ans_with_brack = False
    ans_with_last_brack = False
    ans_with_dot = False
    ans_with_colon = False
    candidates = []
    for choice in all_choices:  # e.g., (A) (B) (C) (D)
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True

    for choice in all_choices:  # e.g., A), B), C), D)
        if f"{choice})" in response:
            candidates.append(choice)
            ans_with_last_brack = True

    for choice in all_choices:  # e.g., A. B. C. D.
        if f"{choice}." in response:
            candidates.append(choice)
            ans_with_dot = True

    for choice in all_choices:  # e.g., A: B: C: D:
        if f"{choice}:" in response:
            candidates.append(choice)
            ans_with_colon = True

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A B C D
            if response.strip() == choice:
                return choice
            if f" {choice} " in response:
                candidates.append(choice)

    # if all above doesn't get candidates, check if the content is larger than
    # 5 tokens and try to parse the example
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False  # it's content ans.

    if len(candidates) == 0:  # still not get answer, choose one deterministically.
        pred_index = _fallback_rng(stable_id).choice(all_choices)
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack:
                for can in candidates:
                    start_indexes.append(response.rfind(f"({can})"))
            elif ans_with_last_brack:
                for can in candidates:
                    start_indexes.append(response.rfind(f"{can})"))
            elif ans_with_dot:
                for can in candidates:
                    start_indexes.append(response.rfind(f"{can}."))
            elif ans_with_colon:
                for can in candidates:
                    start_indexes.append(response.rfind(f"{can}:"))
            else:
                for can in candidates:
                    start_indexes.append(response.rfind(f" {can} "))
        else:
            for can in candidates:
                start_indexes.append(response.lower().rfind(index2ans[can].lower()))
        # get the last one
        pred_index = candidates[_argmax(start_indexes)]
    else:  # if only one candidate, use it.
        pred_index = candidates[0]

    return pred_index


# ----------- Process Open -------------
def check_is_number(value: str) -> bool:
    """Check if the given string is a number."""
    try:
        float(value.replace(",", ""))
        return True
    except ValueError:
        return False


def normalize_str(value: str) -> list:
    """Normalize a string to lower case, converting to float when possible."""
    value = value.strip()

    if check_is_number(value):
        value = value.replace(",", "")
        number = round(float(value), 2)
        return [number]
    value = value.lower()
    if len(value) == 1:
        return [" " + value, value + " "]  # avoid trivial matches
    return [value]


def extract_numbers(value: str) -> list[str]:
    """Extract all forms of numbers from a string with regex."""
    pattern_commas = r"-?\b\d{1,3}(?:,\d{3})+\b"
    pattern_scientific = r"-?\d+(?:\.\d+)?[eE][+-]?\d+"
    pattern_simple = r"-?(?:\d+\.\d+|\.\d+|\d+\b)(?![eE][+-]?\d+)(?![,\d])"

    numbers_with_commas = re.findall(pattern_commas, value)
    numbers_scientific = re.findall(pattern_scientific, value)
    numbers_simple = re.findall(pattern_simple, value)

    return numbers_with_commas + numbers_scientific + numbers_simple


def parse_open_response(response: str) -> list:
    """Parse predicted strings/numbers from an open-ended response."""

    def get_key_subresponses(resp_text: str) -> list[str]:
        resp_text = resp_text.strip().strip(".").lower()
        sub_responses = re.split(r"\.\s(?=[A-Z])|\n", resp_text)
        indicators_of_keys = [
            "could be ",
            "so ",
            "is ",
            "thus ",
            "therefore ",
            "final ",
            "answer ",
            "result ",
        ]
        key_responses = []
        for index, resp in enumerate(sub_responses):
            # if last one, accept it's an equation (the entire response can be
            # just one sentence with equation)
            if index == len(sub_responses) - 1:
                indicators_of_keys.extend(["="])
            shortest_key_response = None
            for indicator in indicators_of_keys:
                if indicator in resp:
                    if not shortest_key_response:
                        shortest_key_response = resp.split(indicator)[-1].strip()
                    else:
                        if len(resp.split(indicator)[-1].strip()) < len(shortest_key_response):
                            shortest_key_response = resp.split(indicator)[-1].strip()

            # accept the shortest key response if it's not trivial
            if shortest_key_response and shortest_key_response.strip() not in [
                ":",
                ",",
                ".",
                "!",
                "?",
                ";",
                ":",
                "'",
            ]:
                key_responses.append(shortest_key_response)
        if len(key_responses) == 0:
            return [resp_text]
        return key_responses

    key_responses = get_key_subresponses(response)

    pred_list = key_responses.copy()  # keep the original string response
    for resp in key_responses:
        pred_list.extend(extract_numbers(resp))

    tmp_pred_list = []
    for pred in pred_list:
        tmp_pred_list.extend(normalize_str(pred))

    # remove duplicates
    return list(set(tmp_pred_list))


# ----------- Evaluation -------------
def eval_multi_choice(gold_i: list | str, pred_i: str) -> bool:
    """Evaluate a multiple-choice instance."""
    if isinstance(gold_i, list):
        return any(answer == pred_i for answer in gold_i)
    return gold_i == pred_i


def eval_open(gold_i: list | str, pred_i: list) -> bool:
    """Evaluate an open-question instance."""
    correct = False
    if isinstance(gold_i, list):
        norm_answers = []
        for answer in gold_i:
            norm_answers.extend(normalize_str(answer))
    else:
        norm_answers = normalize_str(gold_i)
    for pred in pred_i:  # pred is already normalized in parse response phase
        if isinstance(pred, str):  # if it's a string, then find if ans in the pred_i
            for norm_ans in norm_answers:
                if isinstance(norm_ans, str) and norm_ans in pred:
                    if not correct:
                        correct = True
                    break
        else:  # it's a float number
            if pred in norm_answers:
                if not correct:
                    correct = True
                break
    return correct


def mmmu_score(
    target: list[str] | str,
    response: str,
    question_type: str,
    options: list[str],
    stable_id: str | None = None,
) -> float:
    """Score one MMMU example following the official protocol."""
    if question_type == "multiple-choice":
        options = [opt for opt in options if len(opt) > 0]
        all_choices = list(string.ascii_uppercase[: len(options)])
        index2ans = dict(zip(all_choices, options, strict=False))
        parsed_pred = parse_multi_choice_response(
            response, all_choices, index2ans, stable_id=stable_id
        )
        correct = eval_multi_choice(target, parsed_pred)
    else:  # open
        parsed_pred = parse_open_response(response)
        correct = eval_open(target, parsed_pred)
    return float(correct)
