"""MathVista scoring: offline answer extraction + official normalization.

Vendored from ``mm_olmo/olmo/eval/math_vista_utils.py`` and the offline
(`use_api=False`) branch of ``math_vista_score`` in ``mm_olmo/olmo/eval/vqa.py``.

The official MathVista protocol extracts the final answer from the model
response with GPT-4 (``gpt-4-0613``) before comparison; :data:`DEMO_PROMPT`
and :func:`create_test_prompt` are vendored here for the optional GPT-backed
scorer.  The offline path replaces only the extraction step (letter matching
for multiple choice, int/float parsing otherwise); normalization and the
final comparison are identical.
"""

from __future__ import annotations

import re

from olmo_eval.evals.vision.scoring.vqa_normalization import levenshtein, select_mc_option

DEMO_PROMPT = """
Please read the following example. Then extract the answer from the model response and type it at the end of the prompt.

Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end.
Question: Which number is missing?

Model response: The number missing in the sequence is 14.

Extracted answer: 14

Hint: Please answer the question requiring a floating-point number with one decimal place and provide the final value, e.g., 1.2, 1.3, 1.4, at the end.
Question: What is the fraction of females facing the camera?

Model response: The fraction of females facing the camera is 0.6, which means that six out of ten females in the group are facing the camera.

Extracted answer: 0.6

Hint: Please answer the question requiring a floating-point number with two decimal places and provide the final value, e.g., 1.23, 1.34, 1.45, at the end.
Question: How much money does Luca need to buy a sour apple candy and a butterscotch candy? (Unit: $)

Model response: Luca needs $1.45 to buy a sour apple candy and a butterscotch candy.

Extracted answer: 1.45

Hint: Please answer the question requiring a Python list as an answer and provide the final list, e.g., [1, 2, 3], [1.2, 1.3, 1.4], at the end.
Question: Between which two years does the line  graph saw its maximum peak?

Model response: The line graph saw its maximum peak between 2007 and 2008.

Extracted answer: [2007, 2008]

Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.
Question: What fraction of the shape is blue?
Choices:
A: 3/11
B: 8/11
C: 6/11
D: 3/5

Model response: The correct answer is B: 8/11.

Extracted answer: B
"""


def create_test_prompt(query: str, response: str) -> str:
    """Build the GPT-4 answer-extraction prompt (official MathVista)."""
    demo = DEMO_PROMPT.strip()
    test_prompt = f"{query}\n\n{response}"
    return f"{demo}\n\n{test_prompt}\n\nExtracted answer: "


def get_most_similar(prediction: str, choices: list[str]) -> str:
    """Return the choice closest to ``prediction`` by edit distance."""
    distances = [levenshtein(prediction, choice) for choice in choices]
    return choices[distances.index(min(distances))]


def normalize_extracted_answer(
    extraction,
    choices: list[str],
    question_type: str,
    answer_type: str,
    precision,
):
    """Normalize the extracted answer to match the answer type (official)."""
    if question_type == "multi_choice":
        if isinstance(extraction, str):
            extraction = extraction.strip()
        else:
            try:
                extraction = str(extraction)
            except Exception:
                extraction = ""

        # extract "A" from "(A) text"
        letter = re.findall(r"([a-zA-Z]):", extraction)
        if len(letter) > 0:
            extraction = letter[0].upper()

        options = [chr(ord("A") + i) for i in range(len(choices))]

        if extraction in options:
            ind = options.index(extraction)
            extraction = choices[ind]
        else:
            extraction = get_most_similar(extraction, choices)
        assert extraction in choices

    elif answer_type == "integer":
        try:
            extraction = str(int(float(extraction)))
        except Exception:
            extraction = None

    elif answer_type == "float":
        try:
            extraction = str(round(float(extraction), precision))
        except Exception:
            extraction = None

    elif answer_type == "list":
        try:
            extraction = str(extraction)
        except Exception:
            extraction = None

    return extraction


def safe_equal(prediction, answer) -> bool:
    """Compare prediction and answer, tolerating type mismatches."""
    try:
        return prediction == answer
    except Exception:
        return False


def extract_answer_offline(
    response: str,
    question_type: str,
    answer_type: str,
    choices: list[str],
) -> str:
    """Offline answer extraction (no GPT call).

    Applies the official ``extract_answer`` deterministic short-circuits first
    (empty response, response verbatim in choices, int/float parsing), then
    falls back to the mm_olmo ``use_api=False`` branch: letter matching via
    :func:`select_mc_option` for multiple choice, raw response otherwise.
    """
    quick = extract_answer_quick(response, question_type, answer_type, choices)
    if quick is not None:
        return quick
    if question_type == "multi_choice":
        options = [chr(ord("A") + i) for i in range(len(choices))]
        pred_idx = select_mc_option(response, options)
        return choices[pred_idx]
    return response


def extract_answer_quick(
    response: str,
    question_type: str,
    answer_type: str,
    choices: list[str],
) -> str | None:
    """The deterministic pre-GPT short-circuits of the official ``extract_answer``.

    Returns None when GPT extraction would be required.
    """
    if response == "":
        return ""
    if question_type == "multi_choice" and response in choices:
        return response
    if answer_type == "integer":
        try:
            return str(int(response))
        except Exception:
            pass
    if answer_type == "float":
        try:
            return str(float(response))
        except Exception:
            pass
    return None


def math_vista_score_offline(
    response: str,
    *,
    question_type: str,
    answer_type: str,
    choices: list[str],
    precision,
    target,
) -> bool:
    """Score one MathVista example with offline extraction."""
    extraction = extract_answer_offline(response, question_type, answer_type, choices)
    prediction = normalize_extracted_answer(
        extraction, choices, question_type, answer_type, precision
    )
    return safe_equal(prediction, target)


def math_vista_score_from_extraction(
    extraction,
    *,
    question_type: str,
    answer_type: str,
    choices: list[str],
    precision,
    target,
) -> bool:
    """Score from an already-extracted answer (offline or GPT-based)."""
    prediction = normalize_extracted_answer(
        extraction, choices, question_type, answer_type, precision
    )
    return safe_equal(prediction, target)
