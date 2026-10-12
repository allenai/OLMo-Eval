"""VQA-family answer normalization and matching metrics.

Vendored from ``mm_olmo/olmo/eval/vqa.py`` (reference implementation for the
Molmo2 image-QA benchmarks).  The normalization tables and several regex
quirks are preserved byte-for-byte so scores match the original evaluation
exactly — do not "fix" them.

The only intentional change is that ``editdistance.eval`` is replaced by the
pure-python :func:`levenshtein` below to avoid a new dependency.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence

# ---------------------------------------------------------------------------
# Official VQA v2 normalization tables (verbatim)
# ---------------------------------------------------------------------------

CONTRACTIONS = {
    "aint": "ain't",
    "arent": "aren't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hadnt've": "hadn't've",
    "hadn'tve": "hadn't've",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hed": "he'd",
    "hed've": "he'd've",
    "he'dve": "he'd've",
    "hes": "he's",
    "howd": "how'd",
    "howll": "how'll",
    "hows": "how's",
    "Id've": "I'd've",
    "I'dve": "I'd've",
    "Im": "I'm",
    "Ive": "I've",
    "isnt": "isn't",
    "itd": "it'd",
    "itd've": "it'd've",
    "it'dve": "it'd've",
    "itll": "it'll",
    "let's": "let's",
    "maam": "ma'am",
    "mightnt": "mightn't",
    "mightnt've": "mightn't've",
    "mightn'tve": "mightn't've",
    "mightve": "might've",
    "mustnt": "mustn't",
    "mustve": "must've",
    "neednt": "needn't",
    "notve": "not've",
    "oclock": "o'clock",
    "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at",
    "'ows'at": "'ow's'at",
    "'ow'sat": "'ow's'at",
    "shant": "shan't",
    "shed've": "she'd've",
    "she'dve": "she'd've",
    "she's": "she's",
    "shouldve": "should've",
    "shouldnt": "shouldn't",
    "shouldnt've": "shouldn't've",
    "shouldn'tve": "shouldn't've",
    "somebody'd": "somebodyd",
    "somebodyd've": "somebody'd've",
    "somebody'dve": "somebody'd've",
    "somebodyll": "somebody'll",
    "somebodys": "somebody's",
    "someoned": "someone'd",
    "someoned've": "someone'd've",
    "someone'dve": "someone'd've",
    "someonell": "someone'll",
    "someones": "someone's",
    "somethingd": "something'd",
    "somethingd've": "something'd've",
    "something'dve": "something'd've",
    "somethingll": "something'll",
    "thats": "that's",
    "thered": "there'd",
    "thered've": "there'd've",
    "there'dve": "there'd've",
    "therere": "there're",
    "theres": "there's",
    "theyd": "they'd",
    "theyd've": "they'd've",
    "they'dve": "they'd've",
    "theyll": "they'll",
    "theyre": "they're",
    "theyve": "they've",
    "twas": "'twas",
    "wasnt": "wasn't",
    "wed've": "we'd've",
    "we'dve": "we'd've",
    "weve": "we've",
    "werent": "weren't",
    "whatll": "what'll",
    "whatre": "what're",
    "whats": "what's",
    "whatve": "what've",
    "whens": "when's",
    "whered": "where'd",
    "wheres": "where's",
    "whereve": "where've",
    "whod": "who'd",
    "whod've": "who'd've",
    "who'dve": "who'd've",
    "wholl": "who'll",
    "whos": "who's",
    "whove": "who've",
    "whyll": "why'll",
    "whyre": "why're",
    "whys": "why's",
    "wont": "won't",
    "wouldve": "would've",
    "wouldnt": "wouldn't",
    "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've",
    "yall": "y'all",
    "yall'll": "y'all'll",
    "y'allll": "y'all'll",
    "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've",
    "y'all'dve": "y'all'd've",
    "youd": "you'd",
    "youd've": "you'd've",
    "you'dve": "you'd've",
    "youll": "you'll",
    "youre": "you're",
    "youve": "you've",
}

MANUAL_MAP = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}

ARTICLES = ["a", "an", "the"]

PUNCT = [
    ";",
    r"/",
    "[",
    "]",
    '"',
    "{",
    "}",
    "(",
    ")",
    "=",
    "+",
    "\\",
    "_",
    "-",
    ">",
    "<",
    "@",
    "`",
    ",",
    "?",
    "!",
]

# NOTE: both regex quirks below are upstream VQA-eval bugs preserved on
# purpose: `(?!<=\d)` is a (useless) negative lookahead for the literal text
# "<=<digit>", not the intended look-behind, and the original code passes
# ``re.UNICODE`` (== 32) as the *count* argument of ``periodStrip.sub``.
_PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
_COMMA_STRIP = re.compile(r"(\d)(\,)(\d)")


def process_punctuation(in_text: str) -> str:
    out_text = in_text
    for p in PUNCT:
        if (p + " " in in_text or " " + p in in_text) or (
            re.search(_COMMA_STRIP, in_text) is not None
        ):
            out_text = out_text.replace(p, "")
        else:
            out_text = out_text.replace(p, " ")
    out_text = _PERIOD_STRIP.sub("", out_text, re.UNICODE)
    return out_text


def process_digit_article(in_text: str) -> str:
    out_text = []
    temp_text = in_text.lower().split()
    for word in temp_text:
        word = MANUAL_MAP.setdefault(word, word)
        if word not in ARTICLES:
            out_text.append(word)
    for word_id, word in enumerate(out_text):
        if word in CONTRACTIONS:
            out_text[word_id] = CONTRACTIONS[word]
    return " ".join(out_text)


_PREPROCESS_CACHE: dict[str, str] = {}


def preprocess_answer(ans: str) -> str:
    """Official VQA v2 answer normalization (cached)."""
    if ans in _PREPROCESS_CACHE:
        return _PREPROCESS_CACHE[ans]
    out = ans.replace("\n", " ").replace("\t", " ").lower().strip()
    preprocessed = process_digit_article(process_punctuation(out))
    _PREPROCESS_CACHE[ans] = preprocessed
    return preprocessed


# ---------------------------------------------------------------------------
# Edit distance (replaces the `editdistance` dependency)
# ---------------------------------------------------------------------------


def levenshtein(a: str, b: str) -> int:
    """Plain Levenshtein edit distance (insert/delete/substitute, cost 1)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _argmin(values: Sequence[float]) -> int:
    """Index of the minimum value, first occurrence (matches np.argmin)."""
    best_ix = 0
    for ix in range(1, len(values)):
        if values[ix] < values[best_ix]:
            best_ix = ix
    return best_ix


# ---------------------------------------------------------------------------
# Prediction cleanup (from mm_olmo VqaEval.__call__)
# ---------------------------------------------------------------------------


def clean_prediction(pred: str) -> str:
    """Cleanup applied by the original ``VqaEval`` before every metric.

    Strips whitespace; takes the text after the first "Answer:" if present;
    for multi-line output keeps the most frequent line; otherwise collapses
    inner whitespace.  Counting/MathVista evaluators do NOT use this.
    """
    pred = pred.strip()
    if "Answer:" in pred:
        pred = pred.split("Answer:")[1].strip()
    elif "\n" in pred:
        preds = [" ".join(x.strip().split()) for x in pred.split("\n")]
        counts = Counter(preds)
        max_count = max(counts.values())
        pred = [x for x in preds if counts[x] == max_count][0]
    else:
        pred = " ".join(pred.strip().split())
    return pred


# ---------------------------------------------------------------------------
# Metrics (verbatim logic)
# ---------------------------------------------------------------------------


def vqa_score(target: list[str] | str, pred: str) -> float:
    """Official VQA v2 accuracy: min(#matching annotator answers / 3, 1)."""
    pred = preprocess_answer(pred)
    if isinstance(target, list):
        counts = Counter(preprocess_answer(x) for x in target)
        return min(counts[pred] / 3.0, 1)
    return float(pred == target)


def select_mc_option(target: str, options: list[str] | str) -> int:
    """Select a multiple-choice option index from the model output.

    Exact match, then unique prefix containment in both directions, then
    unique substring, then minimum edit distance.
    """
    target = target.lower().strip()
    n = len(options)
    options = [x.lower().strip() for x in options]
    assert len(set(options)) == n

    for ix, option in enumerate(options):
        if option == target:
            return ix

    contains = [ix for ix, option in enumerate(options) if target.startswith(option)]
    if len(contains) == 1:
        return contains[0]

    contains = [ix for ix, option in enumerate(options) if option.startswith(target)]
    if len(contains) == 1:
        return contains[0]

    contains = [ix for ix, option in enumerate(options) if target in option]
    if len(contains) == 1:
        return contains[0]

    distances = [levenshtein(opt, target) for opt in options]
    return _argmin(distances)


def anls_metric(target: str, prediction: str, theta: float = 0.5) -> float:
    """ANLS for DocVQA/InfographicVQA (case-insensitive, θ=0.5)."""
    if not target and not prediction:
        # Degenerate case (placeholder test-split answers); mm_olmo would
        # divide by zero here. Treat two empty strings as an exact match.
        return 1.0
    edit_distance = levenshtein(target.lower(), prediction.lower())
    normalized_ld = edit_distance / max(len(target), len(prediction))
    return 1 - normalized_ld if normalized_ld < theta else 0


def relaxed_correctness(target: str, prediction: str, max_relative_change: float = 0.05) -> bool:
    """ChartQA relaxed accuracy: 5% numeric tolerance, exact match otherwise."""

    def _to_float(text: str) -> float | None:
        try:
            if text.endswith("%"):
                return float(text.rstrip("%")) / 100.0
            return float(text)
        except ValueError:
            return None

    prediction_float = _to_float(prediction)
    target_float = _to_float(target)
    if prediction_float is not None and target_float:
        relative_change = abs(prediction_float - target_float) / abs(target_float)
        return relative_change <= max_relative_change
    return prediction.lower() == target.lower()


def scifi_relaxed_correctness(
    target: str, prediction: str, max_relative_change: float = 0.05
) -> bool:
    """Lenient ChartQA variant: number extraction, word→digit, ÷100, substring."""

    def _to_float(text: str) -> float | None:
        try:
            return float(text)
        except ValueError:
            return None

    def compute_relative_change(target_f: float, prediction_f: float) -> float:
        if target_f == 0:
            return abs(target_f - prediction_f)
        return abs(target_f - prediction_f) / abs(target_f)

    def extract_short_answer(text: str) -> str:
        if "answer:" in text:
            return text.split("answer:")[1].strip()
        return text

    prediction = extract_short_answer(prediction.lower().strip())
    target = extract_short_answer(target.lower().strip())

    if len(prediction) == 0:
        return False

    if prediction[-1] == ".":
        prediction = prediction[:-1]

    word_to_num = {k: v for k, v in MANUAL_MAP.items() if k != "none"}

    target_float = _to_float(target)
    if target_float is not None:
        if "," in prediction:
            prediction = prediction.replace(",", "")

        for word, num in word_to_num.items():
            prediction = prediction.replace(word, str(num))

        match = re.search(r"[-+]?\d*\.\d+|\d+", prediction)
        prediction_float = _to_float(match.group()) if match else None
        if prediction_float is None:
            return False

        relative_change = compute_relative_change(target_float, prediction_float)

        prediction_float_normalized = prediction_float / 100
        relative_change_normalized = compute_relative_change(
            target_float, prediction_float_normalized
        )

        return bool(
            relative_change <= max_relative_change
            or relative_change_normalized <= max_relative_change
        )

    if "[" in target and "," in target:
        # target is a list
        targets = target.replace("[", "").replace("]", "").split(",")
        return all(t.strip().lower() in prediction for t in targets)

    return target.strip().lower() in prediction


def real_world_qa_score(target: str, prediction: str, question_type: str) -> float:
    """RealWorldQA: A–D letter selection for MC, VQA2-normalized EM otherwise."""
    if question_type == "multiple_choice":
        options = ["A", "B", "C", "D"]
        pred_idx = select_mc_option(prediction, options)
        gt_idx = options.index(target)
        return float(pred_idx == gt_idx)
    pred = preprocess_answer(prediction)
    gt = preprocess_answer(target)
    return float(pred == gt)
