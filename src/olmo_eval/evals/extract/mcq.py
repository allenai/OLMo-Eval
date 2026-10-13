"""Multiple-choice answer extraction from free-form model output.

Patterns are tried in priority order; the last match from the first
pattern that hits wins.  Priority reflects specificity — explicit
instruction-following and LaTeX formats are unambiguous, while
parenthesised or bold letters appear throughout running text and are
used only as fallbacks.

This is olmo-eval's native extractor (used by BBQ, WMDP, MedQA). Tasks
ported for parity with oe-eval use ``answer_format.extract_answer_with_format``
instead, whose cascade and priority order mirror oe-eval's and yield
different answers on the same text.
"""

import re

# A letter that stands alone on its line or labels an option ("B", "**B**",
# "B)", "(B) Paris", "B."), as opposed to the first letter of a sentence.
# Used as a lookahead before the letter is captured.
_LONE_LETTER = r"[*_(]*{letter}(?:[*_]*[).:]|[*_]*[^\S\n]*$)"

# "ANSWER: X" — the explicit instruction-following format. Tolerates markdown
# emphasis and parentheses around the label and letter ("**Answer:** B",
# "Answer: **(C)**", "Answer: _B_"). The letter must not be followed by another
# letter or digit, so "ANSWER: the ..." does not capture the "t" of "the".
# After the colon, the letter may also start the next non-blank line
# ("**Answer:**\n\nC"), but only as a lone letter, so "Answer:\n\nA good
# approach ..." does not capture the article "A".
# The colon may be omitted ("Answer C", "**Answer** (B)"). Without it, the
# letter must be on the same line, uppercase, and a lone letter, so prose such
# as "answer a) first" or "answer A good question" does not match.
# [^\S\n] = whitespace excluding newline.
ANSWER_LINE_PATTERN: re.Pattern[str] = re.compile(
    r"ANSWER[*_]*"
    r"(?:"
    r"[^\S\n]*:[*_]*"
    rf"(?:[^\S\n]*|[^\S\n]*\n\s*(?={_LONE_LETTER.format(letter='[A-Z]')}))"
    rf"|[^\S\n]+(?={_LONE_LETTER.format(letter='(?-i:[A-Z])')})"
    r")"
    r"[*_(]*([A-Z])(?![A-Z0-9])",
    re.IGNORECASE | re.MULTILINE,
)

# Ordered from most specific → least specific.  Each regex must have
# exactly one capture group containing the answer letter.
_PATTERNS: tuple[re.Pattern[str], ...] = (
    ANSWER_LINE_PATTERN,
    # \boxed{X} or \boxed{\text{X}} — LaTeX (common with thinking-mode models)
    re.compile(r"\\boxed\{(?:\\text\{)?([A-Z])"),
    # (X) — parenthesized letter
    re.compile(r"\(([A-Z])\)"),
    # **X) or **X. — bold-markdown letter
    re.compile(r"\*\*([A-Z])[.)]\s"),
)


def extract_mcq_answer(text: str) -> str | None:
    """Return the MCQ letter from *text*, or ``None``.

    Tries each pattern in priority order and returns the last match from
    the first pattern that fires.
    """
    for pattern in _PATTERNS:
        matches = list(pattern.finditer(text))
        if matches:
            return matches[-1].group(1).upper()
    return None
