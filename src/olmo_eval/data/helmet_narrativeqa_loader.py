"""HELMET NarrativeQA (LongQA) data loading.

Ports HELMET's `load_narrativeqa` (https://github.com/princeton-nlp/HELMET,
data.py): a novel or movie script plus a question about it, graded by an LLM
judge rather than string overlap.

HELMET keeps only documents longer than 131072 reference tokens, so every
instance is a genuinely long document truncated down to the target length --
never a short one padded up. Establishing that requires tokenizing candidate
documents, which is the expensive part of loading this task; see
`_iter_long_documents` for how that cost is bounded.
"""

import logging
from typing import Any

from datasets import load_dataset

from olmo_eval.data.helmet_infbench_loader import (
    REFERENCE_TOKENIZER,
    TRUNCATION_POSTFIX,
    _load_reference_tokenizer,
    _truncate_context,
)

logger = logging.getLogger(__name__)

# HELMET keeps documents strictly longer than this many reference tokens.
MIN_DOCUMENT_TOKENS = 131072

NARRATIVEQA_REPO = "deepmind/narrativeqa"
NARRATIVEQA_REVISION = "2e643e7363944af1c33a652d1c87320d0871c4e4"

_USER_TEMPLATE = (
    "You are given a story, which can be either a novel or a movie script, and a "
    "question. Answer the question as concisely as you can, using a single phrase if "
    "possible.\n\n{demo}{context}\n\nQuestion: {question}"
)
_SYSTEM_TEMPLATE = "Answer:"
_DEMO_TEMPLATE = "Question: {question}\nAnswer: {answer}"


def _iter_long_documents(rows, tokenizer, needed: int | None):
    """Yield rows whose document exceeds MIN_DOCUMENT_TOKENS, in order.

    Two shortcuts keep this from tokenizing the whole 10.5k-row test split,
    without changing which rows are selected:

    - A document can't have more tokens than characters, so anything shorter
      than the threshold in characters is rejected without tokenizing. This is
      an exact bound, not a heuristic.
    - NarrativeQA asks many questions about each document, so token counts are
      memoized per document.

    HELMET shuffles before filtering and then takes a prefix, so stopping once
    `needed` rows have been found yields exactly the same instances it would.
    """
    token_counts: dict[str, int] = {}
    found = 0

    for row in rows:
        text = row["document"]["text"]
        if len(text) <= MIN_DOCUMENT_TOKENS:
            continue

        count = token_counts.get(text)
        if count is None:
            count = len(tokenizer(text)["input_ids"])
            token_counts[text] = count
        if count <= MIN_DOCUMENT_TOKENS:
            continue

        yield row
        found += 1
        if needed is not None and found >= needed:
            return


def load_narrativeqa_dataset(
    max_context_tokens: int,
    shots: int = 2,
    max_samples: int | None = None,
    seed: int = 42,
    reference_tokenizer: str = REFERENCE_TOKENIZER,
) -> dict[str, Any]:
    """Load NarrativeQA truncated to a target context length.

    Args:
        max_context_tokens: Token budget for the story, measured with the
            reference tokenizer.
        shots: Number of worked question/answer examples to prepend. The demos
            show no story text, so they cost little against the budget.
        max_samples: Cap on the number of instances. Supplying this is strongly
            recommended -- without it every candidate document in the split has
            to be tokenized.
        seed: Seed for instance shuffling and demo selection.
        reference_tokenizer: Tokenizer used for the length filter and for
            truncation. Changing it changes how much text every model sees.

    Returns:
        Dictionary with `data` (processed records) and the HELMET prompt templates.
    """
    all_data = load_dataset(NARRATIVEQA_REPO, revision=NARRATIVEQA_REVISION)
    tokenizer = _load_reference_tokenizer(reference_tokenizer)

    demo_text = ""
    if shots > 0:
        # HELMET shuffles the demo pool without a seed, so its demos vary run
        # to run; seeding here makes the prompt reproducible.
        demos = all_data["train"].shuffle(seed=seed).select(range(shots))
        rendered = "\n\n".join(
            _DEMO_TEMPLATE.format(question=d["question"]["text"], answer=d["answers"][0]["text"])
            for d in demos
        )
        demo_text = (
            f"For example:\n\n{rendered}\n\n"
            "Now, use the following story to answer the question:\n\n"
        )

    rows = all_data["test"].shuffle(seed=seed)
    separator_length = len(tokenizer(TRUNCATION_POSTFIX)["input_ids"])

    truncation_cache: dict[str, str] = {}
    data: list[dict[str, Any]] = []

    for row in _iter_long_documents(rows, tokenizer, max_samples):
        text = row["document"]["text"]
        if text not in truncation_cache:
            truncation_cache[text] = _truncate_context(
                text, max_context_tokens, tokenizer, separator_length
            )
        data.append(
            {
                "context": truncation_cache[text],
                "question": row["question"]["text"],
                "answer": [a["text"] for a in row["answers"]],
                "demo": demo_text,
            }
        )

    return {
        "data": data,
        "prompt_template": _USER_TEMPLATE + "\n" + _SYSTEM_TEMPLATE,
        "user_template": _USER_TEMPLATE,
        "system_template": _SYSTEM_TEMPLATE,
    }
