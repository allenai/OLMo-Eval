"""HELMET Multi-LexSum (summarization) data loading.

Ports HELMET's `load_multi_lexsum` (https://github.com/princeton-nlp/HELMET,
data.py): the filings from a civil rights lawsuit, summarized in a paragraph
and graded by an LLM judge.

Reads the data from `allenai/helmet-plus` rather than calling
`load_dataset("allenai/multi_lexsum")`, which no longer works: `datasets` 4.0
dropped script-based loading and the Hub's parquet conversion has no
`v20230518` config. The hosted `multi_lexsum_val.jsonl` is the full validation
split -- sources, both summary lengths, and the pre-extracted key points the
judge needs -- so nothing is lost by bypassing the broken path, and pinning the
data makes runs reproducible besides.
"""

import json
import logging
from typing import Any

from olmo_eval.data.helmet_infbench_loader import (
    REFERENCE_TOKENIZER,
    TRUNCATION_POSTFIX,
    _load_reference_tokenizer,
    _truncate_context,
)
from olmo_eval.data.helmet_loader import download_helmet_plus_file

logger = logging.getLogger(__name__)

MULTI_LEXSUM_FILE = "multi_lexsum/multi_lexsum_val.jsonl"

_USER_TEMPLATE = (
    "You are given the legal documents in a civil rights lawsuit, and you are tasked "
    "to summarize the case. Write a concise summary of one paragraph (200 to 250 "
    "words). The summary should contain a short description of the background, the "
    "parties involved, and the outcomes of the case.\n\n{demo}Legal documents:\n"
    "{context}\n\nNow please summarize the case."
)
_SYSTEM_TEMPLATE = "Summary:"
_DEMO_TEMPLATE = "Summary: {summary}"


def load_multi_lexsum_dataset(
    max_context_tokens: int,
    shots: int = 2,
    max_samples: int | None = None,
    seed: int = 42,
    reference_tokenizer: str = REFERENCE_TOKENIZER,
) -> dict[str, Any]:
    """Load Multi-LexSum truncated to a target context length.

    Args:
        max_context_tokens: Token budget for the documents, measured with the
            reference tokenizer. HELMET reserves more of the tier here than for
            LongQA (300 tokens rather than 200), so this is computed by the
            caller from the task config.
        shots: Number of example summaries to prepend. They carry no source
            documents, so they cost little against the budget.
        max_samples: Cap on the number of instances.
        seed: Seed for instance sampling and demo selection.
        reference_tokenizer: Tokenizer used for truncation.

    Returns:
        Dictionary with `data` (processed records) and the HELMET prompt templates.
    """
    import random

    path = download_helmet_plus_file(MULTI_LEXSUM_FILE)
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    # HELMET keeps only cases that actually have a short gold summary
    rows = [r for r in rows if r.get("summary/short")]

    demo_text = ""
    if shots > 0:
        # upstream draws demos from the train split; only the validation split
        # is published, so they come from cases held out of the sample below
        pool = rows[-shots:] if len(rows) > shots else rows
        rendered = "\n\n".join(_DEMO_TEMPLATE.format(summary=d["summary/short"]) for d in pool)
        demo_text = (
            f"Example summaries:\n\n{rendered}\n\n"
            "Now, write a summary of the following legal documents.\n"
        )
        demo_ids = {d["id"] for d in pool}
        rows = [r for r in rows if r["id"] not in demo_ids]

    if max_samples is not None and len(rows) > max_samples:
        rng = random.Random(seed)
        shuffled = list(rows)
        rng.shuffle(shuffled)
        rows = shuffled[:max_samples]

    tokenizer = _load_reference_tokenizer(reference_tokenizer)
    separator_length = len(tokenizer(TRUNCATION_POSTFIX)["input_ids"])

    data = []
    for row in rows:
        context = "\n\n".join(row["sources"])
        data.append(
            {
                "context": _truncate_context(
                    context, max_context_tokens, tokenizer, separator_length
                ),
                "question": "",
                "demo": demo_text,
                "answer": [row["summary/short"]],
                "keypoints": row.get("summary/short_keypoints") or [],
                # the precision rubric grades the model's sentences against the
                # expert *long* summary, not the short one it is asked to match
                "expert_summary": row.get("summary/long") or "",
            }
        )

    missing = sum(1 for row in data if not row["keypoints"])
    if missing:
        # the judge cannot grade these; each scores 0.0 and is flagged in its
        # judge_result
        logger.warning("%d Multi-LexSum rows have no key points", missing)

    return {
        "data": data,
        "prompt_template": _USER_TEMPLATE + "\n\n" + _SYSTEM_TEMPLATE,
        "user_template": _USER_TEMPLATE,
        "system_template": _SYSTEM_TEMPLATE,
    }
