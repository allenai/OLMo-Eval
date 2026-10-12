"""Audit K=1 prediction artifacts; never replace the task's aggregate scorer."""

import argparse
import json
from collections import Counter
from pathlib import Path


def summarize(rows: list[dict], expected: int, cap: int) -> dict:
    """Count item-level diagnostics, preserving unknowns and missing records."""
    if expected < 1 or cap < 1:
        raise ValueError("expected and cap must be positive")
    ids = [row["doc_id"] for row in rows]
    if len(ids) != len(set(ids)) or len(rows) > expected:
        raise ValueError("duplicate or excess prediction records")
    counts = Counter()
    reasons = Counter()
    for row in rows:
        outputs = row.get("model_output") or []
        if len(outputs) > 1:
            raise ValueError("screen diagnostics require K=1")
        if not outputs:
            counts["missing_output"] += 1
            counts["cap_unknown"] += 1
            continue
        output = outputs[0]
        reason = output.get("finish_reason")
        reasons[str(reason)] += 1
        tokens = output.get("num_tokens_all")
        if tokens is None:
            tokens = output.get("completion_tokens")
        if reason == "length" or (tokens is not None and tokens >= cap):
            counts["cap_hit"] += 1
        elif reason is None and tokens is None:
            counts["cap_unknown"] += 1
        text = output.get("text")
        if text is None:
            counts["empty_unknown"] += 1
        elif not text.strip():
            counts["empty"] += 1
        # Candidate code failures are scored answers, not infrastructure errors.
        if output.get("scoring_errors") or reason == "error":
            counts["errored"] += 1
    missing = expected - len(rows)
    return {
        "expected_items": expected,
        "observed_items": len(rows),
        "missing_items": missing,
        **{
            key: counts[key]
            for key in (
                "cap_hit",
                "cap_unknown",
                "empty",
                "empty_unknown",
                "errored",
                "missing_output",
            )
        },
        "finish_reasons": dict(reasons),
        "diagnostics_complete": not (
            missing or counts["cap_unknown"] or counts["empty_unknown"] or counts["missing_output"]
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--expected", type=int, required=True)
    parser.add_argument("--cap", type=int, required=True)
    args = parser.parse_args(argv)
    with args.predictions.open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    print(json.dumps(summarize(rows, args.expected, args.cap), indent=2))


if __name__ == "__main__":
    main()
