"""Offline builder for the corrected FairStress dataset mirrors.

Applies the FairStress paper's own validated question-polarity and
D2/F6-F9 case-content corrections (fairstress_corrections.json, next to
this script) directly to the released data, rather than at eval scoring
time, so the eval task (src/olmo_eval/evals/tasks/fairstress.py) can point
straight at a dataset that's already correct -- one documented, re-runnable
build instead of logic duplicated into every consumer.

Two corrections, both GT-item-only:

1. Question-polarity negation: 5 of 300 scenario templates ask a
   negative-outcome question (terminate, revoke, demote, hold liable); the
   `expected_correct` field for these 5 was generated without checking
   question direction. Flips the trailing A/B letter on exactly those 5
   scenarios' GT rows.
2. D2/F6-F9 case-content exclusion: drops rows where a Degree-2 implicit
   correlate overlaps the scenario's own decision criteria, or where an
   F6/F9 pressure sentence's hiring-specific wording reads incoherently
   outside hiring domains.

A 6th scenario template (severance vs. contested termination) is genuinely
ambiguous in which direction its question runs -- no product decision has
been made on the intended framing, so it is deliberately left untouched by
either correction, exactly as the paper's own validation treats it.

This script is intentionally not imported or run by tests.

Usage (needs write access to the target HF repo):

    pip install "datasets>=3.2" huggingface_hub
    hf auth login
    python build_fairstress_corrected.py \
        --source PardisSzah/fairstress-core --target PardisSzah/fairstress-core-corrected
    python build_fairstress_corrected.py \
        --source PardisSzah/fairstress --target PardisSzah/fairstress-corrected --num-proc 8

Add --dry-run to print row counts without pushing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset

_CORRECTIONS = json.loads((Path(__file__).parent / "fairstress_corrections.json").read_text())
POLARITY_FLAGGED_SCENARIOS: frozenset[str] = frozenset(_CORRECTIONS["polarity_flagged_scenarios"])
D2_CONFOUND_EXCLUSIONS: dict[str, frozenset[str]] = {
    scenario_id: frozenset(groups)
    for scenario_id, groups in _CORRECTIONS["d2_confound_exclusions"].items()
}
F6F9_EXCLUDED_INJECTION_IDS: frozenset[str] = frozenset(_CORRECTIONS["f6f9_excluded_injection_ids"])
F6F9_INCOHERENT_DOMAINS: frozenset[str] = frozenset(_CORRECTIONS["f6f9_incoherent_domains"])


def _is_d2_confound_excluded(row: dict[str, Any]) -> bool:
    if row.get("signaling_level") != 2:
        return False
    flagged = D2_CONFOUND_EXCLUSIONS.get(row.get("scenario_id", ""))
    if not flagged:
        return False
    return row.get("minority_group") in flagged or row.get("majority_group") in flagged


def _is_f6f9_domain_excluded(row: dict[str, Any]) -> bool:
    injection_id = (row.get("injection") or {}).get("id")
    if injection_id not in F6F9_EXCLUDED_INJECTION_IDS:
        return False
    return row.get("domain") in F6F9_INCOHERENT_DOMAINS


def is_excluded(row: dict[str, Any]) -> bool:
    return _is_d2_confound_excluded(row) or _is_f6f9_domain_excluded(row)


def _flip_letter(label: str) -> str:
    stripped = label.rstrip()
    trailing = stripped[-1]
    flipped = "B" if trailing == "A" else "A" if trailing == "B" else trailing
    return stripped[:-1] + flipped


def correct_row(row: dict[str, Any]) -> dict[str, Any]:
    if (
        row.get("condition") == "GT"
        and row.get("scenario_id") in POLARITY_FLAGGED_SCENARIOS
        and row.get("expected_correct")
    ):
        row["expected_correct"] = _flip_letter(row["expected_correct"])
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="e.g. PardisSzah/fairstress-core")
    parser.add_argument("--target", required=True, help="e.g. PardisSzah/fairstress-core-corrected")
    parser.add_argument("--num-proc", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ds = load_dataset(args.source, split="train")
    print(f"loaded {args.source}: {len(ds)} rows")

    n_negated = sum(
        1
        for r in ds
        if r.get("condition") == "GT" and r.get("scenario_id") in POLARITY_FLAGGED_SCENARIOS
    )
    n_excluded = sum(1 for r in ds if is_excluded(r))
    print(f"rows to negate: {n_negated}")
    print(f"rows to exclude: {n_excluded}")

    corrected = ds.map(correct_row, num_proc=args.num_proc)
    corrected = corrected.filter(lambda r: not is_excluded(r), num_proc=args.num_proc)
    print(f"corrected dataset: {len(corrected)} rows (was {len(ds)})")

    if args.dry_run:
        print("\nDRY RUN -- not pushing.")
        return

    corrected.push_to_hub(args.target, split="train")
    print(f"\npushed to {args.target}")


if __name__ == "__main__":
    main()
