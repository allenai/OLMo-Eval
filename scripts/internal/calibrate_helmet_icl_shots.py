"""Measure -- and optionally recalibrate -- HELMET ICL shot counts.

For the ICL tasks, context length is set by how many demonstrations get packed
into the prompt, so each (dataset, length) pair needs a shot count that lands
the rendered prompt on its target token length.

This is a diagnostic, NOT the source of the values in
`olmo_eval/data/helmet_tasks.py`. Those are HELMET's own counts, kept verbatim
so our ICL numbers stay comparable to published HELMET results. HELMET fit
them against a Llama-2-era tokenizer, so under Olmo 3 they render at
~0.78-0.85x nominal; the `ratio` column below quantifies that shortfall for
whatever tokenizer you point it at, which is worth knowing when interpreting
an ICL length sweep.

The `_ICL_SHOTS` block it prints at the end is what you would paste into
helmet_tasks.py *if* you decided to recalibrate -- trading comparability with
published HELMET for prompts that actually hit their nominal lengths. Don't
paste it in without making that call deliberately.

Whichever counts are used, they stay hardcoded rather than resolved at eval
time: every model must see the same prompt for its scores to be comparable, so
the shot count has to be fixed by one reference tokenizer rather than by
whichever model is being evaluated.

Prompt length is very close to linear in the shot count, so rather than a full
binary search this fits a line through two probes, solves for the target, and
refines a few times -- which keeps the expensive large-shot renders to a
handful per cell.

Usage:
    uv run python scripts/internal/calibrate_helmet_icl_shots.py
    uv run python scripts/internal/calibrate_helmet_icl_shots.py --datasets banking77 --sizes 4096
"""

import argparse
import statistics

from transformers import AutoTokenizer

from olmo_eval.data.helmet_icl_loader import ICL_DATASETS, load_icl_dataset
from olmo_eval.data.helmet_tasks import STANDARD_CONTEXT_SIZES

DEFAULT_TOKENIZER = "allenai/Olmo-3-1025-7B"
# Number of instances to average over per measurement. Prompt length varies a
# little between instances (different demos, different question), so a single
# render is a noisy target to calibrate against.
SAMPLES_PER_MEASUREMENT = 3
TOLERANCE = 0.005
MAX_REFINEMENTS = 6


def measure_tokens(tokenizer, dataset: str, shots: int, seed: int) -> tuple[float, float]:
    """Return (mean, stdev) rendered prompt length in tokens for a shot count."""
    loaded = load_icl_dataset(dataset, shots=shots, max_samples=SAMPLES_PER_MEASUREMENT, seed=seed)
    lengths = []
    for record in loaded["data"]:
        prompt = loaded["user_template"].format(**record) + "\n" + loaded["system_template"]
        lengths.append(len(tokenizer(prompt)["input_ids"]))
    stdev = statistics.stdev(lengths) if len(lengths) > 1 else 0.0
    return statistics.fmean(lengths), stdev


def calibrate(tokenizer, dataset: str, target: int, seed: int) -> tuple[int, float, float]:
    """Find the shot count whose mean prompt length is closest to `target`."""
    # two probes to fit tokens ~= intercept + slope * shots
    lo_shots, hi_shots = 32, 128
    lo_tokens, _ = measure_tokens(tokenizer, dataset, lo_shots, seed)
    hi_tokens, _ = measure_tokens(tokenizer, dataset, hi_shots, seed)

    slope = (hi_tokens - lo_tokens) / (hi_shots - lo_shots)
    intercept = lo_tokens - slope * lo_shots

    shots = max(1, round((target - intercept) / slope))
    best = (shots, *measure_tokens(tokenizer, dataset, shots, seed))

    for _ in range(MAX_REFINEMENTS):
        mean, stdev = best[1], best[2]
        if abs(mean - target) / target <= TOLERANCE:
            break
        # re-solve using the observed point, holding the fitted slope
        shots = max(1, shots + round((target - mean) / slope))
        mean, stdev = measure_tokens(tokenizer, dataset, shots, seed)
        if abs(mean - target) < abs(best[1] - target):
            best = (shots, mean, stdev)

    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--datasets", nargs="+", default=sorted(ICL_DATASETS))
    parser.add_argument("--sizes", nargs="+", type=int, default=STANDARD_CONTEXT_SIZES)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    results: dict[str, list[int]] = {}
    print(f"calibrating against {args.tokenizer}\n")
    print(f"{'dataset':14s} {'target':>8s} {'shots':>7s} {'mean tok':>9s} {'sd':>6s} {'ratio':>6s}")

    for dataset in args.datasets:
        results[dataset] = []
        for size in args.sizes:
            shots, mean, stdev = calibrate(tokenizer, dataset, size, args.seed)
            results[dataset].append(shots)
            print(f"{dataset:14s} {size:8d} {shots:7d} {mean:9.0f} {stdev:6.0f} {mean / size:6.2f}")

    print(
        "\n# Olmo-3-calibrated counts. helmet_tasks.py deliberately keeps HELMET's"
        "\n# originals instead, for comparability with published HELMET results --"
        "\n# only paste this in if you are making that trade deliberately."
    )
    print("_ICL_SHOTS = {")
    for dataset, shots in results.items():
        print(f'    "{dataset}": {shots},')
    print("}")


if __name__ == "__main__":
    main()
