"""Fit the Arena-Hard v2.0 style-control constants from upstream's published judgments.

Rebuilds upstream's battles from the GPT-4.1 judgments of every leaderboard model,
fits the joint Bradley-Terry model with style features that upstream's
``show_result.py --control-features markdown length`` fits, and prints the
``StyleModel`` constants that ``olmo_eval.evals.tasks.arena_hard`` pins.

It then checks the approximation the task makes. For each leaderboard model, the
constants are refit without that model and only the model's own strength is fitted
against them, as the task does for a new model. The script prints that score next
to the model's score in the full joint fit.

Usage:
    uv run python scripts/internal/fit_arena_hard_style_model.py
    uv run python scripts/internal/fit_arena_hard_style_model.py \
        --data-dir <dir with model_answer/ and model_judgment/>
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download

from olmo_eval.evals.tasks.arena_hard import (
    ARENA_HARD_DATA_DIR,
    ARENA_HARD_REPO,
    ARENA_HARD_REVISION,
    CREATIVE_WRITING,
    HARD_PROMPT,
    VERDICT_OUTCOMES,
    ArenaHardCategory,
    StyleCounts,
    StyleModel,
    baseline_answer,
    battle_outcomes,
    fit_win_rate,
    sigmoid,
    style_counts,
    style_features,
)

JUDGE = "gpt-4.1"


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _short_name(model: str) -> str:
    """Upstream drops the organization prefix from model names."""
    return model.split("/")[-1]


def _upstream_counts(metadata: dict) -> StyleCounts:
    return StyleCounts(
        tokens=int(metadata["token_len"]),
        headers=sum(metadata["header_count"].values()),
        lists=sum(metadata["list_count"].values()),
        bold=sum(metadata["bold_count"].values()),
    )


def load_style(data_dir: Path) -> dict[str, dict[str, StyleCounts]]:
    """Upstream's stored style counts, keyed by short model name and question id."""
    style: dict[str, dict[str, StyleCounts]] = {}
    for path in sorted((data_dir / "model_answer").glob("*.jsonl")):
        records = _read_jsonl(path)
        style[_short_name(records[0]["model"])] = {
            record["uid"]: _upstream_counts(record["metadata"]) for record in records
        }
    return style


def load_battles(
    data_dir: Path, category: ArenaHardCategory, style: dict[str, dict[str, StyleCounts]]
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Rebuild upstream's weighted battles: model, style features, and outcome per row."""
    models: list[str] = []
    features: list[tuple[float, ...]] = []
    outcomes: list[float] = []
    skipped: Counter[str] = Counter()
    baseline_style = style[category.baseline_model]
    for path in sorted((data_dir / "model_judgment" / JUDGE).glob("*.jsonl")):
        for record in _read_jsonl(path):
            if record["category"] != category.name:
                continue
            games = record["games"]
            verdicts = [game["score"] if game else None for game in games]
            if any(verdict is None for verdict in verdicts):
                skipped["no verdict"] += 1
                continue
            if any(verdict not in VERDICT_OUTCOMES for verdict in verdicts):
                skipped["unknown verdict"] += 1
                continue
            model = _short_name(record["model"])
            row_features = style_features(
                style[model][record["uid"]], baseline_style[record["uid"]]
            )
            for outcome in battle_outcomes(verdicts[0], verdicts[1]):
                models.append(model)
                features.append(row_features)
                outcomes.append(outcome)
    if skipped:
        print(f"  skipped judgments: {dict(skipped)}")
    return models, np.array(features), np.array(outcomes)


def fit_joint(
    models: list[str], features: np.ndarray, outcomes: np.ndarray
) -> tuple[dict[str, float], StyleModel]:
    """Fit upstream's Bradley-Terry model with style features by Newton's method.

    Each battle's logit is the model's strength (the baseline's is fixed at 0) plus
    the style coefficients times the standardized features. Returns each model's win
    rate against the baseline and the fitted style model.
    """
    names = sorted(set(models))
    index = {name: column for column, name in enumerate(names)}
    means = features.mean(axis=0)
    stds = features.std(axis=0, ddof=1)
    design = np.zeros((len(models), len(names) + features.shape[1]))
    design[np.arange(len(models)), [index[model] for model in models]] = 1.0
    design[:, len(names) :] = (features - means) / stds

    weights = np.zeros(design.shape[1])
    for _ in range(100):
        probabilities = np.exp(-np.logaddexp(0.0, -(design @ weights)))
        gradient = design.T @ (probabilities - outcomes)
        hessian = (design * (probabilities * (1.0 - probabilities))[:, None]).T @ design
        step = np.linalg.solve(hessian, gradient)
        weights -= step
        if np.max(np.abs(step)) < 1e-10:
            break

    win_rates = {name: sigmoid(float(weights[index[name]])) for name in names}
    style_model = StyleModel(
        means=tuple(float(value) for value in means),
        stds=tuple(float(value) for value in stds),
        coefs=tuple(float(value) for value in weights[len(names) :]),
    )
    return win_rates, style_model


def _format_style_model(style_model: StyleModel) -> str:
    """Render the constants as the Python literal the task module pins."""

    def values(numbers: tuple[float, ...]) -> str:
        return "(" + ", ".join(f"{number:.6g}" for number in numbers) + ")"

    return (
        "StyleModel(\n"
        f"    means={values(style_model.means)},\n"
        f"    stds={values(style_model.stds)},\n"
        f"    coefs={values(style_model.coefs)},\n"
        ")"
    )


def check_style_counts(data_dir: Path, category: ArenaHardCategory) -> None:
    """Compare this package's style counts with upstream's stored ones on the baseline."""
    questions = {
        record["uid"]: record["category"] for record in _read_jsonl(data_dir / "question.jsonl")
    }
    answers_file = Path(category.baseline_answers_file).relative_to(ARENA_HARD_DATA_DIR)
    records = _read_jsonl(data_dir / answers_file)
    mismatches = 0
    checked = 0
    for record in records:
        if questions.get(record["uid"]) != category.name:
            continue
        checked += 1
        ours = style_counts(baseline_answer(record))
        if ours != _upstream_counts(record["metadata"]):
            mismatches += 1
    print(f"  style counts differing from upstream: {mismatches}/{checked} baseline answers")


def report(data_dir: Path, category: ArenaHardCategory) -> None:
    print(f"\n## {category.name} (baseline {category.baseline_model}, judge {JUDGE})")
    check_style_counts(data_dir, category)
    style = load_style(data_dir)
    models, features, outcomes = load_battles(data_dir, category, style)
    joint, style_model = fit_joint(models, features, outcomes)

    print(f"{'model':<44}{'raw':>7}{'joint SC':>10}{'single SC':>11}{'diff':>7}")
    model_array = np.array(models)
    rows = []
    for name in sorted(joint, key=joint.get, reverse=True):
        mine = model_array == name
        others = ~mine
        _, held_out = fit_joint(list(model_array[others]), features[others], outcomes[others])
        offsets = [held_out.offset(row) for row in features[mine]]
        single = fit_win_rate(list(outcomes[mine]), offsets)
        raw = float(outcomes[mine].mean())
        rows.append((name, raw, joint[name], single))
        print(
            f"{name:<44}{raw * 100:>7.1f}{joint[name] * 100:>10.1f}"
            f"{single * 100:>11.1f}{(single - joint[name]) * 100:>7.1f}"
        )
    diffs = [abs(single - joint_rate) * 100 for _, _, joint_rate, single in rows]
    print(f"  max |single - joint| = {max(diffs):.2f} points, mean = {np.mean(diffs):.2f}")
    print(f"\n{category.name.upper()}_STYLE_MODEL = {_format_style_model(style_model)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Directory holding question.jsonl, model_answer/ and model_judgment/; "
        "downloaded from the pinned Hub revision when omitted.",
    )
    args = parser.parse_args()
    data_dir = args.data_dir
    if data_dir is None:
        root = snapshot_download(
            ARENA_HARD_REPO,
            repo_type="dataset",
            revision=ARENA_HARD_REVISION,
            allow_patterns=[
                f"{ARENA_HARD_DATA_DIR}/question.jsonl",
                f"{ARENA_HARD_DATA_DIR}/model_answer/*.jsonl",
                f"{ARENA_HARD_DATA_DIR}/model_judgment/{JUDGE}/*.jsonl",
            ],
        )
        data_dir = Path(root) / ARENA_HARD_DATA_DIR
    for category in (HARD_PROMPT, CREATIVE_WRITING):
        report(data_dir, category)


if __name__ == "__main__":
    main()
