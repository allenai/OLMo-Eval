"""Reproduce the OMEGA out manifest from immutable Hugging Face revisions."""

import argparse
import hashlib
import re
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

from olmo_eval.evals.tasks.omega_500 import OMEGA_500_REVISION
from olmo_eval.evals.tasks.omega_500_out import OMEGA_EXPLORATIVE_REVISION

FAMILY_ALIASES = {
    "geometry_rotation": "geometry_polygon_rotation",
    "logic_puzzles_blocked_grid": "logic_gridworld_blocked",
    "logic_puzzles_grid_knight": "logic_gridworld_knight_move",
    "logic_puzzles_grid_rook": "logic_gridworld_rookmove",
    "logic_puzzles_zebralogic": "logic_zebralogic",
    "number_theory_digit_sum": "numbertheory_qr_sum",
    "number_theory_prime_mod": "numbertheory_lte_qr",
    "number_theory_triple_count": "numbertheory_ordered_lte",
}


def freeze_manifest(output: Path, omega_revision: str, explorative_revision: str) -> None:
    """Select family-matched IDs from pinned snapshots and record their provenance."""
    for revision in (omega_revision, explorative_revision):
        if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
            raise ValueError("Revisions must be immutable 40-character commit hashes")
    anchor_root = Path(
        snapshot_download(
            "allenai/omega-500",
            repo_type="dataset",
            revision=omega_revision,
            allow_patterns=["train-00000-of-00001.parquet"],
        )
    )
    explorative_root = Path(
        snapshot_download(
            "allenai/omega-explorative",
            repo_type="dataset",
            revision=explorative_revision,
            allow_patterns=["*/test_out-00000-of-00001.parquet"],
        )
    )
    counts = Counter(
        r["family"] for r in pq.read_table(anchor_root / "train-00000-of-00001.parquet").to_pylist()
    )
    text = f'''"""Frozen OMEGA test_out dev IDs, stratified to match omega-500 families.

Revisions: omega-500 {omega_revision};
omega-explorative {explorative_revision}.
Within each mapped family, select the required count by ascending SHA256
of ``olmo35-omega-out-v1:<id>``. No answers or model results enter selection.
The alias map reconciles the datasets' family naming; the number-theory
families map by task definition (digit constraints, prime powers, tuple counts).
"""

'''
    text += "FAMILY_ALIASES = " + repr(FAMILY_ALIASES) + "\n\n"
    text += "FAMILY_COUNTS = " + repr(dict(sorted(counts.items()))) + "\n\n"
    text += "IDS_BY_CONFIG = {\n"
    for family, n in sorted(counts.items()):
        config = FAMILY_ALIASES.get(family, family)
        rows = pq.read_table(
            explorative_root / config / "test_out-00000-of-00001.parquet"
        ).to_pylist()
        ids = sorted(r["id"] for r in rows)
        if len(ids) != len(set(ids)):
            raise ValueError(f"{config} has duplicate test_out IDs")
        if len(ids) < n:
            raise ValueError(f"{config} has {len(ids)} test_out IDs; {n} are required")
        chosen = sorted(
            sorted(
                ids, key=lambda x: hashlib.sha256(("olmo35-omega-out-v1:" + x).encode()).hexdigest()
            )[:n]
        )
        text += f'    "{config}": (\n' + "".join(f'        "{i}",\n' for i in chosen) + "    ),\n"
    text += "}\n"
    output.write_text(text)
    print("Frozen", sum(counts.values()), "IDs across", len(counts), "families")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Output Python manifest; format with ruff")
    parser.add_argument("--omega-revision", default=OMEGA_500_REVISION)
    parser.add_argument("--explorative-revision", default=OMEGA_EXPLORATIVE_REVISION)
    args = parser.parse_args(argv)
    freeze_manifest(args.output, args.omega_revision, args.explorative_revision)


if __name__ == "__main__":
    main()
