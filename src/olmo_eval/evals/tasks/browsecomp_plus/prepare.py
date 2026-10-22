"""Prepare the complete BrowseComp-Plus dataset and both retrieval indexes."""

from __future__ import annotations

import argparse
from pathlib import Path

from olmo_eval.evals.tasks.browsecomp_plus.data import (
    DATASET_FILENAME,
    DATASET_REPO,
    INDEX_REPO,
    RETRIEVERS,
    artifact_home,
    prepare_questions,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, default=artifact_home())
    args = parser.parse_args()
    home = args.home.expanduser().resolve()
    home.mkdir(parents=True, exist_ok=True)
    dataset_path = home / DATASET_FILENAME
    if not dataset_path.exists():
        from datasets import load_dataset

        rows = load_dataset(DATASET_REPO, split="test")
        count = prepare_questions(rows, dataset_path)
        print(f"Prepared {count} records at {dataset_path}", flush=True)
    else:
        print(f"Reusing {dataset_path}", flush=True)
    from huggingface_hub import snapshot_download

    for retriever in RETRIEVERS:
        print(f"Downloading {retriever} index", flush=True)
        snapshot_download(
            INDEX_REPO,
            repo_type="dataset",
            allow_patterns=[f"{retriever}/*"],
            local_dir=home / "indexes",
        )
    print(f"Artifacts ready at {home}", flush=True)


if __name__ == "__main__":
    main()
