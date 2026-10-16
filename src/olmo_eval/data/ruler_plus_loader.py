"""HuggingFace-backed data loading for the RULER-plus (ruler-plus) dataset.

Unlike ruler_loader.py's tarball-based RULER release, ruler-plus is generated
with https://github.com/jopetty/RULER (scripts/generate-data.sh, at commit
7512d2416ac7a966bd817ae89a443001aa442f20) and published to the
allenai/ruler-plus HuggingFace dataset repo. Each (context_size, task)
condition is its own JSONL shard, some of which are many GB (contexts run up
to 2097152 tokens), so only the single shard a task actually needs is
downloaded (rather than the whole repo), and it is read with reservoir
sampling so peak memory scales with the requested sample count rather than
the shard's full size.

allenai/ruler-plus is private to Ai2, so these tasks are internal-only: the
haystacks include excerpts of Paul Graham essays, which we can't relicense for
redistribution. Running them needs an HF_TOKEN with access to the repo, or a
locally generated copy via OLMO_EVAL_RULER_PLUS_DATA_DIR.
"""

import json
import logging
import os
from typing import Any, NamedTuple

import numpy as np
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import RepositoryNotFoundError
from huggingface_hub.utils import disable_progress_bars as disable_hf_hub_progress_bars

logger = logging.getLogger(__name__)

RULER_PLUS_REPO_ID = "allenai/ruler-plus"
# Pinned so that re-uploading a shard can't change results under the same task
# name; publish changed data under a new revision and a versioned task name.
RULER_PLUS_REVISION = "b1c317b593ffe0ea34ef3c045f5bb7842f797cc0"
RULER_PLUS_DATA_DIR_ENV_VAR = "OLMO_EVAL_RULER_PLUS_DATA_DIR"

# Field each loaded record carries with its position in the shard. The
# generator's own `index` field is not a sample id (for NIAH it is the
# needle's character offset), so this is what identifies a record.
ROW_FIELD = "_row"


class RulerPlusDataFile(NamedTuple):
    """A local shard path, plus where its contents came from."""

    path: str
    source: str


def get_ruler_plus_data_file(relative_path: str) -> RulerPlusDataFile:
    """Return the local path to a single ruler-plus validation shard.

    ``relative_path`` is relative to the dataset root, e.g.
    ``"4096/niah_single_1/validation.jsonl"`` (see ``data_template`` in
    ruler_plus_tasks.py). Downloads (and caches) just that one shard from the
    allenai/ruler-plus HuggingFace dataset repo at the pinned revision; set
    OLMO_EVAL_RULER_PLUS_DATA_DIR to read a local copy with the same layout
    instead. The task hash doesn't cover the data, so ``source`` records which
    of the two was read.
    """
    override_dir = os.environ.get(RULER_PLUS_DATA_DIR_ENV_VAR)
    if override_dir:
        file_path = os.path.join(os.path.expanduser(override_dir), relative_path)
        if not os.path.isfile(file_path):
            raise FileNotFoundError(
                f"RULER-plus data file not found: {file_path}. Check {RULER_PLUS_DATA_DIR_ENV_VAR}."
            )
        logger.warning(
            f"Reading RULER-plus data from {file_path} ({RULER_PLUS_DATA_DIR_ENV_VAR}) "
            f"rather than {RULER_PLUS_REPO_ID}@{RULER_PLUS_REVISION}"
        )
        return RulerPlusDataFile(path=file_path, source=f"local:{os.path.abspath(file_path)}")

    disable_hf_hub_progress_bars()
    filename = f"data/{relative_path}"
    logger.info(f"Fetching {filename} from {RULER_PLUS_REPO_ID}@{RULER_PLUS_REVISION}...")
    try:
        path = hf_hub_download(
            repo_id=RULER_PLUS_REPO_ID,
            repo_type="dataset",
            filename=filename,
            revision=RULER_PLUS_REVISION,
        )
    except RepositoryNotFoundError as e:
        # The Hub reports a private repo the token can't read as not found
        raise PermissionError(
            f"Could not read {RULER_PLUS_REPO_ID}, which is private to Ai2 (ruler-plus tasks "
            f"are internal-only). Set HF_TOKEN to a token with access, or point "
            f"{RULER_PLUS_DATA_DIR_ENV_VAR} at a locally generated copy."
        ) from e
    return RulerPlusDataFile(
        path=path, source=f"hf:{RULER_PLUS_REPO_ID}@{RULER_PLUS_REVISION}/{filename}"
    )


def _parse_row(line: str, row: int) -> dict[str, Any]:
    record = json.loads(line)
    record[ROW_FIELD] = row
    return record


def load_ruler_plus_shard(
    data_path: str, max_samples: int | None = None, seed: int = 42
) -> list[dict[str, Any]]:
    """Load a ruler-plus JSONL shard, reservoir-sampling up to ``max_samples`` records.

    Streams the shard line-by-line using reservoir sampling (Algorithm R) rather
    than reading the full file into memory before subsampling, so peak memory
    stays proportional to ``max_samples`` instead of the shard's full size.

    Each record gets its 0-based position among the shard's non-blank lines
    under ``ROW_FIELD``, taken before sampling so that it stays the same
    whatever ``max_samples`` is.
    """
    rng = np.random.default_rng(seed)
    reservoir: list[dict[str, Any]] = []
    with open(data_path, encoding="utf-8") as f:
        seen = 0
        for line in f:
            line = line.strip()
            if not line:
                continue
            if max_samples is None or seen < max_samples:
                reservoir.append(_parse_row(line, seen))
            else:
                j = int(rng.integers(0, seen + 1))
                if j < max_samples:
                    reservoir[j] = _parse_row(line, seen)
            seen += 1

    return reservoir
