"""HELMET-plus data loading utilities.

Loads HELMET (https://github.com/princeton-nlp/HELMET) data from the
ai2-internal `allenai/helmet-plus` dataset on the Hub, which re-hosts
HELMET's pre-generated and pre-retrieved files so consumers can fetch them
per task. The synthetic `json_kv` tiers are regenerated with HELMET's own
generator, calibrated against the Olmo 3 tokenizer.

Downloads are pinned to a fixed revision of the dataset so that results stay
reproducible as the dataset grows.
"""

import json
import logging
import os
import random
from collections.abc import Callable
from typing import Any

import numpy as np
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import disable_progress_bars as disable_hf_hub_progress_bars
from huggingface_hub.utils import silent_tqdm

logger = logging.getLogger(__name__)

HELMET_PLUS_REPO_ID = "allenai/helmet-plus"
HELMET_PLUS_REVISION = "f46550958fafdff9340d4c24c50a98aaccd5d202"


def _disable_helmet_progress_bars() -> None:
    """Avoid HF tqdm `_lock` failures in the helmet-plus download path."""
    disable_hf_hub_progress_bars()


def download_helmet_plus_file(filename: str) -> str:
    """Download a single file from the helmet-plus dataset repo, using the HF cache.

    Args:
        filename: Path within the `allenai/helmet-plus` repo (e.g. "json_kv/manifest.json").

    Returns:
        Local path to the downloaded file.
    """
    _disable_helmet_progress_bars()
    return hf_hub_download(  # ty: ignore[no-matching-overload]
        repo_id=HELMET_PLUS_REPO_ID,
        filename=filename,
        repo_type="dataset",
        revision=HELMET_PLUS_REVISION,
        tqdm_class=silent_tqdm,
    )


def load_json_kv_manifest() -> dict[str, dict[str, Any]]:
    """Download and parse the json_kv long-context manifest.

    The manifest maps length names (e.g. "256k", "2m") to the resolved
    `num_kvs` and data file used to generate that length tier, so the
    exact file path never needs to be hardcoded here.
    """
    manifest_path = download_helmet_plus_file("json_kv/manifest.json")
    with open(manifest_path, encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as err:
                raise ValueError(f"Invalid JSONL in {path}:{line_num}: {err}") from err
            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected JSON object in {path}:{line_num}, got {type(record).__name__}"
                )
            records.append(record)
    return records


def sample_jsonl_rows(path: str, max_samples: int | None, seed: int) -> list[dict[str, Any]]:
    """Load a seeded random sample of rows from a JSONL file without parsing the rest.

    Selection matches permuting the fully loaded file with the same seed and
    taking the first `max_samples` rows, so results are unchanged from a full
    load; only the lines that survive are parsed. With no cap the file loads
    directly.
    """
    if max_samples is None:
        return _load_jsonl(path)

    with open(path, encoding="utf-8") as f:
        nonblank = [bool(line.strip()) for line in f]
    positions = [i for i, present in enumerate(nonblank) if present]
    permutation = np.random.default_rng(seed).permutation(len(positions))
    chosen = {positions[int(idx)]: rank for rank, idx in enumerate(permutation[:max_samples])}

    rows: list[dict[str, Any] | None] = [None] * len(chosen)
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            if line_num in chosen:
                rows[chosen[line_num]] = json.loads(line)
    return [row for row in rows if row is not None]


def sample_jsonl_by_key(
    path: str,
    max_samples: int | None,
    seed: int,
    key: Callable[[dict[str, Any]], Any],
    keep: Callable[[dict[str, Any]], bool] | None = None,
) -> list[dict[str, Any]]:
    """Load rows grouped by a key, sampling keys without holding the whole file parsed.

    Some HELMET files repeat each question once per gold-passage depth, and
    the largest tiers are gigabytes of JSONL that expand severalfold when
    parsed. So: one pass parses rows only long enough to record each line's
    key (and apply `keep`, an optional row filter), the kept keys are sampled,
    and a second pass parses only the lines that survive. Selection is
    identical to sampling after a full load -- same sorted unique key set,
    same RNG draw -- just without the resident memory. With no cap there is
    nothing to skip, so the file loads directly.
    """
    if max_samples is None:
        rows = _load_jsonl(path)
        return [r for r in rows if keep(r)] if keep is not None else rows

    line_keys: list[Any] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                line_keys.append(None)
                continue
            row = json.loads(line)
            if keep is not None and not keep(row):
                line_keys.append(None)
                continue
            line_keys.append(key(row))

    unique = sorted({k for k in line_keys if k is not None})
    kept = set(random.Random(seed).sample(unique, min(max_samples, len(unique))))

    rows = []
    with open(path, encoding="utf-8") as f:
        for line, line_key in zip(f, line_keys, strict=True):
            if line_key in kept:
                rows.append(json.loads(line))
    return rows


# Matches HELMET's own load_json_kv prompt, from
# https://github.com/nelson-liu/lost-in-the-middle/blob/main/src/lost_in_the_middle/prompts/kv_retrieval.prompt
_USER_TEMPLATE = (
    "{context}\n\n"
    "Extract the value corresponding to the specified key in the JSON object below.\n\n"
    "{demos}Key: {question}"
)
_SYSTEM_TEMPLATE = "Corresponding value:"
_DEMO_TEMPLATE = "Key: {key}\nCorresponding value: {value}"


def load_json_kv_dataset(
    length_name: str, shots: int = 2, max_samples: int | None = None, seed: int = 42
) -> dict[str, Any]:
    """Load a helmet-plus json_kv dataset for a specific long-context length tier.

    Args:
        length_name: Manifest key identifying the length tier (e.g. "256k", "2m").
        shots: Number of few-shot key/value demos to prepend to each example.
        max_samples: Maximum number of examples to load (for testing).
        seed: Random seed for sampling.

    Returns:
        Dictionary with `data` (processed records) and the HELMET prompt templates.
    """
    manifest = load_json_kv_manifest()
    if length_name not in manifest:
        raise ValueError(
            f"Unknown helmet-plus json_kv length '{length_name}'. Available: {sorted(manifest)}"
        )

    remote_path = f"json_kv/{os.path.basename(manifest[length_name]['test_file'])}"
    data_path = download_helmet_plus_file(remote_path)

    def process_example(example: dict[str, Any]) -> dict[str, Any]:
        demos = example.get("demos", [])[:shots]
        demo_text = "\n\n".join(
            _DEMO_TEMPLATE.format(key=key, value=value) for key, value in demos
        ) + ("\n\n" if demos else "")
        return {**example, "demos": demo_text}

    data = [process_example(record) for record in sample_jsonl_rows(data_path, max_samples, seed)]

    return {
        "data": data,
        "prompt_template": _USER_TEMPLATE + "\n" + _SYSTEM_TEMPLATE,
        "user_template": _USER_TEMPLATE,
        "system_template": _SYSTEM_TEMPLATE,
    }
