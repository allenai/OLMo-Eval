"""RULER data loading utilities.

Ported from HELMET: https://github.com/princeton-nlp/HELMET
"""

import json
import logging
import os
import re
import tarfile
from typing import Any

import numpy as np
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import disable_progress_bars as disable_hf_hub_progress_bars
from huggingface_hub.utils import silent_tqdm

logger = logging.getLogger(__name__)

RULER_DATA_REPO = "allenai/ruler_data"
RULER_DATA_REVISION = "b2d1d948dc4f6ca37c0135123f3b3afbed741130"


def _disable_ruler_progress_bars() -> None:
    """Avoid HF tqdm `_lock` failures in the RULER download path."""
    disable_hf_hub_progress_bars()


def _hf_datasets_cache() -> str:
    cache_dir = os.environ.get("HF_DATASETS_CACHE")
    if cache_dir:
        return os.path.expanduser(cache_dir)

    hf_home = os.environ.get("HF_HOME")
    if hf_home is None:
        hf_home = os.path.join(os.environ.get("XDG_CACHE_HOME", "~/.cache"), "huggingface")
    return os.path.expanduser(os.path.join(hf_home, "datasets"))


def download_ruler_data() -> str:
    """Download and extract RULER dataset from HuggingFace.

    Returns:
        Path to the extracted RULER data directory.
    """
    _disable_ruler_progress_bars()

    # Keyed by revision, so moving the pin re-extracts rather than reusing an
    # older copy.
    root_dir = os.path.join(_hf_datasets_cache(), f"allenai--RULER-{RULER_DATA_REVISION[:12]}")
    data_dir = os.path.join(root_dir, "data")

    if not os.path.exists(data_dir):
        logger.info(f"Local RULER data not found in {root_dir}, downloading...")
        my_file = hf_hub_download(  # ty: ignore[no-matching-overload]
            repo_id=RULER_DATA_REPO,
            filename="data_100_samples.tgz",
            repo_type="dataset",
            revision=RULER_DATA_REVISION,
            tqdm_class=silent_tqdm,
        )
        os.makedirs(root_dir, exist_ok=True)
        logger.info(f"Extracting RULER data to {root_dir}...")
        with tarfile.open(my_file) as tar:
            tar.extractall(root_dir, filter="data")
        if not os.path.exists(data_dir):
            raise RuntimeError(f"Extraction failed: {data_dir} does not exist")
    else:
        logger.info(f"Using cached RULER data in {root_dir}.")

    return root_dir


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


def get_ruler_templates(task_type: str) -> tuple[str, str, str]:
    """Get prompt templates for a specific RULER task type.

    Args:
        task_type: RULER task identifier (e.g., "niah_s_1", "vt", "cwe")

    Returns:
        Tuple of (user_template, system_template, prompt_template)
    """
    # Based on https://github.com/hsiehjackson/RULER/blob/main/scripts/data/synthetic/constants.py
    # and HELMET implementation

    if "niah_mv" in task_type or "niah_mq" in task_type:
        user_template = (
            "Some special magic {type_needle_v} are hidden within the following text. "
            "Make sure to memorize it. I will quiz you about the {type_needle_v} afterwards.\n"
            "{context}\n"
            "What are all the special magic {type_needle_v} for {query} mentioned in the "
            "provided text?"
        )
        system_template = (
            "The special magic {type_needle_v} for {query} mentioned in the provided text are"
        )
    elif "niah" in task_type:
        user_template = (
            "A special magic {type_needle_v} is hidden within the following text. "
            "Make sure to memorize it. I will quiz you about the {type_needle_v} afterwards.\n"
            "{context}\n"
            "What is the special magic {type_needle_v} for {query} mentioned in the provided text?"
        )
        system_template = (
            "The special magic {type_needle_v} for {query} mentioned in the provided text is"
        )
    elif "vt" in task_type:
        user_template = (
            "{example}Memorize and track the chain(s) of variable assignment hidden in the "
            "following text.\n\n"
            "{context}\n"
            "Question: Find all variables that are assigned the value {query} in the text above."
        )
        system_template = (
            "Answer: According to the chain(s) of variable assignment in the text above, "
            "{num_v} variables are assigned the value {query}, they are:"
        )
    elif "cwe" in task_type:
        user_template = (
            "{example}Below is a numbered list of words. In these words, some appear more "
            "often than others. Memorize the ones that appear most often.\n"
            "{context}\n"
            "Question: What are the 10 most common words in the above list?"
        )
        system_template = "Answer: The top 10 words that appear most often in the list are:"
    elif "fwe" in task_type:
        user_template = (
            "Read the following coded text and track the frequency of each coded word. "
            "Find the three most frequently appeared coded words.\n"
            "{context}\n"
            "Question: Do not provide any explanation. Please ignore the dots '....'. "
            "What are the three most frequently appeared words in the above coded text?"
        )
        system_template = (
            "Answer: According to the coded text above, the three most frequently "
            "appeared words are:"
        )
    elif "qa" in task_type:
        user_template = (
            "Answer the question based on the given documents. "
            "Only give me the answer and do not output any other words.\n\n"
            "The following are given documents.\n\n"
            "{context}\n\n"
            "Answer the question based on the given documents. "
            "Only give me the answer and do not output any other words.\n\n"
            "Question: {question}"
        )
        system_template = "Answer:"
    else:
        raise NotImplementedError(f"Unknown RULER task type: {task_type}")

    prompt_template = user_template + "\n" + system_template
    return user_template, system_template, prompt_template


def load_ruler_dataset(
    task_name: str, data_path: str, max_samples: int | None = None, seed: int = 42
) -> dict[str, Any]:
    """Load RULER dataset for a specific task.

    Args:
        task_name: RULER task name (e.g., "niah_s_1__4096")
        data_path: Path to the JSONL data file
        max_samples: Maximum number of samples to load (for testing)
        seed: Random seed for sampling

    Returns:
        Dictionary containing:
            - data: list of processed records
            - prompt_template: Full prompt template
            - user_template: User message template
            - system_template: System/assistant prefix template
    """
    _disable_ruler_progress_bars()

    # Extract task type from task name (remove context size)
    task_type = re.findall(r"^(.*)__\d+$", task_name)[0]

    # Get templates for this task type
    user_template, system_template, prompt_template = get_ruler_templates(task_type)

    # Process examples to standardize field names
    def process_example(example: dict[str, Any]) -> dict[str, Any]:
        processed = dict(example)
        processed.update(
            {
                "question": (example.get("query") or example.get("question") or ""),
                "example": (
                    example.get("example", "") + "\n\n" if example.get("example", "") != "" else ""
                ),
                "answer": example.get("answer") or example.get("outputs"),
            }
        )
        return processed

    data = [process_example(record) for record in _load_jsonl(data_path)]

    # Sample if requested
    if max_samples is not None:
        permutation = np.random.default_rng(seed).permutation(len(data))
        data = [data[int(idx)] for idx in permutation[: min(len(data), max_samples)]]

    return {
        "data": data,
        "prompt_template": prompt_template,
        "user_template": user_template,
        "system_template": system_template,
    }
