"""HELMET-plus ICL (in-context learning) data loading.

Ports HELMET's `load_icl` (https://github.com/princeton-nlp/HELMET, data.py):
each instance presents a long list of labelled examples and asks the model to
label one more. Context length is controlled by the *number of demonstrations*
rather than by truncating a document, so unlike the recall tasks there is no
generated data to download -- the source datasets come straight from the Hub
and the length tier only sets how many shots get packed into the prompt.

Labels are mapped to shuffled integers rather than their natural names, which
is HELMET's default: it forces the model to learn the mapping from context
instead of leaning on label semantics it already knows.
"""

import csv
import hashlib
import math
import random
import re
from typing import Any

from datasets import ClassLabel, Dataset, DatasetDict, DownloadManager, load_dataset

# Source datasets, one entry per HELMET ICL task.
#
# HELMET loads several of these with `trust_remote_code=True`, but `datasets`
# dropped script-based loading in 4.0, so three of the five resolve to another
# source here: CogComp/trec via the Hub's auto-converted `refs/convert/parquet`
# branch (byte-identical content), banking77 via `legacy-datasets/banking77`,
# since PolyAI/banking77 hosts only the loading script and no data, and nlu
# from the CSV its loading script read (see `_load_nlu_csv`), since the Hub
# copy is no longer available. Split sizes and label counts were verified
# against the values HELMET hardcodes.
#
# Every source is pinned to a commit. For trec that is a commit on the
# `refs/convert/parquet` branch, which the Hub regenerates; if it is ever
# garbage-collected the load fails loudly rather than silently drifting.
ICL_DATASETS: dict[str, dict[str, Any]] = {
    "trec_coarse": {
        "path": "CogComp/trec",
        "revision": "65752bf53af25bc935a0dce92fb5b6c930728450",
        "train_split": "train",
        "test_split": "test",
        "text_field": "text",
        "label_field": "coarse_label",
        "num_labels": 6,
    },
    "trec_fine": {
        "path": "CogComp/trec",
        "revision": "65752bf53af25bc935a0dce92fb5b6c930728450",
        "train_split": "train",
        "test_split": "test",
        "text_field": "text",
        "label_field": "fine_label",
        "num_labels": 50,
    },
    "banking77": {
        "path": "legacy-datasets/banking77",
        "revision": "f54121560de48f2852f90be299010d1d6dc612ec",
        "train_split": "train",
        "test_split": "test",
        "text_field": "text",
        "label_field": "label",
        "num_labels": 77,
    },
    "clinic150": {
        "path": "clinc/clinc_oos",
        "revision": "155b9c710419136e17307b80d0a13e68cd46b4ec",
        "name": "plus",
        "train_split": "train",
        "test_split": "validation",
        "text_field": "text",
        "label_field": "intent",
        "num_labels": 151,
    },
    "nlu": {
        "csv_url": (
            "https://raw.githubusercontent.com/xliuhw/NLU-Evaluation-Data/"
            "6e495bf0d371d18b611398f205bf53ddb9b1ad5d/AnnotatedData/"
            "NLU-Data-Home-Domain-Annotated-All.csv"
        ),
        # nlu ships a single split; HELMET carves out a test set with a
        # seeded 90/10 train_test_split.
        "split_from_train": 0.1,
        "train_split": "train",
        "text_field": "text",
        "label_field": "label",
        "num_labels": 68,
    },
}

_ITEM_TEMPLATE = "{text}\nlabel: {label}"
# The doubled braces are intentional and load-bearing: they survive .format()
# as a literal "label: {label}" in the instruction, matching HELMET verbatim.
_USER_TEMPLATE = (
    "Use the provided mapping from the text to label to assign a label to the text. "
    'Only output "label: {{label}}" and nothing else. \n\n{context}\n\n{question}'
)
_SYSTEM_TEMPLATE = "label:"


def _balance_labels(
    records: list[dict[str, Any]], label_field: str, shots: int, seed: int
) -> list[dict[str, Any]]:
    """Sample `shots` demonstrations with (near-)uniform coverage of every label.

    Faithful port of HELMET's inner `balance_labels`. Demonstrations are laid
    out in consecutive rounds, each round holding one example per label in a
    random order, so labels stay interleaved rather than clustered no matter
    where the prompt gets cut off.
    """
    rand = random.Random(seed)

    label_to_samples: dict[Any, list[dict[str, Any]]] = {}
    for record in records:
        label_to_samples.setdefault(record[label_field], []).append(record)

    num_rounds = math.ceil(shots / len(label_to_samples))
    rounds: list[list[dict[str, Any]]] = [[] for _ in range(num_rounds)]

    for samples in label_to_samples.values():
        indices = rand.sample(range(len(samples)), num_rounds % len(samples))
        while len(indices) < num_rounds:
            # sample with replacement when a label has fewer examples than rounds
            indices += rand.sample(
                range(len(samples)), min(num_rounds - len(indices), len(samples))
            )
        for i, idx in enumerate(indices):
            rounds[i].append(samples[idx])

    for round_samples in rounds:
        rand.shuffle(round_samples)

    return [record for round_samples in rounds for record in round_samples][:shots]


# Entity markup in the nlu annotations, e.g. "[time : five am]".
_NLU_ANNOTATION = re.compile(r"\[(.+?)\s+\:+\s(.+?)\]")


def _load_nlu_csv(url: str) -> DatasetDict:
    """Rebuild the nlu_evaluation_data dataset from its source CSV.

    Follows the dataset's original loading script (huggingface/datasets 1.18,
    `nlu_evaluation_data.py`), which HELMET used: rows with a "null"
    annotation are dropped, the text is the annotation with its entity markup
    removed, and the label is `{scenario}_{intent}`, indexed in sorted order.
    Rows keep their file order, which the seeded train/test split depends on.
    """
    path = DownloadManager().download(url)
    texts, labels = [], []
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(
            f, quotechar='"', delimiter=";", quoting=csv.QUOTE_ALL, skipinitialspace=True
        )
        next(reader)
        for row in reader:
            scenario, intent, annotation = row[2], row[3], row[5]
            if annotation == "null":
                continue
            texts.append(_NLU_ANNOTATION.sub(r"\2", annotation))
            labels.append(f"{scenario}_{intent}")

    label_feature = ClassLabel(names=sorted(set(labels)))
    dataset = Dataset.from_dict({"text": texts, "label": labels})
    return DatasetDict({"train": dataset.cast_column("label", label_feature)})


def _load_splits(spec: dict[str, Any], seed: int):
    """Load the train/test splits for one ICL source dataset.

    Returns HuggingFace `Dataset`s rather than lists so callers can use
    `.shuffle(seed=...)`, whose permutation HELMET's instance selection
    depends on.
    """
    load_kwargs: dict[str, Any] = {}
    if "name" in spec:
        load_kwargs["name"] = spec["name"]
    if "revision" in spec:
        load_kwargs["revision"] = spec["revision"]

    if "csv_url" in spec:
        dataset = _load_nlu_csv(spec["csv_url"])
    else:
        dataset = load_dataset(spec["path"], **load_kwargs)

    if "split_from_train" in spec:
        split = dataset[spec["train_split"]].train_test_split(
            test_size=spec["split_from_train"], seed=seed
        )
        return split["train"], split["test"]

    return dataset[spec["train_split"]], dataset[spec["test_split"]]


def load_icl_dataset(
    icl_dataset: str,
    shots: int,
    max_samples: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Load a HELMET ICL dataset at a given number of in-context demonstrations.

    Args:
        icl_dataset: Key into `ICL_DATASETS` (e.g. "banking77").
        shots: Number of demonstrations to pack into each prompt. This is what
            sets the context length -- see `helmet_tasks.py` for the per-length
            shot counts HELMET calibrated.
        max_samples: Cap on the number of test instances.
        seed: Base seed; each instance additionally derives its own seed from
            its text, so demo selection is stable per instance regardless of
            how many instances are evaluated.

    Returns:
        Dictionary with `data` (processed records) and the HELMET prompt templates.
    """
    if icl_dataset not in ICL_DATASETS:
        raise ValueError(
            f"Unknown HELMET ICL dataset '{icl_dataset}'. Available: {sorted(ICL_DATASETS)}"
        )

    spec = ICL_DATASETS[icl_dataset]
    text_field = spec["text_field"]
    label_field = spec["label_field"]
    num_labels = spec["num_labels"]

    train_dataset, test_dataset = _load_splits(spec, seed)
    train_records = train_dataset.to_list()

    if max_samples is not None and len(test_dataset) > max_samples:
        # shuffle first so the balanced subset isn't drawn from the head of the
        # split, then balance so every label is represented in the test set too
        test_records = _balance_labels(
            test_dataset.shuffle(seed=seed).to_list(), label_field, max_samples, seed
        )
    else:
        test_records = test_dataset.to_list()

    def process_example(sample: dict[str, Any]) -> dict[str, Any]:
        # deterministic per-instance seed, so an instance gets the same demos
        # and the same label permutation on every run
        local_seed = (
            int(hashlib.sha256(sample[text_field].encode("utf-8")).hexdigest(), 16) + seed
        ) % 2**31

        demos = _balance_labels(train_records, label_field, shots, local_seed)

        # map each label id to a shuffled integer, so the model has to read the
        # mapping out of the context rather than recognizing the label names
        label_mapping = list(range(num_labels))
        random.Random(local_seed).shuffle(label_mapping)

        context = "\n\n".join(
            _ITEM_TEMPLATE.format(
                text=demo[text_field], label=str(label_mapping[int(demo[label_field])])
            )
            for demo in demos
        )
        return {
            "context": context,
            "question": sample[text_field],
            "answer": str(label_mapping[int(sample[label_field])]),
        }

    data = [process_example(record) for record in test_records]

    return {
        "data": data,
        "prompt_template": _USER_TEMPLATE + "\n" + _SYSTEM_TEMPLATE,
        "user_template": _USER_TEMPLATE,
        "system_template": _SYSTEM_TEMPLATE,
    }
