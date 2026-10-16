"""CC-OCR multi-scene OCR (https://arxiv.org/abs/2412.02210).

The multi-scene track of CC-OCR: scene text, documents and web/UGC images in English and
Chinese, grouped into sub-datasets. The model transcribes all the text in each image; the
reference is compared as a multiset of words (characters for the Chinese sub-datasets), so
reading order does not matter.

Each row of the dataset carries its own prompt, sent verbatim in a single user turn with the
image to an instruction-tuned checkpoint. A stage-1 checkpoint (``-o prompt_templates=none -o
system_prompt_style=style_and_length_v2``) gets the ``textocr:`` tag alone instead, whose
trained answer form (every piece of text, joined by spaces) is exactly what this track scores.
Scoring is the official evaluator's (:mod:`olmo_eval.evals.vision.scoring.cc_ocr`): the primary
``macro_f1`` is the unweighted mean over sub-datasets of each one's mean per-image F1, the track
score the paper reports; ``micro_f1`` and every sub-dataset's score are reported alongside.
Metrics are 0-1 (the paper reports x100).

``cc_ocr_multi_scene_en`` keeps the English sub-datasets (every one without ``zh`` in its
name) and averages over those. Molmo2 models read English only, so it is the default setting
for evaluating them; ``cc_ocr_multi_scene`` runs every sub-dataset, for multilingual models.

The benchmark prescribes no decoding settings; this task decodes greedily. Data is fetched from
the Hub at a pinned revision; set ``CC_OCR_DIR`` to a local copy of the dataset repository to
read it from disk instead.
"""

from __future__ import annotations

import base64
import csv
import functools
import io
import os
import sys
from collections.abc import Iterator
from pathlib import Path

from olmo_eval.common.metrics.base import Metric
from olmo_eval.common.types import Instance, SamplingParams, Split
from olmo_eval.evals.tasks.common import register
from olmo_eval.evals.vision.scoring.cc_ocr import CcOcrDatasetMetric, CcOcrScorer, CcOcrTrackMetric
from olmo_eval.evals.vision.tasks.ocr import TEXTOCR_STYLE, OcrTask

HF_REPO = "wulipc/CC-OCR"
HF_REVISION = "c64517e92179991d509776064174776700cdd5a2"
TRACK = "multi_scene_ocr"

#: The track's sub-datasets (the ``split`` column), in the official index order.
SUBSETS: tuple[str, ...] = (
    "TotalText",
    "IC15",
    "InverseText",
    "Hieragent",
    "zh_scene",
    "FUNSD",
    "CORD",
    "IAM",
    "zh_doc",
    "zh_handwriting",
    "ugc_laion",
    "zh_vertical",
    "zh_dense",
)

#: The English sub-datasets: every one without ``zh`` in its name.
ENGLISH_SUBSETS: tuple[str, ...] = tuple(s for s in SUBSETS if "zh" not in s)

_SCORER = CcOcrScorer()
_MACRO_F1 = CcOcrTrackMetric(name="macro_f1", scorer=_SCORER, kind="macro_f1")


def _metrics(subsets: tuple[str, ...]) -> tuple[Metric, ...]:
    """The track scores (averaged over whichever sub-datasets ran) and one per sub-dataset."""
    return (
        _MACRO_F1,
        CcOcrTrackMetric(name="micro_f1", scorer=_SCORER, kind="micro_f1"),
        *(CcOcrDatasetMetric(name=dataset, scorer=_SCORER, dataset=dataset) for dataset in subsets),
    )


_METRICS = _metrics(SUBSETS)


@functools.lru_cache(maxsize=64)
def _load_tsv(path: str) -> list[dict[str, str]]:
    """Rows of one sub-dataset file; cached because its images are decoded row by row."""
    csv.field_size_limit(sys.maxsize)
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _decode_tsv_image(path: str, row: int):
    """Decode one base64 image cell (module-level so the owning instance stays picklable)."""
    from PIL import Image

    data = base64.b64decode(_load_tsv(path)[row]["image"])
    return Image.open(io.BytesIO(data)).convert("RGB")


def _data_dir() -> Path:
    local = os.environ.get("CC_OCR_DIR")
    if local:
        return Path(local)
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=HF_REPO,
            repo_type="dataset",
            revision=HF_REVISION,
            allow_patterns=[f"{TRACK}/**/*.tsv"],
        )
    )


def _subset_of(stem: str) -> str:
    """The sub-dataset a ``<name>_<count>.tsv`` file holds (``zh_scene_450`` -> ``zh_scene``)."""
    for subset in SUBSETS:
        rest = stem[len(subset) :]
        if stem.startswith(subset) and (not rest or (rest[0] == "_" and rest[1:].isdigit())):
            return subset
    return stem


@register("cc_ocr_multi_scene")
class CcOcrMultiSceneTask(OcrTask):
    dependencies = ["pillow", "huggingface-hub"]
    sampling_params = SamplingParams(temperature=0.0, max_tokens=4096)
    metrics = _METRICS
    primary_metric = _MACRO_F1
    split = Split.TEST
    #: A stage-1 checkpoint is prompted with ``textocr:``; see the module docstring.
    ocr_style = TEXTOCR_STYLE
    #: The sub-datasets this task runs and averages over.
    subsets: tuple[str, ...] = SUBSETS

    def _build_instances(self) -> Iterator[Instance]:
        # Each sub-dataset is one ``<name>_<count>.tsv``; only the ones this task runs are
        # decoded, since the files embed every image.
        paths = {_subset_of(path.stem): path for path in (_data_dir() / TRACK).rglob("*.tsv")}
        if set(paths) != set(SUBSETS):
            raise RuntimeError(
                f"CC-OCR {TRACK} has sub-datasets {sorted(paths)}, expected {sorted(SUBSETS)}"
            )
        for subset in self.subsets:
            for instance in self._tsv_instances(str(paths[subset])):
                if instance.metadata["dataset"] != subset:
                    raise RuntimeError(
                        f"{paths[subset]}: row of sub-dataset {instance.metadata['dataset']!r}"
                    )
                yield instance

    def _tsv_instances(self, tsv_path: str) -> Iterator[Instance]:
        for row_index, row in enumerate(_load_tsv(tsv_path)):
            if row["category"] != TRACK:
                raise RuntimeError(f"{tsv_path}: row of track {row['category']!r} under {TRACK!r}")
            # Image names repeat across sub-datasets; only the pair is unique.
            key = f"{row['split']}/{row['image_name']}"
            yield Instance(
                question=self._question(row["question"]),
                gold_answer=row["answer"],
                metadata={
                    "id": key,
                    "example_id": key,
                    "dataset": row["split"],
                    "image_name": row["image_name"],
                    "answer": row["answer"],
                    "image": functools.partial(_decode_tsv_image, tsv_path, row_index),
                },
            )


@register("cc_ocr_multi_scene_en")
class CcOcrMultiSceneEnglishTask(CcOcrMultiSceneTask):
    """The English sub-datasets of the multi-scene track; ``macro_f1`` is their
    unweighted mean."""

    metrics = _metrics(ENGLISH_SUBSETS)
    subsets = ENGLISH_SUBSETS
