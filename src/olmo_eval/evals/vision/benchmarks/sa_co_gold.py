"""SACoGold — image pointing benchmark (test split).

Mirrors mm_olmo's ``SACoGoldSubset`` (``sa-co-gold-subset-v3``) + ``SACoGoldPointEvaluator``:
loads the prepared subset at ``torch_datasets/sa-co-gold/molmo-subset-v1.json`` (4994 examples) and
asks ``"Point to (the) <text_input>."`` per example. Each example has **three** annotators' instance
masks (``annotations.{a,b,c}``) and a per-example ``weight``; scoring is point-in-mask
precision/recall/f1 taken as the best over annotators, with **weighted** primaries (matching the
balanced subset) and all/single/low/med/high-frequency simple-mean buckets.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Iterator

from olmo_eval.common.types import Instance, SamplingParams, Split
from olmo_eval.evals.tasks.common import register
from olmo_eval.evals.vision.data.paths import rebase_data_path, torch_datasets_dir
from olmo_eval.evals.vision.scoring.pointing import PointingScorer
from olmo_eval.evals.vision.tasks.pointing import (
    ModelPromptPointingTask,
    PointingTask,
    StylePrefixMixin,
    pointing_metrics,
    presence_metrics,
)

_SCORER = PointingScorer()
_BUCKETS = ("all", "single", "low_freq", "med_freq", "high_freq")
_METRICS = pointing_metrics(_SCORER, buckets=_BUCKETS, weighted_primary=True)

#: The full gold test set carries no per-example weight, and its abstention behavior is
#: reported alongside the point metrics.
_POINT_METRICS = _METRICS + presence_metrics(_SCORER)


def _format_query(text_input: str) -> str:
    """mm_olmo ``SACoGold._format_query``: prepend "the" unless the phrase already starts a/the."""
    parts = text_input.split()
    if parts and parts[0].lower() in ("a", "the"):
        return f"Point to {text_input}."
    return f"Point to the {text_input}."


def _strip_article(text_input: str) -> str:
    """mm_olmo's ``_mp`` label: drop a leading "a"/"an"/"the"."""
    parts = text_input.split()
    if parts and parts[0].lower() in ("a", "an", "the"):
        return " ".join(parts[1:])
    return text_input


def _subset_instances(filename: str, question_for: Callable[[str, int], str]) -> Iterator[Instance]:
    """Yield the prepared-subset examples in file order, prompted by ``question_for``."""
    path = torch_datasets_dir() / "sa-co-gold" / filename
    with open(path) as f:
        data = json.load(f)
    for idx, ex in enumerate(data):
        md = ex["metadata"]
        ann = md["annotations"]
        annotators = [[a["segmentation"] for a in ann[k]] for k in ("a", "b", "c")]
        yield Instance(
            question=question_for(ex["text_input"], idx),
            gold_answer=None,
            metadata={
                "pointing_annotators": annotators,
                "image_size": (int(md["width"]), int(md["height"])),
                "image_path": rebase_data_path(ex["image"]),
                "weight": float(md["weight"]),
                "example_id": md["id"],
                "subset": md["subset"],
                "label": ex["text_input"],
            },
        )


@register("sa_co_gold_subset")
class SaCoGoldSubsetTask(StylePrefixMixin, PointingTask):
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1024)
    metrics = _METRICS
    primary_metric = _METRICS[2]  # weighted f1
    split = Split.TEST

    def _build_instances(self) -> Iterator[Instance]:
        return _subset_instances(
            "molmo-subset-v1.json",
            lambda text, _idx: self.apply_family_prefix(_format_query(text)),
        )


@register("sa_co_gold_subset_mp")
class SaCoGoldSubsetMpTask(ModelPromptPointingTask):
    """``sa_co_gold_subset_mp:test`` — mm_olmo's ``SACoGoldSubset(prompt_kind="model")``.

    Reads ``molmo-subset-v2.json`` (the file mm_olmo's ``SACoGoldSubset`` uses), not the
    v1 file the non-mp task above reads.
    """

    sampling_params = SamplingParams(temperature=0.0, max_tokens=1024)
    metrics = _METRICS
    primary_metric = _METRICS[2]  # weighted f1
    split = Split.TEST

    def _build_instances(self) -> Iterator[Instance]:
        return _subset_instances(
            "molmo-subset-v2.json",
            lambda text, idx: self.format_query(_strip_article(text), idx),
        )


#: mm_olmo ``SACoGold.SUBSETS``; the per-subset sample is drawn in this order.
SACO_SUBSETS: tuple[str, ...] = (
    "attributes",
    "crowded",
    "fg_food",
    "fg_sports_equipment",
    "metaclip",
    "sa1b",
    "wiki_common",
)

#: Seed mm_olmo shuffles each subset with before taking the first ``sample`` examples.
_SACO_SAMPLE_SEED = 68181


def _load_gold_examples(sample: int | None) -> list[dict]:
    """Build mm_olmo's ``SACoGold`` example list from the per-annotator gold annotations.

    Keeps only images every annotator marked instance-exhaustive, preserves the
    first-seen image order while scanning annotators a, b, c, and takes ``sample``
    examples **per subset** after a seeded shuffle — so the result is 7x ``sample``.
    """
    import numpy as np

    home = torch_datasets_dir() / "sa-co-gold"
    image_roots = {"sa1b": home / "sa1b-images", "metaclip": home / "metaclip-images"}

    examples: list[dict] = []
    for subset in SACO_SUBSETS:
        images: dict[object, list[dict]] = defaultdict(list)
        annotations: dict[object, dict[str, list]] = defaultdict(
            lambda: {"a": [], "b": [], "c": []}
        )
        for part in ("a", "b", "c"):
            path = home / "gt-annotations" / f"gold_{subset}_merged_{part}_release_test.json"
            with open(path) as f:
                data = json.load(f)
            for image in data["images"]:
                images[image["id"]].append(image)
            for annotation in data["annotations"]:
                annotations[annotation["image_id"]][part].append(annotation)

        subset_examples: list[dict] = []
        for image_id, image_datas in images.items():
            if any(not x["is_instance_exhaustive"] for x in image_datas):
                continue
            file_name = image_datas[0]["file_name"]
            root = image_roots["sa1b" if file_name.startswith("sa_") else "metaclip"]
            subset_examples.append(
                {
                    "image": str(root / file_name),
                    "text_input": image_datas[0]["text_input"],
                    "metadata": dict(
                        image_datas[0], annotations=annotations[image_id], subset=subset
                    ),
                }
            )

        if sample is not None and sample < len(subset_examples):
            # Shuffling an index array draws the same permutation as shuffling the list
            # in place, which is what mm_olmo does.
            order = np.arange(len(subset_examples))
            np.random.RandomState(_SACO_SAMPLE_SEED).shuffle(order)
            subset_examples = [subset_examples[i] for i in order[:sample]]
        examples.extend(subset_examples)
    return examples


class SaCoGoldPointMpBase(ModelPromptPointingTask):
    """The gold test set scored as pointing, with mm_olmo's ``_mp`` prompt.

    Unlike the prepared subsets this carries no per-example weight (so the primaries are
    plain means) and is dominated by absent phrases, which is what ``is_absent_acc`` and
    ``is_present_acc`` measure. Subclasses set how many examples to draw per subset.
    """

    sampling_params = SamplingParams(temperature=0.0, max_tokens=2048)
    metrics = _POINT_METRICS
    primary_metric = _POINT_METRICS[2]  # f1
    split = Split.TEST

    #: mm_olmo's ``sample``, applied per subset; ``None`` keeps every example.
    sample_per_subset: int | None = None

    def _build_instances(self) -> Iterator[Instance]:
        for idx, ex in enumerate(_load_gold_examples(self.sample_per_subset)):
            md = ex["metadata"]
            annotators = [
                [a["segmentation"] for a in md["annotations"][k]] for k in ("a", "b", "c")
            ]
            yield Instance(
                question=self.format_query(_strip_article(ex["text_input"]), idx),
                gold_answer=None,
                metadata={
                    "pointing_annotators": annotators,
                    "image_size": (int(md["width"]), int(md["height"])),
                    "image_path": ex["image"],
                    "example_id": md["id"],
                    "subset": md["subset"],
                    "label": ex["text_input"],
                },
            )


@register("sa_co_gold_point_4k_mp")
class SaCoGoldPoint4kMpTask(SaCoGoldPointMpBase):
    """``sa_co_gold_point_4k_mp:test`` — 4096 examples per subset (28,672 in total)."""

    sample_per_subset = 4096


@register("sa_co_gold_point_mp")
class SaCoGoldPointMpTask(SaCoGoldPointMpBase):
    """``sa_co_gold_point_mp:test`` — every example in the gold test set, unsampled."""

    sample_per_subset = None
