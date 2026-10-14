"""GEOBench-VLM — geospatial multiple-choice benchmark (ICCV 2025, arXiv 2411.19325).

Data comes from the Hub release ``aialliance/GEOBench-VLM`` (``Single.zip`` / ``Temporal.zip``,
pinned below): each ``qa.json`` record is one five-option question (A-E) with five paraphrased
``prompts``. Following the official scripts (``eval_geobenchvlm/*_cls_single.py`` and
``temporal/*_cls_temporal.py`` in The-AI-Alliance/GEO-Bench-VLM), every (question, prompt)
pair is one request, with the official instruction verbatim::

    For the given the Multiple Choice Question Answer below, ... {cls_description}
    {prompt}
    Options: A. ...   B. ...   C. ...   D. ...   E. ...

Answers are parsed with the official first-character rule
(:func:`olmo_eval.evals.vision.scoring.geobench_vlm.parse_geobench_answer`) and decoded
greedily with 128 new tokens (the official Qwen2-VL script).

Tasks:

* ``geobench_vlm`` — the single-image MCQ tasks of the paper's Figure 4, without the xBD
  imagery (see below): 16 tasks, 2,984 questions, 14,920 requests.
* ``geobench_vlm_full`` — all 17 Figure 4 tasks, 3,211 questions, 16,055 requests; needs xBD.
* ``geobench_vlm_temporal`` — the multi-temporal tasks of Table 3 without xBD: crop type,
  farm-pond change detection and land use, 436 questions, 2,180 requests.
* ``geobench_vlm_temporal_full`` — all five Table 3 tasks, 1,713 questions, 8,565 requests;
  needs xBD.

Metrics (0-1): one accuracy per task; ``average``, the unweighted mean of the per-task
accuracies (Figure 4's "Average"; the 17-task value is the paper's headline, e.g. 0.417 for
LLaVA-OneVision); and the Table 2 categories, pooled over their tasks' questions (pooling
reproduces Table 2 from Figure 4). The released code does not score, and the paper does not
say how a question's five prompts combine. Every reported accuracy is a whole number of
questions out of the task's question count, so per-question outcomes are scored: the
headline metrics use the first prompt, and ``average_majority`` (correct on at least three
of five prompts) and ``average_prompt_mean`` (every request counted) are reported beside it.
``unparsed`` is the fraction of replies with no A-E first character.

Parity: Qwen2-VL-7B-Instruct (HF, bf16) scores a 16-task ``average`` of 0.400 against
0.393 from the paper's Figure 4 over the same tasks. Four per-task values match exactly
(more than any other prompt combination gives), and the rest are within a few points.

Deviations:

* **xBD imagery.** 210 of the 227 disaster-type questions (single and temporal) and all
  1,050 damaged-building counts use xBD images, which the release omits at the xBD authors'
  request (GEO-Bench-VLM issue #1). The ``_full`` tasks read them from
  ``$GEOBENCH_XBD_DIR`` (the xView2 download, holding ``hold/images`` and ``test/images``,
  as the release's ``preprocess_xbd.py`` expects) and fail if it is unset or incomplete. The
  default tasks drop those two tasks entirely rather than score them on a remnant. Their
  ``average`` is a 16-task mean, so compare it with the paper's per-task numbers over the
  same 16 tasks. Their ``event_detection`` category is fire-risk assessment alone.
* **Temporal image placement.** The official temporal prompt interleaves text and images
  ("This is the 'pre' image:" <image> "This is the 'post' image:" <image>). Providers here
  place images before the text, so the request sends the pre and post images (the first and
  last of the sequence, as the official scripts do) followed by "The following images show
  the condition before and after the event. The first image is the 'pre' image and the
  second image is the 'post' image." and the question.
* Referring-expression detection, referring segmentation, captioning and the non-optical
  tasks are not in the public ``qa.json`` files and are not implemented.
"""

from __future__ import annotations

import functools
import io
import json
import os
import zipfile
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from olmo_eval.common.metrics.base import Metric
from olmo_eval.common.scorers.base import Scorer, get_scorer_result
from olmo_eval.common.types import Instance, Response, SamplingParams, Split
from olmo_eval.evals.tasks.common import register
from olmo_eval.evals.vision.scoring.geobench_vlm import GeobenchMcScorer
from olmo_eval.evals.vision.tasks.image_qa import ImageQATask
from olmo_eval.evals.vision.tasks.multi_image import MultiImageQATask

_REPO = "aialliance/GEOBench-VLM"
_REVISION = "b3485fe4971213ee13f91960638ddc5c4a098e70"
XBD_DIR_ENV = "GEOBENCH_XBD_DIR"
_XBD_PREFIX = "xBD_"

#: Official instruction (``*_cls_single.py`` / ``*_cls_temporal.py``), verbatim, typos included.
_INSTRUCTION = (
    "For the given the Multiple Choice Question Answer below, analyze the question and answer "
    "strictly from one of the options below. Strictly answer the choice only. No additional "
    "text. Provide only the letter (A., B., C., D. or E.) corresponding to the correct answer "
    "for the multiple-choice question given. {cls_description}\n{question}\n{choices}"
)
_TEMPORAL_PREFIX = (
    "The following images show the condition before and after the event. The first image is "
    "the 'pre' image and the second image is the 'post' image.\n"
)

#: Task name in ``qa.json`` (typos as released) -> metric name.
SINGLE_TASKS: dict[str, str] = {
    "Aircraft Type Classification": "aircraft_type_classification",
    "Building Counting": "building_counting",
    "Crop Type Classification": "crop_type_classification",
    "Disaster Type Classification": "disaster_type_classification",
    "Fire Risk Assessment": "fire_risk_assessment",
    "General Aircraft Counting": "general_aircraft_counting",
    "General Vehicle Counting": "general_vehicle_counting",
    "Land Use Classification": "land_use_classification",
    "Marine Debirs Counting": "marine_debris_counting",
    "Scene Classification": "scene_classification",
    "Ship Type Classification": "ship_type_classification",
    "Specific Aircraft Type Counting": "specific_aircraft_type_counting",
    "Specific Vehicle Type Counting": "specific_vehicle_type_counting",
    "Sptatial Relation Classification": "spatial_relation_classification",
    "Tree Health Assessment": "tree_health_assessment",
    "Trees Counting": "trees_counting",
    "Water Bodies Counting": "water_bodies_counting",
}
TEMPORAL_TASKS: dict[str, str] = {
    "Crop Type Classification": "crop_type_classification",
    "Damaged Building Counting": "damaged_building_counting",
    "Disaster Type Classification": "disaster_type_classification",
    "Farm Pond Change Detection": "farm_pond_change_detection",
    "Land Use Classification": "land_use_classification",
}
#: Tasks whose images are (mostly) xBD; the default tasks drop them.
XBD_TASKS = frozenset({"Disaster Type Classification", "Damaged Building Counting"})

#: Table 2 categories (pooled over their tasks' questions).
CATEGORIES: dict[str, tuple[str, ...]] = {
    "event_detection": ("Disaster Type Classification", "Fire Risk Assessment"),
    "object_classification": ("Aircraft Type Classification", "Ship Type Classification"),
    "counting": (
        "Building Counting",
        "General Aircraft Counting",
        "General Vehicle Counting",
        "Marine Debirs Counting",
        "Specific Aircraft Type Counting",
        "Specific Vehicle Type Counting",
        "Trees Counting",
        "Water Bodies Counting",
        "Tree Health Assessment",
        "Sptatial Relation Classification",
    ),
    "scene_understanding": (
        "Crop Type Classification",
        "Land Use Classification",
        "Scene Classification",
    ),
}

_SCORER = GeobenchMcScorer()

#: How a question's five prompt outcomes combine into one question outcome.
_SCHEMES = ("first_prompt", "majority", "prompt_mean")


def _question_outcomes(
    responses: Sequence[Response], scheme: str, tasks: frozenset[str] | None
) -> dict[str, list[float]]:
    """Per-task list of question outcomes (0-1) under ``scheme``."""
    by_question: dict[tuple[str, int], dict[int, float]] = defaultdict(dict)
    for response in responses:
        meta = response.instance.metadata
        if tasks is not None and meta["task"] not in tasks:
            continue
        score = float(response.scores.get(_SCORER.name, 0.0))
        by_question[(meta["task"], meta["question_index"])][meta["prompt_index"]] = score
    out: dict[str, list[float]] = defaultdict(list)
    for (task, _), prompt_scores in by_question.items():
        if scheme == "first_prompt":
            if 0 not in prompt_scores:
                continue
            value = prompt_scores[0]
        elif scheme == "majority":
            value = float(sum(prompt_scores.values()) * 2 > len(prompt_scores))
        else:
            value = sum(prompt_scores.values()) / len(prompt_scores)
        out[task].append(value)
    return out


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


@dataclass(frozen=True)
class GeobenchAccuracyMetric(Metric):
    """Question accuracy pooled over ``tasks`` (all tasks when ``None``) under ``scheme``.

    ``prompt_mean`` counts every request; ``first_prompt`` and ``majority`` score each
    question once (see the module docstring).
    """

    name: str  # type: ignore[misc]
    scorer: Scorer  # type: ignore[misc]
    tasks: frozenset[str] | None = None
    scheme: str = "first_prompt"

    def compute(self, responses: Sequence[Response]) -> float:
        outcomes = _question_outcomes(responses, self.scheme, self.tasks)
        return _mean([v for values in outcomes.values() for v in values])

    def compute_instance(self, response: Response) -> float | None:
        """The request's own score where it is the question outcome, else ``None``."""
        meta = response.instance.metadata
        if self.tasks is not None and meta["task"] not in self.tasks:
            return None
        if self.scheme == "majority":
            return None  # a vote over five requests has no per-request value
        if self.scheme == "first_prompt" and meta["prompt_index"] != 0:
            return None
        return float(response.scores.get(_SCORER.name, 0.0))

    def supports_pairwise_scorer_fallback(self) -> bool:
        return False


@dataclass(frozen=True)
class GeobenchAverageMetric(Metric):
    """Unweighted mean over the tasks present of each task's accuracy under ``scheme``."""

    name: str  # type: ignore[misc]
    scorer: Scorer  # type: ignore[misc]
    scheme: str = "first_prompt"

    def compute(self, responses: Sequence[Response]) -> float:
        outcomes = _question_outcomes(responses, self.scheme, None)
        return _mean([_mean(values) for values in outcomes.values()])

    def compute_instance(self, response: Response) -> float | None:
        return None  # a macro average has no per-request decomposition

    def supports_pairwise_scorer_fallback(self) -> bool:
        return False


@dataclass(frozen=True)
class GeobenchUnparsedMetric(Metric):
    """Fraction of replies whose first character is not an option letter."""

    name: str  # type: ignore[misc]
    scorer: Scorer  # type: ignore[misc]

    def compute(self, responses: Sequence[Response]) -> float:
        values = [v for v in map(self.compute_instance, responses) if v is not None]
        return _mean(values)

    def compute_instance(self, response: Response) -> float | None:
        for output in response.outputs:
            result = get_scorer_result(output, _SCORER.name)
            if result is not None:
                return float(result["pred"] is None)
        return None

    def supports_pairwise_scorer_fallback(self) -> bool:
        return False


def _metrics(task_names: dict[str, str], tasks: Sequence[str]) -> tuple[Metric, ...]:
    present = frozenset(tasks)
    categories = [
        GeobenchAccuracyMetric(name=name, scorer=_SCORER, tasks=frozenset(members) & present)
        for name, members in CATEGORIES.items()
        if frozenset(members) & present
    ]
    return (
        GeobenchAverageMetric(name="average", scorer=_SCORER),
        *(
            GeobenchAccuracyMetric(name=task_names[t], scorer=_SCORER, tasks=frozenset({t}))
            for t in tasks
        ),
        *categories,
        GeobenchAverageMetric(name="average_majority", scorer=_SCORER, scheme="majority"),
        GeobenchAverageMetric(name="average_prompt_mean", scorer=_SCORER, scheme="prompt_mean"),
        GeobenchUnparsedMetric(name="unparsed", scorer=_SCORER),
    )


def _zip_image(zip_path: str, member: str):
    """Decode one image stored in a release zip (module-level so it is picklable)."""
    from PIL import Image

    with zipfile.ZipFile(zip_path) as archive:
        data = archive.read(member)
    return Image.open(io.BytesIO(data)).convert("RGB")


def _file_image(path: str):
    """Decode one image file (module-level so it is picklable)."""
    from PIL import Image

    with Image.open(path) as image:
        return image.convert("RGB")


def _xbd_index() -> dict[str, Path]:
    """Map xBD file names to paths under ``$GEOBENCH_XBD_DIR`` ({hold,test}/images)."""
    root = os.environ.get(XBD_DIR_ENV)
    if not root:
        raise FileNotFoundError(
            f"The full GEOBench-VLM tasks need the xBD images; download xBD from "
            f"https://xview2.org and set {XBD_DIR_ENV} to the folder holding hold/ and test/."
        )
    index: dict[str, Path] = {}
    for subset in ("hold", "test"):
        images = Path(root) / subset / "images"
        if images.is_dir():
            for path in images.iterdir():
                index[path.name] = path
    return index


class _GeobenchMixin:
    """Instance building shared by the single-image and temporal tasks."""

    archive: ClassVar[str]
    task_names: ClassVar[dict[str, str]]
    include_xbd: ClassVar[bool] = False

    def _records(self) -> Iterator[tuple[int, dict[str, Any], list[Any]]]:
        from huggingface_hub import hf_hub_download

        zip_path = hf_hub_download(
            _REPO, f"{self.archive}.zip", repo_type="dataset", revision=_REVISION
        )
        with zipfile.ZipFile(zip_path) as archive:
            records = json.loads(archive.read(f"{self.archive}/qa.json"))
            members = set(archive.namelist())
        xbd = _xbd_index() if self.include_xbd else {}
        for index, record in enumerate(records):
            task = record["task"]
            if task not in self.task_names:
                raise ValueError(f"GEOBench-VLM record {index} has an unknown task {task!r}")
            if task in XBD_TASKS and not self.include_xbd:
                continue
            paths = record["image_path"]
            paths = [paths[0], paths[-1]] if isinstance(paths, list) else [paths]
            images = []
            for path in paths:
                name = path.split("/")[-1]
                if path in members:
                    images.append(functools.partial(_zip_image, zip_path, path))
                elif name.startswith(_XBD_PREFIX) and name[len(_XBD_PREFIX) :] in xbd:
                    images.append(
                        functools.partial(_file_image, str(xbd[name[len(_XBD_PREFIX) :]]))
                    )
                else:
                    raise FileNotFoundError(
                        f"GEOBench-VLM image {path} (record {index}, {task}) is neither in "
                        f"{self.archive}.zip nor under ${XBD_DIR_ENV}"
                    )
            yield index, record, images

    def _instances(self, prefix: str = "") -> Iterator[tuple[Instance, list[Any]]]:
        for index, record, images in self._records():
            choices = "Options: " + record["options"]
            for prompt_index, prompt in enumerate(record["prompts"]):
                question = prefix + _INSTRUCTION.format(
                    cls_description=record["cls_description"], question=prompt, choices=choices
                )
                instance = Instance(
                    question=question,
                    gold_answer=record["ground_truth_option"],
                    metadata={
                        "answer": record["ground_truth_option"],
                        "answer_text": record["ground_truth"],
                        "options": list(record["options_list"]),
                        "task": record["task"],
                        "question_index": index,
                        "prompt_index": prompt_index,
                        "example_id": f"{index}_{prompt_index}",
                    },
                )
                yield instance, images


def _single_tasks(include_xbd: bool) -> list[str]:
    return [t for t in SINGLE_TASKS if include_xbd or t not in XBD_TASKS]


def _temporal_tasks(include_xbd: bool) -> list[str]:
    return [t for t in TEMPORAL_TASKS if include_xbd or t not in XBD_TASKS]


_SINGLE_METRICS = _metrics(SINGLE_TASKS, _single_tasks(False))
_SINGLE_FULL_METRICS = _metrics(SINGLE_TASKS, _single_tasks(True))
_TEMPORAL_METRICS = _metrics(TEMPORAL_TASKS, _temporal_tasks(False))
_TEMPORAL_FULL_METRICS = _metrics(TEMPORAL_TASKS, _temporal_tasks(True))


@register("geobench_vlm")
class GeobenchVlmTask(_GeobenchMixin, ImageQATask):
    """Single-image GEOBench-VLM MCQs, without the xBD-backed disaster-type task."""

    dependencies = ["pillow", "huggingface-hub"]
    sampling_params = SamplingParams(temperature=0.0, max_tokens=128)
    metrics = _SINGLE_METRICS
    primary_metric = _SINGLE_METRICS[0]  # average
    split = Split.TEST
    archive = "Single"
    task_names = SINGLE_TASKS

    def _build_instances(self) -> Iterator[Instance]:
        for instance, images in self._instances():
            instance.metadata["image"] = images[0]
            yield instance


@register("geobench_vlm_full")
class GeobenchVlmFullTask(GeobenchVlmTask):
    """All 17 single-image GEOBench-VLM tasks (needs ``$GEOBENCH_XBD_DIR``)."""

    metrics = _SINGLE_FULL_METRICS
    primary_metric = _SINGLE_FULL_METRICS[0]
    include_xbd = True


@register("geobench_vlm_temporal")
class GeobenchVlmTemporalTask(_GeobenchMixin, MultiImageQATask):
    """Pre/post-image GEOBench-VLM MCQs, without the xBD-backed tasks."""

    dependencies = ["pillow", "huggingface-hub"]
    sampling_params = SamplingParams(temperature=0.0, max_tokens=128)
    metrics = _TEMPORAL_METRICS
    primary_metric = _TEMPORAL_METRICS[0]  # average
    split = Split.TEST
    archive = "Temporal"
    task_names = TEMPORAL_TASKS

    def _build_instances(self) -> Iterator[Instance]:
        for instance, images in self._instances(prefix=_TEMPORAL_PREFIX):
            instance.metadata["images"] = tuple(images)
            instance.metadata["num_images"] = len(images)
            yield instance


@register("geobench_vlm_temporal_full")
class GeobenchVlmTemporalFullTask(GeobenchVlmTemporalTask):
    """All five temporal GEOBench-VLM tasks (needs ``$GEOBENCH_XBD_DIR``)."""

    metrics = _TEMPORAL_FULL_METRICS
    primary_metric = _TEMPORAL_FULL_METRICS[0]
    include_xbd = True
