"""μ-Bench (Micro-Bench) — closed-VQA microscopy perception (NeurIPS 2024 D&B).

Two tasks reproduce the generative-VLM protocol of the official evaluation code
(``alejandro-lozano-dev/eVLLM``, linked from the paper as the μ-Bench eval code):

* ``micro_bench_coarse`` — coarse-grained perception: five closed-VQA questions per image about
  the scientific domain, subdomain, microscopy modality, submodality and stain (79,716 questions).
* ``micro_bench_fine`` — fine-grained perception: one dataset-specific classification question
  per image (cell type, cell-cycle phase, organelle, tissue class, ...; 16,130 questions).

Data is the ``test`` split (the only split) of ``jnirschl/uBench`` (perception config, 17,315
images), pinned to a Hub revision. Each image's ``questions`` dict carries the question, its
candidate options (the last is always "none of the above") and the answer.

Protocol (``inference/generative_inference.py`` + ``eval/eval_utils.py``):

* Prompt: ``"Answer with a single letter, no extra details.\\nQuestion:" + question + "\\n"`` +
  options joined by newlines, each prefixed ``"A) "``, ``"B) "``, ... (no space after
  ``Question:``, exactly as the code builds it; the paper's figure shows one).
* Answer matching (``check_prediction``): correct iff the lettered gold option (e.g.
  ``"B) basophil"``) is a case-sensitive substring of the response, or the response's first
  character is the gold letter. Anything else is wrong.
* Metrics are pooled accuracies over all questions (``process_models`` /
  ``process_models_question_only`` average ``is_correct`` over the concatenated per-dataset
  outputs), which is what the paper's Table 1 reports as "macro-average accuracy": ``accuracy``
  (primary) plus the five per-question-type accuracies for coarse, and per-dataset accuracies for
  fine. The official fine-grained total (``eval_posthoc_rebutall_finegrained.py``) leaves out
  the Eulenberg et al. 2017 darkfield and epifluorescence datasets (1,486 questions), so fine's
  ``accuracy`` does too (14,644 questions); ``accuracy_all_datasets`` pools all 16,130.
  ``pathology`` is the official pathology-only slice (Table 7), and fine's ``macro_dataset`` is
  the unweighted mean over datasets (an extra, not an official number).
* Datasets: the official tables cover the 23 datasets of ``tasks_metadata``. The Hub release
  also ships ``opencell`` (1,105 images) and ``sirinukunwattana_et_al_2016`` (80) outside that
  list, which are excluded here; its Held et al. 2010 rows have a null ``dataset`` and are named
  by their marker (``held_et_al_2010_h2b`` for H2B-mCherry, ``held_et_al_2010_galt`` for GalT).
  Questions the release leaves null (stain for ``nirschl_unpub_fluorescence``) are skipped, as the
  official loop only visits the questions an image has.

Parity: ``gpt-4o-2024-05-13`` at the reference's temperature 1, on a 5,000-question random
sample per split, scores 0.632 coarse against the paper's 0.627 and 0.483 fine against 0.517.
The fine gap likely comes from ``held_et_al_2010_mt``, which the official total includes but
the release lacks, and from sampling noise at temperature 1.

Deviations:

* Decoding is greedy with a 128-token cap; the reference ran GPT-4o at temperature 1 and each
  open model at its own default generation settings.
* Leading/trailing whitespace is stripped from the response before matching (the reference
  wrappers return stripped text, except where the decoded string had none to strip).
* μ-Bench Cognition (121 expert questions) is not part of the public release, and the
  localization split is scored with GRIT on bounding-box outputs; neither is included.
"""

from __future__ import annotations

import functools
import io
import threading
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from olmo_eval.common.metrics.base import Metric
from olmo_eval.common.scorers.base import Scorer
from olmo_eval.common.types import Instance, LMOutput, Response, SamplingParams, Split
from olmo_eval.evals.tasks.common import register
from olmo_eval.evals.vision.scoring.common import response_text
from olmo_eval.evals.vision.tasks.image_qa import ImageQATask

_DATASET = "jnirschl/uBench"
#: Hub revision of jnirschl/uBench these tasks were checked against.
_REVISION = "3f2c5b590bc7a208d5b60f3527ce4c76a331aa2b"

#: The official generative prompt (``evaluate_dataset(prompt=...)``).
PROMPT = "Answer with a single letter, no extra details."

_LETTERS = "ABCDEFGH"  # ``add_alphabet``'s ``idx_to_option``

#: The datasets the official tables cover (``eval_utils.tasks_metadata``). ``held_et_al_2010_mt``
#: is listed there but absent from the Hub release.
DATASETS: tuple[str, ...] = (
    "acevedo_et_al_2020",
    "burgess_et_al_2024_contour",
    "burgess_et_al_2024_eccentricity",
    "burgess_et_al_2024_texture",
    "empiar_sbfsem",
    "colocalization_benchmark",
    "eulenberg_et_al_2017_brightfield",
    "eulenberg_et_al_2017_darkfield",
    "eulenberg_et_al_2017_epifluorescence",
    "held_et_al_2010_galt",
    "held_et_al_2010_h2b",
    "hussain_et_al_2019",
    "icpr2020_pollen",
    "jung_et_al_2022",
    "kather_et_al_2016",
    "kather_et_al_2018",
    "kather_et_al_2018_val7k",
    "nirschl_et_al_2018",
    "nirschl_unpub_fluorescence",
    "tang_et_al_2019",
    "wong_et_al_2022",
    "wu_et_al_2023",
)

#: The official pathology-only slice (``eval_posthoc.py``, ``_pathology`` tables).
PATHOLOGY_DATASETS: tuple[str, ...] = (
    "hussain_et_al_2019",
    "acevedo_et_al_2020",
    "jung_et_al_2022",
    "kather_et_al_2016",
    "kather_et_al_2018",
    "kather_et_al_2018_val7k",
    "nirschl_et_al_2018",
)

#: Coarse-grained question types (the ``questions`` keys other than ``classification``).
COARSE_TYPES: tuple[str, ...] = ("domain", "modality", "stain", "subdomain", "submodality")

#: Held et al. 2010 rows have a null ``dataset`` on the Hub; their stain names the sub-dataset.
_HELD_BY_STAIN: dict[str, str] = {
    "H2B-mCherry": "held_et_al_2010_h2b",
    "GalT–EGFP": "held_et_al_2010_galt",
}


def build_prompt(question: str, options: Sequence[str]) -> str:
    """The official generative-VLM query for one closed-VQA question."""
    lettered = [f"{_LETTERS[i]}) {option}" for i, option in enumerate(options)]
    return PROMPT + "\n" + "Question:" + question + "\n" + "\n".join(lettered)


def is_correct(prediction: str, answer_idx: int, answer: str) -> bool:
    """The official ``check_prediction``: lettered gold option in the text, or gold first char."""
    letter = _LETTERS[answer_idx]
    return f"{letter}) {answer}" in prediction or prediction[0:1] == letter


_LOAD_LOCK = threading.Lock()


@functools.cache
def _load_dataset() -> Any:
    import datasets

    return datasets.load_dataset(_DATASET, split="test", revision=_REVISION)


@functools.cache
def _load_image_column() -> Any:
    import datasets

    return (
        _load_dataset().select_columns(["image"]).cast_column("image", datasets.Image(decode=False))
    )


def _dataset() -> Any:
    """The pinned test split, loaded once per process."""
    with _LOAD_LOCK:
        return _load_dataset()


def _decode_image(index: int):
    """Decode the image of row ``index`` (module-level so a ``partial`` over it is picklable).

    Instances reference images by row index and the table is opened once per process: a
    ``Dataset`` inside each pickled instance would re-memory-map the Arrow shards on every
    unpickle, which exhausts the process's mappings at ~80k instances.
    """
    from PIL import Image

    with _LOAD_LOCK:
        images = _load_image_column()
    cell = images[index]["image"]
    if cell.get("bytes"):
        return Image.open(io.BytesIO(cell["bytes"]))
    return Image.open(cell["path"])


def _dataset_name(ex: dict) -> str | None:
    if ex["dataset"] is not None:
        return ex["dataset"]
    return _HELD_BY_STAIN.get(ex["stain"])


@dataclass(frozen=True, slots=True)
class MicroBenchScorer(Scorer):
    """Official μ-Bench answer matching on the whitespace-stripped response."""

    name: str = "micro_bench"

    def score(self, instance: Instance, output: LMOutput) -> float:
        meta = instance.metadata
        prediction = response_text(output).strip()
        return float(is_correct(prediction, meta["answer_idx"], meta["answer"]))


_SCORER = MicroBenchScorer()


@dataclass(frozen=True)
class MicroBenchAccuracyMetric(Metric):
    """Pooled accuracy, optionally restricted to instances whose ``field`` is in ``values``."""

    name: str  # type: ignore[misc]
    scorer: Scorer  # type: ignore[misc]
    field: str | None = None
    values: tuple[str, ...] = ()

    def _in_scope(self, response: Response) -> bool:
        return self.field is None or response.instance.metadata.get(self.field) in self.values

    def compute(self, responses: Sequence[Response]) -> float:
        scorer_name = self.scorer().name
        vals = [r.scores.get(scorer_name, 0.0) for r in responses if self._in_scope(r)]
        return sum(vals) / len(vals) if vals else 0.0

    def compute_instance(self, response: Response) -> float | None:
        """The example's own score, or ``None`` when it is outside this slice."""
        if not self._in_scope(response):
            return None
        value = response.scores.get(self.scorer().name)
        return float(value) if isinstance(value, (int, float)) else None

    def supports_pairwise_scorer_fallback(self) -> bool:
        return False


@dataclass(frozen=True)
class MicroBenchMacroMetric(Metric):
    """Unweighted mean over the groups of ``field`` of each group's accuracy."""

    name: str  # type: ignore[misc]
    scorer: Scorer  # type: ignore[misc]
    field: str = "dataset"

    def compute(self, responses: Sequence[Response]) -> float:
        scorer_name = self.scorer().name
        groups: dict[str, list[float]] = {}
        for r in responses:
            groups.setdefault(r.instance.metadata[self.field], []).append(
                r.scores.get(scorer_name, 0.0)
            )
        if not groups:
            return 0.0
        return sum(sum(v) / len(v) for v in groups.values()) / len(groups)

    def compute_instance(self, response: Response) -> float | None:
        # A macro average over groups has no exact per-instance decomposition.
        return None

    def supports_pairwise_scorer_fallback(self) -> bool:
        return False


_COARSE_METRICS: tuple[Metric, ...] = (
    MicroBenchAccuracyMetric(name="accuracy", scorer=_SCORER),
    *(
        MicroBenchAccuracyMetric(
            name=question_type, scorer=_SCORER, field="question_type", values=(question_type,)
        )
        for question_type in COARSE_TYPES
    ),
    MicroBenchAccuracyMetric(
        name="pathology", scorer=_SCORER, field="dataset", values=PATHOLOGY_DATASETS
    ),
)

#: ``eval_posthoc_rebutall_finegrained.py`` drops these two before computing the fine-grained
#: total that Table 1 reports.
FINE_EXCLUDED_DATASETS: tuple[str, ...] = (
    "eulenberg_et_al_2017_darkfield",
    "eulenberg_et_al_2017_epifluorescence",
)

_FINE_METRICS: tuple[Metric, ...] = (
    MicroBenchAccuracyMetric(
        name="accuracy",
        scorer=_SCORER,
        field="dataset",
        values=tuple(d for d in DATASETS if d not in FINE_EXCLUDED_DATASETS),
    ),
    MicroBenchAccuracyMetric(name="accuracy_all_datasets", scorer=_SCORER),
    MicroBenchMacroMetric(name="macro_dataset", scorer=_SCORER),
    MicroBenchAccuracyMetric(
        name="pathology", scorer=_SCORER, field="dataset", values=PATHOLOGY_DATASETS
    ),
    *(
        MicroBenchAccuracyMetric(name=dataset, scorer=_SCORER, field="dataset", values=(dataset,))
        for dataset in DATASETS
    ),
)


class _MicroBenchTask(ImageQATask):
    sampling_params = SamplingParams(temperature=0.0, max_tokens=128)
    split = Split.TEST  # the dataset's only split

    #: The ``questions`` keys this task asks.
    QUESTION_TYPES: ClassVar[tuple[str, ...]] = ()

    def _build_instances(self) -> Iterator[Instance]:
        rows = _dataset().select_columns(["image_id", "dataset", "stain", "domain", "questions"])
        allowed = set(DATASETS)
        for idx, ex in enumerate(rows):
            dataset = _dataset_name(ex)
            if dataset not in allowed:
                continue
            for question_type in self.QUESTION_TYPES:
                q = ex["questions"][question_type]
                if q is None or q["question"] is None:
                    continue
                options = list(q["options"])
                answer_idx = options.index(q["answer"])
                yield Instance(
                    question=build_prompt(q["question"], options),
                    gold_answer=_LETTERS[answer_idx],
                    metadata={
                        "answer": q["answer"],
                        "answer_idx": answer_idx,
                        "options": options,
                        "question_type": question_type,
                        "dataset": dataset,
                        "domain": ex["domain"],
                        "example_id": f"{ex['image_id']}:{question_type}",
                        "image": functools.partial(_decode_image, idx),
                    },
                )


@register("micro_bench_coarse")
class MicroBenchCoarseTask(_MicroBenchTask):
    metrics = _COARSE_METRICS
    primary_metric = _COARSE_METRICS[0]  # accuracy
    QUESTION_TYPES = COARSE_TYPES


@register("micro_bench_fine")
class MicroBenchFineTask(_MicroBenchTask):
    metrics = _FINE_METRICS
    primary_metric = _FINE_METRICS[0]  # accuracy
    QUESTION_TYPES = ("classification",)
