"""OmniDocBench — end-to-end document parsing (https://github.com/opendatalab/OmniDocBench).

Annotated page images — books, papers, slides, exam sheets, newspapers, notes, in English
and Chinese. The model converts each page to markdown; the official evaluator matches its
text blocks, display formulas and tables against the annotations and reports, averaged over
pages:

* ``text_edit`` — normalized edit distance of text blocks (lower is better),
* ``table_teds`` / ``table_teds_s`` — table TEDS and structure-only TEDS (x100),
* ``formula_cdm`` — display-formula CDM (x100),
* ``read_order_edit`` — reading-order edit distance (lower is better),
* ``overall`` = ``((1 - text_edit) * 100 + table_teds + formula_cdm) / 3``, the leaderboard
  number.

Two benchmark versions are registered, each scored by its own pinned official evaluator
run out of process (:mod:`olmo_eval.evals.vision.scoring.omnidocbench`), once per task after every
page is generated:

* ``omnidocbench`` — **v1.6**: v1.5's pages plus a set of hard pages, with annotation fixes,
  a new matching algorithm, and zero-filled page averages for tables and formulas.
* ``omnidocbench_v15`` — **v1.5**: the release behind the v1.5 leaderboard.

The two leaderboards are not comparable (the matching and the page averaging both changed),
which is why both are kept.

CDM renders formulas through LaTeX, ImageMagick and Ghostscript. Each task installs its
version's verified toolchain at job start unless it is already on ``PATH`` and checks that it
renders a formula, all before inference
(:func:`~olmo_eval.evals.vision.scoring.omnidocbench.ensure_cdm_toolchain`; the README lists
the versions), so the whole evaluation, CDM included, runs in one job.

The prompt is the one the official inference scripts share across general VLMs
(``tools/model_infer``), sent in a single user turn with the page image as shipped, to an
instruction-tuned checkpoint. A stage-1 checkpoint (``-o prompt_templates=none -o
system_prompt_style=style_and_length_v2``) gets the ``olmocr:`` tag alone instead: olmOCR's page
transcription writes math as ``\\( \\)`` / ``\\[ \\]`` and tables as HTML, which the official
evaluator reads, and its ``![alt](file)`` figure placeholders are deleted before matching.

Each version has an ``_en`` variant (``omnidocbench_en``, ``omnidocbench_v15_en``) that runs
and scores only the pages the annotations mark ``english``. Molmo2 models read English only, so
the ``_en`` variants are the default setting for evaluating them; the plain names run every
page, the leaderboards' setting, for multilingual models.
Decoding is greedy. Data is fetched from the Hub at each version's pinned revision; set
``OMNIDOCBENCH_DIR`` (v1.6) / ``OMNIDOCBENCH_V15_DIR`` (v1.5) to a local copy
(``OmniDocBench.json`` + ``images/``) to read from disk.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from olmo_eval.common.metrics.base import Metric
from olmo_eval.common.scorers.base import set_scorer_result
from olmo_eval.common.types import Instance, Response, SamplingParams, Split
from olmo_eval.evals.tasks.common import register
from olmo_eval.evals.vision.scoring.omnidocbench import (
    EVALUATORS,
    OmniDocBenchOverallMetric,
    OmniDocBenchPageMetric,
    OmniDocBenchScorer,
    ensure_cdm_toolchain,
    ensure_evaluator,
    run_official_evaluation,
)
from olmo_eval.evals.vision.tasks.ocr import OcrTask

if TYPE_CHECKING:
    from olmo_eval.common.execution import ScoringContext

logger = logging.getLogger(__name__)

HF_REPO = "opendatalab/OmniDocBench"
#: Dataset revision per version: the "add v1.6" commit (the repo has no version tags and
#: ``main`` keeps moving) and the head of the ``v1_5`` branch.
HF_REVISIONS = {
    "v1.6": "d386947f7fc3bafdcd756c8485845a2f43a19875",
    "v1.5": "91fe284bbfacfa687959ae3eb00846ca852aa907",
}
_LOCAL_DIR_ENV = {"v1.6": "OMNIDOCBENCH_DIR", "v1.5": "OMNIDOCBENCH_V15_DIR"}

#: ``tools/model_infer/Qwen3-VL-235B_img2md.py`` of the official repo, verbatim (the copies
#: in the other scripts differ only by string-escaping accidents).
PROMPT = (
    "You are an AI assistant specialized in converting PDF images to Markdown format. "
    "Please follow these instructions for the conversion:\n\n"
    "1. Text Processing:\n"
    "- Accurately recognize all text content in the PDF image without guessing or inferring.\n"
    "- Convert the recognized text into Markdown format.\n"
    "- Maintain the original document structure, including headings, paragraphs, lists, etc.\n\n"
    "2. Mathematical Formula Processing:\n"
    "- Convert all mathematical formulas to LaTeX format.\n"
    "- Enclose inline formulas with \\( \\). For example: This is an inline formula "
    "\\( E = mc^2 \\)\n"
    "- Enclose block formulas with \\[ \\]. For example: "
    "\\[ \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a} \\]\n\n"
    "3. Table Processing:\n"
    "- Convert tables to HTML format.\n"
    "- Wrap the entire table with <table> and </table>.\n\n"
    "4. Figure Handling:\n"
    "- Ignore figures content in the PDF image. Do not attempt to describe or convert "
    "images.\n\n"
    "5. Output Format:\n"
    "- Ensure the output Markdown document has a clear structure with appropriate line "
    "breaks between elements.\n"
    "- For complex layouts, try to maintain the original document's structure and format "
    "as closely as possible.\n\n"
    "Please strictly follow these guidelines to ensure accuracy and consistency in the "
    "conversion. Your task is to accurately convert the content of the PDF image into "
    "Markdown format without adding any extra explanations or comments."
)

_SCORER = OmniDocBenchScorer()

_LANGUAGES = {"english": "english", "chinese": "simplified_chinese", "mixed": "en_ch_mixed"}
#: ``page_attribute.subset``: the v1.5 pages and the three hard sets v1.6 added.
_SUBSETS = {
    "v15": "v1.5",
    "equation_hard": "equation_hard",
    "layout_hard": "layout_hard",
    "table_hard": "table_hard",
}


def _page(name: str, component: str, **kwargs) -> OmniDocBenchPageMetric:
    return OmniDocBenchPageMetric(name=name, scorer=_SCORER, component=component, **kwargs)


def _component_metrics(version: str, *, by_language: bool = True) -> tuple[Metric, ...]:
    """Every metric except CDM-dependent ones, for one benchmark version.

    :param by_language: Also slice text edit by page language; off for tasks that run one
        language, where the other slices would read 0.0.
    """
    zero_fill = EVALUATORS[version].zero_fills_missing_pages
    table_fill = "has_table" if zero_fill else None
    subset_slices: tuple[Metric, ...] = ()
    if version == "v1.6":
        subset_slices = tuple(
            _page(f"text_edit_{key}", "text_edit", attribute="subset", attribute_value=value)
            for key, value in _SUBSETS.items()
        )
    return (
        _page("text_edit", "text_edit"),
        _page("table_teds", "table_teds", zero_fill=table_fill, scale=100.0),
        _page("table_teds_s", "table_teds_s", zero_fill=table_fill, scale=100.0),
        _page("formula_edit", "formula_edit"),
        _page("read_order_edit", "read_order_edit"),
        *(
            _page(f"text_edit_{key}", "text_edit", attribute="language", attribute_value=value)
            for key, value in (_LANGUAGES.items() if by_language else ())
        ),
        *subset_slices,
    )


def _cdm_metrics(version: str) -> tuple[Metric, Metric]:
    """``(overall, formula_cdm)`` for one benchmark version."""
    zero_fill = EVALUATORS[version].zero_fills_missing_pages
    return (
        OmniDocBenchOverallMetric(name="overall", scorer=_SCORER, zero_fill=zero_fill),
        _page(
            "formula_cdm",
            "formula_cdm",
            zero_fill="has_formula" if zero_fill else None,
            scale=100.0,
        ),
    )


#: Metric name -> factor from the evaluator's raw (0-1) value to ours.
_SUMMARY_SCALES = {
    "text_edit": 1.0,
    "read_order_edit": 1.0,
    "table_teds": 100.0,
    "table_teds_s": 100.0,
    "formula_cdm": 100.0,
    "overall": 1.0,
}


def _data_dir(version: str) -> Path:
    local = os.environ.get(_LOCAL_DIR_ENV[version])
    if local:
        return Path(local)
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=HF_REPO,
            repo_type="dataset",
            revision=HF_REVISIONS[version],
            allow_patterns=["OmniDocBench.json", "images/*"],
        )
    )


def _load_pages(data_dir: Path) -> list[dict]:
    with open(data_dir / "OmniDocBench.json", encoding="utf-8") as f:
        return json.load(f)


def _has_block(page: dict, category_type: str) -> bool:
    return any(
        block.get("category_type") == category_type and not block.get("ignore", False)
        for block in page.get("layout_dets", ())
    )


@register("omnidocbench")
class OmniDocBenchTask(OcrTask):
    #: Provisioning the evaluator also needs `git` and `uv` on PATH.
    dependencies = ["pillow", "huggingface-hub", "pyyaml", "filelock"]
    sampling_params = SamplingParams(temperature=0.0, max_tokens=8192)
    metrics = (*_cdm_metrics("v1.6"), *_component_metrics("v1.6"))
    primary_metric = metrics[0]
    split = Split.TEST
    #: Benchmark version: which data revision and official evaluator to use.
    version: Literal["v1.6", "v1.5"] = "v1.6"
    #: Page languages to run (the ``page_attribute.language`` annotation); ``None`` runs all.
    languages: tuple[str, ...] | None = None

    def _build_instances(self) -> Iterator[Instance]:
        ensure_cdm_toolchain(self.version)
        ensure_evaluator(self.version)
        data_dir = _data_dir(self.version)
        question = self._question(PROMPT)
        for page in _load_pages(data_dir):
            info = page["page_info"]
            image_name = os.path.basename(info["image_path"])
            attributes = info.get("page_attribute") or {}
            if self.languages is not None and attributes.get("language") not in self.languages:
                continue
            yield Instance(
                question=question,
                gold_answer=None,
                metadata={
                    "id": image_name,
                    "example_id": image_name,
                    "image_name": image_name,
                    "image_path": str(data_dir / "images" / image_name),
                    "data_source": attributes.get("data_source"),
                    "language": attributes.get("language"),
                    "layout": attributes.get("layout"),
                    "subset": attributes.get("subset"),
                    "has_table": _has_block(page, "table"),
                    "has_formula": _has_block(page, "equation_isolated"),
                },
            )

    async def score_responses(
        self,
        responses: Sequence[Response],
        context: ScoringContext | None = None,
    ) -> Sequence[Response]:
        # Pages are graded together in `compute_metrics`; nothing is scored per response.
        self._extract_answers(responses)
        return responses

    def compute_metrics(self, responses: Sequence[Response]) -> dict[str, dict[str, float]]:
        predictions = {
            r.instance.metadata["image_name"]: (r.outputs[0].text or "") if r.outputs else ""
            for r in responses
        }
        gt_pages = [
            page
            for page in _load_pages(_data_dir(self.version))
            if os.path.basename(page["page_info"]["image_path"]) in predictions
        ]
        try:
            evaluation = run_official_evaluation(
                predictions, gt_pages, with_cdm=True, version=self.version
            )
        except RuntimeError as exc:
            # Every page has been generated by now, and raising would lose all of them. Mark
            # each output's score as lost to infrastructure instead: the run then reports an
            # error, its predictions are still written for rescoring, and no metric is made up.
            logger.exception("OmniDocBench %s evaluation failed.", self.version)
            error = {
                "phase": "compute_metrics",
                "type": type(exc).__qualname__,
                "message": str(exc),
                "infrastructure": "true",
            }
            for response in responses:
                for output in response.outputs:
                    if output.metadata is None:
                        output.metadata = {}
                    output.metadata.setdefault("scoring_errors", {})[_SCORER.name] = dict(error)
            return {}

        for response in responses:
            if not response.outputs:
                continue
            output = response.outputs[0]
            page = evaluation.per_page.get(response.instance.metadata["image_name"], {})
            set_scorer_result(output, _SCORER.name, dict(page))
            score = _SCORER.score(response.instance, output)
            output.metadata[f"score:{_SCORER.name}"] = score
            response.scores[_SCORER.name] = score

        metrics = super().compute_metrics(responses)
        self._check_against_evaluator(metrics, evaluation.summary)
        # Run-level facts that explain why two evaluations of the same outputs can differ:
        # pages zeroed by the evaluator's own time limits, and the workers it ran with.
        metrics["n_scorer_failures"] = {_SCORER.name: float(evaluation.num_scorer_failures)}
        metrics["eval_workers"] = {_SCORER.name: float(evaluation.workers)}
        return metrics

    @staticmethod
    def _check_against_evaluator(
        metrics: dict[str, dict[str, float]], summary: dict[str, float]
    ) -> None:
        """Warn when a page average disagrees with the evaluator's own leaderboard numbers."""
        for name, factor in _SUMMARY_SCALES.items():
            if name not in summary or name not in metrics:
                continue
            ours, theirs = metrics[name][_SCORER.name], summary[name] * factor
            if abs(ours - theirs) > 1e-6 * max(1.0, factor):
                logger.warning(
                    "OmniDocBench %s: page average %.6f differs from the evaluator's %.6f",
                    name,
                    ours,
                    theirs,
                )


@register("omnidocbench_v15")
class OmniDocBenchV15Task(OmniDocBenchTask):
    """OmniDocBench v1.5, scored by the v1.5 evaluator."""

    metrics = (*_cdm_metrics("v1.5"), *_component_metrics("v1.5"))
    primary_metric = metrics[0]
    version = "v1.5"


#: The ``page_attribute.language`` value of the English pages.
_ENGLISH = ("english",)


@register("omnidocbench_en")
class OmniDocBenchEnglishTask(OmniDocBenchTask):
    """OmniDocBench v1.6, English pages only."""

    metrics = (*_cdm_metrics("v1.6"), *_component_metrics("v1.6", by_language=False))
    primary_metric = metrics[0]
    languages = _ENGLISH


@register("omnidocbench_v15_en")
class OmniDocBenchV15EnglishTask(OmniDocBenchV15Task):
    """OmniDocBench v1.5, English pages only."""

    metrics = (*_cdm_metrics("v1.5"), *_component_metrics("v1.5", by_language=False))
    primary_metric = metrics[0]
    languages = _ENGLISH
