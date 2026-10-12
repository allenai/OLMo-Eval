"""olmOCR-bench — unit-test-graded document OCR (https://huggingface.co/datasets/allenai/olmOCR-bench).

Single-page PDFs carry hand-checked unit tests: a phrase is present or absent (headers and
footers must be dropped), two passages come in reading order, a table cell has the right
neighbours, a rendered equation matches. The model transcribes each page to markdown once
and the page's tests run against that text.

Scoring is the official implementation (:mod:`olmo_eval.evals.vision.scoring.olmocr_bench`) and
follows ``olmocr.bench.benchmark``: every PDF without an explicit ``baseline`` test gets one
in a ``baseline`` pseudo-category, each category's score is the pass rate of its tests, and
the primary ``overall`` is the unweighted mean of the eight category pass rates. Metrics are
0-1 (the leaderboard reports x100).

Choices the benchmark leaves to the system under test, and what this task does:

* **Prompt** — ``build_basic_prompt`` from ``olmocr/bench/prompts.py``, verbatim; it is the
  prompt behind the published general-VLM rows (e.g. Qwen2.5-VL). A stage-1 checkpoint
  (``-o prompt_templates=none -o system_prompt_style=style_and_length_v2``) gets the
  ``olmocr:`` tag alone instead, the prompt it trained page transcription on.
* **Page image** — the benchmark ships PDFs only. Pages are rasterized so the longest side
  is ``target_longest_image_dim`` pixels, the official recipe, but with MuPDF instead of
  poppler. 2048 is what the official runners use for API models; each model's own
  preprocessor downscales from there.
* **Sampling** — greedy, one generation per page (the published rows use temperature 0.1).

Data is fetched from the Hub at a pinned revision; set ``OLMOCR_BENCH_DIR`` to a local
``bench_data`` directory to read it from disk instead.
"""

from __future__ import annotations

import functools
import json
import os
from collections.abc import Iterator
from pathlib import Path

from olmo_eval.common.metrics.base import Metric
from olmo_eval.common.types import Instance, SamplingParams, Split
from olmo_eval.evals.tasks.common import register
from olmo_eval.evals.vision.scoring.olmocr_bench import (
    BASELINE_CATEGORY,
    OlmocrBenchCategoryMetric,
    OlmocrBenchOverallMetric,
    OlmocrBenchScorer,
    OlmocrBenchTestErrorsMetric,
    OlmocrBenchTestTypeMetric,
    ensure_olmocr_bench_runtime,
)
from olmo_eval.evals.vision.tasks.ocr import OcrTask

HF_REPO = "allenai/olmOCR-bench"
HF_REVISION = "54a96a6fb6a2bd3b297e59869491db4d3625b711"

#: ``olmocr/bench/prompts.py::build_basic_prompt``, verbatim.
BASIC_PROMPT = (
    "Please provide a natural, plain text representation of the document, formatted in "
    "Markdown. Skip any headers and footers. For ALL mathematical expressions, use LaTeX "
    "notation with \\( and \\) for inline equations and \\[ and \\] for display equations. "
    "Convert any tables into Markdown format."
)

#: Test categories = the benchmark's JSONL files, plus the synthesized baseline tests.
CATEGORIES: tuple[str, ...] = (
    "arxiv_math",
    "headers_footers",
    "long_tiny_text",
    "multi_column",
    "old_scans",
    "old_scans_math",
    "table_tests",
    BASELINE_CATEGORY,
)
TEST_TYPES: tuple[str, ...] = ("present", "absent", "order", "table", "math", "baseline")

_SCORER = OlmocrBenchScorer()
_OVERALL = OlmocrBenchOverallMetric(name="overall", scorer=_SCORER)
_METRICS: tuple[Metric, ...] = (
    _OVERALL,
    *(OlmocrBenchCategoryMetric(name=c, scorer=_SCORER, category=c) for c in CATEGORIES),
    *(
        OlmocrBenchTestTypeMetric(name=f"type_{t}", scorer=_SCORER, test_type=t)
        for t in TEST_TYPES
        if t != "baseline"
    ),
    OlmocrBenchTestErrorsMetric(name="n_test_errors", scorer=_SCORER),
)


def _render_pdf_page(pdf_path: str, target_longest_image_dim: int):
    """Rasterize a single-page PDF to a PIL image (module-level so it stays picklable)."""
    import pymupdf
    from PIL import Image

    with pymupdf.open(pdf_path) as doc:
        page = doc[0]
        zoom = target_longest_image_dim / max(page.rect.width, page.rect.height)
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
        return Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)


def _bench_data_dir() -> Path:
    local = os.environ.get("OLMOCR_BENCH_DIR")
    if local:
        return Path(local)
    from huggingface_hub import snapshot_download

    root = snapshot_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        revision=HF_REVISION,
        allow_patterns=["bench_data/*.jsonl", "bench_data/pdfs/**/*.pdf"],
    )
    return Path(root) / "bench_data"


@register("olmocr_bench")
class OlmocrBenchTask(OcrTask):
    #: The official scorer, pinned to the release behind the published numbers; its
    #: ``bench`` extra omits numpy, which the scorer imports.
    dependencies = ["pillow", "pymupdf", "huggingface-hub", "olmocr[bench]==0.4.27", "numpy"]
    sampling_params = SamplingParams(temperature=0.0, max_tokens=8000)
    metrics = _METRICS
    primary_metric = _OVERALL
    split = Split.TEST
    #: Longest side, in pixels, of the rasterized page handed to the model.
    target_longest_image_dim: int = 2048

    def _build_instances(self) -> Iterator[Instance]:
        ensure_olmocr_bench_runtime()
        bench_dir = _bench_data_dir()
        pdf_dir = bench_dir / "pdfs"

        tests_by_pdf: dict[str, list[dict]] = {}
        for jsonl in sorted(bench_dir.glob("*.jsonl")):
            with open(jsonl) as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        tests_by_pdf.setdefault(row["pdf"], []).append(
                            {"category": jsonl.stem, "test": row}
                        )

        for pdf_path in sorted(pdf_dir.rglob("*.pdf")):
            pdf = pdf_path.relative_to(pdf_dir).as_posix()
            tests = list(tests_by_pdf.get(pdf, ()))
            if not any(t["test"]["type"] == "baseline" for t in tests):
                baseline = {"pdf": pdf, "page": 1, "id": f"{pdf}_baseline", "type": "baseline"}
                tests.append({"category": BASELINE_CATEGORY, "test": baseline})
            yield Instance(
                question=self._question(BASIC_PROMPT),
                gold_answer=None,
                metadata={
                    "id": pdf,
                    "example_id": pdf,
                    "pdf": pdf,
                    "tests": tests,
                    "image": functools.partial(
                        _render_pdf_page, str(pdf_path), self.target_longest_image_dim
                    ),
                },
            )
