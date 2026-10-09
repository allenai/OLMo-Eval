"""Scorer for olmOCR-bench: run a page's unit tests against the model's markdown.

olmOCR-bench (https://huggingface.co/datasets/allenai/olmOCR-bench) grades a page by a set
of binary unit tests — text presence/absence, reading order, table cell relations, rendered
math equivalence, and a degenerate-output ``baseline`` check. The tests are executed by the
official implementation (``olmocr.bench.tests``), so a pass here means exactly what it means
on the published leaderboard.

Math tests compare KaTeX renderings, which the official code produces in a headless Chromium
through Playwright. :func:`ensure_olmocr_bench_runtime` verifies that stack up front and
installs the browser when it is missing, so a run fails before inference rather than scoring
every math test as a failure afterwards.

Required ``instance.metadata``:

==========  ==========================================================================
``tests``   list of ``{"category": str, "test": dict}``; ``test`` is one verbatim JSONL
            row of the benchmark (or a synthesized ``baseline`` row)
==========  ==========================================================================

The per-test outcomes are recorded as the scorer's result (saved with the predictions) for
:class:`OlmocrBenchCategoryMetric` / :class:`OlmocrBenchOverallMetric` to aggregate; the
scorer itself returns the fraction of the page's tests that passed. A test that raises counts
as failed, as in the official runner, and is also counted by :class:`OlmocrBenchTestErrorsMetric`,
so a broken renderer cannot pass for model failures.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import threading
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from olmo_eval.common.metrics.base import Metric
from olmo_eval.common.scorers.base import Scorer, get_scorer_result, set_scorer_result
from olmo_eval.common.types import Instance, LMOutput, Response
from olmo_eval.evals.vision.scoring.common import response_text

logger = logging.getLogger(__name__)

SCORER_NAME = "olmocr_bench"

#: Pseudo-category the official aggregation assigns the synthesized per-PDF baseline tests.
BASELINE_CATEGORY = "baseline"

_RUNTIME_LOCK = threading.Lock()
_runtime_ready = False

#: Renders a reference equation, exiting non-zero when the browser cannot be started. It
#: runs in a fresh interpreter because the renderer keeps one Playwright per pool thread and
#: a failed launch leaves that thread unusable, so a probe must not share threads with the
#: retry after installing the browser, or with the scoring that follows.
_PROBE_SCRIPT = r"""
import sys
from olmocr.bench.katex.render import render_equation
sys.exit(0 if render_equation(r"e^{i\pi} + 1 = 0", use_cache=False) is not None else 1)
"""


def _probe_render() -> tuple[bool, str]:
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE_SCRIPT], capture_output=True, text=True, timeout=600
    )
    return proc.returncode == 0, (proc.stderr or proc.stdout).strip()[-1500:]


def ensure_olmocr_bench_runtime() -> None:
    """Verify the official scorer can run, installing the headless browser if needed.

    Raises ``RuntimeError`` when the scorer package is missing or equations still cannot be
    rendered after installing Chromium.
    """
    global _runtime_ready
    with _RUNTIME_LOCK:
        if _runtime_ready:
            return
        try:
            import olmocr.bench.tests  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "olmOCR-bench scoring needs the official scorer: pip install 'olmocr[bench]' numpy"
            ) from exc

        ready, detail = _probe_render()
        if not ready:
            logger.info("KaTeX render probe failed; installing Chromium. %s", detail)
            for extra in ([], ["--with-deps"]):
                cmd = [sys.executable, "-m", "playwright", "install", *extra, "chromium"]
                logger.info("Running: %s", " ".join(cmd))
                subprocess.run(cmd, check=False)
                ready, detail = _probe_render()
                if ready:
                    break
                logger.warning("KaTeX render probe still failing: %s", detail)
        if not ready:
            raise RuntimeError(
                "olmOCR-bench math tests need a headless Chromium for KaTeX rendering and it "
                "could not be started. Install it with "
                f"`python -m playwright install --with-deps chromium`. Last error:\n{detail}"
            )
        _runtime_ready = True


def run_page_tests(tests: Sequence[dict[str, Any]], markdown: str) -> list[dict[str, Any]]:
    """Run every unit test of one page; an exception inside a test counts as a failure."""
    from olmocr.bench.tests import load_single_test

    results: list[dict[str, Any]] = []
    for entry in tests:
        row = entry["test"]
        raised = False
        try:
            passed, explanation = load_single_test(dict(row)).run(markdown)
        except Exception as exc:
            passed, explanation, raised = False, f"{type(exc).__name__}: {exc}", True
        results.append(
            {
                "id": row["id"],
                "type": row["type"],
                "category": entry["category"],
                "passed": bool(passed),
                "explanation": explanation,
                "raised": raised,
            }
        )
    return results


@dataclass(frozen=True, slots=True)
class OlmocrBenchScorer(Scorer):
    """Fraction of a page's olmOCR-bench unit tests that the output passes."""

    name: str = SCORER_NAME

    def score(self, instance: Instance, output: LMOutput) -> float:
        ensure_olmocr_bench_runtime()
        results = run_page_tests(instance.metadata["tests"], response_text(output))
        passed = sum(r["passed"] for r in results)
        set_scorer_result(
            output, self.name, {"tests": results, "passed": passed, "total": len(results)}
        )
        return passed / len(results) if results else 0.0


def _page_tests(response: Response) -> list[dict[str, Any]] | None:
    """The page's test outcomes, or ``None`` when it was not scored."""
    if not response.outputs:
        return None
    result = get_scorer_result(response.outputs[0], SCORER_NAME)
    return result["tests"] if result else None


def _test_results(responses: Sequence[Response]) -> Iterator[dict[str, Any]]:
    for response in responses:
        yield from _page_tests(response) or ()


def _pass_rate(tests: Sequence[dict[str, Any]]) -> float | None:
    return sum(t["passed"] for t in tests) / len(tests) if tests else None


def _category_pass_rates(
    tests: Iterator[dict[str, Any]] | Sequence[dict[str, Any]],
) -> dict[str, float]:
    by_category: dict[str, list[dict[str, Any]]] = {}
    for test in tests:
        by_category.setdefault(test["category"], []).append(test)
    return {category: _pass_rate(group) or 0.0 for category, group in by_category.items()}


@dataclass(frozen=True)
class OlmocrBenchCategoryMetric(Metric):
    """Pass rate over the unit tests of one category (one JSONL file, or ``baseline``)."""

    name: str  # type: ignore[misc]
    scorer: Scorer  # type: ignore[misc]
    category: str = ""

    def compute(self, responses: Sequence[Response]) -> float:
        return _category_pass_rates(_test_results(responses)).get(self.category, 0.0)

    def compute_instance(self, response: Response) -> float | None:
        """The page's pass rate over its tests of this category; ``None`` if it has none."""
        tests = _page_tests(response) or ()
        return _pass_rate([t for t in tests if t["category"] == self.category])

    def supports_pairwise_scorer_fallback(self) -> bool:
        return False


@dataclass(frozen=True)
class OlmocrBenchOverallMetric(Metric):
    """The leaderboard number: unweighted mean of the per-category pass rates."""

    name: str  # type: ignore[misc]
    scorer: Scorer  # type: ignore[misc]

    def compute(self, responses: Sequence[Response]) -> float:
        rates = _category_pass_rates(_test_results(responses))
        return sum(rates.values()) / len(rates) if rates else 0.0

    def compute_instance(self, response: Response) -> float | None:
        """The same mean over the page's own categories (its file's tests and its baseline)."""
        tests = _page_tests(response)
        if not tests:
            return None
        rates = _category_pass_rates(tests)
        return sum(rates.values()) / len(rates)

    def supports_pairwise_scorer_fallback(self) -> bool:
        return False


@dataclass(frozen=True)
class OlmocrBenchTestTypeMetric(Metric):
    """Pass rate over every unit test of one type, across categories."""

    name: str  # type: ignore[misc]
    scorer: Scorer  # type: ignore[misc]
    test_type: str = ""

    def compute(self, responses: Sequence[Response]) -> float:
        vals = [t["passed"] for t in _test_results(responses) if t["type"] == self.test_type]
        return sum(vals) / len(vals) if vals else 0.0

    def compute_instance(self, response: Response) -> float | None:
        """The page's pass rate over its tests of this type; ``None`` if it has none."""
        tests = _page_tests(response) or ()
        return _pass_rate([t for t in tests if t["type"] == self.test_type])

    def supports_pairwise_scorer_fallback(self) -> bool:
        return False


@dataclass(frozen=True)
class OlmocrBenchTestErrorsMetric(Metric):
    """Unit tests that raised instead of returning a result (counted as failed, as officially).

    A non-zero count points at the test runtime (for example the KaTeX renderer) rather than
    the model.
    """

    name: str  # type: ignore[misc]
    scorer: Scorer  # type: ignore[misc]

    def compute(self, responses: Sequence[Response]) -> float:
        return float(sum(t.get("raised", False) for t in _test_results(responses)))

    def compute_instance(self, response: Response) -> float | None:
        tests = _page_tests(response)
        return float(sum(t.get("raised", False) for t in tests)) if tests is not None else None

    def supports_pairwise_scorer_fallback(self) -> bool:
        return False

    def pairwise_higher_is_better(self) -> bool:
        return False
