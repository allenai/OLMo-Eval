"""Adversarial validation harness for the citation-grounding scorer.

Motivation: the citation scorer must not become an unvalidated oracle that
rewards citation-shaped prose. This module constructs adversarial cited-report
cases with KNOWN-correct expected behavior and checks that the scorer penalizes
them. It has two layers:

1. Pipeline layer (deterministic, CI-safe): run each case through an oracle judge
   that returns the case's ground-truth verdicts, and assert the scoring pipeline
   (aggregation, half-credit, precision/recall) converts correct verdicts into
   the expected scores. This proves the plumbing, GIVEN a correct judge. See
   tests/core/test_citation_validation.py.

2. Judge layer (needs OPENAI_API_KEY or OPENROUTER_API_KEY, run manually): run
   the same cases through real judges to test whether they ARE correct judges on
   these adversarial cases. Run `python -m olmo_eval.common.scorers.citation_validation` to sweep a
   ladder of judges; pass rung specs as args (`model` or `model:effort`, e.g.
   `gpt-5.5:low`) and `--repeat N` to average over runs (the gpt-5 series runs at
   temperature 1, so single runs are noisy).

The adversarial half needs no human labels (expected behavior is known by
construction). The complementary human-agreement study (does the scorer agree
with human citation-faithfulness labels on real scientific text) is scaffolded at
the bottom but needs labeled data; see plans/002_science_literature_evals.md.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from olmo_eval.common.scorers.citation import score_citations_for_sections
from olmo_eval.common.scorers.llm_judge import JudgeFn, build_openai_judge_fn

# An expectation on a resulting metric: (metric_name, op, threshold), op in {">=", "<="}.
Expectation = tuple[str, str, float]


@dataclass(frozen=True)
class AdversarialCase:
    """A constructed cited-report case with known-correct expected scoring.

    Attributes:
        name: Short identifier.
        description: What failure mode this probes.
        parsed_response: Report in the scorer's sections format.
        ground_truth_claims: Claims JSON a correct judge would return for the
            single section group (used by the oracle judge).
        expectations: Bounds the resulting scores must satisfy.
    """

    name: str
    description: str
    parsed_response: dict[str, Any]
    ground_truth_claims: list[dict[str, Any]]
    expectations: tuple[Expectation, ...]


def _section(text: str, citations: list[dict[str, Any]]) -> dict[str, Any]:
    return {"sections": [{"title": "", "text": text, "citations": citations}]}


def _claim(text: str, supporting: list[str], non_supporting: list[str], supported: bool) -> dict:
    return {
        "text": text,
        "supporting": supporting,
        "non_supporting": non_supporting,
        "is_fully_supported": supported,
    }


def build_cases() -> list[AdversarialCase]:
    """Construct the adversarial validation cases."""
    return [
        AdversarialCase(
            name="supporting_control",
            description="Citations genuinely support their claims; should score high.",
            parsed_response=_section(
                "Transformers rely on self-attention to model dependencies [1]. "
                "Batch normalization stabilizes training of deep networks [2].",
                [
                    {
                        "id": "[1]",
                        "snippets": [
                            "The Transformer relies entirely on self-attention to draw "
                            "global dependencies between input and output."
                        ],
                        "title": "Attention Is All You Need",
                    },
                    {
                        "id": "[2]",
                        "snippets": [
                            "Batch Normalization allows higher learning rates and "
                            "stabilizes training of deep neural networks."
                        ],
                        "title": "Batch Normalization",
                    },
                ],
            ),
            ground_truth_claims=[
                _claim("Transformers rely on self-attention.", ["[1]"], [], True),
                _claim("Batch normalization stabilizes training.", ["[2]"], [], True),
            ],
            expectations=(
                ("citation_precision", ">=", 0.99),
                ("citation_recall", ">=", 0.99),
            ),
        ),
        AdversarialCase(
            name="topical_non_supporting",
            description="Citation is on-topic but does not support the specific claim.",
            parsed_response=_section(
                "Transformers rely on self-attention to model long-range dependencies [1].",
                [
                    {
                        "id": "[1]",
                        "snippets": [
                            "Recurrent neural networks process sequences step by step and "
                            "struggle with long-range dependencies due to vanishing gradients."
                        ],
                        "title": "On the difficulty of training recurrent networks",
                    }
                ],
            ),
            ground_truth_claims=[
                _claim("Transformers rely on self-attention.", [], ["[1]"], False),
            ],
            expectations=(
                ("citation_precision", "<=", 0.1),
                ("citation_recall", "<=", 0.1),
            ),
        ),
        AdversarialCase(
            name="title_only_half_credit",
            description="Citation has only a title (no usable snippet); should earn half credit.",
            parsed_response=_section(
                "Self-attention enables parallel computation over sequence positions [1].",
                [{"id": "[1]", "snippets": [], "title": "Attention Is All You Need"}],
            ),
            ground_truth_claims=[
                _claim("Self-attention enables parallel computation.", ["[1]"], [], True),
            ],
            expectations=(
                ("citation_precision", ">=", 0.4),
                ("citation_precision", "<=", 0.6),
                ("citation_recall", ">=", 0.4),
                ("citation_recall", "<=", 0.6),
            ),
        ),
        AdversarialCase(
            name="citation_stuffed",
            description="One supporting citation plus four irrelevant ones; precision should drop.",
            parsed_response=_section(
                "Self-attention computes pairwise interactions between all positions "
                "[1][2][3][4][5].",
                [
                    {
                        "id": "[1]",
                        "snippets": [
                            "Self-attention relates all positions of a sequence to compute "
                            "a representation of that sequence."
                        ],
                        "title": "Attention Is All You Need",
                    },
                    {"id": "[2]", "snippets": ["Batch normalization normalizes layer inputs."]},
                    {
                        "id": "[3]",
                        "snippets": ["Dropout randomly zeroes activations at train time."],
                    },
                    {"id": "[4]", "snippets": ["Random crops and flips augment image datasets."]},
                    {"id": "[5]", "snippets": ["SGD with momentum accelerates gradient descent."]},
                ],
            ),
            ground_truth_claims=[
                _claim(
                    "Self-attention computes pairwise interactions.",
                    ["[1]"],
                    ["[2]", "[3]", "[4]", "[5]"],
                    True,
                ),
            ],
            expectations=(
                ("citation_precision", "<=", 0.3),
                ("citation_recall", ">=", 0.9),
            ),
        ),
        AdversarialCase(
            name="shuffled_citations",
            description="Each claim cites the other's evidence; both should be non-supporting.",
            parsed_response=_section(
                "Dropout reduces overfitting in neural networks [1]. "
                "Residual connections ease optimization of very deep networks [2].",
                [
                    {
                        "id": "[1]",
                        "snippets": [
                            "Residual connections let gradients flow through very deep "
                            "networks, easing their optimization."
                        ],
                        "title": "Deep Residual Learning",
                    },
                    {
                        "id": "[2]",
                        "snippets": [
                            "Dropout prevents co-adaptation of feature detectors and "
                            "reduces overfitting."
                        ],
                        "title": "Dropout",
                    },
                ],
            ),
            ground_truth_claims=[
                _claim("Dropout reduces overfitting.", [], ["[1]"], False),
                _claim("Residual connections ease optimization.", [], ["[2]"], False),
            ],
            expectations=(
                ("citation_precision", "<=", 0.1),
                ("citation_recall", "<=", 0.1),
            ),
        ),
        AdversarialCase(
            name="uncited_claim",
            description="A claim with no citation at all; should be extrapolatory (zero recall).",
            parsed_response=_section(
                "Large language models exhibit emergent few-shot capabilities.", []
            ),
            ground_truth_claims=[],  # no citations -> the judge is not consulted
            expectations=(
                ("citation_precision", "<=", 0.1),
                ("citation_recall", "<=", 0.1),
            ),
        ),
    ]


def make_oracle_judge(case: AdversarialCase) -> JudgeFn:
    """A judge that returns the case's ground-truth verdicts, ignoring the prompt.

    Lets the deterministic layer exercise the full scoring pipeline as if the
    judge were perfectly correct. Only valid for single-section cases (one group).
    """

    async def judge(prompt: str, **kwargs: Any) -> str:
        return json.dumps({"claims": case.ground_truth_claims})

    return judge


def check_expectations(scores: dict[str, float], expectations: Sequence[Expectation]) -> list[str]:
    """Return a list of human-readable failures (empty means all satisfied)."""
    failures: list[str] = []
    for metric, op, threshold in expectations:
        value = scores.get(metric, 0.0)
        ok = value >= threshold if op == ">=" else value <= threshold
        if not ok:
            failures.append(f"{metric}={value:.3f} not {op} {threshold}")
    return failures


@dataclass
class CaseResult:
    name: str
    description: str
    scores: dict[str, float]
    failures: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures


async def run_case(case: AdversarialCase, judge_fn: JudgeFn) -> CaseResult:
    """Score one case with the given judge and check its expectations."""
    scores = await score_citations_for_sections(judge_fn, case.parsed_response)
    return CaseResult(
        name=case.name,
        description=case.description,
        scores=scores,
        failures=check_expectations(scores, case.expectations),
    )


async def run_validation(
    judge_fn: JudgeFn, cases: list[AdversarialCase] | None = None
) -> list[CaseResult]:
    """Run every case through a single judge (use the real judge for the kill test)."""
    cases = cases if cases is not None else build_cases()
    return [await run_case(case, judge_fn) for case in cases]


def format_report(results: Sequence[CaseResult]) -> str:
    """Render a pass/fail report."""
    lines = []
    passed = sum(r.passed for r in results)
    lines.append(f"Citation scorer adversarial validation: {passed}/{len(results)} cases passed\n")
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        scores = ", ".join(f"{k}={v:.3f}" for k, v in sorted(r.scores.items()))
        lines.append(f"[{status}] {r.name}: {scores}")
        for failure in r.failures:
            lines.append(f"         - {failure}")
    return "\n".join(lines)


# =============================================================================
# Judge ladder: run the kill test across a progression of judge models
# =============================================================================

# Default progression, cheapest -> most expensive (per OpenAI list pricing).
# Override by passing rung specs as CLI args. A rung is "model" or "model:effort"
# (reasoning effort for the gpt-5 series, e.g. "gpt-5.5:low"). Unavailable model
# ids / unsupported efforts are reported as ERROR, not fatal, so it is safe to
# include rungs you may not have. Suggested workflow: sweep models at default
# effort to find the cheapest that passes, then effort-tune that winner.
#
# Judge selection from a representative sweep (--repeat 5, cases passing all runs):
# gpt-4o-mini accepts wrong/irrelevant evidence and must not be used for citation
# judging; gpt-5-mini is the cheapest reliable choice for iteration; gpt-5.5:medium
# was the only fully-clean config. Re-run the sweep below to refresh these for new
# models rather than trusting these figures as current.
DEFAULT_JUDGE_LADDER = [
    "gpt-5-nano",
    "gpt-5.4-nano",
    "gpt-5-mini",
    "gpt-5.4-mini",
    "gpt-5",
    "gpt-5.4",
    "gpt-5.5",
    "gpt-5.5-pro",
]


def _parse_rung(spec: str) -> tuple[str, str | None]:
    """Parse a rung spec 'model' or 'model:effort' into (model, effort)."""
    model, sep, effort = spec.partition(":")
    return model, (effort if sep else None)


@dataclass
class RungReport:
    """Results for one ladder rung (a model + optional reasoning effort), possibly
    over multiple repeats so per-case pass rates capture run-to-run variance."""

    label: str  # the rung spec, e.g. "gpt-5.5:low"
    model: str
    effort: str | None
    runs: list[list[CaseResult]]  # one CaseResult list per repeat
    error: str | None = None

    @property
    def total_cases(self) -> int:
        return len(self.runs[0]) if self.runs else 0

    def case_pass_counts(self) -> list[tuple[str, int, int]]:
        """Per case: (name, runs_passed, runs_total), aggregated by case index."""
        if not self.runs:
            return []
        n = len(self.runs)
        return [
            (self.runs[0][i].name, sum(run[i].passed for run in self.runs), n)
            for i in range(self.total_cases)
        ]

    @property
    def reliable(self) -> int:
        """Cases that passed on every repeat."""
        return sum(1 for _, passes, total in self.case_pass_counts() if passes == total)


async def run_ladder(
    specs: Sequence[str], repeat: int = 1, cases: list[AdversarialCase] | None = None
) -> list[RungReport]:
    """Run the kill test for each rung, `repeat` times. A rung that errors (e.g.
    unavailable model) is recorded and does not abort the ladder."""
    reports: list[RungReport] = []
    for spec in specs:
        model, effort = _parse_rung(spec)
        judge = build_openai_judge_fn(
            model=model,
            reasoning_effort=effort,
            scorer_name=f"citation_validation[{spec}]",
            max_tokens=4096,
        )
        runs: list[list[CaseResult]] = []
        error: str | None = None
        try:
            for _ in range(repeat):
                runs.append(await run_validation(judge, cases=cases))
        except Exception as e:  # surface API/model errors per rung, keep climbing
            error = str(e)
        reports.append(RungReport(spec, model, effort, runs, error))
    return reports


def format_ladder_report(reports: Sequence[RungReport]) -> str:
    """Render per-rung, per-case pass rates plus a one-line summary."""
    lines: list[str] = []
    for rep in reports:
        if rep.error is not None:
            lines.append(f"=== {rep.label}: ERROR — {rep.error}")
            continue
        n = len(rep.runs)
        lines.append(
            f"=== {rep.label}: {rep.reliable}/{rep.total_cases} cases passed all {n} run(s)"
        )
        for name, passes, total in rep.case_pass_counts():
            flag = "" if passes == total else "  <- flaky"
            lines.append(f"  [{passes}/{total}] {name}{flag}")
    summary = " | ".join(
        f"{r.label} ERROR" if r.error is not None else f"{r.label} {r.reliable}/{r.total_cases}"
        for r in reports
    )
    lines.append("")
    lines.append(f"Ladder summary (cases passing every run): {summary}")
    return "\n".join(lines)


# =============================================================================
# Human-agreement audit scaffold (needs labeled data; not yet populated)
# =============================================================================


@dataclass
class LabeledCitationExample:
    """A human-labeled citation-faithfulness example for scorer agreement studies.

    `human_supported` is the human judgment of whether the section's cited claims
    are genuinely supported by the cited snippets.
    """

    section_text: str
    citations: list[dict[str, Any]]
    human_supported: bool
    note: str = ""


# TODO: populate from 50-100 human-labeled ScholarQA/ExpertQA outputs across
# 2-3 models, including the adversarial variants above. See the review gate in
# plans/002_science_literature_evals.md. Empty until that labeling exists.
AUDIT_SET: list[LabeledCitationExample] = []

# Recall above this is treated as the scorer calling a group "supported" when
# computing agreement. Crude placeholder; revisit when the audit set exists.
_SUPPORTED_RECALL_THRESHOLD = 0.5


async def citation_scorer_agreement(
    examples: Sequence[LabeledCitationExample], judge_fn: JudgeFn
) -> dict[str, float]:
    """Agreement between the scorer's supported/not verdict and human labels.

    Returns accuracy and counts. A scaffold: with no labeled data this returns
    zeros. The supported/not threshold is a placeholder pending the audit design.
    """
    if not examples:
        return {"accuracy": 0.0, "n": 0.0}

    correct = 0
    for ex in examples:
        parsed = _section(ex.section_text, ex.citations)
        scores = await score_citations_for_sections(judge_fn, parsed)
        scorer_supported = scores.get("citation_recall", 0.0) >= _SUPPORTED_RECALL_THRESHOLD
        correct += int(scorer_supported == ex.human_supported)
    return {"accuracy": correct / len(examples), "n": float(len(examples))}


if __name__ == "__main__":
    import asyncio
    import sys

    async def _main() -> None:
        argv = sys.argv[1:]
        repeat = 1
        specs: list[str] = []
        i = 0
        while i < len(argv):
            if argv[i] in ("--repeat", "-r"):
                repeat = int(argv[i + 1])
                i += 2
            else:
                specs.append(argv[i])
                i += 1
        reports = await run_ladder(specs or DEFAULT_JUDGE_LADDER, repeat=repeat)
        print(format_ladder_report(reports))

    asyncio.run(_main())
