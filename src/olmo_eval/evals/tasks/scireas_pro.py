"""SciReas-Pro: reasoning-intensive scientific problems (arXiv 2508.19202).

SciReas-Pro is the hard subset of SciReas from "Demystifying Scientific
Problem-Solving in LLMs by Probing Knowledge and Reasoning" (Li et al., 2025). It
holds 1,356 items drawn from eight existing benchmarks, kept where the authors
judged the item reasoning-intensive (o3-mini and o4-mini answer agreement plus an
LLM screen). Rows come from ``yale-nlp/SciReas-Pro`` (the authors' harness loads it
under its old name ``lihaoxin2020/SciReas-Pro``) pinned to one revision. Each row is
the rendered prompt the authors' harness sends, the gold answer, and the source
task and document index.

======================  =====  ==========================================
Source                  Items  Answer format
======================  =====  ==========================================
SuperGPQA (sci + eng)     702  multiple choice, up to ten options
LAB-Bench                 197  multiple choice
UGPhysics                 196  boxed free-form (some boxed letters)
OlympiadBench (en, TO)    117  boxed free-form math and physics
MMLU-Pro                   42  multiple choice, ten options
GPQA (main)                41  multiple choice, four options
SciBench                   40  boxed numeric
SciEval                    21  multiple choice; 11 name a molecule (SMILES)
======================  =====  ==========================================

Tasks
-----
``scireas_pro``
    All 1,356 items, in dataset order (grouped by source). ``-o limit=N`` draws a
    seeded random sample, which is not stratified.

``scireas_pro:fast500``
    A fixed 500-item sample, stratified by source task (proportional allocation,
    largest remainder, seed 20261022), listed in ``scireas_pro_fast500_ids.json``
    in a seeded random order. Every model sees the same items.

Scoring
-------
Each item is scored the way the authors' harness scores its source benchmark
(``lihaoxin2020/SciReas-Eval``, ``scillm-pro`` task configs), and the primary
``accuracy`` is the mean over items, the suite's ``micro`` aggregate:

- GPQA, LAB-Bench and SciEval: oe-eval's answer-regex cascade
  (:func:`~olmo_eval.evals.extract.extract_answer_with_format`) with each source's
  format, prefix and answer regexes, scored as ``exact_match_flex`` (the extracted
  answer whatever its format score). SciEval's molecule items match the text after
  ``Answer:`` exactly, case-insensitively.
- MMLU-Pro and SuperGPQA: the MMLU-Pro reference extractor (format regex, then
  ``answer is``, then ``Answer:``, then the last standalone letter A-J).
- SciBench: the boxed (or Minerva-format) answer, matched symbolically, by string
  normalization, or numerically within 5% relative error.
- OlympiadBench: OlympiadBench's rule-based ``MathJudger`` on the text after
  ``Therefore, the final answer is:``, with each problem's own tolerance from the
  source dataset (``Hothan/OlympiadBench``).
- UGPhysics: the extracted answer is first compared by rule (``MathJudger`` at
  tolerance 1e-8); an answer that fails the rule check goes to a model judge with
  UGPhysics' judge prompt and the problem's reference solution (from
  ``UGPhysics/ugphysics``). An output with no extractable answer is wrong without
  a judge call. This is the only judged source: 196 of 1,356 items (14.5%) can
  reach the judge, and only when the rule check fails.

Reasoning and strict scoring: everything through the last ``</think>`` is dropped
before extraction. An output that never closes its reasoning scores 0 without
extraction: one with an opening ``<think>`` and no ``</think>``, or one with no
``</think>`` that stopped at the token limit (chat templates that open the block in
the prompt leave no ``<think>`` in the output, so the length stop is the only
sign). ``unclosed_think_rate`` reports the share of outputs with no ``</think>``.

Judge
-----
The authors judge with ``gpt-4o-2024-08-06``. The default here is the judge
olmo-eval uses for FrontierScience (``gpt-5.5-2026-04-23`` at high reasoning
effort), reached through ``OPENAI_API_KEY`` or ``OPENROUTER_API_KEY``. Override it
per task with ``-o judge_model=<model>`` or for a run with
``OLMO_EVAL_JUDGE=<model>[:<effort>]``. A judge failure is recorded as an
incomplete score and left out of the averages, never counted as wrong.

Deviations from the reference harness
-------------------------------------
- Generation defaults to temperature 0.6 (as the reference) with no token cap
  (``max_tokens=None``, generate to the context limit) instead of per-source caps
  of 16,384 or 32,768 tokens.
- The UGPhysics rule check uses OlympiadBench's ``MathJudger`` instead of
  UGPhysics' own rule judger (a sibling of it). Both send every rule miss to the
  model judge, so the judge decides the same items.
- The reference fills the UGPhysics judge prompt with ``{{problem}}`` while the
  template reads ``{problem}``, so its judge never sees the problem statement. The
  problem is filled in here.
- The judge model (see Judge) and strict scoring of unfinished reasoning.

Metrics: ``accuracy`` (primary, mean over items), ``accuracy_<source>``,
``macro_accuracy`` (unweighted mean over the eight sources), ``unclosed_think_rate``,
``truncation_rate`` and ``judge_call_rate`` (share of items sent to the judge).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
import re
import signal
import string
import threading
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from importlib.resources import files
from typing import Any, Literal

import sympy
from sympy.parsing.latex import parse_latex

from olmo_eval.common.scorers.llm_judge import JudgeFn
from olmo_eval.common.types import (
    Instance,
    LMOutput,
    LMRequest,
    RequestType,
    Response,
    SamplingParams,
    Split,
)
from olmo_eval.data import DataLoader, DataSource
from olmo_eval.evals.extract import extract_answer_with_format
from olmo_eval.evals.extract.math import (
    get_unnormalized_answer,
    hendrycks_is_equiv,
    minerva_is_equiv,
    normalize_final_answer,
)
from olmo_eval.evals.tasks._olympiadbench_judger import DEFAULT_PRECISION, MathJudger
from olmo_eval.evals.tasks.common import OutputScoreAggregation, register
from olmo_eval.evals.tasks.common.judge import (
    JudgedMacroScoreMetric,
    JudgedScoreMetric,
    JudgedTask,
    JudgeScorer,
    TruncationRateMetric,
    is_truncated,
    visible_response,
)
from olmo_eval.evals.tasks.frontierscience import (
    FRONTIERSCIENCE_DEFAULT_JUDGE_MODEL,
    FRONTIERSCIENCE_DEFAULT_JUDGE_REASONING_EFFORT,
)

logger = logging.getLogger(__name__)

SCIREAS_PRO_REPO = "yale-nlp/SciReas-Pro"
# Pinned so a Hub update cannot silently change the item set mid-comparison.
SCIREAS_PRO_REVISION = "795fe6cd17e94d43761e8c5f05fbe4ef7c63acad"
SCIREAS_PRO_SIZE = 1356

#: Per-problem tolerances and answer types for the OlympiadBench items.
OLYMPIADBENCH_REPO = "Hothan/OlympiadBench"
OLYMPIADBENCH_REVISION = "91184b52131e7fc9455fef848035173aea8cc01a"
OLYMPIADBENCH_FILES = {
    "olympiadbench_OE_TO_maths_en_COMP": (
        "OlympiadBench/OE_TO_maths_en_COMP/OE_TO_maths_en_COMP.parquet"
    ),
    "olympiadbench_OE_TO_physics_en_COMP": (
        "OlympiadBench/OE_TO_physics_en_COMP/OE_TO_physics_en_COMP.parquet"
    ),
}

#: Reference solutions for the UGPhysics judge.
UGPHYSICS_REPO = "UGPhysics/ugphysics"
UGPHYSICS_REVISION = "523e71bc4e33356bb19de4af38c128796e8a8770"

FAST500_FILE = "scireas_pro_fast500_ids.json"
FAST500_SIZE = 500
FAST500_SEED = 20261022

UGPHYSICS_JUDGE_PROMPT_FILE = "scireas_pro_ugphysics_judge_prompt.json"
UGPHYSICS_JUDGE_PROMPT_SHA256 = "b10fead779ce41172a710952500911f1d5d58eebc19c907d65b9305e3e3a94da"
#: Tolerance of the UGPhysics rule check, as the reference metric sets it.
UGPHYSICS_RULE_PRECISION = 1e-8

SCIREAS_PRO_JUDGE_MAX_TOKENS = 8192
SCIREAS_PRO_SCORER_NAME = "scireas_pro"

#: Seconds allowed for one symbolic comparison before it counts as no match.
RULE_TIMEOUT_SECONDS = 30

SOURCES: tuple[str, ...] = (
    "supergpqa",
    "lab_bench",
    "ugphysics",
    "olympiadbench",
    "mmlu_pro",
    "gpqa",
    "scibench",
    "scieval",
)

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def source_of(taskname: str) -> str:
    """Map a SciReas-Pro ``taskname`` such as ``supergpqa_Physics:cot`` to its source."""
    for source in sorted(SOURCES, key=lambda name: len(name), reverse=True):
        if taskname == source or taskname.startswith(f"{source}_"):
            return source
    raise ValueError(f"unknown SciReas-Pro source task {taskname!r}")


def item_id(taskname: str, doc_id: int) -> str:
    """The id of one item: its source task and its index in that task."""
    return f"{taskname}:{doc_id}"


# ---------------------------------------------------------------------------
# Rule-based scoring
# ---------------------------------------------------------------------------


def _call_with_timeout[T](
    fn: Callable[[], T], default: T, seconds: int = RULE_TIMEOUT_SECONDS
) -> T:
    """Run ``fn``, returning ``default`` if it takes longer than ``seconds``.

    Symbolic comparisons can stall on a pathological expression. The alarm only
    works on the main thread; elsewhere ``fn`` runs without a limit.
    """
    if threading.current_thread() is not threading.main_thread():
        return fn()

    def on_alarm(signum, frame):
        raise TimeoutError

    previous = signal.signal(signal.SIGALRM, on_alarm)
    signal.alarm(seconds)
    try:
        return fn()
    except TimeoutError:
        logger.warning("SciReas-Pro rule check timed out after %ds; counted as no match", seconds)
        return default
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


_ANSWER_PREFIX_REGEXES = (
    "(?i)Therefore,? the answer is",
    "(?i)the answer is",
    "(?i)answer is",
    "(?i)answer:",
)


@dataclass(frozen=True)
class MCQRule:
    """How the reference harness extracts and matches one source's choice answers."""

    #: ``cascade`` is oe-eval's regex cascade; ``mmlu_pro`` the MMLU-Pro extractor.
    style: Literal["cascade", "mmlu_pro"]
    answer_format_regex: str
    answer_regexes: tuple[str, ...] = ()
    #: SciEval drops ``*`` before extracting and reduces ``"B. text"`` to ``"B"``.
    scieval_cleanup: bool = False
    ignore_punctuation: bool = True


MCQ_RULES: dict[str, MCQRule] = {
    "gpqa": MCQRule(
        "cascade",
        r"(?i)answer: \(([A-D])\)",
        (r"\(([A-D])\)", r"\b([A-D])\b", r"\*([A-D])\*"),
    ),
    "lab_bench": MCQRule(
        "cascade",
        r"(?i)answer: ([A-Z])",
        (r"\(([A-Z])\)", r"\b([A-Z])\b", r"\*([A-Z])\*"),
    ),
    "scieval": MCQRule(
        "cascade",
        r"(?i)answer: (.*)",
        (r"\(([A-D])\)", r"\b([A-D])\b"),
        scieval_cleanup=True,
        ignore_punctuation=False,
    ),
    "mmlu_pro": MCQRule("mmlu_pro", r"(?i)answer: ([A-Z])"),
    "supergpqa": MCQRule("mmlu_pro", r"(?i)answer: \(([A-Z])\)"),
}

_SCIEVAL_LETTER_WITH_TEXT = re.compile(r"^([A-F])\.\s.*$")


def extract_choice(text: str, rule: MCQRule) -> str:
    """Extract an answer the way the reference harness does for this source."""
    if rule.style == "mmlu_pro":
        matches = re.findall(rule.answer_format_regex, text)
        if matches:
            answer = matches[-1]
        elif (match := re.search(r"(?i)answer is:?\s*\(?([A-J])\)?", text)) or (
            match := re.search(r".*[aA]nswer:\s*([A-J])", text)
        ):
            answer = match.group(1)
        else:
            letters = re.findall(r".*\b([A-J])\b", text)
            answer = letters[-1] if letters else ""
        return re.sub(r"\(|\)", "", answer)

    if rule.scieval_cleanup:
        text = text.replace("*", "")
    answer = extract_answer_with_format(
        text,
        answer_format_regex=rule.answer_format_regex,
        answer_regexes=rule.answer_regexes,
        prefix_regexes=_ANSWER_PREFIX_REGEXES,
    ).answer
    if rule.scieval_cleanup:
        if match := _SCIEVAL_LETTER_WITH_TEXT.match(answer.strip()):
            return match.group(1)
        return answer.strip()
    return re.sub(r"\(|\)", "", answer)


_PUNCTUATION = str.maketrans("", "", string.punctuation)


def choice_matches(prediction: str, gold: str, rule: MCQRule) -> bool:
    """Hugging Face ``exact_match`` with ``ignore_case`` and the source's punctuation rule."""
    prediction, gold = prediction.lower(), gold.lower()
    if rule.ignore_punctuation:
        prediction, gold = prediction.translate(_PUNCTUATION), gold.translate(_PUNCTUATION)
    return prediction == gold


def _brace_content(text: str, start: int) -> str | None:
    """Return the text inside the brace group opening just before ``start``."""
    depth = 1
    index = start
    while index < len(text):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start:index]
        index += 1
    return None


def last_boxed_content(text: str) -> str | None:
    """Content of the last ``\\boxed{...}``, or None if there is none or it never closes."""
    index = text.rfind("\\boxed{")
    if index < 0:
        return None
    return _brace_content(text, index + len("\\boxed{"))


def _scibench_boxed(text: str) -> str | None:
    """SciBench's boxed answer: the last ``oxed{...}``, right side of any ``=``."""
    index = text.rfind("oxed")
    if index < 0:
        return None
    opening = text.find("{", index)
    if opening < 0:
        return None
    content = _brace_content(text, opening + 1)
    if content is None or text[index : opening + 1] != "oxed{":
        return None
    if "=" in content:
        content = content.split("=")[-1].lstrip(" ")
    return content


def _close_equiv(x1: str, x2: str, rel_tol: float = 0.05) -> bool:
    """SciBench's tolerance match: equal within ``rel_tol`` once both parse as numbers."""
    return _call_with_timeout(lambda: _close_equiv_unguarded(x1, x2, rel_tol), default=False)


def _close_equiv_unguarded(x1: str, x2: str, rel_tol: float) -> bool:
    try:
        parsed_x1 = parse_latex(x1)
        parsed_x2 = parse_latex(x2)
    except Exception:
        return False
    try:
        return math.isclose(float(sympy.N(parsed_x1)), float(sympy.N(parsed_x2)), rel_tol=rel_tol)
    except (TypeError, ValueError, AttributeError):
        try:
            return bool(sympy.simplify(parsed_x1 - parsed_x2) == 0)
        except Exception:
            return False
    except Exception:
        return False


def scibench_candidates(text: str) -> list[str]:
    """Normalized answers SciBench's ``exact_match_flex`` tries, in order."""
    candidates: list[str] = []
    minerva = get_unnormalized_answer(text)
    if minerva != "[invalidanswer]":
        candidates.append(normalize_final_answer(minerva))
    boxed = _scibench_boxed(text)
    if boxed is not None:
        candidates.append(normalize_final_answer(boxed))
    else:
        dollars = [match.start() for match in re.finditer(r"\$", text)]
        if len(dollars) > 1:
            candidates.append(normalize_final_answer(text[dollars[-2] + 1 : dollars[-1]]))
    if not candidates:
        candidates.append(normalize_final_answer(text))
    return candidates


def scibench_correct(text: str, gold: str) -> tuple[bool, str]:
    """Whether a SciBench response matches, and the first extracted answer."""
    candidates = scibench_candidates(text)
    # minerva_is_equiv sets its own alarm, so only the tolerance check is guarded here.
    correct = any(
        minerva_is_equiv(answer, gold)
        or hendrycks_is_equiv(answer, gold)
        or _close_equiv(answer, gold)
        for answer in candidates
    )
    return correct, candidates[0]


_OLYMPIAD_FINAL_ANSWER = re.compile(r"(?i)Therefore, the final answer is: (.*)")


def olympiad_precision(answer_type: str, error: str | None) -> float | list[float]:
    """The tolerance OlympiadBench sets for a problem, as the reference metric reads it."""
    if "Tuple" in answer_type or not error:
        return DEFAULT_PRECISION
    if "," in error:
        return [float(part) if part else DEFAULT_PRECISION for part in error.split(",")]
    return float(error)


def olympiad_correct(text: str, gold: str, precision: float | list[float]) -> tuple[bool, str]:
    """Judge an OlympiadBench response; returns the verdict and the extracted answer."""
    matches = _OLYMPIAD_FINAL_ANSWER.findall(text)
    answer = matches[-1].strip() if matches else text
    verdict = _call_with_timeout(lambda: MathJudger().judge(answer, gold, precision), default=False)
    return verdict, answer


_UGPHYSICS_ANSWER_PHRASES = ("is the ans", "is our ans")


def ugphysics_extract(text: str) -> str:
    """UGPhysics' strict answer extraction: the boxed answer, else an explicit phrase.

    Returns "" when the response states no answer, which the reference scores wrong
    without asking the judge.
    """
    if "herefore" in text:
        text = text.split("herefore")[-1].strip()
    if "oxed{" in text:
        boxed = last_boxed_content(text)
        if boxed is None:
            first = text.find("oxed{")
            boxed = _brace_content(text, first + len("oxed{"))
        return (boxed or "").strip()
    for phrase in _UGPHYSICS_ANSWER_PHRASES:
        if phrase in text:
            return re.split(r"(,|\.|\!\|?)", text.split(phrase)[-2].strip())[-1].strip()
    for phrase in ("answer is", "answer:", "answer :"):
        if phrase in text:
            answer = text.split(phrase)[-1].strip()
            if answer.startswith("$") and answer.endswith("$"):
                answer = answer[1:-1]
            return answer
    return ""


def ugphysics_gold(answer: str) -> str:
    """The content of a UGPhysics gold answer such as ``\\boxed{0.0732, 574.2}``."""
    boxed = last_boxed_content(answer)
    return (boxed if boxed is not None else answer).strip()


def ugphysics_rule_correct(prediction: str, gold: str) -> bool:
    """The rule check before the judge: every answer pairs with an equal gold answer."""
    return _call_with_timeout(
        lambda: MathJudger().judge(prediction, gold, UGPHYSICS_RULE_PRECISION), default=False
    )


@cache
def ugphysics_judge_template() -> str:
    """UGPhysics' judge prompt, verbatim, checked against its recorded digest."""
    resource = files(__package__).joinpath(UGPHYSICS_JUDGE_PROMPT_FILE)
    template = json.loads(resource.read_text(encoding="utf-8"))["template"]
    digest = hashlib.sha256(template.encode("utf-8")).hexdigest()
    if digest != UGPHYSICS_JUDGE_PROMPT_SHA256:
        raise ValueError(f"{UGPHYSICS_JUDGE_PROMPT_FILE} does not match its recorded digest")
    return template


def ugphysics_judge_prompt(
    problem: str, solution: str, gold: str, response: str, prediction: str
) -> str:
    """Fill UGPhysics' judge prompt for one response."""
    judger = MathJudger()
    gold_answers = judger.split_by_comma(gold)
    predicted = judger.split_by_comma(prediction)
    return (
        ugphysics_judge_template()
        .replace("{problem}", problem)
        .replace("{{RS}}", solution)
        .replace("{{RA}}", ", ".join(gold_answers))
        .replace("{{SS}}", response)
        .replace("{{SA}}", ", ".join(predicted))
    )


_UGPHYSICS_VERDICT = re.compile(
    r"Equivalence\s+Judgement[\s*_#:]*\**\s*(TRUE|FALSE)\b", re.IGNORECASE
)


def parse_ugphysics_verdict(raw: str) -> bool | None:
    """Read the TRUE / FALSE equivalence verdict, or None if the reply has none."""
    matches = _UGPHYSICS_VERDICT.findall(raw)
    if not matches:
        return None
    return matches[-1].upper() == "TRUE"


# ---------------------------------------------------------------------------
# The fixed fast sample
# ---------------------------------------------------------------------------


def allocate_proportional(counts: Mapping[str, int], size: int) -> dict[str, int]:
    """Split ``size`` across strata in proportion to ``counts`` (largest remainder).

    Ties in the fractional part go to the stratum whose name sorts first.
    """
    total = sum(counts.values())
    if not 0 <= size <= total:
        raise ValueError(f"cannot draw {size} items from {total}")
    exact = {name: size * count / total for name, count in counts.items()}
    allocation = {name: math.floor(value) for name, value in exact.items()}
    remainder = size - sum(allocation.values())
    by_fraction = sorted(counts, key=lambda name: (allocation[name] - exact[name], name))
    for name in by_fraction[:remainder]:
        allocation[name] += 1
    return allocation


def stratified_sample_ids(items: Sequence[tuple[str, int]], size: int, seed: int) -> list[str]:
    """Draw a stratified sample of item ids, one stratum per source task.

    Each stratum gets its proportional share; within it, documents are drawn
    uniformly. The ids come back in a seeded random order, so a prefix of the list
    is itself a random sample.
    """
    by_task: dict[str, list[int]] = defaultdict(list)
    for taskname, doc_id in items:
        by_task[taskname].append(doc_id)
    allocation = allocate_proportional({name: len(ids) for name, ids in by_task.items()}, size)
    rng = random.Random(seed)
    picked: list[str] = []
    for taskname in sorted(by_task):
        drawn = rng.sample(sorted(by_task[taskname]), allocation[taskname])
        picked.extend(item_id(taskname, doc_id) for doc_id in sorted(drawn))
    rng.shuffle(picked)
    return picked


@cache
def fast500_ids() -> tuple[str, ...]:
    """The ``scireas_pro:fast500`` item ids, in their recorded order."""
    resource = files(__package__).joinpath(FAST500_FILE)
    ids = tuple(json.loads(resource.read_text(encoding="utf-8"))["ids"])
    if len(ids) != FAST500_SIZE or len(set(ids)) != FAST500_SIZE:
        raise ValueError(f"{FAST500_FILE} must list {FAST500_SIZE} distinct ids")
    return ids


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SciReasProScorer(JudgeScorer):
    """Serves each output's stored SciReas-Pro score.

    The pinned auxiliary sources and the judge prompt digest are fields so that
    they reach the task hash: they change scores without changing the requests.
    """

    name: str = SCIREAS_PRO_SCORER_NAME
    olympiadbench_revision: str = OLYMPIADBENCH_REVISION
    ugphysics_revision: str = UGPHYSICS_REVISION
    ugphysics_judge_prompt_sha256: str = UGPHYSICS_JUDGE_PROMPT_SHA256
    rule_timeout_seconds: int = RULE_TIMEOUT_SECONDS


SCIREAS_PRO_SCORER = SciReasProScorer()


@dataclass(frozen=True)
class LowerIsBetterRateMetric(JudgedScoreMetric):
    """A judged-task rate where lower is better."""

    def higher_is_better(self) -> bool:
        return False


ACCURACY = JudgedScoreMetric(name="accuracy", score_key="accuracy", scorer=SCIREAS_PRO_SCORER)
SCIREAS_PRO_METRICS = (
    ACCURACY,
    *(
        JudgedScoreMetric(
            name=f"accuracy_{source}",
            score_key="accuracy",
            metadata_key="source",
            metadata_value=source,
            scorer=SCIREAS_PRO_SCORER,
        )
        for source in SOURCES
    ),
    JudgedMacroScoreMetric(
        name="macro_accuracy",
        score_key="accuracy",
        metadata_key="source",
        groups=SOURCES,
        scorer=SCIREAS_PRO_SCORER,
    ),
    LowerIsBetterRateMetric(
        name="unclosed_think_rate", score_key="unclosed_think", scorer=SCIREAS_PRO_SCORER
    ),
    TruncationRateMetric(scorer=SCIREAS_PRO_SCORER),
    JudgedScoreMetric(name="judge_call_rate", score_key="judge_called", scorer=SCIREAS_PRO_SCORER),
)


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


def _load_olympiadbench(tasknames: set[str]) -> dict[str, list[dict[str, Any]]]:
    loader = DataLoader()
    rows: dict[str, list[dict[str, Any]]] = {}
    for taskname in sorted(tasknames):
        source = DataSource(
            path=OLYMPIADBENCH_REPO,
            data_files=OLYMPIADBENCH_FILES[taskname],
            revision=OLYMPIADBENCH_REVISION,
            split="train",
        )
        rows[taskname] = list(loader.load(source))
    return rows


def _load_ugphysics(subjects: set[str]) -> dict[str, dict[str, Any]]:
    """UGPhysics English rows keyed by stripped problem text."""
    source = DataSource(
        path=UGPHYSICS_REPO,
        data_files=tuple(f"{subject}/en.jsonl" for subject in sorted(subjects)),
        revision=UGPHYSICS_REVISION,
        split="train",
    )
    by_problem: dict[str, dict[str, Any]] = {}
    for row in DataLoader().load(source):
        problem = str(row.get("problem") or "").strip()
        if problem:
            by_problem.setdefault(problem, row)
    return by_problem


def _find_ugphysics_reference(
    question: str, by_problem: Mapping[str, dict[str, Any]]
) -> dict[str, Any] | None:
    """The UGPhysics row whose problem statement ends the prompt.

    A prompt is UGPhysics' instruction, a blank line, then the problem; the problem
    can itself contain blank lines, so each split point is tried in turn.
    """
    question = question.strip()
    start = question.find("\n\n")
    while start >= 0:
        reference = by_problem.get(question[start:].strip())
        if reference is not None:
            return reference
        start = question.find("\n\n", start + 2)
    return None


def attach_references(rows: list[dict[str, Any]]) -> None:
    """Add the source-dataset fields scoring needs to OlympiadBench and UGPhysics rows.

    OlympiadBench rows get ``answer_type`` and ``error`` (the problem's tolerance),
    joined by document index; UGPhysics rows get ``reference_problem`` and
    ``reference_solution``, joined by problem text. A row that does not join
    raises, since it would otherwise be scored against the wrong reference.
    """
    olympiad = {r["taskname"] for r in rows if source_of(r["taskname"]) == "olympiadbench"}
    ugphysics = {
        r["taskname"].split("_", 1)[1] for r in rows if source_of(r["taskname"]) == "ugphysics"
    }
    olympiad_rows = _load_olympiadbench(olympiad) if olympiad else {}
    ugphysics_rows = _load_ugphysics(ugphysics) if ugphysics else {}
    for row in rows:
        taskname = row["taskname"]
        source = source_of(taskname)
        if source == "olympiadbench":
            reference = olympiad_rows[taskname][int(row["doc_id"])]
            if json.dumps(list(reference["final_answer"])) != row["answer"]:
                missing = item_id(taskname, row["doc_id"])
                raise ValueError(f"OlympiadBench answer mismatch for {missing}")
            row["answer_type"] = str(reference.get("answer_type") or "")
            error = reference.get("error")
            row["error"] = error if isinstance(error, str) else None
        elif source == "ugphysics":
            reference = _find_ugphysics_reference(row["question"], ugphysics_rows)
            if reference is None or reference.get("answers") != row["answer"]:
                missing = item_id(taskname, row["doc_id"])
                raise ValueError(f"no UGPhysics reference for {missing}")
            row["reference_problem"] = str(reference["problem"])
            row["reference_solution"] = str(reference.get("solution") or "")


@register("scireas_pro")
class SciReasPro(JudgedTask):
    """All of SciReas-Pro, each item scored as its source benchmark scores it."""

    data_source = DataSource(path=SCIREAS_PRO_REPO, revision=SCIREAS_PRO_REVISION, split="train")
    split = Split.TRAIN
    metrics = SCIREAS_PRO_METRICS
    primary_metric = ACCURACY
    # The reference samples at 0.6; no cap means generating to the context limit.
    sampling_params = SamplingParams(temperature=0.6, max_tokens=None)
    output_score_aggregation = OutputScoreAggregation.MEAN

    judge_model = FRONTIERSCIENCE_DEFAULT_JUDGE_MODEL
    judge_reasoning_effort = FRONTIERSCIENCE_DEFAULT_JUDGE_REASONING_EFFORT
    judge_max_tokens = SCIREAS_PRO_JUDGE_MAX_TOKENS
    judge_scorer_name = SCIREAS_PRO_SCORER_NAME
    judge_extra_score_keys = ("unclosed_think", "judge_called")

    @property
    def instances(self) -> Iterator[Instance]:
        if self._instances_cache is None:
            rows = list(DataLoader().load(self.config.get_data_source()))
            if len(rows) != SCIREAS_PRO_SIZE:
                raise ValueError(
                    f"{SCIREAS_PRO_REPO}@{SCIREAS_PRO_REVISION} has {len(rows)} rows, "
                    f"expected {SCIREAS_PRO_SIZE}"
                )
            rows = self._select(rows)
            attach_references(rows)
            instances = [self.process_doc(row, index) for index, row in enumerate(rows)]
            self._instances_cache = [instance for instance in instances if instance is not None]
        yield from self._instances_cache

    def _select(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """The rows this task evaluates, in order."""
        return rows

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        question = str(doc.get("question") or "")
        answer = str(doc.get("answer") or "").strip()
        taskname = str(doc.get("taskname") or "")
        if not question.strip() or not answer or not taskname:
            return None
        doc_id = int(doc["doc_id"])
        metadata: dict[str, Any] = {
            "id": item_id(taskname, doc_id),
            "taskname": taskname,
            "doc_id": doc_id,
            "source": source_of(taskname),
            "index": index,
        }
        for key in ("answer_type", "error", "reference_problem", "reference_solution"):
            if key in doc:
                metadata[key] = doc[key]
        return Instance(question=question, gold_answer=answer, metadata=metadata)

    def format_request(self, instance: Instance) -> LMRequest:
        """Send the released prompt verbatim; it carries the source's answer instruction."""
        return LMRequest(
            request_type=RequestType.CHAT,
            messages=({"role": "user", "content": instance.question},),
        )

    async def _score_output(self, response: Response, output: LMOutput, judge_fn: JudgeFn) -> float:
        instance = response.instance
        source = instance.metadata["source"]
        gold = instance.gold_answer or ""
        raw = output.metadata.get("original_text") or output.text or ""
        closed = _THINK_CLOSE in raw
        judge_result: dict[str, Any] = {
            "scorer": SCIREAS_PRO_SCORER_NAME,
            "source": source,
            "method": "rule",
        }
        scores = {
            "accuracy": 0.0,
            "unclosed_think": 0.0 if closed else 1.0,
            "judge_called": 0.0,
        }

        unfinished = not closed and (_THINK_OPEN in raw or is_truncated(output))
        if unfinished:
            judge_result["method"] = "unfinished_reasoning"
            output.extracted_answer = None
            return self._record_judgement(output, judge_result, scores)

        text = visible_response(raw)
        correct: bool
        if source in MCQ_RULES:
            rule = MCQ_RULES[source]
            prediction = extract_choice(text, rule)
            correct = bool(prediction) and choice_matches(prediction, gold, rule)
        elif source == "scibench":
            correct, prediction = scibench_correct(text, gold)
        elif source == "olympiadbench":
            gold_answers = json.loads(gold)
            precision = olympiad_precision(
                str(instance.metadata.get("answer_type") or ""), instance.metadata.get("error")
            )
            correct, prediction = olympiad_correct(text, gold_answers[0], precision)
        elif source == "ugphysics":
            prediction = ugphysics_extract(text)
            gold_answer = ugphysics_gold(gold)
            if not prediction:
                correct = False
                judge_result["method"] = "no_answer"
            elif ugphysics_rule_correct(prediction, gold_answer):
                correct = True
            else:
                judge_result["method"] = "judge"
                scores["judge_called"] = 1.0
                prompt = ugphysics_judge_prompt(
                    problem=str(instance.metadata.get("reference_problem") or ""),
                    solution=str(instance.metadata.get("reference_solution") or ""),
                    gold=gold_answer,
                    response=text,
                    prediction=prediction,
                )
                correct = await self._judge_output(
                    output, judge_fn, prompt, parse_ugphysics_verdict, judge_result
                )
        else:
            raise ValueError(f"no scoring rule for SciReas-Pro source {source!r}")

        output.extracted_answer = prediction
        judge_result["extracted_answer"] = prediction
        judge_result["correct"] = correct
        scores["accuracy"] = 1.0 if correct else 0.0
        return self._record_judgement(output, judge_result, scores)


@register("scireas_pro:fast500")
class SciReasProFast500(SciReasPro):
    """The fixed, stratified 500-item sample of SciReas-Pro.

    Items are the ones listed in ``scireas_pro_fast500_ids.json``, in that file's
    order; everything else matches ``scireas_pro``.
    """

    def _select(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        by_id = {item_id(row["taskname"], int(row["doc_id"])): row for row in rows}
        ids = fast500_ids()
        missing = [sample_id for sample_id in ids if sample_id not in by_id]
        if missing:
            raise ValueError(
                f"{FAST500_FILE} names {len(missing)} item(s) absent from the dataset, "
                f"starting with {missing[0]}"
            )
        return [by_id[sample_id] for sample_id in ids]
