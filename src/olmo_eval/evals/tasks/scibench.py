"""SciBench: college-level scientific problems with free-form numeric answers.

SciBench (https://arxiv.org/abs/2307.10635) collects 692 problems from ten
undergraduate chemistry, physics and mathematics textbooks. Every gold answer is a
decimal number, so scoring compares the model's value against the gold value within
a 5 percent relative tolerance. That matches the reference implementation at
https://github.com/mandyyyyii/scibench, whose ``equiv`` function is identical across
all four of its evaluation drivers.

Three details of the reference grader are load bearing and reproduced here.

Units are never checked. The unit is injected into the prompt and the system prompt
tells the model to leave it out of the answer.

The ``unit`` field doubles as a scientific-notation carrier for 108 of the 692 rows,
where ``answer_number`` holds only the mantissa and the unit holds a ``10^k`` factor.
When text follows that factor, the reference strips the factor from the prompt and
multiplies the gold answer back out, so the model is asked for a full value in base
units. When the unit is only the factor, it stays in the prompt and the comparison
runs on the mantissa. Both paths are reproduced. Skipping them changes 16 percent of
the benchmark.

Comparison is tried twice, once on the whole answer and once on its first
whitespace-delimited token, with commas stripped. ``abs_tol`` keeps its default of
zero, so a gold value of exactly zero demands an exact match.

One deliberate deviation. Four rows in the ``diff`` textbook hold a gold value that
``float`` rejects: two write the sign as U+2212 MINUS SIGN and two use thousands
separators. The reference strips commas from the model's answer and never from the
gold, so it scores those four incorrect for every model. Both forms are normalized
here, which keeps all 692 rows scoreable and can raise a score by at most 0.58 points
against the published numbers. Everything else matches the reference exactly, verified
by replaying its grader over all 692 rows: identical prompts, identical gold values and
identical verdicts.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from olmo_eval.common.formatters import ChatFormatter
from olmo_eval.common.metrics import (
    AccuracyMetric,
    MacroSubsetAccuracyMetric,
    SubsetAccuracyMetric,
)
from olmo_eval.common.scorers.base import Scorer
from olmo_eval.common.types import Instance, LMOutput, LMRequest, SamplingParams, Split
from olmo_eval.data import DataSource
from olmo_eval.evals.tasks.common import Task, register

# The reference ``sys_cal_box2`` prompt, verbatim from eval/prompt/prompt_scai.py.
# The 5 percent tolerance is only calibrated against this wording, so it is copied
# rather than paraphrased.
_SYSTEM_PROMPT = (
    "Please provide a clear and step-by-step solution for a scientific problem in "
    "the categories of Chemistry, Physics, or Mathematics. The problem will specify "
    "the unit of measurement, which should not be included in the answer. Express "
    "the final answer as a decimal number with three digits after the decimal point. "
    'Conclude the answer by stating "The answer is therefore \\boxed{[ANSWER]}."'
)

# Reference ``remove_not``. Matches a ``10^k`` factor with optional dollar signs and
# braces. It deliberately consumes a leading ``$`` and can leave the unit's closing
# ``$`` behind; that is what the reference shows the model, so it is preserved.
_TEN_POWER = re.compile(r"[\$]?\ *10\^[{]?\ *-?[0-9]+\ *[}]?\ *[\$]?")
# Reference ``cal_not``. Same factor without the dollar signs, used to read k.
_BARE_TEN_POWER = re.compile(r"10\^[{]?\ *-?[0-9]+\ *[}]?")

_TEXTBOOKS = (
    "atkins",
    "calculus",
    "chemmc",
    "class",
    "diff",
    "fund",
    "matter",
    "quan",
    "stat",
    "thermo",
)

_REL_TOL = 0.05


def _unit_after_ten_power(unit: str) -> str | None:
    """Reference ``remove_not``: the unit text following a ``10^k`` factor, else None.

    Returns an empty string when the unit is nothing but the factor. The reference
    treats that as falsy and keeps the original unit, which this module mirrors.
    """
    if _TEN_POWER.search(unit) is None:
        return None
    return _TEN_POWER.split(unit)[-1]


def _ten_exponent(text: str) -> float | None:
    """Read the k out of a ``10^k`` factor, following reference ``cal_not``."""
    match = _BARE_TEN_POWER.search(text)
    if match is None:
        return None
    body = match.group(0)
    body = body[body.find("^") + 1 :]
    if "{" in body:
        body = body[body.find("{") + 1 :]
    if "}" in body:
        body = body[: body.find("}")]
    try:
        return float(body)
    except ValueError:
        return None


def _last_boxed(text: str) -> str:
    """Reference ``last_boxed_only_string``.

    It searches for the substring ``oxed`` so a leading backslash is optional. A bare
    ``\\box{}`` is therefore not matched, matching the reference.
    """
    idx = text.rfind("oxed")
    if idx < 0:
        idx = text.rfind("\\fbox")
        if idx < 0:
            return ""
    depth = 0
    for i in range(idx, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[idx : i + 1]
    return ""


def _unwrap_boxed(boxed: str) -> str:
    """Reference ``remove_boxed``, including its trailing ``=`` handling."""
    left = "oxed{"
    if not boxed.startswith(left) or not boxed.endswith("}"):
        return ""
    answer = boxed[len(left) : -1]
    if "=" in answer:
        answer = answer.split("=")[-1].lstrip(" ")
    return answer


def parse_answer(text: str) -> str:
    """Reference ``parse_math_answer``: the last boxed value in the response."""
    return _unwrap_boxed(_last_boxed(text))


def collapse_scientific_notation(text: str) -> str:
    """Reference ``cal_not(parse_not(...))`` applied to a model answer.

    ``2.5 \\times 10^6`` becomes ``2500000.0``. Anything that is not a mantissa and
    factor pair is returned unchanged, matching the reference's bare ``except``.
    """
    for separator in ("\\times", "*"):
        if separator in text:
            mantissa, _, factor = text.partition(separator)
            exponent = _ten_exponent(factor)
            if exponent is None:
                return text
            try:
                return str(float(mantissa.strip()) * 10.0**exponent)
            except ValueError:
                return text
    return text


def _matches(answer: str, gold: float, rel_tol: float) -> bool:
    """Reference ``equiv``: whole answer first, then its first token, commas stripped."""
    cleaned = answer.replace(",", "").strip()
    candidates = [cleaned]
    tokens = cleaned.split()
    if tokens:
        candidates.append(tokens[0])
    for candidate in candidates:
        try:
            if math.isclose(float(candidate), gold, rel_tol=rel_tol):
                return True
        except (ValueError, OverflowError):
            continue
    return False


@dataclass(frozen=True, slots=True)
class SciBenchToleranceScorer(Scorer):
    """Relative-tolerance numeric match, as the SciBench reference defines it.

    Nothing else in olmo-eval compares numbers with a tolerance. The math scorers are
    exact or symbolic, so they reject ``1.52`` against a gold of ``1.5``, which is a
    routine SciBench case.
    """

    name: str = "exact_match"
    rel_tol: float = _REL_TOL

    def score(self, instance: Instance, output: LMOutput) -> float:
        if instance.gold_answer is None:
            return 0.0
        try:
            gold = float(instance.gold_answer)
        except ValueError:
            return 0.0
        answer = parse_answer(output.text or "")
        # The reference collapses scientific notation in the model's answer only for
        # rows whose unit carried a factor that was stripped from the prompt.
        if instance.metadata.get("ten_exponent") is not None:
            answer = collapse_scientific_notation(answer)
        if output.metadata is not None:
            output.metadata["extracted_value"] = answer
        return 1.0 if _matches(answer, gold, self.rel_tol) else 0.0


_SCORER = SciBenchToleranceScorer()
_ACCURACY = AccuracyMetric(name="exact_match", scorer=_SCORER)
_SUBSETS = tuple(f"source__{book}" for book in _TEXTBOOKS)


@register("scibench")
class SciBench(Task):
    """SciBench open-ended numeric problems, scored within 5 percent relative error."""

    data_source = DataSource(
        path="xw27/scibench",
        split="train",
        revision="93931252bc1b71d495e67390235940643d926958",
    )
    split = Split.TRAIN
    formatter = ChatFormatter(system_prompt=_SYSTEM_PROMPT)
    metrics = (
        _ACCURACY,
        MacroSubsetAccuracyMetric(name="macro_accuracy", scorer=_SCORER, subsets=_SUBSETS),
        *(SubsetAccuracyMetric(name=subset, scorer=_SCORER) for subset in _SUBSETS),
    )
    primary_metric = _ACCURACY
    num_fewshot = 0
    strip_thinking = True
    # The reference decodes greedily. max_tokens is left uncapped so the derivation has
    # room on any context size; cap it with ``-o max_tokens=N`` for a faster run.
    sampling_params = SamplingParams(max_tokens=None, temperature=0.0)

    @property
    def instances(self) -> Iterator[Instance]:
        yield from self._load_instances_cached()

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        # Fields are used raw, exactly as the reference driver does. Trailing
        # whitespace is load bearing: a unit of ``"$10^6$ "`` leaves a space after the
        # factor, the reference reads that space as a real unit suffix, and that is
        # what decides whether the gold answer gets multiplied out. Stripping the unit
        # first silently drops the factor on two rows.
        problem = str(doc.get("problem_text") or "")
        raw_answer = str(doc.get("answer_number") or "")
        if not problem.strip() or not raw_answer.strip():
            return None
        # Four ``diff`` rows hold a gold value that ``float`` rejects: two write the
        # sign as U+2212 MINUS SIGN and two use thousands separators. The reference
        # strips commas from the model's answer but not from the gold, so it scores all
        # four incorrect for every model. Both are normalized here. See the module
        # docstring; this is the only place the task departs from the reference.
        try:
            gold = float(raw_answer.replace("−", "-").replace(",", "").strip())
        except ValueError:
            return None

        unit = str(doc.get("unit") or "")
        suffix = _unit_after_ten_power(unit)
        # An empty suffix means the unit is only the factor. The reference keeps the
        # original unit then, and skips the multiplication on both sides.
        shown_unit = suffix if suffix else unit
        exponent = _ten_exponent(unit) if shown_unit != unit else None
        if exponent is not None:
            gold *= 10.0**exponent

        return Instance(
            question=f"{problem} The unit of the answer is {shown_unit}.",
            gold_answer=repr(gold),
            metadata={
                "id": str(doc.get("problemid") or index).strip(),
                "source": str(doc.get("source") or "").strip(),
                "unit": unit,
                "ten_exponent": exponent,
                # The ``<book>_sol.json`` files contribute 112 problems that carry a
                # worked solution. They are disjoint from the other 580 and can only be
                # told apart by this field.
                "has_solution": bool(str(doc.get("solution") or "").strip()),
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        assert self.config.formatter is not None
        return self.config.formatter.format(instance, self.get_fewshot())

    def extract_answer(self, output: LMOutput) -> str:
        return parse_answer(output.text or "")
