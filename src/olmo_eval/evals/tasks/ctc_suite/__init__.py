"""
The CTC (corpus-tracking-capacity) long-context suite: 22 tasks, context ladders from 2k to 1M
tokens.

Each task gives the model a corpus of N documents in-prompt and asks a question whose difficulty
scales with how much of the corpus must be *simultaneously* tracked -- from O(N) retrieval (find
the answer-bearing passage) to O(N^2) relational tasks (find every contradicting pair) and O(NM)
structural ones (cluster everything). Every task has a ladder of context rungs; a rung label is the
measured median prompt length through the reference prompt path, not a document count.

**Provenance.** Prompt templates, answer parsers, per-task metrics, gold-index conventions and stop
rules are vendored verbatim from the ``ctc`` package (AI2 OLMo-core branch ``prasann/ctc``, commit
:data:`UPSTREAM_COMMIT`) under ``_vendor/``, where they are golden-fixture-tested against the
pre-migration implementation. That implementation produced the suite's published numbers, but six
rows have been re-graded since -- the README's "Grading changed after the published grid" table
says which, and an old number for one of those rows came from the old grader. Do not edit the
vendored files here; fix upstream and re-vendor. Only the subtrees this harness actually calls are
vendored -- see ``_vendor/MANIFEST.md`` for what was taken and what was deliberately left behind.
One spec (plain ``grouping``) is registered locally below from the vendored factory.

**Data.** Public HF dataset ``PrasannSinghal/ctc-suite-eval``, pinned at :data:`HF_REVISION`: one
config per task, one split per rung (``r2k`` ... ``r1m``). The pin is part of the task config and
therefore part of the task hash, so a dataset rebuild shows up as a new hash instead of silently
changing what an old number meant. Set ``CTC_SUITE_DATA_ROOT=/path/to/ladders`` to read local
``<task>/rung_<tokens>.jsonl`` files instead (same files the HF dataset is built from); the local
path *replaces* the config's data source, so a run over an unvalidated local ladder also gets its
own task hash rather than being stored under the published data's.

**Comparability notes.**

* Rungs at 256k and above carry ``eval_size=125`` (seeded subsample of the same question set;
  binomial SE ~ +/-0.041 at f1~0.7) -- quote the size next to any number from them. ``scifact``
  is 300 examples and ``obliq_twitter`` 123-126 at every rung; same rule. Every rung whose size is
  not the 500-example default is recorded in :attr:`RosterRow.eval_size`.
* Prompts here are the plain-text prompt path (``spec.build_prompt``). The original suite's
  vLLM/native runs additionally wrapped each document in Qwen-reserved marker tokens at
  tokenization time; scores are therefore comparable *within* this harness, and near but not
  bit-identical to the historical grid.
* ``query_position`` is pinned to ``"both"`` -- the setting every published rung was graded with.
* **Reasoning models.** A generation whose ``<think>`` block is never closed is counted as a parse
  failure rather than being handed to the parser: the ids inside an unfinished trace are the ones
  the model was *considering*, and scoring them credits a model for thinking out loud. Give such a
  model room to finish instead by overriding ``max_tokens``: the suite's budget is the config's
  ``sampling_params``, the runner applies overrides on top of it field by field, and the result is
  both what is sent and what is hashed. The suite's own budgets are sized for a direct answer.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

from olmo_eval.common.metrics.base import Metric
from olmo_eval.common.scorers.base import Scorer
from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, SamplingParams
from olmo_eval.data import DataSource
from olmo_eval.evals.tasks.common import Task, TaskConfig, register, register_variant

from ._vendor.ctc.eval.stopping import STOP_PRESETS
from ._vendor.ctc.eval.stopping import apply as _apply_stop
from ._vendor.ctc.eval.stopping import in_unclosed_think as _in_unclosed_think
from ._vendor.ctc.format import registry as _ctc_registry
from ._vendor.ctc.format.prompts import GROUPING_INSTRUCTION
from ._vendor.ctc.tasks import load_all as _load_all_specs
from ._vendor.ctc.tasks._grouping import make_grouping_spec

__all__ = [
    "HF_DATASET",
    "HF_REVISION",
    "UPSTREAM_COMMIT",
    "ROSTER",
    "RUNG_TOKENS",
    "CTCClass",
    "CTCScorer",
    "CTCParseScorer",
    "CTCSubMetricScorer",
    "CTCMeanMetric",
    "CTCSuiteTask",
]

#: Public HF dataset holding every rung file. One config per ROSTER row, one split per rung.
HF_DATASET = "PrasannSinghal/ctc-suite-eval"

#: The dataset commit every rung is read at. Pinned because the dataset has been rebuilt during
#: this suite's life (the xabsence exact-copy rebuild, the rerank repricing), and each rebuild
#: changes what a stored task hash means. ``DataSource.revision`` is serialized into
#: ``TaskConfig.to_dict()``, so bumping this deliberately produces a new hash.
HF_REVISION = "04ab86006124b3f34c566f44bbc82d4fec98fcda"

#: The ``ctc`` upstream commit ``_vendor/`` was taken from (AI2 OLMo-core branch ``prasann/ctc``).
#: Also named in the README and in ``_vendor/MANIFEST.md`` -- a test checks the three agree -- and
#: carried as a field on every CTC scorer, so a re-vendor changes the task hash.
UPSTREAM_COMMIT = "40a5c60d143b427b8c2cd276fa5009b809be6be3"

#: Env var pointing at a local ladder tree (``<subset>/rung_<tokens>.jsonl``) for offline runs.
DATA_ROOT_ENV = "CTC_SUITE_DATA_ROOT"

#: The setting every published rung was graded with. Tasks that hardcode their own position
#: (qdmatch, grouping, outlier) ignore this, and declare so via ``honors_query_position``.
QUERY_POSITION = "both"

# Import the vendored spec registrations (side-effect: populates the ctc registry).
_load_all_specs()

# Plain ``grouping`` (OpenAlex, unlabeled clusters) is in the suite roster but not in the vendored
# canonical set -- register it from the vendored factory, mirroring grouping_labeled minus labels.
# The header the factory builds already carries GROUPING_INSTRUCTION, so the query builder must NOT
# prepend it again: doing so printed the instruction twice in every plain-grouping prompt.
if "grouping" not in _ctc_registry.names():
    _ctc_registry.register(
        make_grouping_spec(
            name="grouping",
            description="Partition abstracts into their (unnamed) field clusters.",
            instruction=GROUPING_INSTRUCTION,
            rungs=("2k", "4k", "8k", "16k", "32k"),
            query_builder=lambda ex: ex["queries"][0] if ex.get("queries") else "",
            sources=("openalex",),
        )
    )


#: Rung label -> token budget in the local filenames. ``contradiction_iid`` aliases r2k to 2560:
#: its 2048-doc-count file sat below the training minimum, so the IID ladder starts one step up
#: (the remap is explicit in the source repo's ``build_suite_table.py`` as well).
RUNG_TOKENS: dict[str, int] = {
    "r2k": 2048,
    "r4k": 4096,
    "r8k": 8192,
    "r16k": 16384,
    "r32k": 32768,
    "r64k": 65536,
    "r128k": 131072,
    "r256k": 262144,
    "r512k": 524288,
    "r1m": 1048576,
}

_LADDER_2K_32K = ("r2k", "r4k", "r8k", "r16k", "r32k")
_LADDER_FULL = tuple(RUNG_TOKENS)  # 2k .. 1m
_LADDER_TO_512K = _LADDER_FULL[:-1]


class CTCClass(StrEnum):
    """How much of the corpus a row forces the model to track at once.

    This is the axis the suite is named for, so it is declared per row rather than inferred:
    :data:`LOW` rows are answerable by finding the right document(s) -- work that scales O(N) in
    corpus size and that a retriever could in principle do -- while :data:`HIGH` rows require
    relating documents to each other (O(N^2) pair-finding, O(NM) clustering, O(N^3) triples) and
    have no single answer-bearing span. The split is 12 low / 10 high and is what ``ctc:low`` /
    ``ctc:high`` select; it is orthogonal to context length, which ``ctc:figure`` / ``ctc:xlong``
    select.
    """

    LOW = "low"
    HIGH = "high"


@dataclass(frozen=True)
class RosterRow:
    """One row of the 22-task suite.

    :param subset: HF config name == local ladder directory name.
    :param spec: The vendored :class:`TaskSpec` that formats and grades this row.
    :param ctc_class: :class:`CTCClass` -- low (O(N), retrieval-shaped) or high (O(N^2)+,
        relational/structural). Declared with no default: a new row must classify itself, because
        a silently-defaulted row would quietly change what ``ctc:low``/``ctc:high`` mean.
    :param complexity: The class in the notation the README table uses (``"O(N)"``, ``"O(N^2)"``,
        ``"O(NM)"``, ``"O(N^3)"``), recorded so the coarse two-way split stays auditable.
    :param rungs: Rung labels with data, ascending.
    :param eval_size: ``{rung_label: rows}`` for every rung whose size is not the 500-example
        default -- above it as well as below. A number quoted from a small rung must carry the
        size inline, and a rung that is *larger* than the default has to be declared too or the
        README's "everything else is 500" reads as a measured fact when it is not.
    :param rung_alias: Rung label -> token budget override for the local filename.
    :param max_new_tokens: Decode budget override for this row, when the spec's own budget cannot
        hold this row's longest correct answer. Sized from the measured gold, not guessed.
    :param note: Anything a reader of the numbers has to know.
    """

    subset: str
    spec: str
    ctc_class: CTCClass
    complexity: str
    rungs: tuple[str, ...] = _LADDER_2K_32K
    eval_size: dict[str, int] = field(default_factory=dict)
    rung_alias: dict[str, int] = field(default_factory=dict)
    max_new_tokens: int | None = None
    note: str = ""


_SUB500_XLONG = {"r256k": 125, "r512k": 125, "r1m": 125}

#: The frozen 22-row roster. Row order is figure order. The record it was frozen from
#: (``records/ctc-final-suite.md``, 2026-08-12) lives in the OLMo-core working repo, not here --
#: this dict is the copy that runs, and the roster tests are what pin it.
ROSTER: dict[str, RosterRow] = {
    "ctc_fiqa": RosterRow(
        ctc_class=CTCClass.LOW,
        complexity="O(N)",
        subset="fiqa",
        spec="retrieval",
        rungs=_LADDER_FULL,
        eval_size=dict(_SUB500_XLONG),
    ),
    "ctc_nq": RosterRow(
        ctc_class=CTCClass.LOW,
        complexity="O(N)",
        subset="nq",
        spec="retrieval",
        rungs=_LADDER_FULL,
        eval_size=dict(_SUB500_XLONG),
    ),
    "ctc_hpqa": RosterRow(
        ctc_class=CTCClass.LOW,
        complexity="O(N)",
        subset="hotpotqa",
        spec="retrieval",
        rungs=_LADDER_FULL,
        eval_size=dict(_SUB500_XLONG),
    ),
    "ctc_qdmatch_fiqa": RosterRow(
        ctc_class=CTCClass.HIGH,
        complexity="O(N^2)",
        subset="qdmatch_fiqa",
        spec="qdmatch",
        rungs=_LADDER_TO_512K,
        eval_size=dict(_SUB500_XLONG),
        note="caps at 512k: the 6,148 recoverable BEIR-FiQA units are the whole labeled universe, "
        "and a 1M example needs ~8k distinct units",
    ),
    "ctc_qdmatch_nq": RosterRow(
        ctc_class=CTCClass.HIGH,
        complexity="O(N^2)",
        subset="qdmatch_nq",
        spec="qdmatch",
        rungs=_LADDER_FULL,
        eval_size=dict(_SUB500_XLONG),
    ),
    "ctc_qdmatch_hpqa": RosterRow(
        ctc_class=CTCClass.HIGH,
        complexity="O(N^2)",
        subset="qdmatch_hpqa",
        spec="qdmatch",
        rungs=_LADDER_FULL[:-2],
        eval_size=dict(_SUB500_XLONG),
        note="caps at 256k: 4,000 recoverable HotpotQA units, and a 512k example needs ~7k",
    ),
    "ctc_outlier_amzn": RosterRow(
        ctc_class=CTCClass.LOW,
        complexity="O(N)",
        subset="outlier_amzn",
        spec="outlier",
        rungs=_LADDER_FULL,
        eval_size=dict(_SUB500_XLONG),
        note="the Amazon product-review build -- the corpus the shared outlier instruction's "
        "'product reviews' wording describes",
    ),
    "ctc_outlier": RosterRow(
        ctc_class=CTCClass.HIGH,
        complexity="O(NM)",
        subset="outlier",
        spec="outlier",
        rungs=_LADDER_FULL,
        eval_size=dict(_SUB500_XLONG),
        note="64k+ rungs are the true scale-K construction (K ~ n/9); the retired fixed-K xlong "
        "files are not used. Wikipedia topic chunks, graded under the reference's shared outlier "
        "instruction, whose wording says 'product reviews' -- kept verbatim because every "
        "published number for this row was produced under it",
    ),
    "ctc_outlier_fixedm": RosterRow(
        ctc_class=CTCClass.LOW,
        complexity="O(N)",
        subset="outlier_fixedM",
        spec="outlier",
        rungs=_LADDER_TO_512K,
        eval_size=dict(_SUB500_XLONG),
        note="K pinned at 3 -- the control for the scale-K row. Caps at 512k: three majority "
        "topics need ~2,400 same-topic chunks each at 1M and the wiki pool cannot supply that. "
        "Wikipedia topic chunks under the same verbatim 'product reviews' instruction as "
        "ctc_outlier",
    ),
    "ctc_oolong": RosterRow(
        ctc_class=CTCClass.LOW,
        complexity="O(N)",
        subset="oolong",
        spec="oolong",
        rungs=_LADDER_FULL,
        eval_size={**_SUB500_XLONG, "r64k": 668, "r128k": 669},
        note="r64k and r128k hold more than the 500-example default (668 and 669)",
    ),
    "ctc_grouping": RosterRow(
        ctc_class=CTCClass.HIGH,
        complexity="O(NM)",
        subset="grouping",
        spec="grouping",
    ),
    "ctc_absence": RosterRow(
        ctc_class=CTCClass.LOW,
        complexity="O(N)",
        subset="absence_gutenberg",
        spec="absence",
        rungs=("r2k", "r4k", "r8k", "r16k"),
        eval_size={"r16k": 148},
        note="corpus is a contiguous Gutenberg passage; rung ceiling is bounded by book length, "
        "and so is the r16k example count (148). "
        "LOW despite the absence framing: the modified version is the original in the same order "
        "with whole sentences deleted, so the deletions fall out of a single aligned pass -- O(N). "
        "Contrast ctc_xabsence, whose two corpora are unordered, so every item has to be checked "
        "against every item in the other corpus",
    ),
    "ctc_xabsence": RosterRow(
        ctc_class=CTCClass.HIGH,
        complexity="O(N^2)",
        subset="xabsence",
        spec="xabsence",
        note="the one-sided EXACT-COPY Gutenberg build (2026-08-14): B-side twins are "
        "byte-identical and orphans sit A-side only. Supersedes the PubMed paraphrase build, "
        "kept in the dataset as the xabsence_paraphrase config",
    ),
    "ctc_rerank": RosterRow(
        ctc_class=CTCClass.LOW,
        complexity="O(N)",
        subset="rerank",
        spec="rerank",
        rungs=_LADDER_TO_512K,
        eval_size=dict(_SUB500_XLONG),
        note="14 of 500 rows at r2k and at r32k have no document with cross-encoder score > 0; "
        "ce_pos_recall is undefined on those and they are excluded from the mean (the count is "
        "reported as ce_ref_available)",
    ),
    "ctc_msmarco": RosterRow(
        ctc_class=CTCClass.LOW,
        complexity="O(N)",
        subset="msmarco",
        spec="retrieval",
        rungs=_LADDER_TO_512K,
        eval_size=dict(_SUB500_XLONG),
    ),
    "ctc_reorder": RosterRow(
        ctc_class=CTCClass.HIGH,
        complexity="O(N^2)",
        subset="reorder",
        spec="reorder",
        rungs=("r2k", "r4k", "r8k", "r16k"),
        note="chunks are contiguous in one book; rung ceiling is bounded by book length. Its "
        "primary metric, kendall_tau, ranges over [-1, 1] and not [0, 1] -- 0.0 is chance, not "
        "the floor, and it must not be averaged against the f1-style rows",
    ),
    "ctc_obliq": RosterRow(
        ctc_class=CTCClass.LOW,
        complexity="O(N)",
        subset="obliq_twitter",
        spec="retrieval",
        rungs=_LADDER_FULL,
        eval_size={
            "r2k": 123,
            **{r: 126 for r in _LADDER_FULL[1:7]},
            "r256k": 125,
            "r512k": 125,
            "r1m": 125,
        },
        max_new_tokens=512,
        note="123-126 examples per rung -- flag the size inline. Unlike the rest of the retrieval "
        "family (at most 3 gold ids) its gold sets reach 64 ids, so it overrides the retrieval "
        "spec's 64-token budget: the longest perfect answer measures 311 tokens at r32k and 404 "
        "at r1m through the Qwen3 tokenizer, and under a 64-token budget 27/126 rows at r32k were "
        "truncated into partial credit (mean f1 ceiling 0.953)",
    ),
    "ctc_niah": RosterRow(
        ctc_class=CTCClass.LOW,
        complexity="O(N)",
        subset="niah",
        spec="retrieval",
        rungs=_LADDER_FULL,
        eval_size=dict(_SUB500_XLONG),
    ),
    "ctc_contradiction": RosterRow(
        ctc_class=CTCClass.HIGH,
        complexity="O(N^2)",
        subset="contradiction_iid",
        spec="contradiction",
        rungs=_LADDER_FULL,
        eval_size=dict(_SUB500_XLONG),
        rung_alias={"r2k": 2560},
        note="the IID realistic-mode ladder; never mix with the retired both-mode ladder",
    ),
    "ctc_strmatch": RosterRow(
        ctc_class=CTCClass.HIGH,
        complexity="O(N^2)",
        subset="strmatch",
        spec="strmatch",
    ),
    "ctc_textgroups": RosterRow(
        ctc_class=CTCClass.HIGH,
        complexity="O(N^3)",
        subset="textgroups",
        spec="textgroups",
    ),
    "ctc_scifact": RosterRow(
        ctc_class=CTCClass.LOW,
        complexity="O(N)",
        subset="scifact",
        spec="retrieval",
        eval_size={r: 300 for r in _LADDER_2K_32K},
        note="300 examples at every rung -- flag the size inline",
    ),
}

#: Which rung a bare task name (no variant) evaluates: the top of the 2k-32k figure ladder.
DEFAULT_RUNG = "r32k"

#: Specs whose gold is not a top-level example field: ``score`` reads it off the whole example.
#: oolong's answers live in ``_meta.gold_list`` while its ``gold_doc_indices`` is present and
#: *empty*, so "fall back when the field is falsy" would be indistinguishable from a genuinely
#: empty gold elsewhere -- absence/xabsence raise on an example passed where a list belongs, and a
#: pair spec scores a correct ``[]`` answer 0.0. The fallback is therefore declared, not inferred.
_GOLD_IS_THE_EXAMPLE = frozenset({"oolong"})

#: Sub-metrics a row reports as their own metric, beyond the spec's primary. Only rerank needs one:
#: ``ce_ref_available`` is what says how many rows its mean was actually computed over.
_EXTRA_METRIC_KEYS: dict[str, tuple[str, ...]] = {"rerank": ("ce_ref_available",)}

#: Rows whose primary-metric mean skips examples where this sub-metric is 0, because the metric is
#: undefined there rather than zero. rerank's ce_pos_recall needs at least one CE-positive
#: document; without one, a perfect qrel-first ranking still scores 0 and the ceiling sits at 0.972.
_SKIP_WHEN_ZERO: dict[str, str] = {"rerank": "ce_ref_available"}


def _resolve_spec(spec_name: str):
    return _ctc_registry.get(spec_name)


def _data_source(row: RosterRow, rung: str) -> DataSource:
    """The canonical HF source. The split IS the rung label, which is also how
    :meth:`CTCSuiteTask._load_instances` knows which file it is reading.

    ``data_files`` pins the ONE parquet this rung needs. Without it the loader resolves the whole
    config and builds every split before handing back the requested one, so ``-t ctc_nq:r2k``
    downloads the entire 2k-to-1M nq ladder (~900MB) to read 500 short rows -- slow on every rung,
    and a download failure anywhere in the ladder fails a rung that did not need those files.

    ``revision`` pins the dataset commit; it is serialized by :meth:`DataSource.to_dict`, so a
    rebuild of the dataset cannot land under an existing task hash.
    """
    return DataSource(
        path=HF_DATASET,
        subset=row.subset,
        split=rung,
        data_files={rung: f"data/{row.subset}/{rung}.parquet"},
        revision=HF_REVISION,
    )


def _local_data_source(row: RosterRow, rung: str, root: str) -> DataSource:
    """The local-ladder source that ``CTC_SUITE_DATA_ROOT`` substitutes for the published data.

    A bare JSONL is a single unnamed split, so the split is ``train``. The path is the whole point:
    it reaches ``TaskConfig.to_dict()`` and therefore the task hash, so a run over an unvalidated
    local tree is stored separately from a run over the published dataset. Reading the env var and
    quietly swapping the *rows* while leaving the config untouched is what let the two share a
    hash.
    """
    tokens = row.rung_alias.get(rung, RUNG_TOKENS[rung])
    return DataSource(path=os.path.join(root, row.subset, f"rung_{tokens}.jsonl"), split="train")


def _decode_budget(row: RosterRow, spec) -> int:
    """The row's ``max_tokens``: its own override, else the larger of the stop preset's and the
    spec's budgets. Too small truncates a correct answer into a parse failure, which reads as a
    capability limit rather than as a config mistake."""
    if row.max_new_tokens is not None:
        return row.max_new_tokens
    return max(STOP_PRESETS[spec.stop].max_new_tokens, spec.max_new_tokens)


def _score_output(spec, instance: Instance, output: LMOutput) -> tuple[Any, dict[str, float]]:
    """Run the spec's stop rules, parser and metric on one generation, once.

    Every CTC scorer needs the same three calls, and :meth:`Task._apply_scorers` runs each scorer
    over every output, so the result is memoized on ``output.metadata``: without it a row with a
    primary metric, a parse-rate metric and a sub-metric parses each 1M-token generation three
    times.

    The three calls mirror the reference runner exactly: stop-rule cleanup, then
    ``spec.parse(text, n_docs)``, then ``spec.score(parsed, gold)`` with the spec's declared gold
    field. One deviation from the reference is deliberate: a generation still inside an unclosed
    ``<think>`` block is treated as unparseable (see the module docstring).

    :returns: ``(parsed, scored)`` -- the parser's output (``None`` on failure) and the spec's
        full metric dict.
    """
    cached = output.metadata.get("ctc_scored") if output.metadata else None
    if cached is not None:
        return cached["parsed"], cached["scored"]

    example = instance.metadata["example"]
    cleaned = _apply_stop(output.text or "", STOP_PRESETS[spec.stop])
    # An unfinished trace holds only the ids the model was still weighing. Scoring them credits
    # thinking out loud; recording a parse failure says what actually happened.
    truncated_trace = _in_unclosed_think(cleaned)
    parsed = None if truncated_trace else spec.parse(cleaned, len(example["documents"]))

    # Gold resolution mirrors the reference runner: the whole example when the spec asks for it
    # (rerank's scorer needs ce_scores, oolong's gold lives in _meta.gold_list), else the declared
    # field. A present-but-empty gold list is NOT a reason to fall back -- see _GOLD_IS_THE_EXAMPLE.
    if spec.extra.get("score_takes_example") or spec.name in _GOLD_IS_THE_EXAMPLE:
        gold = example
    else:
        gold = example.get(spec.extra.get("gold_field", "gold_doc_indices"))
        if gold is None:
            gold = example
    scored = {k: float(v) for k, v in spec.score(parsed, gold).items()}

    if output.metadata is None:
        output.metadata = {}
    output.metadata["ctc_scored"] = {"parsed": parsed, "scored": scored}
    output.metadata["ctc_parse_ok"] = parsed is not None
    output.metadata["ctc_all_metrics"] = scored
    # `score:`-prefixed metadata keys are persisted per output as ``sample_metrics`` by the
    # predictions writer, which is the only route a sub-metric has into the shipped files --
    # ``output.metadata`` itself is dropped. These are the diagnostics (k_exact, coverage,
    # ce_ref_available, recall@k) someone reading a low score needs. The spec's own ``parsed`` flag
    # is skipped: it is already persisted as ``ctc_parse_ok``, and one flag under two names is a
    # question nobody should have to ask of a predictions file.
    for key, value in scored.items():
        if key != "parsed":
            output.metadata[f"score:ctc_{key}"] = value
    return parsed, scored


@dataclass(frozen=True)
class _CTCScorer(Scorer):
    """What the three CTC scorers share.

    ``upstream`` and ``query_position`` take no part in scoring. They are here because a scorer's
    fields are serialized into the task config and so into the task hash, and these two *are* the
    grader's identity: which vendored commit parsed and scored the generation, and which prompt
    layout it was answering. Without them a re-vendor that changes a parser, a prompt or a stop
    rule keeps the old hash -- which is how the rows in the README's "Grading changed" table ended
    up stored beside numbers from the old grader.
    """

    spec_name: str = ""
    upstream: str = UPSTREAM_COMMIT
    query_position: str = QUERY_POSITION


@dataclass(frozen=True)
class CTCScorer(_CTCScorer):
    """Score a generation with the task's own parser and metric.

    A ``None`` parse scores 0 on every metric, which is the reference behaviour -- but parse *rate*
    should be watched separately: a parse-rate collapse is a decoding/stopping regression wearing
    an accuracy drop's clothes.
    """

    name: str = "ctc"

    def score(self, instance: Instance, output: LMOutput) -> float:
        spec = _resolve_spec(self.spec_name)
        _, scored = _score_output(spec, instance, output)
        return scored.get(spec.primary_metric, 0.0)


@dataclass(frozen=True)
class CTCParseScorer(_CTCScorer):
    """1.0 when the task's own parser got a usable answer out of the generation, else 0.0.

    Exists because a parse-rate collapse is a decoding/stopping regression wearing an accuracy
    drop's clothes, and the shipped output files previously had no way to tell them apart:
    ``ctc_parse_ok`` lived on ``output.metadata``, which the predictions writer does not persist.
    Declaring it as a scorer puts the number in ``metrics.json`` and in every prediction's
    ``instance_metrics`` instead, which is where anyone reading a low score looks.
    """

    name: str = "ctc_parse_ok"

    def score(self, instance: Instance, output: LMOutput) -> float:
        parsed, _ = _score_output(_resolve_spec(self.spec_name), instance, output)
        return float(parsed is not None)


@dataclass(frozen=True)
class CTCSubMetricScorer(_CTCScorer):
    """One named key out of the spec's metric dict, promoted to a metric of its own.

    Used where a sub-metric is not a diagnostic but a precondition for reading the primary number
    -- rerank's ``ce_ref_available`` says whether ``ce_pos_recall`` was defined for that example at
    all, and its mean is the share of rows the primary mean was computed over.
    """

    name: str = "ctc_sub"
    key: str = ""

    def score(self, instance: Instance, output: LMOutput) -> float:
        _, scored = _score_output(_resolve_spec(self.spec_name), instance, output)
        return scored.get(self.key, 0.0)


@dataclass(frozen=True, slots=True)
class CTCMeanMetric(Metric):
    """Mean of one scorer across responses (named for the metric it reports, e.g. mrr@10).

    :param skip_when_zero: Name of a sub-metric that gates the mean. Responses whose value for it
        is 0 are left out entirely rather than averaged in as zeros, for the case where the metric
        is *undefined* on that example rather than failed -- rerank's ``ce_pos_recall`` needs at
        least one CE-positive document, and 14 of 500 rows have none, so counting them as zeros put
        the ceiling a perfect model could reach at 0.972. A row left out of the mean has no
        per-instance value either (:meth:`compute_instance`), so the predictions file cannot
        disagree with ``metrics.json``. When no row at all counts -- a ``limit`` run that drew only
        CE-less rows -- the mean is 0.0 and so is the ``ce_ref_available`` metric reported beside
        it; read the pair as "nothing here was gradable", not as a model that scored zero.
    """

    name: str = "ctc"
    scorer: type[Scorer] | Scorer = CTCScorer()
    skip_when_zero: str | None = None

    def _counts(self, response) -> bool:
        if self.skip_when_zero is None:
            return True
        for output in response.outputs:
            metrics = (output.metadata or {}).get("ctc_all_metrics")
            if metrics and metrics.get(self.skip_when_zero, 0.0) > 0.0:
                return True
        return False

    def compute(self, responses) -> float:
        counted = [r for r in responses if self._counts(r)]
        if not counted:
            return 0.0
        scorer_name = self.scorer().name
        total = sum(r.scores.get(scorer_name, 0.0) for r in counted)
        return total / len(counted)

    def compute_instance(self, response) -> float | None:
        # A row the mean excluded gets no value: the scorer channel still holds its 0.0, and
        # persisting that would make a pairwise analysis over the predictions average rows the
        # aggregate never saw.
        if not self._counts(response):
            return None
        value = response.scores.get(self.scorer().name)
        return float(value) if isinstance(value, (int, float)) else None

    def supports_scorer_fallback(self) -> bool:
        # With a gate, the scorer channel is NOT a per-instance view of this metric -- it has
        # values for the excluded rows -- so a reader must not fall back to it.
        return self.skip_when_zero is None

    def to_dict(self) -> dict[str, Any]:
        # The gate changes what the aggregate means, so it has to change the task hash. (Explicit
        # base call: zero-argument super() does not work in a slots dataclass.)
        return {**Metric.to_dict(self), "skip_when_zero": self.skip_when_zero}


class CTCSuiteTask(Task):
    """Base class for every suite row.

    A subclass declares one thing -- ``row_name`` -- and :meth:`__init_subclass__` derives the data
    source, the metrics and the docstring from ``ROSTER[row_name]``. The 22 rows are written out as
    real classes below rather than built with ``type()`` so that each one is importable, greppable
    and picklable under its own name.
    """

    row_name: str = ""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not cls.row_name:
            return
        row = ROSTER[cls.row_name]
        spec = _resolve_spec(row.spec)
        metrics: list[Metric] = [
            CTCMeanMetric(
                name=spec.primary_metric,
                scorer=CTCScorer(spec_name=row.spec),
                skip_when_zero=_SKIP_WHEN_ZERO.get(row.spec),
            ),
            CTCMeanMetric(name="parse_rate", scorer=CTCParseScorer(spec_name=row.spec)),
        ]
        metrics += [
            CTCMeanMetric(
                name=key, scorer=CTCSubMetricScorer(name=f"ctc_{key}", spec_name=row.spec, key=key)
            )
            for key in _EXTRA_METRIC_KEYS.get(row.spec, ())
        ]
        # Task._get_scorers() dedups by scorer name and silently keeps the first, so a sub-metric
        # whose key collided with a fixed scorer name would quietly report that scorer's value.
        names = [m.scorer().name for m in metrics]
        if len(set(names)) != len(names):
            raise ValueError(f"{cls.row_name}: scorer names must be unique, got {names}")
        cls.__doc__ = f"CTC suite row {row.subset!r} ({row.spec} spec). {row.note}".strip()
        cls.data_source = _data_source(
            row, DEFAULT_RUNG if DEFAULT_RUNG in row.rungs else row.rungs[-1]
        )
        cls.metrics = tuple(metrics)
        cls.primary_metric = metrics[0]
        # The decode budget is config, not a per-instance decision, so it is declared here where
        # `register` serializes it into the task hash and the runner lays overrides over it field
        # by field -- what is sent is what is stored. No decode-time text stops on purpose: the
        # reference stop rules suppress text stops inside an unclosed <think> block and before
        # first content, which a stop string cannot honour, so the same rules run post-hoc in
        # CTCScorer. The only cost is decode tokens on models that never emit EOS, bounded by
        # max_tokens.
        cls.sampling_params = SamplingParams(max_tokens=_decode_budget(row, spec), temperature=0)

    def __init__(self, config: TaskConfig) -> None:
        # CTC_SUITE_DATA_ROOT is resolved HERE, into the config, rather than at load time: the
        # local tree is a different dataset and must produce a different task hash. Read at
        # construction (not at import) so the env var still works the way an env var should. It
        # stands in for the published dataset only: a source that is already local, or that a
        # caller pointed somewhere else on purpose, is left alone -- which also keeps construction
        # idempotent, since the split of an already-local source is not a rung label.
        root = os.environ.get(DATA_ROOT_ENV)
        source = config.data_source
        if root and isinstance(source, DataSource) and source.path == HF_DATASET:
            config = replace(config, data_source=_local_data_source(self.row, source.split, root))
        super().__init__(config)

    @property
    def row(self) -> RosterRow:
        return ROSTER[self.row_name]

    @property
    def spec(self):
        return _resolve_spec(self.row.spec)

    @property
    def instances(self) -> Iterator[Instance]:
        yield from self._load_instances_cached()

    def _load_instances(self, split: str | None = None) -> Iterator[Instance]:
        """Same as the base loader, except the variant's :class:`DataSource` is used AS-IS.

        The base loader routes through ``config.get_data_source()``, which substitutes the
        config-level default split (``test``) for the source's own -- and this suite has no
        ``test`` split anywhere, so every rung would 404. That is a framework deficiency rather
        than a property of this suite; see the follow-up issue on ``get_data_source(split=None)``
        keeping a ``DataSource``'s own split.
        """
        from olmo_eval.data import DataLoader

        for index, doc in enumerate(DataLoader().load(self.config.data_source)):
            instance = self.process_doc(doc, index)
            if instance is not None:
                yield instance

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        # The raw unified example travels whole: the prompt builder, parser and scorer all read
        # different parts of it, and slicing it here is how field conventions historically drifted.
        return Instance(
            question=doc["queries"][0] if doc.get("queries") else "",
            gold_answer=None,
            metadata={"id": index, "example": doc},
        )

    def format_request(self, instance: Instance) -> LMRequest:
        prompt = self.spec.build_prompt(instance.metadata["example"], query_position=QUERY_POSITION)
        return LMRequest(request_type=RequestType.COMPLETION, prompt=prompt)


@register("ctc_fiqa")
class CTCFiqa(CTCSuiteTask):
    row_name = "ctc_fiqa"


@register("ctc_nq")
class CTCNq(CTCSuiteTask):
    row_name = "ctc_nq"


@register("ctc_hpqa")
class CTCHpqa(CTCSuiteTask):
    row_name = "ctc_hpqa"


@register("ctc_qdmatch_fiqa")
class CTCQdmatchFiqa(CTCSuiteTask):
    row_name = "ctc_qdmatch_fiqa"


@register("ctc_qdmatch_nq")
class CTCQdmatchNq(CTCSuiteTask):
    row_name = "ctc_qdmatch_nq"


@register("ctc_qdmatch_hpqa")
class CTCQdmatchHpqa(CTCSuiteTask):
    row_name = "ctc_qdmatch_hpqa"


@register("ctc_outlier_amzn")
class CTCOutlierAmzn(CTCSuiteTask):
    row_name = "ctc_outlier_amzn"


@register("ctc_outlier")
class CTCOutlier(CTCSuiteTask):
    row_name = "ctc_outlier"


@register("ctc_outlier_fixedm")
class CTCOutlierFixedM(CTCSuiteTask):
    row_name = "ctc_outlier_fixedm"


@register("ctc_oolong")
class CTCOolong(CTCSuiteTask):
    row_name = "ctc_oolong"


@register("ctc_grouping")
class CTCGrouping(CTCSuiteTask):
    row_name = "ctc_grouping"


@register("ctc_absence")
class CTCAbsence(CTCSuiteTask):
    row_name = "ctc_absence"


@register("ctc_xabsence")
class CTCXabsence(CTCSuiteTask):
    row_name = "ctc_xabsence"


@register("ctc_rerank")
class CTCRerank(CTCSuiteTask):
    row_name = "ctc_rerank"


@register("ctc_msmarco")
class CTCMsmarco(CTCSuiteTask):
    row_name = "ctc_msmarco"


@register("ctc_reorder")
class CTCReorder(CTCSuiteTask):
    row_name = "ctc_reorder"


@register("ctc_obliq")
class CTCObliq(CTCSuiteTask):
    row_name = "ctc_obliq"


@register("ctc_niah")
class CTCNiah(CTCSuiteTask):
    row_name = "ctc_niah"


@register("ctc_contradiction")
class CTCContradiction(CTCSuiteTask):
    row_name = "ctc_contradiction"

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        # Contradiction pairs are unordered, and the vendored scorer requires each gold pair
        # sorted low-high because parsed pairs are. The published r64k-r1m rungs store most of
        # their pairs high-low (r64k: 744 of 1,500), and the scorer raises on every such example
        # rather than score it. Sorting here is a no-op on r2k-r32k, which have none.
        doc = {**doc, "gold_doc_indices": [sorted(pair) for pair in doc["gold_doc_indices"]]}
        return super().process_doc(doc, index)


@register("ctc_strmatch")
class CTCStrmatch(CTCSuiteTask):
    row_name = "ctc_strmatch"


@register("ctc_textgroups")
class CTCTextgroups(CTCSuiteTask):
    row_name = "ctc_textgroups"


@register("ctc_scifact")
class CTCScifact(CTCSuiteTask):
    row_name = "ctc_scifact"


# One variant per rung with data. The base name evaluates DEFAULT_RUNG; a variant only changes the
# data source, since every other setting is a property of the row and not of the rung.
for _name, _row in ROSTER.items():
    for _rung in _row.rungs:
        register_variant(_name, _rung, data_source=_data_source(_row, _rung))
