"""HELMET tasks: all seven categories at HELMET's standard context lengths.

HELMET (https://github.com/princeton-nlp/HELMET) evaluates seven task
categories at 4k-128k tokens of context: recall, RAG, re-ranking, LongQA,
summarization, ICL, and citation (ALCE). All seven are registered here.

Pre-generated and pre-retrieved data lives on the ai2-internal
`allenai/helmet-plus` Hub dataset; the remaining tasks pull their sources
from the Hub at load time. `narrativeqa` and both summarization tasks are
graded by an LLM judge (see `helmet_judge.py`) and therefore need a judge
configured -- the `helmet_nojudge__*` suites exclude them.
"""

import os
import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, cast

from olmo_eval.common.metrics import AccuracyMetric, NDCGMetric, RecallMetric, RougeLF1Metric
from olmo_eval.common.scorers import Scorer
from olmo_eval.common.scorers.alce import AlceQampariRecTop5Scorer, AlceStrEmScorer
from olmo_eval.common.scorers.base import _squad_normalize_answer
from olmo_eval.common.scorers.helmet_judge import HelmetLongQAJudgeScorer, HelmetSummJudgeScorer
from olmo_eval.common.scorers.substring import SubstringExactMatchScorer
from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, SamplingParams
from olmo_eval.data import (
    helmet_alce_loader,
    helmet_icl_loader,
    helmet_infbench_loader,
    helmet_kilt_loader,
    helmet_loader,
    helmet_msmarco_loader,
    helmet_multilexsum_loader,
    helmet_narrativeqa_loader,
)
from olmo_eval.data.helmet_alce_loader import load_alce_dataset
from olmo_eval.data.helmet_icl_loader import ICL_DATASETS, load_icl_dataset
from olmo_eval.data.helmet_infbench_loader import (
    INFBENCH_SUBSETS,
    INFINITEBENCH_REPO,
    INFINITEBENCH_REVISION,
    REFERENCE_TOKENIZER,
    REFERENCE_TOKENIZER_REVISION,
    load_infbench_dataset,
)
from olmo_eval.data.helmet_kilt_loader import load_kilt_dataset
from olmo_eval.data.helmet_loader import (
    HELMET_PLUS_REPO_ID,
    HELMET_PLUS_REVISION,
    load_json_kv_dataset,
)
from olmo_eval.data.helmet_msmarco_loader import load_msmarco_dataset, parse_rankings
from olmo_eval.data.helmet_multilexsum_loader import load_multi_lexsum_dataset
from olmo_eval.data.helmet_narrativeqa_loader import (
    NARRATIVEQA_REPO,
    NARRATIVEQA_REVISION,
    load_narrativeqa_dataset,
)
from olmo_eval.data.helmet_tasks import HELMET_TASKS
from olmo_eval.data.ruler_loader import (
    RULER_DATA_REPO,
    RULER_DATA_REVISION,
    download_ruler_data,
    get_ruler_templates,
    load_ruler_dataset,
)
from olmo_eval.data.ruler_tasks import RULER_TASKS
from olmo_eval.evals.tasks.common.base import Task, TaskConfig
from olmo_eval.evals.tasks.common.registry import register


class HelmetTask(Task):
    """Shared plumbing for HELMET-plus tasks.

    Subclasses implement `_load_dataset` to fetch their data; everything from
    prompt assembly onward is common, since all HELMET tasks share the same
    `user_template` / `system_template` prompt shape.
    """

    def __init__(self, config: TaskConfig) -> None:
        super().__init__(config)
        task_name = config.name.removeprefix("helmet_")
        self.task_name = task_name
        self.helmet_config = HELMET_TASKS[task_name]

        task_type, context_size_str = task_name.rsplit("__", 1)
        self.task_type = task_type
        self.context_size = int(context_size_str)

        self._dataset = None
        self._templates = None

    def _load_dataset(self) -> dict[str, Any]:
        """Return the loader payload: `data` plus the HELMET prompt templates."""
        raise NotImplementedError

    def _load_data(self) -> None:
        if self._dataset is not None:
            return

        loaded = self._load_dataset()

        self._dataset = loaded["data"]
        self._templates = {
            "prompt": loaded["prompt_template"],
            "user": loaded["user_template"],
            "system": loaded["system_template"],
        }

    @property
    def instances(self) -> Iterator[Instance]:
        self._load_data()

        if self._instances_cache is not None:
            yield from self._instances_cache
            return

        self._instances_cache = []
        for idx, doc in enumerate(self._dataset):  # type: ignore
            instance = self.process_doc(cast(dict[str, Any], doc), index=idx)
            if instance is not None:
                self._instances_cache.append(instance)
                yield instance

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        if self._templates is None:
            raise RuntimeError("Templates not loaded. Call _load_data() first.")

        question = self._templates["user"].format(**doc)

        # With a chat template the answer prefix is left off rather than fed in
        # as a partial assistant turn, matching how HELMET runs its chat-format
        # tasks (and how ruler.py handles the same distinction).
        # RULER's answer prefixes name the needle and query, so the system
        # template is filled from the row too; for every other task it has no
        # fields and formatting leaves it unchanged
        prepend_text = (
            ""
            if self.helmet_config.get("use_chat_template")
            else self._templates["system"].format(**doc)
        )

        answer = doc.get("answer")

        metadata: dict = {
            "id": index,
            "task_type": self.task_type,
            "context_size": self.context_size,
            "prepend_text": prepend_text,
            "tag": self.helmet_config["tag"],
        }
        if isinstance(answer, list):
            metadata["all_gold_answers"] = answer
        for scoring_field in ("qa_pairs", "qampari_answers"):
            if scoring_field in doc:
                # ALCE scoring inputs, kept out of the prompt
                metadata[scoring_field] = doc[scoring_field]
        for judge_field in ("keypoints", "expert_summary"):
            if judge_field in doc:
                # inputs the summarization judge needs; extracted ahead of time
                # and shipped with the data rather than derived at scoring time
                metadata[judge_field] = doc[judge_field]
        if "qrel" in doc:
            # graded relevance judgements for the re-ranking scorer
            metadata["qrel"] = doc["qrel"]
        if "question" in doc:
            # The bare question, kept separate from `Instance.question`, which is
            # the whole rendered prompt. An LLM judge needs this one -- handing it
            # the prompt would ship a book-length context to the judge.
            metadata["judge_question"] = doc["question"]

        return Instance(
            question=question,
            gold_answer=answer,
            metadata=metadata,
        )

    @property
    def request_type(self) -> RequestType:
        return RequestType.COMPLETION

    def format_request(self, instance: Instance) -> LMRequest:
        if self.config.formatter is not None:
            return self.config.formatter.format(instance, self.get_fewshot())

        prompt = instance.question
        prepend_text = (instance.metadata or {}).get("prepend_text", "")
        if prepend_text:
            prompt = prompt + "\n" + prepend_text

        return LMRequest(
            request_type=self.request_type,
            prompt=prompt,
        )

    def extract_answer(self, output: LMOutput) -> Any:
        return output.text


class HelmetJsonKvTask(HelmetTask):
    """HELMET-plus json_kv task: extract a value for a given key from a long JSON blob."""

    def _load_dataset(self) -> dict[str, Any]:
        """Load the helmet-plus json_kv data for this task's length tier.

        HELMET-plus data is pre-generated at specific context lengths and
        published as JSONL files on the Hub, so it's downloaded and cached via
        huggingface_hub rather than the standard dataset pipeline.
        """
        return load_json_kv_dataset(
            length_name=self.helmet_config["length_name"],
            shots=self.helmet_config["shots"],
            max_samples=self.config.limit,
            seed=self.config.seed,
        )


def _parse_labeled_output(text: str, prefix: str) -> str | None:
    """Pull the answer out of a `label: N`-style completion.

    Mirrors HELMET's `parse_output` (utils.py): prefer the text following the
    prefix, otherwise fall back to the first line, then strip a repeated
    prefix that chat-style models often echo back.
    """
    patterns = [
        re.compile(f"(?:{re.escape(prefix)})(.*)(?:\n|$)", flags=re.IGNORECASE),
        re.compile(r"(?:^)(.*)(?:\n|$)"),
    ]
    for pattern in patterns:
        match = pattern.search(text)
        if match is not None:
            return re.sub(
                f"^{re.escape(prefix)}", "", match[1].strip(), flags=re.IGNORECASE
            ).strip()
    return None


class HelmetIclTask(HelmetTask):
    """HELMET ICL task: label a text given many labelled demonstrations in context."""

    def _load_dataset(self) -> dict[str, Any]:
        return load_icl_dataset(
            icl_dataset=self.helmet_config["icl_dataset"],
            shots=self.helmet_config["shots"],
            max_samples=self.config.limit,
            seed=self.config.seed,
        )

    def extract_answer(self, output: LMOutput) -> Any:
        return _parse_labeled_output(output.text, prefix="label:")


class HelmetInfbenchTask(HelmetTask):
    """HELMET LongQA task: answer a question about a book-length story."""

    def _load_dataset(self) -> dict[str, Any]:
        return load_infbench_dataset(
            subset=self.helmet_config["infbench_subset"],
            max_context_tokens=self.helmet_config["max_context_tokens"],
            shots=self.helmet_config["shots"],
            max_samples=self.config.limit,
            seed=self.config.seed,
        )

    def extract_answer(self, output: LMOutput) -> Any:
        # HELMET scores the raw generation and the "Answer:"-stripped version,
        # keeping whichever is better; stripping here is the closer of the two
        # for a chat model that echoes the prefix, and a no-op otherwise.
        parsed = _parse_labeled_output(output.text, prefix="Answer:")
        return parsed if parsed else output.text


@dataclass(frozen=True, slots=True)
class HelmetExactMatchScorer(Scorer):
    """Exact match under HELMET's answer normalization.

    HELMET's `exact_match` (drqa_exact_match_score in utils.py) compares
    answers after lowercasing, stripping punctuation, dropping articles, and
    collapsing whitespace. The generic ExactMatchScorer only lowercases and
    strips, so a model answering `label: 42.` or `"42"` would score 0 here but
    1 under HELMET -- a real divergence on realistic ICL outputs, since the
    stop-at-newline sampling leaves trailing punctuation intact.
    """

    name: str = "exact_match"

    def score(self, instance: Instance, output: LMOutput) -> float:
        if output.extracted_answer is None:
            return 0.0

        golds = (instance.metadata or {}).get("all_gold_answers")
        if not golds:
            gold = instance.gold_answer
            if gold is None:
                return 0.0
            golds = list(gold) if isinstance(gold, (list, tuple)) else [gold]

        prediction = _squad_normalize_answer(str(output.extracted_answer))
        return float(any(_squad_normalize_answer(str(g)) == prediction for g in golds))


@dataclass(frozen=True, slots=True)
class InfbenchChoiceScorer(Scorer):
    """Exact match for InfiniteBench multiple choice, following HELMET.

    HELMET accepts the answer in any of the forms a model realistically emits,
    so this counts a response correct when the raw generation, or its
    "Answer:"-stripped form, normalizes to either the bare letter ("B") or the
    letter with its option text ("B. Paris") -- or when the latter appears
    anywhere in the generation, which is what catches a verbose model that
    answers in a sentence.

    It reads `output.text` rather than only `extracted_answer` because that
    last substring rule is defined against the untouched generation.
    """

    name: str = "exact_match"

    def score(self, instance: Instance, output: LMOutput) -> float:
        metadata = instance.metadata or {}
        golds = metadata.get("all_gold_answers") or []
        if not golds:
            gold = instance.gold_answer
            if gold is None:
                return 0.0
            golds = list(gold) if isinstance(gold, (list, tuple)) else [gold]
        golds = [str(g) for g in golds]

        raw = output.text or ""
        candidates = [raw]
        parsed = _parse_labeled_output(raw, prefix="Answer:")
        if parsed:
            candidates.append(parsed)

        normalized_golds = {_squad_normalize_answer(g) for g in golds}
        if any(_squad_normalize_answer(c) in normalized_golds for c in candidates):
            return 1.0

        # golds[1] is the "letter. option text" form
        if len(golds) > 1 and golds[1].lower() in raw.lower():
            return 1.0

        return 0.0


class HelmetNarrativeQaTask(HelmetTask):
    """HELMET LongQA task: answer a question about a novel or movie script.

    Graded by an LLM judge (see `HelmetLongQAJudgeScorer`), so running it needs
    a judge configured -- `OPENAI_API_KEY` for the default OpenAI judge.
    """

    def _load_dataset(self) -> dict[str, Any]:
        return load_narrativeqa_dataset(
            max_context_tokens=self.helmet_config["max_context_tokens"],
            shots=self.helmet_config["shots"],
            max_samples=self.config.limit,
            seed=self.config.seed,
        )

    def extract_answer(self, output: LMOutput) -> Any:
        parsed = _parse_labeled_output(output.text, prefix="Answer:")
        return parsed if parsed else output.text


class HelmetKiltTask(HelmetTask):
    """HELMET RAG task: answer an open-domain question from retrieved passages.

    The question cap is `max_questions`, not `config.limit`: every question
    appears once per gold-passage depth, and the runner applies `limit` to
    instances, which would keep some depths of a question and drop others.
    These tasks leave `limit` unset by default; setting one (e.g. for a smoke
    test) caps instances like any other task, at the cost of the depth sweep.
    """

    def _load_dataset(self) -> dict[str, Any]:
        return load_kilt_dataset(
            task=self.helmet_config["kilt_task"],
            length_name=self.helmet_config["length_name"],
            shots=self.helmet_config["shots"],
            max_samples=self.helmet_config["max_questions"],
            seed=self.config.seed,
            popularity_threshold=self.helmet_config.get("popularity_threshold"),
            max_prompt_tokens=self.helmet_config["max_prompt_tokens"],
        )

    def extract_answer(self, output: LMOutput) -> Any:
        # the prompt asks for "Answer: [answer]", and HELMET scores the parsed
        # form as well as the raw text, keeping whichever is better
        parsed = _parse_labeled_output(output.text, prefix="Answer:")
        return parsed if parsed else output.text


class HelmetMsMarcoTask(HelmetTask):
    """HELMET re-ranking task: order candidate passages by relevance to a query."""

    def _load_dataset(self) -> dict[str, Any]:
        return load_msmarco_dataset(
            length_name=self.helmet_config["length_name"],
            shots=self.helmet_config["shots"],
            max_samples=self.config.limit,
            seed=self.config.seed,
        )

    def extract_answer(self, output: LMOutput) -> Any:
        # a ranked list of document ids, which NDCGScorer scores against qrel
        return parse_rankings(output.text or "")


class HelmetMultiLexSumTask(HelmetTask):
    """HELMET summarization task: summarize the filings of a civil rights lawsuit."""

    def _load_dataset(self) -> dict[str, Any]:
        return load_multi_lexsum_dataset(
            max_context_tokens=self.helmet_config["max_context_tokens"],
            shots=self.helmet_config["shots"],
            max_samples=self.config.limit,
            seed=self.config.seed,
        )


class HelmetAlceTask(HelmetTask):
    """HELMET Cite task: answer a question and cite the documents used.

    Only ALCE's answer-correctness metrics are scored here. Whether the
    citations actually support the claims is AutoAIS's job, which needs an NLI
    model this task does not yet wire up.
    """

    def _load_dataset(self) -> dict[str, Any]:
        return load_alce_dataset(
            task=self.helmet_config["alce_task"],
            length_name=self.helmet_config["length_name"],
            shots=self.helmet_config["shots"],
            max_samples=self.config.limit,
            seed=self.config.seed,
        )


class HelmetRulerTask(HelmetTask):
    """HELMET recall task: one of the RULER needle-in-a-haystack variants.

    Reads the same pre-generated RULER files as the `ruler_*` tasks, but builds
    the prompt the way HELMET does: from its templates and the row's fields,
    with the answer prefix on its own line. The files' prebuilt `input`, which
    the `ruler_*` tasks use, joins the two with a space instead.
    """

    def _load_dataset(self) -> dict[str, Any]:
        ruler_name = f"{self.helmet_config['ruler_task']}__{self.context_size}"
        return load_ruler_dataset(
            task_name=ruler_name,
            data_path=os.path.join(download_ruler_data(), RULER_TASKS[ruler_name]["data"]),
            max_samples=self.config.limit,
            seed=self.config.seed,
        )


_TASK_CLASSES: dict[str, type[HelmetTask]] = {
    "json_kv": HelmetJsonKvTask,
    "ruler": HelmetRulerTask,
    "icl": HelmetIclTask,
    "infbench": HelmetInfbenchTask,
    "narrativeqa": HelmetNarrativeQaTask,
    "kilt": HelmetKiltTask,
    "msmarco": HelmetMsMarcoTask,
    "multi_lexsum": HelmetMultiLexSumTask,
    "alce": HelmetAlceTask,
}

# Per-kind metric configuration. HELMET scores json_kv with substring exact
# match (RecallMetric's substring scorer is equivalent for a single gold
# answer) and ICL with exact match; see HELMET's scripts/collect_results.py.
_TASK_METRICS: dict[str, tuple] = {
    "json_kv": ((RecallMetric(),), "recall"),
    # HELMET's ruler_recall: the fraction of gold values found in the output
    "ruler": ((RecallMetric(),), "recall"),
    # HELMET-normalized exact match, not the generic scorer -- see
    # HelmetExactMatchScorer for why the difference is load-bearing
    "icl": ((AccuracyMetric(name="exact_match", scorer=HelmetExactMatchScorer),), "exact_match"),
    "infbench": ((RougeLF1Metric(),), "rougeL_f1"),
    "infbench_choice": (
        (AccuracyMetric(name="exact_match", scorer=InfbenchChoiceScorer),),
        "exact_match",
    ),
    # HELMET's headline re-ranking metric
    "msmarco": ((NDCGMetric(),), "ndcg_at_10"),
    # HELMET scores RAG with substring exact match over the answer aliases
    "kilt": (
        (AccuracyMetric(name="substring_exact_match", scorer=SubstringExactMatchScorer),),
        "substring_exact_match",
    ),
    # ALCE answer correctness. The citation-grounding half (citation_rec /
    # citation_prec) needs AutoAIS and is not wired up yet.
    "alce_asqa": ((AccuracyMetric(name="str_em", scorer=AlceStrEmScorer),), "str_em"),
    "alce_qampari": (
        (AccuracyMetric(name="qampari_rec_top5", scorer=AlceQampariRecTop5Scorer),),
        "qampari_rec_top5",
    ),
    # HELMET's gpt-4-f1: fluency-gated F1 over key points, via three judge calls
    "summ_book": (
        (AccuracyMetric(name="gpt4_f1", scorer=HelmetSummJudgeScorer(is_book=True)),),
        "gpt4_f1",
    ),
    "summ_lawsuit": (
        (AccuracyMetric(name="gpt4_f1", scorer=HelmetSummJudgeScorer(is_book=False)),),
        "gpt4_f1",
    ),
    # HELMET's gpt-4-score, normalized to [0, 1]; see HelmetLongQAJudgeScorer
    "narrativeqa": (
        (AccuracyMetric(name="gpt4_score", scorer=HelmetLongQAJudgeScorer()),),
        "gpt4_score",
    ),
}


# The loader module behind each task kind, whose prompt templates feed the task hash.
_LOADER_MODULES = {
    "json_kv": helmet_loader,
    "icl": helmet_icl_loader,
    "infbench": helmet_infbench_loader,
    "narrativeqa": helmet_narrativeqa_loader,
    "kilt": helmet_kilt_loader,
    "msmarco": helmet_msmarco_loader,
    "multi_lexsum": helmet_multilexsum_loader,
    "alce": helmet_alce_loader,
}

# Kinds that truncate or trim with the reference tokenizer.
_USES_REFERENCE_TOKENIZER = {"infbench", "narrativeqa", "kilt", "multi_lexsum"}


def _task_settings(task_cfg: dict) -> dict[str, Any]:
    """Everything that shapes a task's prompts but has no TaskConfig field of its own.

    Serialized into the task hash, so a change to shots, token budgets, prompt
    templates or a pinned source revision gives the task a new hash.
    """
    kind = task_cfg["kind"]
    sources: dict[str, Any] = {}
    if kind == "icl":
        sources["icl"] = ICL_DATASETS[task_cfg["icl_dataset"]]
    elif kind == "ruler":
        sources["ruler_data"] = f"{RULER_DATA_REPO}@{RULER_DATA_REVISION}"
    elif kind == "narrativeqa":
        sources["narrativeqa"] = f"{NARRATIVEQA_REPO}@{NARRATIVEQA_REVISION}"
    else:
        sources["helmet_plus"] = f"{HELMET_PLUS_REPO_ID}@{HELMET_PLUS_REVISION}"
    if kind == "infbench":
        sources["infinitebench"] = f"{INFINITEBENCH_REPO}@{INFINITEBENCH_REVISION}"
        sources["infinitebench_subset"] = INFBENCH_SUBSETS[task_cfg["infbench_subset"]]
    if kind in _USES_REFERENCE_TOKENIZER:
        sources["reference_tokenizer"] = f"{REFERENCE_TOKENIZER}@{REFERENCE_TOKENIZER_REVISION}"

    if kind == "ruler":
        user, system, _ = get_ruler_templates(task_cfg["ruler_task"])
        templates = {"user": user, "system": system}
    else:
        templates = {
            name: value
            for name, value in vars(_LOADER_MODULES[kind]).items()
            if "TEMPLATE" in name and isinstance(value, str)
        }
    return {"helmet": task_cfg, "sources": sources, "templates": templates}


def _make_helmet_task_class(task_name: str, task_cfg: dict) -> type[HelmetTask]:
    """Create a task subclass for a HELMET-plus task variant.

    Subclasses carry only class-level attributes (metrics, sampling_params, limit);
    all runtime state is derived from config.name inside HelmetTask.__init__.
    """
    base_cls = _TASK_CLASSES[task_cfg["kind"]]
    metrics, primary_metric = _TASK_METRICS[task_cfg["metrics_key"]]
    # Hand over the Metric itself, not its name: TaskConfig only resolves a
    # Metric instance, and an unresolved primary drops the task out of its
    # suite's average-of-averages.
    primary_metric = next(m for m in metrics if m.name == primary_metric)

    stop_sequences = ("\n",) if task_cfg.get("stop_new_line") else None

    return type(
        f"Helmet_{task_name}",
        (base_cls,),
        {
            "__module__": __name__,
            "metrics": metrics,
            "primary_metric": primary_metric,
            "sampling_params": SamplingParams(
                temperature=0.0,
                top_p=1.0,
                max_tokens=task_cfg["max_gen_toks"],
                stop_sequences=stop_sequences,
            ),
            "limit": task_cfg["limit"],
            "task_settings": _task_settings(task_cfg),
        },
    )


# Dynamically register all HELMET-plus tasks
for _task_name, _task_config in HELMET_TASKS.items():
    _cls = _make_helmet_task_class(_task_name, _task_config)
    # Inject into module globals so pickle can find the class by name
    globals()[_cls.__name__] = _cls
    register(f"helmet_{_task_name}")(_cls)
