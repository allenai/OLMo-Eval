"""LitSearch fixed-corpus reranking (plan 005).

597 literature-search queries over a frozen 64K-paper corpus (Ajith et al.,
EMNLP 2024, arXiv:2407.18940). Unlike the agentic ``litsearch`` task (a live
Semantic Scholar smoke test), this is a reproducible, judge-free, tool-free
reranking eval: each query carries a fixed pool of BM25-retrieved candidate
papers, the model selects/ranks the most relevant, and we score Recall@k over
the model's own selection.

The candidate pools are built offline by
``scripts/internal/build_litsearch_pools.py`` (BM25 over the LitSearch
``corpus_clean`` corpus) and hosted on the Hub at
``allenai/litsearch-rerank-pools`` with candidate text baked in, so the eval
needs no corpus download at run time. The BM25 retriever Recall@k baseline is
reported by that build script.

Scoring is deterministic: parse the ranked candidate numbers the model returns,
map them back to corpus IDs, and intersect the top-k with the query's gold IDs.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from olmo_eval.common.metrics import Metric
from olmo_eval.common.scorers.base import Scorer
from olmo_eval.common.types import (
    Instance,
    LMOutput,
    LMRequest,
    RequestType,
    Response,
    SamplingParams,
    Split,
)
from olmo_eval.data import DataSource
from olmo_eval.evals.tasks.common import Task, register

logger = logging.getLogger(__name__)

POOLS_REPO = "allenai/litsearch-rerank-pools"
#: Where the build script writes the artifact locally before it is pushed to the
#: Hub; the regression test reads this path when present.
POOLS_PATH = Path(__file__).parent / "data" / "litsearch_rerank_pools.jsonl"

RECALL_KS = (5, 20)
DEFAULT_SELECT = 20  # how many ranked papers we ask the model to return

_LIST_RE = re.compile(r"\[[^\[\]]*\]")
_RANKED_KEY_RE = re.compile(r'"ranked_papers"\s*:\s*(\[[^\[\]]*\])')

RERANK_PROMPT = (
    "You are helping a researcher find the most relevant scientific papers for a "
    "search query. Below are {n} candidate papers, each with a number, title, and "
    "abstract. Rank the papers most relevant to the query, most relevant first, and "
    "return the numbers of the top {select} as a JSON list, e.g. "
    '{{"ranked_papers": [3, 1, 12]}}. Return only papers that are genuinely '
    "relevant; do not pad the list.\n\nQuery: {query}\n\nCandidates:\n{candidates}"
)


def _coerce_int(x: Any) -> int | None:
    """Coerce a JSON list element to an int, accepting digit strings like "3"."""
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return int(x)
    if isinstance(x, str) and x.strip().lstrip("-").isdigit():
        return int(x.strip())
    return None


def _coerce_int_list(raw: str) -> list[int]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [n for n in (_coerce_int(x) for x in parsed) if n is not None]


def parse_ranked_numbers(text: str) -> list[int]:
    """Extract an ordered list of 1-based candidate numbers from model output.

    Reasoning models emit a long ``<think>`` block (which can contain stray
    bracketed groups) before the answer. The explicit ``"ranked_papers": [...]``
    field is the final answer, so take the last such field and return it as-is --
    including an empty list, which means the model selected nothing. Only when no
    ``ranked_papers`` field is present at all do we fall back to the last
    bracketed list in the text.
    """
    if not text:
        return []
    ranked_matches = list(_RANKED_KEY_RE.finditer(text))
    if ranked_matches:
        return _coerce_int_list(ranked_matches[-1].group(1))
    for match in reversed(list(_LIST_RE.finditer(text))):
        numbers = _coerce_int_list(match.group(0))
        if numbers:
            return numbers
    return []


def recall_at_k(gold: set[int], ranked_ids: list[int], k: int) -> float:
    """Fraction of gold IDs in the model's top-k selected corpus IDs."""
    if not gold:
        return 0.0
    return len(gold & set(ranked_ids[:k])) / len(gold)


@dataclass(frozen=True)
class RerankRecallScorer(Scorer):
    """Placeholder; recall is computed per response in ``score_responses``."""

    name: str = "litsearch_rerank"
    score_key: str = "recall@5"

    def score(self, instance: Instance, output: LMOutput) -> float:
        return (output.metadata or {}).get(self.score_key, 0.0)


@dataclass(frozen=True)
class RecallAtKMetric(Metric):
    """Mean Recall@k over the model's reranked selection, precomputed per response."""

    name: str = "recall@5"
    scorer: type[Scorer] = RerankRecallScorer

    def compute(self, responses: Sequence[Response]) -> float:
        if not responses:
            return 0.0
        return sum(r.scores.get(self.name, 0.0) for r in responses) / len(responses)

    def supports_scorer_fallback(self) -> bool:
        return False


LITSEARCH_RERANK_METRICS = tuple(RecallAtKMetric(name=f"recall@{k}") for k in RECALL_KS)


@register("litsearch_rerank")
class LitSearchRerank(Task):
    """Fixed-corpus LitSearch reranking: model reranks BM25 candidates, scored Recall@k."""

    data_source = DataSource(path=POOLS_REPO, split="train")
    split = Split.TRAIN
    metrics = LITSEARCH_RERANK_METRICS
    primary_metric = RecallAtKMetric(name="recall@5")
    # Reasoning models spend most of their budget thinking before emitting the
    # ranked list. Leave generation uncapped (bounded only by the model's context
    # window) so the JSON answer is never truncated away.
    sampling_params = SamplingParams(temperature=0.0, max_tokens=None)

    #: How many candidates to present in the prompt (<= stored pool size).
    num_candidates: int = 50
    #: How many ranked papers to ask the model to return.
    num_select: int = DEFAULT_SELECT

    @property
    def instances(self) -> Iterator[Instance]:
        yield from self._load_instances_cached()

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        query = doc.get("query", "")
        gold_ids = [int(c) for c in doc.get("gold_corpusids", []) if c is not None]
        candidates = doc.get("candidates", [])[: self.num_candidates]
        if not query or not gold_ids or not candidates:
            return None

        return Instance(
            question=query,
            metadata={
                "case_id": doc.get("query_id", index),
                "gold_corpusids": gold_ids,
                "candidate_corpusids": [int(c["corpusid"]) for c in candidates],
                "candidates": candidates,
                "query_set": doc.get("query_set", ""),
                "specificity": doc.get("specificity"),
                "quality": doc.get("quality"),
                "index": index,
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        candidates = instance.metadata["candidates"]
        lines = []
        for number, cand in enumerate(candidates, start=1):
            title = (cand.get("title") or "").strip()
            abstract = (cand.get("abstract") or "").strip()
            lines.append(f"[{number}] {title}\n{abstract}")
        prompt = RERANK_PROMPT.format(
            n=len(candidates),
            select=self.num_select,
            query=instance.question,
            candidates="\n\n".join(lines),
        )
        return LMRequest(
            request_type=RequestType.CHAT,
            messages=({"role": "user", "content": prompt},),
        )

    async def score_responses(
        self,
        responses: Sequence[Response],
        context: Any = None,
    ) -> Sequence[Response]:
        """Map the model's ranked numbers to corpus IDs and score Recall@k."""
        for response in responses:
            gold = set(response.instance.metadata.get("gold_corpusids", []))
            candidate_ids = response.instance.metadata.get("candidate_corpusids", [])

            output = response.outputs[0] if response.outputs else None
            numbers = parse_ranked_numbers(output.text if output else "")
            # 1-based candidate numbers -> corpus IDs, preserving model order, deduped.
            ranked_ids: list[int] = []
            seen: set[int] = set()
            for number in numbers:
                pos = number - 1
                if 0 <= pos < len(candidate_ids):
                    cid = candidate_ids[pos]
                    if cid not in seen:
                        seen.add(cid)
                        ranked_ids.append(cid)

            for k in RECALL_KS:
                response.scores[f"recall@{k}"] = recall_at_k(gold, ranked_ids, k)

            if output is not None:
                if output.metadata is None:
                    output.metadata = {}
                output.extracted_answer = ranked_ids
                output.metadata["litsearch_rerank_selected"] = ranked_ids
                for k in RECALL_KS:
                    output.metadata[f"recall@{k}"] = response.scores[f"recall@{k}"]

        return responses
