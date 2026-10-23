"""HELMET passage re-ranking (MS MARCO) data loading.

Ports HELMET's `load_msmarco_rerank` (https://github.com/princeton-nlp/HELMET,
data.py): a query plus a list of ID-tagged candidate passages, which the model
must return in relevance order. Scored with NDCG@10 against the graded
relevance labels shipped with each candidate.

Context length is set by how many candidates were retrieved, so each length
tier is a separate pre-retrieved file on `allenai/helmet-plus`; see
`msmarco/manifest.json` there for the tier -> file mapping. As with the RAG
tasks, that fixes the ceiling at standard HELMET's 128k.
"""

import hashlib
import json
import logging
import random
import re
from typing import Any

from olmo_eval.data.helmet_loader import download_helmet_plus_file, sample_jsonl_by_key

logger = logging.getLogger(__name__)

MSMARCO_MANIFEST = "msmarco/manifest.json"
MSMARCO_TASK = "msmarco_rerank_psg"

_USER_TEMPLATE = (
    "You are provided with a list of documents, each indicated by their ID. Rank "
    "each document based on their relevance to the question in descending order "
    "from most relelvant to least relevant texts. Include all documents in the "
    "rankings. Write your answer using the unique IDs, with the following format:\n"
    "Ranking: ID3 > ID1 > ID2\n\n{demos}{context}\n\nQuery: {question}"
)
_SYSTEM_TEMPLATE = "Ranking:"


def parse_rankings(output: str) -> list[str]:
    """Parse a `ID3 > ID1 > ID2` ranking out of a model completion.

    Port of HELMET's `parse_rankings` (utils.py). HELMET returns a dict of
    id -> descending score for pytrec_eval; this returns the equivalent ranked
    list, which is what `NDCGScorer` consumes. Duplicate ids keep their first
    (best) position, as upstream does.
    """
    # strip the bracket/ID decoration the prompt asks for, leaving bare numbers
    output = re.sub(r"[\[\]:]", "", output)
    output = output.lower().replace("id", "")

    # ids are integers, so take the longest `n > n > n` run in the output
    longest = ""
    for match in re.finditer(r"(\d+)(?:\s*>\s*(\d+))*", output):
        if len(match.group(0)) > len(longest):
            longest = match.group(0)

    if longest:
        ranked = [num.strip() for num in longest.split(">") if num.strip().isdigit()]
    else:
        # nothing ranking-shaped in the output; upstream falls back to the raw
        # string, which simply scores zero
        ranked = [output]

    seen, unique = set(), []
    for doc_id in ranked:
        if doc_id not in seen:
            seen.add(doc_id)
            unique.append(doc_id)
    return unique


def load_msmarco_manifest() -> dict[str, dict[str, Any]]:
    with open(download_helmet_plus_file(MSMARCO_MANIFEST), encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _query_key(row: dict[str, Any]) -> Any:
    return row["qid"] if "qid" in row else row["query"]


def _passage_template(contexts: list[dict[str, Any]]) -> str:
    # the MS MARCO candidates carry no titles, but the template switches on it
    # upstream, so keep both forms
    return (
        "[ID: {id}] Document (Title: {title}): {text}"
        if contexts and "title" in contexts[0]
        else "[ID: {id}] Document: {text}"
    )


def _gold_ranking(contexts: list[dict[str, Any]]) -> str:
    return " > ".join(c["id"] for c in sorted(contexts, key=lambda c: c["label"], reverse=True))


def load_msmarco_dataset(
    length_name: str,
    shots: int = 2,
    max_samples: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Load the MS MARCO re-ranking task at one length tier.

    Args:
        length_name: Tier key, e.g. "4k" or "128k".
        shots: Number of worked ranking examples to prepend, drawn from the
            separate k10 demo pool so they stay short.
        max_samples: Cap on the number of queries.
        seed: Seed for query sampling. Demo choice is keyed per query, so it is
            stable regardless of this cap.

    Returns:
        Dictionary with `data` (processed records) and the HELMET prompt templates.
    """
    manifest = load_msmarco_manifest()[MSMARCO_TASK]
    if length_name not in manifest:
        raise ValueError(f"Unknown length '{length_name}'. Available: {sorted(manifest)}")

    entry = manifest[length_name]
    logger.info("Fetching re-ranking data for %s (%s)...", length_name, entry["test_file"])
    data = sample_jsonl_by_key(
        download_helmet_plus_file(entry["test_file"]), max_samples, seed, key=_query_key
    )
    demo_pool = _load_jsonl(download_helmet_plus_file(entry["demo_file"]))

    # when the demo pool is much larger than the eval set, upstream drops the
    # eval queries from it once rather than per instance
    demo_filtered = False
    if len(demo_pool) > 2 * len(data):
        eval_qids = {r["qid"] for r in data}
        demo_pool = [d for d in demo_pool if d["qid"] not in eval_qids]
        demo_filtered = True

    def build(sample: dict[str, Any]) -> dict[str, Any]:
        template = _passage_template(sample["ctxs"])
        passages = "\n\n".join(template.format(**c) for c in sample["ctxs"])

        demo_text = ""
        if shots > 0:
            candidates = demo_pool
            if not demo_filtered:
                candidates = [d for d in demo_pool if d["qid"] != sample["qid"]]

            shuffled = list(candidates)
            seed_value = abs(
                int(hashlib.sha256(str(sample["qid"]).encode("utf-8")).hexdigest(), 16) % 2**31
            )
            random.Random(seed_value).shuffle(shuffled)

            chosen, seen_qids = [], set()
            for demo in shuffled:
                if demo["qid"] in seen_qids:
                    continue
                seen_qids.add(demo["qid"])
                chosen.append(demo)
                if len(chosen) >= shots:
                    break

            for demo in chosen:
                demo_template = _passage_template(demo["ctxs"])
                demo_text += (
                    "\n\n".join(demo_template.format(**c) for c in demo["ctxs"])
                    + f"\n\nQuery: {demo['query']}\nRanking: {_gold_ranking(demo['ctxs'])}"
                    + "\n\n"
                )

        return {
            "context": passages,
            "question": sample["query"],
            "demos": demo_text,
            "answer": _gold_ranking(sample["ctxs"]),
            # relevance judgements, consumed by NDCGScorer
            "qrel": [[c["id"], str(c["label"])] for c in sample["ctxs"]],
        }

    return {
        "data": [build(r) for r in data],
        "prompt_template": _USER_TEMPLATE + "\n" + _SYSTEM_TEMPLATE,
        "user_template": _USER_TEMPLATE,
        "system_template": _SYSTEM_TEMPLATE,
    }
