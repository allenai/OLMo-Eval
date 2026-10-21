"""Runtime search-backend shim for DeepScholar-Bench generation.

Injected into the sandbox and run instead of ``python -m deepscholar_base.main``.
It monkeypatches ``deepscholar_base.search.recursive_search._process_single_lotus_search_task``
so recursive search uses an alternate retrieval backend without editing the
upstream clone (Python resolves module globals at call time, so overriding the
attribute takes effect even though the caller imported the name):

- "s2":     route every search to the Semantic Scholar API (keyed, request
            timeouts, no arXiv rate-limit hangs). The eval sets
            ``enable_web_search=false`` so only one search pass runs.
- "tavily": skip the hardwired ARXIV corpus; let LOTUS handle the TAVILY corpus.

The upstream standardized row schema is ``title, url, snippet, query, context,
date``; we also populate ``authors`` and ``id`` (used by citation formatting).

Top-level imports are stdlib only so the pure mapping helpers can be unit-tested
without pandas/requests/deepscholar_base present.
"""

from __future__ import annotations

import os
import time
from typing import Any

S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_FIELDS = "title,abstract,url,authors,year,publicationDate,externalIds"
REQUIRED_COLUMNS = ["title", "url", "snippet", "query", "context", "date"]

# Hard wall-clock budget for a single S2 search across retries. The full validation
# encountered frequent 429 responses, so each search needs bounded backoff rather
# than retrying indefinitely.
S2_SEARCH_BUDGET_SEC = 45.0


def _authors_str(paper: dict[str, Any]) -> str:
    names = [a.get("name", "") for a in (paper.get("authors") or []) if a.get("name")]
    return ", ".join(names)


def map_s2_paper(paper: dict[str, Any], query: str) -> dict[str, Any]:
    """Map one Semantic Scholar paper record to a standardized row dict.

    The upstream eval only registers a citation when the markdown link URL matches
    ``arxiv.org/abs/<id>`` (eval/parsers/deepscholar_base.py). So when the paper has
    an ArXiv external ID we emit that URL and id — otherwise the citation would be
    unscorable. Non-arXiv S2 hits keep their S2 URL (usable as synthesis context,
    but not counted as citations by this eval). The URL is never fetched, so this
    does not reintroduce arXiv API calls.
    """
    title = paper.get("title") or ""
    snippet = paper.get("abstract") or ""
    # Emit a %Y-%m-%d date: downstream parses dates strictly and a bare year
    # ("2006") raises, failing the search/filter step. Pad year-only to Jan 1.
    date = paper.get("publicationDate") or ""
    if not date and paper.get("year"):
        date = f"{int(paper['year'])}-01-01"
    ext = paper.get("externalIds") or {}
    arxiv_id = ext.get("ArXiv")
    if arxiv_id:
        url = f"https://arxiv.org/abs/{arxiv_id}"
        paper_id = arxiv_id
    else:
        url = paper.get("url") or ""
        paper_id = ext.get("DOI") or paper.get("paperId") or ""
    return {
        "title": title,
        "url": url,
        "snippet": snippet,
        "query": query,
        "context": f"{title}[{url}]: {snippet}",
        "date": date,
        "authors": _authors_str(paper),
        "id": paper_id,
    }


def _within_cutoff(paper: dict[str, Any], end_date: Any) -> bool:
    """Keep papers strictly before end_date, conservatively handling year-only dates."""
    if end_date is None:
        return True
    from datetime import datetime

    publication_date = paper.get("publicationDate")
    if publication_date:
        text = str(publication_date)
        try:
            return datetime.strptime(text[:10], "%Y-%m-%d").date() < end_date.date()
        except ValueError:
            try:
                return int(text[:4]) < end_date.year
            except ValueError:
                return True

    year = paper.get("year")
    if year:
        try:
            return int(year) < end_date.year
        except (TypeError, ValueError):
            pass
    return True


def s2_search_rows(
    query: str,
    k: int,
    end_date: Any = None,
    api_key: str | None = None,
    logger: Any = None,
    budget_sec: float = S2_SEARCH_BUDGET_SEC,
) -> list[dict[str, Any]]:
    """Query Semantic Scholar and return standardized row dicts (performs I/O).

    Bounded by ``budget_sec`` total across retries. On exhaustion it returns no
    results rather than delaying the query indefinitely.
    """
    import requests

    params: dict[str, Any] = {
        "query": query,
        "limit": max(1, min(int(k), 100)),
        "fields": S2_FIELDS,
    }
    if end_date is not None:
        params["year"] = f"-{end_date.year}"
    headers = {"x-api-key": api_key} if api_key else {}

    start = time.time()
    attempt = 0
    while time.time() - start < budget_sec:
        attempt += 1
        try:
            resp = requests.get(S2_SEARCH_URL, params=params, headers=headers, timeout=(5, 15))
            if resp.status_code == 429:
                remaining = budget_sec - (time.time() - start)
                if logger:
                    logger.warning(
                        f"S2 rate-limited (429) for {query!r} (attempt {attempt}); "
                        f"backing off, {remaining:.0f}s budget left"
                    )
                if remaining <= 0:
                    break
                time.sleep(min(2.0 * attempt, remaining))
                continue
            resp.raise_for_status()
            data = resp.json().get("data") or []
            return [map_s2_paper(p, query) for p in data if _within_cutoff(p, end_date)]
        except (requests.RequestException, ValueError) as e:
            if logger:
                logger.error(f"S2 search failed for {query!r} (attempt {attempt}): {e}")
            time.sleep(1.0)
    if logger:
        logger.warning(
            f"S2 search for {query!r} exhausted {budget_sec:.0f}s budget; returning no results"
        )
    return []


def _make_patched(backend: str, original: Any) -> Any:
    """Build the replacement for ``_process_single_lotus_search_task``."""
    import importlib

    import pandas as pd

    # deepscholar_base exists only inside the sandbox; resolve it at runtime.
    recursive_search = importlib.import_module("deepscholar_base.search.recursive_search")
    arxiv_corpus = recursive_search.WebSearchCorpus.ARXIV

    async def patched(
        configs: Any,
        query: str,
        corpus: Any,
        k: int,
        sort_by_date: bool = False,
        end_date: Any = None,
    ) -> Any:
        if backend == "tavily":
            if corpus == arxiv_corpus:
                return pd.DataFrame(columns=REQUIRED_COLUMNS)
            return await original(configs, query, corpus, k, sort_by_date, end_date)

        # backend == "s2": route every corpus to Semantic Scholar.
        api_key = os.environ.get("S2_API_KEY")
        if not api_key:
            raise RuntimeError("S2_API_KEY is required for search_backend=s2")
        rows = s2_search_rows(query, k, end_date=end_date, api_key=api_key, logger=configs.logger)
        if not rows:
            return pd.DataFrame(columns=REQUIRED_COLUMNS)
        return pd.DataFrame(rows)

    return patched


def _patch_stage_max_tokens(max_tokens: int) -> None:
    """Raise the default stage-LM token budget.

    Upstream ``Configs.initialize_lms`` drops the configured token budget when it
    builds the filter/search/taxonomize LMs, so they fall back to LOTUS's default
    (512) and truncate structured outputs (notably the taxonomize ``Categories``
    JSON, which then fails to parse). Default a missing ``max_tokens`` to the
    configured budget instead. ``max_tokens`` is a cap, not a target, so this does
    not lengthen short stage calls. The explicitly configured generation LM and the
    judge pass ``max_tokens`` themselves and are unaffected.
    """
    import importlib

    lm_attr = "LM"
    lm_cls = getattr(importlib.import_module("lotus.models"), lm_attr)
    init_name = "__init__"
    original_init = getattr(lm_cls, init_name)

    def init(self: Any, *args: Any, **kwargs: Any) -> None:
        if kwargs.get("max_tokens") is None:
            kwargs["max_tokens"] = max_tokens
        original_init(self, *args, **kwargs)

    setattr(lm_cls, init_name, init)


def main() -> None:
    import importlib

    stage_max_tokens = os.environ.get("DEEPSCHOLAR_STAGE_MAX_TOKENS")
    if stage_max_tokens and int(stage_max_tokens) > 0:
        _patch_stage_max_tokens(int(stage_max_tokens))

    backend = os.environ.get("DEEPSCHOLAR_SEARCH_BACKEND", "arxiv")
    if backend in ("s2", "tavily"):
        rs = importlib.import_module("deepscholar_base.search.recursive_search")
        target = "_process_single_lotus_search_task"
        setattr(rs, target, _make_patched(backend, getattr(rs, target)))

    import runpy

    # deepscholar_base.main parses sys.argv[1:], which already holds the flags we
    # were invoked with, so no argv rewriting is needed.
    runpy.run_module("deepscholar_base.main", run_name="__main__")


if __name__ == "__main__":
    main()
