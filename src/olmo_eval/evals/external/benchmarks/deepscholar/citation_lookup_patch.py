"""In-place patch for the pinned deepscholar-bench citation lookup.

Upstream ``eval/utils.py::get_citation_count_from_title`` hardcodes
``mailto="your@email.com"`` and sends no API key, so at benchmark volume every
request lands in OpenAlex's anonymous tier and 429s. Since it returns ``None`` on
any failure, ``document_importance`` (median system-citation / median exemplar-
citation) collapses toward 0 on rate-limited runs.

This script replaces that lookup with an authenticated, cached, throttled version
that retries transient failures. It is applied to the cloned repo at setup and
fails loudly if the upstream function has changed (i.e. the pinned ref moved),
which is the signal to re-verify the substitution.

Usage: ``python citation_lookup_patch.py <path-to-eval/utils.py>``
"""

from __future__ import annotations

import pathlib
import sys

_FUNCTION_OLD = """def get_citation_count_from_title(
    title, mailto="your@email.com", similarity_threshold=0.8
):
    try:
        search_url = f"https://api.openalex.org/works?search={title}&mailto={mailto}"
        response = requests.get(search_url, timeout=10)
        response.raise_for_status()
        results = response.json().get("results", [])

        if results:
            top_result = results[0]
            paper_title = top_result.get("display_name", "")
            citation_count = top_result.get("cited_by_count", 0)

            # Compare similarity
            similarity = jaccard_similarity(title, paper_title)
            if similarity >= similarity_threshold:
                return citation_count
            else:
                return None
    except Exception as e:
        print(f"Error fetching citation count: {e}")
        return None
"""

_FUNCTION_NEW = '''_OPENALEX_CACHE = {}
_OPENALEX_LAST_REQUEST_AT = 0.0
_OPENALEX_MIN_INTERVAL_SECONDS = 0.1
_OPENALEX_MAX_ATTEMPTS = 6


def get_citation_count_from_title(title, mailto=None, similarity_threshold=0.8):
    """Fetch a citation count without silently converting API failures to missing data."""
    api_key = os.environ.get("OPENALEX_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENALEX_API_KEY is required for document_importance; "
            "inject it with the benchmark launch command"
        )

    normalized_title = " ".join(str(title).split())
    cache_key = (normalized_title.casefold(), float(similarity_threshold))
    if cache_key in _OPENALEX_CACHE:
        return _OPENALEX_CACHE[cache_key]

    # OpenAlex interprets ``?`` and ``*`` in ``search`` as wildcards, which
    # require a different API mode and turn ordinary paper titles into HTTP 400s.
    # Strip search syntax and quote the remaining words as an exact phrase. This
    # also gives substantially better title matching than an unquoted word query.
    search_title = " ".join(
        "".join(
            character if character.isalnum() or character.isspace() else " "
            for character in normalized_title
        ).split()
    )
    if not search_title:
        _OPENALEX_CACHE[cache_key] = None
        return None

    params = {
        "search": f'"{search_title}"',
        "api_key": api_key,
        "per_page": 1,
        "select": "display_name,cited_by_count",
    }
    email = mailto or os.environ.get("OPENALEX_EMAIL")
    if email:
        params["mailto"] = email

    global _OPENALEX_LAST_REQUEST_AT
    last_error = None
    for attempt in range(_OPENALEX_MAX_ATTEMPTS):
        elapsed = time.monotonic() - _OPENALEX_LAST_REQUEST_AT
        if elapsed < _OPENALEX_MIN_INTERVAL_SECONDS:
            time.sleep(_OPENALEX_MIN_INTERVAL_SECONDS - elapsed)

        try:
            response = requests.get(
                "https://api.openalex.org/works", params=params, timeout=10
            )
        except requests.RequestException as error:
            # RequestException text can contain a prepared URL with credentials.
            last_error = f"{type(error).__name__}: request failed"
            if attempt + 1 < _OPENALEX_MAX_ATTEMPTS:
                time.sleep(min(2 ** attempt, 30.0))
            continue

        _OPENALEX_LAST_REQUEST_AT = time.monotonic()
        if response.status_code == 429 or response.status_code >= 500:
            last_error = RuntimeError(
                f"OpenAlex transient HTTP {response.status_code}"
            )
            retry_after = response.headers.get("Retry-After")
            try:
                delay = float(retry_after) if retry_after else 2 ** attempt
            except (TypeError, ValueError):
                delay = 2 ** attempt
            # Per-process hash randomization gives concurrent jobs different jitter.
            jitter = 0.75 + (abs(hash(normalized_title)) % 500) / 1000
            time.sleep(min(max(delay, 0.25) * jitter, 30.0))
            continue

        if response.status_code >= 400:
            # Do not use raise_for_status(): requests includes the prepared
            # URL (and therefore our api_key query parameter) in HTTPError.
            detail = str(response.text or "").strip()[:500]
            detail = detail.replace(api_key, "<redacted>")
            raise RuntimeError(
                f"OpenAlex HTTP {response.status_code} for title "
                f"{normalized_title!r}: {detail}"
            )

        results = response.json().get("results", [])
        citation_count = None
        if results:
            top_result = results[0]
            paper_title = top_result.get("display_name", "")
            similarity = jaccard_similarity(search_title, paper_title)
            if similarity >= similarity_threshold:
                citation_count = top_result.get("cited_by_count", 0)
        _OPENALEX_CACHE[cache_key] = citation_count
        return citation_count

    raise RuntimeError(
        f"OpenAlex lookup failed after {_OPENALEX_MAX_ATTEMPTS} attempts "
        f"for title {normalized_title!r}: {last_error}"
    )
'''


def main() -> None:
    path = pathlib.Path(sys.argv[1])
    source = path.read_text()
    if source.count(_FUNCTION_OLD) != 1:
        raise SystemExit(
            f"citation patch: expected function not found exactly once in {path}; "
            "the pinned deepscholar-bench ref may have changed."
        )
    patched = source.replace(_FUNCTION_OLD, _FUNCTION_NEW)
    path.write_text(patched)


if __name__ == "__main__":
    main()
