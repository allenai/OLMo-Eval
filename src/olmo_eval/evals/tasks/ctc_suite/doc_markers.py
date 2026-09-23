"""
Document-boundary markers for CTC prompts, so the eval can match CTC SFT data built with them.

The OLMo-core CTC SFT converter (``src/scripts/data/convert_unified_to_document_landmark.py``,
default mode) renders the SAME prompt body as ``CTC_SUITE_PROMPT_FORMAT=chat`` and then wraps the
document block in ``<|box_start|>`` / ``<|box_end|>`` before applying the chat template. A model
trained on those shards has only ever seen the markers; ``CTC_SUITE_DOC_MARKERS=1`` applies the same
wrap here so the eval prompt is token-identical to its training prompts.

This is a verbatim port of OLMo-core's ``olmo_core.data.document_chunk_landmark._wrap_documents`` /
``_wrap_item_lines`` (summary spans omitted: no CTC shard uses them). Keep the two in sync -- a
divergence here is a silent train/eval format mismatch, not an error.

Per task, as the converter chunks them: ``oolong`` wraps each ``||``-delimited item line; every
other task wraps each ``documents[i]`` body, with the chunks made contiguous so only the
instruction/question prefix and the trailing query stay outside a chunk.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from re import Pattern

DOC_START_STR = "<|box_start|>"
DOC_END_STR = "<|box_end|>"

#: Env var: ``1`` wraps documents in the markers (chat format only). Unset/``0`` = no markers.
DOC_MARKERS_ENV = "CTC_SUITE_DOC_MARKERS"

#: Tasks the converter chunks by item LINE (``--chunk-by line``); all others by document.
LINE_CHUNKED = {"oolong": re.compile(r"\|\|")}

#: What the prompt renderer does to a document's text before embedding it (so the verbatim search
#: matches). reorder collapses a passage's paragraph breaks to single newlines.
_RENDERER_TEXT_NORMALIZERS: dict[str, Callable[[str], str]] = {
    "reorder": lambda body: body.replace("\n\n", "\n"),
}


def doc_markers_enabled() -> bool:
    """:returns: Whether ``CTC_SUITE_DOC_MARKERS`` asks for markers, read at request time."""
    v = os.environ.get(DOC_MARKERS_ENV, "0").strip().lower()
    if v not in ("0", "1", "false", "true", ""):
        raise ValueError(f"{DOC_MARKERS_ENV}={v!r}; expected 0 or 1")
    return v in ("1", "true")


def wrap_item_lines(text: str, item_re: Pattern, start_str: str, end_str: str) -> str:
    """Wrap each line matching ``item_re``; a newline between two item lines joins the chunk."""
    lines = text.split("\n")
    is_item = [bool(line.strip() and item_re.search(line)) for line in lines]
    out: list[str] = []
    for i, line in enumerate(lines):
        lead = "" if i == 0 else "\n"
        if is_item[i] and i > 0 and is_item[i - 1]:
            out.append(f"{start_str}{lead}{line}{end_str}")
        elif is_item[i]:
            out.append(f"{lead}{start_str}{line}{end_str}")
        else:
            out.append(f"{lead}{line}")
    return "".join(out)


def wrap_documents(
    text: str, documents: list[dict], start_str: str, end_str: str, task: str = ""
) -> str:
    """Wrap the document block in contiguous chunks, one per document found verbatim in ``text``."""
    normalize = _RENDERER_TEXT_NORMALIZERS.get(task)
    body_spans: list[tuple[int, int]] = []
    cursor = 0
    for d in documents:
        body = str(d.get("text", "")).strip()
        if not body:
            continue
        if normalize is not None:
            body = normalize(body)
        idx = text.find(body, cursor)
        if idx == -1:
            idx = text.find(body)
        if idx == -1:
            continue  # the converter leaves such a document unwrapped too
        body_spans.append((idx, idx + len(body)))
        cursor = idx + len(body)
    if not body_spans:
        return text

    chunk_spans: list[tuple[int, int]] = []
    for i, (bstart, bend) in enumerate(body_spans):
        if i == 0:
            para = text.rfind("\n\n", 0, bstart)
            cstart = 0 if para == -1 else para
        else:
            cstart = chunk_spans[i - 1][1]
        cend = bend
        if i == len(body_spans) - 1:
            while cend < len(text) and text[cend] in "\n\r\t ":
                cend += 1
        chunk_spans.append((cstart, cend))

    pieces: list[str] = []
    pos = 0
    for cstart, cend in chunk_spans:
        if cstart > pos:
            pieces.append(text[pos:cstart])
        pieces += [start_str, text[cstart:cend], end_str]
        pos = cend
    if pos < len(text):
        pieces.append(text[pos:])
    return "".join(pieces)


def add_doc_markers(body: str, example: dict, task: str) -> str:
    """
    Wrap a rendered prompt body exactly as the CTC SFT converter does for ``task``.

    :param body: The chat-format prompt body (``build_prompt(..., use_alpaca=False)``).
    :param example: The unified example the body was rendered from.
    :param task: The spec name (``oolong``, ``retrieval``, ``reorder``, ...).
    """
    if task in LINE_CHUNKED:
        return wrap_item_lines(body, LINE_CHUNKED[task], DOC_START_STR, DOC_END_STR)
    return wrap_documents(body, example.get("documents", []), DOC_START_STR, DOC_END_STR, task)
