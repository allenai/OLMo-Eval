"""
Fit CTC rows to a model's position window by shortening DISTRACTOR documents, so no row is
left-truncated at generation time.

Why: the r256k rung labels a size, not a token count -- through the chat + doc-marker prompt path
its prompts run to ~272k Qwen3.5 tokens, and the ``olmo_core`` provider left-truncates anything
longer than ``max_model_len - max_tokens``. That cuts the chat header and the instruction off
43-100% of the r256k rows of nq / hpqa / qdmatch_nq / outlier / contradiction / rerank (measured
2026-09-29), so those cells score a prompt the model was never trained on.

What it changes: only the ``text`` of documents that carry no part of the answer, each cut at a
word boundary by the same fraction, just enough that the rendered prompt plus the decode budget
fits (with a small margin). The document COUNT, the order and every index field stay as they are,
so the gold, the grader and the corpus-size semantics (e.g. outlier's K) are untouched. Protected,
per spec:

* ``retrieval`` / ``outlier`` / ``contradiction``: every document named in ``gold_doc_indices``;
* ``qdmatch``: both members of every ``gold_pairs`` entry;
* ``rerank``: ``gold_doc_indices``, ``hard_neg_indices`` and the top-``--rerank-keep`` documents by
  ``ce_scores`` (its metric grades against those scores).

Index bases differ by task (contradiction 1-based, the others 0-based), so each named index ``g``
protects both ``g`` and ``g - 1``. ``oolong`` is refused: its single document is the list of items
the answer counts over, so no part of it is a distractor.

Rows that already fit are written unchanged. Writes ``<out>/<subset>/rung_<tokens>.jsonl`` -- the
layout ``CTC_SUITE_DATA_ROOT`` reads -- with each row's fields exactly as the HF parquet stores
them, plus ``_fit`` (``{"shaved_fraction", "prompt_tokens_before", "prompt_tokens_after"}``).

    python scripts/ctc_suite/fit_to_window.py --cells ctc_nq:r256k ctc_outlier:r256k \\
        --out /tmp/ctc_fitted --tokenizer Qwen/Qwen3.5-0.8B --max-model-len 262144
"""

from __future__ import annotations

import argparse
import json
import os
import sys

REFUSED = {"oolong"}
MARGIN = 64  # tokens of slack under the limit


def _log(msg: str) -> None:
    print(f"[fit] {msg}", file=sys.stderr, flush=True)


def protected_indices(example: dict, spec_name: str, rerank_keep: int) -> set[int]:
    """Indices of the documents that carry part of the answer (see the module docstring)."""
    named: list[int] = []
    if spec_name == "qdmatch":
        for pair in example.get("gold_pairs") or []:
            named += [int(x) for x in pair]
    else:
        for g in example.get("gold_doc_indices") or []:
            named += [int(x) for x in g] if isinstance(g, (list, tuple)) else [int(g)]
    if spec_name == "rerank":
        named += [int(x) for x in example.get("hard_neg_indices") or []]
        scores = example.get("ce_scores") or []
        # CE=None marks pooled-foreign documents: distractors the grader never ranks
        scored = [i for i, v in enumerate(scores) if v is not None]
        top = sorted(scored, key=lambda i: -float(scores[i]))[:rerank_keep]
        named += top
    out = set()
    for g in named:
        out |= {g, g - 1}
    return out


class _SpecOnly:
    """Just enough of a CTCSuiteTask for ``get_sampling_params`` (which reads only ``spec``)."""

    def __init__(self, spec) -> None:
        self.spec = spec


def _shave(text: str, frac: float) -> str:
    """Keep the leading ``1 - frac`` of ``text``, cut back to the last word boundary."""
    keep = int(len(text) * (1.0 - frac))
    if keep >= len(text):
        return text
    cut = text.rfind(" ", 0, keep)
    return text[: cut if cut > 0 else keep].rstrip()


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--cells", nargs="+", required=True, help="<row>:<rung>, e.g. ctc_nq:r256k")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--max-model-len", type=int, default=262144)
    ap.add_argument("--rerank-keep", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0, help="first N rows only (for testing)")
    args = ap.parse_args()

    os.environ.setdefault("CTC_SUITE_PROMPT_FORMAT", "chat")
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    from transformers import AutoTokenizer

    import olmo_eval.evals.tasks.ctc_suite as S
    from olmo_eval.evals.tasks.ctc_suite.doc_markers import add_doc_markers, doc_markers_enabled

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    fs = HfFileSystem()

    def n_tokens(example: dict, spec) -> int:
        # exactly the request CTCSuiteTask.format_request builds in chat format
        body = spec.build_prompt(example, query_position=S.QUERY_POSITION, use_alpaca=False)
        if doc_markers_enabled():
            body = add_doc_markers(body, example, spec.name)
        text = tok.apply_chat_template(
            [{"role": "user", "content": body}], tokenize=False, add_generation_prompt=True
        )
        return len(tok(text, add_special_tokens=False)["input_ids"])

    for cell in args.cells:
        row_name, rung = cell.split(":")
        row = S.ROSTER.get(row_name) or S.OOD_ROSTER[row_name]
        spec = S._resolve_spec(row.spec)
        if spec.name in REFUSED:
            raise SystemExit(f"[fit] {cell}: {spec.name} has no distractor documents to shorten")
        # the decode budget the eval will reserve (the task's own rule, incl. the rerank cap env)
        max_tokens = S.CTCSuiteTask.get_sampling_params(_SpecOnly(spec), None).max_tokens
        room = args.max_model_len - max_tokens - MARGIN
        path = f"datasets/{S.HF_DATASET}/data/{row.subset}/{rung}.parquet"
        examples = pq.read_table(fs.open(path)).to_pylist()
        if args.limit:
            examples = examples[: args.limit]
        tokens = row.rung_alias.get(rung, S.RUNG_TOKENS[rung])
        os.makedirs(os.path.join(args.out, row.subset), exist_ok=True)
        dst = os.path.join(args.out, row.subset, f"rung_{tokens}.jsonl")
        shaved = 0
        with open(dst, "w") as f:
            for i, ex in enumerate(examples):
                before = n_tokens(ex, spec)
                frac, after = 0.0, before
                if before > room:
                    keep = protected_indices(ex, spec.name, args.rerank_keep)
                    free = [j for j in range(len(ex["documents"])) if j not in keep]
                    originals = [ex["documents"][j]["text"] for j in free]
                    free_tok = sum(
                        len(tok(t, add_special_tokens=False)["input_ids"]) for t in originals
                    )
                    frac = min(0.95, (before - room) / max(free_tok, 1))
                    for _ in range(8):  # word-boundary cuts and marker tokens: re-measure, top up
                        for j, t in zip(free, originals, strict=True):
                            ex["documents"][j]["text"] = _shave(t, frac)
                        after = n_tokens(ex, spec)
                        if after <= room:
                            break
                        frac = min(0.95, frac + (after - room) / max(free_tok, 1) + 0.002)
                    if after > room:
                        raise SystemExit(
                            f"[fit] {cell} row {i}: {after} tokens > {room} after shaving"
                        )
                    shaved += 1
                ex["_fit"] = {
                    "shaved_fraction": round(frac, 4),
                    "prompt_tokens_before": before,
                    "prompt_tokens_after": after,
                }
                f.write(json.dumps(ex) + "\n")
        _log(
            f"{cell}: {shaved}/{len(examples)} rows shortened to fit {room} prompt tokens -> {dst}"
        )


if __name__ == "__main__":
    main()
