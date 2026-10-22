# CTC suite: 22 long-context tasks, ladders from 2k to 1M tokens

Each task puts a corpus of N documents in-prompt and asks a question whose difficulty scales with
how much of the corpus must be tracked *simultaneously* — from O(N) retrieval, through O(N²)
relational tasks (find every contradicting pair), to O(NM)/O(N³) structural ones (cluster
everything, find planted triples). Every task has a context ladder; a rung label is a token budget,
not a document count — see the accuracy caveat below before treating it as a measured length.

## Quickstart

```bash
# no token needed: the dataset (PrasannSinghal/ctc-suite-eval) is public

# preview without a model
uv run olmo-eval run -m mock -t ctc_contradiction:r32k --dry-run

# one task, one rung
uv run olmo-eval run -m <model> -t ctc_nq:r64k --save-predictions

# suites -- by context length
uv run olmo-eval run -m <model> -t ctc:figure       # all 22 tasks, 2k-32k grid (108 runs)
uv run olmo-eval run -m <model> -t ctc:r128k        # every task at 128k (one x-axis column)
uv run olmo-eval run -m <model> -t ctc:oolong       # one task's whole ladder
uv run olmo-eval run -m <model> -t ctc:xlong        # everything above 32k (69 runs)

# suites -- by corpus-tracking demand (the axis the suite is named for)
uv run olmo-eval run -m <model> -t ctc:low          # the 12 O(N) rows, every rung (106 runs)
uv run olmo-eval run -m <model> -t ctc:high         # the 10 O(N^2)+ rows, every rung (71 runs)
uv run olmo-eval run -m <model> -t ctc:high:figure  # ... 2k-32k only (49 runs)
uv run olmo-eval run -m <model> -t ctc:low:xlong    # the two axes compose (47 runs)

uv run olmo-eval run -m <model> -t ctc              # all 177 task x rung combinations
```

## low-CTC vs high-CTC

`ctc:low` is the 12 rows where an answer-bearing document exists and the work is finding it --
O(N) in corpus size, and in principle solvable by a retriever. `ctc:high` is the 10 rows where the
answer is a *relation over* documents with no single span to retrieve: every contradicting pair
(O(N^2)), the clustering of everything (O(NM)), the planted triple (O(N^3)). The split is 12/10,
declared per row as `RosterRow.ctc_class` and pinned by test; the per-row `complexity` field
records which class of the four it is.

This axis is **orthogonal to context length** -- `ctc:figure`/`ctc:xlong` cut the same 22 tasks by
rung. `ctc:high:figure` is the cheapest useful probe of the two: it is where a model that merely
retrieves well separates from one that tracks a corpus, at grid-sized cost. Aggregation is
DISPLAY_ONLY for these suites too; read low-vs-high per task or as a gap on a shared metric, never
as one mean against another.

A bare task name (`-t ctc_nq`) evaluates the 32k rung. Suite aggregation is DISPLAY_ONLY on
purpose: the metrics are heterogeneous (f1, pair f1, kendall tau, ce_pos_recall, partial credit)
and a cross-task average would be meaningless.

## The 22 tasks

| task | class | metric | ladder top | notes |
|---|---|---|---|---|
| ctc_fiqa, ctc_nq, ctc_hpqa, ctc_msmarco, ctc_scifact, ctc_obliq, ctc_niah | O(N) | f1 (gold ids) | 1M (msmarco 512k) | retrieval family |
| ctc_rerank | O(N) | **ce_pos_recall** | 512k | see metric note below |
| ctc_oolong | O(N) | partial credit | 1M | aggregate questions over a line stream |
| ctc_outlier_amzn, ctc_outlier_fixedm | O(N) | f1 (set) | 1M / 512k | fixed-K controls |
| ctc_absence | O(N) | f1 (set) | 16k | deleted sentences, versions aligned and in order |
| ctc_outlier | O(NM) | f1 (set) | 1M | K grows with N (~n/9) — the scale-K row |
| ctc_qdmatch_fiqa/nq/hpqa | O(N²) | pair f1 | 512k / 1M / 256k | query↔document matching |
| ctc_contradiction | O(N²) | f1 (pairs) | 1M | PubMed claims, IID realistic mode |
| ctc_xabsence | O(N²) | f1 (set) | 32k | exact-copy orphans across two unordered corpora |
| ctc_strmatch | O(N²) | f1 (pairs) | 32k | planted shared word-runs |
| ctc_reorder | O(N²) | kendall tau | 16k | restore reading order |
| ctc_grouping | O(NM) | pairwise f1 | 32k | cluster abstracts |
| ctc_textgroups | O(N³) | f1 (groups) | 32k | planted feature-sum triples |

Ladder tops below 1M are **source-corpus ceilings, not laziness** — e.g. qdmatch_hpqa exhausts all
4,000 labeled HotpotQA units at 256k; absence/reorder are bounded by contiguous-book length. Each
cap is documented on its RosterRow.

## Numbers that must travel with results

- Small base models often cannot answer in the suite's answer space at all, which floors every
  row below chance rather than ranking them. Measured on the OLMo hybrid ladder at r2k: the share
  of generations emitting any `[id]` is 3.5-13.5% at 450M and 5-61% at 810M, reaching ~100% at
  1.4B. Read `parse_rate` before reading a score from a sub-1B checkpoint.
- Rungs ≥256k hold **125 examples** (seeded subsample; SE ≈ ±0.041 at f1≈0.7). `ctc_scifact` is
  300 and `ctc_obliq` 123–126 at every rung. `ctc_absence` r16k is 148 (bounded by book length) and
  `ctc_oolong` is *larger* than the default at r64k (668) and r128k (669). Everything else is 500.
  Every exception is recorded in `RosterRow.eval_size` and pinned by test; quote sizes inline.
- **rerank's metric is ce_pos_recall**: the fraction of documents with cross-encoder score > 0
  (median 3, p90 5 per example) present in the model's first 10 emitted ids. Single-qrel MRR@10
  saturates at ~0.98 and is emitted only as a secondary. Relevance in this data is bimodal
  (nothing between CE −5 and 0), which is also why an NDCG@10 over CE gains would collapse to
  the top-3 — measured before this metric was chosen. **14 of 500 rows at r2k and at r32k have no
  CE-positive document at all**, so the metric is undefined there rather than zero: those rows are
  excluded from the mean, and `ce_ref_available` reports what fraction of rows the mean covers.
  Averaging them in as zeros held a perfect model's ceiling at 0.972.
- **`ctc_reorder`'s kendall_tau ranges over [-1, 1]**, not [0, 1]: 0.0 is chance, not the floor,
  and −1.0 is an exactly reversed ordering. Suite aggregation is DISPLAY_ONLY partly for this
  reason; never average it by hand against the f1-style rows.
- **`ctc_obliq` decodes with a 512-token budget**, not the retrieval family's 64. Its gold sets
  reach 64 ids where the rest of the family has at most 3, and the longest perfect answer measures
  311 Qwen3 tokens at r32k and 404 at r1m — under a 64-token budget 27/126 rows at r32k were
  truncated into partial credit, capping the row at mean f1 0.953.
- **`ctc_outlier` and `ctc_outlier_fixedm` are graded under an instruction that says "product
  reviews"** even though their corpora are Wikipedia topic chunks (only `ctc_outlier_amzn` is
  reviews). The wording is the reference's and is kept verbatim, because every published number for
  those rows was produced under it; changing it would reprice the row and invalidate the
  checkpoints' format fingerprint.
- **Reasoning models.** A generation whose `<think>` block is never closed is scored as a parse
  failure, not handed to the parser: the ids in an unfinished trace are the ones the model was
  still weighing, and crediting them rewards thinking out loud. Give such a model room to finish
  instead: the suite's budget is the config's `sampling_params`, the runner lays `-o` overrides
  over it field by field, and the result is both what is sent and what is hashed — so
  `-t ctc_nq:r2k -o max_tokens=4096` genuinely raises it and genuinely changes the task hash. (An
  earlier version of this suite stored one budget and sent another.)
- **A rung label is a build target, not a per-task guarantee.** Labels were set from the reference
  prompt path, and the xlong rungs were confirmed against it (real 1M rows measure p50 1.03–1.07M
  tokens). But on the 2k–32k ladder, measurement through the Qwen3.5 tokenizer found two rows
  systematically short of their label — `ctc_contradiction` ~1.5x and `ctc_niah` ~2.9x, both
  consistent across that row's rungs — and `ctc_xabsence`'s labels are estimates pending a prefill
  measure. Trends *within* a task are unaffected, since the scaling is consistent down the row;
  **a cross-task comparison "at the same rung" is not comparing the same context length.** Quote
  measured tokens on any absolute-length claim.
- **Every row reports `parse_rate` alongside its primary metric.** A parse-rate collapse is a
  decoding/stopping regression wearing an accuracy drop's clothes, so check it before believing a
  low score. It is in `metrics.json` and in each prediction's `instance_metrics`. (It used to be
  written only to `output.metadata` as `ctc_parse_ok`, which the predictions writer drops -- the
  advice was unfollowable from the shipped files.)
- **A near-zero score is a parser hypothesis until you have read the raw generations.** Measured
  2026-09-09 on Qwen3.5-4B-Base at r2k: `ctc_textgroups` scores 0.000 with `parse_rate` 0.00
  because the model never emits a pair list -- it opens with a plan and then repeats
  "Combination N: ... Invalid" verbatim until the budget ends it. Raising the budget 200 -> 1024
  was tried and reverted; 8/10 still hit the larger cap. That row is repetition-gated, not
  truncation-gated.
- Contexts ≥256k exceed most models' native windows; the serving side (YaRN etc.) is the caller's
  responsibility and belongs next to any reported number.
- **`--save-requests` (on by default) writes each corpus twice.** The raw example travels whole in
  `Instance.metadata` -- the prompt builder, parser and scorer all read different parts of it, and
  slicing it is how field conventions historically drifted -- and `requests.jsonl` serializes that
  metadata next to the rendered prompt that already contains the same documents. Measured at r2k
  the two are ~1:1 (7.3 KB each per row); at 1M tokens that is ~8 MB per row and roughly 1 GB per
  task×rung, half of it duplication. Pass `--no-save-requests` on xlong runs unless you need the
  prompts on disk; predictions are unaffected.

### Grading changed after the published grid — for seven rows

The vendored graders are golden-fixture tested against the pre-migration implementation, but five
fixes have landed *since* the reference grid was produced, and each one raises a correct answer's
score. **Numbers for these rows are not comparable across the change**, and where an old number
exists it came from the old grader:

| Row(s) | What changed | Effect |
|---|---|---|
| `ctc_outlier`, `ctc_outlier_amzn`, `ctc_outlier_fixedm` | stop preset `newline` → `outliers`: the instruction mandates a sentence before the `Outliers:` line, so the newline stop fired there and the ids never reached the parser | a gold-derived perfect answer went from a parse failure to 1.0 |
| `ctc_oolong` | the parser and the stop rule both read the **last** of three templated markers (`Answer:`/`Label:`/`User:`) seen *so far*; previously the parser matched only `answer:` and the stop rule anchored on the earliest marker. The whole-string stop rule now replays the decode loop prefix by prefix, so a corpus line echoed after the answer (every oolong line carries `User:`) can no longer become the graded span | correct answers that were scored 0 now score correctly |
| `ctc_grouping` | `pairwise_metrics` scored an all-singleton gold partition 0 even for an exact match | 48/500 r2k rows; the perfect-answer ceiling moves 0.904 → 1.0 |
| `ctc_obliq` | decode budget 64 → 512 tokens | 27/126 r32k rows were truncated; the ceiling moves 0.953 → 1.0 |
| `ctc_contradiction` r64k–r1m | gold pairs are sorted low-high at load. The published rungs from r64k up store most pairs high-low (r64k: 744 of 1,500 pairs, on 435 of 500 examples), and the vendored scorer raises on an unsorted gold pair instead of scoring it | those examples were left unscored, and the runner recorded 0 for them; they are now scored. The published grid stops at 32k, so the old numbers for these rungs are internal runs, not grid numbers. r2k–r32k have no high-low pairs and are unchanged |

Everything else is unchanged in code — but **parity with the reference grid has not been measured
for any row.** No per-row comparison of one model's scores through this harness against the
reference grid's numbers exists yet, for the 15 rows whose graders did not change any more than
for the seven that did. Until that run happens (tracked in
[#443](https://github.com/allenai/olmo-eval/issues/443)),
treat numbers from this harness as internally consistent and comparable to each other, not as
reproductions of the published grid.

## Design and provenance

- Prompt templates, parsers, metrics, gold-index conventions and stop rules are **vendored
  byte-faithful** under `_vendor/` from the `ctc` package (AI2 OLMo-core branch `prasann/ctc`, at
  commit **`40a5c60d143b427b8c2cd276fa5009b809be6be3`**), where they are golden-fixture-tested
  against the implementation that produced the suite's published numbers. That commit is also
  `ctc_suite.UPSTREAM_COMMIT` and a field on every CTC scorer, so a re-vendor changes the task hash;
  a test checks the three places that name it agree.
  Only the subtrees this harness reads are vendored; `_vendor/MANIFEST.md` lists exactly what was
  taken, the one deliberate omission, and the re-vendoring steps. **Fix upstream and re-vendor; do
  not edit `_vendor/`** (it is ruff-excluded to stay diffable, but `ty check src/` does cover it).
  This rule was broken once: the stop-rule work in `69729fad` was written straight into the
  vendored copy, so the copy silently ran ahead of the code it is supposed to mirror and a routine
  re-vendor would have reverted it. Those changes are upstream now.
- Scoring mirrors the reference runner call-for-call: stop-rule cleanup →
  `spec.parse(text, n_docs)` → `spec.score(parsed, gold)` with the spec's declared gold field.
- Gold conventions are pinned by test: the pair family stores 1-based indices, the retrieval
  family 0-based, and answering with the wrong base scores zero silently — the class of bug the
  vendoring exists to prevent (`tests/evals/tasks/test_ctc_suite.py`).
- Data: `PrasannSinghal/ctc-suite-eval` (public HF; parquet, one config per task, one split per
  rung), **pinned at revision `04ab8600…`** (`ctc_suite.HF_REVISION`). The pin is serialized into
  the task config, so a dataset rebuild — and this one has been rebuilt, for the xabsence
  exact-copy build and the rerank repricing — produces a new task hash instead of quietly changing
  what an already-stored number meant. Gold answers included: exclude from pretraining corpora.
  `CTC_SUITE_DATA_ROOT=/path` substitutes a local `<subset>/rung_<tokens>.jsonl` tree; the local
  path *replaces* the config's data source, so an unvalidated local ladder also gets its own task
  hash rather than being filed under the published data's. Two metadata fields differ in kind:
  `absence_gutenberg` carries a builder-side `meta` serialized to a JSON string (schema stability
  across splits), which grading never reads; `oolong` carries a structured `_meta` whose
  `gold_list` and `answer_type` *are* the gold, and the scorer reads them.
- The namespace is personal for now. If the dataset moves under `allenai/`, that is a new
  `HF_DATASET` and a new `HF_REVISION`, and therefore new task hashes — better done before a large
  grid is persisted against it.
- Data generation, ladder builders, and per-rung realized-token measurements live in the OLMo-core
  working repo (`debug/ctc_1m_ladders/REPORT.md` is the build provenance record).
