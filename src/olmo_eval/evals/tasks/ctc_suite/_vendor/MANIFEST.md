# What is vendored here, and what is not

`_vendor/ctc/` is a copy of the `ctc` package from the AI2 OLMo-core branch `prasann/ctc`, at
commit **`40a5c60d143b427b8c2cd276fa5009b809be6be3`**. The same commit is named by
`ctc_suite.UPSTREAM_COMMIT` and by the suite README; a test checks the three agree, so they move
together or not at all. `UPSTREAM_COMMIT` is also a field on every CTC scorer, so it is serialized
into the task config and a re-vendor changes the task hash.

**Do not edit these files.** Fix upstream and re-vendor. The suite's prompts, parsers, metrics,
gold-index conventions and stop rules are golden-fixture tested upstream against the implementation
that produced the suite's published numbers; an edit made here and not there is an edit nothing
tests, and it silently un-pins this harness from the reference. That had already happened once: the
stop-rule work in `69729fad` (the multi-marker `require_before`, the `bracketed_answer`
suppression, the `outliers` preset, and oolong's multi-marker `parse`) was written into this copy
and never sent upstream, so a later re-vendor would have quietly reverted it. Those changes now
live upstream, which is why this manifest exists.

Every `.py` file below `_vendor/ctc/` is byte-identical to its upstream counterpart at that commit,
with exactly one deliberate omission, recorded in the file itself:

| Omitted | Where | Why |
|---|---|---|
| the `redundancy`, `mathmatch`, `cycle`, `groups4`, `qa`, `grouping_labeled` and `summarization` task packages | `tasks/` and `tasks/__init__.py:TASK_MODULES` | None is in `ROSTER`, so none is reachable from any registered task. `summarization` additionally imported `rouge_score`, which this repository does not depend on, so vendoring it made `ty check` fail on code nothing could run. |

The two other optional dependencies upstream code touches — scikit-learn in
`format/metrics.clustering_extras` and scipy in `format/metrics.ordering_extras` — are kept. They
are reachable (`tasks/_grouping.py` calls `clustering_extras`) and they now load through
`importlib.import_module`, so an install without the extra sees an empty metric dict rather than an
unresolved import in the type check.

## What is taken

Only the subtrees this harness actually calls:

- `format/` — `assemble.py`, `documents.py`, `fingerprint.py`, `metrics.py`, `parsing.py`,
  `prompts.py`, `registry.py`, `rungs.py`. `fingerprint.py` is not called by the harness directly
  but `registry.py` imports it, so it comes along.
- `tasks/` — the 11 packages in `TASK_MODULES` plus the shared factories `_absence.py`,
  `_cycles.py`, `_grouping.py`, `_pairs.py`, `_retrieval.py`. `_grouping.py` is what the harness
  uses to register the plain `grouping` spec, which upstream does not ship as a package.
- `eval/stopping.py` — the stop rules.
- `data/ladders.py` — the pure rung table, read by `contradiction/spec.py`.

Not taken: the data generators, the decode backends, the runner, the training-side modules, and
the upstream test suite.

## Re-vendoring

1. Make the change upstream, with a test, and commit it there.
2. Copy every file listed above from `<olmo-core>/ctc/src/ctc/` over `_vendor/ctc/`.
3. Re-apply the omission in the table (trim `TASK_MODULES`; do not vendor the seven packages).
4. Update the commit in this file, in `ctc_suite.UPSTREAM_COMMIT`, and in the suite README.
5. `ruff` does not lint `_vendor/` (it is `extend-exclude`d so the copy stays diffable against
   upstream), but `ty check src/` does cover it — a type error upstream is a red CI here.
