# Agent guide for olmo-eval

Operational instructions for coding agents. [CONTRIBUTING.md](CONTRIBUTING.md) has the
complete rules and workflow, [docs/design.md](docs/design.md) the reasoning behind them,
and [README.md](README.md) the user and API reference.

Protect the meaning of stored results: a change to what is measured must be identifiable,
and the configuration, code, data, and environment behind a result must be recoverable.

## Working sequence

1. Read the relevant module and its tests, check the working tree, and check open PRs for
   overlapping changes. Preserve existing user edits.
2. Identify which task specs, suites, or execution paths the change affects. For a port,
   identify the reference config and revision before implementing.
3. Make one focused change, reuse existing components, and add tests.
4. Run focused tests while iterating, then the checks below. For a new eval, follow
   [Validate a new eval in stages](CONTRIBUTING.md#validate-a-new-eval-in-stages).
5. Report what changed, which checks actually ran, and remaining gaps. A submitted Beaker
   job is not a verified result.

Committing and pushing to a feature branch is fine. Never push to `main`, force-push, or
rewrite history on a branch someone else is using, and merge a PR only when your human
asks. Before merging, retarget PRs stacked on its branch (`gh pr edit <n> --base main`):
the branch is deleted on merge, which can close them for good. Beaker runs use a pushed
commit, so finish the local code, tests, and launch command before pushing for a remote
trial.

## Commands

```bash
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run ty check src/
uv run pytest tests/ --ignore=tests/integration -v
```

- Use `uv run` for everything; `uv add` for dependencies, never `pip install`. Don't
  hand-edit `uv.lock` or change it outside dependency PRs.
- CI uses `uv run --frozen --no-group vllm` on Python 3.12 and 3.13.
- Unit tests must not need a GPU, network, or Docker. Use local fixtures and the `mock`
  provider; GPU tests need `--gpu`, Docker-backed tests live in `tests/integration/`.
- For docs-only changes, check links, paths, and documented commands instead.
- Discovery without a model: `olmo-eval tasks -f <name>`, `olmo-eval suites`,
  `olmo-eval suite inspect <suite>`, `olmo-eval task inspect <spec> --request`,
  `olmo-eval run -m mock -t <spec> -o limit=5`.

## Invariants

- **Spec meaning is fixed.** Don't change an existing spec's prompt, data, extraction,
  metric, sampling budget, or a suite's membership or aggregation in place. Add a variant
  or versioned spec. If existing results must change, add an entry to
  [docs/result-compatibility.md](docs/result-compatibility.md).
- **Task hash.** Every setting that changes requests or scores must reach
  `TaskConfig.to_dict()`. A new optional field must be serialized only when set, so
  existing hashes don't move. Test both directions.
- **Pins.** Pin a new dataset's `revision=` and judge models to dated snapshots when the
  task is introduced.
- **Failures aren't scores.** No broad `except` around provider, judge, or scoring calls
  that substitutes a score; no sentinel scores; missing logprobs are not zero; a task
  with no scored responses fails. Diagnostics must reach saved predictions.
- **Per-instance metrics.** Override `Metric.compute_instance()` when the metric differs
  from the scorer value, and return `None` for out-of-scope instances.
- **Parity.** Match the cited reference exactly; keep reference bugs under the parity name
  and put fixes in a new variant.
- **Boundaries.** Tasks own the measurement; harnesses own execution. Tasks don't import
  suites or depend on Beaker. Ai2 infrastructure defaults are fine; credentials,
  personal values (an individual's image, workspace, or secret names), and internal-only
  URLs are not.
- **Upload boundary.** olmo-eval reaches the dashboard only through `src/olmo_eval/upload/`
  over HTTP and never imports `olmo_eval_api` (`dashboard/api`). Upload payloads follow
  `dashboard/contract/ingest-v1.schema.json`; change the contract there first.

## Common mistakes

- `num_fewshot=0` treated as unset; use `is None`.
- A class-level `split` silently overriding a requested split.
- An explicit `max_tokens` larger than the model's context window.
- A new extractor, scorer, or formatter that duplicates `MultipleChoiceScorer`,
  `MCQAChatFormatter`, `LogprobMCAccuracyMetric`, or `evals/extract/`.
- Prompt suffixes concatenated in `process_doc` instead of set on the formatter.
- Vision benchmarks not added to the explicit import list in
  `evals/vision/benchmarks/__init__.py`; verify discovery through the CLI.
- A `SamplingParams` or `LMRequest` change tested against one provider only, or a new
  provider kind missing from `_GPU_PROVIDERS` / `_SEQUENTIAL_ONLY`.
- Optional or CUDA-only packages imported at module level in shared code.
- Mutating a cached harness preset when applying overrides.
- Docstrings with hard-coded counts, or comments describing in-progress work.
- Tests that only call private helpers, or regression tests that pass on the old code.
- `-o` overrides: they attach to the preceding `-t` or `--harness`. Task overrides accept
  `TaskConfig` fields and `SamplingParams` fields such as `max_tokens=128`.

## Scope and handoff

- One behavior change per PR. Keep dependency and lockfile changes, image bumps, prompt
  edits to other tasks, and unrelated refactors out. Report unrelated bugs instead of
  fixing them in the same change. Add fields and hooks only with their first consumer.
- Don't bump the package version; evaluations are identified by commit and configuration.
- PR titles are sentence-case imperatives without a conventional-commit prefix. The
  description states the reference and deviations, affected specs, exact validation
  commands, and Beaker experiment IDs, and doesn't claim checks that didn't run.
- For a requested Beaker trial, verify the pushed revision, model, harness, task limit,
  workspace, budget, cluster, GPU count, `--min-runtime`, `--timeout`, and secrets before
  submitting. Pass credentials through secret mappings, not plaintext `--env`. Afterwards,
  inspect logs, completion status, failure counts, and saved predictions. If credentials
  or infrastructure block a check, state the gap and leave a runnable command.
