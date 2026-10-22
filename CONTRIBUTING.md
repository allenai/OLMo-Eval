# Contributing to olmo-eval

This guide covers setup, the rules a change must follow, how to validate a new eval, and
what a reviewable PR looks like. [docs/design.md](docs/design.md) explains why the rules
exist; [README.md](README.md) is the user and API reference. Coding agents read
[AGENTS.md](AGENTS.md), which condenses these rules and links back here.

## Setup

Follow the [README quick start](README.md#quick-start), then from the repository root:

```bash
uv sync --frozen   # install from the checked-in lockfile
make setup         # install git hooks
```

vLLM and other CUDA dependencies have Linux platform markers, so the repository installs
and unit-tests on macOS.

## Checks

Run these before opening a PR that changes code:

```bash
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run ty check src/
uv run pytest tests/ --ignore=tests/integration -v
```

- CI runs the same commands as `uv run --frozen --no-group vllm ...` on Python 3.12 and
  3.13. Use those flags to reproduce a CI-only failure.
- `./scripts/fix.sh` formats and applies lint fixes. `./scripts/verify.sh` syncs
  dependencies and runs everything, including Docker-backed integration tests
  (`--no-docker` skips them; `--gpu` enables GPU tests).
- Unit tests must run without a GPU, network access, or Docker. GPU tests use the `gpu`
  marker and run with `--gpu`; Docker-backed tests live in `tests/integration/`.
- For a docs-only change, check links, paths, and every command you documented.

## Find your way around

```bash
uv run olmo-eval tasks -f gsm8k                # tasks and their variants
uv run olmo-eval suites                        # registered suites
uv run olmo-eval suite inspect olmobase:code   # expand a suite
uv run olmo-eval harnesses                     # also: scaffolds, models, external-evals
uv run olmo-eval run -m mock -t gsm8k --dry-run
```

Follow [`gsm8k.py`](src/olmo_eval/evals/tasks/gsm8k.py) through `process_doc`,
`format_request`, and `extract_answer`. Before writing a new task, read a nearby task that
uses the same request and scoring types, and its tests. Existing tasks include legacy
conventions; when they disagree with this guide, follow this guide. The
[task authoring reference](README.md#adding-new-tasks) has the class and variant APIs.

### Where changes belong

Paths are relative to `src/olmo_eval/` unless stated otherwise.

| Change | Start here |
| --- | --- |
| Text benchmark or variant | `evals/tasks/`; base types and registry in `evals/tasks/common/` |
| Frozen subsets, exemplars, long constant tables | `evals/tasks/constants/` |
| Group of benchmarks and its aggregation | `evals/suites/` |
| Shared answer extraction | `evals/extract/` |
| Reusable formatting, scoring, metrics | `common/formatters.py`, `common/scorers/`, `common/metrics/` |
| Multimodal benchmark | `evals/vision/benchmarks/`, with shared tasks, data, and scoring under `evals/vision/` |
| Benchmark with its own runner | `evals/external/benchmarks/` |
| Data access | `data/` |
| Model backend | `inference/providers/`; shared request/response types in `common/types/` |
| Tools, scaffolds, sandboxes, execution presets | `harness/` |
| Scheduling, aggregation, saved artifacts | `runners/` |
| CLI and remote submission | `cli/`, `launch/` |
| Results upload to the dashboard | `upload/`; the dashboard itself is in repository-root `dashboard/` |
| Tests and maintenance scripts | Repository-root `tests/` and `scripts/` |

Registration happens on import. Text task modules, suite modules, and external benchmark
packages are discovered automatically. Vision benchmarks have an explicit import list in
[`evals/vision/benchmarks/__init__.py`](src/olmo_eval/evals/vision/benchmarks/__init__.py).
Check that the CLI lists a new benchmark; importing it directly in a test can hide a
missing registration.

## Rules for changes

### Task identity

A task's serialized configuration (`TaskConfig.to_dict()`) is what gets stored with its
results and hashed to identify it.

- Every task setting that changes requests or scores must be part of that serialized
  configuration, including prompt, formatter, scorer, few-shot, and judge settings.
  Otherwise two different runs share a hash and artifact name. Don't read
  output-affecting settings from environment variables at scoring time.
- A new optional field must leave existing hashes unchanged when unused, so serialize it
  only when set. Test both that changing the field changes the hash and that an unrelated
  existing task keeps its hash.
- Pin a new dataset's `revision=` when the task is introduced, and pin judge models to
  dated snapshots. Adding a pin later changes the task's identity after results exist.
- Variant names describe the measurement or regime (`mc`, `rc`, `bpb`, `gen`, `cot`,
  `judge`, `3shot`), not a model. Task specs and suite names must not collide.
- Don't change an existing spec's measurement in place; see
  [Changing or retiring an existing spec](#changing-or-retiring-an-existing-spec).

### Reference parity

- Identify the reference (oe-eval config, paper, or repository revision) before coding,
  and cite it in the task module and the PR.
- Match the split, few-shot count and selection, prompt text and suffixes, stop sequences,
  token budget, extraction, normalization, and aggregation. If settings come from a
  differently named reference configuration, name the variant after that regime.
- Keep reference bugs under the parity name and put corrections in a new variant.
- Put prompt suffixes in formatter configuration where the formatter supports it, not
  string concatenation in `process_doc`.
- Compare requests and per-instance scores as well as aggregates, and report our result
  beside the reference under the same model and regime where possible. Say precisely
  whether prompt, code, or numerical parity was checked. A mock run or a similar score
  does not establish numerical parity.
- Suites declare their aggregation. oe-eval often reports instance-weighted averages,
  which is `WEIGHTED_AVERAGE`.

### Failures and scoring

- Don't catch broad exceptions around provider, judge, or scoring calls and substitute a
  score. Exhausted retries and judge parse failures need explicit failure accounting and
  the raw response preserved. Catch narrowly when the recovery is defined.
- A scorer returns 0 for a wrong or unparseable answer and raises when it cannot decide.
  The runner records the error on the output, leaves it unscored, and counts instances
  with no scored output against the task's failure budget (`max_hard_failure_rate`).
  Raise `ScoringIncompleteError` for a judge or grader that gave no verdict, and let
  sandbox failures raise `SandboxInfrastructureError`, so the counts say why.
- Missing logprobs are never logprob zero. Don't use sentinel scores such as `-1` or NaN
  without an explicit error. A task with no scored responses must fail the run.
- Diagnostics such as format correctness, judge results, and raw output must reach saved
  predictions. Arbitrary instance metadata is not copied automatically, so check the
  predictions file, not just the logs.
- A metric whose per-instance value differs from its scorer's value must say so
  explicitly, and instances outside the metric's scope should have no value. Check saved
  per-instance values as well as the aggregate.
- Bounded quality metrics such as accuracy and F1 use a 0-1 scale. Bits per byte and
  perplexity keep their native units.
- Before writing a scorer, formatter, metric, or extractor, look for an existing one in
  `common/` and `evals/extract/`. Shared extraction belongs in `evals/extract/`.
- Test extractors against negative numbers, thousands separators, malformed answers, and
  answers inside `<think>` traces. Treat `num_fewshot=0` as a real value (`is None` means
  unset), keep explicit token budgets within the model's context, and don't let a
  class-level `split` overwrite a requested one.

### Shared interfaces and environment

- `SamplingParams` and `LMRequest` are shared by every provider. A change must work for
  `vllm`, `vllm_server`, `hf`, `litellm`, `olmo_core`, and the vision providers, or state
  and validate a narrower scope.
- A new provider kind must be added everywhere existing kinds are enumerated; search for
  an existing kind's name.
- Don't import CUDA-only or optional packages at module import time in shared code. Keep
  heavy or side-effectful imports, such as downloads, inside the code that needs them.
- Deep-copy presets before applying overrides. Keep multiprocessing payloads small and
  picklable; pass paths instead of decoded images.
- Tasks don't import suite modules or depend on Beaker, and runners work without Beaker.
- olmo-eval reaches the dashboard only through `upload/` over HTTP and never imports
  `olmo_eval_api` (the separate package in `dashboard/api`). Upload payloads follow
  `dashboard/contract/ingest-v1.schema.json`; change the contract there first.
- Beaker and Ai2 infrastructure are the supported way to run remotely, and Ai2 defaults
  are fine. Keep credentials, personal values (an individual's image, workspace, or
  secret names), and internal-only URLs out of source, and keep deployment values
  configurable where that is cheap.

### Code and tests

- Keep interfaces stable and general; don't couple shared methods to one benchmark's
  config fields. Add fields and hooks together with their first consumer.
- Docstrings describe stable behavior, not current implementation details or counts of
  tasks or examples. Comments explain why and stay short. Project status and PR context
  belong in the PR, not in code.
- Type dictionaries with a fixed schema that cross interfaces (`TypedDict`), use `Literal`
  or validation for fixed string choices, and tuples for immutable collections. Don't add
  `ty` ignores.
- Add unit tests for new code, in pytest style rather than `unittest.TestCase`. A
  regression test should fail against the old code. Exercise public paths such as
  `get_task(spec)` and the CLI, not only private helpers.
- For a new task, test registration and every variant, representative documents,
  formatted requests, extraction edge cases, and any new configuration or hash behavior,
  using local fixtures and the mock provider.
- Scripts under `scripts/` that generate checked-in data need tests in `tests/scripts/`.

## Validate a new eval in stages

### 1. Inspect locally and exercise the pipeline

Replace `gsm8k` with the new task spec once it is registered:

```bash
# Loads task data and shows requests before inference.
uv run olmo-eval task inspect gsm8k --request -n 2

# Prints the run plan without loading a model.
uv run olmo-eval run -m mock -t gsm8k -o limit=5 --dry-run

# Exercises data loading, formatting, mock inference, scoring, and artifact writing.
uv run olmo-eval run -m mock -t gsm8k -o limit=5 \
    --inspect-request --inspect-response --output-dir /tmp/olmo-eval-gsm8k-smoke
```

Task inspection and mock runs may download data. `mock` replaces the model provider, but
judges, tools, and code-scoring sandboxes may still need credentials or services. A mock
score says nothing about model quality or parity.

Inspect the output directory's `requests/`, `predictions/`, and `metrics.json`: prompt
text, gold labels, extracted answers, per-instance metrics, and failure counts. Use a
fresh directory for each trial so artifacts stay attributable.

### 2. Run a small real-model trial

GPU trials use `olmo-eval beaker launch`, which submits through gantry. Gantry clones the
repository at a pushed commit, so uncommitted or unpushed changes are not part of a remote
run. Commit and push the branch before submitting.

Confirm Beaker access, a workspace and budget, a cluster where the workspace can schedule,
and any model, data, judge, or tool credentials. Choose a GPU count that fits the model,
set `--min-runtime` to cover setup and the expected run, and set `--timeout` to bound it.
Preview a limited trial first, replacing every angle-bracketed value:

```bash
uv run olmo-eval beaker launch \
    --name '<trial-name>' \
    --model '<model-preset-or-checkpoint>' \
    --harness '<harness-preset>' \
    --task '<task-spec>' -o limit=10 \
    --cluster '<cluster>' --workspace '<workspace>' --budget '<budget>' \
    --gpus '<gpu-count>' --min-runtime '<duration>' --timeout '<duration>' \
    --inspect-request --inspect-response --no-follow --dry-run
```

Check the resolved task, provider, image, resources, and command. The preview can require
Beaker authentication and set up credentials, but does not submit the evaluation. Remove
`--dry-run` to submit. The [Beaker reference](README.md#launching-on-beaker) covers YAML
configs, provider overrides, secrets, and multi-model runs. For an external eval, use
`--external-eval`/`-E` with `--eval-arg`/`-A` instead of `--task`; external evals don't
share the task `limit` override.

### 3. Inspect the trial before expanding it

Keep the experiment ID and the submitted command. Follow logs with
`uv run olmo-eval beaker watch -e '<experiment-id>'`, or use the Beaker CLI:

```bash
beaker experiment logs '<experiment-id>' --tail 200
beaker experiment tasks '<experiment-id>' --format=json
beaker experiment results '<experiment-id>' -o '/tmp/olmo-eval-<trial-name>'
```

A successful launch only confirms submission. Check completion status, scored and failed
counts, and real requests and predictions for prompt mismatches, truncation, extraction
errors, and judge failures. Diagnose a failure before relaunching. Don't raise failure
tolerance to make a trial pass. Expand to the full dataset and compare with the reference
only after the trial looks right, and keep trial results distinguishable from full ones.

## Changing or retiring an existing spec

A task spec's default measurement is fixed once results exist under it.

- A change to prompt, extraction, data, metric, sampling budget, suite membership, or
  suite aggregation gets a new variant or versioned name (`:v2`; `omega_500:hillclimb` is a
  precedent) instead of an in-place edit.
- If a change to existing results is unavoidable, such as a fix to shared extraction or a
  bug that made the old behavior meaningless, add an entry to the
  [result compatibility log](docs/result-compatibility.md) in the same PR, naming the
  affected specs, whether hashes change, and which results remain comparable.
- Audit callers when changing shared helpers; one extractor can affect many specs.
- To retire a spec, remove it or make it fail with a message naming its replacement.
  Never point an old name at a different measurement, since stored results under that
  name would then appear to describe it. Record the retirement in the compatibility log.

## Dependencies and images

- Don't hand-edit `uv.lock`. Use `uv add` or `uv lock`, and only in PRs about
  dependencies; keep incidental lockfile changes out of other PRs.
- Explain each pin in `pyproject.toml` with a comment naming the breakage it prevents.
- A core upgrade (torch, vLLM, transformers, the OpenAI client) needs a representative
  real-model canary. The usual sequence:
  1. Change the dependency and regenerate `uv.lock`.
  2. If the runtime image changes, build and push a candidate with
     `./scripts/build_image.sh` and `./scripts/beaker/push_beaker_image.sh` (see the
     [Docker reference](README.md#docker-image-management)).
  3. Run the canary from a pushed commit with the candidate image passed explicitly
     (`--image`), and compare against a run on the current default.
  4. Promote the image by updating `BEAKER_DEFAULT_IMAGE` or `BEAKER_SANDBOX_IMAGE` in
     `common/constants/infrastructure.py`, in the same PR or a follow-up.

## Versions and releases

Evaluations are identified by their source commit and resolved configuration. Package
version numbers do not currently provide a reliable record of evaluation compatibility,
and there is no automated publication workflow; release ownership and policy remain to be
defined. Don't bump the package version as part of an eval change; record changes that
affect existing results in the [compatibility log](docs/result-compatibility.md).

## Pull requests

- Check open PRs for overlapping work before starting.
- One behavior change per PR: a task port with its tests and any suite entry it needs, or
  a fix with its regression test. Keep unrelated refactors, prompt edits to other tasks,
  dependency upgrades, and image changes in separate PRs, and report unrelated bugs
  separately. Split a large feature into a numbered series of independently working PRs.
- PRs are squash-merged, so the title becomes the commit on `main`. Use a sentence-case
  imperative title without a conventional-commit prefix.
- The description explains the measurement, the reference and any intentional deviations,
  which existing specs are affected, and what was actually verified: exact commands, code
  revision, Beaker experiment IDs, and reference comparisons as applicable. An unrun check
  belongs under remaining work. List gaps and link follow-up issues, and update the
  description if the code changes during review.
- Review owners are listed in [.github/CODEOWNERS](.github/CODEOWNERS).
- When a workflow changes, update this guide or the README in the same PR.
