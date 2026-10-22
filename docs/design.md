# Design

olmo-eval supports base-model evaluation during pretraining, post-training checks, and
tool-augmented or agentic evaluation. The same task definitions serve all three. The
central requirement is that a stored result can be traced back to the data,
configuration, code, and environment that produced it, so a number means the same thing
whether it came from a laptop trial, a Beaker sweep, or a comparison months later.

This page explains the design choices that follow from that requirement.
[CONTRIBUTING.md](../CONTRIBUTING.md) turns them into concrete rules and workflow;
[README.md](../README.md) is the user and API reference.

## How an evaluation is put together

```text
DataSource -> Task.process_doc -> Instance -> Task.format_request -> LMRequest
                                                                    |
                                                       Harness / Provider
                                                                    |
                   saved results <- Metrics <- extraction / scoring / responses
```

A **task** defines the measurement: data, split, few-shot examples, prompts, sampling
parameters, answer extraction, scoring, and aggregation. A **formatter** turns instances
into requests. A **scorer** evaluates outputs; a **metric** aggregates scores and exposes
per-instance values where meaningful. Requests use `CHAT` or `COMPLETION` for generation,
and `LOGLIKELIHOOD` for likelihood-based multiple choice and bits-per-byte evaluation.

A **harness** defines the execution setup: provider, tools, scaffold, sandboxes, auxiliary
providers, concurrency, and failure tolerance. A **provider** adapts requests to a model
backend. A **scaffold** controls a multi-turn interaction.

## Measurement and execution are separate

The task owns what is measured; the harness owns how it is run. This lets one task run
with different models and agent setups, and lets a baseline and a tool-augmented run of
the same task be compared directly. The task should not need to know whether it is running
locally or on Beaker, and the harness should not contain benchmark-specific logic.

Tools, system prompts, and scaffolds can change performance, so the harness is part of
the experimental condition and must be recorded with the result. For judge tasks, the
grading protocol and judge settings belong to the measurement; the harness supplies any
auxiliary provider needed to execute it.

An **external eval** is the deliberate exception. It wraps a benchmark that ships its own
runner, such as tau2 or SciCode, and uses `run-external` and the external-eval result
contract instead of forcing that execution loop into the task pipeline.

## A task spec names a measurement

A **task spec** such as `humaneval:3shot:bpb` selects a registered task and applies
registered variants from left to right. Specs are what people type, what suites list, and
what gets stored beside results. Because a spec is how people refer to a result, its
default measurement is treated as fixed once results exist under it. A change to the
prompt, extraction, data, metric, or sampling budget produces a new variant or versioned
name instead of silently redefining the old one.

Named variants are preferred to ad-hoc runtime overrides. `:3shot`, `:bpb`, or `:cot` are
discoverable with `olmo-eval tasks` and reproducible from the name alone. A runtime
override such as `-o limit=5` is useful for trials, but a result produced with it describes
that limited run and needs its overrides reported with it.

The spec is a name; the **resolved configuration** records the actual task, including
overrides. `TaskConfig.to_dict()` supplies its stored representation and the task hash.
The hash does not fingerprint the source code or capture the model and harness
configuration, so reproducing an experiment also requires the code revision, the model,
checkpoint, and tokenizer, the harness, data revisions, and the dependency environment.
Stochastic generation and hosted services can still vary between runs; the goal is that
the procedure is recoverable.

**Package versions.** Evaluations are identified by their source commit and resolved
configuration. Package version numbers do not currently provide a reliable record of
evaluation compatibility, and there is no automated publication workflow; release
ownership and policy remain to be defined. Changes that affect existing results are
recorded in the [result compatibility log](result-compatibility.md) instead.

## Composition over configuration

The core vocabulary is small: task, suite, formatter, scorer, metric, harness, provider.
New benchmarks are assembled from existing parts where possible. When a part almost fits,
extend it so the next benchmark can reuse it, and keep benchmark-specific quirks in the
benchmark's own module. Hooks for hypothetical future tasks are avoided; a shared
interface grows when a real consumer needs it.

Registries for tasks, variants, suites, harness presets, tools, external evals, and model
presets are populated by importing modules, so adding a benchmark is mostly adding a
module rather than editing a central list.

A **suite** lists tasks or nested suites and states its aggregation explicitly, because
equal task weighting and instance weighting answer different questions. A suite that
cannot compute the aggregate it promises reports nothing instead of a different number
under the same name.

## Parity before improvement

Many tasks port oe-eval, a paper's release, or an official repository, and downstream
comparisons depend on matching them. A port first matches the reference: split, few-shot
selection, prompt text, stop sequences, token budget, extraction, normalization, and
aggregation. Where the reference has a bug, the parity name keeps the reference behavior
and the correction gets a new, explicitly named variant. A similar aggregate is not
evidence of parity, since two incompatible implementations can produce close headline
numbers.

## Failures are not scores

A model's wrong or unparseable answer may legitimately score zero under a benchmark's
rules. A provider outage, an exhausted judge retry, or a failed scoring process must
remain an explicit failure. An eval that quietly scores a crashed instance as zero
produces a believable wrong number, which is worse than a failed run.

When a scorer cannot reach a verdict for an output, the output records the error and gets
no score from that scorer. An instance the scorer could not score at all is left out of
that scorer's metrics, and one it scored from only some of its outputs is scored from
those. Instances left out this way
and instances that produced no output share one failure budget, `max_hard_failure_rate`
(5% by default). A task within it publishes its metrics with the counts beside them; a
task over it, or with nothing scored, fails and publishes none. Leaving failures out can
bias a score, because they tend to cluster on unusual outputs, and the budget bounds that
bias while the published counts make it visible. Requests and predictions are saved by
default so prompts, extracted answers, per-instance scores and scoring errors can be
checked before an aggregate is trusted.
Progress reporting and telemetry are the exception: they must never terminate an
evaluation, so their errors are caught and logged.

## Reproducible environments

Dependencies are locked in `uv.lock` and installed with `uv sync --frozen`. Version pins in
`pyproject.toml` carry a comment naming the breakage they prevent, so the next person knows
when a pin can be lifted. GPU-only dependencies are gated by platform markers so the
repository installs and unit-tests on macOS. Beaker jobs run from a git checkout of a
pushed commit inside a recorded image.

olmo-eval is built for Ai2's model development, and Beaker and Ai2 infrastructure are its
supported way to run remotely. The source is public so others can use it, but it is not
designed around other deployments. Credentials, personal values, and internal-only URLs
stay out of source, and deployment values stay configurable where that is cheap.
