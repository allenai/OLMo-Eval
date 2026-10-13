# BFCL (Berkeley Function Calling Leaderboard)

The BFCL v3 single-turn categories.

A prediction is graded by BFCL's own checker: the predicted calls are compared
against a list of accepted values per parameter, so an answer is right when it
names the right function and supplies values the benchmark accepts — not when
it matches one reference string.

Paper: <https://arxiv.org/abs/2502.17858> ·
Dataset: `gorilla-llm/Berkeley-Function-Calling-Leaderboard`

## Three prompting regimes

Every category is registered in three regimes, which differ only in how the
functions reach the model and how its calls are read back.

| Spec | Regime | Use for |
|------|--------|---------|
| `bfcl_simple` | Native function calling — functions go out as tool schemas, calls come back as `tool_calls` | An instruction-tuned model behind an OpenAI-compatible endpoint |
| `bfcl_simple:prompt` | BFCL's prompting mode — functions in a system prompt, calls as `[func(arg=value)]` text | Any chat endpoint, with no server-side tool parsing |
| `bfcl_simple:base` | Plain completion with curated exemplars | A pretrained model with no chat template |

### Native function calling

Needs no server flags:

```bash
uv run olmo-eval run -m my-model -t bfcl
```

vLLM only emits `tool_calls` when started with `--enable-auto-tool-choice`, so a
vLLM server the run starts itself gets that flag whenever a task sends tool
schemas. vLLM infers a `--tool-call-parser` from the model name; add
`-o provider.kwargs.tool_call_parser=<name>` to choose one explicitly. Provider
overrides follow `--harness`, not `-t`.

If the server would still answer in plain text, because
`provider.kwargs.enable_auto_tool_choice=false` was passed, the run stops with
an error saying so rather than reporting a score of zero. An external server
given with `base_url` is trusted to have been started with the flag.

A provider that cannot carry tool schemas at all — the in-process `vllm`
provider and `litellm` both ignore them — is refused the same way, naming the
provider and pointing at the other regimes. Only `vllm_server` (the default)
and `mock` accept them.

### Prompting mode

Needs no server flags:

```bash
uv run olmo-eval run -m my-model -t bfcl:prompt
```

### Base models

The `:base` regime lays the task out as one block of text — instruction,
functions, question — that the model continues with its calls. Hand-written
exemplars carry the answer format; `:0shot` through `:5shot` change how many
are shown, and the default is five:

```bash
uv run olmo-eval run -m my-base-model -t bfcl:base
uv run olmo-eval run -m my-base-model -t bfcl_simple:base:2shot
```

They are hand-written rather than sampled from the data, so no exemplar is
drawn from the evaluated set, and they cover the shapes a category can take —
one call, one of several candidate functions, several calls, and declining
when no function fits. The declining exemplar matters: without it a base model
calls something on nearly every instance and the irrelevance categories
measure nothing.

Each language has its own set, since the Java and JavaScript categories expect
calls written in those languages.

The expected answer format is BFCL's `[func(arg=value)]` in every text regime, so
numbers are comparable across models and checkpoints. Because a base model has
never been taught that format, the decoder also accepts the JSON tool-call
shapes models pick up during pretraining — a bare list of
`{"name": ..., "arguments": {...}}` objects, or one wrapped in `<tool_call>`
tags — so the score reflects which function was chosen rather than which
surface form the model reached for.

## Categories and suites

```bash
uv run olmo-eval suite inspect bfcl
```

| Suite | What it reports |
|-------|-----------------|
| `bfcl` | BFCL's single-turn overall, without the executable categories |
| `bfcl:non_live_ast` | BFCL's non-live AST summary |
| `bfcl:non_live` | Non-live AST summary with irrelevance |
| `bfcl:non_live_simple` | Simple AST across Python, Java and JavaScript |
| `bfcl:categories` | Every category reported separately |

Each suite also exists as `:prompt` and `:base` (for example
`bfcl:non_live_ast:base`).

BFCL weights its live summaries by how many instances each category holds,
which a suite average cannot express, so those summaries are tasks that pool
their categories' instances: `bfcl_live_ast` and `bfcl_live`. The per-category
tasks are `bfcl_simple`, `bfcl_multiple`, `bfcl_parallel`,
`bfcl_parallel_multiple`, `bfcl_java`, `bfcl_javascript`, `bfcl_irrelevance`,
and the six `bfcl_live_*` categories.

`bfcl_java` and `bfcl_javascript` read calls written in those languages, and
declare the `tree-sitter-java` and `tree-sitter-javascript` grammars as
task-specific dependencies (see [Task-Specific
Dependencies](../../../../../README.md#task-specific-dependencies)). No other
category needs them.

## The checker

The checker is vendored under `olmo_eval/common/scorers/bfcl/` rather than
depended on: the `bfcl-eval` package pins `numpy==1.26.4` and pulls faiss,
sentence-transformers, anthropic and cohere, which does not coexist with the
vLLM environment here.

Two deliberate departures from the reference, both commented where they
happen:

- An arithmetic argument is folded rather than passed to `eval`, which would
  run whatever a model wrote.
- The v3 prompt is reproduced verbatim, including its wording slips and its
  rendering of function documents as a Python repr under a sentence calling it
  JSON, because changing either moves the scores.

Replaying BFCL's own possible answers back through the checker reproduces
999/1000 non-live Python, 1344/1351 live, 92/100 Java and 49/50 JavaScript.
Every remaining case is a dataset quirk the reference checker fails
identically, such as `simple_363`, whose possible answer names `find_closest`
while its function document names `restaurant_search.find_closest`.

## Multi-turn

The multi-turn categories are a different kind of measurement. An entry is a
conversation against stateful APIs, and a prediction is graded on what its
calls did: after each turn the involved instances must hold the state the
ground truth path leaves them in, and everything the ground truth's calls
returned must also have come back from the model's. Nothing the model said is
read.

Driving a rollout needs the `bfcl_multi_turn` scaffold, so these tasks run
under a harness that carries it:

```bash
uv run olmo-eval run -m my-model --harness bfcl_multi_turn -t bfcl:multi_turn

uv run olmo-eval run -m my-model --harness bfcl_multi_turn -t bfcl:multi_turn:prompt
```

| Task | What it perturbs |
|------|------------------|
| `bfcl_multi_turn_base` | Nothing; decompose a request and carry it out |
| `bfcl_multi_turn_miss_func` | A needed function is withheld, then offered partway through |
| `bfcl_multi_turn_miss_param` | A request omits a parameter, so the model should ask rather than guess |
| `bfcl_multi_turn_long_context` | The same tasks with the state inflated by filler |

`bfcl:multi_turn` averages the four as equals, as the leaderboard does. Each
also exists as `:prompt`. There is no `:base` regime: a pretrained model is not
asked to drive a twenty-step tool-executing rollout.

Within a turn the model is asked for calls, the calls are run, and each result
comes back as a tool message, named after its call when the model was prompted.
The turn ends when the model stops calling. A turn that runs past twenty steps
stops the rollout, and an entry stopped before its last turn scores zero, as
does one whose conversation outgrows the model's context window. The rollout's
own instances only serve to answer the model; the score comes from replaying
its calls against fresh ones. As in the reference implementation, a turn whose
ground truth makes no call is not graded, though its calls still shape the
state the next turn is compared on.

Each prediction's `scorer_results` records whether the entry passed, why it
failed if it did, and whether a turn ran out of steps.

The API classes are vendored under
`olmo_eval/common/scorers/bfcl/multi_turn/api/`, because an instance's
attributes after a turn are what a prediction is compared against.

`multi_turn_composite` is not registered. Its ground truth calls an older
signature than the other categories do and some of its calls are not valid
Python, so it cannot be executed against the classes any version of the
reference implementation ships; it is also absent from the v3 multi-turn
summary.

## What is not implemented

- **`multi_turn_composite`**: see above.
- **Executable and REST** (`exec_*`, `rest`): these grade by running the
  predicted calls against live third-party APIs.

Because the executable categories are missing, `bfcl:non_live` is BFCL's
non-live overall with those terms left out and will not match a published
non-live number, which averages them in. `bfcl:non_live_ast` and the live
summaries are directly comparable.
