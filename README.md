# olmo-eval

[![CI](https://github.com/allenai/olmo-eval/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/allenai/olmo-eval/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/allenai/olmo-eval/blob/main/LICENSE)

## Overview
This project provides a unified workbench for evaluating language models throughout the model development loop.

Features:

- Registry of benchmark tasks and composable suites, with named variants for few-shot settings, formatting, and scoring (e.g. humaneval:3shot:bpb).
- Support for inference via vLLM, LiteLLM for commercial APIs, and a mock provider for dry runs and debugging.
- Harness abstraction that separates execution policy from task definition, so any task can be run baseline or tool-augmented without modification.
- Multi-turn agentic evaluation with tool calling, scaffolds, and sandboxed environments via Docker, Podman, or Modal.
- LLM-as-judge scoring with auxiliary providers, including locally served judge models.
- Aggregate and instance-level prediction storage.
- Inspection tooling for viewing instances, formatted prompts, token arrays, and model responses.

To contribute, start with [CONTRIBUTING.md](CONTRIBUTING.md). The reasoning behind the
design is in [docs/design.md](docs/design.md), and [AGENTS.md](AGENTS.md) holds the
instructions for coding agents.

## Quick Start

This project uses [uv](https://docs.astral.sh/uv/) with a checked-in `uv.lock`
for reproducible builds. To get started, sync the repo with `uv`, browse the
available tasks and suites, and preview a run with the built-in `mock` provider.

### Run Your First Eval

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Python 3.12 if your machine does not already have it
uv python install 3.12

# Install dependencies + the package (editable) from the lockfile.
# The default groups (`dev` + `vllm`) are installed automatically, which
# pulls in storage, beaker, hf, and the vLLM inference provider. vLLM
# deps are marked Linux-only via PEP 508 markers, so this works on macOS
# too — no extra flags needed.
uv sync --frozen

# Install pre-commit hooks
make setup

# To update the lockfile after changing pyproject.toml
uv lock

# Add an optional extra on top of the defaults (e.g. agents, litellm)
uv sync --frozen --extra agents

# `openhands` conflicts with vllm — opt out of the vllm group when using it
uv sync --frozen --no-group vllm --extra openhands

# Browse a few suites
uv run olmo-eval suite inspect mmlu
uv run olmo-eval suite inspect gpqa
uv run olmo-eval suite inspect olmobase:code

# Preview a run without loading a model
uv run olmo-eval run -m mock -t gsm8k --dry-run

# Preview another run with a different task spec
uv run olmo-eval run -m mock -t humaneval:3shot:bpb --dry-run
```

## Key Concepts

The evaluation framework is built around these core abstractions:

| Abstraction | Description |
|-------------|-------------|
| **Task** | Benchmark specification defining dataset slice, request construction, and scoring logic |
| **Suite** | Benchmark collection that composes tasks and/or nested suites and defines result aggregation |
| **Harness** | Execution runtime around the inference provider, tools, scaffolds, and runtime behavior |
| **Formatter** | Prompt renderer from an instance and few-shot context to an LM request |
| **Scorer** | Per-example evaluator from model output to raw score or judgment |
| **Metric** | Dataset-level aggregator over per-example scores |

### Tasks

Tasks define how to load data, format prompts, and score outputs. Register with `@register`:

```python
from olmo_eval.evals.tasks.common import Task, register
from olmo_eval.data import DataSource

@register("my_task")
class MyTask(Task):
    # DataSource specifies path, subset (optional), and split
    data_source = DataSource(path="cais/mmlu", subset="abstract_algebra", split="test")
    ...
```

**Variants** can also act as named evaluation presets (for example, few-shot settings):

```python
from olmo_eval.evals.tasks.common import register_variant

register_variant("my_task", "3shot", num_fewshot=3, fewshot_seed=42)
# Built-in example: uv run olmo-eval run -m llama3.1-8b -t humaneval:3shot:bpb
```

**Runtime Dependencies** allow tasks to specify packages installed at job startup:

```python
@register("code_eval")
class CodeEvalTask(Task):
    data_source = DataSource(path="my-org/code-dataset", split="test")
    dependencies = ["code-sandbox==1.0", "git+https://github.com/user/repo@v2.0"]
    ...
```

### Suites

Suites group multiple tasks for batch evaluation:

```python
from olmo_eval.evals.suites import Suite, register

register(Suite(
    name="my_suite",
    tasks=("task_a:3shot", "task_b:3shot", "task_c:3shot"),
))
```

#### Aggregation

Suites support different strategies for combining task results:

| Strategy | Description |
|----------|-------------|
| `AVERAGE` | Simple average of all task scores (default) |
| `WEIGHTED_AVERAGE` | Average of all task scores, each weighted by the task's instance count |
| `AVERAGE_OF_AVERAGES` | Average over child suite averages (equal weight per child) |
| `DISPLAY_ONLY` | Display child results without computing suite average |
| `NONE` | No aggregation - just collect individual task results |

**Average of Averages Example:**

```python
from olmo_eval.evals.suites import Suite, AggregationStrategy, register

# Nested suite with 3 tasks
multilingual_code = Suite(
    name="multilingual_code",
    tasks=("mbpp_python", "mbpp_java", "mbpp_rust"),
    aggregation=AggregationStrategy.AVERAGE,
)

# Parent suite using average of averages
register(Suite(
    name="code_eval",
    tasks=(
        "humaneval",        # Single task (score: 0.80)
        multilingual_code,  # Nested suite with 3 tasks (scores: 0.40, 0.50, 0.60)
    ),
    aggregation=AggregationStrategy.AVERAGE_OF_AVERAGES,
))

# Results:
# - humaneval: 0.80
# - multilingual_code average: (0.40 + 0.50 + 0.60) / 3 = 0.50
#
# AVERAGE_OF_AVERAGES: (0.80 + 0.50) / 2 = 0.65
# vs AVERAGE:          (0.80 + 0.40 + 0.50 + 0.60) / 4 = 0.575
```

Note: `AVERAGE_OF_AVERAGES` gives each child equal weight regardless of how many tasks it contains.

**Weighted Average Example:**

```python
register(Suite(
    name="mmlu_pro",
    tasks=("mmlu_pro_math", "mmlu_pro_history"),  # 1351 and 381 questions
    aggregation=AggregationStrategy.WEIGHTED_AVERAGE,
))

# With scores of 0.30 (math) and 0.45 (history):
# WEIGHTED_AVERAGE: (0.30 * 1351 + 0.45 * 381) / 1732 = 0.333
# vs AVERAGE:       (0.30 + 0.45) / 2 = 0.375
```

`WEIGHTED_AVERAGE` matches the instance-weighted "micro" average that oe-eval reports for some suites. It weights each task by the number of instances that task scored.

A weighted suite reports its weighted mean or no score at all. If a contributing task has no instance count, the suite aggregate is omitted and the runner logs which tasks were missing, rather than publishing an unweighted mean under the same suite name. Instance counts are required when a result is stored, so this only affects results written before that check existed.

### Formatters

Formatters convert instances into LM requests. See `olmo_eval.common.formatters` for available options.

```python
from olmo_eval.common.formatters import MultipleChoiceFormatter, ChatFormatter

# Multiple choice with logprob scoring
formatter = MultipleChoiceFormatter(template="Q: {question}\n\nA:")

# Chat-based formatting
formatter = ChatFormatter(system_prompt="You are a helpful assistant.")
```

### Scorers

Scorers compute a score for each instance/output pair. See `olmo_eval.common.scorers` for available options.

```python
from olmo_eval.common.scorers import ExactMatchScorer, MultipleChoiceScorer

# Exact string match
scorer = ExactMatchScorer()

# Multiple choice comparison
scorer = MultipleChoiceScorer()
```

### Metrics

Metrics aggregate scores across responses. See `olmo_eval.common.metrics` for available options.

```python
from olmo_eval.common.metrics import AccuracyMetric, F1Metric
from olmo_eval.common.scorers import ExactMatchScorer, F1Scorer

# Mean accuracy
metric = AccuracyMetric(scorer=ExactMatchScorer)

# Mean F1 score
metric = F1Metric(scorer=F1Scorer)
```

### Model Presets

Pre-configured model settings in `olmo_eval/common/constants/models.py`:

```python
from olmo_eval.common.constants import get_model_presets

# Returns dict of preset name -> ModelConfig
presets = get_model_presets()
# {
#     "llama3.1-8b": ModelConfig(model="meta-llama/Meta-Llama-3.1-8B"),
#     "olmo-2-7b": ModelConfig(model="allenai/OLMo-2-1124-7B"),
#     ...
# }
```

### Harness

A **Harness** is the runtime orchestration layer for an evaluation run. It combines the primary inference provider with execution policy such as system prompts, tools, auxiliary providers, sandboxing, metrics collection, and an optional scaffold for multi-turn control. This lets the same task run in plain, tool-using, or scaffolded modes without changing the task definition.

**Key concept**: Any task can be run with or without tools—that's determined by the Harness configuration, not the task definition. This allows comparing baseline vs tool-augmented performance on the same task.

#### Using Harness via CLI

```bash
# Run task without tools or a scaffold (baseline)
uv run olmo-eval run -m llama3.1-8b -t simpleqa:judge

# Run task with search tools via harness preset
uv run olmo-eval run -m llama3.1-8b -t simpleqa:judge --harness dr_tulu

# Use a custom harness config file
uv run olmo-eval run -m llama3.1-8b -t simpleqa:judge --harness-config ./my_harness.yaml
```

#### HarnessConfig

Configuration for a harness:

```python
from olmo_eval.harness import HarnessConfig, ProviderConfig, get_harness_preset
from olmo_eval.harness.tools.search import (
    semantic_scholar_search,
    serper_web_search,
    serper_fetch_page,
)

# Get a preset
config = get_harness_preset("dr_tulu")

# Or create custom config with tools
config = HarnessConfig(
    name="my_harness",
    provider=ProviderConfig(model="gpt-4o", kind="litellm"),
    tools=(semantic_scholar_search, serper_web_search, serper_fetch_page),
    system_prompt="You are a helpful assistant with search tools.",
    max_turns=10,
    max_concurrency=8,
    scaffold="openai_agents",
    required_secrets=("S2_API_KEY", "SERPER_API_KEY"),
)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | Required | Harness identifier |
| `provider` | `ProviderConfig` | `ProviderConfig()` | Model provider configuration |
| `tools` | `tuple[Tool \| str, ...]` | `()` | Tool instances or registered tool names |
| `system_prompt` | `str \| None` | `None` | System prompt to inject |
| `tool_choice` | `str` | `"auto"` | Tool selection mode (`auto`, `none`, `required`) |
| `scaffold` | `str \| None` | `None` | Execution scaffold (e.g., `openai_agents`) |
| `max_turns` | `int \| None` | `None` | Max turns for multi-turn execution |
| `max_concurrency` | `int \| None` | `None` | Concurrent executions |
| `scoring_concurrency` | `int \| None` | `None` | Max concurrent scoring operations |
| `sandboxes` | `tuple[SandboxConfig, ...]` | `()` | Sandbox configurations for isolated tool execution |
| `scaffold_kwargs` | `dict[str, Any]` | `{}` | Scaffold-specific options (e.g., `enable_compaction`) |
| `metrics` | `MetricsConfig \| None` | `None` | Inference metrics collection config |
| `batching` | `BatchConfig \| None` | `None` | Batching strategy configuration |
| `max_hard_failure_rate` | `float \| None` | `None` | The failure budget: fraction of a task's instances allowed to produce no output or have no scored output before the task is marked failed, its metrics are withheld and the run exits non-zero (default `0.05`). `1.0` never fails a run on partial failures; a task with nothing scored still fails |
| `required_secrets` | `tuple[str, ...]` | `()` | Required environment variables |

#### Scaffolds

Scaffolds define how the Harness executes multi-turn requests with tool calling. A scaffold handles the agentic loop: calling the model, executing tools, and feeding results back.

```bash
# List available scaffolds
uv run olmo-eval scaffolds
```

**When to use a scaffold:**
- For multi-turn execution with `harness.run()`, you must specify a scaffold
- For single-turn generation with `harness.generate()`, no scaffold is needed

```python
# Multi-turn execution requires a scaffold
config = HarnessConfig(
    name="my_agent",
    provider=ProviderConfig(model="gpt-4o", kind="litellm"),
    tools=(semantic_scholar_search, serper_web_search),
    scaffold="openai_agents",  # Required for run()
)
harness = Harness(config)
result = await harness.run(request)  # Uses the scaffold

# Single-turn generation works without a scaffold
config = HarnessConfig(
    name="simple",
    provider=ProviderConfig(model="gpt-4o", kind="litellm"),
)
harness = Harness(config)
outputs = harness.generate(requests)  # No scaffold needed
```

#### Inference Metrics

Harness configurations can include `MetricsConfig` to collect inference performance metrics during evaluation. A run without `--harness` or `--harness-config` collects them with the default `MetricsConfig()`, writing `metrics/<provider>_<model>-inference.jsonl` in the output directory, as the built-in presets do:

```python
from olmo_eval.harness import HarnessConfig, ProviderConfig
from olmo_eval.inference.metrics import MetricsConfig

config = HarnessConfig(
    name="with_metrics",
    provider=ProviderConfig(model="llama3.1-8b", kind="vllm_server"),
    metrics=MetricsConfig(
        enabled=True,
        reporters=("file", "db"),  # Save to file and database
        collect_vllm_server=True,  # Poll vLLM server /metrics endpoint
    ),
)
```

**Visualizing Metrics:**

```bash
# Plot metrics from database (requires at least one filter)
uv run olmo-eval metrics plot -G my-benchmark-group
uv run olmo-eval metrics plot -m OLMo-3 --metric throughput

# Show statistics table without interactive plots
uv run olmo-eval metrics plot -e experiment_123 --stats-only
```

When using the `db` reporter, metrics are stored in a PostgreSQL database (default name: `olmo_eval_metrics`). You must configure your own database connection using the `OLMO_EVAL_DB_*` environment variables (see [Database Configuration](#database-configuration)).

#### Auxiliary Providers and Local Judge Models

Some tasks or custom scorers use LLM-as-judge scoring, where a separate model evaluates responses. The `auxiliary_providers` configuration lets you specify additional inference providers for scoring or judging. Harness overrides must come immediately after `--harness`, while task overrides like `limit=...` must come after `-t`.

**Local example with `uv run olmo-eval run`:**

```bash
uv run olmo-eval run \
    --harness default \
    -o provider.max_model_len=16384 \
    -o provider.num_instances=1 \
    -o 'metrics.reporters=[file]' \
    -o 'metrics.collect_gpu=true' \
    -o 'provider.kwargs.timeout=300' \
    -o auxiliary_providers.judge.kind=vllm_server \
    -o auxiliary_providers.judge.model=Qwen/Qwen3-8B \
    -o auxiliary_providers.judge.num_instances=1 \
    -o scoring_concurrency=4 \
    -m Qwen/Qwen3-8B \
    -t simpleqa:judge \
    -o limit=10
```

**Key configuration options:**

| Option | Description |
|--------|-------------|
| `auxiliary_providers.judge.kind` | Provider type: `vllm_server`, `litellm`, etc. |
| `auxiliary_providers.judge.model` | Model to use for judging |
| `auxiliary_providers.judge.num_instances` | Number of parallel vLLM instances |
| `auxiliary_providers.judge.base_url` | URL for external servers (when not spawning locally) |
| `scoring_concurrency` | Number of concurrent scoring requests |

#### Defining Tools

Tools combine schema (for the LLM) and implementation (for execution) in a single definition:

```python
from olmo_eval.harness import tool, registered_tool

# Option 1: @tool decorator (local use)
@tool(description="Search the web for information")
async def web_search(query: str) -> str:
    """Search implementation."""
    return await search_api(query)

# Option 2: @registered_tool decorator (global registry, for cross-process use)
@registered_tool(description="Fetch a webpage")
async def fetch_page(url: str) -> str:
    """Fetch implementation."""
    return await fetch_url(url)
```

Tools are automatically registered when using `@registered_tool`, making them available by name in HarnessConfig.

#### Custom Harness Config File

Create a YAML file for custom harness configurations:

```yaml
# my_harness.yaml
name: custom_search
tool_names:
  - semantic_scholar_snippet_search
  - serper_google_webpage_search
system_prompt: |
  You are a research assistant with web search capabilities.
  Use search tools to find accurate information before answering.
max_turns: 15
max_concurrency: 4
required_secrets:
  - S2_API_KEY
  - SERPER_API_KEY
```

```bash
uv run olmo-eval run -m llama3.1-8b -t simpleqa:judge --harness-config my_harness.yaml
```

#### Programmatic Usage

```python
from olmo_eval.harness import Harness, HarnessConfig, ProviderConfig, get_harness_preset
from olmo_eval.harness.tools.search import (
    semantic_scholar_search,
    serper_web_search,
)

# Create harness with preset and provider override
config = get_harness_preset("dr_tulu").with_provider(
    ProviderConfig(model="meta-llama/Llama-3.1-8B-Instruct", kind="vllm")
)
harness = Harness(config)

# Or create from scratch
config = HarnessConfig(
    name="my_harness",
    provider=ProviderConfig(model="gpt-4o", kind="litellm"),
    tools=(semantic_scholar_search, serper_web_search),
    system_prompt="You are a helpful assistant.",
    scaffold="openai_agents",
)
harness = Harness(config)

# Multi-turn execution with tool calling
result = await harness.run(request, sampling_params)
print(result.trajectory)  # Shows all turns including tool calls
print(result.final_output)  # Final model response
```

## Adding New Tasks

This section explains how to create new evaluation tasks.

### Quick Start: Minimal Task Example

A multiple-choice task scored by the log-likelihood of each answer choice:

```python
"""Example: a minimal multiple-choice task scored by log-likelihood."""

from collections.abc import Iterator
from typing import Any

from olmo_eval.common.formatters import MultipleChoiceFormatter
from olmo_eval.common.metrics import LogprobMCAccuracyMetric
from olmo_eval.common.types import Instance, LMRequest
from olmo_eval.data import DataSource
from olmo_eval.evals.tasks.common import Task, register


@register("my_task")
class MyTask(Task):
    """MMLU abstract algebra, scored by the highest-likelihood answer choice."""

    # Pin the revision so the task's data cannot change underneath stored results.
    data_source = DataSource(
        path="cais/mmlu",
        subset="abstract_algebra",
        split="test",
        revision="c30699e8356da336a370243923dbaf21066bb9fe",
    )
    formatter = MultipleChoiceFormatter(
        template="Question: {question}",
        choice_template=" {choice}",
        prompt_suffix="\nAnswer:",
    )
    metrics = (LogprobMCAccuracyMetric(),)

    @property
    def instances(self) -> Iterator[Instance]:
        # Loads the configured data source, calls process_doc, and caches the result.
        yield from self._load_instances_cached()

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        return Instance(
            question=doc["question"],
            choices=tuple(doc["choices"]),
            gold_answer=doc["choices"][doc["answer"]],
            metadata={"id": f"my_task_{index}", "gold_idx": doc["answer"]},
        )

    def format_request(self, instance: Instance) -> LMRequest:
        assert self.config.formatter is not None
        return self.config.formatter.format(instance, self.get_fewshot())
```

Class attributes that match `TaskConfig` fields (`data_source`, `formatter`, `metrics`,
and so on) become the task's default configuration. Put the module in
`src/olmo_eval/evals/tasks/` and it is registered on import. Check it with
`uv run olmo-eval task inspect my_task --request -n 1` and
`uv run olmo-eval run -m mock -t my_task -o limit=5`. Generation tasks use a chat or
completion formatter instead, extract an answer in `extract_answer`, and score it with a
metric such as `AccuracyMetric(scorer=ExactMatchScorer)`.

### Task Class Overview

| Method | Required | Purpose |
|--------|----------|---------|
| `instances` | Yes | Property that yields `Instance` objects; usually `yield from self._load_instances_cached()` |
| `format_request(instance)` | Yes | Converts an `Instance` into an `LMRequest` for the model |
| `process_doc(doc, index)` | When using `_load_instances*` | Converts a raw document into an `Instance`, or `None` to skip it |
| `extract_answer(output)` | No | Extracts the answer from `LMOutput`; defaults to `config.answer_extractor` or the raw text |
| `_build_fewshot()` | No | Override to customize few-shot example loading |
| `score_responses(...)` | No | Override to customize scoring logic |
| `compute_metrics(...)` | No | Override to customize metric computation |

### TaskConfig Reference

Commonly used fields. `TaskConfig` in `src/olmo_eval/evals/tasks/common/base.py`
documents the rest.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | Required | Task identifier used in CLI |
| `data_source` | `DataSource \| str \| None` | `None` | Dataset source (HuggingFace, S3, GCS, local, or URI string) |
| `formatter` | `Formatter \| None` | `None` | Request formatter |
| `metrics` | `tuple[Metric, ...]` | `()` | Evaluation metrics (scorers are inferred from metrics) |
| `num_fewshot` | `int` | `0` | Number of few-shot examples |
| `limit` | `int \| None` | `None` | Max instances to evaluate |
| `split` | `Split` | `Split.TEST` | Dataset split to use |
| `primary_metric` | `MetricName \| Metric \| None` | `None` | Primary metric for ranking (defaults to single metric if only one) |
| `sampling_params` | `SamplingParams \| None` | `None` | Default sampling parameters for this task |
| `answer_extractor` | `Callable[[str], str] \| None` | `None` | Function applied to output text by the default `extract_answer` |
| `strip_thinking` | `bool` | `False` | Drop `<think>...</think>` traces before scoring |
| `dependencies` | `list[str] \| None` | `None` | Runtime packages to install (e.g., `["pkg==1.0"]`) |
| `judge_model` | `str \| None` | `None` | Judge model for LLM-as-judge scoring |

### Data Sources

Tasks can load data from multiple sources using `DataSource`:

```python
from olmo_eval.data import DataSource

# HuggingFace datasets - specify path, subset, and split
DataSource(path="cais/mmlu", subset="abstract_algebra", split="test")

# Using URI string (alternative syntax)
DataSource.from_uri("hf://cais/mmlu?subset=abstract_algebra&split=test")

# Without subset (for datasets that don't have subsets)
DataSource(path="openai_humaneval", split="test")

# With specific data files and revision
DataSource(path="my-org/dataset", data_files="data/test.jsonl", revision="v1.0")

# Local JSONL files
DataSource(path="/path/to/dataset.jsonl")

# S3
DataSource(path="s3://my-bucket/datasets/data.jsonl")

# GCS
DataSource(path="gs://my-bucket/datasets/data.parquet")
```

**DataSource Fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `path` | `str` | Required | Dataset path (HuggingFace repo, S3/GCS URI, or local path) |
| `subset` | `str \| None` | `None` | Dataset subset/config name |
| `split` | `str` | `"test"` | Dataset split |
| `source_type` | `SourceType \| None` | `None` | Backend type; inferred from `path` when unset |
| `data_files` | `str \| tuple[str, ...] \| None` | `None` | Specific data files to load |
| `revision` | `str \| None` | `None` | Dataset revision/version |

### Common Patterns

**Multiple Choice Tasks:**
```python
formatter=MultipleChoiceFormatter(template="Question: {question}\n\nAnswer:")
metrics=(AccuracyMetric(scorer=MultipleChoiceScorer),)
```

**Generation Tasks (exact match):**
```python
formatter=CompletionFormatter(template="{question}")
metrics=(AccuracyMetric(scorer=ExactMatchScorer),)
```

**Tasks with Multiple Subsets** (like MMLU with 57 subjects):
```python
# Base class with shared logic
class MMLUTask(Task):
    ...

# Register each subset - the subset is specified in DataSource
@register("mmlu_anatomy")
class MMLUAnatomy(MMLUTask):
    data_source = DataSource(path="cais/mmlu", subset="anatomy", split="test")

@register("mmlu_physics")
class MMLUPhysics(MMLUTask):
    data_source = DataSource(path="cais/mmlu", subset="high_school_physics", split="test")
```

### Adding Variants

**Variants** modify how a task is formatted/scored (e.g., `:mc`, `:bpb`):
```python
from olmo_eval.evals.tasks.common import register_variant

# Register after task is defined
register_variant("my_task", "bpb", formatter=PPLFormatter(), metrics=(BPBMetricByteAvg(scorer=BitsPerByteScorer),))
```

**Variants** can also encode configuration presets (e.g., `:3shot`, `:zero`):
```python
from olmo_eval.evals.tasks.common import register_variant

register_variant("my_task", "3shot", num_fewshot=3, fewshot_seed=1234)
register_variant("my_task", "zero", num_fewshot=0)
```

Usage: `uv run olmo-eval run -m llama3.1-8b -t humaneval:3shot:bpb`

## Tool-Augmented Evaluation

olmo-eval supports evaluating models with tool use through the **Harness** abstraction. This enables comparing baseline model performance against tool-augmented performance on the same tasks.

The **Harness** is the preferred way to add tools to evaluations. It separates tool configuration from task definition, allowing any task to be run with or without tools:

```bash
# Baseline evaluation (no tools)
uv run olmo-eval run -m llama3.1-8b -t simpleqa:judge

# Same task with search tools
uv run olmo-eval run -m llama3.1-8b -t simpleqa:judge --harness dr_tulu
```

See the [Harness](#harness) section above for full documentation on:
- Creating custom harness configurations
- Defining tools with the `@tool` decorator
- Programmatic usage

## Function Calling (BFCL)

The Berkeley Function Calling Leaderboard v3 single-turn categories are
registered as `bfcl_*` tasks, reformulated as a completion so a pretrained
model with no chat template can be measured on them. See
[`src/olmo_eval/evals/tasks/bfcl/README.md`](src/olmo_eval/evals/tasks/bfcl/README.md)
for the task list, suites, exemplar settings, and what is not implemented.

```bash
uv run olmo-eval run -m my-base-model -t bfcl
```

## Multimodal Evaluation

olmo-eval evaluates vision-language models on image benchmarks. Tasks attach images
to `LMRequest.images` and the provider decides how to render them, so the task
definition is the same regardless of which multimodal provider runs it.

Five families of image tasks are built in:

| Family | Suite | Tasks |
| --- | --- | --- |
| Image-QA | `molmo2_imageqa` | `chart_qa`, `vqa2`, `doc_qa`, `info_qa`, `text_vqa`, `real_world_qa`, `mmmu`, `mmmu_pro`, `math_vista`, `countbench_qa`, `pixmo_count`, `ai2d`, `charxiv_descriptive`, `charxiv_reasoning` |
| Dense caption | `molmo2_imageqa_caption` | the image-QA tasks plus `dense_caption` |
| Pointing | `molmo2_pointing` | `pixmo_points_eval`, `sa_co_gold_subset` |
| Pointing (model prompts) | `molmo2_pointing_mp` | `pixmo_points_eval_mp`, `sa_co_gold_subset_mp`, `sa_co_gold_point_4k_mp` |
| Multi-image | `molmo2_multiimage` | `muir_bench`, `mmiu`, `blink` |
| Document OCR, English (default for Molmo2) | `ocr_en` | `olmocr_bench`, `cc_ocr_multi_scene_en`, `omnidocbench_en` |
| Document OCR, every language | `ocr` | `olmocr_bench`, `cc_ocr_multi_scene`, `omnidocbench` |

Image-QA primary metrics are all 0-1, so `molmo2_imageqa` averages them.
`dense_caption` reports on a 0-100 scale, so `molmo2_imageqa_caption` is display-only
and computes no cross-task average. Pointing tasks score point-in-mask
precision/recall/f1 by maximum bipartite matching rather than VQA-style answers,
which is why they are a separate suite. Multi-image tasks attach a *list* of images
per instance (capped at 20, matching the mm_olmo eval config) and score multiple
choice answers by MMMU-style option-letter parsing; besides the primary `all`
accuracy each task reports per-category breakdowns (MuirBench's 12 task types,
BLINK's 14 subtasks, MMIU's 7 relationship types plus image-count buckets).

Document-OCR tasks hand the model one page image and grade the markdown (or JSON)
it writes back, each with its benchmark's official scoring:

- `olmocr_bench` — [olmOCR-bench](https://huggingface.co/datasets/allenai/olmOCR-bench):
  1,403 PDF pages and 7,019 unit tests (text presence/absence, reading order, table
  cells, rendered math). Tests run through the official `olmocr` scorer; `overall` is
  the mean of the eight category pass rates (0-1). Math tests render with KaTeX in a
  headless Chromium, which is installed on first use.
- `cc_ocr_multi_scene` — the multi-scene OCR track of
  [CC-OCR](https://huggingface.co/datasets/wulipc/CC-OCR): 2,750 scene-text, document and
  web images in English and Chinese. The primary `macro_f1` is the official track score
  (0-1): per-image F1 of the word (Chinese: character) multiset, averaged within and then
  across the 13 sub-datasets.
- `omnidocbench` — [OmniDocBench](https://github.com/opendatalab/OmniDocBench) v1.6:
  1,651 pages scored by the pinned official evaluator, which runs in a virtualenv of its
  own (needs `git` and `uv`). `overall` is the leaderboard's 0-100 number, which is why
  the `ocr` suite is display-only. Its formula metric (CDM) renders LaTeX; the task sets
  up the toolchain itself at job start, so it runs in one step anywhere (see
  [OmniDocBench formula scoring](#omnidocbench-formula-scoring-cdm)). `omnidocbench_v15`
  is the 1,355-page v1.5 release with its own evaluator; the two leaderboards are not
  comparable.

English-only variants keep the benchmarks' official scoring on a subset:
`cc_ocr_multi_scene_en` runs the 8 English sub-datasets (2,000 images) and averages over
those, and each OmniDocBench version has an `_en` variant (`omnidocbench_en`,
`omnidocbench_v15_en`) that runs and scores only the pages annotated `english` (755 of
v1.6's pages, 628 of v1.5's). Molmo2 models read English only, so the `_en` variants (suite
`ocr_en`) are the default setting for evaluating them. The plain names run every language,
as the leaderboards do, for multilingual models; `olmocr_bench` is English throughout and
has no variant.

By default each OCR task sends its benchmark's own instruction, for instruction-tuned
checkpoints. A stage-1 checkpoint (`-o prompt_templates=none -o
system_prompt_style=style_and_length_v2`) gets the OCR style tag it was trained on, alone:
`olmocr:` for `olmocr_bench` and OmniDocBench, `textocr:` for CC-OCR.

The OCR datasets are downloaded from the Hugging Face Hub at pinned revisions rather
than read from `$MOLMO_DATA_DIR`.

### Setup

The multimodal providers live behind the `hf` and `olmo_core_vlm` extras, and the
benchmarks' own scoring dependencies (`pycocotools`, `scipy`) behind `vision`, which
every provider needs:

```bash
# HuggingFace multimodal path
uv sync --extra hf --extra vision

# Also evaluate OLMo-core MultimodalLM checkpoints
uv sync --extra hf --extra olmo_core_vlm --extra vision
```

Beaker jobs install each task's scoring dependencies automatically.

Image data is read from a **read-only** tree — loaders raise rather than build
caches, so the data must already be staged:

```bash
export MOLMO_DATA_DIR=/weka/oe-training-default/mm-olmo  # this is the default
```

A missing data root fails at task setup with the variable named, rather than
reading a nonexistent path (the Ai2 defaults are tracked for removal in #381).

Images live under `$MOLMO_DATA_DIR/torch_datasets/`. Manifests that recorded
absolute paths on another machine are re-anchored under the current root
automatically.

Four tasks call the OpenAI API and need `OPENAI_API_KEY` or `OPENROUTER_API_KEY`
(see [OpenAI models through OpenRouter](#openai-models-through-openrouter)), so the
`molmo2_imageqa` suite needs one of them as a whole:

- `math_vista` defaults to the official `gpt-4-0613` answer extraction. Use the
  `math_vista:offline` variant for an API-free heuristic instead.
- `charxiv_descriptive` and `charxiv_reasoning` run the official `gpt-4o` grader.
- `dense_caption` runs a `gpt-4o` recall+consistency judge. Judge responses are
  cached, so the key is only needed on a cache miss.

The launcher resolves the key as the user-scoped beaker secret
`{beaker-username}_OPENAI_API_KEY`, or `{beaker-username}_OPENROUTER_API_KEY` when
that is missing. If neither exists, the launch fails up front and prints the
`beaker secret write` command to run.

### Running

```bash
# Released HuggingFace checkpoint, full image-QA suite
uv run olmo-eval run -m molmo2-4b -t molmo2_imageqa

# A single benchmark
uv run olmo-eval run -m molmo2-4b -t mmmu_pro

# Pointing benchmarks
uv run olmo-eval run -m molmo2-4b -t molmo2_pointing

# Multi-image benchmarks
uv run olmo-eval run -m molmo2-4b -t molmo2_multiimage

# Inspect the suite without loading a model
uv run olmo-eval suite inspect molmo2_imageqa
```

The `molmo2-4b` preset runs `allenai/Molmo2-4B` with `dtype=float32`,
`autocast_dtype=bfloat16` and `max_crops=24` (8 per image in multi-image prompts),
matching the reference evaluation numerics.

### OmniDocBench formula scoring (CDM)

`omnidocbench`, `omnidocbench_v15` and their `_en` variants score formulas with CDM,
which renders every formula through LaTeX and ImageMagick. Before inference each task
installs whatever part of that toolchain is missing or at another version, then renders
a test formula through its version's own CDM template. The whole evaluation, CDM
included, finishes in the same job; no separate rescoring step is needed. Setup takes
about 5-9 minutes on a fresh machine.

The versions are the ones each release documents, because CDM follows the renderer
(Ghostscript 10.02 instead of 9.55 moves a formula score by about 0.2 points):

| | v1.6 (`omnidocbench`) | v1.5 (`omnidocbench_v15`) |
| --- | --- | --- |
| LaTeX | TeX Live 2025, `pdflatex` + `CJK` | TeX Live 2025, `xelatex` + `xeCJK` |
| Font | `gkai` (TeX Live `arphic`) | Source Han Sans SC |
| ImageMagick | 7.1.1-47 | 7.1.1-47 |
| Ghostscript | 9.55.0 | 9.55.0 |
| Other | | Node.js 16.13.1 (KaTeX tokenizer) |

What the setup needs:

- **Network access**: TeX Live 2025 comes from the frozen `tlnet-final` repository in the
  TeX historic archive, ImageMagick and Ghostscript from their GitHub releases, the font
  from Adobe's release, and Node.js from nodejs.org. Each download is checked against its
  pinned SHA-256, and a mismatch stops the task.
- **Linux x86_64**, where the installed binaries run. Elsewhere, put the toolchain on
  `PATH` yourself.
- **apt** (root, as in Beaker jobs), for the system libraries the ImageMagick AppImage
  links against and for `fontconfig`.
- TeX Live, ImageMagick and Ghostscript already on `PATH` at exactly these versions are
  used as they are, as is any `node`; anything else is installed alongside, first on
  `PATH` for the evaluation.

If a download fails or the test formula does not render, the task stops before inference
with the error, rather than scoring every formula zero. On a machine without network
access or root, put the pinned versions on `PATH` first (the task then uses them as they
are) or point `OMNIDOCBENCH_CDM_DIR` at a toolchain installed elsewhere.

Environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `OMNIDOCBENCH_CDM_DIR` | `~/.cache/olmo_eval/omnidocbench/cdm` | Where the toolchain is installed; point several runs at one shared install to reuse it |
| `OMNIDOCBENCH_EVAL_DIR` | `~/.cache/olmo_eval/omnidocbench` | Where each version's evaluator checkout and virtualenv are provisioned |
| `OMNIDOCBENCH_EVAL_WORKERS` | `min(16, cpus / 4)` | Evaluator worker threads (matching, TEDS, CDM); 48 suits a Beaker GPU node |
| `OMNIDOCBENCH_EVAL_TIMEOUT_S` | `43200` (12 h) | Wall-clock cap on evaluation; past it, as on any evaluator failure, the task reports no metrics and an error, and keeps the predictions for rescoring |
| `OMNIDOCBENCH_DIR` / `OMNIDOCBENCH_V15_DIR` | Hub download | Local copy of the dataset (`OmniDocBench.json` + `images/`) |

For example, on Beaker:

```bash
olmo-eval beaker launch -n molmo2-4b-omnidocbench \
  -m /weka/oe-training-default/mm-olmo/released-models-molmo2-1225/Molmo2-4B/step2000 \
  -t omnidocbench_en \
  --harness default -o provider.kind=olmo_core_vlm -o provider.max_crops=24 \
  -o provider.num_instances=2 \
  --gpus 2 --env OMNIDOCBENCH_EVAL_WORKERS=48 \
  -c ai2/holmes -p urgent --min-runtime 8h -w <workspace> -B <budget>
```

The job log shows the toolchain setup (`Running: ...install-tl...`, `tlmgr install ...`),
then inference, then one `overall` for the task.

### Providers

**`huggingface`** handles image-text-to-text models via `AutoProcessor` and
`AutoModelForImageTextToText`. Pass `multimodal=True` in the provider kwargs (the
`molmo2-4b` preset does this for you).

**`olmo_core_vlm`** runs OLMo-core `MultimodalLM` checkpoints in three on-disk
formats: raw trainer saves (`config.json` + `model_and_optim/`), consolidated
safetensors exports, and mm_olmo trainer checkpoints, which are config-translated
and key-remapped at load time. It drives its own decode loop because OLMo-core's
vision branch has no multimodal generation module.

Prefer `provider.kind=olmo_core_vlm` explicitly — remote jobs then install the
matching `olmo_core_vlm` extra. `provider.kind=olmo_core` also detects
multimodal checkpoints and reroutes (with a warning), but on Beaker that route
installs the text extra and fails at worker init:

```bash
uv run olmo-eval run \
    --harness default -o provider.kind=olmo_core \
    -m /weka/path/to/multimodal-checkpoint \
    -t molmo2_imageqa
```

`-o` must follow `--harness` or `-t`, so the provider override needs an explicit
`--harness default` ahead of it.

A single image is cropped to the checkpoint's training `max_crops`
(`-o provider.max_crops=N`), and each image of a multi-image prompt to its
`max_multi_image_crops` (8 when the checkpoint names none;
`-o provider.max_multi_image_crops=N`), as mm_olmo does. The context window is
64,000 tokens, the length mm_olmo's `eval_molmo2.py` evaluates at, or the
checkpoint's training window if that is longer (`-o provider.max_model_len=N`); a
prompt longer than the window fails its instance rather than being truncated.
mm_olmo evaluates the released Molmo2 models at 24 crops per single image, so pass
`-o provider.max_crops=24` to compare against their published numbers; document QA
is the most crop-sensitive.

No released `ai2-olmo-core` ships the multimodal classes (`olmo_core.nn.vision`,
`MultimodalLM`) yet, so the `olmo_core_vlm` extra pins OLMo-core's vision branch
by commit. It is declared conflicting with the text `olmo_core` extra — the two
cannot be installed together — and beaker jobs install it automatically for
`provider.kind=olmo_core_vlm`, so no package override is needed:

```bash
uv run olmo-eval beaker launch \
    --harness default -o provider.kind=olmo_core_vlm \
    -m /weka/path/to/multimodal-checkpoint \
    -t molmo2_imageqa
```

Once a release includes the multimodal classes, the extras collapse into one and
the pin goes away.

### Adding a multimodal task

Image-QA tasks subclass `ImageQATask` (`evals/vision/tasks/image_qa.py`),
which resolves images from `instance.metadata["image_path"]` or
`instance.metadata["image"]` (a PIL image or a zero-arg callable returning one —
use `load_instance_image` from `evals/vision/data/images.py` for either form) and anchors data reads under
`$MOLMO_DATA_DIR`. Pointing tasks subclass `PointingTask`
(`evals/vision/tasks/pointing.py`). Multi-image tasks subclass
`MultiImageQATask` (`evals/vision/tasks/multi_image.py`), which resolves
`instance.metadata["images"]` — a callable returning the image list, or a sequence
of PIL images / zero-arg callables / file paths — and sends them all on one
request. Registration and variants work exactly as described in
[Adding New Tasks](#adding-new-tasks).

## Querying Results

Evaluation results can be stored in PostgreSQL and queried via the CLI.

### Basic Queries

```bash
# Query by experiment ID
uv run olmo-eval results query --experiment exp_001

# Query by model
uv run olmo-eval results query --model llama3.1-8b

# Query by task (shows comparison matrix)
uv run olmo-eval results query --task mmlu --task gsm8k

# Query by experiment group
uv run olmo-eval results query -G my-benchmark-group --format json

# Combine filters
uv run olmo-eval results query --model llama3.1-8b --task mmlu --format json
```

### Instance-Level Predictions

Include `--instances` to retrieve instance-level predictions:

```bash
# Get instances for an experiment
uv run olmo-eval results query --experiment exp_001 --task mmlu --instances --format json

# Paginate through large result sets using keyset pagination
uv run olmo-eval results query --task mmlu --instances --limit 1000 --format json

# Get next page using last_id from previous response
uv run olmo-eval results query --task mmlu --instances --limit 1000 --after-id 1000 --format json
```

JSON output includes pagination metadata:
```json
{
  "experiments": [...],
  "pagination": {
    "last_id": 12345,
    "has_more": true
  }
}
```

### Output Formats

| Format | Flag | Description |
|--------|------|-------------|
| Table | `--format table` | Rich terminal tables (default) |
| JSON | `--format json` | Structured JSON with pagination metadata |
| CSV | `--format csv` | CSV output to stdout |

### Database Configuration

#### AI2 Users (Recommended)

Set these two environment variables to connect to the shared database:

```bash
export OLMO_EVAL_DB_HOST="<database-host>"
export OLMO_EVAL_DB_SECRET_ARN="arn:aws:secretsmanager:us-west-2:..."
```

The password is automatically fetched from AWS Secrets Manager on first connection.
This requires AWS credentials configured (via `~/.aws/credentials` or environment variables).

#### All Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLMO_EVAL_DB_HOST` | `localhost` | Database host |
| `OLMO_EVAL_DB_PORT` | `5432` | Database port |
| `OLMO_EVAL_DB_NAME` | `olmo_eval` | Database name |
| `OLMO_EVAL_DB_USER` | `postgres` | Database user |
| `OLMO_EVAL_DB_PASSWORD` | - | Database password (use this OR `OLMO_EVAL_DB_SECRET_ARN`) |
| `OLMO_EVAL_DB_SECRET_ARN` | - | AWS Secrets Manager ARN for password (fetched on auth failure) |

## Advanced Usage

### Multi-GPU and Tool-Augmented Evaluation

```bash
# Basic evaluation
uv run olmo-eval run -m llama3.1-8b -t mmlu -t gsm8k -t arc_easy

# Large models with multi-GPU tensor parallelism
uv run olmo-eval run -m llama3.1-70b -t mmlu --num-gpus 4

# Refresh Hugging Face cache before loading a remote model
uv run olmo-eval run -m allenai/OLMo-2-1124-7B -t mmlu --force-download-model

# Tool-augmented evaluation with harness
uv run olmo-eval run -m llama3.1-8b -t simpleqa:judge --harness dr_tulu
```

## Debugging and Inspection

olmo-eval provides tools for inspecting tasks, requests, and responses at various stages of evaluation.

### Task Inspection (`uv run olmo-eval task inspect`)

Inspect task instances without running evaluation:

```bash
# View raw instance data
uv run olmo-eval task inspect arc_easy

# View multiple instances
uv run olmo-eval task inspect arc_easy -n 5 --skip 10

# View the LM request that will be sent to the model
uv run olmo-eval task inspect arc_easy:mc --request

# View formatted prompt with chat template applied
uv run olmo-eval task inspect humaneval -T meta-llama/Llama-3.1-8B-Instruct --formatted

# View tokenized representation
uv run olmo-eval task inspect humaneval -T meta-llama/Llama-3.1-8B-Instruct --tokens

# Export as JSON for programmatic use
uv run olmo-eval task inspect arc_easy --json
```

| Option | Description |
|--------|-------------|
| `-n, --count` | Number of instances to display |
| `-s, --skip` | Number of instances to skip |
| `--instance` | Show instance details (default if no other flags) |
| `--request` | Show the LM request |
| `-T, --tokenizer` | Tokenizer for formatting/tokenization |
| `--formatted` | Show prompt after template applied (requires `-T`) |
| `--tokens` | Show token array (requires `-T`) |
| `--max-tokens` | Max tokens to display (0 for no limit) |
| `--max-chars` | Max chars for formatted prompt (0 for no limit) |
| `--max-string-length` | Max chars for instance field values (0 for no limit) |
| `--json` | Output as JSON |

### Runtime Inspection Flags

Inspect data during evaluation runs with `uv run olmo-eval run`:

```bash
# Enable all inspection flags at once
uv run olmo-eval run -m llama3.1-8b -t mmlu --inspect

# Or use individual flags for specific inspection
uv run olmo-eval run -m llama3.1-8b -t mmlu --inspect-instance --inspect-request

# Inspect the response after model generation
uv run olmo-eval run -m llama3.1-8b -t mmlu --inspect-response

# Combine multiple inspection flags
uv run olmo-eval run -m llama3.1-8b -t mmlu \
    --inspect-instance \
    --inspect-request \
    --inspect-response
```

| Flag | Description |
|------|-------------|
| `--inspect` | Enable all inspection flags below |
| `--inspect-instance` | Print the first instance of each task before running |
| `--inspect-request` | Print the first LM request before model generation |
| `--inspect-formatted` | Show formatted prompt (after chat template applied) |
| `--inspect-tokens` | Show token array before evaluation |
| `--inspect-response` | Print the first response after model generation |

### Mock Provider for Testing

Use the `mock` provider to test inspection tools without loading a real model:

```bash
# Quick inspection without vLLM or PyTorch
uv run olmo-eval run -m mock -t humaneval:3shot:bpb --inspect-request

# Dry run with mock to preview configuration
uv run olmo-eval run -m mock -t mmlu --dry-run
```

## External Evals

External evals are standalone evaluations that run outside the normal task pipeline.
Use them when a benchmark already comes with its own harness, verifier, or environment
and does not fit cleanly into the usual task formatter/scorer flow. They are a good fit
for agent-style benchmarks like `terminal_bench_2`, `tau2_bench`, and `asta_bench`
that need sandbox orchestration, benchmark-specific setup, or end-to-end execution
against an external repo or runner.

### Defining an External Eval

`SandboxedExternalEval` runs a benchmark inside one container. This sketch runs a
benchmark's own CLI against the model under evaluation and reads its score file:

```python
import json
import time
from typing import Any

from olmo_eval.evals.external import (
    ExternalEvalResult,
    SandboxedExternalEval,
    register_external_eval,
)
from olmo_eval.harness.sandbox import SandboxExecutor


class MyBenchmarkExternalEval(SandboxedExternalEval):
    """My benchmark evaluation."""

    name = "my_benchmark"
    description = "Evaluates a model on my benchmark"
    timeout_seconds = 3600.0
    required_secrets = ("MY_API_KEY",)
    sandbox_image = "my-benchmark:latest"
    working_dir = "/workspace"
    setup_command = ("pip install -r requirements.txt",)

    @property
    def arguments(self) -> dict[str, tuple[str, Any | None]]:
        # arg_name -> (description, default_value)
        return {"subset": ("Which subset to evaluate", "default")}

    async def execute(self, provider, args, output_dir=None, container_runtime="podman"):
        start_time = time.time()
        output: list[str] = []
        config = self._create_sandbox_config(container_runtime, output_dir)
        async with SandboxExecutor(config, name=self.name) as executor:
            if error := await self._run_setup(executor, output, start_time):
                return error
            run = await executor.execute_command(
                f"my-benchmark --model {provider.model_name} --subset {args['subset']} "
                f"--out {self.results_dir}/score.json",
                timeout=self.timeout_seconds,
            )
            output.append(run.output)
            if not run.success:
                return self._error_result("Benchmark run failed", start_time, "\n".join(output))
            scores = await executor.execute_command(f"cat {self.results_dir}/score.json")
        return ExternalEvalResult(
            name=self.name,
            metrics={"accuracy": json.loads(scores.output)["accuracy"]},
            raw_output="\n".join(output),
            duration_seconds=time.time() - start_time,
        )


register_external_eval(MyBenchmarkExternalEval())
```

Benchmark modules under `src/olmo_eval/evals/external/benchmarks/` are registered
automatically. See `benchmarks/tau2/eval.py` for a complete implementation, including how
a sandboxed benchmark reaches a locally served model.

### Running External Evals

```bash
# List available external evals
uv run olmo-eval external-evals

# Run a built-in external eval
uv run olmo-eval run-external -e tau2_bench --model llama3.1-8b -a domain=airline -a num_tasks=1
```

### ExternalEvalResult

External evals return structured results:

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Eval identifier |
| `metrics` | `dict[str, float]` | Evaluation metrics |
| `metadata` | `dict` | Additional metadata |
| `success` | `bool` | Whether the eval completed successfully |
| `error` | `str \| None` | Error message if failed |
| `duration_seconds` | `float \| None` | Execution time |
| `raw_output` | `str \| None` | Raw stdout/stderr from the evaluation |
| `predictions` | `list[dict] \| None` | Instance-level predictions |

## Sandboxes

Sandboxes provide isolated execution environments for code execution, tool use, and external evals.

### Configuration

```python
from olmo_eval.harness.sandbox import SandboxConfig, SandboxMode, Capability

config = SandboxConfig(
    image="python:3.12-slim",
    mode=SandboxMode.DOCKER,
    command_timeout=30.0,
    startup_timeout=60.0,
    instances=4,  # Run 4 parallel executors
    working_dir="/workspace",
    environment=(("MY_VAR", "value"),),
    volumes=(("/host/path", "/container/path"),),
    capabilities=Capability.BASH | Capability.PYTHON,  # Union of frozensets
)
```

### SandboxConfig Fields

Commonly used fields. `SandboxConfig` in `src/olmo_eval/harness/sandbox/config.py`
documents the rest, including Modal and SWE-ReX options.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `image` | `str` | Required | Container image |
| `mode` | `SandboxMode` | Required | `LOCAL`, `DOCKER`, or `MODAL` |
| `capabilities` | `frozenset[str]` | `Capability.DEFAULT` | Capabilities this sandbox provides, such as `Capability.BASH` or `Capability.PYTHON` |
| `instances` | `int \| None` | `None` | Executors to create; `None` lets the caller allocate capacity, otherwise one executor |
| `container_runtime` | `str` | `"podman"` | `"docker"` or `"podman"` |
| `command_timeout` | `float` | `30.0` | Default timeout per command (seconds) |
| `working_dir` | `str` | `"/workspace"` | Working directory in the container |
| `environment` | `tuple[tuple[str, str], ...]` | `()` | Environment variables as `(name, value)` pairs |
| `volumes` | `tuple[tuple[str, str], ...]` | `()` | Volume mounts as `(host_path, container_path)` pairs |

### Using SandboxManager

The `SandboxManager` manages multiple executors and routes each command to an executor
with the required capabilities:

```python
import asyncio

from olmo_eval.harness.sandbox import Capability, SandboxConfig, SandboxManager, SandboxMode

configs = [
    SandboxConfig(image="python:3.12", mode=SandboxMode.DOCKER, capabilities=Capability.PYTHON, instances=2),
    SandboxConfig(image="ubuntu:22.04", mode=SandboxMode.DOCKER, capabilities=Capability.BASH),
]

manager = SandboxManager(configs, owner="my-scorer")
await manager.start()

# Run on an executor that provides the python capability
output = await manager.execute("python -c 'print(1)'", capabilities=Capability.PYTHON)

# Commands that need the same capability are spread across matching executors
outputs = await asyncio.gather(*[
    manager.execute(command, capabilities=Capability.PYTHON) for command in commands
])

# execute_command and execute_code return an ExecutionResult with exit code and output
result = await manager.execute_code("print('hello')", capabilities=Capability.PYTHON)

await manager.stop()
```

### Sandbox Modes

| Mode | Description |
|------|-------------|
| `LOCAL` | Run commands locally (development only) |
| `DOCKER` | Run in Docker/Podman containers |
| `MODAL` | Run on Modal cloud platform |

## Launching on Beaker

olmo-eval includes built-in support for launching evaluation jobs on [Beaker](https://beaker.org).

### Installation

The `beaker` extra is included in the default `dev` group, so a plain
`uv sync --frozen` is enough. If you previously opted out of the default
groups, re-enable it with:

```bash
uv sync --frozen --extra beaker
```

### CLI Usage

Launch an evaluation job:

```bash
# Basic evaluation
uv run olmo-eval beaker launch -n "eval-llama3-mmlu" -m llama3.1-8b -t mmlu \
    --cluster h100 \
    -w "ai2/olmo-eval-debug" \
    -B "ai2/oe-other"

# Multiple tasks
uv run olmo-eval beaker launch -n "eval-llama3-suite" \
    -m llama3.1-8b \
    -t mmlu -t gsm8k -t hellaswag \
    --cluster h100 \
    -w "ai2/olmo-eval-debug" \
    -B "ai2/oe-other"

# Large model with multiple GPUs
uv run olmo-eval beaker launch \
    --name "eval-70b-full" \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --task mmlu --task gsm8k --task arc_easy \
    --cluster h100 \
    --workspace "ai2/olmo-eval-debug" \
    --budget "ai2/oe-other" \
    --gpus 4 \
    --timeout 48h

# Preview the Beaker spec without launching
uv run olmo-eval beaker launch -n "test" -m llama3.1-8b -t arc_easy \
    --cluster h100 \
    -w "ai2/olmo-eval-debug" \
    -B "ai2/oe-other" \
    --dry-run

# With a harness preset for tool-augmented evaluation
uv run olmo-eval beaker launch -n "eval-with-tools" \
    -m llama3.1-8b \
    -t simpleqa:judge \
    --harness dr_tulu \
    --cluster h100 \
    -w "ai2/olmo-eval-debug" \
    -B "ai2/oe-other"

# With inspection flags for debugging
uv run olmo-eval beaker launch -n "debug-eval" \
    -m llama3.1-8b \
    -t mmlu -o limit=10 \
    --inspect-request \
    --inspect-response \
    --cluster h100 \
    -w "ai2/olmo-eval-debug" \
    -B "ai2/oe-other"

# Run external evaluations
uv run olmo-eval beaker launch -n "external-eval" \
    -E tau2_bench \
    -m llama3.1-8b \
    -A domain=airline \
    -A num_tasks=1 \
    --cluster h100 \
    -w "ai2/olmo-eval-debug" \
    -B "ai2/oe-other"
```

### Advanced Usage

#### Local Judge Models

For tasks or custom scorers that use a named auxiliary judge provider, you can run a local judge model alongside the main model. Put harness overrides immediately after `--harness`, then put task overrides after `-t`.

```bash
uv run olmo-eval beaker launch \
    --harness default \
    -o provider.max_model_len=16384 \
    -o provider.num_instances=1 \
    -o 'metrics.reporters=[file]' \
    -o 'metrics.collect_gpu=true' \
    -o 'provider.kwargs.timeout=300' \
    -o auxiliary_providers.judge.kind=vllm_server \
    -o auxiliary_providers.judge.model=Qwen/Qwen3-8B \
    -o auxiliary_providers.judge.num_instances=1 \
    -o scoring_concurrency=4 \
    -m Qwen/Qwen3-8B \
    -t "simpleqa:judge@urgent" \
    -o limit=10 \
    -w "ai2/olmo-eval-debug" \
    -B "ai2/oe-other" \
    --cluster h100 \
    --inspect \
    --group olmo-eval-local-judge-2 -y
```

### Per-Task Priorities

Tasks can include an optional `@priority` suffix to set different priorities per task.
Tasks with different priorities will be launched as separate Beaker experiments:

```bash
# Mixed priorities - creates separate experiments per priority level
uv run olmo-eval beaker launch -n "eval-suite" -m llama3.1-8b \
    -t "mmlu@high" \
    -t "gsm8k@normal" \
    -t "arc_easy@low" \
    --cluster h100 \
    -w "ai2/olmo-eval-debug" \
    -B "ai2/oe-other"

# Creates 3 experiments:
#   eval-suite-high:   runs mmlu at high priority
#   eval-suite-normal: runs gsm8k at normal priority
#   eval-suite-low:    runs arc_easy at low priority

# With task variants (@ comes after the task spec)
uv run olmo-eval beaker launch -n "eval" -m llama3.1-8b -t "arc_easy:mc@high" \
    --cluster h100 \
    -w "ai2/olmo-eval-debug" \
    -B "ai2/oe-other"

# Tasks without @priority use the config file priority (default: normal)
```

### Experiment Groups

Groups logically organize experiments for management and result retrieval:

```bash
# Launch with grouping
uv run olmo-eval beaker launch -n "benchmark-v1" --group "benchmark-2024" \
    -m llama3.1-8b -m olmo-2-7b \
    -t mmlu -t gsm8k -t hellaswag \
    --cluster h100 \
    -w "ai2/olmo-eval-debug" \
    -B "ai2/oe-other"

# Creates experiment and adds it to "benchmark-2024" group

# Check group status and results
uv run olmo-eval beaker group info benchmark-2024

# Show detailed task info
uv run olmo-eval beaker group info benchmark-2024 --verbose

# Wait for completion and export as CSV
uv run olmo-eval beaker group info benchmark-2024 --wait --format csv > results.csv

# Export as JSON
uv run olmo-eval beaker group info benchmark-2024 --format json

# Watch experiment logs
uv run olmo-eval beaker watch -e <experiment-id>

# Cancel all experiments in a group
uv run olmo-eval beaker group cancel benchmark-2024

# List groups in a workspace
uv run olmo-eval beaker group list -w <workspace>
```

### Inference Provider Configuration

Docker images do NOT include inference providers (vllm, transformers, litellm) by default.
Each model must resolve to a provider configuration, either from a built-in model preset or from harness overrides.

**Via config file (recommended):**

```yaml
name: eval-mixed-providers
models:
  - llama3.1-8b
  - gpt-4o
tasks:
  - mmlu
cluster: h100
workspace: ai2/olmo-eval-debug
budget: ai2/oe-other
```

### CLI Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--config` | `-f` | none | YAML config file (CLI args override config values) |
| `--name` | `-n` | auto | Experiment name (auto-generated from model/tasks if not provided) |
| `--model` | `-m` | required | Model name or HuggingFace path (can specify multiple) |
| `--task` | `-t` | required | Task name with optional `@priority` suffix (can specify multiple) |
| `--harness` | `-H` | none | Harness preset name |
| `--override` | `-o` | none | Override for preceding `-t` or `-H` (can specify multiple) |
| `--cluster` | `-c` | required | Cluster alias (`h100`, `a100`, `aus`) or full name |
| `--gpus` | `-G` | auto | Number of GPUs (defaults to 1 for GPU providers, 0 otherwise) |
| `--max-gpus-per-node` | | `8` | Maximum GPUs per node (tasks split if exceeded) |
| `--priority` | `-p` | `normal` | Job priority (`low`, `normal`, `high`, `urgent`) |
| `--preemptible` | | `true` | Allow preemption (deprecated by Beaker; prefer `--min-runtime`) |
| `--min-runtime` | | none | Minimum runtime before Beaker may preempt the job (e.g., `2h`) |
| `--timeout` | `-T` | `24h` | Job timeout (e.g., `24h`, `30m`) |
| `--retries` | `-r` | none | Number of retries on failure |
| `--workspace` | `-w` | required | Beaker workspace |
| `--budget` | `-B` | required | Beaker budget |
| `--image` | `-I` | default | Custom Beaker image |
| `--group` | `-g` | auto | Add experiments to Beaker group(s) (auto-generated if not specified) |
| `--external-eval` | `-E` | none | External evaluation name(s) to run instead of tasks |
| `--eval-arg` | `-A` | none | Arguments for external evals (`key=value`) |
| `--provider-kwarg` | `-K` | none | Provider kwargs for external evals (`key=value`) |
| `--force-download-model` | | `false` | Refresh Hugging Face model/tokenizer cache before loading |
| `--uv-cache-dir` | | default | UV cache directory for package downloads |
| `--weka-bucket` | | none | Extra Weka bucket to mount at `/weka/<bucket>`, on top of `oe-training-default`, `oe-eval-default` and `oe-adapt-default` (can specify multiple) |
| `--dry-run` | `-d` | `false` | Print spec without launching |
| `--yes` | `-y` | `false` | Skip confirmation prompt |
| `--follow/--no-follow` | | `true` | Follow logs after launch |
| `--secret-env` | | none | Map Beaker secret to env var (`SECRET:VAR`) |
| `--aws-credentials` | | auto | Inject AWS credentials (auto-detected from s3:// paths) |
| `--gcp-credentials` | | auto | Inject GCP credentials (auto-detected from gs:// model paths) |
| `--store` | | `false` | Persist results to configured database |

### Preemption

By default, Beaker can preempt a job at any time and requeues it afterward. Use
`--min-runtime` (or `min_runtime:` in a config file) to protect a job from preemption:

```bash
uv run olmo-eval beaker launch -m llama3.1-8b -t mmlu -c h100 --min-runtime 4h
```

- How long the protection lasts depends on the cluster's scheduling policy. Where the
  policy makes jobs interruptible after their min runtime, the job is protected for that
  long, then can be preempted and is requeued. On other clusters, any min runtime
  protects the job for its whole run.
- Beaker accepts 5m to 8h, and no more than the cluster's maximum task timeout.
- Clusters that require an allocation for protected work reject a min runtime unless
  your workspace has an allocation there.
- `min_runtime` cannot be combined with `preemptible` in the CLI or in a config file.
  When the CLI sets `--min-runtime` or `--preemptible/--no-preemptible`, it replaces
  both config file fields.
- External evals (`-E`) take `--min-runtime` from the CLI only; they do not read the
  config file.
- `--no-preemptible` still works. Beaker treats it as an 8h min runtime with auto-resume
  turned off.

### Per-Task Overrides

Use the `-o/--override` flag to apply configuration overrides to the preceding `-t`:

```bash
# Task overrides (apply to the preceding -t)
uv run olmo-eval beaker launch -n "eval" \
    -m llama3.1-8b \
    -t mmlu -o limit=100 -o num_fewshot=5 \
    -t gsm8k -o limit=50 \
    --cluster h100 \
    -w "ai2/olmo-eval-debug" \
    -B "ai2/oe-other"
```

Task overrides accept `TaskConfig` fields and `SamplingParams` fields (sampling shortcuts
such as `temperature` or `max_tokens`). Any other key after `-t` is rejected; harness keys
belong after `--harness`.

| Kind | Example |
|------|---------|
| Number | `-o limit=100`, `-o num_fewshot=5` |
| Sampling shortcut | `-o temperature=0.6`, `-o max_tokens=512` |
| Boolean | `-o strip_thinking=true` |
| List (JSON) | `-o 'dependencies=["pkg1==1.0", "pkg2"]'` |
| Harness, nested | `--harness default -o provider.max_model_len=16384` |

Quote values that contain brackets, braces, or spaces so the shell passes them through
unchanged.

### OpenAI models through OpenRouter

Anything that needs `OPENAI_API_KEY` also accepts an
[OpenRouter](https://openrouter.ai) key in `OPENROUTER_API_KEY` (or
`OPEN_ROUTER_KEY`). This covers LLM
judges, the OpenAI models in the model registry (`gpt-4o`, `gpt-4o-mini`, ...),
and the OpenAI models that external evals such as tau2 and DeepScholar call.
When both keys are set, the OpenAI key is used.

With only an OpenRouter key, requests go to OpenRouter's OpenAI-compatible
endpoint, and OpenAI model names are sent as OpenRouter slugs (`gpt-4o` becomes
`openai/gpt-4o`). Judge response caches are still keyed by the OpenAI model name,
so existing caches keep working.

Locally, export the key:

```bash
export OPENROUTER_API_KEY=sk-or-...
```

On Beaker, store it as a user-scoped secret. A task that requires
`OPENAI_API_KEY` uses `{beaker-username}_OPENROUTER_API_KEY` when
`{beaker-username}_OPENAI_API_KEY` does not exist:

```bash
beaker secret write --workspace <workspace> {beaker-username}_OPENROUTER_API_KEY sk-or-...
```

External evals run third-party code in a sandbox. There, the OpenRouter key is
exported as both `OPENROUTER_API_KEY` and `OPENAI_API_KEY`, and `OPENAI_BASE_URL`
points the OpenAI SDK at OpenRouter. olmo-eval rewrites the model names it passes
to tau2, DeepScholar and AstaBench, but it cannot rewrite names hardcoded inside a benchmark,
such as AstaBench's scorer models. OpenRouter resolves most bare OpenAI names
(`gpt-4o-mini`, `o3-mini-2025-01-31`), but rejects dated snapshots it does not list,
such as `gpt-4-0613`.

The `openai_agents` harness compacts long agent histories through OpenAI's
`responses.compact` endpoint. OpenRouter does not serve that endpoint, so with
only an OpenRouter key the harness skips compaction, logs a warning and keeps
the full history.

### Secret Environment Overrides

By default, Beaker secrets are mapped using the pattern `{username}_{ENV_VAR}` (e.g., `ai2-tylerm_OPENAI_API_KEY`).
Use `--secret-env` to override this with a custom Beaker secret name:

```bash
# Use a team-shared secret instead of your personal secret
uv run olmo-eval beaker launch -n "eval" -m gpt-4o -t mmlu \
    --secret-env team-openai-key:OPENAI_API_KEY \
    --cluster h100 \
    -w "ai2/olmo-eval-debug" \
    -B "ai2/oe-other"

# Multiple secret overrides
uv run olmo-eval beaker launch -n "eval" -m gpt-4o -t simpleqa:judge \
    --harness dr_tulu \
    --secret-env team-openai-key:OPENAI_API_KEY \
    --secret-env shared-serper-key:SERPER_API_KEY \
    --secret-env shared-s2-key:S2_API_KEY \
    --cluster h100 \
    -w "ai2/olmo-eval-debug" \
    -B "ai2/oe-other"
```

Format: `BEAKER_SECRET_NAME:ENV_VAR_NAME`

This is useful for:
- Using team-shared API keys instead of personal secrets
- Testing with different credential sets
- Sharing jobs that use organization-level secrets

### YAML Configuration

For complex or reusable configurations, use YAML config files with the `--config/-f` option.
CLI arguments override values from the config file.

**Basic config file** (`eval_config.yaml`):

```yaml
name: eval-llama3-core
models:
  - llama3.1-8b
tasks:
  - mmlu
  - gsm8k
  - hellaswag
  - arc_challenge

cluster: h100
workspace: ai2/olmo-eval-debug
budget: ai2/oe-other
gpus: 1
priority: normal
timeout: 24h
```

**Usage**:

```bash
# Run from config file
uv run olmo-eval beaker launch -f eval_config.yaml --dry-run

# Override specific values
uv run olmo-eval beaker launch -f eval_config.yaml --gpus 4

# Replace the config's model list from the CLI
uv run olmo-eval beaker launch -f eval_config.yaml -m olmo-2-7b
```

**Multi-model comparison config**:

```yaml
name: eval-model-comparison
models:
  - llama3.1-8b
  - olmo-2-7b
  - mistral-7b
tasks:
  - mmlu
  - gsm8k
  - hellaswag
cluster: h100
workspace: ai2/olmo-eval-debug
budget: ai2/oe-other
gpus: 1
```

**Per-task priorities in config** (`examples/beaker/configs/prioritized_tasks.yaml`):

Use `@priority` suffix on tasks to run different tasks at different priority levels.
Tasks with different priorities create separate Beaker experiments:

```yaml
name: eval-prioritized
models:
  - llama3.1-8b
  - olmo-2-7b
tasks:
  # High priority - run first
  - mmlu@high
  - gsm8k@high
  # Normal priority
  - hellaswag@normal
  - arc_challenge@normal
  # Low priority - run when resources available
  - winogrande@low
  - arc_easy@low
cluster: h100
workspace: ai2/olmo-eval-debug
budget: ai2/oe-other
gpus: 1
timeout: 24h
```

This creates **3 experiments** (one per priority level, with both models in each):

```
eval-prioritized-high:   models=[llama3.1-8b, olmo-2-7b], tasks=[mmlu, gsm8k]
eval-prioritized-normal: models=[llama3.1-8b, olmo-2-7b], tasks=[hellaswag, arc_challenge]
eval-prioritized-low:    models=[llama3.1-8b, olmo-2-7b], tasks=[winogrande, arc_easy]
```

**Large model config**:

```yaml
name: eval-70b-full
models:
  - meta-llama/Llama-3.1-70B-Instruct
tasks:
  - mmlu
  - gsm8k
  - hellaswag
cluster: h100
workspace: ai2/olmo-eval-debug
budget: ai2/oe-other
gpus: 4
priority: high
min_runtime: 8h
timeout: 48h
retries: 2
```

**Config file fields**:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | no | Experiment name (generated from models and tasks if omitted) |
| `models` | list of strings | yes | Model names or presets |
| `tasks` | list | yes | Task specs (with optional `@priority`) |
| `cluster` | string | yes | Cluster alias or full name |
| `workspace` | string | yes | Beaker workspace |
| `budget` | string | when the workspace has no bound budget | Beaker budget |
| `gpus` | int | no | GPUs per model instance (auto-detected from the provider if omitted) |
| `max_gpus_per_node` | int | no | Max GPUs per node, splits tasks if exceeded (default: `8`) |
| `priority` | string | no | Default priority (default: `normal`) |
| `min_runtime` | string | no | Minimum runtime before preemption (e.g., `2h`); see [Preemption](#preemption) |
| `preemptible` | bool | no | Deprecated by Beaker; cannot be combined with `min_runtime` |
| `timeout` | string | no | Job timeout (default: `24h`) |
| `retries` | int | no | Retry count on failure |
| `beaker_image` | string | no | Container image to use (config-only) |
| `groups` | list | no | Beaker groups to add experiments to |

CLI options override config values. `-m` and `-t` replace the config's lists rather than
adding to them.

See `examples/beaker/configs/` for more configuration examples.

### Cluster Aliases

```bash
# List clusters with utilization
uv run olmo-eval beaker clusters

# Only show cluster aliases
uv run olmo-eval beaker clusters --aliases
```

### Programmatic API

```python
from olmo_eval.launch import BeakerJobConfig, BeakerLauncher

config = BeakerJobConfig(
    name="eval-llama3-mmlu",
    command=["uv", "run", "olmo-eval", "run", "-m", "llama3.1-8b", "-t", "mmlu"],
    cluster="h100",
    workspace="<workspace>",
    budget="<budget>",  # optional when the workspace has a bound budget
    num_gpus=1,
)

launcher = BeakerLauncher()
experiment = launcher.launch(config)
print(f"Launched: {launcher.beaker.experiment.url(experiment)}")
```

## Docker Image Management

Docker images provide the runtime environment (Python, PyTorch, CUDA) but do NOT include:
- **Source code** - Gantry mounts your git repository at runtime
- **Inference providers** - Installed at job startup from each model's resolved provider config

This approach allows you to:
- Use any git commit without rebuilding images
- Keep images small and cacheable

### Building Images

Images are tagged with CUDA and PyTorch versions: `cu{version}-trc{version}-{arch}`

```bash
# Build with defaults
./scripts/build_image.sh

# Specific CUDA + PyTorch version
./scripts/build_image.sh --cuda-version 12.8.1 --torch-version 2.10.0

# Production build
./scripts/build_image.sh --platform linux/amd64

# See supported CUDA+PyTorch pairs
./scripts/build_image.sh --help
```

**Supported CUDA versions**: 12.6.1, 12.8.0, 12.8.1, 12.9.1
**PyTorch version**: Configurable via `--torch-version`
**Configuration**: See `scripts/build_config.sh`

### Pushing Images

```bash
# Push most recent build
./scripts/beaker/push_beaker_image.sh

# Preview without pushing
./scripts/beaker/push_beaker_image.sh --dry-run
```

The script auto-detects the image name from the tag (e.g., `olmo-eval-cu1281-trc2100-amd64`)

### What's in the Image

The image contains:
- Python 3.12 (via uv)
- PyTorch with CUDA support
- System dependencies (git, uv, ca-certificates)

The image does NOT contain:
- olmo-eval source code (provided by gantry at runtime)
- olmo-eval dependencies like click, datasets, rich, etc. (installed at job startup)
- Storage backends like boto3, psycopg (installed at job startup if needed)
- Inference providers like vllm, transformers, litellm (installed at job startup)

### Installing Inference Providers at Runtime

Inference providers are NOT baked into images. They are installed at job startup from the resolved provider configuration for each model:

```yaml
# In config file
models:
  - llama3.1-8b
  - gpt-4o
```

```bash
# Or force the provider kind via a harness override
uv run olmo-eval beaker launch -n "eval" \
    --harness default \
    -o provider.kind=vllm_server \
    -m llama3.1-8b \
    -t mmlu \
    --cluster h100 \
    -w "ai2/olmo-eval-debug" \
    -B "ai2/oe-other"

# Manual installation inside container
uv pip install -e '.[vllm]'  # includes vllm[runai]
```

For raw OLMo-core checkpoints, force the provider kind through the harness and
use `provider.package` when you need a specific `ai2-olmo-core` install:

```bash
# Pin the PyPI package version
uv run olmo-eval beaker launch -n "eval" \
    --harness default \
    -o provider.kind=olmo_core \
    -o provider.package=2.3.0 \
    -m /weka/path/to/raw-olmo-core-checkpoint \
    -t mmlu \
    --cluster h100 \
    -w "ai2/olmo-eval-debug" \
    -B "ai2/oe-other"

# Override ai2-olmo-core from a GitHub branch
uv run olmo-eval beaker launch -n "eval" \
    --harness default \
    -o provider.kind=olmo_core \
    -o provider.package=https://github.com/allenai/OLMo-core.git@my-branch \
    -m /weka/path/to/raw-olmo-core-checkpoint \
    -t mmlu \
    --cluster h100 \
    -w "ai2/olmo-eval-debug" \
    -B "ai2/oe-other"
```

The OLMo-core source URL is normalized to install the
`ai2-olmo-core[torchao,transformers]` distribution, so branch overrides replace
the bundled OLMo-core dependency instead of installing a separate repo-named
package.

### Task-Specific Dependencies

Tasks can declare runtime dependencies that are installed at job startup (see [Tasks](#tasks)). Dependencies are automatically merged, deduplicated, and installed after the inference provider.

You can also add or override dependencies via the CLI:

```bash
# Add dependencies to a task via -o flag
uv run olmo-eval beaker launch -n "eval" -m llama3.1-8b \
    -t humaneval:3shot:bpb -o 'dependencies=["code-sandbox==1.0", "git+https://github.com/user/repo@v2.0"]' \
    --cluster h100 \
    -w "ai2/olmo-eval-debug" \
    -B "ai2/oe-other"

# Dependencies from multiple tasks are merged
uv run olmo-eval beaker launch -n "eval" -m llama3.1-8b \
    -t humaneval:3shot:bpb -o 'dependencies=["pkg1"]' \
    -t mbpp:3shot:bpb -o 'dependencies=["pkg2"]' \
    --cluster h100 \
    -w "ai2/olmo-eval-debug" \
    -B "ai2/oe-other"
```

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, checks, and PR expectations.
