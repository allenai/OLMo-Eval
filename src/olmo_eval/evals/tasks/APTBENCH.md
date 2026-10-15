# APTBench

`aptbench_*` implements [APTBench](https://github.com/TencentYoutuResearch/APTBench)
(Tencent Youtu, [arXiv:2510.24397](https://arxiv.org/abs/2510.24397)): *Benchmarking
Agentic Potential of Base LLMs During Pre-Training*.

APTBench is a **base-model** benchmark. Instead of running an agent end to end,
it turns agent trajectories (software-engineering and deep-research runs) into
few-shot completion problems. The model continues a prompt such as
`The correct next step plan is (` and the next few tokens are graded. That makes
it cheap enough to track during pre-training while still predicting downstream
agent performance.

## Files

- [`aptbench.py`](aptbench.py): subtask definitions, answer extractors, the
  `APTBenchTask` base class, and registration of each subtask and its
  `ctx<N>k` variants.
- [`../suites/aptbench.py`](../suites/aptbench.py): per-category and composite
  suites.

Both modules are picked up by task and suite auto-discovery
(`tasks/__init__.py`, `suites/__init__.py`).

## Subtasks

Every task is named `aptbench_<category>_<subtask>`.

| Category | Subtask | N | Asks the model to… | Answer | Max new tokens |
|---|---|---:|---|---|---:|
| `env_setup` | `plan` | 437 | choose a repository setup plan | letter | 10 |
| `env_setup` | `action` | 1084 | write the next setup shell command | command | 30 |
| `env_setup` | `error` | 147 | choose a plan that fixes a setup error | letter | 10 |
| `issue_fix` | `locate` | 283 | choose the buggy code snippet | letter | 10 |
| `issue_fix` | `fix_patch` | 232 | choose the correct fix patch | letter | 10 |
| `issue_fix` | `plan` | 243 | choose the next step in a repair trajectory | letter | 10 |
| `issue_fix` | `action` | 241 | write the next command in a repair trajectory | command | 10 |
| `issue_fix` | `test_patch` | 1060 | choose the test that reproduces an issue | letter | 10 |
| `deepresearch` | `plan_en` / `plan_zh` | 614 / 417 | choose the next search/browse step | letter | 10 |
| `deepresearch` | `summ_ans_en` / `summ_ans_zh` | 212 / 138 | state the answer found in a trajectory | short text | 30 |
| `deepresearch` | `openend_plan_en` | 298 | choose the best report structure | letter | 10 |
| `deepresearch` | `openend_citation_en` / `_zh` | 173 / 189 | pick all statements a web page supports | letter set | 10 |
| `deepresearch` | `openend_quality_en` / `_zh` | 110 / 103 | pick the best of four reports | letter | 10 |
| `tool` | `acebench_api_select`, `bfcl_v4_api_select` | 697, 197 | name the function to call | function name | 30 |
| `tool` | `acebench_api_param`, `bfcl_v4_api_param` | 1538, 565 | write one argument value | JSON value | 30 |
| `agentic_math` | `planning_single` | 1448 | choose the key math capability | letter (A–E) | 10 |
| `agentic_math` | `feedback_tf` | 1198 | judge a solution correct or wrong | `correct`/`wrong` | 10 |
| `agentic_math` | `action_cal` | 1101 | answer a calculation question | letter (A–D) | 10 |

`env_setup` + `issue_fix` form **APTBench-SWE** and `deepresearch` forms
**APTBench-DR**; these are the paper's headline scores. `tool` and
`agentic_math` ship with the upstream repository but its main run script
(`test_tasks_vllm.sh`) does not run them.

### Suites

| Suite | Contents | Aggregation |
|---|---|---|
| `aptbench:<category>` | subtasks of one category | mean over subtasks |
| `aptbench:swe` | `env_setup`, `issue_fix` | mean of category means |
| `aptbench` | `env_setup`, `issue_fix`, `deepresearch` | mean of category means |
| `aptbench:all` | all five categories | mean of category means |

Upstream reports per-subtask numbers only; the suite aggregation is ours.

## How to run

```bash
# Browse
uv run olmo-eval suite inspect aptbench
uv run olmo-eval task inspect aptbench_issue_fix_plan --request

# Smoke test without a model
uv run olmo-eval run -m mock -t aptbench_env_setup_action -o limit=5 --dry-run

# One subtask on a real base model (vLLM server, the default harness)
uv run olmo-eval run -m Qwen/Qwen3-8B-Base -t aptbench_issue_fix_plan:ctx32k

# Headline SWE + DR suite, 32K context
uv run olmo-eval run \
    --harness default -o provider.max_model_len=32768 \
    -m Qwen/Qwen3-8B-Base \
    -t aptbench:ctx32k

# Everything, at a 128K budget (truncates from the start; see "Differences from upstream")
uv run olmo-eval run \
    --harness default -o provider.max_model_len=131072 \
    -m <model> -t aptbench:all:ctx128k
```

Harness overrides (`provider.*`) go right after `--harness`; task overrides
such as `limit=...` go right after `-t`. If the vLLM server is slow to start
(large models or long contexts), raise its startup timeout with
`--harness default -o provider.kwargs.startup_timeout=900`.

### Pick a context variant

Many prompts are long. Median prompt sizes are about 230K characters for
`issue_fix_action`, 150K for `issue_fix_plan` and 200K for
`openend_quality_en`, and the largest `env_setup_plan` prompt is about 1M
characters. A prompt longer than the model's context fails the request.
Choose the `ctx<N>k` variant (`ctx8k`, `ctx32k`, `ctx64k`, `ctx128k`) that
matches the context you serve (`provider.max_model_len`). The variant caps the
prompt at `N*1024 - 64` tokens and drops tokens **from the start**, so the
fixed in-context examples are cut first and the instance being asked about is
kept.

- Truncation is applied by the `vllm_server` provider (the `default` harness).
  The offline `vllm` and `litellm` providers ignore it and log a warning.
- Any budget can be set directly:
  `-t aptbench_issue_fix_action -o truncate_prompt_tokens=60000`.
- Scores are only comparable across models at the same context budget.

## Walkthrough of the implementation

1. **Data.** There is no official HuggingFace mirror, so each subtask reads
   `data/<category>/<subtask>/input_data.jsonl` from the upstream GitHub repo
   at a pinned commit (`APTBENCH_REVISION`). It uses the same
   `DataSource(path="json", data_files=<url>)` pattern as other tasks here,
   which goes through `datasets` and its cache. `split = Split.TRAIN` because
   JSON data files load as a single `train` split.
2. **Prompts.** The upstream template for each subtask
   (`code/prompts/*.txt`, from the same commit) already contains its
   in-context examples, so `num_fewshot` is fixed at 0. `split_template` cuts
   each template at the paragraph holding the first placeholder:
   - The **prefix** (fixed examples) is prepended in `format_request`.
   - The **instance section** has its `$PLACEHOLDER$`s filled in `process_doc`
     exactly as upstream does (`str.replace` with the stripped field value)
     and becomes `Instance.question`.

   So `olmo-eval task inspect` shows just the instance, and the full prompt is
   byte-identical to upstream. This was checked on every row of every
   subtask.
3. **Generation.** `RequestType.COMPLETION`, greedy (`temperature=0`), no
   stop sequences, and upstream's `max_new_tokens` as `max_tokens`.
4. **Extraction.** `extract_answer` strips the output, applies the ported
   upstream extractor, and strips the result. The extractors are:
   letter before `)`/newline, A–E / A–D choice, `correct`/`wrong`, first line
   cut at `;`, text before `]`, text before `)`. They were
   differential-tested against the upstream functions.
5. **Scoring.** Case-sensitive `ExactMatchScorer`; citation subtasks use
   `LabelSetMatchScorer` (from `olmo_eval.common.scorers`), which compares `A,C`-style label sets, ignoring
   order. The primary metric is `accuracy`, the same number upstream's
   per-category `code/results/<category>/get_results.py` reports (the share of
   lines with `"judge": true`). As upstream does, empty generations are
   left out of both numerator and denominator. Failed requests (no output at
   all) count as incorrect. `empty_generation_rate` reports the share of
   responses that were left out.
6. **Instance IDs.** Upstream `uuid`s are not unique within some subtasks
   (the `*_api_param` tasks have one row per parameter), so `metadata["id"]`
   is `<task>_<row index>`. The upstream `uuid` is kept in metadata.

## Differences from upstream

- **BOS token.** Upstream's offline vLLM tokenizes with the tokenizer's
  defaults, which add BOS for models that use one (e.g. Llama). The
  `vllm_server` provider sends no special tokens unless you pass
  `-o provider.kwargs.add_bos_token=true`.
- **Flex match.** For `summ_ans_*`, upstream also prints a secondary "flex"
  accuracy that ignores surrounding quotes. It is not implemented.
- **ROUGE.** Upstream also prints ROUGE-1/2/L, tokenized with a Llama-3
  tokenizer, for `summ_ans_*`, `*_api_param` and `issue_fix_action`. Only
  exact-match accuracy is implemented.
- **Long prompts.** Upstream keeps the first and last 60K tokens of any prompt
  over 120K tokens, using the evaluated model's tokenizer. Here truncation is
  opt-in and drops tokens from the start (see above).
- **Sampling.** Upstream passes `top_p=0.95, top_k=20` alongside
  `temperature=0`, which is greedy either way.

Inherited upstream quirks, kept for fidelity:

- The command extractor cuts at the first `;`, so gold commands that contain
  one (4/241 in `issue_fix_action`, 3/1084 in `env_setup_action`) can never
  be matched.
- `env_setup_plan` has one duplicated row.

## Parity with the paper

Qwen3-8B-Base, evaluated with `aptbench:ctx32k` (`provider.max_model_len=32768`),
compared with the paper's Qwen3-8B base numbers (Tables 2 and 3; accuracy or
exact match, in %).

| Subtask | Paper | olmo-eval (ctx32k) | Δ |
|---|---:|---:|---:|
| `env_setup_plan` | 76.43 | 78.72 | +2.29 |
| `env_setup_action` (EM) | 35.61 | 35.70 | +0.09 |
| `env_setup_error` | 61.22 | 59.86 | −1.36 |
| `issue_fix_locate` | 25.09 | 25.44 | +0.35 |
| `issue_fix_fix_patch` | 25.43 | 21.55 | −3.88 |
| `issue_fix_plan` | 52.67 | 50.21 | −2.46 |
| `issue_fix_action` (EM) | 27.80 | 21.16 | −6.64 |
| `issue_fix_test_patch` | 28.68 | 28.87 | +0.19 |
| **SWE average** | **41.62** | **40.19** | **−1.43** |
| `deepresearch_plan_en` | 73.94 | 74.43 | +0.49 |
| `deepresearch_plan_zh` | 72.90 | 71.94 | −0.96 |
| `deepresearch_summ_ans_en` (EM) | 31.60 | 31.60 | 0.00 |
| `deepresearch_summ_ans_zh` (EM) | 36.96 | 36.23 | −0.73 |
| `deepresearch_openend_plan_en` | 55.03 | 54.36 | −0.67 |
| `deepresearch_openend_citation_en` | 15.61 | 17.92 | +2.31 |
| `deepresearch_openend_citation_zh` | 8.47 | 6.35 | −2.12 |
| `deepresearch_openend_quality_en` | 39.09 | 37.27 | −1.82 |
| `deepresearch_openend_quality_zh` | 29.13 | 28.16 | −0.97 |

The paper's DR average also includes ROUGE-1 columns, which are not
implemented, so DR is compared per subtask only.

The largest gaps are on subtasks whose prompts exceed 32K tokens. Measured with
the Qwen3 tokenizer, every `issue_fix_action` prompt (median about 76K tokens)
and about two thirds of `issue_fix_fix_patch` prompts are truncated at
`ctx32k`, losing their in-context examples. Upstream leaves almost all of these
prompts whole, since it only truncates prompts over 120K tokens. Comparisons
against the paper on these subtasks need a model and variant with enough
context (e.g. `ctx128k`).
