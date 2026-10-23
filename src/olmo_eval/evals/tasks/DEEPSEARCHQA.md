# DeepSearchQA

`deepsearchqa` implements [DeepSearchQA](https://huggingface.co/datasets/google/deepsearchqa)
(Google DeepMind, [arXiv:2601.20975](https://arxiv.org/abs/2601.20975); Kaggle
starter code: [deepsearchqa-starter-code](https://www.kaggle.com/code/andrewmingwang/deepsearchqa-starter-code),
[leaderboard](https://www.kaggle.com/benchmarks/google/dsqa)) inside olmo-eval.

900 hand-crafted, multi-step information-seeking questions across 17 domains
(Politics, Finance, Science, Health, History, Geography, Media, ...). Each
question is either a **Single Answer** (one entity/value) or a **Set Answer**
(an enumeration or composite answer requiring multiple items); the goal is to
test whether an agent can plan and execute a search process that returns a
*complete and precise* answer set, not just a single plausible fact.

Both registered tasks share the dataset, notebook judge prompt, and verdict
parser. They differ in the response presented to the judge:

- **`deepsearchqa`** requests a `FINAL ANSWER: ...` line and grades the
  extracted answer text.
- **`deepsearchqa_official_judge`** allows free-form answers and grades the
  full response.

## Files

- [`deepsearchqa.py`](deepsearchqa.py) — dataset loading (`DeepSearchQABase`)
  and the `deepsearchqa` task implementation.
- [`deepsearchqa_official_judge.py`](deepsearchqa_official_judge.py) — the
  `deepsearchqa_official_judge` task implementation (imports `DeepSearchQABase`
  from `deepsearchqa.py`).
- [`../../../../tests/evals/tasks/test_deepsearchqa.py`](../../../../tests/evals/tasks/test_deepsearchqa.py)
  and [`../../../../tests/evals/tasks/test_deepsearchqa_official_judge.py`](../../../../tests/evals/tasks/test_deepsearchqa_official_judge.py)
  — unit tests for parsing, scoring, and the registered tasks/variants.

No other files were modified; both tasks are picked up automatically by the
package's task auto-discovery (`src/olmo_eval/evals/tasks/__init__.py`), the
same mechanism every other task in this directory relies on.

## Data

- HuggingFace dataset: `google/deepsearchqa`, config `deepsearchqa`, split
  `eval` (900 rows).
- Columns: `problem` (question), `problem_category` (domain), `answer`, `answer_type`
  (`"Single Answer"` or `"Set Answer"`).
- 4 `Set Answer` rows encode "no items satisfy every constraint" as the
  literal text `None` in the source CSV; HuggingFace's CSV loader coerces
  that to a null value. The shared loader restores the reference text `None`.
  The judge receives that reference and decides whether the model correctly
  states that no items qualify. Blank or missing model responses always score
  zero without calling the judge. A missing `answer` on a `Single Answer` row is
  still treated as malformed and dropped, since that combination shouldn't
  occur in valid data.

## How grading works

**The official grader is published.** The technical report
([PDF](https://storage.googleapis.com/deepmind-media/DeepSearchQA/DeepSearchQA_benchmark_paper.pdf))
grades with Gemini 2.5 Flash, zero-shot, using the exact prompt printed in
Appendix A — it is not an unpublished/proprietary prompt. `deepsearchqa_official_judge`
uses that prompt verbatim; `deepsearchqa` uses a different, independently
written mechanic (below). Both deviate from the official leaderboard in the
same unavoidable way: olmo-eval's judge infrastructure (`build_openai_judge_fn`)
is OpenAI-only, so both grade with an OpenAI model instead of Gemini (default
`gpt-5.5:medium`, overridable — see below), so absolute numbers from either
task are not directly comparable to the public Kaggle leaderboard.

### `deepsearchqa`: discrete extraction + index matching

The model is prompted to end its response with:

```
FINAL ANSWER: <item 1>, <item 2>, ...
```

Only the text after the last `FINAL ANSWER` marker is graded (the rest of the
response — search steps, reasoning, etc. — is not scored directly). That line
is split into a predicted item set `S` and compared against the gold item set
`G` with an LLM judge that decides semantic equivalence per item (so
"NZ" ≟ "New Zealand" counts as a match), returning two lists of matched
indices (one per side). From those matched counts the task computes, per
instance:

| Metric                    | Formula                    |
|----------------------------|----------------------------|
| `deepsearchqa_precision`   | `\|S ∩ G\| / \|S\|`        |
| `deepsearchqa_recall`      | `\|S ∩ G\| / \|G\|`        |
| `deepsearchqa_f1`          | harmonic mean of precision/recall (**primary metric**) |
| `deepsearchqa_exact_match` | `1.0` iff the item sets match exactly, else `0.0` |

This mechanic is easier to validate programmatically than the official one:
checking a judge-returned index is valid is just `0 <= i < N` for an `N` you
already know, with no need to reconcile the judge's own text against your
source data. The trade-off is that it requires the model to produce a
specific output format, and it's a bigger departure from the paper's
free-form grading than "different judge model" alone.

### `deepsearchqa_official_judge`: free-form response, official prompt

No special output format is required — the model just answers the question,
and the judge grades the full raw response directly (verbatim Appendix A
prompt). The judge returns, per gold item, a boolean ("was this item found in
the response") plus a free list of "Excessive Answers" (items claimed that
aren't part of the gold answer). The task counts how many gold items were
marked found (capped at the gold count) and how many excessive answers were
listed, then computes the same precision/recall/F1/exact-match formulas under
the `deepsearchqa_official_*` metric names.

Note this task does **not** attempt to reconcile the judge's per-item keys
back to the exact `gold_items` strings (an LLM won't reliably echo them
byte-for-byte) — it only counts `True` values, which is enough to compute the
aggregate metrics without needing per-item traceability. If you need to know
*which* specific gold item was missed, use `deepsearchqa`'s index-matching
metadata instead.

This benchmark is about live information-seeking rather than parametric
knowledge, so it is intended to be evaluated with search tools attached via
olmo-eval's Harness abstraction — running it without tools mostly measures
how much of the answer the model already knows.

## Running it

```bash
# Requires an OpenAI key for the LLM judge (always needed, regardless of harness)
export OPENAI_API_KEY=...

# --harness dr_tulu additionally requires its own search-tool keys
export S2_API_KEY=...       # Semantic Scholar
export SERPER_API_KEY=...   # Google web search

# Full 900-instance run with search tools (the intended way to run this eval)
uv run olmo-eval run -m llama3.1-8b -t deepsearchqa --harness dr_tulu

# Baseline, no search tools (parametric-knowledge reference point only)
uv run olmo-eval run -m llama3.1-8b -t deepsearchqa

# Quick iteration on a 50-instance subset
uv run olmo-eval run -m llama3.1-8b -t deepsearchqa:mini --harness dr_tulu

# The official-prompt variant works the same way; swap the task name
uv run olmo-eval run -m llama3.1-8b -t deepsearchqa_official_judge:mini --harness dr_tulu

# Preview instances/prompts without calling a model or the judge
uv run olmo-eval task inspect deepsearchqa -n 3 --request

# Preview the run config without executing anything
uv run olmo-eval run -m mock -t deepsearchqa --dry-run
```

Use a different judge model with `OLMO_EVAL_JUDGE`, e.g.:

```bash
OLMO_EVAL_JUDGE=gpt-5-mini uv run olmo-eval run -m llama3.1-8b -t deepsearchqa --harness dr_tulu
```

See the main [README's Harness section](../../../../README.md#harness) for
configuring which search tools `--harness` attaches, and [Adding New
Tasks](../../../../README.md#adding-new-tasks) for how this task fits the
general task framework.
