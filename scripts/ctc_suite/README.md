# CTC suite launcher

`launch_ctc_suite.py` evaluates one olmo-core checkpoint on the whole CTC suite on Beaker --
full attention or compressive landmark -- in one command, and pools the results into one table.
The suite itself (tasks, metrics, data, caveats) is described in
[`src/olmo_eval/evals/tasks/ctc_suite/README.md`](../../src/olmo_eval/evals/tasks/ctc_suite/README.md).

```bash
# 1. plan: cells, shards, jobs, estimated GPU-hours -- submits nothing
python scripts/ctc_suite/launch_ctc_suite.py --ckpt <weka step dir> --arm compressive \
    --run-name <name> --dry-run

# 2. submit (one single-GPU Beaker job per bin)
python scripts/ctc_suite/launch_ctc_suite.py --ckpt <weka step dir> --arm compressive --run-name <name>
python scripts/ctc_suite/launch_ctc_suite.py --ckpt <weka step dir> --arm full        --run-name <name>

# 3. after the jobs finish: one table (eval_size + binomial SE, sub-500 cells flagged)
python scripts/ctc_suite/launch_ctc_suite.py --run-name <name> --collect

# jobs are unallocated, so allocated work can preempt them: collect lists jobs with no results,
# and this resubmits exactly those with the settings in the launch manifest
python scripts/ctc_suite/launch_ctc_suite.py --run-name <name> --resubmit
```

## What gets evaluated

| `--rows` | Rows |
|---|---|
| `all` (default) | the 22-row CTC suite + the 2 held-out OOD rows (`ctc_contra_fever`, `ctc_outlier_review`) |
| `roster` | the 22 rows |
| `setA` | the 12 rows the setA SFT data is IID with (nq, hpqa, qdmatch_nq, outlier, oolong, contradiction, xabsence, reorder, rerank, strmatch, textgroups, grouping) + the 2 OOD rows |
| `a,b,c` | an explicit list of row names |

Every rung each row has, up to 256k. Eval sizes per rung come from `--policy` (default
`r2k-r32k:300,r64k:100,r128k:100,r256k:100`), capped by what the rung ships (125 at 256k) and by
`--row-limit ROW=N` (defaults `ctc_grouping=100`, `ctc_reorder=100`: their answers list every
document, up to ~1,400 decoded tokens each; `ROW=0` clears a default).

⚠ 300 and 100 are below the 500-example floor. `--collect` prints each cell's size and binomial SE
next to its value; carry both wherever the number goes (±0.026 at 300, ±0.046 at 100, f1≈0.7).

## How it works

1. **Plan.** One cell per (row, rung). Its cost is `limit × (prefill(rung) + 30 ms × answer
   tokens)` -- prefill GPU-seconds per rung measured on a compressive-landmark Qwen3.5-4B on one
   H100 (0.2 s at 2k … 1.7 s at 32k, 12 s at 128k, ~33 s at 256k extrapolated), answer lengths
   measured from each row's reference answers. The planner has run ~35% above measured wall-clock.
2. **Shard.** A cell bigger than the per-job budget is split with `CTC_SUITE_SHARDS`: the runner's
   seeded `limit` sample is drawn first, then every n-th row is taken, so the shards partition the
   exact rows an unsharded run grades and pool back exactly.
3. **Pack** cells into single-GPU bins that fit `--budget-hours` (default 1.5) minus
   `--setup-minutes` (default 8, install + model load). A bin never holds two shards of one cell or
   two batch sizes.
4. **Submit** one gantry job per bin. Each runs a single `olmo-eval run` over all of its cells, so
   the model loads once:
   - provider: olmo-eval's native `olmo_core` (no HF/vLLM export), `max_model_len` 262,144;
   - `CTC_SUITE_PROMPT_FORMAT=chat`: the prompt body byte-identical to the CTC SFT data, in the
     model's chat template (IID with chat-template SFT checkpoints; not comparable with the
     default `alpaca` numbers);
   - `CTC_SUITE_RERANK_DECODE_TOKENS=160`: rerank is scored on its first 10 distinct ids only, so
     the cap is score-identical;
   - outputs to `<--out-root>/<run-name>/jobNN/` -- by default
     `/weka/oe-training-default/ai2-llm/checkpoints/<your Beaker user>/_olmoeval_ctc/` -- plus
     `launch_manifest.json` with the full plan.
5. **Collect** (run where weka is mounted, with the same `--out-root`) reads every `jobNN/metrics.json`, pools shards by instance count, and writes
   `results.json` (row, rung, metric, value, eval_size, se).

## Arms

| `--arm` | Batch size | Why |
|---|---|---|
| `compressive` | 1 | landmark blocks are tied to absolute position, so rows cannot share a padded batch |
| `full` | `--full-batch-size` (8) at ≤32k, 1 above | several 64k+ KV caches do not fit together |

Planned cost for `--rows all` with the defaults: **compressive ~40 GPU-h in 30 jobs, full ~33 GPU-h
in 26 jobs** (planner estimates; real has been ~35% lower). The 256k rung is the largest single
item (~15 GPU-h across 16 rows) and its per-example cost is still extrapolated.

**Full-arm batching, validated 2026-09-23** on a dense Qwen3.5-4B (32 rows each at r8k): scores
match (nq f1 0.3750 vs 0.3750; grouping pairwise_f1 0.2016 at bs=1 vs 0.1992 at bs=8) and
generations share their first 84% (nq) / 62% (grouping) of text before diverging -- bf16
batch-shape numerics, not a padding bug (that would corrupt outputs from the first token). Not
bitwise-identical, so `--full-batch-size 1` if you need bitwise determinism; bs=8 was ~2-5x faster
(440 s vs 1,378 s wall, the latter including a cold model load).

## Requirements

- **A local olmo-eval environment on this branch** (the planner imports the suite roster):
  `git checkout prasann/ctc-suite-launcher && uv sync`, then prefix commands with `uv run`.
- **Checkpoint:** an olmo-core step dir (`config.json` + `model_and_optim/`) on weka. Its config
  does not need a `dataset.tokenizer`; the launcher passes the tokenizer
  (`--tokenizer`, default `Qwen/Qwen3.5-0.8B`), EOS 248046 and pad 248044 explicitly.
- **This branch pushed.** gantry clones the current branch at its pushed commit; unpushed changes
  are not in the job.
- **`--olmo-core-ref` carrying the FLA autotune fix** (default `prasann/landmark`). Without it every
  new prompt length re-tunes Triton kernels for ~25 s -- ~95% of eval wall-clock on Qwen3.5/GDN.
- **Beaker/gantry** set up locally; defaults are workspace `ai2/flex2`, budget `ai2/oe-other`,
  cluster `ai2/jupiter-cirrascale-2`, priority `urgent`, image
  `tylerr/olmo-core-tch291cu128-2025-11-25`.

## Caveats

- **Left truncation at 256k.** Some r256k rows of `ctc_hpqa`, `ctc_outlier`, `ctc_qdmatch_nq` and
  `ctc_rerank` exceed Qwen3.5's 262,144 positions and are left-truncated by the provider. The plan
  prints the affected cells; quote them with that caveat.
- **Shortcut rows.** `ctc_textgroups` (the shortest document is gold in ~0.55 of rows at 8k-32k)
  and `ctc_strmatch` (only the gold pairs share words) are partly solvable without the task.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `No module named 'olmo_eval.cli'` | the base image ships an old `ai2-olmo-eval` that shadows an editable install; the launcher uninstalls it and installs this checkout non-editably |
| `cannot import name 'silent_tqdm'` | `huggingface_hub` too old; the launcher pins `transformers==5.7.0` / `huggingface_hub==1.12.2` |
| `Invalid OLMo-core checkpoint ... dataset.tokenizer` | pass `validate_checkpoint=false` + `allow_tokenizer_fallback=true` (the launcher does) |
| every example takes ~10-25 s regardless of length | `--olmo-core-ref` lacks the FLA autotune fix |
| a job exits 143 with "preempted by ... allocated workloads are scheduled ahead of unallocated ones" | expected for unallocated jobs; `--resubmit` reruns just the missing ones |
| a score of ~0.4 vs ~0.7 on 25 examples | eval_size 25 is ±0.1; outlier@8k agreed to 0.01 between this path (0.597) and the native OLMo-core driver (0.587) at 500 |
