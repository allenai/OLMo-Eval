# Result compatibility log

This log records merged changes that alter what an existing task spec or suite measures,
so a result stored before the change can be identified and interpreted. New tasks and new
variants do not need an entry. See
[Changing or retiring an existing spec](../CONTRIBUTING.md#changing-or-retiring-an-existing-spec)
for when a change is allowed in place.

Add an entry in the PR that makes the change, newest first. Each entry states:

- **Affected**: task specs, suites, or shared code paths, including suite membership or
  aggregation changes.
- **Change**: what is measured differently.
- **Identity**: whether the task hash changes. When it does not, results from before and
  after share a hash and can only be told apart by commit or date.
- **Comparability**: which earlier results remain comparable and what replaces the old
  behavior, if anything.

## 2026-10-02: Chat-style `ANSWER:` lines in multiple-choice extraction (#433)

- **Affected**: generation variants that extract an answer letter with the shared MCQ
  extractor (`evals/extract/mcq.py`) or `ANSWER_LINE_PATTERN`, including GPQA, LAB-Bench,
  MedQA, BBQ, and WMDP. Log-likelihood variants are unaffected.
- **Change**: answers formatted with markdown emphasis, parentheses, or the letter on the
  next line (`**Answer:** B`, `Answer:\n\nC`) are now extracted instead of falling through
  to later patterns or scoring as unparsed.
- **Identity**: task hashes are unchanged.
- **Comparability**: earlier scores can differ for models that use those answer formats,
  since a later fallback pattern may have picked a different letter or none. Compare runs
  on the same side of the commit.

## 2026-09-24: MMLU-Pro suites use instance-weighted aggregation (#385)

- **Affected**: suites `mmlu_pro`, `mmlu_pro:bpb`, and `mmlu_pro:cot`.
- **Change**: suite aggregation changed from `AVERAGE` (equal weight per category) to
  `WEIGHTED_AVERAGE` (weighted by instance count). Per-task results are unchanged.
- **Identity**: task hashes are unchanged; only the suite-level number differs.
- **Comparability**: suite averages computed before the change are macro averages and are
  not comparable with later suite averages. Per-category task results remain comparable.

## 2026-09-24: `gsm8k` honors `num_fewshot=0` (#335)

- **Affected**: `gsm8k` runs with `num_fewshot` explicitly set to 0.
- **Change**: a zero-shot override previously still included the full few-shot prompt;
  it now includes no examples. Runs at the default or a positive few-shot count are
  unchanged.
- **Identity**: the stored config already recorded `num_fewshot: 0`, so before and after
  share a hash.
- **Comparability**: earlier "zero-shot" `gsm8k` results were few-shot results and are not
  comparable with later zero-shot runs.
