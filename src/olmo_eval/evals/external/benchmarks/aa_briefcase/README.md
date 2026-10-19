# AA-Briefcase-Lite

`aa_briefcase_lite` runs the public week of Artificial Analysis's
[AA-Briefcase](https://artificialanalysis.ai/evaluations/aa-briefcase) benchmark. The
scenario is a commercial due-diligence engagement. An agent gets the scenario overview,
a task brief and a pool of 147 source files, and produces one of four deliverables:

| Task | Deliverable | Binary checks |
|---|---|---|
| `w1_t1` | `market_overview.tex` and `.pdf`, a one-page TikZ market map | 11 |
| `w1_t2` | `market_model.xlsx`, a market sizing and transition model | 12 |
| `w1_t3` | `target_assessment.pptx`, an investment-committee deck | 22 |
| `w1_t4` | `briefing.mp4` and `briefing.srt`, a video briefing with subtitles | 10 |

A model judge grades each deliverable against the dataset's binary rubric checks. The
primary metric, `rubric_pass_rate`, is the share of the 55 checks that pass, pooled
over tasks. The run also reports the pass rate per task and per check type
(`accuracy_pass_rate` for the A checks, `critical_insight_pass_rate` for the C checks),
how often the agent finished or abandoned, and with `n_attempts` above 1, the standard
error of the pass rate across attempts.

```bash
# All four tasks on Beaker
uv run olmo-eval beaker launch -E aa_briefcase_lite -m <model> --gpus 1

# One task, three runs
uv run olmo-eval beaker launch -E aa_briefcase_lite -m <model> --gpus 1 \
    -A task_ids=w1_t2 -A n_attempts=3
```

The judge needs `OPENAI_API_KEY`. `olmo-eval external-evals` lists every argument
with its default.

## How a task runs

Each run follows the AA-Briefcase harness as the dataset card describes it.

- **Agent.** The agent runs on the `stirrup` scaffold, which follows
  [Stirrup](https://github.com/ArtificialAnalysis/Stirrup), the harness Artificial
  Analysis uses. The agent has three tools: `code_exec` runs one shell command,
  `finish` submits deliverable paths after checking that they exist, and
  `abandon_task_finish` gives up. Settings:
  - up to 500 turns, with a reminder once 20 remain;
  - 20 minutes per command;
  - the history is summarized once a reply fills 70% of the context window.

  The system and task prompts are the dataset's own, filled in as published.
- **Sandbox.** The agent works in a container built from `sandbox/Dockerfile`, as the
  unprivileged `user` in `/home/user`. The container has the Python, LaTeX,
  LibreOffice, FFmpeg and other tools the task prompt lists. The task's source files
  are mounted read-only at `/home/user/shared` and `/home/user/week`. The image is
  built on the first run on a host, which takes several minutes.
- **Submission.** Only files named in the `finish` call are graded. A deliverable counts
  when a submitted path has its file name. A run that ends any other way, for example
  by abandoning or running out of turns, submits nothing, and all its checks fail.
- **Parsing.** A fresh container from the same image parses the submitted files for the
  judge (`sandbox/parse_submission.py`):
  - LaTeX and subtitles are passed as text.
  - PDFs, slides and workbooks are passed as text and as page images. Workbook cells
    carry both formulas and recalculated values.
  - Videos are passed as their stream metadata and 16 evenly spaced frames.
- **Grading.** Each check is one judge call. It uses the dataset's binary-rubric prompts
  and the parsed files the check names. The judge is GPT-5.5 (`gpt-5.5-2026-04-23`) at
  high reasoning effort, one of the three judges Artificial Analysis samples from for
  rubric checks. It returns a JSON verdict. A check whose files were not submitted fails
  without a call.

The deliverables and the parsed files are kept under `aa_briefcase_lite/<task>-<attempt>/`
in the output directory. Each prediction records the run's completion reason, its finish
call, every check's verdict and reasoning, and the agent's trajectory.

A run fails rather than reporting a score when the sandbox or the model server fails,
or when a check cannot be graded after its retries. The partial metrics are still saved.

## Differences from the leaderboard harness

- **Pairwise checks are not run.** The 8 pairwise checks for analytical quality and
  presentation compare two models' submissions and have no published prompt. The
  leaderboard's Elo scores therefore cannot be reproduced, and the reported score is the
  rubric pass rate alone.
- **One judge.** The leaderboard samples each check's judge from a panel of three (Claude
  Opus 4.8, GPT-5.5 and Gemini 3.1 Pro); this eval uses GPT-5.5 for every check.
- **The judge's input format is not published.** The parsing above follows Artificial
  Analysis's description for GDPval-AA. Video audio is not transcribed, whereas the
  leaderboard sends video to a model that reads it natively.
- **No view_image tool.** Stirrup offers it only to models that accept images.
- **Command results** give stdout and stderr as one `<output>` stream, because the
  sandbox returns them combined.
- **Network access is not blocked.** The prompt tells the agent the sandbox is offline,
  but the container keeps network access.
