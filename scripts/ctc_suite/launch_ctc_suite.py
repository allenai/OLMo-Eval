"""
Launch the full CTC suite on Beaker for one olmo-core checkpoint -- full attention or compressive
landmark -- and collect the results into one table.

    # plan only
    python scripts/ctc_suite/launch_ctc_suite.py --ckpt <weka step dir> --arm compressive \\
        --run-name q35-4b-setA-cl --dry-run
    # submit
    python scripts/ctc_suite/launch_ctc_suite.py --ckpt <weka step dir> --arm full \\
        --run-name q35-4b-setA-full
    # after the jobs finish: pool every job (and shard) into one table
    python scripts/ctc_suite/launch_ctc_suite.py --run-name q35-4b-setA-full --collect

What it runs. Every row of the 22-row CTC roster by default (``--rows all`` adds the 2 held-out
OOD rows, ``--rows setA`` is the 12 the setA SFT data is IID with + OOD; both are opt-in, since the
OOD ladders are not on the HF dataset yet)
-- at every rung up to 256k, with eval sizes from ``--policy`` (default
300 per rung at 2k-32k, 100 at 64k/128k/256k), through olmo-eval's native ``olmo_core``
provider with ``CTC_SUITE_PROMPT_FORMAT=chat`` -- the prompt body byte-identical to the CTC SFT
data, inside the model's chat template, so the eval is IID with chat-template SFT checkpoints.

How it packs. Each (row, rung) cell is costed from measured throughput (fast-compressive-landmark
Qwen3.5-4B on one H100: prefill GPU-seconds per rung + ~30 ms per decoded token at batch size 1,
times each row's measured answer length). Cells bigger than the per-job budget are split into
exact shards (``CTC_SUITE_SHARDS``; the shards partition the very rows an unsharded run grades),
then packed into single-GPU jobs so every job fits ``--budget-hours``.

Arms. ``compressive``: batch size 1 (landmark blocks are tied to absolute position, so rows cannot
be padded together). ``full``: batch size ``--full-batch-size`` at <=32k, 1 above (the KV cache of
several 64k+ rows does not fit); its cost uses the same prefill and a batched decode, so the plan is
an estimate until a full-attention run is timed.

The r256k rung runs with YaRN by default. Through the chat + doc-marker prompt path its prompts run
to ~272k tokens, past Qwen3.5's 262,144 positions; scored as-is the provider left-truncates 43-100%
of them (chat header and instruction cut off), which reads as a fake 256k collapse. So every
``--yarn-rungs`` cell (default ``r256k``) is packed into its own jobs, which apply YaRN factor 2 to
the checkpoint's RoPE (``preflight.py``), raise ``max_model_len`` to 524,288 and feed prefill in
131,072-token chunks (a single forward past 262,144 tokens crashes; the chunked path is exact).
Every other rung runs unchanged. ``--yarn-rungs ''`` turns this off; ``--fit-window`` replaces it by
shortening distractor documents instead (the two agreed within noise on every r256k cell of the setA
pair, 2026-10-03).

Requirements. The checkpoint must be an olmo-core step dir (``config.json`` + ``model_and_optim``);
the job installs this olmo-eval checkout's commit (so push the branch first) and OLMo-core from
``--olmo-core-ref`` (default ``prasann/landmark``), which must carry the FLA autotune fix (or every
new prompt length re-tunes kernels for ~25 s) and, for the YaRN rungs, chunked prefill
(``GenerationConfig.prefill_chunk_size``).
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass

WEKA_CKPTS = "/weka/oe-training-default/ai2-llm/checkpoints"


def default_out_root() -> str:
    """``<weka checkpoints>/<your Beaker user>/_olmoeval_ctc`` (``$USER`` without beaker)."""
    user = os.environ.get("USER", "unknown")
    try:
        who = subprocess.run(
            ["beaker", "account", "whoami", "--format", "json"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        data = json.loads(who.stdout)
        user = (data[0] if isinstance(data, list) else data).get("name") or user
    except Exception:
        pass
    return os.path.join(WEKA_CKPTS, user, "_olmoeval_ctc")


IID_ROWS = [
    "ctc_nq",
    "ctc_hpqa",
    "ctc_qdmatch_nq",
    "ctc_outlier",
    "ctc_oolong",
    "ctc_contradiction",
    "ctc_xabsence",
    "ctc_reorder",
    "ctc_rerank",
    "ctc_strmatch",
    "ctc_textgroups",
    "ctc_grouping",
]
OOD_ROWS = ["ctc_contra_fever", "ctc_outlier_review"]

RUNG_ORDER = ["r2k", "r4k", "r8k", "r16k", "r32k", "r64k", "r128k", "r256k"]
#: GPU-seconds of prefill per example (compressive landmark, H100, measured; 256k extrapolated).
PREFILL_S = {
    "r2k": 0.2,
    "r4k": 0.3,
    "r8k": 0.5,
    "r16k": 0.9,
    "r32k": 1.7,
    "r64k": 4.2,
    "r128k": 12.0,
    "r256k": 33.0,
}
DECODE_S_PER_TOKEN = 0.030
#: Median decoded length (Qwen3.5 tokens) of each row's reference answer, measured on the eval rows.
#: Rungs above the last measured one reuse it. OOD rows use their in-distribution twin.
ANSWER_TOKENS = {
    "ctc_nq": [5, 6, 6, 6, 7],
    "ctc_hpqa": [9, 10, 10, 10, 12],
    "ctc_qdmatch_nq": [23, 25, 26, 28, 30],
    "ctc_outlier": [15, 16, 17, 17, 19],
    "ctc_oolong": [4, 4, 4, 3, 3],
    "ctc_contradiction": [25, 25, 29, 31, 31],
    "ctc_xabsence": [15, 16, 16, 17, 18],
    "ctc_reorder": [41, 101, 221, 474],
    "ctc_rerank": [88, 180, 368, 832, 1772],
    "ctc_strmatch": [24, 26, 28, 30, 31],
    "ctc_textgroups": [21, 23, 25, 26, 29],
    "ctc_grouping": [61, 143, 245, 647, 1384],
    "ctc_fiqa": [8, 8, 9, 9, 10],
    "ctc_qdmatch_fiqa": [21, 24, 25, 27, 29],
    "ctc_qdmatch_hpqa": [44, 48, 50, 55, 60],
    "ctc_outlier_amzn": [19, 20, 21, 22, 24],
    "ctc_outlier_fixedm": [15, 16, 17, 17, 19],
    "ctc_absence": [16, 17, 18, 19],
    "ctc_msmarco": [6, 6, 6, 7, 7],
    "ctc_obliq": [25, 26, 26, 30, 32],
    "ctc_niah": [6, 6, 6, 7, 7],
    "ctc_scifact": [5, 5, 6, 6, 6],
}
OOD_TWIN = {"ctc_contra_fever": "ctc_contradiction", "ctc_outlier_review": "ctc_outlier"}
#: rerank is scored on its first 10 distinct ids; 160 tokens hold them (score-identical cap).
RERANK_DECODE_TOKENS = 160
LONG_RUNG_ROWS = 125  # rows shipped at r256k and above
OVERFLOW_256K = {"ctc_hpqa", "ctc_outlier", "ctc_qdmatch_nq", "ctc_rerank"}
DEFAULT_POLICY = "r2k-r32k:300,r64k:100,r128k:100,r256k:100"
#: Per-row caps on top of the policy, for the rows whose answers list every document (grouping) or
#: every passage (reorder): hundreds to ~1,400 decoded tokens each at bs=1, the bulk of the cost.
DEFAULT_ROW_LIMITS = {"ctc_grouping": 100, "ctc_reorder": 100}
#: Rungs evaluated with YaRN by default, and the settings their jobs get (see the module docstring).
DEFAULT_YARN_RUNGS = "r256k"
YARN_FACTOR = 2.0
YARN_MAX_MODEL_LEN = 524288
YARN_PREFILL_CHUNK = 131072
BASE_MAX_MODEL_LEN = 262144


@dataclass
class Cell:
    row: str
    subset: str
    rung: str
    limit: int
    cost_s: float
    batch_size: int
    shard: tuple | None = None
    #: evaluated with YaRN (its own jobs). Absent from manifests written before this existed, so
    #: resubmitting an older run reproduces it unchanged.
    yarn: bool = False

    @property
    def task(self) -> str:
        return f"{self.row}:{self.rung}"


def rows_without_data(rows: list[str]) -> list[str]:
    """The rows whose subset has no parquet directory in :data:`HF_DATASET`."""
    from huggingface_hub import HfFileSystem

    from olmo_eval.evals.tasks.ctc_suite import HF_DATASET, OOD_ROSTER, ROSTER

    listed = HfFileSystem().ls(f"datasets/{HF_DATASET}/data", detail=False)
    subsets = {path.rstrip("/").split("/")[-1] for path in listed}
    return [r for r in rows if (ROSTER.get(r) or OOD_ROSTER[r]).subset not in subsets]


def parse_policy(policy: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for part in policy.split(","):
        rungs, n = part.split(":")
        if "-" in rungs:
            lo, hi = rungs.split("-")
            for r in RUNG_ORDER[RUNG_ORDER.index(lo) : RUNG_ORDER.index(hi) + 1]:
                out[r] = int(n)
        else:
            out[rungs] = int(n)
    return out


def plan(
    rows: list[str],
    policy: dict[str, int],
    arm: str,
    full_bs: int,
    row_limits: dict[str, int] | None = None,
    yarn_rungs: frozenset[str] = frozenset(),
) -> list[Cell]:
    from olmo_eval.evals.tasks.ctc_suite import OOD_ROSTER, ROSTER

    cells = []
    for name in rows:
        row = ROSTER.get(name) or OOD_ROSTER[name]
        answers = ANSWER_TOKENS.get(name) or ANSWER_TOKENS.get(OOD_TWIN.get(name, ""), [32])
        for rung in row.rungs:
            if rung not in policy:
                continue
            available = row.eval_size.get(rung, LONG_RUNG_ROWS if rung == "r256k" else 500)
            limit = min(policy[rung], available, (row_limits or {}).get(name, 10**9))
            i = RUNG_ORDER.index(rung)
            ans = answers[min(i, len(answers) - 1)]
            if name == "ctc_rerank":
                ans = min(ans, RERANK_DECODE_TOKENS)
            bs = (
                full_bs
                if arm == "full" and RUNG_ORDER.index(rung) <= RUNG_ORDER.index("r32k")
                else 1
            )
            # batched decode is ~bs/2 x faster per example on an H100 at these sizes (estimate)
            decode = DECODE_S_PER_TOKEN * ans / max(1.0, bs / 2)
            cells.append(
                Cell(
                    name,
                    row.subset,
                    rung,
                    limit,
                    limit * (PREFILL_S[rung] + decode),
                    bs,
                    yarn=rung in yarn_rungs,
                )
            )
    return cells


def pack(cells: list[Cell], budget_s: float) -> list[list[Cell]]:
    """
    Exact-shard cells over budget, then pack into single-GPU bins (one batch size, and YaRN either
    on or off, per bin).
    """
    pieces = []
    for c in cells:
        n = max(1, math.ceil(c.cost_s / budget_s))
        pieces += (
            [c]
            if n == 1
            else [
                Cell(c.row, c.subset, c.rung, c.limit, c.cost_s / n, c.batch_size, (i, n), c.yarn)
                for i in range(n)
            ]
        )
    pieces.sort(key=lambda c: -c.cost_s)
    bins: list[list[Cell]] = []
    loads: list[float] = []
    for p in pieces:
        ok = [
            i
            for i, b in enumerate(bins)
            if loads[i] + p.cost_s <= budget_s
            and b[0].batch_size == p.batch_size
            and b[0].yarn == p.yarn
            and all(c.task != p.task for c in b)
        ]
        if ok:
            j = min(ok, key=lambda i: loads[i])
            bins[j].append(p)
            loads[j] += p.cost_s
        else:
            bins.append([p])
            loads.append(p.cost_s)
    return bins


def bin_settings(args, cells: list[Cell]) -> tuple[float, int, int]:
    """
    The ``(rope_yarn, max_model_len, prefill_chunk_size)`` a bin runs with: an explicit
    ``--rope-yarn`` / ``--max-model-len`` / ``--prefill-chunk-size`` applies to every bin; otherwise
    YaRN bins get the YaRN settings and all others the base window.
    """
    yarn = cells[0].yarn
    rope = args.rope_yarn if args.rope_yarn is not None else (YARN_FACTOR if yarn else 0.0)
    max_len = args.max_model_len or (YARN_MAX_MODEL_LEN if yarn else BASE_MAX_MODEL_LEN)
    chunk = (
        args.prefill_chunk_size
        if args.prefill_chunk_size is not None
        else (YARN_PREFILL_CHUNK if yarn else 0)
    )
    return rope, max_len, chunk


def job_script(
    ckpt: str,
    cells: list[Cell],
    out_dir: str,
    tokenizer: str,
    doc_markers: bool = False,
    fit_window: bool = False,
    rope_yarn: float = 0.0,
    max_model_len: int = 262144,
    prefill_chunk_size: int = 0,
    doc_id_style: str = "seq",
    eos_token_id: int = 248046,
    pad_token_id: int = 248044,
    yarn_old_len: int = 262144,
) -> tuple[str, dict]:
    shards = {f"{c.subset}:{c.rung}": f"{c.shard[0]}/{c.shard[1]}" for c in cells if c.shard}
    env = {
        "CTC_SUITE_PROMPT_FORMAT": "chat",
        "CTC_SUITE_RERANK_DECODE_TOKENS": str(RERANK_DECODE_TOKENS),
    }
    if doc_markers:
        env["CTC_SUITE_DOC_MARKERS"] = "1"
    if doc_id_style != "seq":
        env["CTC_SUITE_DOC_ID_STYLE"] = doc_id_style
    if shards:
        env["CTC_SUITE_SHARDS"] = json.dumps(shards)
    tasks = " ".join(f"-t {c.task} -o limit={c.limit}" for c in cells)
    yarn = f"--rope-yarn {rope_yarn} --yarn-old-len {yarn_old_len} " if rope_yarn else ""
    chunk = (
        f"-o provider.kwargs.prefill_chunk_size={prefill_chunk_size} " if prefill_chunk_size else ""
    )
    # fit_to_window.py writes EVERY cell of this job (rows that already fit unchanged): the data
    # root it sets applies to all of them
    fit = (
        f"python scripts/ctc_suite/fit_to_window.py --cells {' '.join(c.task for c in cells)} "
        f"--out /tmp/ctc_fitted --tokenizer {tokenizer} --max-model-len {max_model_len} "
        "|| exit 1; export CTC_SUITE_DATA_ROOT=/tmp/ctc_fitted; "
        if fit_window
        else ""
    )
    # preflight.py stops the job before the model loads if the installed OLMo-core cannot build the
    # checkpoint's config (dropping only null fields it does not know) or if the tokenizer's chat
    # template would render prompts that are not a token prefix of the SFT sequences.
    cmd = (
        f"set -uo pipefail; {fit}CKPT=$(python scripts/ctc_suite/preflight.py --ckpt {ckpt} "
        f"--tokenizer {tokenizer} --view /tmp/ckpt_view {yarn}| tail -1) || exit 1; "
        f'olmo-eval run -m "$CKPT" {tasks} -H default '
        f"-o provider.kind=olmo_core -o provider.tokenizer={tokenizer} "
        f"-o provider.max_model_len={max_model_len} "
        f"-o provider.kwargs.batch_size={cells[0].batch_size} {chunk}"
        # our checkpoints' config.json has no dataset.tokenizer, so name everything explicitly
        f"-o provider.kwargs.eos_token_id={eos_token_id} "
        f"-o provider.kwargs.pad_token_id={pad_token_id} "
        f"-o provider.kwargs.validate_checkpoint=false "
        f"-o provider.kwargs.allow_tokenizer_fallback=true -O {out_dir} "
        # exit with olmo-eval's status, so Beaker shows a failed job as failed
        f"2>&1 | grep -v Warning | tail -80; rc=${{PIPESTATUS[0]}}; "
        f'echo "=== rc=$rc $(date -u +%T)"; exit $rc'
    )
    return cmd, env


def pack_node_script(group: list[tuple[int, str, dict]]) -> tuple[str, dict]:
    """
    One multi-GPU job running each bin's script on its own GPU, in parallel.

    Each bin keeps its own env (exported in its subshell) and its own /tmp scratch paths; the job
    prints every bin's tail and exits non-zero if any bin failed.
    """
    parts = []
    for gpu, (i, cmd, env) in enumerate(group):
        cmd = cmd.replace("/tmp/ckpt_view", f"/tmp/ckpt_view_{i:02d}").replace(
            "/tmp/ctc_fitted", f"/tmp/ctc_fitted_{i:02d}"
        )
        exports = " ".join(f"export {k}={shlex.quote(v)};" for k, v in env.items())
        sub = f"export CUDA_VISIBLE_DEVICES={gpu}; {exports} {cmd}"
        parts.append(
            f"( bash -c {shlex.quote(sub)} > /tmp/job{i:02d}.log 2>&1; "
            f"echo $? > /tmp/rc_{i:02d} ) &"
        )
    report = " ".join(
        f'echo "=== job{i:02d} rc=$(cat /tmp/rc_{i:02d})"; tail -40 /tmp/job{i:02d}.log; '
        f'[ "$(cat /tmp/rc_{i:02d})" = 0 ] || rc=1;'
        for i, _, _ in group
    )
    return " ".join(parts) + f" wait; rc=0; {report} exit $rc", {}


def submit(args, bins: list[list[Cell]], only: set[int] | None = None) -> None:
    """Submit one job per bin (``only``: just those bin indices, e.g. after a preemption)."""
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, os.pardir, os.pardir))
    branch = subprocess.run(
        ["git", "-C", repo, "branch", "--show-current"], capture_output=True, text=True
    ).stdout.strip()
    # gantry clones THIS repo at its current (pushed) commit; OLMo-core comes from --olmo-core-ref
    # the base image ships an old ai2-olmo-eval that shadows an editable install: remove it and
    # install this checkout non-editably. transformers/huggingface_hub are pinned because the
    # image's huggingface_hub is too old for transformers 5 (`cannot import name 'silent_tqdm'`);
    # the pins are the versions this repo's uv.lock resolves, so a job runs what CI tested (and
    # transformers satisfies the hf extra's >=5.8,<5.13 floor). Bump them together with uv.lock.
    olmo_core_git = f"git+https://github.com/allenai/OLMo-core.git@{args.olmo_core_ref}"
    if args.olmo_core_no_deps:
        # A ref on upstream v3.0.0+ (e.g. prasann/lengen) pins torch>=2.10 (and fla 0.5.2 + torch
        # 2.13 for its ``fla`` extra); resolving it would replace the image's torch 2.9.1 / fla
        # 0.4.1. Install it without dependencies and pin the stack the setA checkpoints trained on.
        olmo_core_install = (
            f"pip install --no-deps 'ai2-olmo-core @ {olmo_core_git}' && pip install "
            "'dataclass-extensions==0.5.0' 'flash-linear-attention==0.4.1' 'fla-core==0.4.1' "
            "bettermap"
        )
    else:
        olmo_core_install = f"pip install 'ai2-olmo-core[fla] @ {olmo_core_git}' dataclass-extensions"
    install = (
        "pip uninstall -y ai2-olmo-eval olmo-eval; pip install . 'transformers==5.8.1' "
        "'huggingface_hub==1.12.2' && " + olmo_core_install
    )
    root = os.path.join(args.out_root, args.run_name)
    failed: list[int] = []
    scripts: list[tuple[int, str, dict]] = []
    for i, cells in enumerate(bins):
        if only is not None and i not in only:
            continue
        rope, max_len, chunk = bin_settings(args, cells)
        cmd, env = job_script(
            args.ckpt,
            cells,
            f"{root}/job{i:02d}",
            args.tokenizer,
            args.doc_markers,
            args.fit_window,
            rope,
            max_len,
            chunk,
            args.doc_id_style,
            args.eos_token_id,
            args.pad_token_id,
            args.yarn_old_len,
        )
        scripts.append((i, cmd, env))
    # --node-gpus N: pack up to N bins into one N-GPU job, one bin per GPU, each writing its own
    # job dir exactly as alone (so --collect is unchanged). Single-GPU urgent jobs can sit for hours
    # on a busy cluster while whole nodes are still granted in minutes.
    groups = (
        [scripts[k : k + args.node_gpus] for k in range(0, len(scripts), args.node_gpus)]
        if args.node_gpus
        else [[x] for x in scripts]
    )
    for group in groups:
        i = group[0][0]
        if len(group) == 1 and not args.node_gpus:
            _, cmd, env = group[0]
            name = f"ctc-{args.run_name}-{i:02d}"
            gpus = 1
        else:
            cmd, env = pack_node_script(group)
            name = f"ctc-{args.run_name}-n{'-'.join(f'{g[0]:02d}' for g in group)}"
            gpus = len(group)
        argv = [
            "gantry",
            "run",
            "--name",
            name[:120],
            "-w",
            args.workspace,
            "-b",
            args.budget,
            "--gpus",
            str(gpus),
            "--priority",
            args.priority,
            "--weka",
            "oe-training-default:/weka/oe-training-default",
            "--beaker-image",
            args.image,
            "--python-manager",
            "conda",
            "--system-python",
            "--install",
            install,
            "--timeout",
            "0",
            "--shared-memory",
            f"{32 * gpus}GiB",
            "--yes",
        ]
        if branch:
            argv += ["--branch", branch]
        # gantry refuses --hostname together with --cluster; the hosts already pin the cluster
        if args.hostname:
            for host in args.hostname:
                argv += ["--hostname", host]
        else:
            argv += ["--cluster", args.cluster]
        for k, v in env.items():
            argv += ["--env", f"{k}={v}"]
        argv += ["--", "bash", "-c", cmd]
        if args.dry_run:
            if i == 0:
                print("\nfirst job:\n  " + " ".join(argv[:-1]) + f" '<{len(cmd)} char script>'")
            continue
        # One at a time, checked: firing every gantry at once in the background (output discarded)
        # silently lost 10 of 24 submissions on 2026-09-28.
        res = subprocess.run(argv, cwd=repo, capture_output=True, text=True)
        url = next((w for w in res.stdout.split() if "beaker.org/ex/" in w), "")
        jobs = ",".join(f"job{g[0]:02d}" for g in group)
        if res.returncode != 0 or not url:
            failed += [g[0] for g in group]
            tail = (res.stderr or res.stdout).strip().splitlines()[-3:]
            print(f"  ⚠ {jobs} NOT submitted (rc={res.returncode}): {' | '.join(tail)}")
        else:
            print(f"  {jobs} {url}")
    if failed:
        print(
            f"\n⚠ {len(failed)} job(s) failed to submit: {failed}. Retry exactly those with "
            f"--run-name {args.run_name} --resubmit --jobs {','.join(map(str, failed))}"
        )
    if only is not None:
        print(f"\nresubmitted {len(only) - len(failed)} job(s) of {sorted(only)} -> {root}")
        return
    keep = (
        "ckpt",
        "arm",
        "policy",
        "rows",
        "row_limit",
        "tokenizer",
        "doc_markers",
        "doc_id_style",
        "fit_window",
        "rope_yarn",
        "max_model_len",
        "prefill_chunk_size",
        "yarn_rungs",
        "olmo_core_ref",
        "olmo_core_no_deps",
        "node_gpus",
        "eos_token_id",
        "pad_token_id",
        "yarn_old_len",
        "cluster",
        "workspace",
        "budget",
        "priority",
        "image",
    )
    manifest = {k: getattr(args, k) for k in keep}
    manifest.update(olmo_eval_branch=branch, jobs=[[c.__dict__ for c in b] for b in bins])
    if not args.dry_run:
        try:
            os.makedirs(root, exist_ok=True)
            path = os.path.join(root, "launch_manifest.json")
            with open(path, "w") as f:
                json.dump(manifest, f, indent=1)
        except OSError as e:  # weka not mounted where we launch from; the jobs are already out
            path = local_manifest_path(args.run_name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                json.dump(manifest, f, indent=1)
            print(f"⚠ could not write the manifest under {root} ({e.strerror}); wrote {path}")
    print(f"\n{len(bins)} jobs {'planned' if args.dry_run else 'submitted'} -> {root}")


def collect(run_name: str, out_root: str) -> None:
    """Pool every job's metrics.json (shards weighted by instance count) into one table."""
    root = os.path.join(out_root, run_name)
    if not os.path.isdir(root):
        raise SystemExit(
            f"{root} is not visible from here (weka not mounted?): run --collect where it is"
        )
    acc: dict[str, dict] = {}
    for path in sorted(glob.glob(os.path.join(root, "job*", "metrics.json"))):
        with open(path) as f:
            tasks = json.load(f).get("tasks", [])
        for t in tasks:
            name, scorer = t["primary_metric"].split(":")
            val, n = t["metrics"][name][scorer], t["num_instances"]
            a = acc.setdefault(t["task"], {"metric": t["primary_metric"], "sum": 0.0, "n": 0})
            a["sum"] += val * n
            a["n"] += n
    rows = []
    for task, a in sorted(acc.items()):
        row, rung = task.split(":")
        v = a["sum"] / a["n"]
        se = math.sqrt(max(v * (1 - v), 0) / a["n"]) if 0 <= v <= 1 else float("nan")
        rows.append(
            {
                "row": row,
                "rung": rung,
                "metric": a["metric"],
                "value": round(v, 4),
                "eval_size": a["n"],
                "se": round(se, 4),
            }
        )
    missing = missing_jobs(root)
    if missing:
        print(
            f"⚠ {len(missing)} job(s) have no metrics.json yet (running, failed or preempted): "
            f"{missing} -- rerun with --resubmit once they are no longer running"
        )
    out = os.path.join(root, "results.json")
    with open(out, "w") as f:
        json.dump(rows, f, indent=1)
    for r in rows:
        flag = (
            f"  ⚠ eval_size={r['eval_size']} only (±{r['se']:.3f})" if r["eval_size"] < 500 else ""
        )
        print(f"{r['row']:<20}{r['rung']:>6}  {r['metric']:<22}{r['value']:.4f}{flag}")
    print(f"\n{len(rows)} cells -> {out}")


def local_manifest_path(run_name: str) -> str:
    """Where the manifest goes when the results root is not writable from the launching machine."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "launches", run_name, "launch_manifest.json")


def manifest_path(root: str) -> str:
    """The run's manifest: under the results root if it is there, else the local fallback."""
    path = os.path.join(root, "launch_manifest.json")
    return path if os.path.exists(path) else local_manifest_path(os.path.basename(root))


def missing_jobs(root: str) -> list[int]:
    """Bin indices from the launch manifest whose job directory has no metrics.json."""
    path = manifest_path(root)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        n = len(json.load(f)["jobs"])
    return [
        i for i in range(n) if not os.path.exists(os.path.join(root, f"job{i:02d}", "metrics.json"))
    ]


def resubmit(args) -> None:
    """Resubmit exactly the bins that produced no metrics.json, with the original settings."""
    root = os.path.join(args.out_root, args.run_name)
    with open(manifest_path(root)) as f:
        manifest = json.load(f)
    for k, v in manifest.items():
        # --hostname is a property of this resubmission (e.g. steering around a bad node)
        if k not in ("jobs", "olmo_eval_branch", "hostname"):
            setattr(args, k, v)
    # Manifests from before per-rung YaRN record no yarn_rungs, and their cells no yarn flag: they
    # resubmit with exactly the window settings they were launched with.
    if "yarn_rungs" not in manifest:
        args.rope_yarn = manifest.get("rope_yarn", 0.0)
        args.max_model_len = manifest.get("max_model_len", BASE_MAX_MODEL_LEN)
        args.prefill_chunk_size = manifest.get("prefill_chunk_size", 0)
    bins = [
        [Cell(**{**c, "shard": tuple(c["shard"]) if c["shard"] else None}) for c in b]
        for b in manifest["jobs"]
    ]
    if args.batch_size:
        # e.g. a bs=8 bin that OOMs at r32k: rerun it smaller (bs=1 bins stay bs=1)
        for b in bins:
            for c in b:
                c.batch_size = min(c.batch_size, args.batch_size)
    if args.jobs:
        missing = sorted(int(j) for j in args.jobs.split(","))
        print(f"resubmitting the named jobs {missing}: check first that none of them is running")
        submit(args, bins, only=set(missing))
        return
    if not os.path.isdir(args.out_root):
        # missing_jobs would report EVERY job as missing and this would resubmit the whole run
        raise SystemExit(
            f"results root {args.out_root} is not visible from here (weka not mounted?): run "
            "--resubmit where it is, or name the jobs with --jobs"
        )
    missing = missing_jobs(root)
    if not missing:
        print(f"every job in {root} has results; nothing to resubmit")
        return
    print(f"resubmitting {missing}: check first that none of them is still running")
    submit(args, bins, only=set(missing))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ckpt", help="olmo-core step dir on weka")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--arm", choices=["full", "compressive"], default="compressive")
    ap.add_argument(
        "--rows",
        default="roster",
        help="'roster' (the 22-row CTC suite; default), 'all' (the 22 + the 2 held-out OOD rows), "
        "'setA' (the 12 rows the setA SFT data is IID with + the 2 OOD rows), or a list. The OOD "
        "ladders are not on the HF dataset yet, so 'all'/'setA' are refused until they are",
    )
    ap.add_argument("--policy", default=DEFAULT_POLICY)
    ap.add_argument(
        "--row-limit",
        action="append",
        default=[],
        metavar="ROW=N",
        help="cap one row's eval size at every rung (repeatable; ROW=0 clears a default). "
        "Defaults: ctc_grouping=100, ctc_reorder=100",
    )
    ap.add_argument("--budget-hours", type=float, default=1.5, help="wall-clock per job")
    ap.add_argument("--setup-minutes", type=float, default=8.0, help="install + model load")
    ap.add_argument("--full-batch-size", type=int, default=8)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument(
        "--doc-markers",
        action="store_true",
        help="wrap each document in <|box_start|>/<|box_end|> (CTC_SUITE_DOC_MARKERS=1), as the "
        "CTC SFT converter's default shards do. Use for checkpoints trained on marker-wrapped "
        "shards (e.g. setA shards_qwen35_256k); leave off for marker-free training data.",
    )
    ap.add_argument(
        "--doc-id-style",
        choices=["seq", "hash"],
        default="seq",
        help="'hash' (CTC_SUITE_DOC_ID_STYLE=hash): show each document as a distinct random "
        "5-digit id seeded from the example, and map the generation's ids back before grading -- "
        "for checkpoints trained on OLMo-core '--doc-id-style hash' shards (e.g. "
        "setA_max20_evaliid_hashid). Default 'seq' = 1..N, every other checkpoint.",
    )
    ap.add_argument(
        "--yarn-rungs",
        default=DEFAULT_YARN_RUNGS,
        help="comma-separated rungs evaluated with YaRN in their own jobs (factor "
        f"{YARN_FACTOR:g}, max_model_len {YARN_MAX_MODEL_LEN}, prefill chunks of "
        f"{YARN_PREFILL_CHUNK}); default '{DEFAULT_YARN_RUNGS}', '' for none. Their prompts exceed "
        "Qwen3.5's 262,144 positions and are otherwise left-truncated",
    )
    ap.add_argument(
        "--fit-window",
        action="store_true",
        help="instead of YaRN, shorten distractor documents so every prompt fits the base window "
        "untruncated (fit_to_window.py; refuses oolong). Turns --yarn-rungs off",
    )
    ap.add_argument(
        "--rope-yarn",
        type=float,
        default=None,
        help="override: YaRN factor for EVERY job (preflight.py); 0 for none. Default: per rung, "
        "from --yarn-rungs",
    )
    ap.add_argument(
        "--max-model-len",
        type=int,
        default=0,
        help=f"override for every job. Default: {BASE_MAX_MODEL_LEN}, or {YARN_MAX_MODEL_LEN} on "
        "YaRN jobs",
    )
    ap.add_argument(
        "--prefill-chunk-size",
        type=int,
        default=None,
        help="override for every job: feed prompts to prefill in slices of this many tokens "
        "(OLMo-core GenerationConfig.prefill_chunk_size); 0 for one shot. Default: "
        f"{YARN_PREFILL_CHUNK} on YaRN jobs, one shot otherwise. A single forward longer than "
        "262,144 tokens crashes on Qwen3.5 (Triton illegal memory access); chunking is exact",
    )
    ap.add_argument("--olmo-core-ref", default="prasann/landmark")
    ap.add_argument(
        "--node-gpus",
        type=int,
        default=0,
        help="pack up to N bins into one N-GPU job (one bin per GPU); 0 = one 1-GPU job per bin",
    )
    ap.add_argument(
        "--eos-token-id",
        type=int,
        default=248046,
        help="generation stop id (default Qwen3.5 <|im_end|>; Qwen3: 151645)",
    )
    ap.add_argument(
        "--pad-token-id",
        type=int,
        default=248044,
        help="padding id (default Qwen3.5 <|endoftext|>; Qwen3: 151643)",
    )
    ap.add_argument(
        "--yarn-old-len",
        type=int,
        default=262144,
        help="the checkpoint's native position window that YaRN extends (default Qwen3.5's 262,144; "
        "Qwen3-4B-Base: 32768)",
    )
    ap.add_argument(
        "--olmo-core-no-deps",
        action="store_true",
        help="install --olmo-core-ref without its dependencies, pinning the image's torch 2.9.1 / "
        "fla 0.4.1 stack (needed for refs on upstream v3.0.0+, e.g. prasann/lengen)",
    )
    ap.add_argument("--cluster", default="ai2/jupiter-cirrascale-2")
    ap.add_argument("--workspace", default="ai2/flex2")
    ap.add_argument("--budget", default="ai2/oe-other")
    ap.add_argument("--priority", default="urgent")
    ap.add_argument("--image", default="tylerr/olmo-core-tch291cu128-2025-11-25")
    ap.add_argument(
        "--out-root",
        default=None,
        help="results root on weka (default: <weka checkpoints>/<your Beaker user>/_olmoeval_ctc)",
    )
    ap.add_argument("--collect", action="store_true", help="pool finished results, submit nothing")
    ap.add_argument(
        "--resubmit",
        action="store_true",
        help="resubmit the jobs of --run-name that have no metrics.json (e.g. preempted), with "
        "the settings recorded in its launch manifest",
    )
    ap.add_argument(
        "--jobs",
        default="",
        help="with --resubmit: resubmit exactly these job indices (e.g. 0,3,7) instead of the ones "
        "with no metrics.json -- for jobs that never submitted, or when weka is not mounted here",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="with --resubmit: cap every resubmitted bin's batch size (e.g. 4 after a bs=8 OOM)",
    )
    ap.add_argument(
        "--hostname",
        action="append",
        default=[],
        help="restrict jobs to these Beaker hostnames (repeatable; gantry --hostname) -- how to "
        "keep jobs off a bad node, since gantry has no exclude",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    args.out_root = args.out_root or default_out_root()

    if args.collect:
        return collect(args.run_name, args.out_root)
    if args.resubmit:
        return resubmit(args)
    if not args.ckpt:
        ap.error("--ckpt is required unless --collect")
    if args.rows == "setA":
        rows = IID_ROWS + OOD_ROWS
    elif args.rows in ("all", "roster"):
        from olmo_eval.evals.tasks.ctc_suite import OOD_ROSTER, ROSTER

        rows = list(ROSTER) + (list(OOD_ROSTER) if args.rows == "all" else [])
    else:
        rows = [r.strip() for r in args.rows.split(",") if r.strip()]

    missing = rows_without_data(rows)
    if missing:
        from olmo_eval.evals.tasks.ctc_suite import OOD_ROSTER

        ood = [r for r in missing if r in OOD_ROSTER]
        why = (
            f" {ood} are the held-out OOD rows, whose ladders are not uploaded yet: they are "
            "opt-in and run only once their data/<subset>/ directories exist."
            if ood
            else ""
        )
        ap.error(
            f"no data in the suite's HF dataset for {missing}: their cells would fail at load (and "
            f"take the rest of their job down with them).{why} Upload the ladders, or leave the "
            "rows out (the default --rows roster, or an explicit --rows list)"
        )

    row_limits = dict(DEFAULT_ROW_LIMITS)
    for kv in args.row_limit:
        k, v = kv.split("=")
        if int(v) <= 0:
            row_limits.pop(k, None)
        else:
            row_limits[k] = int(v)
    budget_s = args.budget_hours * 3600 - args.setup_minutes * 60
    if args.fit_window:
        args.yarn_rungs = ""  # fitting replaces YaRN
    yarn_rungs = frozenset(r.strip() for r in args.yarn_rungs.split(",") if r.strip())
    cells = plan(
        rows, parse_policy(args.policy), args.arm, args.full_batch_size, row_limits, yarn_rungs
    )
    bins = pack(cells, budget_s)
    total = sum(c.cost_s for b in bins for c in b)
    print(
        f"{args.run_name}: arm={args.arm} rows={len(rows)} policy={args.policy} "
        f"row caps={row_limits}"
    )
    print(
        f"{len(bins)} single-GPU jobs, ~{total / 3600:.1f} GPU-h (planner is conservative), "
        f"longest {max(sum(c.cost_s for c in b) for b in bins) / 60:.0f} min + setup"
    )
    for i, b in enumerate(bins):
        print(
            f"  job{i:02d} {sum(c.cost_s for c in b) / 60:4.0f} min bs={b[0].batch_size}"
            f"{' yarn' if b[0].yarn else ''}  "
            + ", ".join(c.task + (f"[{c.shard[0]}/{c.shard[1]}]" if c.shard else "") for c in b)
        )
    risky = sorted(
        {
            c.task
            for b in bins
            for c in b
            if c.rung == "r256k"
            and c.row in OVERFLOW_256K
            and not c.yarn
            and not args.fit_window
            and bin_settings(args, b)[1] <= BASE_MAX_MODEL_LEN
        }
    )
    if risky:
        print(f"  ⚠ left-truncated rows (> 262,144 tokens) in: {', '.join(risky)}")
    submit(args, bins)


if __name__ == "__main__":
    sys.exit(main())
