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

What it runs. Every CTC suite row -- the 22-row roster plus the 2 held-out OOD rows by default
(``--rows roster`` for the 22 only, ``--rows setA`` for the 12 the setA SFT data is IID with + OOD)
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

Requirements. The checkpoint must be an olmo-core step dir (``config.json`` + ``model_and_optim``);
the job installs this olmo-eval checkout's commit (so push the branch first) and OLMo-core from
``--olmo-core-ref`` (which must carry the FLA autotune fix, or every new prompt length re-tunes
kernels for ~25 s). Qwen3.5 has 262,144 positions: some 256k rows of hotpotqa / outlier /
qdmatch_nq / rerank are longer and get left-truncated -- quote those cells with that caveat.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
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


@dataclass
class Cell:
    row: str
    subset: str
    rung: str
    limit: int
    cost_s: float
    batch_size: int
    shard: tuple | None = None

    @property
    def task(self) -> str:
        return f"{self.row}:{self.rung}"


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
                Cell(name, row.subset, rung, limit, limit * (PREFILL_S[rung] + decode), bs)
            )
    return cells


def pack(cells: list[Cell], budget_s: float) -> list[list[Cell]]:
    """Exact-shard cells over budget, then pack into single-GPU bins (one batch size per bin)."""
    pieces = []
    for c in cells:
        n = max(1, math.ceil(c.cost_s / budget_s))
        pieces += (
            [c]
            if n == 1
            else [
                Cell(c.row, c.subset, c.rung, c.limit, c.cost_s / n, c.batch_size, (i, n))
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


def job_script(
    ckpt: str, cells: list[Cell], out_dir: str, tokenizer: str, doc_markers: bool = False
) -> tuple[str, dict]:
    shards = {f"{c.subset}:{c.rung}": f"{c.shard[0]}/{c.shard[1]}" for c in cells if c.shard}
    env = {
        "CTC_SUITE_PROMPT_FORMAT": "chat",
        "CTC_SUITE_RERANK_DECODE_TOKENS": str(RERANK_DECODE_TOKENS),
    }
    if doc_markers:
        env["CTC_SUITE_DOC_MARKERS"] = "1"
    if shards:
        env["CTC_SUITE_SHARDS"] = json.dumps(shards)
    tasks = " ".join(f"-t {c.task} -o limit={c.limit}" for c in cells)
    cmd = (
        f"set -uo pipefail; olmo-eval run -m {ckpt} {tasks} -H default "
        f"-o provider.kind=olmo_core -o provider.tokenizer={tokenizer} "
        f"-o provider.max_model_len=262144 -o provider.kwargs.batch_size={cells[0].batch_size} "
        # our checkpoints' config.json has no dataset.tokenizer, so name everything explicitly
        f"-o provider.kwargs.eos_token_id=248046 -o provider.kwargs.pad_token_id=248044 "
        f"-o provider.kwargs.validate_checkpoint=false "
        f"-o provider.kwargs.allow_tokenizer_fallback=true -O {out_dir} "
        f'2>&1 | grep -v Warning | tail -80; echo "=== rc=${{PIPESTATUS[0]}} $(date -u +%T)"'
    )
    return cmd, env


def submit(args, bins: list[list[Cell]], only: set[int] | None = None) -> None:
    """Submit one job per bin (``only``: just those bin indices, e.g. after a preemption)."""
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, os.pardir, os.pardir))
    branch = subprocess.run(
        ["git", "-C", repo, "branch", "--show-current"], capture_output=True, text=True
    ).stdout.strip()
    # gantry clones THIS repo at its current (pushed) commit; OLMo-core comes from --olmo-core-ref
    # the base image ships an old ai2-olmo-eval that shadows an editable install: remove it and
    # install this checkout non-editably
    install = (
        "pip uninstall -y ai2-olmo-eval olmo-eval; pip install . 'transformers==5.7.0' "
        "'huggingface_hub==1.12.2' && "
        f"pip install 'ai2-olmo-core[fla] @ git+https://github.com/allenai/OLMo-core.git@"
        f"{args.olmo_core_ref}' dataclass-extensions"
    )
    root = os.path.join(args.out_root, args.run_name)
    for i, cells in enumerate(bins):
        if only is not None and i not in only:
            continue
        cmd, env = job_script(
            args.ckpt, cells, f"{root}/job{i:02d}", args.tokenizer, args.doc_markers
        )
        argv = [
            "gantry",
            "run",
            "--name",
            f"ctc-{args.run_name}-{i:02d}"[:120],
            "-w",
            args.workspace,
            "-b",
            args.budget,
            "--cluster",
            args.cluster,
            "--gpus",
            "1",
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
            "32GiB",
            "--yes",
        ]
        if branch:
            argv += ["--branch", branch]
        for k, v in env.items():
            argv += ["--env", f"{k}={v}"]
        argv += ["--", "bash", "-c", cmd]
        if args.dry_run:
            if i == 0:
                print("\nfirst job:\n  " + " ".join(argv[:-1]) + f" '<{len(cmd)} char script>'")
            continue
        subprocess.Popen(
            argv,
            cwd=repo,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    if only is not None:
        print(f"\nresubmitted {len(only)} job(s): {sorted(only)} -> {root}")
        return
    keep = (
        "ckpt",
        "arm",
        "policy",
        "rows",
        "row_limit",
        "tokenizer",
        "doc_markers",
        "olmo_core_ref",
        "cluster",
        "workspace",
        "budget",
        "priority",
        "image",
    )
    manifest = {k: getattr(args, k) for k in keep}
    manifest.update(olmo_eval_branch=branch, jobs=[[c.__dict__ for c in b] for b in bins])
    if not args.dry_run:
        os.makedirs(root, exist_ok=True)
        with open(os.path.join(root, "launch_manifest.json"), "w") as f:
            json.dump(manifest, f, indent=1)
    print(f"\n{len(bins)} jobs {'planned' if args.dry_run else 'submitted'} -> {root}")


def collect(run_name: str, out_root: str) -> None:
    """Pool every job's metrics.json (shards weighted by instance count) into one table."""
    root = os.path.join(out_root, run_name)
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


def missing_jobs(root: str) -> list[int]:
    """Bin indices from the launch manifest whose job directory has no metrics.json."""
    path = os.path.join(root, "launch_manifest.json")
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
    with open(os.path.join(root, "launch_manifest.json")) as f:
        manifest = json.load(f)
    for k, v in manifest.items():
        if k not in ("jobs", "olmo_eval_branch"):
            setattr(args, k, v)
    bins = [
        [Cell(**{**c, "shard": tuple(c["shard"]) if c["shard"] else None}) for c in b]
        for b in manifest["jobs"]
    ]
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
        default="all",
        help="'all' (the 22-row CTC suite + the 2 held-out OOD rows; default), 'roster' (the 22 "
        "only), 'setA' (the 12 rows the setA SFT data is IID with + the 2 OOD rows), or a list",
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
    ap.add_argument("--olmo-core-ref", default="prasann/landmark")
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

    row_limits = dict(DEFAULT_ROW_LIMITS)
    for kv in args.row_limit:
        k, v = kv.split("=")
        if int(v) <= 0:
            row_limits.pop(k, None)
        else:
            row_limits[k] = int(v)
    budget_s = args.budget_hours * 3600 - args.setup_minutes * 60
    cells = plan(rows, parse_policy(args.policy), args.arm, args.full_batch_size, row_limits)
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
            f"  job{i:02d} {sum(c.cost_s for c in b) / 60:4.0f} min bs={b[0].batch_size}  "
            + ", ".join(c.task + (f"[{c.shard[0]}/{c.shard[1]}]" if c.shard else "") for c in b)
        )
    risky = sorted(
        {c.task for b in bins for c in b if c.rung == "r256k" and c.row in OVERFLOW_256K}
    )
    if risky:
        print(f"  ⚠ left-truncated rows (> 262,144 tokens) in: {', '.join(risky)}")
    submit(args, bins)


if __name__ == "__main__":
    sys.exit(main())
