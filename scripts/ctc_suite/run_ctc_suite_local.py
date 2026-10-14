"""
Run the CTC suite LOCALLY on a Berkeley slurm node (default ``lorax``) instead of Beaker: the same
cells, limits, shards, bins and per-bin ``preflight.py`` -> ``olmo-eval run`` command that
``launch_ctc_suite.py`` would submit, with the checkpoint, scratch and output paths rewritten to
node-local ``/data`` and each bin run as a one-GPU slurm job.

Fallback for when Beaker's H100 queues are full. Everything the jobs touch is on the node's
``/data`` (interpreter, caches, checkpoints, logs, outputs); nothing reads NFS or ``/net``.

    R=scripts/ctc_suite/run_ctc_suite_local.py
    # 0. once per node: stage code + build the venv (CPU slurm job; see setup_env_script())
    python $R setup-env
    # 1. evaluate a checkpoint that a Beaker transfer job put on S3 (model-only distcp +
    #    config.json + transfer_meta.json + _TRANSFER_OK under --s3-root/<item>/): one fetch+verify
    #    job, then one job per bin, at most --max-gpus running at once across ALL runs
    python $R submit --ckpt-item q35-4b-dense-ctc-setA-32k-20260923T1537/step888 \\
        --run-name lengen-local-baseline-step888 --arm full --doc-markers \\
        --rows ctc_nq,ctc_hpqa --policy "r2k-r32k:300,r64k:100,r128k:100" --yarn-rungs ""
    # 2. the table (runs launch_ctc_suite.py --collect on the node, where the outputs are)
    python $R collect --run-name lengen-local-baseline-step888

Launcher flags (``--arm --rows --policy --row-limit --budget-hours --setup-minutes
--full-batch-size --tokenizer --doc-markers --doc-id-style --yarn-rungs --fit-window --rope-yarn
--max-model-len --prefill-chunk-size --eos-token-id --pad-token-id --yarn-old-len``) mean exactly
what they mean in ``launch_ctc_suite.py``; the plan/pack/bin_settings/job_script used are that
module's own functions. A ``--tokenizer`` that is a ``--fetch``'d item is rewritten to its local
path (``--tokenizer-item``).

Concurrency: every bin job is named ``<--slot-prefix><k>`` for k < ``--max-gpus`` and submitted
with ``--dependency=singleton``, so at most one job per slot runs at a time -- a global cap of
``--max-gpus`` GPUs over every run submitted with the same prefix. Each bin goes to the slot with
the least queued estimated work (job comments carry the planner's estimate).

Layout on the node (``--root``, default ``/data/prasann/lengen_eval``)::

    ckpts/<item>/                 fetched checkpoint (+ _VERIFIED once loaded and checked)
    results/<run_name>/jobNN/     olmo-eval -O dir of bin NN (metrics.json, predictions)
    results/<run_name>/launch_manifest.json
    logs/<run_name>/jobNN.log     full slurm log of bin NN (olmo-eval output unfiltered)
    tmp/<run_name>/jobNN/         preflight view (YaRN / dropped-null config) + TMPDIR
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
sys.path.insert(0, HERE)

import launch_ctc_suite as L  # noqa: E402  (the Beaker launcher: plan/pack/job_script)

DEFAULT_ROOT = "/data/prasann/lengen_eval"
DEFAULT_VENV = "/data/prasann/oe-eval-venv-lengen"
DEFAULT_S3 = "s3://ai2-llm/checkpoints/prasanns/_transfer/lengen"
OLMO_CORE_REPO = "/accounts/projects/berkeleynlp/prasann/projects/OLMo-core-lengen"
#: what the Beaker launcher's install puts on top of the image (submit(): transformers /
#: huggingface_hub pins, --olmo-core-no-deps stack) plus the image's own torch 2.9.1 stack and
#: OLMo-core's runtime deps at the image's versions (pip freeze of
#: tylerr/olmo-core-tch291cu128-2025-11-25).
TORCH_PINS = ["torch==2.9.1+cu128", "triton==3.5.1"]
LAUNCHER_PINS = ["transformers==5.8.1", "huggingface_hub==1.12.2"]
NO_DEPS_STACK = [
    "dataclass-extensions==0.5.0",
    "flash-linear-attention==0.4.1",
    "fla-core==0.4.1",
    "bettermap",
]
OLMO_CORE_RUNTIME = [
    "cached_path==1.8.0",
    "safetensors==0.7.0",
    "einops==0.8.1",
    "filelock",
    "requests",
    "packaging",
    "pyyaml",
    "rich",
    "importlib_resources",
    "pandas",
    "awscli",
]


def sh(cmd: list[str] | str, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, shell=isinstance(cmd, str), text=True, capture_output=True, **kw)


def srun_stdin(args, script: str, data: bytes | None = None, minutes: int = 10) -> str:
    """Run ``script`` on the node in a tiny CPU srun, optionally streaming ``data`` to its stdin."""
    argv = [
        "srun",
        f"--partition={args.partition}",
        f"--qos={args.qos}",
        f"--nodelist={args.node}",
        "--cpus-per-task=1",
        "--mem=2G",
        f"--time=00:{minutes:02d}:00",
        "bash",
        "-c",
        script,
    ]
    res = subprocess.run(argv, input=data, capture_output=True)
    out = res.stdout.decode(errors="replace")
    if res.returncode != 0:
        raise SystemExit(f"srun failed ({res.returncode}): {res.stderr.decode()[-2000:]}\n{out}")
    return out


def sbatch(script: str) -> str:
    res = subprocess.run(["sbatch", "--parsable"], input=script, text=True, capture_output=True)
    if res.returncode != 0:
        raise SystemExit(f"sbatch failed: {res.stderr}")
    return res.stdout.strip().split(";")[0]


def header(args, name: str, log: str, gpus: int, cpus: int, mem: str, hours: float) -> str:
    h = int(hours)
    m = int(round((hours - h) * 60))
    lines = [
        "#!/bin/bash",
        f"#SBATCH --partition={args.partition}",
        f"#SBATCH --qos={args.qos}",
        f"#SBATCH --nodelist={args.node}",
        f"#SBATCH --job-name={name}",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --mem={mem}",
        f"#SBATCH --time={h:02d}:{m:02d}:00",
        f"#SBATCH --output={log}",
    ]
    if gpus:
        lines.append(f"#SBATCH --gres=gpu:{args.gpu_type}:{gpus}")
    return "\n".join(lines) + "\n"


def env_block(args) -> str:
    """Node-local env for every job: interpreter, caches, HOME, AWS creds -- all on /data."""
    r, v = args.root, args.venv
    return f"""set -uo pipefail
export HOME={r}/home XDG_CACHE_HOME={r}/cache/xdg HF_HOME={r}/cache/hf
export TRITON_CACHE_DIR={r}/cache/triton TORCHINDUCTOR_CACHE_DIR={r}/cache/inductor
export CUDA_CACHE_PATH={r}/cache/cuda FLASHINFER_CACHE_DIR={r}/cache/flashinfer
export UV_CACHE_DIR={r}/cache/uv UV_PYTHON_INSTALL_DIR={r}/uv_python PIP_CACHE_DIR={r}/cache/pip
export AWS_SHARED_CREDENTIALS_FILE={r}/aws/credentials AWS_CONFIG_FILE={r}/aws/config
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export CUDA_HOME=/usr/local/cuda-12.8
export PATH={v}/bin:$CUDA_HOME/bin:/usr/bin:/bin
unset PYTHONPATH CONDA_PREFIX
mkdir -p $HOME $XDG_CACHE_HOME $HF_HOME $TRITON_CACHE_DIR $TORCHINDUCTOR_CACHE_DIR $CUDA_CACHE_PATH
case "{v}" in /data/*) ;; *) echo "!!! venv not node-local"; exit 1;; esac
"""


# --------------------------------------------------------------------------------------------
# setup-env
# --------------------------------------------------------------------------------------------


def stage(args, repo: str, ref: str, dest_parent: str, name: str) -> str:
    """``git archive`` on the login node streamed into the node's /data (never git on the node)."""
    sha = sh(["git", "-C", repo, "rev-parse", ref]).stdout.strip()
    if not sha:
        raise SystemExit(f"cannot resolve {ref} in {repo}")
    dest = f"{dest_parent}/{name}-{sha[:9]}"
    data = subprocess.run(["git", "-C", repo, "archive", sha], capture_output=True).stdout
    srun_stdin(
        args,
        f"if [ -d {dest} ]; then echo exists; else mkdir -p {dest}.part && tar -x -C {dest}.part "
        f"&& mv {dest}.part {dest}; fi",
        data,
    )
    return dest


def setup_env_script(args, oe_dir: str, oc_dir: str) -> str:
    q = shlex.quote
    v = args.venv
    pins = " ".join(q(p) for p in TORCH_PINS)
    idx = "--index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple"
    fa3 = ""
    if args.fa3_tarball:
        # the Beaker image's own FlashAttention-3 build (flash_attn_3 3.0.0b1, torch 2.9.1 cu128,
        # python 3.12), unpacked into the venv's site-packages: same kernels as the image.
        fa3 = f"""
SP=$({v}/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
AWS_PROFILE=S3 {v}/bin/aws s3 cp {q(args.fa3_tarball)} {args.root}/tmp/fa3_site.tgz --only-show-errors || exit 1
tar -xzf {args.root}/tmp/fa3_site.tgz -C $SP || exit 1
# the image installs it as an egg (importable as flash_attn_interface via easy-install.pth)
ls -d $SP/flash_attn_3-*.egg > $SP/flash_attn_3_egg.pth && cat $SP/flash_attn_3_egg.pth
"""
    return f"""{env_block(args)}
export PATH={args.root}/bin:$PATH
echo "=== setup-env $(date -u +%T) on $(hostname)"
mkdir -p {args.root}/tmp
[ -x {v}/bin/python ] || uv venv --python 3.12 --seed {v}
printf '%s\n' {pins} > {args.root}/tmp/constraints.txt
# torch 2.9.1+cu128 / triton 3.5.1: the Beaker image's stack
uv pip install --python {v}/bin/python {idx} --index-strategy unsafe-best-match {pins} || exit 1
# what the launcher installs on the image: this olmo-eval (non-editable) + pins ...
uv pip install --python {v}/bin/python {idx} --index-strategy unsafe-best-match \\
    -c {args.root}/tmp/constraints.txt {oe_dir} {" ".join(q(p) for p in LAUNCHER_PINS)} || exit 1
# ... and OLMo-core --no-deps on the torch 2.9.1 / fla 0.4.1 stack, plus its runtime deps
uv pip install --python {v}/bin/python --no-deps {oc_dir} || exit 1
uv pip install --python {v}/bin/python {idx} --index-strategy unsafe-best-match \\
    -c {args.root}/tmp/constraints.txt {" ".join(q(p) for p in NO_DEPS_STACK + OLMO_CORE_RUNTIME)} || exit 1
{fa3}
{v}/bin/python - <<'PY'
import importlib, torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
for m in ["triton", "fla", "transformers", "huggingface_hub", "olmo_core", "olmo_eval",
          "flash_attn_3.flash_attn_interface", "flash_attn_interface", "flash_attn"]:
    try:
        mod = importlib.import_module(m)
        print(m, getattr(mod, "__version__", "?"), mod.__file__)
    except Exception as e:
        print(m, "MISSING", type(e).__name__, e)
PY
# warm the HF cache: the default tokenizer (the CTC dataset is read at its pinned revision online)
{v}/bin/python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen3.5-0.8B'); print('tokenizer cached')"
uv pip freeze --python {v}/bin/python > {v}/pip_freeze.txt
echo "=== setup-env done $(date -u +%T)"
"""


def cmd_setup_env(args) -> None:
    oe_dir = stage(args, REPO, "HEAD", f"{args.root}/code", "olmo-eval")
    oc_dir = stage(args, args.olmo_core_repo, args.olmo_core_ref, f"{args.root}/code", "OLMo-core")
    log = f"{args.root}/logs/setup_env_{time.strftime('%m%d%H%M')}.log"
    srun_stdin(args, f"mkdir -p {args.root}/logs {args.root}/bin {args.root}/aws")
    # uv (static binary) and the AWS credentials (profile S3) onto the node, never read over NFS
    home = os.path.expanduser("~")
    for src, dst, mode in (
        (os.path.join(home, ".local/bin/uv"), "bin/uv", "755"),
        (os.path.join(home, ".aws/credentials"), "aws/credentials", "600"),
        (os.path.join(home, ".aws/config"), "aws/config", "600"),
    ):
        with open(src, "rb") as f:
            srun_stdin(
                args,
                f"umask 077; cat > {args.root}/{dst}.part && chmod {mode} {args.root}/{dst}.part "
                f"&& mv {args.root}/{dst}.part {args.root}/{dst}",
                f.read(),
            )
    job = sbatch(
        header(args, "ctclocal-setup-env", log, 0, 16, "64G", 1.0)
        + setup_env_script(args, oe_dir, oc_dir)
    )
    print(f"setup-env job {job} on {args.node}; log {log}\n  olmo-eval {oe_dir}\n  OLMo-core {oc_dir}")


# --------------------------------------------------------------------------------------------
# fetch + verify
# --------------------------------------------------------------------------------------------

VERIFY_PY = r"""
import json, os, sys, time, torch, torch.distributed as dist
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.distributed.checkpoint import load_model_and_optim_state
sys.path.insert(0, os.environ["OE_SCRIPTS"])
from preflight import drop_unknown_null_fields
ck = sys.argv[1]
meta = json.load(open(os.path.join(ck, "transfer_meta.json")))
cfg = json.load(open(os.path.join(ck, "config.json")))
import hashlib
assert hashlib.sha256(open(os.path.join(ck, "config.json"), "rb").read()).hexdigest() == meta["config_sha256"]
print("config.json sha256 matches the weka original; model.rope_reset =", cfg["model"].get("rope_reset", "<absent>"))
os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(29500 + os.getpid() % 1000), RANK="0", WORLD_SIZE="1")
dist.init_process_group("gloo", rank=0, world_size=1)
model_cfg = json.loads(json.dumps(cfg["model"]))
print("dropped:", drop_unknown_null_fields(model_cfg))
def cpu_backend(node):  # CPU job: FA3 needs a GPU to import; weights are backend-independent
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "backend" and v in ("flash_3", "flash_2", "flash_4", "te"):
                node[k] = "torch"
            else:
                cpu_backend(v)
    elif isinstance(node, list):
        for v in node:
            cpu_backend(v)
cpu_backend(model_cfg)
t0 = time.time()
m = TransformerConfig.from_dict(model_cfg).build(init_device="cpu").to(torch.bfloat16)
load_model_and_optim_state(os.path.join(ck, "model_and_optim"), m, strict=True)
sd = m.state_dict()
assert set(sd) == set(meta["tensor_sums"]), set(sd) ^ set(meta["tensor_sums"])
bad = [k for k, v in sd.items() if abs(float(v.double().sum()) - meta["tensor_sums"][k]) > 1e-6 * (1 + abs(meta["tensor_sums"][k]))]
assert not bad, f"tensor sums differ from the weka resave: {bad[:5]}"
print(f"VERIFIED {ck}: strict load of {len(sd)} tensors in {time.time()-t0:.0f}s, sums match")
"""


def fetch_script(args, items: list[str], oe_dir: str) -> str:
    parts = [env_block(args), f"export OE_SCRIPTS={oe_dir}/scripts/ctc_suite"]
    for item in items:
        local = f"{args.root}/ckpts/{item}"
        s3 = f"{args.s3_root}/{item}/"
        is_ckpt = not item.startswith("hf_tokenizers/")
        wait = (
            f"  for i in $(seq 1 {int(args.fetch_wait_minutes)}); do\n"
            f"    AWS_PROFILE=S3 aws s3 ls {s3}_TRANSFER_OK >/dev/null 2>&1 && break\n"
            f'    echo "waiting for {s3}_TRANSFER_OK ($i min)"; sleep 60\n  done\n'
            if is_ckpt
            else ""
        )
        verify = (
            f"python - {local} <<'PY' || exit 1\n{VERIFY_PY}\nPY\n" if is_ckpt else ""
        )
        parts.append(
            f"""
echo "=== fetch {item} $(date -u +%T)"
if [ -f {local}/_VERIFIED ]; then echo "already verified"; else
{wait}  mkdir -p {local}
  AWS_PROFILE=S3 aws s3 sync {s3} {local}/ --only-show-errors || exit 1
  {'[ -f ' + local + '/_TRANSFER_OK ] || { echo "!!! no _TRANSFER_OK"; exit 1; }' if is_ckpt else ''}
  du -sh {local}; ls {local}
  {verify}date -u +%FT%TZ > {local}/_VERIFIED
fi"""
        )
    parts.append('echo "=== fetch done $(date -u +%T)"')
    return "\n".join(parts) + "\n"


# --------------------------------------------------------------------------------------------
# submit
# --------------------------------------------------------------------------------------------


def slot_loads(args) -> list[float]:
    """Estimated queued seconds per slot, from my pending/running jobs' comments."""
    loads = [0.0] * args.max_gpus
    out = sh(["squeue", "--me", "-h", "-o", "%j|%k"]).stdout
    for line in out.splitlines():
        name, _, comment = line.partition("|")
        if not name.startswith(args.slot_prefix):
            continue
        try:
            k = int(name[len(args.slot_prefix) :])
            est = float(comment.split("est=")[1].split()[0]) if "est=" in comment else 3600.0
        except (ValueError, IndexError):
            continue
        if k < len(loads):
            loads[k] += est
    return loads


def bin_script(args, cells, i: int, cmd: str, env: dict, log_dir: str, tmp: str, ckpt: str) -> str:
    exports = "\n".join(f"export {k}={shlex.quote(v)}" for k, v in env.items())
    return f"""{env_block(args)}
export TMPDIR={tmp}/tmp
mkdir -p $TMPDIR {tmp}
cd {args.oe_dir}
echo "=== {args.run_name} job{i:02d} on $(hostname) GPU=$CUDA_VISIBLE_DEVICES slurm=$SLURM_JOB_ID $(date -u +%FT%TZ)"
echo "=== cells: {' '.join(c.task + (f'[{c.shard[0]}/{c.shard[1]}]' if c.shard else '') + f'@{c.limit}' for c in cells)}"
python - {ckpt}/config.json <<'PY' || exit 1
# the checkpoint's attention backends, and a FlashAttention-3 kernel call on this GPU: a flash_3
# config cannot build without it (assert_supported), so a run that loads is a run on FA3
import collections, json, sys, torch
def walk(n, acc):
    if isinstance(n, dict):
        for k, v in n.items():
            if k == "backend" and isinstance(v, str):
                acc[v] += 1
            walk(v, acc)
    elif isinstance(n, list):
        for v in n:
            walk(v, acc)
    return acc
print("torch", torch.__version__, torch.cuda.get_device_name(0))
print("config attention backends:", dict(walk(json.load(open(sys.argv[1]))["model"], collections.Counter())))
import flash_attn_interface as fa3
q = torch.randn(1, 512, 8, 128, device="cuda", dtype=torch.bfloat16)
o = fa3.flash_attn_func(q, q, q, causal=True)
o = o[0] if isinstance(o, tuple) else o
ref = torch.nn.functional.scaled_dot_product_attention(*(t.transpose(1, 2) for t in (q, q, q)), is_causal=True).transpose(1, 2)
print(f"FA3 kernel OK on GPU (max |fa3-sdpa| {{(o.float() - ref.float()).abs().max().item():.2e}})")
PY
{exports}
# progress heartbeat: prediction files written so far, every 5 min
( while sleep 300; do echo "--- heartbeat $(date -u +%T): $(find {args.results_dir}/job{i:02d} -type f 2>/dev/null | wc -l) files, \\
$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader -i ${{CUDA_VISIBLE_DEVICES%%,*}} 2>/dev/null)"; done ) &
HB=$!
bash -c {shlex.quote(cmd)}
rc=$?
kill $HB 2>/dev/null
echo "=== job{i:02d} exit rc=$rc $(date -u +%FT%TZ)"
exit $rc
"""


def cmd_submit(args) -> None:
    if not args.ckpt_item:
        raise SystemExit("--ckpt-item is required")
    rows = (
        L.IID_ROWS + L.OOD_ROWS
        if args.rows == "setA"
        else [r.strip() for r in args.rows.split(",") if r.strip()]
    )
    row_limits = dict(L.DEFAULT_ROW_LIMITS)
    for kv in args.row_limit:
        k, v = kv.split("=")
        if int(v) <= 0:
            row_limits.pop(k, None)
        else:
            row_limits[k] = int(v)
    budget_s = args.budget_hours * 3600 - args.setup_minutes * 60
    if args.fit_window:
        args.yarn_rungs = ""
    yarn_rungs = frozenset(r.strip() for r in args.yarn_rungs.split(",") if r.strip())
    cells = L.plan(
        rows, L.parse_policy(args.policy), args.arm, args.full_batch_size, row_limits, yarn_rungs
    )
    bins = L.pack(cells, budget_s)

    # stage this checkout's olmo-eval scripts (the package itself is the venv's install)
    args.oe_dir = stage(args, REPO, "HEAD", f"{args.root}/code", "olmo-eval")
    ckpt = f"{args.root}/ckpts/{args.ckpt_item}"
    fetch_items = [args.ckpt_item] + list(args.fetch)
    if args.tokenizer_item:
        args.tokenizer = f"{args.root}/ckpts/{args.tokenizer_item}"
        fetch_items.append(args.tokenizer_item)
    root = f"{args.root}/results/{args.run_name}"
    args.results_dir = root
    log_dir = f"{args.root}/logs/{args.run_name}"
    exists = srun_stdin(
        args,
        f"mkdir -p {root} {log_dir}; ls {root} | grep -c '^job' || true",
    ).strip()
    if exists not in ("", "0") and not args.allow_existing:
        raise SystemExit(f"{root} already has job dirs ({exists}): pick a fresh --run-name")

    print(
        f"{args.run_name}: arm={args.arm} rows={len(rows)} policy={args.policy} "
        f"row caps={row_limits}\n{len(bins)} bins, "
        f"~{sum(c.cost_s for b in bins for c in b) / 3600:.1f} GPU-h (planner, H100)"
    )
    fetch_job = None
    if not args.dry_run:
        fetch_job = sbatch(
            header(
                args,
                f"ctclocal-fetch-{args.run_name}"[:60],
                f"{log_dir}/fetch.log",
                0,
                8,
                "64G",
                args.fetch_wait_minutes / 60 + 1,
            )
            + fetch_script(args, fetch_items, args.oe_dir)
        )
        print(f"  fetch+verify job {fetch_job}: {', '.join(fetch_items)} -> {args.root}/ckpts/")

    loads = slot_loads(args)
    order = sorted(range(len(bins)), key=lambda i: -sum(c.cost_s for c in bins[i]))
    jobs: dict[int, dict] = {}
    for i in order:
        cells_i = bins[i]
        rope, max_len, chunk = L.bin_settings(args, cells_i)
        cmd, env = L.job_script(
            ckpt,
            cells_i,
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
        tmp = f"{args.root}/tmp/{args.run_name}/job{i:02d}"
        # node-local scratch instead of the Beaker container's /tmp, and the full olmo-eval log
        # instead of the last 80 lines (the command is otherwise byte-identical)
        local_cmd = (
            cmd.replace("/tmp/ckpt_view", f"{tmp}/ckpt_view")
            .replace("/tmp/ctc_fitted", f"{tmp}/ctc_fitted")
            .replace("| grep -v Warning | tail -80;", "| grep --line-buffered -v Warning;")
        )
        assert " /tmp/" not in local_cmd and "=/tmp/" not in local_cmd, local_cmd
        est = sum(c.cost_s for c in cells_i) + args.setup_minutes * 60
        k = min(range(len(loads)), key=lambda s: loads[s])
        loads[k] += est
        hours = max(2.0, args.time_factor * est / 3600)
        script = header(
            args,
            f"{args.slot_prefix}{k}",
            f"{log_dir}/job{i:02d}.log",
            1,
            args.cpus,
            args.mem,
            hours,
        )
        dep = "singleton" + (f",afterok:{fetch_job}" if fetch_job else "")
        script = script.replace(
            "#SBATCH --output",
            f"#SBATCH --dependency={dep}\n#SBATCH --kill-on-invalid-dep=yes\n"
            f"#SBATCH --comment=est={est:.0f} run={args.run_name} job{i:02d}\n"
            f"#SBATCH --requeue\n#SBATCH --output",
        )
        script += bin_script(args, cells_i, i, local_cmd, env, log_dir, tmp, ckpt)
        desc = ", ".join(
            c.task + (f"[{c.shard[0]}/{c.shard[1]}]" if c.shard else "") for c in cells_i
        )
        if args.dry_run:
            if i == order[0]:
                print("---- first bin script ----\n" + script + "--------------------------")
            jid = "-"
        else:
            jid = sbatch(script)
        jobs[i] = {"slurm_job": jid, "slot": k, "est_s": round(est), "rope_yarn": rope,
                   "max_model_len": max_len, "prefill_chunk_size": chunk}
        print(f"  job{i:02d} slurm={jid} slot={k} ~{est / 60:.0f} min bs={cells_i[0].batch_size}"
              f"{' yarn' if rope else ''}  {desc}")

    keep = (
        "arm", "policy", "rows", "row_limit", "tokenizer", "doc_markers", "doc_id_style",
        "fit_window", "rope_yarn", "max_model_len", "prefill_chunk_size", "yarn_rungs",
        "eos_token_id", "pad_token_id", "yarn_old_len", "budget_hours", "setup_minutes",
        "full_batch_size",
    )
    manifest = {k: getattr(args, k) for k in keep}
    manifest.update(
        ckpt=ckpt,
        ckpt_item=args.ckpt_item,
        s3_source=f"{args.s3_root}/{args.ckpt_item}/",
        local=True,
        node=args.node,
        venv=args.venv,
        oe_dir=args.oe_dir,
        olmo_eval_commit=sh(["git", "-C", REPO, "rev-parse", "HEAD"]).stdout.strip(),
        olmo_eval_branch=sh(["git", "-C", REPO, "branch", "--show-current"]).stdout.strip(),
        fetch_job=fetch_job,
        slurm_jobs={f"job{i:02d}": jobs[i] for i in sorted(jobs)},
        submitted=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        jobs=[[c.__dict__ for c in b] for b in bins],
    )
    if args.dry_run:
        print(f"\n{len(bins)} bins planned (dry run) -> {root}")
        return
    srun_stdin(
        args,
        f"cat > {root}/launch_manifest.json && cp {root}/launch_manifest.json {root}/manifest.json",
        json.dumps(manifest, indent=1).encode(),
    )
    print(f"\n{len(bins)} bins submitted -> {root} (logs {log_dir})")
    if args.register:
        ops = "/accounts/projects/berkeleynlp/prasann/bin/ops"
        for i in sorted(jobs):
            j = jobs[i]
            r = sh([ops, "register", "--run-id", f"{args.run_name}-job{i:02d}", "--experiment",
                    args.experiment, "--kind", "eval", "--cluster", "slurm", "--node", args.node,
                    "--job-id", j["slurm_job"], "--expected-minutes",
                    str(int(j["est_s"] / 60) + 10), "--log-path", f"{log_dir}/job{i:02d}.log",
                    "--ckpt-path", ckpt, "--git", REPO]
                   + (["--parent-run", args.parent_run] if args.parent_run else []))
            print(f"  ops register {args.run_name}-job{i:02d}: {(r.stdout or r.stderr).strip()[-80:]}")


def cmd_collect(args) -> None:
    out = srun_stdin(
        args,
        f"{env_block(args)}\ncd $(ls -td {args.root}/code/olmo-eval-* | head -1) && "
        f"python scripts/ctc_suite/launch_ctc_suite.py --collect --run-name {args.run_name} "
        f"--out-root {args.root}/results",
    )
    print(out)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("action", choices=["setup-env", "submit", "collect"])
    # where
    ap.add_argument("--node", default="lorax")
    ap.add_argument("--partition", default="berkeleynlp")
    ap.add_argument("--qos", default="preemptive")
    ap.add_argument("--gpu-type", default="H200")
    ap.add_argument("--max-gpus", type=int, default=5, help="global cap over all runs (slots)")
    ap.add_argument("--slot-prefix", default="ctclocal-slot")
    ap.add_argument("--cpus", type=int, default=12)
    ap.add_argument("--mem", default="160G")
    ap.add_argument("--time-factor", type=float, default=3.0, help="slurm --time = factor x est")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--venv", default=DEFAULT_VENV)
    ap.add_argument("--s3-root", default=DEFAULT_S3)
    # setup-env
    ap.add_argument("--olmo-core-repo", default=OLMO_CORE_REPO)
    ap.add_argument("--olmo-core-ref", default="origin/prasann/lengen")
    ap.add_argument("--fa3-tarball", default=f"{DEFAULT_S3}/image_tch291/fa3_site.tgz")
    # submit
    ap.add_argument("--ckpt-item", help="checkpoint path relative to --s3-root (run/stepN)")
    ap.add_argument("--fetch", action="append", default=[], help="extra S3 items to fetch")
    ap.add_argument("--tokenizer-item", help="S3 item holding the tokenizer (sets --tokenizer)")
    ap.add_argument("--fetch-wait-minutes", type=float, default=240,
                    help="how long the fetch job waits for the item's _TRANSFER_OK on S3")
    ap.add_argument("--run-name")
    ap.add_argument("--allow-existing", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--register", action="store_true", help="ops register every bin job")
    ap.add_argument("--experiment", default="lengen_rope_hash")
    ap.add_argument("--parent-run", default="")
    # launcher flags (same meaning as launch_ctc_suite.py)
    ap.add_argument("--arm", choices=["full", "compressive"], default="compressive")
    ap.add_argument("--rows", default=",".join(L.IID_ROWS))
    ap.add_argument("--policy", default=L.DEFAULT_POLICY)
    ap.add_argument("--row-limit", action="append", default=[])
    ap.add_argument("--budget-hours", type=float, default=1.5)
    ap.add_argument("--setup-minutes", type=float, default=8.0)
    ap.add_argument("--full-batch-size", type=int, default=8)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--doc-markers", action="store_true")
    ap.add_argument("--doc-id-style", choices=["seq", "hash"], default="seq")
    ap.add_argument("--yarn-rungs", default=L.DEFAULT_YARN_RUNGS)
    ap.add_argument("--fit-window", action="store_true")
    ap.add_argument("--rope-yarn", type=float, default=None)
    ap.add_argument("--max-model-len", type=int, default=0)
    ap.add_argument("--prefill-chunk-size", type=int, default=None)
    ap.add_argument("--eos-token-id", type=int, default=248046)
    ap.add_argument("--pad-token-id", type=int, default=248044)
    ap.add_argument("--yarn-old-len", type=int, default=262144)
    args = ap.parse_args()
    if args.action == "setup-env":
        return cmd_setup_env(args)
    if not args.run_name:
        ap.error("--run-name is required")
    if args.action == "collect":
        return cmd_collect(args)
    return cmd_submit(args)


if __name__ == "__main__":
    sys.exit(main())
