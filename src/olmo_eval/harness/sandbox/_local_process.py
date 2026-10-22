"""Container-side process supervision for local sandbox commands."""

COMMAND_SUPERVISOR = r"""
import ctypes
import json
import os
import signal
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

record, marker, duration = sys.argv[1:]
if ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) != 0:
    raise OSError(ctypes.get_errno(), "Could not become a child subreaper")
interrupted = False
reading_payload = False

def interrupt(signum, frame):
    global interrupted
    interrupted = True
    if reading_payload:
        raise InterruptedError("Command cancelled while receiving its payload")

signal.signal(signal.SIGUSR1, interrupt)
cleanup_started = False

def clean_children():
    deadline = time.monotonic() + 2.0
    while True:
        try:
            while os.waitpid(-1, os.WNOHANG)[0]:
                pass
        except ChildProcessError:
            Path(record + ".clean").touch()
            return
        children = Path(f"/proc/self/task/{os.getpid()}/children").read_text().split()
        for pid in children:
            try:
                os.kill(int(pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        if time.monotonic() >= deadline:
            raise RuntimeError("Command still has live descendants after cleanup")
        time.sleep(0.025)

try:
    Path(record).write_text(Path("/proc/self/stat").read_text())
    print(marker, flush=True)
    reading_payload = True
    if interrupted:
        raise InterruptedError("Command cancelled before receiving its payload")
    command = json.load(sys.stdin)
    reading_payload = False
    if len(command) != 3 or command[1] != "-c" or command[0] not in ("python3", "bash", "sh"):
        raise ValueError("Unsupported local command payload")
    with tempfile.TemporaryDirectory(prefix="_sandbox_command_") as directory:
        source = Path(directory) / "source"
        source.write_text(command[2], encoding="utf-8")
        if command[0] == "python3":
            wrapper = (
                f"exec(compile(open({str(source)!r}, encoding='utf-8').read(), "
                "'<string>', 'exec'))"
            )
        else:
            wrapper = f'eval "$(cat {shlex.quote(str(source))})"'
        started = time.monotonic()
        child = subprocess.Popen(
            ["timeout", "--kill-after=5s", duration + "s", command[0], "-c", wrapper],
            stdin=subprocess.DEVNULL,
        )
        while child.poll() is None and not interrupted:
            time.sleep(0.025)
        code = child.poll()
        timed_out = code in (-9, 124, 137) and time.monotonic() - started >= float(duration)
        if interrupted or timed_out:
            cleanup_started = True
            clean_children()
        else:
            try:
                while os.waitpid(-1, os.WNOHANG)[0]:
                    pass
            except ChildProcessError:
                Path(record + ".clean").touch()
        code = 124 if interrupted else code
        code = 128 - code if code < 0 else code
        finished = marker.replace("__OLMO_STARTED_", "__OLMO_FINISHED_", 1)
        report = json.dumps({"exit_code": code, "timed_out": timed_out})
        print("\n" + finished + ":" + report, flush=True)
except BaseException:
    if not cleanup_started:
        clean_children()
    raise
sys.exit(code)
"""

COMMAND_CLEANUP = r"""
ready=0
for attempt in {1..20}; do
    if IFS= read -r row 2>/dev/null < "$1"; then ready=1; break; fi
    sleep 0.1
done
[ "$ready" = 1 ] || { echo "Command supervisor record unavailable" >&2; exit 75; }
if [ ! -f "$1.clean" ]; then
    supervisor=${row%% *}
    read -r -a recorded <<< "${row##*) }"
    if ! IFS= read -r leader 2>/dev/null < /proc/$supervisor/stat; then
        echo "Command supervisor exited without confirmed cleanup" >&2
        exit 75
    fi
    read -r -a current <<< "${leader##*) }"
    if [ "${recorded[19]}" != "${current[19]}" ]; then
        echo "Command supervisor identity changed" >&2
        exit 75
    fi
    kill -USR1 -- "$supervisor" 2>/dev/null || true
    for attempt in {1..60}; do
        [ -f "$1.clean" ] && break
        sleep 0.05
    done
fi
if [ ! -f "$1.clean" ]; then
    echo "Command descendant cleanup not confirmed" >&2
    exit 75
fi
rm -f "$1" "$1.clean"
"""
