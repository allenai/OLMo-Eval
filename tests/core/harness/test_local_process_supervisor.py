"""Exercise the container-side supervisor on compatible Linux hosts."""

import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from olmo_eval.harness.sandbox._local_process import COMMAND_SUPERVISOR

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux process supervision")


def _wait_for(path: Path) -> None:
    deadline = time.monotonic() + 3
    while not path.exists():
        assert time.monotonic() < deadline, f"Child did not create {path}"
        time.sleep(0.025)


@pytest.mark.parametrize("interrupt", ["timeout", "cancel"])
def test_supervisor_reaps_escaped_descendants(tmp_path, interrupt):
    record = tmp_path / "supervisor"
    child_file = tmp_path / "child"
    code = (
        "import os,signal,time\n"
        "signal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
        "if os.fork() == 0:\n"
        "    os.setsid()\n"
        "    if os.fork(): os._exit(0)\n"
        f"    open({str(child_file)!r},'w').write(str(os.getpid()))\n"
        "time.sleep(60)\n"
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-I",
            "-c",
            COMMAND_SUPERVISOR,
            str(record),
            "__OLMO_STARTED_test__",
            "0.2" if interrupt == "timeout" else "60",
            sys.executable,
            "-c",
            code,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for(child_file)
        pid = int(child_file.read_text())
        if interrupt == "cancel":
            os.kill(process.pid, signal.SIGUSR1)
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode in (124, 137), stderr
        assert Path(str(record) + ".clean").exists()
        assert not Path(f"/proc/{pid}").exists()
        report = json.loads(stdout.rsplit("__OLMO_FINISHED_test__:", 1)[1])
        assert report["exit_code"] == process.returncode
        assert report["timed_out"] is (interrupt == "timeout")
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=5)
        if child_file.exists():
            with contextlib.suppress(ProcessLookupError):
                os.kill(int(child_file.read_text()), signal.SIGKILL)


def test_supervisor_isolates_imports_and_reports_ordinary_exit(tmp_path):
    (tmp_path / "subprocess.py").write_text("raise RuntimeError('workspace import')")
    record = tmp_path / "supervisor"
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            COMMAND_SUPERVISOR,
            str(record),
            "__OLMO_STARTED_test__",
            "5",
            sys.executable,
            "-c",
            "print('ok'); raise SystemExit(124)",
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 124, result.stderr
    assert result.stdout.startswith("__OLMO_STARTED_test__\nok\n")
    assert Path(str(record) + ".clean").exists()
    report = json.loads(result.stdout.rsplit("__OLMO_FINISHED_test__:", 1)[1])
    assert report == {"exit_code": 124, "timed_out": False}
