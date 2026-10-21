"""Grade one LiveCodeBench solution against a problem's test cases.

Runs inside a sandbox, not in the harness process. Reads ``problem.json`` and
``solution.py`` from its own directory and prints a single JSON verdict to
stdout.

The solution runs in a forked child, as it does in the reference harness, so
that a solution which crashes the interpreter or exhausts its time budget is
reported as a failed solution rather than leaving no verdict behind. Only a
problem whose test cases cannot be decoded produces a verdict with no tests,
which the harness treats as a grading failure rather than a score.

Problems come in two shapes. A problem with a function name expects the
solution to define that function (on a ``Solution`` class for LeetCode
problems); its inputs and outputs are JSON values compared directly. A problem
without one is a whole program: inputs are fed to stdin and the captured
stdout is compared line by line, falling back to decimal comparison so that
numerically equal output is not rejected over formatting.

Ported from the reference implementation:
https://github.com/LiveCodeBench/LiveCodeBench/blob/main/lcb_runner/evaluation/testing_util.py
"""

import ast
import base64
import faulthandler
import json
import os
import pickle
import select
import signal
import sys
import time
import zlib
from decimal import Decimal
from io import StringIO
from types import ModuleType
from typing import Any
from unittest.mock import mock_open, patch

# Prelude the reference harness prepends to every solution, so that solutions
# relying on names from a bare `from x import *` still resolve.
IMPORT_PRELUDE = (
    "from string import *\nfrom re import *\nfrom datetime import *\n"
    "from collections import *\nfrom heapq import *\nfrom bisect import *\n"
    "from copy import *\nfrom math import *\nfrom random import *\n"
    "from statistics import *\nfrom itertools import *\nfrom functools import *\n"
    "from operator import *\nfrom io import *\nfrom sys import *\nfrom json import *\n"
    "from builtins import *\nfrom typing import *\nimport string\nimport re\n"
    "import datetime\nimport collections\nimport heapq\nimport bisect\nimport copy\n"
    "import math\nimport random\nimport statistics\nimport itertools\n"
    "import functools\nimport operator\nimport io\nimport sys\nimport json\n"
    "sys.setrecursionlimit(50000)\n"
)


class TimeoutException(Exception):
    """Raised by the alarm handler when a test case runs too long."""


def timeout_handler(signum, frame):
    raise TimeoutException


def start_alarm(seconds):
    """Arm the per-case timer. Unlike ``signal.alarm`` this keeps fractions of a second."""
    signal.setitimer(signal.ITIMER_REAL, seconds)


def cancel_alarm():
    signal.setitimer(signal.ITIMER_REAL, 0)


class CapturedOutput(StringIO):
    """Stdout stand-in that survives a solution closing it."""

    def close(self):
        pass


class Capturing(list):
    """Collect everything a block writes to stdout."""

    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = CapturedOutput()
        return self

    def __exit__(self, *args):
        self.append(self._stringio.getvalue())
        del self._stringio
        sys.stdout = self._stdout


class MockBuffer:
    """Byte-oriented view of the stdin stand-in."""

    def __init__(self, inputs):
        self.lines = inputs.split("\n")
        self.index = 0

    def read(self, *args):
        return "\n".join(self.lines).encode()

    def readline(self, *args):
        if self.index >= len(self.lines):
            return b""
        line = self.lines[self.index]
        self.index += 1
        return (line + "\n").encode()


class MockStdin:
    """Stand-in for sys.stdin that also exposes a .buffer."""

    def __init__(self, inputs):
        self.stringio = StringIO(inputs)
        self.buffer = MockBuffer(inputs)

    def read(self, *args):
        return self.stringio.read()

    def readline(self, *args):
        return self.stringio.readline()

    def readlines(self, *args):
        return self.stringio.readlines()

    def __getattr__(self, name):
        return getattr(self.stringio, name)


def clean_if_name(code):
    """Drop an ``if __name__ == '__main__'`` guard, keeping its body."""
    try:
        tree = ast.parse(code)
        last_block = tree.body[-1]
        if isinstance(last_block, ast.If):
            condition = last_block.test
            if ast.unparse(condition).strip() == "__name__ == '__main__'":
                code = ast.unparse(tree.body[:-1]) + "\n" + ast.unparse(last_block.body)
    except Exception:
        pass
    return code


def make_function(code):
    """Wrap a whole program in a function so it can be called per test case."""
    try:
        imports = []
        body = []
        tree = ast.parse(code)
        for stmt in tree.body:
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                imports.append(stmt)
            else:
                body.append(stmt)

        wrapper = ast.parse("def wrapped_function():\n    pass\n")
        function = wrapper.body[0]
        if not isinstance(function, ast.FunctionDef):
            return code
        function.body = body or [ast.Pass()]
        ast.fix_missing_locations(wrapper)

        import_source = "\n".join(ast.unparse(stmt) for stmt in imports)
        return IMPORT_PRELUDE + "\n" + import_source + "\n" + ast.unparse(wrapper)
    except Exception:
        return code


def call_method(method, inputs):
    """Call a wrapped program with `inputs` standing in for stdin."""
    if isinstance(inputs, list):
        inputs = "\n".join(inputs)

    line_iterator = iter(inputs.split("\n"))
    mock_stdin = MockStdin(inputs)

    @patch("builtins.open", mock_open(read_data=inputs))
    @patch("sys.stdin", mock_stdin)
    @patch("sys.stdin.readline", lambda *args: next(line_iterator))
    @patch("sys.stdin.readlines", lambda *args: inputs.split("\n"))
    @patch("sys.stdin.read", lambda *args: inputs)
    def _inner(_method):
        try:
            return _method()
        except SystemExit:
            pass

    return _inner(method)


class CompilationError(Exception):
    """Raised when a solution's module body will not execute."""


def compile_code(code, timeout):
    """Execute a solution's module body and return the object to call into."""
    start_alarm(timeout)
    try:
        module = ModuleType("tmp_sol", "")
        exec(code, module.__dict__)
        # LeetCode problems wrap their answer in a Solution class; everything
        # else exposes the function at module level.
        if "class Solution" in code:
            return module.Solution()
        return module
    except TimeoutException:
        # Module-level work that never finishes is a slow solution, not a
        # broken one.
        raise
    except Exception as exc:
        # A solution that does not compile is a failed solution, reported the
        # same way as one whose entry point is missing.
        raise CompilationError(repr(exc)) from exc
    finally:
        cancel_alarm()


def get_stripped_lines(value):
    return [line.strip() for line in value.strip().split("\n")]


def convert_line_to_decimals(line):
    try:
        return True, [Decimal(element) for element in line.split()]
    except Exception:
        return False, []


def grade_call_based(code, inputs, outputs, fn_name, timeout):
    """Grade a solution by calling `fn_name` with each test case's arguments."""
    code = IMPORT_PRELUDE + "\n\n" + code
    compiled = compile_code(code, timeout)
    method = getattr(compiled, fn_name, None)
    if method is None:
        return False, {"error_code": -4, "error_message": "Function not found in generated code"}

    parsed_inputs = [[json.loads(line) for line in case.split("\n")] for case in inputs]
    parsed_outputs = [json.loads(case) for case in outputs]

    for gt_input, gt_output in zip(parsed_inputs, parsed_outputs, strict=True):
        start_alarm(timeout)
        faulthandler.enable()
        try:
            prediction = method(*gt_input)
            cancel_alarm()

            # Tuples and lists are not distinguished: ground truth is never a tuple.
            if isinstance(prediction, tuple):
                prediction = list(prediction)

            if prediction != gt_output:
                return False, {"error_code": -2, "error_message": "Wrong Answer"}
        except TimeoutException:
            return False, {"error_code": -3, "error_message": "Time Limit Exceeded"}
        except (Exception, SystemExit) as exc:
            # A whole-program solution may exit; a function that does so has
            # no answer.
            return False, {"error_code": -4, "error_message": f"Runtime Error: {exc!r}"}
        finally:
            cancel_alarm()
            faulthandler.disable()

    return True, {}


def grade_stdio(code, inputs, outputs, timeout):
    """Grade a whole-program solution by feeding stdin and reading stdout."""
    code = make_function(clean_if_name(code))
    compiled = compile_code(code, timeout)
    method = getattr(compiled, "wrapped_function", None)
    if method is None:
        return False, {"error_code": -4, "error_message": "Function not found in generated code"}

    for gt_input, gt_output in zip(inputs, outputs, strict=True):
        start_alarm(timeout)
        faulthandler.enable()
        with Capturing() as captured:
            try:
                call_method(method, gt_input)
                cancel_alarm()
            except TimeoutException:
                return False, {"error_code": -3, "error_message": "Time Limit Exceeded"}
            except Exception as exc:
                return False, {"error_code": -4, "error_message": f"Runtime Error: {exc!r}"}
            finally:
                cancel_alarm()
                faulthandler.disable()

        prediction_lines = get_stripped_lines(captured[0])
        expected_lines = get_stripped_lines(gt_output)

        if len(prediction_lines) != len(expected_lines):
            return False, {
                "error_code": -2,
                "error_message": "Wrong answer: mismatched output length",
            }

        for idx, (predicted, expected) in enumerate(
            zip(prediction_lines, expected_lines, strict=True)
        ):
            if predicted == expected:
                continue
            # Compare numerically before rejecting: Decimal avoids the false
            # equalities that float comparison would introduce on large values.
            ok_prediction, decimal_prediction = convert_line_to_decimals(predicted)
            ok_expected, decimal_expected = convert_line_to_decimals(expected)
            if ok_prediction and ok_expected and decimal_prediction == decimal_expected:
                continue
            return False, {
                "error_code": -2,
                "error_message": f"Wrong answer at line {idx}",
            }

    return True, {}


def decode_test_cases(encoded):
    """Decode a test case blob, which may be JSON or compressed pickle.

    Raises ``ValueError`` for a blob in neither form, so that a payload the
    grader cannot read is never mistaken for a problem with no tests.
    """
    if not encoded:
        return []
    try:
        return json.loads(encoded)
    except json.JSONDecodeError:
        pass
    try:
        return json.loads(pickle.loads(zlib.decompress(base64.b64decode(encoded.encode("utf-8")))))
    except Exception as exc:
        raise ValueError(f"test cases are neither JSON nor compressed pickle: {exc!r}") from exc


def reliability_guard():
    """Disable calls that would let a solution damage the run.

    This is defence in depth for accidental damage inside an already isolated
    container, not a security boundary.
    """
    import builtins
    import shutil
    import subprocess

    faulthandler.disable()

    # Disabling a name means replacing it with None, which is by definition
    # not what its declared type allows.
    untyped_builtins: Any = builtins
    untyped_builtins.exit = None
    untyped_builtins.quit = None

    os.environ["OMP_NUM_THREADS"] = "1"

    for module, name in (
        (os, "kill"),
        (os, "system"),
        (os, "remove"),
        (os, "removedirs"),
        (os, "rmdir"),
        (os, "fchdir"),
        (os, "chdir"),
        (os, "setuid"),
        (os, "fork"),
        (os, "forkpty"),
        (os, "killpg"),
        (os, "rename"),
        (os, "renames"),
        (os, "truncate"),
        (os, "unlink"),
        (os, "fchmod"),
        (os, "fchown"),
        (os, "chmod"),
        (os, "chown"),
        (os, "chroot"),
        (shutil, "rmtree"),
        (shutil, "move"),
        (shutil, "chown"),
        (subprocess, "Popen"),
    ):
        if hasattr(module, name):
            setattr(module, name, None)

    loaded_modules: Any = sys.modules
    for blocked in ("ipdb", "joblib", "resource", "psutil", "tkinter"):
        loaded_modules[blocked] = None


MAX_ERROR_MESSAGE_LEN = 2000
VERDICT_PREFIX = "LCB_VERDICT "


def grade(solution, inputs, outputs, fn_name, timeout):
    """Grade a solution in this process and return ``(passed, detail)``."""
    signal.signal(signal.SIGALRM, timeout_handler)
    # Contest answers can be integers far past the interpreter's default limit
    # on converting between int and str; the reference harness raises it too.
    sys.set_int_max_str_digits(50000)
    reliability_guard()
    try:
        if fn_name:
            return grade_call_based(solution, inputs, outputs, fn_name, timeout)
        return grade_stdio(solution, inputs, outputs, timeout)
    except CompilationError as exc:
        return False, {"error_code": -4, "error_message": f"Compilation error: {exc}"}
    except TimeoutException:
        return False, {"error_code": -3, "error_message": "Time Limit Exceeded"}


def grade_in_child(solution, inputs, outputs, fn_name, timeout, budget):
    """Grade in a forked child and return the verdict fields it reports.

    The child sends its verdict back over a pipe. A child that dies before
    reporting, or is still running when ``budget`` seconds are up, is a failed
    solution: the parent never depends on the solution's process surviving.
    """
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(read_fd)
        try:
            passed, detail = grade(solution, inputs, outputs, fn_name, timeout)
            verdict: dict[str, Any] = {"passed": passed, **detail}
        except BaseException as exc:
            verdict = {
                "passed": False,
                "error_code": -5,
                "error_message": f"TestRunnerError: {exc!r}",
            }
        if "error_message" in verdict:
            verdict["error_message"] = verdict["error_message"][:MAX_ERROR_MESSAGE_LEN]
        with os.fdopen(write_fd, "wb") as pipe:
            pipe.write(json.dumps(verdict).encode("utf-8"))
        os._exit(0)

    os.close(write_fd)
    chunks = []
    deadline = time.monotonic() + budget
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
                return {
                    "passed": False,
                    "error_code": -3,
                    "error_message": (
                        f"Time Limit Exceeded: over the {budget:.0f}s budget for all tests"
                    ),
                }
            ready, _, _ = select.select([read_fd], [], [], remaining)
            if not ready:
                continue
            chunk = os.read(read_fd, 65536)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(read_fd)
    _, status = os.waitpid(pid, 0)

    if chunks:
        return json.loads(b"".join(chunks).decode("utf-8"))
    if os.WIFSIGNALED(status):
        # The interpreter itself died: deep recursion, memory exhaustion.
        return {
            "passed": False,
            "error_code": -4,
            "error_message": (
                f"Runtime Error: solution process killed by signal {os.WTERMSIG(status)}"
            ),
        }
    return {
        "passed": False,
        "error_code": -5,
        "error_message": (
            f"TestRunnerError: solution process exited {os.WEXITSTATUS(status)} without a verdict"
        ),
    }


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "problem.json")) as handle:
        problem = json.load(handle)
    with open(os.path.join(here, "solution.py")) as handle:
        solution = handle.read()

    fn_name = problem.get("fn_name")
    timeout = float(problem.get("timeout", 6))

    started = time.time()
    try:
        test_cases = decode_test_cases(problem["public_test_cases"]) + decode_test_cases(
            problem["private_test_cases"]
        )
        inputs = [case["input"] for case in test_cases]
        outputs = [case["output"] for case in test_cases]
    except Exception as exc:
        inputs = []
        verdict: dict[str, Any] = {
            "passed": False,
            "error_code": -5,
            "error_message": f"TestRunnerError: {exc!r}"[:MAX_ERROR_MESSAGE_LEN],
        }
    else:
        # The reference harness gives the whole run this long before killing it.
        budget = (timeout + 1) * len(inputs) + 5
        verdict = grade_in_child(solution, inputs, outputs, fn_name, timeout, budget)

    verdict["num_tests"] = len(inputs)
    verdict["elapsed"] = round(time.time() - started, 3)
    # The harness looks for the marked line. The child may have written to
    # stdout without a trailing newline, so start a fresh line first.
    sys.stdout = sys.__stdout__
    print("\n" + VERDICT_PREFIX + json.dumps(verdict), flush=True)


if __name__ == "__main__":
    main()
