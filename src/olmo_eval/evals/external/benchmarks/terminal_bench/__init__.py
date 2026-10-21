"""Terminal-Bench style external evaluations.

Terminal-Bench evaluates LLM agents on diverse terminal tasks requiring
software engineering skills, system administration, and problem-solving.
The loader pins the 2.1 release of the benchmark. OpenThoughts-TBLite, a
100-task set in the same format, is evaluated with the same harness.

Repositories:
- https://github.com/harbor-framework/terminal-bench-2-1
- https://github.com/open-thoughts/OpenThoughts-TBLite
"""

from olmo_eval.evals.external.benchmarks.terminal_bench.eval import TerminalBenchExternalEval
from olmo_eval.evals.external.benchmarks.terminal_bench.tblite import (
    OpenThoughtsTBLiteExternalEval,
)
from olmo_eval.evals.external.registry import register_external_eval

register_external_eval(TerminalBenchExternalEval())
register_external_eval(OpenThoughtsTBLiteExternalEval())
