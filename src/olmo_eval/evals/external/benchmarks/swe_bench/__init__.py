"""SWE-bench Verified external evaluation.

SWE-bench Verified is a 500-instance, human-validated subset of SWE-bench. Each
instance asks an agent to resolve a real GitHub issue in a Python repository,
and the resulting patch is graded by running the repository's tests.

Dataset: https://huggingface.co/datasets/SWE-bench/SWE-bench_Verified
Repository: https://github.com/SWE-bench/SWE-bench
"""

from olmo_eval.evals.external.benchmarks.swe_bench.eval import SWEBenchVerifiedExternalEval
from olmo_eval.evals.external.registry import register_external_eval

register_external_eval(SWEBenchVerifiedExternalEval())
