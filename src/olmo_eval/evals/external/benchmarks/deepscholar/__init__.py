"""DeepScholar-Bench external evaluation.

DeepScholar-Bench evaluates generative research synthesis: a system retrieves
prior work and writes a paper's related-work section, scored on organization,
nugget/reference coverage, and citation precision.

Repository: https://github.com/guestrin-lab/deepscholar-bench
"""

from olmo_eval.evals.external.benchmarks.deepscholar.eval import DeepScholarExternalEval
from olmo_eval.evals.external.registry import register_external_eval

register_external_eval(DeepScholarExternalEval())
