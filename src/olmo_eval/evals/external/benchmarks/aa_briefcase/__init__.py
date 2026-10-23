"""AA-Briefcase-Lite external evaluation.

AA-Briefcase-Lite is the public week of Artificial Analysis's AA-Briefcase
benchmark: an agent with a code-execution sandbox and a pool of source files
produces four commercial due-diligence deliverables, which a model judge grades
against binary rubric checks.

Dataset: https://huggingface.co/datasets/ArtificialAnalysis/AA-Briefcase-Lite
Harness: https://github.com/ArtificialAnalysis/Stirrup
"""

from olmo_eval.evals.external.benchmarks.aa_briefcase.eval import AABriefcaseLiteExternalEval
from olmo_eval.evals.external.registry import register_external_eval

register_external_eval(AABriefcaseLiteExternalEval())
