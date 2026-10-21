"""
HarmBench Safety Evaluation Task

This module implements the HarmBench Safety evaluation task
as previously implemented in allenai/safety-eval

Paper: https://arxiv.org/abs/2402.04249

Example commands to run:

For a reasoning model:
olmo-eval beaker launch  \
    --harness default   \
    -o provider.kwargs.tensor_parallel_size=2   \
    -o 'metrics.collect_gpu=true'   \
    -o 'provider.kwargs.timeout=300'   \
    -o auxiliary_providers.wg_judge.kind=vllm_server \
    -o auxiliary_providers.wg_judge.model=allenai/wildguard \
    -o auxiliary_providers.wg_judge.kwargs.add_bos_token=true \
    -o scoring_concurrency=4   \
    -m allenai/Olmo-3-7B-Think   \
    -t "harmbench:wg_judge_thinking@high" \
    -w "ai2/WORKSPACE"   \
    -B "ai2/BUDGET"   \
    --cluster h100

For an instruct model:
olmo-eval beaker launch  \
    --harness default   \
    -o 'metrics.collect_gpu=true'   \
    -o 'provider.kwargs.timeout=300'   \
    -o auxiliary_providers.wg_judge.kind=vllm_server \
    -o auxiliary_providers.wg_judge.model=allenai/wildguard \
    -o auxiliary_providers.wg_judge.kwargs.add_bos_token=true \
    -o scoring_concurrency=4   \
    -m allenai/Olmo-3-7B-Instruct   \
    -t "harmbench:wg_judge@high" \
    -w "ai2/WORKSPACE"   \
    -B "ai2/BUDGET"   \
    --cluster h100

For a base model:
olmo-eval beaker launch  \
    --harness default   \
    -o 'metrics.collect_gpu=true'   \
    -o 'provider.kwargs.timeout=300'   \
    -o auxiliary_providers.wg_judge.kind=vllm_server \
    -o auxiliary_providers.wg_judge.model=allenai/wildguard \
    -o auxiliary_providers.wg_judge.kwargs.add_bos_token=true \
    -o scoring_concurrency=4   \
    -m allenai/Olmo-3-1025-7B   \
    -t "harmbench:base@high" \
    -w "ai2/WORKSPACE"   \
    -B "ai2/BUDGET"   \
    --cluster h100

On the OpenAI harness:
olmo-eval beaker launch  \
    --harness default   \
    -o 'metrics.collect_gpu=true'   \
    -o 'provider.kwargs.timeout=300'   \
    -o scoring_concurrency=4   \
    -m allenai/Olmo-3-7B-Instruct   \
    -t "harmbench:openai_judge@high" \
    -w "ai2/WORKSPACE"   \
    -B "ai2/BUDGET"   \
    --cluster h100
"""

import logging
from typing import Any

from olmo_eval.common.types import Instance
from olmo_eval.data import DataSource
from olmo_eval.evals.tasks.common import register
from olmo_eval.evals.tasks.safety_base import SafetyBase, register_safety_variants

logger = logging.getLogger(__name__)


@register("harmbench")
class Harmbench(SafetyBase):
    """Harmbench safety evaluation task."""

    data_source = DataSource("allenai/tulu-3-harmbench-eval")

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        """Convert a dataset document to an Instance."""

        functional_cat = doc["FunctionalCategory"]
        semantic_cat = doc["SemanticCategory"]

        if functional_cat == "contextual":
            question = doc["ContextString"] + "\n\n---\n\n" + doc["Behavior"]
        else:
            question = doc["Behavior"]

        return Instance(
            question=question,
            gold_answer="response_harmfulness=no",
            metadata={
                "id": doc["BehaviorID"],
                "index": index,
                "functional_category": functional_cat,
                "semantic_category": semantic_cat,
            },
        )


_HARMBENCH_SUBSET_METRICS = (
    "functional_category__standard",
    "functional_category__contextual",
    "functional_category__copyright",
    "semantic_category__copyright",
    "semantic_category__misinformation_disinformation",
    "semantic_category__chemical_biological",
    "semantic_category__illegal",
    "semantic_category__harmful",
    "semantic_category__cybercrime_intrusion",
    "semantic_category__harassment_bullying",
)


# =============================================================================
# Variant Registrations
# =============================================================================

register_safety_variants("harmbench", _HARMBENCH_SUBSET_METRICS)
