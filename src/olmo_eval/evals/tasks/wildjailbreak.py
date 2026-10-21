"""
WildJailbreak Safety Evaluation Task

This module implements the WildJailbreak evaluation task
as previously implemented in allenai/safety-eval

Paper: https://arxiv.org/abs/2406.18510

Example commands to run:

For a reasoning model:
olmo-eval beaker launch  \
    --harness default   \
    -o provider.kwargs.tensor_parallel_size=2   \
    -o 'metrics.collect_gpu=true'   \
    -o auxiliary_providers.wg_judge.kind=vllm_server \
    -o auxiliary_providers.wg_judge.model=allenai/wildguard \
    -o auxiliary_providers.wg_judge.kwargs.add_bos_token=true \
    -o scoring_concurrency=4   \
    -m allenai/Olmo-3-7B-Think   \
    -t "wildjailbreak:wg_judge_thinking@high" \
    -w "ai2/WORKSPACE"   \
    -B "ai2/BUDGET"   \
    --cluster h100

For an instruct model:
olmo-eval beaker launch  \
    --harness default   \
    -o 'metrics.collect_gpu=true'   \
    -o auxiliary_providers.wg_judge.kind=vllm_server \
    -o auxiliary_providers.wg_judge.model=allenai/wildguard \
    -o auxiliary_providers.wg_judge.kwargs.add_bos_token=true \
    -o scoring_concurrency=4   \
    -m allenai/Olmo-3-7B-Instruct   \
    -t "wildjailbreak:wg_judge@high" \
    -w "ai2/WORKSPACE"   \
    -B "ai2/BUDGET"   \
    --cluster h100

For a base model:
olmo-eval beaker launch  \
    --harness default   \
    -o 'metrics.collect_gpu=true'   \
    -o auxiliary_providers.wg_judge.kind=vllm_server \
    -o auxiliary_providers.wg_judge.model=allenai/wildguard \
    -o auxiliary_providers.wg_judge.kwargs.add_bos_token=true \
    -o scoring_concurrency=4   \
    -m allenai/Olmo-3-1025-7B   \
    -t "wildjailbreak:base@high" \
    -w "ai2/WORKSPACE"   \
    -B "ai2/BUDGET"   \
    --cluster h100

On the OpenAI harness:
olmo-eval beaker launch  \
    --harness default   \
    -o 'metrics.collect_gpu=true'   \
    -o scoring_concurrency=4   \
    -m allenai/Olmo-3-7B-Instruct   \
    -t "wildjailbreak:openai_judge@high" \
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


@register("wildjailbreak")
class WildJailbreak(SafetyBase):
    """wildjailbreak safety evaluation task."""

    data_source = DataSource(path="allenai/olmo-eval-wildjailbreak", split="test")

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        """Convert a dataset document to an Instance."""

        metadata: dict[str, Any] = {"id": index}

        if doc["data_type"] == "adversarial_benign":
            prompt_type = "benign"
            gold_answer = "response_refusal=no"
        elif doc["data_type"] == "adversarial_harmful":
            prompt_type = "harmful"
            gold_answer = "response_refusal=yes"
            metadata["vanilla_prompt"] = doc["vanilla"]
        else:
            logger.warning("Failed to parse prompt data type: %s", doc)
            return None

        metadata["prompt_type"] = prompt_type
        return Instance(
            question=doc["adversarial"],
            gold_answer=gold_answer,
            metadata=metadata,
        )


_WILDJAILBREAK_SUBSET_METRICS = (
    "prompt_type__benign",
    "prompt_type__harmful",
)

# =============================================================================
# Variant Registrations
# =============================================================================

# The benign and harmful halves are separate tasks upstream, whose headline
# numbers are averaged 50/50. Weighting by instance count instead would let the
# ~2000 harmful prompts bury over-refusal on the ~250 benign ones.
register_safety_variants(
    "wildjailbreak",
    _WILDJAILBREAK_SUBSET_METRICS,
    macro_subsets=_WILDJAILBREAK_SUBSET_METRICS,
)
