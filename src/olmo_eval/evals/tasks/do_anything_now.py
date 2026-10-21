"""
Do_Anything_Now Safety Evaluation Task

This module implements the Do_Anything_Now evaluation task
as previously implemented in allenai/safety-eval

Paper: https://arxiv.org/abs/2308.03825

Usage:

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
    -t "do_anything_now:wg_judge_thinking@high" \
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
    -t "do_anything_now:wg_judge@high" \
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
    -t "do_anything_now:base@high" \
    -w "ai2/WORKSPACE"   \
    -B "ai2/BUDGET"   \
    --cluster h100

On the OpenAI harness:
olmo-eval beaker launch  \
    --harness default   \
    -o 'metrics.collect_gpu=true'   \
    -o scoring_concurrency=4   \
    -m allenai/Olmo-3-7B-Instruct   \
    -t "do_anything_now:openai_judge@high" \
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


@register("do_anything_now")
class DoAnythingNow(SafetyBase):
    """DoAnythingNow safety evaluation task."""

    data_source = DataSource(path="allenai/tulu-3-do-anything-now-eval", split="test")

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        """Convert a dataset document to an Instance."""

        return Instance(
            question=doc["adversarial"],
            gold_answer="response_refusal=yes",
            metadata={
                "id": index,
                "jailbreak_source": doc["source"],
                "platform": doc["platform"],
                "vanilla_prompt": doc["vanilla"],
            },
        )


_DOANYTHING_SUBSET_METRICS = (
    "jailbreak_source__jailbreak_chat",
    "jailbreak_source__BreakGPT",
    "jailbreak_source__ChatGPT",
    "jailbreak_source__LLM Promptwriting",
    "jailbreak_source__AI Prompt Sharing",
    "jailbreak_source__ChatGPTJailbreak",
)


# =============================================================================
# Variant Registrations
# =============================================================================

register_safety_variants("do_anything_now", _DOANYTHING_SUBSET_METRICS)
