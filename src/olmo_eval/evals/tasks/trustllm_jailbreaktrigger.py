"""
TrustLLM_JailbreakTrigger Safety Evaluation Task

This module implements the TrustLLM_JailbreakTrigger evaluation task
as previously implemented in allenai/safety-eval

It is not part of the full safety suite

Paper: https://arxiv.org/abs/2401.05561

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
    -t "trustllm_jailbreaktrigger:wg_judge_thinking@high" \
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
    -t "trustllm_jailbreaktrigger:wg_judge@high" \
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
    -t "trustllm_jailbreaktrigger:base@high" \
    -w "ai2/WORKSPACE"   \
    -B "ai2/BUDGET"   \
    --cluster h100

On the OpenAI harness:
olmo-eval beaker launch  \
    --harness default   \
    -o 'metrics.collect_gpu=true'   \
    -o scoring_concurrency=4   \
    -m allenai/Olmo-3-7B-Instruct   \
    -t "trustllm_jailbreaktrigger:openai_judge@high" \
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


@register("trustllm_jailbreaktrigger")
class TrustLLM_JailbreakTrigger(SafetyBase):
    """trustllm_jailbreaktrigger safety evaluation task."""

    data_source = DataSource(path="allenai/tulu-3-trustllm-jailbreaktrigger-eval", split="test")

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        """Convert a dataset document to an Instance."""

        return Instance(
            question=doc["prompt"],
            gold_answer="response_refusal=yes",
            metadata={
                "id": doc["id"],
                "index": index,
                "source": doc["source"],
                "jailbreak_type": doc["label"][0],
            },
        )


_JAILBREAKTRIGGER_SUBSET_METRICS = (
    "source__QBB",
    "source__ITC",
    "jailbreak_type__Refusal sentence prohibition",
    "jailbreak_type__leetspeak",
    "jailbreak_type__bad words",
    "jailbreak_type__programming",
    "jailbreak_type__fixed sentence",
    "jailbreak_type__cot",
    "jailbreak_type__json_format",
    "jailbreak_type__no_punctuation",
    "jailbreak_type__no_long_word",
    "jailbreak_type__without_the",
    "jailbreak_type__url_encode",
    "jailbreak_type__cou",
    "jailbreak_type__scenario",
    "jailbreak_type__multitask",
)


# =============================================================================
# Variant Registrations
# =============================================================================

register_safety_variants("trustllm_jailbreaktrigger", _JAILBREAKTRIGGER_SUBSET_METRICS)
