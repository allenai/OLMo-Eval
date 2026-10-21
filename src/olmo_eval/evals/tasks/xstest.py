"""
XSTest Safety Evaluation Task

This module implements the XSTest evaluation task
as previously implemented in allenai/safety-eval

Paper: https://arxiv.org/abs/2308.01263

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
    -t "xstest:wg_judge_thinking@high" \
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
    -t "xstest:wg_judge@high" \
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
    -t "xstest:base@high" \
    -w "ai2/WORKSPACE"   \
    -B "ai2/BUDGET"   \
    --cluster h100

On the OpenAI harness:
olmo-eval beaker launch  \
    --harness default   \
    -o 'metrics.collect_gpu=true'   \
    -o scoring_concurrency=4   \
    -m allenai/Olmo-3-7B-Instruct   \
    -t "xstest:openai_judge@high" \
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


@register("xstest")
class XSTest(SafetyBase):
    """xstest safety evaluation task."""

    data_source = DataSource(path="walledai/XSTest", split="test")

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        """Convert a dataset document to an Instance."""
        prompt_type = doc["label"]

        if prompt_type == "safe":
            gold_answer = "response_refusal=no"
        elif prompt_type == "unsafe":
            gold_answer = "response_refusal=yes"
        else:
            logger.warning("Failed to parse prompt data type: %s", doc)
            return None

        return Instance(
            question=doc["prompt"],
            gold_answer=gold_answer,
            metadata={
                "id": index,
                "prompt_type": prompt_type,
                "subcategory": doc["type"],
            },
        )


_XSTEST_SUBSET_METRICS = (
    "prompt_type__safe",
    "prompt_type__unsafe",
    "subcategory__homonyms",
    "subcategory__contrast_homonyms",
    "subcategory__figurative_language",
    "subcategory__contrast_figurative_language",
    "subcategory__safe_targets",
    "subcategory__contrast_safe_targets",
    "subcategory__safe_contexts",
    "subcategory__contrast_safe_contexts",
    "subcategory__definitions",
    "subcategory__contrast_definitions",
    "subcategory__nons_group_real_discr",
    "subcategory__real_group_nons_discr",
    "subcategory__contrast_discr",
    "subcategory__historical_events",
    "subcategory__contrast_historical_events",
    "subcategory__privacy_public",
    "subcategory__privacy_fictional",
    "subcategory__contrast_privacy",
)

# =============================================================================
# Variant Registrations
# =============================================================================

register_safety_variants("xstest", _XSTEST_SUBSET_METRICS)
