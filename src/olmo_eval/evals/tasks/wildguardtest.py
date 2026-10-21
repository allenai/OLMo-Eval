"""
WildGuardTest Safety Evaluation Task

This module implements the WildGuardTest evaluation task
as previously implemented in allenai/safety-eval

Paper: https://arxiv.org/abs/2406.18495

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
    -t "wildguardtest:wg_judge_thinking@high" \
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
    -t "wildguardtest:wg_judge@high" \
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
    -t "wildguardtest:base@high" \
    -w "ai2/WORKSPACE"   \
    -B "ai2/BUDGET"   \
    --cluster h100

On the OpenAI harness:
olmo-eval beaker launch  \
    --harness default   \
    -o 'metrics.collect_gpu=true'   \
    -o scoring_concurrency=4   \
    -m allenai/Olmo-3-7B-Instruct   \
    -t "wildguardtest:openai_judge@high" \
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


@register("wildguardtest")
class WildGuardTest(SafetyBase):
    """wildguardtest safety evaluation task."""

    data_source = DataSource(path="allenai/wildguardmix", subset="wildguardtest", split="test")

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        """Convert a dataset document to an Instance."""

        # Drop the instance if prompt_harm_label or response_harm_label is NA
        if doc["prompt_harm_label"] is None or doc["response_harm_label"] is None:
            return None

        # Drop examples where the prompt label is not harmful
        if doc["prompt_harm_label"] != "harmful":
            return None

        prompt_type = "adversarial" if doc["adversarial"] else "vanilla"

        question = doc["prompt"]

        return Instance(
            question=question,
            gold_answer="response_harmfulness=no",
            metadata={
                "id": index,
                "subcategory": doc["subcategory"],
                "prompt_type": prompt_type,
            },
        )


_WILDGUARD_SUBSET_METRICS = (
    "prompt_type__vanilla",
    "prompt_type__adversarial",
    "subcategory__private_information_individual",
    "subcategory__others",
    "subcategory__cyberattack",
    "subcategory__sexual_content",
    "subcategory__social_stereotypes_and_unfair_discrimination",
    "subcategory__toxic_language_hate_speech",
    "subcategory__violence_and_physical_harm",
    "subcategory__defamation_encouraging_unethical_or_unsafe_actions",
    "subcategory__fraud_assisting_illegal_activities",
    "subcategory__mental_health_over-reliance_crisis",
    "subcategory__causing_material_harm_by_disseminating_misinformation",
    "subcategory__disseminating_false_or_misleading_information_encouraging_disinformation_campaigns",
    "subcategory__copyright_violations",
    "subcategory__sensitive_information_organization_government",
)


# =============================================================================
# Variant Registrations
# =============================================================================

register_safety_variants("wildguardtest", _WILDGUARD_SUBSET_METRICS)
