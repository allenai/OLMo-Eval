r"""
StrongReject Safety Evaluation Task

This module implements the StrongReject evaluation task
as previously implemented in allenai/safety-eval

Paper: https://arxiv.org/abs/2402.10260


For a reasoning model:
olmo-eval beaker launch  \
    --harness default   \
    -o provider.kwargs.tensor_parallel_size=2   \
    -o 'metrics.collect_gpu=true'   \
    -o auxiliary_providers.sr_judge.kind=vllm_server   \
    -o auxiliary_providers.sr_judge.model=google/gemma-2b   \
    -o auxiliary_providers.sr_judge.tokenizer=qylu4156/strongreject-15k-v1   \
    -o auxiliary_providers.sr_judge.kwargs.enable_lora=true   \
    -o auxiliary_providers.sr_judge.kwargs.lora_modules=\
    [strongreject=qylu4156/strongreject-15k-v1] \
    -o auxiliary_providers.sr_judge.kwargs.gpu_memory_utilization=0.2  \
    -o auxiliary_providers.sr_judge.kwargs.add_bos_token=true \
    -o scoring_concurrency=4   \
    -m allenai/Olmo-3-7B-Think   \
    -t "strongreject:sr_judge_thinking@high" \
    -w "ai2/WORKSPACE"   \
    -B "ai2/BUDGET"   \
    --cluster h100

For an instruct model:
olmo-eval beaker launch  \
    --harness default   \
    -o 'metrics.collect_gpu=true'   \
    -o auxiliary_providers.sr_judge.kind=vllm_server   \
    -o auxiliary_providers.sr_judge.model=google/gemma-2b   \
    -o auxiliary_providers.sr_judge.tokenizer=qylu4156/strongreject-15k-v1   \
    -o auxiliary_providers.sr_judge.kwargs.enable_lora=true   \
    -o auxiliary_providers.sr_judge.kwargs.lora_modules=\
    [strongreject=qylu4156/strongreject-15k-v1] \
    -o auxiliary_providers.sr_judge.kwargs.gpu_memory_utilization=0.2  \
    -o auxiliary_providers.sr_judge.kwargs.add_bos_token=true \
    -o scoring_concurrency=4   \
    -m allenai/Olmo-3-7B-Instruct   \
    -t "strongreject:sr_judge@high" \
    -w "ai2/WORKSPACE"   \
    -B "ai2/BUDGET"   \
    --cluster h100

For a base model:
olmo-eval beaker launch  \
    --harness default   \
    -o 'metrics.collect_gpu=true'   \
    -o auxiliary_providers.sr_judge.kind=vllm_server   \
    -o auxiliary_providers.sr_judge.model=google/gemma-2b   \
    -o auxiliary_providers.sr_judge.tokenizer=qylu4156/strongreject-15k-v1   \
    -o auxiliary_providers.sr_judge.kwargs.enable_lora=true   \
    -o auxiliary_providers.sr_judge.kwargs.lora_modules=\
    [strongreject=qylu4156/strongreject-15k-v1] \
    -o auxiliary_providers.sr_judge.kwargs.gpu_memory_utilization=0.2  \
    -o auxiliary_providers.sr_judge.kwargs.add_bos_token=true \
    -o scoring_concurrency=4   \
    -m allenai/Olmo-3-1025-7B   \
    -t "strongreject:base@high" \
    -w "ai2/WORKSPACE"   \
    -B "ai2/BUDGET"   \
    --cluster h100

On the OpenAI harness:
olmo-eval beaker launch  \
    --harness default   \
    -o 'metrics.collect_gpu=true'   \
    -o scoring_concurrency=4   \
    -m allenai/Olmo-3-7B-Instruct   \
    -t "strongreject:openai_judge@high" \
    -w "ai2/WORKSPACE"   \
    -B "ai2/BUDGET"   \
    --cluster h100
"""

import logging
from typing import Any

from olmo_eval.common.scorers import SafetyScorer
from olmo_eval.common.types import Instance
from olmo_eval.data import DataSource
from olmo_eval.evals.tasks.common import register
from olmo_eval.evals.tasks.safety_base import SafetyBase, register_safety_variants

logger = logging.getLogger(__name__)


@register("strongreject")
class StrongReject(SafetyBase):
    """strongreject safety evaluation task."""

    data_source = DataSource(path="allenai/olmo-eval-strongreject", split="test")

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        """Convert a dataset document to an Instance."""
        if not doc["subsample"]:
            return None

        return Instance(
            question=doc["jailbroken_prompt"],
            gold_answer="response_harmfulness=no",  # used for openai judge only
            metadata={
                "id": doc["id"],
                "index": index,
                "jailbreak": doc["jailbreak"],
                "category": doc["category"],
                "forbidden_prompt": doc["forbidden_prompt"],
            },
        )


strongreject_safety_metrics = (
    "category__Disinformation and deception",
    "category__Hate, harassment and discrimination",
    "category__Illegal goods and services",
    "category__Non-violent crimes",
    "category__Sexual content",
    "category__Violence",
    "jailbreak__aim",
    "jailbreak__auto_obfuscation",
    "jailbreak__auto_payload_splitting",
    "jailbreak__base64",
    "jailbreak__base64_input_only",
    "jailbreak__base64_output_only",
    "jailbreak__base64_raw",
    "jailbreak__combination_2",
    "jailbreak__combination_3",
    "jailbreak__dev_mode_v2",
    "jailbreak__dev_mode_with_rant",
    "jailbreak__disemvowel",
    "jailbreak__distractors",
    "jailbreak__distractors_negated",
    "jailbreak__evil_confidant",
    "jailbreak__few_shot_json",
    "jailbreak__gcg_transfer_harmbench",
    "jailbreak__gcg_transfer_universal_attacks",
    "jailbreak__none",
    "jailbreak__pair",
    "jailbreak__pap_authority_endorsement",
    "jailbreak__pap_evidence_based_persuasion",
    "jailbreak__pap_expert_endorsement",
    "jailbreak__pap_logical_appeal",
    "jailbreak__pap_misrepresentation",
    "jailbreak__poems",
    "jailbreak__prefix_injection",
    "jailbreak__refusal_suppression",
    "jailbreak__rot_13",
    "jailbreak__style_injection_json",
    "jailbreak__style_injection_short",
    "jailbreak__translation_guarani",
    "jailbreak__translation_hmong",
    "jailbreak__translation_scotts_gaelic",
    "jailbreak__translation_zulu",
    "jailbreak__wikipedia",
    "jailbreak__wikipedia_with_title",
)


# =============================================================================
# Variant Registrations
# =============================================================================

sr_scorer = SafetyScorer(provider_name="sr_judge", judge_format="strongreject")

register_safety_variants("strongreject", strongreject_safety_metrics, sr_scorer, "sr_judge")
