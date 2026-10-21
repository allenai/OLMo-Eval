from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from olmo_eval.harness.config import ProviderConfig


# =============================================================================
# Model Presets for Evaluation
# =============================================================================


def get_model_presets() -> dict[str, ProviderConfig]:
    """Get model presets dictionary.

    Returns a dictionary mapping preset names to ProviderConfig instances.
    """
    from olmo_eval.common.types import ProviderKind
    from olmo_eval.harness.config import ProviderConfig

    return {
        "llama3.1-8b": ProviderConfig(
            kind=ProviderKind.VLLM,
            model="meta-llama/Meta-Llama-3.1-8B",
        ),
        "llama3.1-8b-instruct": ProviderConfig(
            kind=ProviderKind.VLLM,
            model="meta-llama/Llama-3.1-8B-Instruct",
        ),
        "llama3.1-70b": ProviderConfig(
            kind=ProviderKind.VLLM,
            model="meta-llama/Meta-Llama-3.1-70B",
        ),
        "llama3.1-70b-instruct": ProviderConfig(
            kind=ProviderKind.VLLM,
            model="meta-llama/Llama-3.1-70B-Instruct",
        ),
        "olmo-3-1025-7b": ProviderConfig(
            kind=ProviderKind.VLLM,
            model="allenai/Olmo-3-1025-7B",
            trust_remote_code=True,
            max_model_len=4096,
            revision="stage2-step47684",
            kwargs={"gpu_memory_utilization": 0.7, "add_bos_token": False},
        ),
        "olmo-2-7b": ProviderConfig(
            kind=ProviderKind.VLLM,
            model="allenai/OLMo-2-1124-7B",
            trust_remote_code=True,
        ),
        "olmo-2-13b": ProviderConfig(
            kind=ProviderKind.VLLM,
            model="allenai/OLMo-2-1124-13B",
            trust_remote_code=True,
        ),
        "qwen2.5-7b": ProviderConfig(
            kind=ProviderKind.VLLM,
            model="Qwen/Qwen2.5-7B",
        ),
        "qwen3-coder-30b": ProviderConfig(
            kind=ProviderKind.VLLM_SERVER,
            model="Qwen/Qwen3-Coder-30B-A3B-Instruct",
            kwargs={"enable_expert_parallel": True, "tool_call_parser": "qwen3_coder"},
        ),
        # TODO(undfined): Leaving this here as reference. DeepGEMM is more involved
        # and we can add it to base image and toggle it on when needed.
        # "qwen3-coder-30b-fp8": ProviderConfig(
        #     kind=ProviderKind.VLLM_SERVER,
        #     model="Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8",
        #     kwargs={"enable_expert_parallel": True, "tool_call_parser": "qwen3_coder"},
        #     dependencies=(
        #         "git+https://github.com/deepseek-ai/DeepGEMM.git@v2.1.1.post3 --no-build-isolation",
        #     ),
        # ),
        "deepseek-r1-distill-8b": ProviderConfig(
            kind=ProviderKind.VLLM,
            model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            max_model_len=32768,
        ),
        "mistral-7b": ProviderConfig(
            kind=ProviderKind.VLLM,
            model="mistralai/Mistral-7B-v0.3",
        ),
        "o3-mini-2025-01-31-medium": ProviderConfig(
            kind=ProviderKind.LITELLM,
            model="openai/o3-mini-2025-01-31",
            api_base="https://api.openai.com/v1",
            required_secrets=("OPENAI_API_KEY",),
            kwargs={"reasoning_effort": "medium", "drop_params": True},
        ),
        "gpt-4o": ProviderConfig(
            kind=ProviderKind.LITELLM,
            model="gpt-4o",
            api_base="https://api.openai.com/v1",
            required_secrets=("OPENAI_API_KEY",),
        ),
        "gpt-4o-mini": ProviderConfig(
            kind=ProviderKind.LITELLM,
            model="openai/gpt-4o-mini",
            api_base="https://api.openai.com/v1",
            required_secrets=("OPENAI_API_KEY",),
        ),
        "gpt-4-turbo": ProviderConfig(
            kind=ProviderKind.LITELLM,
            model="openai/gpt-4-turbo",
            api_base="https://api.openai.com/v1",
            required_secrets=("OPENAI_API_KEY",),
        ),
        "claude-3-opus": ProviderConfig(
            kind=ProviderKind.LITELLM,
            model="anthropic/claude-3-opus-20240229",
            api_base="https://api.anthropic.com",
            required_secrets=("ANTHROPIC_API_KEY",),
        ),
        "claude-3-sonnet": ProviderConfig(
            kind=ProviderKind.LITELLM,
            model="anthropic/claude-3-sonnet-20240229",
            api_base="https://api.anthropic.com",
            required_secrets=("ANTHROPIC_API_KEY",),
        ),
        "mock": ProviderConfig(
            kind=ProviderKind.MOCK,
            model="mock",
        ),
        # Ai2 deployed models on Litellm Proxy
        "cirrascale-olmo-3-7b-instruct": ProviderConfig(
            kind=ProviderKind.LITELLM,
            model="litellm_proxy/openai/Olmo-3-7B-Instruct",
            api_base="https://ai2-model-hub.allen.ai",
            required_secrets=("LITELLM_PROXY_API_KEY",),
        ),
        "modal-olmo-3-7b-instruct": ProviderConfig(
            kind=ProviderKind.LITELLM,
            model="litellm_proxy/openai/ai2-release-partners/Olmo-3-7B-Instruct",
            api_base="https://ai2-model-hub.allen.ai",
            required_secrets=("LITELLM_PROXY_API_KEY",),
        ),
        # Multimodal (image-text-to-text) models. Run the released HF Molmo2
        # checkpoint directly via AutoProcessor + AutoModelForImageTextToText;
        # tasks attach the image to LMRequest.images. Requires the `hf` extra.
        # fp32 weights + bf16 autocast matches mm_olmo's official amp_bf16 eval
        # numerics (the released config declares dtype=float32, float32_attention).
        "molmo2-4b": ProviderConfig(
            kind=ProviderKind.HF,
            model="allenai/Molmo2-4B",
            trust_remote_code=True,
            dtype="float32",
            kwargs={"multimodal": True, "max_crops": 24, "autocast_dtype": "bfloat16"},
        ),
        "molmo2-8b": ProviderConfig(
            kind=ProviderKind.HF,
            model="allenai/Molmo2-8B",
            trust_remote_code=True,
            dtype="float32",
            kwargs={"multimodal": True, "max_crops": 24, "autocast_dtype": "bfloat16"},
        ),
        "molmo2-o-7b": ProviderConfig(
            kind=ProviderKind.HF,
            model="allenai/Molmo2-O-7B",
            trust_remote_code=True,
            dtype="float32",
            kwargs={"multimodal": True, "max_crops": 24, "autocast_dtype": "bfloat16"},
        ),
    }
