"""Literal type definitions for constrained string values."""

from enum import StrEnum
from typing import Literal


class ProviderKind(StrEnum):
    """Provider types for inference backends."""

    VLLM = "vllm"
    VLLM_SERVER = "vllm_server"
    HF = "hf"
    OLMO_CORE = "olmo_core"
    OLMO_CORE_VLM = "olmo_core_vlm"
    MOCK = "mock"
    LITELLM = "litellm"


ProviderLiteral = Literal[
    "vllm", "vllm_server", "hf", "olmo_core", "olmo_core_vlm", "mock", "litellm"
]
DtypeLiteral = Literal["auto", "float16", "bfloat16", "float32"]
PriorityLiteral = Literal["low", "normal", "high", "urgent"]
LoadFormatLiteral = Literal[
    "auto", "pt", "safetensors", "runai_streamer", "tensorizer", "bitsandbytes"
]
