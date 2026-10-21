"""Multimodal (vision-language) OLMo-core inference provider.

Split by responsibility: :mod:`checkpoint` (detect/config/weights),
:mod:`preprocessing` (tokenizer and crop hints), :mod:`cache` (KV-cached
decoding), and :mod:`provider` (the provider itself).
"""

from olmo_eval.inference.providers.olmo_core_vlm.checkpoint import (
    is_multimodal_checkpoint,
)
from olmo_eval.inference.providers.olmo_core_vlm.provider import OlmoCoreVLMProvider

__all__ = ["OlmoCoreVLMProvider", "is_multimodal_checkpoint"]
