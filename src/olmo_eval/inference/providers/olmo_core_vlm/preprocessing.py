"""Tokenizer and preprocessing hints resolved from a multimodal checkpoint.

The Molmo2 image special tokens are resolved to vocab IDs through the tokenizer so
the provider works for any vocab layout that defines them.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol, overload

import olmo_eval.inference.providers.olmo_core_utils as core_utils
from olmo_eval.inference.providers.olmo_core_vlm.checkpoint import (
    MultimodalCheckpointInfo,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

DEFAULT_TOKENIZER = "allenai/Molmo2-4B"

# Molmo2 image special-token *names* — resolved to IDs through the tokenizer so
# the provider works for any vocab layout that defines them.
IMAGE_SPECIAL_TOKENS = (
    "<im_start>",
    "<im_end>",
    "<im_patch>",
    "<im_col>",
    "<low_res_im_start>",
)
IMAGE_PLACEHOLDER_TOKEN = "<|image|>"
END_OF_TURN_TOKEN = "<|im_end|>"


class VLMTokenizerProtocol(core_utils.TokenizerProtocol, Protocol):
    """Tokenizer surface the multimodal provider needs on top of the text one.

    ``convert_tokens_to_ids`` maps the image special-token names above to the
    vocab IDs the provider splices into prompts.
    """

    bos_token_id: int | None

    @overload
    def convert_tokens_to_ids(self, tokens: str) -> int | None: ...

    @overload
    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]: ...


def resolve_tokenizer_path(info: MultimodalCheckpointInfo, explicit_tokenizer: str | None) -> str:
    """Pick the tokenizer to load: explicit > checkpoint hint > Molmo2 default.

    OLMo-core multimodal checkpoints record the HF model they were bootstrapped
    from as ``model_id``; mm_olmo checkpoints record only the *base* tokenizer
    (e.g. ``Qwen/Qwen3-4B``) which lacks the Molmo2 image special tokens and
    chat template, so those fall through to the Molmo2 default.
    """
    if explicit_tokenizer is not None:
        return explicit_tokenizer
    if info.format in ("olmo_core_dcp", "olmo_core_unsharded"):
        model_id = info.config.get("model_id")
        if isinstance(model_id, str) and model_id:
            return model_id
    return DEFAULT_TOKENIZER


def resolve_max_crops(info: MultimodalCheckpointInfo, explicit_max_crops: int | None) -> int:
    """Pick the multi-crop budget: explicit > checkpoint hint > OLMo-core default."""
    from olmo_core.nn.vision.molmo2_tokens import DEFAULT_MAX_CROPS

    if explicit_max_crops is not None:
        return explicit_max_crops
    if info.format == "mm_olmo_dcp":
        image_cfg = (info.model_config.get("mm_preprocessor") or {}).get("image") or {}
        max_crops = image_cfg.get("max_crops")
        if isinstance(max_crops, int) and max_crops > 0:
            return max_crops
    else:
        dataset = info.config.get("dataset")
        if isinstance(dataset, dict):
            max_crops = dataset.get("max_crops")
            if isinstance(max_crops, int) and max_crops > 0:
                return max_crops
    return DEFAULT_MAX_CROPS


#: Per-image crop budget for multi-image prompts when the checkpoint names none: the
#: released Molmo2 ``max_multi_image_crops``, which mm_olmo's ``MultiImagePreprocessor``
#: uses whenever an example has more than one image.
DEFAULT_MAX_MULTI_IMAGE_CROPS = 8

#: The sequence length mm_olmo's ``eval_molmo2.py`` evaluates Molmo2 models at. A
#: checkpoint's training window is often shorter (16,384 for Molmo2-4B), and many-image
#: prompts such as MMIU's exceed it.
EVAL_MAX_LENGTH = 64_000


def resolve_max_multi_image_crops(
    info: MultimodalCheckpointInfo, explicit_max_multi_image_crops: int | None
) -> int:
    """Pick the per-image crop budget for multi-image prompts: explicit > checkpoint hint > 8."""
    if explicit_max_multi_image_crops is not None:
        return explicit_max_multi_image_crops
    if info.format == "mm_olmo_dcp":
        image_cfg = (info.model_config.get("mm_preprocessor") or {}).get("image") or {}
        max_crops = image_cfg.get("max_multi_image_crops")
    else:
        dataset = info.config.get("dataset")
        max_crops = dataset.get("max_multi_image_crops") if isinstance(dataset, dict) else None
    if isinstance(max_crops, int) and max_crops > 0:
        return max_crops
    return DEFAULT_MAX_MULTI_IMAGE_CROPS


def resolve_max_length(info: MultimodalCheckpointInfo, explicit_max_length: int | None) -> int:
    """Pick the max sequence length: explicit, else the longer of the checkpoint's training
    window and :data:`EVAL_MAX_LENGTH`.

    Only prompts longer than the training window see the difference: they are answered,
    as mm_olmo answers them, instead of failing their instance.
    """
    if explicit_max_length is not None:
        return explicit_max_length
    return max(_checkpoint_max_length(info), EVAL_MAX_LENGTH)


def _checkpoint_max_length(info: MultimodalCheckpointInfo) -> int:
    """The checkpoint's training sequence length, or 4096 when it records none."""
    if info.format == "mm_olmo_dcp":
        max_length = (info.model_config.get("llm") or {}).get("max_sequence_length")
        if isinstance(max_length, int) and max_length > 0:
            return max_length
    else:
        for section in ("dataset", "collator", "train_module"):
            value = info.config.get(section)
            if isinstance(value, dict):
                max_length = value.get("sequence_length") or value.get("max_sequence_length")
                if isinstance(max_length, int) and max_length > 0:
                    return max_length
    return 4096
