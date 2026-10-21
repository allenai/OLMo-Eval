"""Multimodal checkpoint detection, model-config building and weight loading.

Handles the three on-disk flavors of OLMo-core ``MultimodalLM`` weights (raw
OLMo-core trainer checkpoints, consolidated safetensors exports, and mm_olmo /
Molmo2-trainer checkpoints, key-remapped into the OLMo-core layout at load time).
"""

from __future__ import annotations

import json
import logging
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

CheckpointFormat = Literal["olmo_core_dcp", "olmo_core_unsharded", "mm_olmo_dcp"]

_EXPECTED_FORMATS = (
    "expected one of: a raw OLMo-core multimodal checkpoint (config.json with a "
    "MultimodalLMConfig under 'model' + model_and_optim/.metadata), a consolidated "
    "OLMo-core export (olmo_core_config.json + model.safetensors with lm.*/vision.*/"
    "connector.* keys), or an mm_olmo/Molmo2-trainer checkpoint (config.yaml + "
    "model_and_optim/.metadata with model.transformer.*/model.vision_backbone.* keys)"
)

# Old name of MultimodalLMConfig kept by earlier vision-branch checkpoints.
_LEGACY_MULTIMODAL_CLASS_NAMES = {
    "olmo_core.nn.vision.multimodal.MultimodalTransformerConfig": (
        "olmo_core.nn.vision.multimodal.MultimodalLMConfig"
    ),
}

_MULTIMODAL_CLASS_SUFFIXES = ("MultimodalLMConfig", "MultimodalTransformerConfig")

# Default tokenizer for Molmo2-family models: Qwen tokenizer + Molmo2 image


@dataclass(frozen=True)
class MultimodalCheckpointInfo:
    """Resolved facts about a multimodal checkpoint directory."""

    format: CheckpointFormat
    config: dict[str, Any]
    """The full config dict (ExperimentConfig for OLMo-core, the yaml for mm_olmo)."""
    model_config: dict[str, Any]
    """The model section (MultimodalLMConfig dict, or the mm_olmo model yaml)."""


def _config_error(checkpoint_dir: str, reason: str) -> ValueError:
    return ValueError(
        f"Invalid multimodal OLMo-core checkpoint {checkpoint_dir!r}: {reason}. "
        f"{_EXPECTED_FORMATS}."
    )


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open() as f:
        return yaml.safe_load(f)


def _is_multimodal_model_config(model_config: Any) -> bool:
    if not isinstance(model_config, dict):
        return False
    class_name = model_config.get("_CLASS_", "")
    return isinstance(class_name, str) and class_name.endswith(_MULTIMODAL_CLASS_SUFFIXES)


def _is_mm_olmo_model_config(model_config: Any) -> bool:
    return (
        isinstance(model_config, dict)
        and "llm" in model_config
        and "vision_backbone" in model_config
    )


def is_multimodal_checkpoint(checkpoint_dir: str) -> bool:
    """Cheaply check whether a local directory holds multimodal OLMo-core weights.

    Used to route ``provider.kind=olmo_core`` requests to the multimodal
    provider. Returns ``False`` on any read error so callers can fall back to
    the text provider's own validation.
    """
    try:
        detect_checkpoint_format(checkpoint_dir)
        return True
    except Exception:
        return False


def detect_checkpoint_format(checkpoint_dir: str) -> MultimodalCheckpointInfo:
    """Identify which multimodal checkpoint flavor ``checkpoint_dir`` holds.

    :raises ValueError: if the directory matches none of the known formats.
    """
    root = Path(checkpoint_dir)
    if not root.is_dir():
        raise _config_error(checkpoint_dir, "not a local directory")

    unsharded_config = root / "olmo_core_config.json"
    if unsharded_config.is_file() and (root / "model.safetensors").is_file():
        config = _load_json(unsharded_config)
        model_config = config.get("model")
        if not (isinstance(model_config, dict) and _is_multimodal_model_config(model_config)):
            raise _config_error(
                checkpoint_dir,
                "olmo_core_config.json 'model' is not a MultimodalLMConfig",
            )
        return MultimodalCheckpointInfo(
            format="olmo_core_unsharded", config=config, model_config=model_config
        )

    has_dcp = (root / "model_and_optim" / ".metadata").is_file()
    json_config = root / "config.json"
    yaml_config = root / "config.yaml"

    if json_config.is_file() and has_dcp:
        config = _load_json(json_config)
        model_config = config.get("model")
        if isinstance(model_config, dict) and _is_multimodal_model_config(model_config):
            return MultimodalCheckpointInfo(
                format="olmo_core_dcp", config=config, model_config=model_config
            )
        raise _config_error(
            checkpoint_dir,
            "config.json 'model' is not a MultimodalLMConfig (text-only OLMo-core "
            "checkpoints should use the 'olmo_core' provider)",
        )

    if yaml_config.is_file() and has_dcp:
        config = _load_yaml(yaml_config)
        model_config = config.get("model")
        if isinstance(model_config, dict) and _is_mm_olmo_model_config(model_config):
            return MultimodalCheckpointInfo(
                format="mm_olmo_dcp", config=config, model_config=model_config
            )
        raise _config_error(
            checkpoint_dir,
            "config.yaml 'model' does not look like an mm_olmo multimodal model "
            "(missing 'llm'/'vision_backbone')",
        )

    raise _config_error(checkpoint_dir, "no recognizable config/weights layout found")


def _rewrite_legacy_class_names(config: Any) -> Any:
    """Recursively map renamed ``_CLASS_`` names to their current spelling."""
    if isinstance(config, dict):
        out = {key: _rewrite_legacy_class_names(value) for key, value in config.items()}
        class_name = out.get("_CLASS_")
        if isinstance(class_name, str) and class_name in _LEGACY_MULTIMODAL_CLASS_NAMES:
            out["_CLASS_"] = _LEGACY_MULTIMODAL_CLASS_NAMES[class_name]
        return out
    if isinstance(config, list):
        return [_rewrite_legacy_class_names(value) for value in config]
    return config


def build_model_config(
    info: MultimodalCheckpointInfo,
    *,
    image_patch_token_id: int,
    attention_backend: str | None = "torch",
) -> Any:
    """Build the OLMo-core ``MultimodalLMConfig`` for a checkpoint.

    :param image_patch_token_id: The ``<im_patch>`` vocab ID resolved from the
        tokenizer; only used for the mm_olmo format (OLMo-core configs carry
        their own).
    :param attention_backend: Normalize every attention config to this backend
        (``None`` keeps the checkpoint's). Training configs often pin ``flex``
        or ``flash_2``, but eval needs the dense ``torch`` backend: it is the
        one that supports the bidirectional image-token ``or_mask`` AND the
        KV-cached decoding installed by :func:`enable_kv_cache`.
    """
    from olmo_core.nn.vision.multimodal import MultimodalLMConfig

    if info.format in ("olmo_core_dcp", "olmo_core_unsharded"):
        model_config = _rewrite_legacy_class_names(info.model_config)
        if attention_backend is not None:
            # Enum coercion inside from_dict runs before any post-hoc normalization
            # could, and training configs pin backends (`flex`) that a given
            # OLMo-core release may not define — rewrite them in the raw dict.
            model_config = _rewrite_backend_values(model_config, attention_backend)
        cfg = MultimodalLMConfig.from_dict(model_config)
    else:
        cfg = _mm_olmo_model_config_to_multimodal_lm_config(
            info.model_config, image_patch_token_id=image_patch_token_id
        )
    if attention_backend is not None:
        _normalize_attention_backend(cfg, attention_backend)
    return cfg


def _rewrite_backend_values(config: Any, backend: str) -> Any:
    """Replace attention ``backend`` string values in a raw config tree."""
    if isinstance(config, dict):
        return {
            key: (
                backend
                if key == "backend" and isinstance(value, str)
                else _rewrite_backend_values(value, backend)
            )
            for key, value in config.items()
        }
    if isinstance(config, list):
        return [_rewrite_backend_values(value, backend) for value in config]
    return config


def _normalize_attention_backend(cfg: Any, backend: str) -> None:
    """Set ``backend`` on every nested attention/sequence-mixer config."""
    from olmo_core.nn.attention import AttentionBackendName

    backend_value = AttentionBackendName(backend)

    def set_backend(config: Any) -> None:
        mixer = getattr(config, "sequence_mixer", None) or getattr(config, "attention", None)
        if mixer is None and hasattr(config, "backend"):
            mixer = config
        if mixer is not None and hasattr(mixer, "backend"):
            setattr(mixer, "backend", backend_value)  # noqa: B010

    for section in (cfg.lm, cfg.vision, cfg.connector):
        apply = getattr(section, "apply", None)
        if callable(apply):
            apply(set_backend)
        else:
            set_backend(section)


def _mm_olmo_model_config_to_multimodal_lm_config(
    model_config: dict[str, Any], *, image_patch_token_id: int
) -> Any:
    """Translate an mm_olmo (``video_olmo``) model config into a ``MultimodalLMConfig``.

    Builds lightweight HF-Molmo2-config lookalikes from the mm_olmo yaml fields
    and reuses OLMo-core's ``molmo2_config_from_hf_config`` so the architecture
    dispatch (qwen3_4B / qwen3_8B / olmo3_7B) and ViT-truncation logic stay in
    one place.
    """
    try:
        from olmo_core.nn.vision.molmo2_loader import molmo2_config_from_hf_config
    except ImportError as e:
        raise ImportError(
            "Loading mm_olmo-format checkpoints needs olmo_core.nn.vision.molmo2_loader, "
            "which no released ai2-olmo-core ships yet — install OLMo-core's vision branch "
            "(e.g. -o provider.package=https://github.com/allenai/OLMo-core.git@vision). "
            "Native OLMo-core checkpoints do not need it."
        ) from e

    llm = model_config["llm"]
    backbone = model_config["vision_backbone"]
    vit = backbone["vit"]

    text_config = types.SimpleNamespace(
        hidden_size=llm["d_model"],
        num_hidden_layers=llm["n_layers"],
        qk_norm_type=(
            llm.get("attention_layer_norm_type") if llm.get("attention_layer_norm") else None
        ),
        rope_theta=llm["rope_theta"],
        vocab_size=llm["embedding_size"],
        additional_vocab_size=llm["additional_vocab_size"],
    )
    vit_config = types.SimpleNamespace(
        image_default_input_size=tuple(vit["image_default_input_size"]),
        image_patch_size=vit["image_patch_size"],
        hidden_size=vit["image_emb_dim"],
        num_attention_heads=vit["image_num_heads"],
        num_key_value_heads=vit["image_num_key_value_heads"],
        num_hidden_layers=vit["image_num_layers"],
        head_dim=vit["image_head_dim"],
        intermediate_size=vit["image_mlp_dim"],
        hidden_act=vit["image_mlp_activations"],
        image_num_pos=vit["image_num_pos"],
        layer_norm_eps=vit["image_norm_eps"],
    )
    # mm_olmo's image_pooling_2d reuses the ViT attention dims and half the LM's
    # fused MLP width, matching the released Molmo2 adapter_config.
    adapter_config = types.SimpleNamespace(
        vit_layers=list(backbone["vit_layers"]),
        num_attention_heads=vit["image_num_heads"],
        num_key_value_heads=vit["image_num_key_value_heads"],
        head_dim=vit["image_head_dim"],
        text_hidden_size=llm["d_model"],
        pooling_attention_mask=backbone["pooling_attention_mask"],
        intermediate_size=llm["mlp_hidden_size"] // 2,
    )
    hf_like_config = types.SimpleNamespace(
        text_config=text_config,
        vit_config=vit_config,
        adapter_config=adapter_config,
        image_patch_id=image_patch_token_id,
    )
    if backbone.get("image_pooling_2d") != "attention_meanq":
        raise ValueError(
            "mm_olmo checkpoint uses unsupported image pooling "
            f"{backbone.get('image_pooling_2d')!r}; only 'attention_meanq' is supported"
        )
    if backbone.get("image_projector") != "mlp":
        raise ValueError(
            "mm_olmo checkpoint uses unsupported image projector "
            f"{backbone.get('image_projector')!r}; only 'mlp' is supported"
        )
    return molmo2_config_from_hf_config(hf_like_config)


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------


def _untie_lm_head(model: Any) -> None:
    """Give ``lm_head.w_out`` its own storage when word embeddings are tied.

    Checkpoints in the wild store *different* tensors for
    ``lm.embeddings.weight`` and ``lm.lm_head.w_out.weight`` even when the
    config says ``tie_word_embeddings`` (the extra image-token rows legitimately
    differ). Loading both into a shared parameter would let one silently
    clobber the other, so break the tie before loading.
    """
    import torch.nn as nn

    lm = model.lm
    if getattr(lm, "tie_word_embeddings", False):
        lm.lm_head.w_out.weight = nn.Parameter(lm.embeddings.weight.detach().clone())


def _ensure_mm_olmo_unpickle_shim() -> None:
    """Register a stub ``olmo.train.remote_filesystem`` module for metadata unpickling.

    mm_olmo distributed checkpoints pickle a custom ``_StorageInfo`` dataclass
    (fields ``relative_path`` / ``offset`` / ``length``, mirroring torch's) into
    ``model_and_optim/.metadata``. The mm_olmo package isn't installed in the
    eval environment, so provide a minimal class with the same module path for
    pickle to resolve. No-op when the real package (or the shim) is importable.
    """
    import importlib.util

    module_name = "olmo.train.remote_filesystem"
    # The shim itself registers with ``__spec__ = None``, which makes ``find_spec``
    # raise ValueError on the next call — so consult ``sys.modules`` first.
    if module_name in sys.modules:
        return
    try:
        if importlib.util.find_spec(module_name) is not None:
            return
    except (ImportError, ModuleNotFoundError, ValueError):
        pass

    from dataclasses import dataclass as _dataclass

    @_dataclass
    class _StorageInfo:
        relative_path: str
        offset: int
        length: int

    olmo_mod = sys.modules.setdefault("olmo", types.ModuleType("olmo"))
    train_mod = sys.modules.setdefault("olmo.train", types.ModuleType("olmo.train"))
    fs_mod = sys.modules.setdefault(
        "olmo.train.remote_filesystem", types.ModuleType("olmo.train.remote_filesystem")
    )
    setattr(olmo_mod, "train", train_mod)  # noqa: B010
    setattr(train_mod, "remote_filesystem", fs_mod)  # noqa: B010
    _StorageInfo.__module__ = "olmo.train.remote_filesystem"
    setattr(fs_mod, "_StorageInfo", _StorageInfo)  # noqa: B010


def _mm_olmo_key_to_hf_key(key: str) -> str:
    """Rename an mm_olmo trainer tensor key to released-Molmo2 HF naming.

    The mm_olmo trainer keeps attention/MLP tensors directly on the block
    (``blocks.N.att_proj``); the HF release nests them (``blocks.N.self_attn.
    att_proj`` / ``blocks.N.mlp.ff_proj``). Everything else (wte, ln_f, the
    whole vision backbone) matches HF naming already.

    The one non-block rename is the untied LM head: mm_olmo stores it as the
    transformer's own ``ff_out``, which HF calls ``lm_head``. Weight-tied
    checkpoints have no such tensor.
    """
    parts = key.split(".")
    if parts == ["model", "transformer", "ff_out", "weight"]:
        return "lm_head.weight"
    if len(parts) >= 4 and parts[:2] == ["model", "transformer"] and parts[2] == "blocks":
        leaf = parts[4]
        if leaf in ("att_proj", "attn_out", "q_norm", "k_norm"):
            parts.insert(4, "self_attn")
        elif leaf in ("ff_proj", "ff_out"):
            parts.insert(4, "mlp")
        return ".".join(parts)
    return key


def _load_mm_olmo_state_dict(checkpoint_dir: str, model_cfg: Any) -> dict[str, torch.Tensor]:
    """Consolidate an mm_olmo distributed checkpoint into MultimodalLM keys.

    Reads the (unsharded) ``model.*`` tensors, renames them to released-Molmo2
    HF naming, synthesizes the tied ``lm_head`` when absent, and runs OLMo-core's
    ``molmo2_hf_state_dict_to_multimodal_lm`` for the fused-weight splits and
    patch-embedding permutation.
    """
    from olmo_core.distributed.checkpoint import get_checkpoint_metadata, load_keys

    try:
        from olmo_core.nn.vision.molmo2_loader import molmo2_hf_state_dict_to_multimodal_lm
    except ImportError as e:
        raise ImportError(
            "Loading mm_olmo-format checkpoints needs olmo_core.nn.vision.molmo2_loader, "
            "which no released ai2-olmo-core ships yet — install OLMo-core's vision branch "
            "(e.g. -o provider.package=https://github.com/allenai/OLMo-core.git@vision). "
            "Native OLMo-core checkpoints do not need it."
        ) from e

    _ensure_mm_olmo_unpickle_shim()

    dcp_dir = str(Path(checkpoint_dir) / "model_and_optim")
    metadata = get_checkpoint_metadata(dcp_dir)
    model_keys = sorted(key for key in metadata.state_dict_metadata if key.startswith("model."))
    if not model_keys:
        raise _config_error(checkpoint_dir, "distributed checkpoint has no 'model.*' keys")

    logger.info("Loading %d unsharded tensors from %s", len(model_keys), dcp_dir)
    hf_state_dict: dict[str, Any] = {}
    for key, tensor in zip(model_keys, load_keys(dcp_dir, model_keys), strict=True):
        hf_state_dict[_mm_olmo_key_to_hf_key(key)] = tensor

    if "lm_head.weight" not in hf_state_dict:
        # Weight-tied mm_olmo model (`can_predict_extra_tokens=False`): logits use
        # the base-vocab embedding table; the converter zero-pads the extra rows.
        # Substituting the embeddings for a head the model does *not* tie produces
        # fluent-looking garbage rather than an error, so say when it happens.
        base_embedding = hf_state_dict.get("model.transformer.wte.embedding")
        if base_embedding is None:
            raise _config_error(
                checkpoint_dir, "missing both 'lm_head' and 'wte.embedding' tensors"
            )
        logger.info(
            "No LM head tensor in %s; treating the checkpoint as weight-tied and using "
            "the embedding table for the head.",
            checkpoint_dir,
        )
        hf_state_dict["lm_head.weight"] = base_embedding

    state_dict = molmo2_hf_state_dict_to_multimodal_lm(hf_state_dict, model_cfg)
    _verify_lm_head_is_untied(state_dict, model_cfg, checkpoint_dir)
    return state_dict


def _verify_lm_head_is_untied(
    state_dict: dict[str, torch.Tensor], model_cfg: Any, checkpoint_dir: str
) -> None:
    """Reject a load that gave an untied model the embedding table as its LM head.

    An untied checkpoint keeps its output projection in a separate tensor whose
    shape matches the embedding table, so a read that returns the wrong one of
    the pair loads cleanly and leaves every other weight intact. The model then
    generates from the wrong projection: the output is noise, the score sits at
    chance, and nothing raises. Compare the two tensors instead, since a trained
    untied head is never bit-for-bit equal to the embeddings.
    """
    import torch

    lm_cfg = getattr(model_cfg, "lm", None)
    if lm_cfg is None or getattr(lm_cfg, "tie_word_embeddings", False):
        return

    head = state_dict.get("lm.lm_head.w_out.weight")
    embeddings = state_dict.get("lm.embeddings.weight")
    if head is None or embeddings is None or head.shape != embeddings.shape:
        return

    if torch.equal(head, embeddings):
        raise _config_error(
            checkpoint_dir,
            "the LM head loaded byte-identical to the token embeddings even though the "
            "model is not weight-tied, which means the checkpoint read returned the wrong "
            "tensor; retry the load",
        )


def load_checkpoint_weights(
    info: MultimodalCheckpointInfo, checkpoint_dir: str, model: Any
) -> None:
    """Load checkpoint weights into a freshly built ``MultimodalLM`` (on CPU)."""
    _untie_lm_head(model)

    if info.format == "olmo_core_dcp":
        from olmo_core.distributed.checkpoint import load_model_and_optim_state

        load_model_and_optim_state(str(Path(checkpoint_dir) / "model_and_optim"), model)
        return

    if info.format == "olmo_core_unsharded":
        from safetensors.torch import load_file

        state_dict = load_file(str(Path(checkpoint_dir) / "model.safetensors"))
        model.load_state_dict(state_dict)
        return

    state_dict = _load_mm_olmo_state_dict(checkpoint_dir, model.cfg)
    model.load_state_dict(state_dict)
