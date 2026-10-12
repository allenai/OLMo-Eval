#!/usr/bin/env python
"""Export an OLMo-core Molmo2 DCP checkpoint to released-Molmo2 HF format.

Training writes a *sharded distributed* checkpoint. Both OLMo-core and mm_olmo layouts
are supported (``olmo_core_dcp`` and ``mm_olmo_dcp``)::

    step30000/
      config.json          # OLMo-core: serialized MultimodalLMConfig under "model"
      config.yaml          # mm_olmo: the released-model layout
      model_and_optim/     # __0_0.distcp, __0_1.distcp, ...

vLLM cannot read that. It needs an HF directory -- ``config.json`` with
``architectures``, HF-named ``*.safetensors``, and the processor/tokenizer/modeling
files. This script produces one, which unlocks the vLLM provider (measured 5-10x over
the HF provider, and far more over ``olmo_core_vlm``, which decodes at a fixed chunk
size) for our own checkpoints rather than only for external comparison models.

The weight mapping is not reimplemented here: it calls OLMo-core's
:func:`multimodal_lm_state_dict_to_hf`, documented as the inverse of the loader used to
*read* released Molmo2 weights. Auxiliary files are copied from the reference HF repo
recorded as ``model_id`` in the checkpoint config (e.g. ``allenai/Molmo2-4B``), since the
architecture is unchanged by fine-tuning -- only the weights differ.

**Verify before trusting.** A subtly wrong conversion is worse than none: it produces
plausible scores that are quietly incorrect. Run the same task through
``provider.kind=olmo_core_vlm`` on the DCP and ``provider.kind=vllm`` on the export and
require the scores to agree.

Usage:
    python tools/olmo_core_to_hf/export.py \\
        --checkpoint /path/to/run/step30000 \\
        --out /path/to/hf-exports/run-step30000 \\
        --reference allenai/Molmo2-4B

``--reference`` supplies the HF config/processor/modeling files. It defaults to the
``model_id`` recorded in an OLMo-core ``config.json``, but **released mm_olmo checkpoints
have no ``model_id``**, so it is required for those. Missing it fails loudly.

``--verify`` compares the export against the reference tensor by tensor -- no GPU, no
inference. On an unmodified checkpoint it proves the conversion bitwise, which is strictly
stronger than comparing eval scores: matching scores can hide compensating differences.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

logger = logging.getLogger("olmo_core_to_hf")

#: Everything in the reference repo EXCEPT its weights is needed verbatim: HF config,
#: remote-code modeling/processing, tokenizer, and -- easy to miss -- the per-modality
#: preprocessor configs whose ``auto_map`` tells transformers where each class lives.
#: An allowlist was tried first and silently omitted three: without
#: ``video_preprocessor_config.json`` the processor cannot find ``Molmo2VideoProcessor``
#: (it is defined in ``video_processing_molmo2.py``, but nothing points there), and
#: ``added_tokens.json`` / ``chat_template.jinja`` carry the image special tokens and the
#: chat template. Excluding only the weights is the safe direction to be wrong in.
_SKIP_EXACT = frozenset({"model.safetensors.index.json"})
_SKIP_SUFFIXES = (".png", ".jpg", ".jpeg")


def _is_weight_file(name: str) -> bool:
    return name.startswith("model") and name.endswith(".safetensors")


def should_copy(name: str) -> bool:
    """Copy every reference file except its weights, index and images."""
    if _is_weight_file(name) or name in _SKIP_EXACT:
        return False
    return not name.lower().endswith(_SKIP_SUFFIXES)


def load_olmo_core_state_dict(checkpoint: Path):
    """Build the MultimodalLM from the checkpoint config and load its DCP weights.

    Mirrors ``OlmoCoreVLMProvider.__init__`` rather than reimplementing it: resolve the
    tokenizer the checkpoint names, take ``<im_patch>`` from it (``build_model_config``
    needs the id and cross-checks it against the config), build on CPU, then load.
    """
    from transformers import AutoTokenizer

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from olmo_eval.inference.providers.olmo_core_vlm import checkpoint as vlm_checkpoint
    from olmo_eval.inference.providers.olmo_core_vlm.preprocessing import (
        resolve_tokenizer_path,
    )

    info = vlm_checkpoint.detect_checkpoint_format(str(checkpoint))
    logger.info("checkpoint format: %s", info.format)

    tokenizer_path = resolve_tokenizer_path(info, None)
    logger.info("tokenizer: %s", tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    patch_id = tokenizer.convert_tokens_to_ids("<im_patch>")

    cfg = vlm_checkpoint.build_model_config(
        info, image_patch_token_id=patch_id, attention_backend="torch"
    )
    logger.info("building MultimodalLM on CPU and loading DCP weights (the slow part)")
    model = cfg.build(init_device="cpu")
    vlm_checkpoint.load_checkpoint_weights(info, str(checkpoint), model)
    return model.state_dict(), cfg, info


def verify_export(out: Path, reference: str, ref_dir: Path) -> int:
    """Compare an export against the reference repo tensor by tensor.

    No GPU and no inference: for an *unmodified* checkpoint this proves the conversion
    bitwise, which is strictly stronger than comparing eval scores -- 72.50 vs 72.50 at 98%
    text agreement means some outputs differed and happened to net to zero on a lossy metric.

    A fine-tuned checkpoint will legitimately differ in values; the structural checks
    (missing / extra / shape) still apply and are what catch a broken key mapping.

    :returns: process exit code -- non-zero if the structure disagrees.
    """
    import torch
    from safetensors.torch import load_file

    export = load_file(str(out / "model.safetensors"))
    ref: dict[str, torch.Tensor] = {}
    shards = sorted(ref_dir.glob("model*.safetensors"))
    if not shards:
        raise SystemExit(f"no model*.safetensors under {ref_dir}")
    for shard in shards:
        ref.update(load_file(str(shard)))

    missing = sorted(set(ref) - set(export))
    extra = sorted(set(export) - set(ref))
    shared = sorted(set(ref) & set(export))
    mismatched = [k for k in shared if tuple(export[k].shape) != tuple(ref[k].shape)]

    print(f"export vs {reference}")
    print(
        f"  tensors: {len(shared)} / {len(ref)} present, {len(missing)} missing, "
        f"{len(extra)} extra, {len(mismatched)} shape mismatches"
    )
    for label, keys in (("missing", missing), ("extra", extra), ("shape", mismatched)):
        for key in keys[:10]:
            print(f"    {label}: {key}")
        if len(keys) > 10:
            print(f"    ... and {len(keys) - 10} more {label}")

    identical, differing_tensors, differing_elements = 0, [], 0
    for key in shared:
        if key in mismatched:
            continue
        # Cast the reference to the export's dtype: the released weights ship fp32 while
        # the export defaults to bf16, and that cast is the only intended difference.
        same = torch.equal(export[key], ref[key].to(export[key].dtype))
        if same:
            identical += 1
        else:
            differing_tensors.append(key)
            differing_elements += int((export[key] != ref[key].to(export[key].dtype)).sum())
    print(
        f"  export == reference.to({export[shared[0]].dtype if shared else 'n/a'}), bitwise: "
        f"{identical}/{len(shared) - len(mismatched)} tensors, "
        f"{differing_elements} differing elements"
    )
    for key in differing_tensors[:10]:
        print(f"    differs: {key}")
    if len(differing_tensors) > 10:
        print(f"    ... and {len(differing_tensors) - 10} more differing")

    structural_ok = not missing and not extra and not mismatched
    if not structural_ok:
        print("STRUCTURAL MISMATCH -- the key mapping is wrong, do not use this export")
        return 1
    if differing_tensors:
        print(
            "Structure matches; values differ. Expected for a fine-tuned checkpoint, "
            "a bug for an unmodified one."
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="OLMo-core step directory (DCP)")
    parser.add_argument("--out", required=True, help="Destination HF directory")
    parser.add_argument(
        "--reference",
        default=None,
        help="HF repo/dir supplying config+processor+modeling files. Defaults to the "
        "checkpoint config's model_id.",
    )
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Compare an existing export against the reference tensor by tensor and exit. "
        "No GPU, no inference.",
    )
    parser.add_argument(
        "--skip-weights",
        action="store_true",
        help="Only refresh the auxiliary files, reusing an existing model.safetensors.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    )

    import torch
    from safetensors.torch import save_file

    checkpoint = Path(args.checkpoint)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.skip_weights or args.verify:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
        from olmo_eval.inference.providers.olmo_core_vlm.checkpoint import (
            detect_checkpoint_format,
        )

        info = detect_checkpoint_format(str(checkpoint))
        hf_state = None
    else:
        oc_state, cfg, info = load_olmo_core_state_dict(checkpoint)
        logger.info("loaded %d OLMo-core tensors", len(oc_state))

    reference = args.reference or info.config.get("model_id")
    if not reference:
        raise SystemExit("no --reference and no model_id in the checkpoint config")
    logger.info("reference HF repo for config/processor/modeling: %s", reference)

    from transformers import AutoConfig

    ref_cfg = AutoConfig.from_pretrained(reference, trust_remote_code=True)
    base_vocab_size = ref_cfg.text_config.vocab_size
    logger.info("base_vocab_size from reference: %d", base_vocab_size)

    if args.verify:
        ref_dir_v = Path(reference)
        if not ref_dir_v.is_dir():
            from huggingface_hub import snapshot_download

            ref_dir_v = Path(snapshot_download(reference))
        return verify_export(out, str(reference), ref_dir_v)

    weights_path = out / "model.safetensors"
    if args.skip_weights:
        if not weights_path.exists():
            raise SystemExit(f"--skip-weights but {weights_path} is absent")
        total = 0
        logger.info("reusing existing %s", weights_path)
    else:
        from olmo_core.nn.vision.molmo2_loader import multimodal_lm_state_dict_to_hf

        hf_state = multimodal_lm_state_dict_to_hf(oc_state, cfg, base_vocab_size=base_vocab_size)
        del oc_state
        logger.info("converted to %d HF tensors", len(hf_state))

        target_dtype = getattr(torch, args.dtype)
        hf_state = {k: v.to(target_dtype).contiguous() for k, v in hf_state.items()}
        save_file(hf_state, str(weights_path), metadata={"format": "pt"})
        total = sum(v.numel() for v in hf_state.values())
        logger.info(
            "wrote %s (%d params, %.1f GB)", weights_path, total, weights_path.stat().st_size / 1e9
        )

    # Auxiliary files: resolve the reference locally (snapshot_download is a no-op when
    # already cached) then copy, so the export is self-contained and needs no Hub access.
    from huggingface_hub import snapshot_download

    ref_dir = Path(reference)
    if not ref_dir.is_dir():
        ref_dir = Path(snapshot_download(reference))
    copied, skipped = [], []
    for src in sorted(ref_dir.iterdir()):
        if not src.is_file():
            continue
        if should_copy(src.name):
            shutil.copy2(src, out / src.name)
            copied.append(src.name)
        else:
            skipped.append(src.name)
    # The copied config carries torch_dtype: null, so nothing in the output directory
    # would record the precision the weights were actually written in.
    config_path = out / "config.json"
    if config_path.is_file():
        config_json = json.loads(config_path.read_text())
        config_json["torch_dtype"] = args.dtype
        config_path.write_text(json.dumps(config_json, indent=2))
        logger.info("recorded torch_dtype=%s in %s", args.dtype, config_path)

    logger.info("copied %d aux files: %s", len(copied), ", ".join(copied))
    logger.info("skipped %d: %s", len(skipped), ", ".join(skipped))

    # Record provenance so a stray export directory is traceable.
    (out / "olmo_core_export.json").write_text(
        json.dumps(
            {
                "source_checkpoint": str(checkpoint),
                "reference_model": str(reference),
                "dtype": args.dtype,
                "num_tensors": len(hf_state) if hf_state else None,
                "num_params": total,
            },
            indent=2,
        )
    )
    logger.info("done: %s", out)
    logger.warning(
        "VERIFY before use: run the same task via provider.kind=olmo_core_vlm on the DCP and "
        "provider.kind=vllm on this export, and require the scores to agree."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
