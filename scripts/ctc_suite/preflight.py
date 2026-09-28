"""
Pre-flight for one CTC suite eval job: run inside the job, after the install, before ``olmo-eval run``.
Prints the checkpoint directory to evaluate as its last stdout line; exits non-zero (so the job stops
before loading the model) if the eval would not see the model the way training did.

1. **Checkpoint config vs the installed OLMo-core.** A checkpoint trained on another OLMo-core branch
   can carry config fields this one does not define (e.g. the setA SFT runs, trained on
   ``prasann/ctc-setA-sft``, serialize ``AttentionConfig.vec_dim`` / ``n_summary_tokens`` / ... as
   ``null``), and ``TransformerConfig.from_dict`` rejects any unknown field. An unknown field that is
   ``null`` configures nothing, so it is dropped: the config is rewritten into ``--view`` with every
   other entry of the checkpoint symlinked beside it. An unknown field with a VALUE turns on a feature
   this OLMo-core lacks -- that is a hard failure, not something to drop.

2. **Chat-template alignment.** CTC SFT data is rendered as ONE chat-template call over (prompt,
   answer), so the tokens a model saw before its answer are a prefix of that full render. The eval
   renders the prompt alone with ``add_generation_prompt=True``. That is only train-identical if the
   template's generation prompt is a token prefix of its own rendered conversation. Qwen3.5-0.8B's
   template is (``<think>\\n\\n</think>\\n\\n``); Qwen3.5-4B's is NOT (it opens ``<think>\\n`` for a
   generation prompt but renders a past answer as ``<think>\\n\\n</think>\\n\\n...``), so passing
   ``--tokenizer Qwen/Qwen3.5-4B`` would evaluate every prompt out of format. The check fails instead.

    python scripts/ctc_suite/preflight.py --ckpt <step dir> --tokenizer Qwen/Qwen3.5-0.8B \\
        --view /tmp/ckpt_view
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import os
import sys

CLASS_FIELD = "_CLASS_"


def _log(msg: str) -> None:
    print(f"[preflight] {msg}", file=sys.stderr, flush=True)


def _resolve(cls_name: str):
    module, _, name = cls_name.rpartition(".")
    try:
        return getattr(importlib.import_module(module), name, None)
    except ImportError:
        return None


def drop_unknown_null_fields(node, path: str = "model") -> list[str]:
    """
    Remove, in place, every field that the installed OLMo-core's config class does not define and
    whose value is ``None``.

    :param node: A decoded ``config.json`` subtree (dicts carry their class in ``_CLASS_``).
    :param path: Dotted path of ``node``, for messages.

    :returns: The dotted paths of the dropped fields.

    :raises SystemExit: If an unknown field has a non-null value.
    """
    dropped: list[str] = []
    if isinstance(node, dict):
        cls = _resolve(node[CLASS_FIELD]) if isinstance(node.get(CLASS_FIELD), str) else None
        if cls is not None and dataclasses.is_dataclass(cls):
            known = {f.name for f in dataclasses.fields(cls)}
            known |= set(getattr(cls, "_IGNORE_FIELDS", None) or ()) | {CLASS_FIELD, "type"}
            for key in [k for k in node if k not in known]:
                if node[key] is not None:
                    raise SystemExit(
                        f"[preflight] {path}.{key}={node[key]!r} is set in the checkpoint config, "
                        f"but the installed OLMo-core's {cls.__name__} has no such field: install the "
                        "OLMo-core the checkpoint was trained with (--olmo-core-ref)"
                    )
                del node[key]
                dropped.append(f"{path}.{key}")
        for key, value in node.items():
            dropped += drop_unknown_null_fields(value, f"{path}.{key}")
    elif isinstance(node, list):
        for i, value in enumerate(node):
            dropped += drop_unknown_null_fields(value, f"{path}.{i}")
    return dropped


def resolve_checkpoint(ckpt: str, view: str) -> str:
    """
    :returns: ``ckpt`` if the installed OLMo-core builds its model config as-is, else ``view``
        holding the compatible config (see the module docstring).
    """
    from olmo_core.nn.transformer import TransformerConfig

    with open(os.path.join(ckpt, "config.json")) as f:
        config = json.load(f)
    dropped = drop_unknown_null_fields(config["model"])
    TransformerConfig.from_dict(config["model"])  # raises if it still does not build
    if not dropped:
        _log(f"checkpoint config builds as-is: {ckpt}")
        return ckpt
    _log(f"dropped {len(dropped)} null field(s) unknown to this OLMo-core: {', '.join(dropped)}")
    os.makedirs(view, exist_ok=True)
    for name in os.listdir(ckpt):
        if name != "config.json" and not os.path.lexists(os.path.join(view, name)):
            os.symlink(os.path.join(ckpt, name), os.path.join(view, name))
    with open(os.path.join(view, "config.json"), "w") as f:
        json.dump(config, f)
    _log(f"evaluating through {view} (weights symlinked from {ckpt})")
    return view


def check_chat_prefix(tokenizer: str) -> None:
    """
    :raises SystemExit: If ``tokenizer``'s generation prompt is not a token prefix of the same
        conversation rendered with an assistant answer (see the module docstring).
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer)
    user = [{"role": "user", "content": "Documents:\n<|box_start|>[1] a<|box_end|>\nWhich one?"}]
    prompt = tok.apply_chat_template(user, tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template(
        user + [{"role": "assistant", "content": "[1]"}], tokenize=False
    )
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    f_ids = tok(full, add_special_tokens=False)["input_ids"]
    if not full.startswith(prompt) or f_ids[: len(p_ids)] != p_ids:
        raise SystemExit(
            f"[preflight] {tokenizer}'s generation prompt ends {prompt[-24:]!r}, which is not a token "
            f"prefix of the trained conversation ({full[len(prompt) - 24 :][:48]!r}): every prompt "
            "would be out of format for SFT data rendered in one call. Use a tokenizer whose "
            "template closes the empty thinking block (e.g. Qwen/Qwen3.5-0.8B)"
        )
    _log(f"chat template OK: generation prompt ends {prompt[-24:]!r}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ckpt", required=True, help="olmo-core step dir")
    ap.add_argument("--tokenizer", required=True, help="the provider's tokenizer")
    ap.add_argument("--view", required=True, help="where to write a compatible config if needed")
    args = ap.parse_args()
    check_chat_prefix(args.tokenizer)
    print(resolve_checkpoint(args.ckpt, args.view))


if __name__ == "__main__":
    main()
