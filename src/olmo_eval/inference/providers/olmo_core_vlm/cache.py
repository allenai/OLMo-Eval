"""KV-cached decoding support for the multimodal provider.

The stock ``torch``/``flex`` attention backends that support the bidirectional
image-token mask reject KV caching, so a cache-capable dense-SDPA subclass is
installed at load time, driven by explicit per-row RoPE positions.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


def build_decode_attention_mask(
    *,
    seq_len: int,
    pos: int,
    total: int,
    device: Any,
    or_mask: Any = None,
    and_mask: Any = None,
    cache_leftpad: Any = None,
) -> Any:
    """The boolean SDPA mask for one cached decode forward, or ``None`` for plain causal.

    ``seq_len`` query rows sit at absolute positions ``pos .. pos+seq_len-1`` over
    ``total`` cached keys. Causality is expressed against absolute positions;
    ``or_mask`` re-opens key columns (bidirectional image tokens), ``and_mask``
    closes them, and both are left-padded to the key axis when they only cover the
    current step. With left padding, real queries never see pad-slot keys while
    pad-slot queries keep their causal rows so no softmax row is fully masked.
    """
    import torch

    has_leftpad = cache_leftpad is not None
    if seq_len <= 1 and or_mask is None and and_mask is None and not has_leftpad:
        return None

    base = torch.ones(seq_len, total, device=device, dtype=torch.bool).tril(diagonal=pos)

    def _align_keys(mask: Any, pad_value: bool) -> Any:
        mask = mask.to(device=device, dtype=torch.bool)
        missing = total - mask.shape[-1]
        if missing > 0:
            pad = torch.full((*mask.shape[:-1], missing), pad_value, device=device)
            mask = torch.cat([pad, mask], dim=-1)
        return mask

    if or_mask is not None:
        base = base | _align_keys(or_mask, False)
    if and_mask is not None:
        base = base & _align_keys(and_mask, True)
    if has_leftpad:
        leftpad = cache_leftpad.to(device=device, dtype=torch.long)
        key_ok = torch.arange(total, device=device) >= leftpad[:, None]
        q_abs = pos + torch.arange(seq_len, device=device)
        pad_query = q_abs[None, :] < leftpad[:, None]
        base = base & (key_ok[:, None, None, :] | pad_query[:, None, :, None])
    return base


_CACHED_BACKEND_CLS: type | None = None


def _cached_torch_backend_class() -> type:
    """Build (once) a ``TorchAttentionBackend`` subclass that supports KV caching.

    OLMo-core's dense ``torch`` backend is the only one that supports the
    multimodal ``or_mask`` / ``and_mask``, but it rejects KV caching — so
    multimodal decoding would otherwise re-run the LM over the whole sequence
    for every generated token. The ``Attention`` module already carries all the
    cache plumbing (``KVCacheManager``, RoPE ``start_pos`` from
    ``current_position()``); this subclass adds the missing piece: write the
    new K/V into the cache and run SDPA over the cached prefix.

    Scope (asserted): no sliding window, no context parallelism, no
    intra-document masking, no attention sinks. Queries at absolute positions ``pos..pos+T-1``
    attend keys ``0..pos+T-1`` causally; ``or_mask`` / ``and_mask`` (sized
    ``(B, 1, T, T)`` over the current forward's tokens) are aligned to the key
    axis by left-padding, which is exact for the prefill call (``pos == 0``,
    the only call that passes them).

    Variable-length batches are supported through cache left-padding: rows are
    right-aligned in the cache (``kv_cache_manager.cache_leftpad`` holds each
    row's pad length, set by :func:`prepare_kv_caches`) so every row's last
    prompt token lands in the final prefill slot and all rows share one write
    position per decode step. Real queries then must not attend the pad slots
    (their K/V is garbage from pad tokens), while pad-slot queries keep their
    causal rows so no softmax row is fully masked (a fully-masked row turns
    into NaN attention that would poison later layers through the cached K/V).
    """
    global _CACHED_BACKEND_CLS
    if _CACHED_BACKEND_CLS is not None:
        return _CACHED_BACKEND_CLS

    import torch
    import torch.nn.functional as F
    from olmo_core.nn.attention.backend import TorchAttentionBackend, _repeat_kv

    class CachedTorchAttentionBackend(TorchAttentionBackend):
        @classmethod
        def assert_supports_kv_cache(cls) -> None:
            pass

        def forward(  # noqa: C901
            self,
            qkv,
            cu_doc_lens=None,
            cu_doc_lens_q=None,
            cu_doc_lens_k=None,
            max_doc_len=None,
            max_doc_len_q=None,
            max_doc_len_k=None,
            local_k_slice=None,
            kv_cache_manager=None,
            or_mask=None,
            and_mask=None,
            sinks=None,
        ):
            if kv_cache_manager is None:
                # `sinks` exists only in OLMo-core builds whose attention backends take
                # attention-sink logits; pass it through only when set, so older builds work.
                sink_kwargs = {} if sinks is None else {"sinks": sinks}
                return super().forward(
                    qkv,
                    cu_doc_lens=cu_doc_lens,
                    cu_doc_lens_q=cu_doc_lens_q,
                    cu_doc_lens_k=cu_doc_lens_k,
                    max_doc_len=max_doc_len,
                    max_doc_len_q=max_doc_len_q,
                    max_doc_len_k=max_doc_len_k,
                    local_k_slice=local_k_slice,
                    or_mask=or_mask,
                    and_mask=and_mask,
                    **sink_kwargs,
                )

            if isinstance(qkv, torch.Tensor):
                raise RuntimeError(f"'{type(self).__name__}' doesn't support packed QKV")
            if self.window_size != (-1, -1):
                raise RuntimeError(
                    f"'{type(self).__name__}' doesn't support KV caching with sliding windows"
                )
            if any(
                opt is not None
                for opt in (
                    cu_doc_lens,
                    cu_doc_lens_q,
                    cu_doc_lens_k,
                    max_doc_len,
                    max_doc_len_q,
                    max_doc_len_k,
                )
            ):
                raise RuntimeError(
                    f"'{type(self).__name__}' doesn't support intra-document masking"
                )
            if self.cp_enabled:
                raise RuntimeError(
                    f"'{type(self).__name__}' doesn't support KV caching with context parallelism"
                )
            if sinks is not None:
                raise RuntimeError(
                    f"'{type(self).__name__}' doesn't support KV caching with attention sinks"
                )

            q, k, v = qkv
            seq_len = q.shape[1]
            # CPU-side mirror of ``cache_seqlens`` to avoid a GPU sync per layer
            # per step. ``Attention.sdpa`` calls ``update_seqlen(seq_len)`` right
            # after this forward, so advance the mirror by the same amount.
            pos = getattr(kv_cache_manager, "_position_mirror", None)
            if pos is None:
                pos = int(kv_cache_manager.cache_seqlens.item())
            kv_cache_manager._position_mirror = pos + seq_len
            total = pos + seq_len

            k_cache, v_cache = kv_cache_manager.k_cache, kv_cache_manager.v_cache
            if total > k_cache.shape[1]:
                raise RuntimeError(f"KV cache overflow: {total} > allocated {k_cache.shape[1]}")
            k_cache[:, pos:total] = k.to(k_cache.dtype)
            v_cache[:, pos:total] = v.to(v_cache.dtype)
            k_full = k_cache[:, :total].to(q.dtype)
            v_full = v_cache[:, :total].to(q.dtype)

            has_leftpad = getattr(kv_cache_manager, "_has_leftpad", False)
            attn_mask = build_decode_attention_mask(
                seq_len=seq_len,
                pos=pos,
                total=total,
                device=q.device,
                or_mask=or_mask,
                and_mask=and_mask,
                cache_leftpad=kv_cache_manager.cache_leftpad if has_leftpad else None,
            )

            n_rep = self.n_heads // self.n_kv_heads
            k_full = _repeat_kv(k_full, n_rep)
            v_full = _repeat_kv(v_full, n_rep)
            q, k_full, v_full = q.transpose(1, 2), k_full.transpose(1, 2), v_full.transpose(1, 2)
            att = F.scaled_dot_product_attention(
                q,
                k_full,
                v_full,
                attn_mask=attn_mask,
                dropout_p=self.dropout_p,
                is_causal=False,
                scale=self.scale,
            )
            return att.transpose(1, 2).contiguous()

    _CACHED_BACKEND_CLS = CachedTorchAttentionBackend
    return CachedTorchAttentionBackend


def _lm_attention_modules(model: Any) -> list[Any]:
    return [
        block.attention
        for block in model.lm.blocks.values()
        if getattr(block, "attention", None) is not None
    ]


def enable_kv_cache(model: Any) -> bool:
    """Swap each LM attention's dense ``torch`` backend for the cached subclass.

    Returns ``False`` (leaving the model untouched) when any block uses a
    different backend, a sliding window, or has no attention module — callers
    should then fall back to no-cache decoding.
    """
    from olmo_core.nn.attention.backend import TorchAttentionBackend

    cached_cls = _cached_torch_backend_class()
    attentions = _lm_attention_modules(model)
    if len(attentions) != len(model.lm.blocks):
        return False
    for attention in attentions:
        backend = getattr(attention, "backend", None)
        if backend is None or type(backend) not in (TorchAttentionBackend, cached_cls):
            return False
        if backend.window_size != (-1, -1):
            return False
    for attention in attentions:
        attention.backend.__class__ = cached_cls
    return True


_EXPLICIT_POSITION_KV_MANAGER_CLS: type | None = None


def _explicit_position_kv_manager_class() -> type:
    """Build (once) a ``KVCacheManager`` subclass for explicit-position decoding.

    ``Attention.forward`` derives the RoPE ``start_pos`` from
    ``kv_cache_manager.current_position()``, but RoPE forbids combining
    ``start_pos`` with explicit ``position_ids``. The decode loop drives RoPE
    entirely through per-row ``position_ids`` (required for variable-length
    batches, where each row sits at a different absolute position), so this
    manager reports no position of its own.
    """
    global _EXPLICIT_POSITION_KV_MANAGER_CLS
    if _EXPLICIT_POSITION_KV_MANAGER_CLS is not None:
        return _EXPLICIT_POSITION_KV_MANAGER_CLS

    from olmo_core.nn.attention.kv_cache import KVCacheManager

    class ExplicitPositionKVCacheManager(KVCacheManager):
        def current_position(self) -> Any:
            return None

    _EXPLICIT_POSITION_KV_MANAGER_CLS = ExplicitPositionKVCacheManager
    return ExplicitPositionKVCacheManager


def prepare_kv_caches(
    model: Any,
    batch_size: int,
    max_seq_len: int,
    leftpad: torch.Tensor | None = None,
) -> None:
    """Initialize (or reset) a KV cache on every LM attention block.

    :param leftpad: Per-row left-pad lengths ``(batch_size,)`` for
        variable-length batches; rows are right-aligned in the cache and the
        cached backend masks the pad slots out of every real query's row.
    """
    manager_cls = _explicit_position_kv_manager_class()
    has_leftpad = leftpad is not None and bool((leftpad > 0).any())
    for attention in _lm_attention_modules(model):
        manager = attention.kv_cache_manager
        if isinstance(manager, manager_cls):
            manager.reset(batch_size, max_seq_len)
        else:
            manager = manager_cls(
                batch_size=batch_size,
                max_seq_len=max_seq_len,
                num_kv_heads=attention.n_kv_heads,
                head_dim=attention.head_dim,
                device=attention.w_k.weight.device,
            )
            attention.kv_cache_manager = manager
        if leftpad is not None:
            manager.cache_leftpad.copy_(leftpad)
        manager._has_leftpad = has_leftpad
        manager._position_mirror = 0


def free_kv_caches(model: Any) -> None:
    """Drop all LM KV caches so subsequent forwards run uncached."""
    for attention in _lm_attention_modules(model):
        attention.kv_cache_manager = None


def rope_buffers(model: Any, max_seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Full-length RoPE sin/cos tables covering positions ``0..max_seq_len - 1``.

    Explicit-``position_ids`` RoPE gathers rows from the sin/cos tables, but
    ``RotaryEmbedding.forward`` sizes its internal tables from the current
    forward's ``seq_len`` (1 during decode) when ``start_pos`` is absent —
    far short of the decode positions. The tables must therefore be passed in
    explicitly; this builds them once per batch. The Transformer broadcasts one
    buffer pair to every block, so all blocks must share one RoPE config
    (asserted here; per-position table rows do not depend on table length, so
    these buffers match what shorter internal tables would hold).
    """
    ropes = [
        rope
        for attention in _lm_attention_modules(model)
        if (rope := getattr(attention, "rope", None)) is not None
    ]
    if not ropes:
        raise ValueError("Model has no RoPE modules; cannot run explicit-position decoding")
    configs = {
        (
            type(rope).__name__,
            getattr(rope, "theta", None),
            getattr(rope, "rotary_dim", None),
            repr(getattr(rope, "scaling", None)),
        )
        for rope in ropes
    }
    if len(configs) != 1:
        raise ValueError(
            f"LM blocks use differing RoPE configs ({sorted(configs)}); batched "
            "explicit-position decoding requires one shared config"
        )
    buffers = ropes[0].get_buffers(max_seq_len, next(iter(model.lm.parameters())).device)
    if buffers.pos_sin is None or buffers.pos_cos is None:
        raise ValueError(
            f"{type(ropes[0]).__name__} does not expose sin/cos RoPE buffers; batched "
            "explicit-position decoding requires RotaryEmbedding"
        )
    return buffers.pos_sin, buffers.pos_cos
