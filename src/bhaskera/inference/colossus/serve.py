"""Huge-model serving loop for COLOSSUS tiers (chunk 2c).

`serve_huge_moe`: prefill (optionally chunked) -> lockstep greedy decode
over a wrapped (TieredMoEWrapper) model with per-step ledger.
Model-agnostic: no names, no templates, no architecture branches.
"""
from __future__ import annotations

import time
from typing import Any

import torch

from .loading import ShardHandles, materialize, split_routed
from .placement import TieredMoEWrapper, wrap_moe_layers

_compat_installed = False


def install_cache_compat() -> None:
    """Polyfills for modeling code written against older transformers cache APIs.

    Some custom modeling files (e.g. DeepSeek-V2) call
    `DynamicCache.get_usable_length` and expect a non-None causal 4D mask.
    Both shims are idempotent and model-agnostic in form.
    """
    global _compat_installed
    if _compat_installed:
        return
    import torch as _torch
    from transformers.cache_utils import DynamicCache as _DC
    from transformers.modeling_attn_mask_utils import AttentionMaskConverter as _AMC

    def _get_usable_length(self, *args, **kwargs):
        layer_idx = 0
        if len(args) > 1 and isinstance(args[1], int):
            layer_idx = args[1]
        elif "layer_idx" in kwargs:
            layer_idx = kwargs["layer_idx"]
        return self.get_seq_length(layer_idx)

    _DC.get_usable_length = _get_usable_length
    _orig = _AMC.to_causal_4d

    def _patched_to_causal_4d(self, batch_size, query_length, key_value_length,
                              dtype, device="cpu"):
        mask = _orig(self, batch_size, query_length, key_value_length, dtype, device)
        if mask is None:
            mask = _torch.zeros((batch_size, 1, query_length, key_value_length),
                                dtype=dtype, device=device)
        return mask

    _AMC.to_causal_4d = _patched_to_causal_4d
    _compat_installed = True


def prepare_model(model: torch.nn.Module, profile: Any, handles: ShardHandles,
                  device: torch.device, capacity: int,
                  dtype: torch.dtype = torch.bfloat16,
                  placement: str = "slots", dma_stream: Any = None,
                  zssr: bool = False, prefetch_topk: int = 8,
                  prefetch_conf: float = 0.0) -> list[TieredMoEWrapper]:
    """Materialize resident weights + wrap MoE layers. Returns wrappers."""
    resident, _ = split_routed(handles.weight_map)
    materialize(model, resident, handles, device, dtype)
    model.eval()
    # RoPE-style computed buffers are meta after empty init; best-effort init
    # via duck-typed hooks (any attention exposing _init_rope).
    for m in model.modules():
        init_rope = getattr(m, "_init_rope", None)
        if callable(init_rope):
            try:
                init_rope()
            except Exception:
                pass
        remb = getattr(m, "rotary_emb", None)
        if remb is not None:
            try:
                remb.to(device)
            except Exception:
                pass
    wrappers = wrap_moe_layers(
        model, profile, handles, device, capacity, dma_stream,
        zssr_prefetch=zssr, prefetch_topk=prefetch_topk,
        prefetch_conf=prefetch_conf,
        cpu_exec=(placement == "cpu"))
    return wrappers


def serve_huge_moe(model, tokenizer, profile, handles: ShardHandles,
                   device: torch.device, prompts: list[str],
                   max_new_tokens: int = 16, capacity: int = 12,
                   placement: str = "slots", prefill_chunk: int = 0,
                   use_cache: bool = True, zssr: bool = False,
                   prefetch_topk: int = 8, prefetch_conf: float = 0.0,
                   log_routing: str | None = None,
                   teacher_tokens: list[int] | None = None,
                   audit_logits: str | None = None,
                   ) -> dict[str, Any]:
    """Greedy lockstep serve with ledger. Returns results dict."""
    from transformers.cache_utils import DynamicCache

    install_cache_compat()
    wrappers = prepare_model(model, profile, handles, device, capacity,
                             placement=placement,
                             dma_stream=(torch.cuda.Stream(device=device)
                                         if device.type == "cuda" else None),
                             zssr=zssr, prefetch_topk=prefetch_topk,
                             prefetch_conf=prefetch_conf)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    enc = tokenizer(prompts, padding=True, return_tensors="pt")
    input_ids = enc["input_ids"]
    attn_mask = enc.get("attention_mask")
    B, prompt_len = input_ids.shape
    input_ids = input_ids.to(device)
    if attn_mask is not None:
        attn_mask = attn_mask.to(device)
    generated_ids = input_ids.clone()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    t_prefill = time.perf_counter()
    past = DynamicCache() if use_cache else None
    with torch.no_grad():
        if prefill_chunk > 0 and generated_ids.shape[1] > prefill_chunk:
            logits, seqlen = None, generated_ids.shape[1]
            for s in range(0, seqlen, prefill_chunk):
                chunk = generated_ids[:, s:s + prefill_chunk]
                cmask = torch.ones((B, s + chunk.shape[1]), dtype=torch.long,
                                   device=device)
                out = model(input_ids=chunk, attention_mask=cmask,
                            past_key_values=past, use_cache=use_cache)
                logits = out.logits
                past = getattr(out, "past_key_values", None)
        else:
            out = model(input_ids=generated_ids, attention_mask=attn_mask,
                        past_key_values=past, use_cache=use_cache)
            logits = out.logits
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        past = getattr(out, "past_key_values", None)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t_prefill = time.perf_counter() - t_prefill
    if next_token.device != device:
        next_token = next_token.to(device)
    # Teacher-forced flip audit (B=1 only): feed reference tokens, record own
    # argmax + fp32 margin per position (isolates per-position noise).
    audit: list[dict] = []
    if teacher_tokens is not None:
        assert B == 1, "teacher protocol requires batch size 1"
        assert len(teacher_tokens) >= max_new_tokens, "teacher too short"
        tv, ti = logits[0, -1, :].float().topk(2)
        own = int(ti[0])
        audit.append({"pos": 1, "own": own, "ref": teacher_tokens[0],
                      "match": own == teacher_tokens[0],
                      "margin": float(tv[0] - tv[1])})
        next_token = torch.tensor([[teacher_tokens[0]]], device=device)

    finished = [False] * B
    eos_id = tokenizer.eos_token_id
    new_counts = [1] * B
    latencies: list[float] = []
    if log_routing:
        for w in wrappers:
            w.routing_log = []
    for step in range(max_new_tokens - 1):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            if use_cache and past is not None:
                out = model(input_ids=next_token, past_key_values=past, use_cache=True)
            else:
                out = model(input_ids=torch.cat([generated_ids, next_token], dim=1),
                            use_cache=False)
            logits = out.logits
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            if use_cache:
                past = getattr(out, "past_key_values", None)
            if teacher_tokens is not None:
                pos = step + 2
                if pos - 1 < len(teacher_tokens):
                    tv, ti = logits[0, -1, :].float().topk(2)
                    own = int(ti[0])
                    ref = teacher_tokens[pos - 1]
                    audit.append({"pos": pos, "own": own, "ref": ref,
                                  "match": own == ref,
                                  "margin": float(tv[0] - tv[1])})
                    next_token = torch.tensor([[ref]], device=device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        latencies.append(time.perf_counter() - t0)
        if next_token.device != device:
            next_token = next_token.to(device)
        for bi in range(B):
            if finished[bi]:
                continue
            new_counts[bi] += 1
            if int(next_token[bi, 0].item()) == eos_id:
                finished[bi] = True
        generated_ids = torch.cat([generated_ids, next_token], dim=1)
        if all(finished):
            break

    texts = tokenizer.batch_decode(generated_ids[:, prompt_len:],
                                   skip_special_tokens=True)
    total_hits = sum(w.hits for w in wrappers)
    total_misses = sum(w.misses for w in wrappers)
    total_dma_mb = sum(w.dma_bytes for w in wrappers) / (1024 ** 2)
    total_decode = sum(latencies)
    agg_tokens = sum(new_counts)
    peak = (torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            if device.type == "cuda" else 0.0)
    routing_steps = 0
    if log_routing:
        import json as _json
        entries = [{"layer": lyr, "experts": exps}
                   for w in wrappers for (lyr, exps) in (w.routing_log or [])]
        with open(log_routing, "w") as f:
            _json.dump(entries, f)
        routing_steps = (len(wrappers[0].routing_log or [])
                         if wrappers else 0)
    flip_audit = None
    if audit_logits:
        import json as _json2
        with open(audit_logits, "w") as f:
            _json2.dump({"audit": audit}, f)
        flip_audit = audit
    return {
        "texts": texts,
        "batch_size": B,
        "prompt_tokens": prompt_len,
        "generated_tokens": max_new_tokens,
        "prefill_s": t_prefill,
        "decode_avg_ms": (total_decode / max(1, len(latencies))) * 1000.0,
        "batch_decode_tps": agg_tokens / total_decode if total_decode > 0 else 0.0,
        "total_hits": total_hits,
        "total_misses": total_misses,
        "total_dma_mb": total_dma_mb,
        "peak_vram_gb": peak,
        "placement": placement,
        "capacity": capacity,
        "routing_log_steps": routing_steps,
        "flip_audit": flip_audit,
    }
