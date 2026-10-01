"""Model-agnostic MoE layer interface (chunk 1 of Bhaskera+COLOSSUS integration).

Problem: serving code assumed DeepSeek attribute layout
(`moe_block.experts`, 3-tuple gate, `moe_infer`, `shared_experts`).
Bhaskera forbids model-specific names, so this module translates any MoE
block into a uniform spec the execution machinery consumes:

    MoE block (any arch) --from_block--> MoELayerSpec --executor--> y

Gate conventions normalized here (NOT in the hot path):
  - "deepseek_3tuple": gate(x) -> (topk_idx, topk_weight[, aux])
  - "logits":          gate(x) -> logits -> softmax/topk/renorm (Mixtral/Qwen)

Expert invocation: bulk `moe_infer(x, idx, w)` when present, else the
caller falls back to the per-expert sorted-dispatch loop.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

GATE_DEEPSEEK_3TUPLE = "deepseek_3tuple"
GATE_LOGITS = "logits"


def find_proj(expert: nn.Module, *name_hints: str) -> nn.Linear:
    """Locate a projection Linear inside an expert by name hints.

    Primary: child whose name contains f"{hint}_proj" (gate_proj, ...).
    Fallback: error listing candidate Linear children (explicit > guessing).
    """
    named = dict(expert.named_modules())
    for hint in name_hints:
        for name, mod in named.items():
            if isinstance(mod, nn.Linear) and f"{hint}_proj" in name:
                return mod
    cands = [n for n, m in named.items() if isinstance(m, nn.Linear)]
    raise KeyError(f"no projection matching {name_hints}; Linear candidates: {cands}")


def detect_gate_style(moe_block: Any) -> str:
    """DeepSeek-style bulk gate vs logits gate (no model names)."""
    if hasattr(moe_block, "moe_infer"):
        return GATE_DEEPSEEK_3TUPLE
    return GATE_LOGITS


@dataclass
class MoELayerSpec:
    """Uniform, model-agnostic view of one MoE layer."""
    experts: Sequence[nn.Module]
    gate: nn.Module
    gate_style: str
    shared: Optional[nn.Module]
    top_k: int
    n_routed: int
    hidden: int
    inter: int
    dtype: torch.dtype

    @classmethod
    def from_block(cls, moe_block: Any, top_k: Optional[int] = None) -> "MoELayerSpec":
        experts = list(moe_block.experts)
        gate = getattr(moe_block, "gate", None)
        if gate is None:
            raise AttributeError("MoE block has no gate/router module")
        style = detect_gate_style(moe_block)
        if top_k is None:
            top_k = int(getattr(moe_block, "num_experts_per_tok",
                                getattr(moe_block, "top_k", 2)))
        shared = getattr(moe_block, "shared_experts", None)
        cfg = getattr(moe_block, "config", None)
        if shared is None and cfg is not None and \
                getattr(cfg, "num_shared_experts", None):
            # Config-declared shared experts under another attribute name.
            for name in ("shared_expert", "shared_moe", "shared_mlp"):
                if hasattr(moe_block, name):
                    shared = getattr(moe_block, name)
                    break
        g0 = find_proj(experts[0], "gate")
        return cls(experts=experts, gate=gate, gate_style=style, shared=shared,
                   top_k=top_k, n_routed=len(experts),
                   hidden=g0.weight.shape[1], inter=g0.weight.shape[0],
                   dtype=g0.weight.dtype)

    def route(self, hidden_2d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Unified routing -> (topk_idx [N,K] long, topk_weight [N,K])."""
        if self.gate_style == GATE_DEEPSEEK_3TUPLE:
            out = self.gate(hidden_2d)
            idx, w = out[0], out[1]
            return idx.long(), w.to(hidden_2d.dtype)
        logits = self.gate(hidden_2d)
        if logits.dim() > 2:
            logits = logits.view(-1, logits.shape[-1])
        scores = F.softmax(logits.float(), dim=-1)
        w, idx = torch.topk(scores, k=self.top_k, dim=-1)
        w = w / w.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        return idx.long(), w.to(hidden_2d.dtype)

    def router_weight(self) -> Optional[torch.Tensor]:
        """Detached fp32 router matrix for ZSSR probing (None if absent)."""
        g = self.gate
        if hasattr(g, "weight"):
            return g.weight.detach().float()
        inner = getattr(g, "gate", None)
        if inner is not None and hasattr(inner, "weight"):
            return inner.weight.detach().float()
        lin = getattr(g, "linear", None)
        if isinstance(lin, nn.Linear):
            return lin.weight.detach().float()
        return None
