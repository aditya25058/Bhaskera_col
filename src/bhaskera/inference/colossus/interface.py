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

# Expert activation fns by config/hidden_act names. SwiGLU variants share the
# silu-gated form; plain FFNs use the elementwise one.
ACT_SILU = ("silu", "swiglu", "sigmoid")
ACT_GELU = ("gelu", "gelu_new", "gelu_pytorch_tanh")
ACT_RELU = ("relu",)


def normalize_activation(name: Any) -> str:
    """Canonical activation id from a config string (default: silu)."""
    low = str(name or "silu").lower()
    if any(k in low for k in ACT_GELU):
        return "gelu"
    if any(k in low for k in ACT_RELU):
        return "relu"
    return "silu"

# Expert-container attribute names across architectures.
EXPERT_CONTAINERS = ("experts", "local_experts", "routed_experts")

# Projection-role hints over parameter leaf names, in priority order.
# SwiGLU: gate/up/down (DeepSeek/Qwen/Llama-MoE), w1/w3/w2 (Mixtral).
ROLE_HINTS = {
    "gate": ("gate_proj", "gate", "w1", "wi"),
    "up": ("up_proj", "up", "w3", "w0"),
    "down": ("down_proj", "down", "w2", "wo"),
}


def experts_of(moe_block: Any) -> Tuple[str, Sequence[nn.Module]]:
    """(container_attr, expert_modules) trying known container names."""
    for attr in EXPERT_CONTAINERS:
        if hasattr(moe_block, attr):
            return attr, list(getattr(moe_block, attr))
    raise AttributeError(
        f"MoE block {type(moe_block).__name__} has no expert container "
        f"(tried {EXPERT_CONTAINERS})")


# MoE-block attribute names on decoder layers across architectures.
MOE_BLOCK_ATTRS = ("mlp", "block_sparse_moe", "moe", "sparse_moe")


def find_moe_block(decoder_layer: Any) -> Optional[Tuple[str, Any]]:
    """(attr, block) for the first attr holding an experts container; None."""
    for attr in MOE_BLOCK_ATTRS:
        block = getattr(decoder_layer, attr, None)
        if block is None:
            continue
        try:
            experts_of(block)
            return attr, block
        except AttributeError:
            continue
    return None


def expert_weight_keys(expert_dotted: str, weight_keys) -> dict:
    """{gate|up|down: full key} for one expert's dotted path.

    Prefers ".weight"-suffixed entries on ambiguity; raises KeyError listing
    what was found when a role is missing (explicit > silent miss).
    """
    prefix = expert_dotted + "."
    cands: dict = {}
    for k in weight_keys:
        if not k.startswith(prefix):
            continue
        role = classify_role(k[len(prefix):])
        if role is None:
            continue
        cands.setdefault(role, []).append(k)
    out = {}
    for role, ks in cands.items():
        w = [k for k in ks if k.endswith(".weight")] or sorted(ks)
        out[role] = w[0]
    missing = {"gate", "up", "down"} - set(out)
    if missing:
        raise KeyError(f"expert {expert_dotted}: missing roles {sorted(missing)} "
                       f"(saw {sorted(cands)})")
    return out


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


def classify_role(relname: str) -> Optional[str]:
    """Map an expert-relative param name to gate/up/down (None if unknown)."""
    low = relname.lower()
    for role, hints in ROLE_HINTS.items():
        if any(h in low for h in hints):
            return role
    return None


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
    container_attr: str = "experts"
    activation: str = "silu"

    @classmethod
    def from_block(cls, moe_block: Any, top_k: Optional[int] = None) -> "MoELayerSpec":
        container_attr, experts = experts_of(moe_block)
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
                   dtype=g0.weight.dtype, container_attr=container_attr,
                   activation=normalize_activation(
                       getattr(getattr(moe_block, "config", None),
                               "hidden_act", "silu")))

    def route(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Unified routing -> (topk_idx, topk_weight), rank mirrors input.

        DeepSeek-style gates consume the tensor as given (often 3D); logits
        gates flatten, top-k, then restore the leading shape.
        """
        if self.gate_style == GATE_DEEPSEEK_3TUPLE:
            out = self.gate(hidden)
            idx, w = out[0], out[1]
            return idx.long(), w.to(hidden.dtype)
        lead = hidden.shape[:-1]
        flat = hidden.reshape(-1, hidden.shape[-1])
        logits = self.gate(flat)
        if logits.dim() > 2:
            logits = logits.view(-1, logits.shape[-1])
        scores = F.softmax(logits.float(), dim=-1)
        w, idx = torch.topk(scores, k=self.top_k, dim=-1)
        w = w / w.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        return (idx.long().view(*lead, self.top_k),
                w.to(hidden.dtype).view(*lead, self.top_k))

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
