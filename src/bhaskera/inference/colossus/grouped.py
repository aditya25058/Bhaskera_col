"""Grouped-weight MoE adapter (COLOSSUS generality layer).

Two MoE representations exist in the wild:

1. module-based: an experts container holding one submodule per expert
   (DeepSeek / Qwen / Mixtral / Param2). Handled by interface.py.
2. grouped-weight: the container holds fused 3D parameters with a leading
   expert dimension, e.g. gate_up [E, 2I, H] + down [E, H, I].
   Handled HERE.

This module exposes representation (2) through the same execution
contract the tiered machinery already understands:

    grouped [E, ...] --slice--> expert e view --split--> gate/up halves

No architecture names appear anywhere in this file: discovery is by
tensor rank (3D params sharing a leading dim) and container position
(experts-leaf without an expert index), never by model family.

Zero-copy: expert slices are views on the mmap'd grouped tensor; only
the DMA copy into a slot materializes bytes.
"""
from __future__ import annotations

import re
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

# Container leaf names shared with interface.EXPERT_CONTAINERS
# (asserted in sync by the test suite, not imported to avoid a cycle).
GROUPED_CONTAINER_NAMES = ("experts", "local_experts", "routed_experts")

# Weight-map key for a grouped expert tensor: the experts container is
# followed directly by a parameter name (no ".<expert_idx>." segment).
# Structural (position-based); no parameter-name knowledge.
GROUPED_KEY_RES = tuple(
    re.compile(r"\." + name + r"\.[A-Za-z_][A-Za-z_0-9]*$") for name in GROUPED_CONTAINER_NAMES
)


def is_grouped_expert_key(key: str) -> bool:
    """True iff key names a fused grouped-expert tensor (no expert index)."""
    return any(rx.search(key) is not None for rx in GROUPED_KEY_RES)


def classify_grouped_pair(shapes: dict[str, tuple]) -> dict[str, str]:
    """{fused, down} key assignment from 3D shapes alone.

    Fused gate+up has shape [E, 2I, H]; down has [E, H, I]. Structural
    rule: the fused tensor's dim1 is exactly twice the down tensor's
    dim2, and fused dim2 equals down dim1. Needs exactly two tensors.
    """
    if len(shapes) != 2:
        raise KeyError(f"grouped experts need exactly 2 tensors, saw {sorted(shapes)}")
    (ka, sa), (kb, sb) = list(shapes.items())
    if len(sa) != 3 or len(sb) != 3 or sa[0] != sb[0]:
        raise KeyError(f"grouped experts need 3D tensors sharing dim0: {shapes}")
    if sa[1] == 2 * sb[2] and sa[2] == sb[1]:
        return {"fused": ka, "down": kb}
    if sb[1] == 2 * sa[2] and sb[2] == sa[1]:
        return {"fused": kb, "down": ka}
    raise KeyError(f"no fused/down pair among {shapes} (need [E,2I,H]+[E,H,I])")


class GroupedSlot(nn.Module):
    """HBM slot holding one sliced grouped expert.

    gate_up [2I, H] (fused halves, split at use) + down [H, I].
    Activation from spec (silu/gelu/gelu_tanh/relu); gelu_tanh matches
    tanh-approximated GELU variants (see placement.apply_gate).
    """

    def __init__(self, hidden: int, inter: int, device: torch.device,
                 dtype: torch.dtype = torch.bfloat16,
                 activation: str = "gelu_tanh"):
        super().__init__()
        self.gate_up_proj = nn.Linear(hidden, 2 * inter, bias=False,
                                      device=device, dtype=dtype)
        self.down_proj = nn.Linear(inter, hidden, bias=False,
                                   device=device, dtype=dtype)
        self.activation = activation
        self.requires_grad_(False)


class GroupedTieredMoEWrapper(nn.Module):
    """Tiered execution for one grouped-expert container.

    Drop-in replacement for the native experts module: same call
    signature (hidden, topk_idx, topk_w) -> tensor, same op order
    (ascending experts, index_add accumulation) so outputs match the
    native implementation bitwise when weights are resident.

    Resident: nothing (router/norms stay native upstream). Slots hold C
    sliced experts, LRU-managed, demand-loaded from `handles` via the
    two grouped keys. Telemetry mirrors TieredMoEWrapper
    (hits/misses/dma_bytes/routing_log) so serve.py ledgers just work.
    """

    def __init__(
        self,
        layer_idx: int,
        fused_key: str,
        down_key: str,
        n_experts: int,
        hidden: int,
        inter: int,
        device: torch.device,
        capacity: int,
        handles: Any,
        dtype: torch.dtype = torch.bfloat16,
        activation: str = "gelu_tanh",
        dma_stream: Any = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.fused_key = fused_key
        self.down_key = down_key
        self.n_experts = int(n_experts)
        self.hidden = int(hidden)
        self.inter = int(inter)
        self.device = device
        self.handles = handles
        self.dma_stream = dma_stream
        self.activation = activation
        self.streaming = (capacity == 0)
        if self.streaming:
            capacity = 1
        self.capacity = capacity
        self.slots: list[nn.Module] = nn.ModuleList([
            GroupedSlot(hidden, inter, device, dtype, activation)
            for _ in range(capacity)
        ])
        self.slot_to_expert: dict[int, int] = {}
        self.expert_to_slot: dict[int, int] = {}
        self.slot_lru: list[int] = list(range(capacity))
        self.hits = 0
        self.misses = 0
        self.dma_bytes = 0
        self.zssr_predictions = 0
        self.zssr_correct = 0
        self.zssr_suppressed = 0
        self.routing_log = None
        self.cpu_ms = 0.0
        self.cpu_n = 0

    # -- loading ------------------------------------------------------
    def _expert_slice(self, expert_id: int):
        fused = self.handles.get_tensor(self.fused_key)[expert_id]
        down = self.handles.get_tensor(self.down_key)[expert_id]
        return fused, down

    def _load_expert_to_slot(self, expert_id: int, slot_idx: int) -> int:
        slot = self.slots[slot_idx]
        t_fused, t_down = self._expert_slice(expert_id)
        nbytes = int(t_fused.nbytes + t_down.nbytes)
        use_stream = (self.dma_stream is not None and self.device.type == "cuda")
        with torch.no_grad():
            if use_stream:
                with torch.cuda.stream(self.dma_stream):
                    slot.gate_up_proj.weight.copy_(t_fused, non_blocking=True)
                    slot.down_proj.weight.copy_(t_down, non_blocking=True)
            else:
                slot.gate_up_proj.weight.copy_(t_fused)
                slot.down_proj.weight.copy_(t_down)
        if slot_idx in self.slot_to_expert:
            self.expert_to_slot.pop(self.slot_to_expert.pop(slot_idx), None)
        self.slot_to_expert[slot_idx] = expert_id
        self.expert_to_slot[expert_id] = slot_idx
        self.dma_bytes += nbytes
        return nbytes

    def _sync_dma(self) -> None:
        if self.dma_stream is not None and self.device.type == "cuda":
            torch.cuda.current_stream(self.device).wait_stream(self.dma_stream)

    def _ensure(self, expert_id: int) -> int:
        if expert_id in self.expert_to_slot:
            self.hits += 1
            slot_idx = self.expert_to_slot[expert_id]
            self.slot_lru.remove(slot_idx)
            self.slot_lru.append(slot_idx)
            return slot_idx
        self.misses += 1
        slot_idx = self.slot_lru.pop(0)
        self._load_expert_to_slot(expert_id, slot_idx)
        self.slot_lru.append(slot_idx)
        self._sync_dma()
        return slot_idx

    # -- forward -------------------------------------------------------
    def forward(self, hidden_states: torch.Tensor,
                topk_indices: torch.Tensor,
                topk_weights: torch.Tensor) -> torch.Tensor:
        from .placement import apply_gate
        lead = hidden_states.shape[:-1]
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        K = topk_indices.shape[-1]
        TI = topk_indices.reshape(-1, K).long()
        # Weights keep their incoming dtype (native multiplies in promoted
        # precision, then casts at accumulation — required for bitwise parity).
        TW = topk_weights.reshape(-1, K)
        needed = sorted({int(e) for e in TI.unique().tolist()} & set(range(self.n_experts)))
        if self.routing_log is not None:
            self.routing_log.append((self.layer_idx, needed))
        out = torch.zeros_like(flat)
        # Native op order, literally: ascending experts, positions in
        # torch.where order, index_add accumulation (bitwise parity).
        mask = F.one_hot(TI, num_classes=self.n_experts + 1).permute(2, 1, 0)
        for e in needed:
            slot_idx = self._ensure(e)
            slot = self.slots[slot_idx]
            kpos, rows = torch.where(mask[e])
            xe = flat[rows]
            with torch.no_grad():
                g, u = F.linear(xe, slot.gate_up_proj.weight).chunk(2, dim=-1)
                ye = F.linear(apply_gate(g, u, self.activation),
                              slot.down_proj.weight)
                ye = ye * TW[rows, kpos].unsqueeze(-1)
            out.index_add_(0, rows, ye.to(out.dtype))
        if self.streaming:
            self.expert_to_slot.clear()
            self.slot_to_expert.clear()
            self.slot_lru = list(range(len(self.slots)))
        return out.reshape(*lead, flat.shape[-1])


def grouped_container_attr(layer: Any) -> str | None:
    """Attribute name of a grouped experts container on a layer (None)."""
    for attr in GROUPED_CONTAINER_NAMES:
        mod = getattr(layer, attr, None)
        if mod is None or isinstance(mod, nn.ModuleList):
            continue
        try:
            params = [p for _, p in mod.named_parameters(recurse=False)]
        except Exception:
            continue
        shapes = [tuple(p.shape) for p in params if p.dim() == 3]
        if len(shapes) >= 2 and len({s[0] for s in shapes}) == 1:
            return attr
    return None


def wrap_grouped_layers(model: nn.Module, profile: Any, handles: Any,
                        device: torch.device, capacity: int,
                        dma_stream: Any = None,
                        **opts) -> list[GroupedTieredMoEWrapper]:
    """Replace every grouped experts container with a tiered wrapper.

    Discovery: profile.decoder_layer_cls instances -> grouped_container_attr.
    Weight keys: index keys under the container dotted path (exactly the
    fused + down pair, classified by shape). wrapper_cls: alternate
    executor with a compatible constructor (e.g. column-granular slots);
    extra opts (e.g. hot_frac) pass through. Returns wrappers in order.
    """
    from .interface import normalize_activation

    wrapper_cls = opts.pop("wrapper_cls", GroupedTieredMoEWrapper)

    decoder_cls = getattr(profile, "decoder_layer_cls", None)
    if decoder_cls is not None:
        layers = [m for m in model.modules() if isinstance(m, decoder_cls)]
    else:
        layers = list(getattr(getattr(model, "model", model), "layers", []))
    dotted = {id(m): name for name, m in model.named_modules()}
    weight_keys = list(handles.weight_map.keys())
    wrappers = []
    for layer_idx, layer in enumerate(layers):
        attr = grouped_container_attr(layer)
        if attr is None:
            continue
        prefix = dotted.get(id(layer), "")
        if not prefix:
            raise KeyError(f"layer {layer_idx}: no dotted path")
        cands = [k for k in weight_keys
                 if k.startswith(prefix + "." + attr + ".")]
        if len(cands) != 2:
            raise KeyError(f"layer {layer_idx}: expected fused+down pair, "
                            f"saw {cands}")
        shapes = {k: tuple(handles.header(k)["shape"]) for k in cands}
        pair = classify_grouped_pair(shapes)
        fused_shape = shapes[pair["fused"]]
        n_exp, inter2, hidden = fused_shape
        inter = inter2 // 2
        act_mod = next((m for _, m in getattr(layer, attr)
                        .named_children()), None)
        act_name = type(act_mod).__name__ if act_mod is not None else "silu"
        if "tanh" in act_name.lower():
            activation = "gelu_tanh"
        else:
            activation = normalize_activation(
                getattr(getattr(layer, "config", None), "hidden_act", act_name))
        wrapper = wrapper_cls(
            layer_idx=layer_idx, fused_key=pair["fused"], down_key=pair["down"],
            n_experts=n_exp, hidden=hidden, inter=inter, device=device,
            capacity=capacity, handles=handles, activation=activation,
            dma_stream=dma_stream, **opts)
        setattr(layer, attr, wrapper)
        wrappers.append(wrapper)
    return wrappers
