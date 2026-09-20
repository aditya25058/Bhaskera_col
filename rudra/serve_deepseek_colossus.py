#!/usr/bin/env python3
"""
serve_deepseek_colossus.py
==========================
Full End-to-End Serving Engine for DeepSeek-Coder-V2 (236B MoE, 160 Experts)
Consolidating an 8x H100 Cluster onto 2x NVIDIA H100 NVL (Hopper SM 9.0)
via COLOSSUS Dynamic Expert Streaming.

Pipeline Partitioning:
- GPU 0 (NVIDIA H100 NVL 94GB):
  - Embeddings (model.embed_tokens)
  - Layer 0: Dense Decoder Layer (MLA + Dense MLP + RMSNorms)
  - Layers 1..29: 29 MoE Layers (MLA + Shared Experts + C=12 Dynamic Slots)
- GPU 1 (NVIDIA H100 NVL 94GB):
  - NVLink P2P Activation Bridge
  - Layers 30..59: 30 MoE Layers (MLA + Shared Experts + C=12 Dynamic Slots)
  - Final RMSNorm (model.norm)
  - LM Head (lm_head)
- Host DDR5 (503 GB RAM / Page Cache):
  - 445 GB cold expert pool accessed via zero-copy mmap safetensors
  - Dynamic PCIe Gen5 DMA streaming for cache misses
"""

import os
import sys
import time
import json
import argparse
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from accelerate.utils import set_module_tensor_to_device
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter

# ─────────────────────────────────────────────────────────────────────────────
# Compatibility Polyfills for DeepSeek-V2 with Modern Transformers (v5.x)
# ─────────────────────────────────────────────────────────────────────────────
def get_usable_length(self, *args, **kwargs):
    layer_idx = 0
    if len(args) > 1 and isinstance(args[1], int):
        layer_idx = args[1]
    elif "layer_idx" in kwargs:
        layer_idx = kwargs["layer_idx"]
    return self.get_seq_length(layer_idx)

DynamicCache.get_usable_length = get_usable_length

_orig_to_causal_4d = AttentionMaskConverter.to_causal_4d

def patched_to_causal_4d(self, batch_size, query_length, key_value_length, dtype, device="cpu"):
    mask = _orig_to_causal_4d(self, batch_size, query_length, key_value_length, dtype, device)
    if mask is None:
        mask = torch.zeros((batch_size, 1, query_length, key_value_length), dtype=dtype, device=device)
    return mask

AttentionMaskConverter.to_causal_4d = patched_to_causal_4d

MODEL_PATH = "/home/palakm/MoEServingSim/aditya/models/DeepSeek-Coder-V2-Instruct"


# ─────────────────────────────────────────────────────────────────────────────
# 1. P2P NVLink Bridge between Layer 29 (GPU 0) and Layer 30 (GPU 1)
# ─────────────────────────────────────────────────────────────────────────────
class Layer30P2PBridge(nn.Module):
    """Bridges GPU 0 and GPU 1 over NVLink for Layer 30."""
    def __init__(self, inner_layer: nn.Module, target_dev: torch.device):
        super().__init__()
        self.inner_layer = inner_layer
        self.target_dev = target_dev

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        if hidden_states.device != self.target_dev:
            hidden_states = hidden_states.to(self.target_dev, non_blocking=True)
        # Also transfer any tensor in args or kwargs if on dev0
        new_args = [
            a.to(self.target_dev, non_blocking=True) if isinstance(a, torch.Tensor) and a.device != self.target_dev else a
            for a in args
        ]
        new_kwargs = {
            k: (v.to(self.target_dev, non_blocking=True) if isinstance(v, torch.Tensor) and v.device != self.target_dev else v)
            for k, v in kwargs.items()
        }
        return self.inner_layer(hidden_states, *new_args, **new_kwargs)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.inner_layer, name)



# Set expandable segments to avoid fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ─────────────────────────────────────────────────────────────────────────────
# 2. COLOSSUS Fast Dynamic Expert Slot (Instant HBM Allocation)
# ─────────────────────────────────────────────────────────────────────────────
class FastExpertSlot(nn.Module):
    """Direct HBM-allocated slot module for DeepSeek-V2 MoE expert."""
    def __init__(self, cfg, device: torch.device):
        super().__init__()
        H = cfg.hidden_size
        I = cfg.moe_intermediate_size
        self.gate_proj = nn.Linear(H, I, bias=False, device=device, dtype=torch.bfloat16)
        self.up_proj = nn.Linear(H, I, bias=False, device=device, dtype=torch.bfloat16)
        self.down_proj = nn.Linear(I, H, bias=False, device=device, dtype=torch.bfloat16)
        self.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class GlobalColumnPool:
    """GPU-only column-block cache keyed (layer, expert, block). Plain LFU + recency tiebreak.

    Physical unit is a column block (default 128 cols of intermediate dim) so every
    transfer (PCIe DMA) and every operand (GEMM slice) stays contiguous.
    Logical granularity stays column-level; residency/execution are fine-grained:
      hit  -> HBM entry -> slot slice (no PCIe, GPU computes)
      miss -> DMA master slice -> pool entry -> slot slice (PCIe once), GPU computes
    Assembly fills a full slot, so compute stays expert-granular (one GEMM, exact).
    No CPU execution anywhere. Capacity derives from an explicit VRAM budget:
      cap_blocks = budget_bytes / bytes_per_block.
    """

    def __init__(self, device: torch.device, dtype: torch.dtype, hidden: int, inter: int,
                 block: int = 128, budget_gb: float = 8.0):
        self.device = device
        self.dtype = dtype
        self.H = hidden
        self.I = inter
        self.B = max(1, int(block))
        self.nblocks = (inter + self.B - 1) // self.B
        elem = torch.tensor([], dtype=dtype).element_size()
        self.per_block_bytes = 3 * self.B * hidden * elem
        self.budget_bytes = int(budget_gb * (1024 ** 3))
        self.cap = max(1, self.budget_bytes // self.per_block_bytes)
        self.entries: Dict[tuple, List[torch.Tensor]] = {}
        self.freq: Dict[tuple, int] = {}
        self.stamp: Dict[tuple, int] = {}
        self.tick = 0
        self.hits = 0
        self.misses = 0
        self.dma_bytes = 0

    def reset_stats(self):
        self.hits = 0
        self.misses = 0
        self.dma_bytes = 0

    def _evict_one(self):
        victim = min(self.entries.keys(), key=lambda k: (self.freq.get(k, 0), self.stamp.get(k, 0)))
        for t in self.entries.pop(victim):
            del t
        self.freq.pop(victim, None)
        self.stamp.pop(victim, None)

    def _block_range(self, bi: int):
        s = bi * self.B
        return s, min(self.I, s + self.B)

    def assemble(self, layer_idx: int, expert_id: int, masters, slot_mod, dma_stream) -> int:
        """Assemble full expert weights into slot_mod from pool (hits) + DMA (misses).

        masters: (Wg_full, Wu_full, Wd_full) CPU tensors. Returns DMA bytes moved.
        Slot ends up holding the complete expert -> downstream forward is exact.
        """
        Wg_full, Wu_full, Wd_full = masters
        moved = 0
        with torch.no_grad():
            with torch.cuda.stream(dma_stream):
                for bi in range(self.nblocks):
                    s, e = self._block_range(bi)
                    key = (layer_idx, expert_id, bi)
                    self.tick += 1
                    ent = self.entries.get(key)
                    if ent is None:
                        while len(self.entries) >= self.cap:
                            self._evict_one()
                        g = torch.empty((e - s, self.H), dtype=self.dtype, device=self.device)
                        u = torch.empty((e - s, self.H), dtype=self.dtype, device=self.device)
                        d = torch.empty((self.H, e - s), dtype=self.dtype, device=self.device)
                        g.copy_(Wg_full[s:e], non_blocking=True)
                        u.copy_(Wu_full[s:e], non_blocking=True)
                        d.copy_(Wd_full[:, s:e], non_blocking=True)
                        self.entries[key] = [g, u, d]
                        self.freq[key] = 1
                        self.stamp[key] = self.tick
                        self.misses += 1
                        moved += int((g.nbytes + u.nbytes + d.nbytes))
                        ent = self.entries[key]
                    else:
                        self.freq[key] = self.freq.get(key, 0) + 1
                        self.stamp[key] = self.tick
                        self.hits += 1
                    g, u, d = ent
                    slot_mod.gate_proj.weight[s:e].copy_(g, non_blocking=True)
                    slot_mod.up_proj.weight[s:e].copy_(u, non_blocking=True)
                    slot_mod.down_proj.weight[:, s:e].copy_(d, non_blocking=True)
        self.dma_bytes += moved
        return moved


class DeepSeekColossusMoEWrapper(nn.Module):
    """
    Dynamic Slot Residency MoE Wrapper with Heterogeneous Compute-to-Data Engine
    + ADETR per-slot column streaming (COLOSSUS column-level reunification):
    - Permanently resident: Gate, Shared Experts on GPU HBM.
    - Dynamic pool: C dynamic slots allocated on GPU HBM.
    - Whole-expert (adetr_ratio=1.0): stream full 45MB expert into slot (scaffold).
    - Column-level (adetr_ratio<1.0): split intermediate dim into hot columns
      (CPU, zero weight movement, 10KB activation) + cold columns (DMA into slot
      slices, GPU computes y_cold). y = y_hot + y_cold, bitwise exact SA-FFN.
      No permanent 216GB hot table: residency lives per-slot, streaming per-miss.
    - Cold experts with small token load (M <= cpu_token_threshold):
      Directly evaluated on CPU via AVX-512 in native BF16 without PCIe weight DMA.
    """
    def __init__(
        self,
        layer_idx: int,
        moe_module: nn.Module,
        cfg,
        device: torch.device,
        capacity: int,
        handles: Dict[str, any],
        weight_map: Dict[str, str],
        dma_stream: torch.cuda.Stream,
        enable_hetero: bool = True,
        cpu_token_threshold: int = 1,
        adetr_ratio: float = 1.0,
        col_pool=None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.cfg = cfg
        self.device = device
        self.capacity = capacity
        self.handles = handles
        self.weight_map = weight_map
        self.dma_stream = dma_stream
        self.enable_hetero = enable_hetero
        self.cpu_token_threshold = cpu_token_threshold
        self.adetr_ratio = float(adetr_ratio)
        self.col_pool = col_pool

        # Resident modules on GPU
        self.gate = moe_module.gate
        self.shared_experts = moe_module.shared_experts

        # Dynamic slots on GPU (allocated directly in HBM)
        self.slots: List[nn.Module] = nn.ModuleList([
            FastExpertSlot(cfg, device=device)
            for _ in range(capacity)
        ])

        self.slot_to_expert: Dict[int, int] = {}
        self.expert_to_slot: Dict[int, int] = {}
        self.slot_lru: List[int] = list(range(capacity))
        self.slot_col_partial: Dict[int, bool] = {}

        # Heterogeneous activation streaming buffers
        self.max_cpu_batch = 64
        self.cpu_act_in = torch.empty((self.max_cpu_batch, cfg.hidden_size), dtype=torch.bfloat16, pin_memory=True)
        self.cpu_act_out = torch.empty((self.max_cpu_batch, cfg.hidden_size), dtype=torch.bfloat16, pin_memory=True)
        self.act_stream = torch.cuda.Stream(device=device)
        self.cpu_expert_weights: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

        # Metrics
        self.hits = 0
        self.misses = 0
        self.cpu_dispatches = 0
        self.adetr_col_dispatches = 0
        self.dma_bytes = 0
        self.expert_freq: Dict[int, int] = {}
        self.hetero_max_tokens = 8
        self.hetero_freq_retain = 3

    def _get_cpu_expert_weights(self, expert_id: int):
        if expert_id not in self.cpu_expert_weights:
            pfx = f"model.layers.{self.layer_idx}.mlp.experts.{expert_id}"
            k_gate = f"{pfx}.gate_proj.weight"
            k_up = f"{pfx}.up_proj.weight"
            k_down = f"{pfx}.down_proj.weight"

            shard_gate = self.weight_map[k_gate]
            shard_up = self.weight_map[k_up]
            shard_down = self.weight_map[k_down]

            t_gate = self.handles[shard_gate].get_tensor(k_gate)
            t_up = self.handles[shard_up].get_tensor(k_up)
            t_down = self.handles[shard_down].get_tensor(k_down)
            self.cpu_expert_weights[expert_id] = (t_gate, t_up, t_down)
        return self.cpu_expert_weights[expert_id]

    def _cpu_expert_exec(self, expert_id: int, toks_gpu: torch.Tensor) -> torch.Tensor:
        """Executes cold expert on CPU in native BF16, streaming activations over PCIe."""
        M = toks_gpu.shape[0]
        if M > self.max_cpu_batch:
            self.max_cpu_batch = M
            self.cpu_act_in = torch.empty((self.max_cpu_batch, self.cfg.hidden_size), dtype=torch.bfloat16, pin_memory=True)
            self.cpu_act_out = torch.empty((self.max_cpu_batch, self.cfg.hidden_size), dtype=torch.bfloat16, pin_memory=True)

        with torch.cuda.stream(self.act_stream):
            self.cpu_act_in[:M].copy_(toks_gpu, non_blocking=True)
        self.act_stream.synchronize()

        Wg, Wu, Wd = self._get_cpu_expert_weights(expert_id)
        toks_cpu = self.cpu_act_in[:M]

        with torch.no_grad():
            h_g = F.linear(toks_cpu, Wg)
            h_u = F.linear(toks_cpu, Wu)
            act = F.silu(h_g) * h_u
            y = F.linear(act, Wd)
            self.cpu_act_out[:M].copy_(y)

        out = torch.empty_like(toks_gpu)
        with torch.cuda.stream(self.act_stream):
            out.copy_(self.cpu_act_out[:M], non_blocking=True)
        torch.cuda.current_stream(self.device).wait_stream(self.act_stream)
        return out

    def _adetr_column_exec(self, expert_id: int, slot_idx: int, toks_gpu: torch.Tensor, skip_dma: bool = False) -> torch.Tensor:
        """ADETR per-slot column split (COLOSSUS column-level, exact SA-FFN).

        Splits intermediate dim I into hot columns (CPU, zero weight movement)
        + cold columns (DMA into slot slices, GPU computes y_cold).
        y = y_hot + y_cold == native full-expert output.
        DMA per miss = adetr_ratio * 45MB (0.5 -> 22.5MB, 0.25 -> 11.25MB).
        No permanent hot table: residency lives per-slot, streaming per-miss.
        """
        Wg_full, Wu_full, Wd_full = self._get_cpu_expert_weights(expert_id)
        I = Wg_full.shape[0]
        i_cold = max(1, int(I * self.adetr_ratio))
        i_hot = I - i_cold
        slot_mod = self.slots[slot_idx]
        M = toks_gpu.shape[0]

        # 1. DMA cold slices only into slot (gate/up rows, down cols).
        # skip_dma=True when slot already holds this expert's cold slices (column-hit reuse).
        if not skip_dma:
            with torch.no_grad():
                with torch.cuda.stream(self.dma_stream):
                    slot_mod.gate_proj.weight[i_hot:].copy_(Wg_full[i_hot:], non_blocking=True)
                    slot_mod.up_proj.weight[i_hot:].copy_(Wu_full[i_hot:], non_blocking=True)
                    slot_mod.down_proj.weight[:, i_hot:].copy_(Wd_full[:, i_hot:], non_blocking=True)
            torch.cuda.current_stream(self.device).wait_stream(self.dma_stream)
            self.dma_bytes += int((Wg_full[i_hot:].nbytes + Wu_full[i_hot:].nbytes + Wd_full[:, i_hot:].nbytes))

        # 2. GPU cold partial from slot slices
        with torch.no_grad():
            h_g_c = F.linear(toks_gpu, slot_mod.gate_proj.weight[i_hot:])
            h_u_c = F.linear(toks_gpu, slot_mod.up_proj.weight[i_hot:])
            y_cold = F.linear(F.silu(h_g_c) * h_u_c, slot_mod.down_proj.weight[:, i_hot:])

        # 3. CPU hot partial (weights stay in DDR5; only 10KB activation moves)
        if M > self.max_cpu_batch:
            self.max_cpu_batch = M
            self.cpu_act_in = torch.empty((self.max_cpu_batch, self.cfg.hidden_size), dtype=torch.bfloat16, pin_memory=True)
            self.cpu_act_out = torch.empty((self.max_cpu_batch, self.cfg.hidden_size), dtype=torch.bfloat16, pin_memory=True)
        with torch.cuda.stream(self.act_stream):
            self.cpu_act_in[:M].copy_(toks_gpu, non_blocking=True)
        self.act_stream.synchronize()
        with torch.no_grad():
            x_cpu = self.cpu_act_in[:M]
            y_hot = F.linear(F.silu(F.linear(x_cpu, Wg_full[:i_hot])) * F.linear(x_cpu, Wu_full[:i_hot]), Wd_full[:, :i_hot])
            self.cpu_act_out[:M].copy_(y_hot)
        y_hot_gpu = torch.empty_like(toks_gpu)
        with torch.cuda.stream(self.act_stream):
            y_hot_gpu.copy_(self.cpu_act_out[:M], non_blocking=True)
        torch.cuda.current_stream(self.device).wait_stream(self.act_stream)

        # 4. Bind slot to this expert. Hot region is stale by design: record partial
        # so future hits reuse cold slices (skip_dma) instead of full-slot forward.
        if not skip_dma:
            if slot_idx in self.slot_to_expert:
                old_exp = self.slot_to_expert[slot_idx]
                self.expert_to_slot.pop(old_exp, None)
            self.slot_to_expert[slot_idx] = expert_id
            self.expert_to_slot[expert_id] = slot_idx
            self.slot_col_partial[slot_idx] = True
        self.adetr_col_dispatches += 1
        return (y_hot_gpu + y_cold).to(toks_gpu.dtype)

    def warm_up_slots(self, initial_experts: List[int]):
        """Pre-populates dynamic slots with initial experts."""
        for slot_idx, exp_id in enumerate(initial_experts[:self.capacity]):
            self._load_expert_to_slot(exp_id, slot_idx)

    def _load_expert_to_slot(self, expert_id: int, slot_idx: int):
        slot_mod = self.slots[slot_idx]
        t_gate, t_up, t_down = self._get_cpu_expert_weights(expert_id)

        with torch.no_grad():
            with torch.cuda.stream(self.dma_stream):
                slot_mod.gate_proj.weight.copy_(t_gate, non_blocking=True)
                slot_mod.up_proj.weight.copy_(t_up, non_blocking=True)
                slot_mod.down_proj.weight.copy_(t_down, non_blocking=True)

        if slot_idx in self.slot_to_expert:
            old_exp = self.slot_to_expert[slot_idx]
            self.expert_to_slot.pop(old_exp, None)

        self.slot_to_expert[slot_idx] = expert_id
        self.expert_to_slot[expert_id] = slot_idx
        self.slot_col_partial[slot_idx] = False
        self.dma_bytes += (t_gate.nbytes + t_up.nbytes + t_down.nbytes)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        identity = hidden_states
        orig_shape = hidden_states.shape

        # 1. Gate routing (returns topk_idx, topk_weight, aux_loss)
        topk_indices, topk_weights, _ = self.gate(hidden_states)
        needed_experts = topk_indices.unique().tolist()
        flat_topk = topk_indices.view(-1)

        # 2. Compute shared experts (permanently resident)
        shared_out = self.shared_experts(identity)

        # 3. Dynamic Slot Management & Expert Streaming
        gpu_hits = set([e for e in needed_experts if e in self.expert_to_slot])
        miss_ids = [e for e in needed_experts if e not in self.expert_to_slot]

        # Protect resident hits in LRU
        for exp_id in gpu_hits:
            self.hits += 1
            slot_idx = self.expert_to_slot[exp_id]
            self.slot_lru.remove(slot_idx)
            self.slot_lru.append(slot_idx)

        cpu_ids = set()
        gpu_misses = []
        # Job 1814 fix: freq-aware retention + prefill guard. Frequent experts (seen>=3)
        # stay on GPU even if M small; prefill (N>8) never goes CPU (fixes 67s prefill).
        n_tokens = flat_topk.numel() // max(1, topk_indices.shape[-1])
        use_hetero = self.enable_hetero and n_tokens <= self.hetero_max_tokens
        for _e in needed_experts:
            self.expert_freq[_e] = self.expert_freq.get(_e, 0) + 1
        if use_hetero:
            for exp_id in miss_ids:
                tok_count = (flat_topk == exp_id).sum().item()
                if tok_count <= self.cpu_token_threshold and self.expert_freq.get(exp_id, 0) < self.hetero_freq_retain:
                    cpu_ids.add(exp_id)
                else:
                    gpu_misses.append(exp_id)
        else:
            gpu_misses = miss_ids

        # ADETR column-level: all misses go through per-slot hot/cold split
        # (supersedes hetero full-CPU path; hot half stays in DDR5, cold half DMA'd).
        # Global column pool (Experiment C): GPU-only, supersedes both paths above.
        use_adetr = self.adetr_ratio < 1.0
        use_pool = self.col_pool is not None
        adetr_ids = set(miss_ids) if (use_adetr and not use_pool) else set()
        pool_ids = set(miss_ids) if use_pool else set()
        if use_pool:
            cpu_ids.clear()
            gpu_misses = []

        # Stream heavy misses into GPU slots
        if gpu_misses:
            for exp_id in gpu_misses:
                self.misses += 1
                slot_idx = self.slot_lru.pop(0)
                self._load_expert_to_slot(exp_id, slot_idx)
                self.slot_lru.append(slot_idx)
            torch.cuda.current_stream(self.device).wait_stream(self.dma_stream)

        # 4. Compute routed experts
        cnts = topk_indices.new_zeros((topk_indices.shape[0], self.cfg.n_routed_experts))
        cnts.scatter_(1, topk_indices, 1)
        tokens_per_expert = cnts.sum(dim=0).cpu().numpy()
        idxs = topk_indices.view(-1).argsort()
        flat_x = hidden_states.view(-1, hidden_states.shape[-1])
        sorted_tokens = flat_x[idxs // topk_indices.shape[1]]

        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue

            tokens_for_this = sorted_tokens[start_idx:end_idx]
            if i in cpu_ids:
                self.cpu_dispatches += 1
                out_this = self._cpu_expert_exec(i, tokens_for_this)
            elif i in pool_ids:
                self.misses += 1
                slot_idx = self.slot_lru.pop(0)
                moved = self.col_pool.assemble(
                    self.layer_idx, i, self._get_cpu_expert_weights(i),
                    self.slots[slot_idx], self.dma_stream)
                self.dma_bytes += moved
                if slot_idx in self.slot_to_expert:
                    self.expert_to_slot.pop(self.slot_to_expert[slot_idx], None)
                self.slot_to_expert[slot_idx] = i
                self.expert_to_slot[i] = slot_idx
                self.slot_col_partial[slot_idx] = False
                self.slot_lru.append(slot_idx)
                torch.cuda.current_stream(self.device).wait_stream(self.dma_stream)
                out_this = self.slots[slot_idx](tokens_for_this)
            elif i in adetr_ids:
                self.misses += 1
                slot_idx = self.slot_lru.pop(0)
                out_this = self._adetr_column_exec(i, slot_idx, tokens_for_this)
                self.slot_lru.append(slot_idx)
            else:
                if i in self.expert_to_slot:
                    slot_idx = self.expert_to_slot[i]
                    if use_adetr and self.slot_col_partial.get(slot_idx, False):
                        # Column-hit: cold slices already resident, skip DMA
                        self.hits += 1
                        self.slot_lru.remove(slot_idx)
                        self.slot_lru.append(slot_idx)
                        out_this = self._adetr_column_exec(i, slot_idx, tokens_for_this, skip_dma=True)
                    else:
                        out_this = self.slots[slot_idx](tokens_for_this)
                else:
                    self.misses += 1
                    slot_idx = self.slot_lru.pop(0)
                    self._load_expert_to_slot(i, slot_idx)
                    self.slot_lru.append(slot_idx)
                    torch.cuda.current_stream(self.device).wait_stream(self.dma_stream)
                out_this = self.slots[slot_idx](tokens_for_this)

            outputs.append(out_this)
            start_idx = end_idx

        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)
        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        final_out = (
            new_x.view(*topk_indices.shape, -1)
            .type(topk_weights.dtype)
            .mul_(topk_weights.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )

        return shared_out + final_out.view(*orig_shape)


# ─────────────────────────────────────────────────────────────────────────────
# 3. End-to-End Serving Engine Execution
# ─────────────────────────────────────────────────────────────────────────────
def serve_deepseek(args):
    print("=" * 80)
    print("  COLOSSUS PRODUCTION SERVING: DeepSeek-Coder-V2 (236B MoE)")
    if args.num_gpus == 1:
        print("  Hardware Target: 1x NVIDIA H100 NVL (Consolidating 8-GPU Cluster onto 1 GPU!)")
    else:
        print("  Hardware Target: 2x NVIDIA H100 NVL (Consolidating 8-GPU Cluster)")
    print("=" * 80)

    assert torch.cuda.is_available(), "CUDA GPU required!"
    if args.num_gpus == 1:
        dev0 = torch.device("cuda:0")
        dev1 = dev0
        p0 = torch.cuda.get_device_properties(dev0)
        p1 = p0
        print(f"  GPU 0: {p0.name} | Total HBM3: {p0.total_memory / (1024**3):.1f} GB")
    else:
        assert torch.cuda.device_count() >= 2, "Dual GPUs required for 2-GPU serving!"
        dev0 = torch.device("cuda:0")
        dev1 = torch.device("cuda:1")
        p0 = torch.cuda.get_device_properties(dev0)
        p1 = torch.cuda.get_device_properties(dev1)
        print(f"  GPU 0: {p0.name} | Total HBM3: {p0.total_memory / (1024**3):.1f} GB")
        print(f"  GPU 1: {p1.name} | Total HBM3: {p1.total_memory / (1024**3):.1f} GB")
    print(f"  Dynamic Slot Capacity C = {args.capacity} slots per MoE layer")


    # Load Tokenizer & Config
    print(f"\n[1] Loading Tokenizer & Architecture Config from {MODEL_PATH}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    cfg = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print(f"  Loaded in {time.time() - t0:.2f}s | Vocab: {len(tokenizer):,} | Layers: {cfg.num_hidden_layers} | Experts: {cfg.n_routed_experts} (top-{cfg.num_experts_per_tok})")

    # Open Safetensors Handles (Direct Zero-Copy mmap)
    print(f"\n[2] Pre-opening Safetensors Shard Handles (Host DDR5 Zero-Copy mmap)...")
    t0 = time.time()
    with open(f"{MODEL_PATH}/model.safetensors.index.json", "r") as f:
        weight_map = json.load(f)["weight_map"]
    shards = sorted(list(set(weight_map.values())))
    handles = {s: safe_open(os.path.join(MODEL_PATH, s), framework="pt", device="cpu") for s in shards}
    print(f"  Pre-opened {len(handles)} shards in {time.time() - t0:.2f}s.")

    # Instantiate Meta Model Skeleton
    print(f"\n[3] Instantiating Meta Model Skeleton...")
    t0 = time.time()
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True, torch_dtype=torch.bfloat16)
    print(f"  Instantiated 60-layer skeleton in {time.time() - t0:.2f}s.")

    # Load Non-Routed Weights to Respective GPUs
    print(f"\n[4] Materializing Non-Routed Weights (~25 GB) across Dual H100 NVL...")
    t0 = time.time()
    non_routed_keys = {k: v for k, v in weight_map.items() if not (".mlp.experts." in k and ".shared_experts" not in k)}

    for shard_file in sorted(list(set(non_routed_keys.values()))):
        handle = handles[shard_file]
        for k in handle.keys():
            if k not in non_routed_keys:
                continue
            if k.startswith("model.embed_tokens."):
                target_dev = dev0
            elif k.startswith("model.layers."):
                l_idx = int(k.split(".")[2])
                target_dev = dev0 if (args.num_gpus == 1 or l_idx < 30) else dev1
            else:
                target_dev = dev0 if args.num_gpus == 1 else dev1
            t = handle.get_tensor(k)
            set_module_tensor_to_device(model, k, target_dev, value=t.to(torch.bfloat16))

    torch.cuda.synchronize(dev0)
    if args.num_gpus > 1:
        torch.cuda.synchronize(dev1)
    print(f"  Non-routed parameters materialized in {time.time() - t0:.2f}s.")
    print(f"    GPU 0 Allocated: {torch.cuda.memory_allocated(dev0) / (1024**3):.2f} GB")
    if args.num_gpus > 1:
        print(f"    GPU 1 Allocated: {torch.cuda.memory_allocated(dev1) / (1024**3):.2f} GB")

    # Re-initialize RoPE rotary embeddings on target devices to eliminate meta buffers
    print(f"\n[4.5] Initializing RoPE Rotary Embeddings on Target GPUs...")
    for l_idx in range(cfg.num_hidden_layers):
        layer = model.model.layers[l_idx]
        target_dev = dev0 if (args.num_gpus == 1 or l_idx < 30) else dev1
        layer.self_attn._init_rope()
        layer.self_attn.rotary_emb.to(target_dev)

    # Install COLOSSUS Dynamic Slot Wrappers for Layers 1..59
    print(f"\n[5] Installing COLOSSUS Dynamic Slot Wrappers (Layers 1..59)...")
    t0 = time.time()
    dma_stream0 = torch.cuda.Stream(device=dev0)
    dma_stream1 = dma_stream0 if args.num_gpus == 1 else torch.cuda.Stream(device=dev1)
    colossus_wrappers: List[DeepSeekColossusMoEWrapper] = []
    # Experiment C: one global column pool per GPU (shared across all layers on that device).
    # Budget-derived capacity; 0 disables (whole-expert / hetero / adetr paths unchanged).
    col_pools = {}
    if args.col_pool_gb > 0:
        for dev in ([dev0] if args.num_gpus == 1 else [dev0, dev1]):
            col_pools[str(dev)] = GlobalColumnPool(
                device=dev, dtype=torch.bfloat16, hidden=cfg.hidden_size,
                inter=cfg.moe_intermediate_size, block=args.col_block,
                budget_gb=args.col_pool_gb / (1 if args.num_gpus == 1 else 2))
        p0pool = col_pools[str(dev0)]
        print(f"  GlobalColumnPool: block={args.col_block} cols, budget={args.col_pool_gb}GB total, "
              f"cap~{p0pool.cap} blocks/pool ({p0pool.cap * p0pool.per_block_bytes / (1024**3):.2f}GB), "
              f"universe/layer={cfg.n_routed_experts * p0pool.nblocks} blocks")

    for l_idx in range(1, cfg.num_hidden_layers):
        layer = model.model.layers[l_idx]
        dev = dev0 if (args.num_gpus == 1 or l_idx < 30) else dev1
        dma_stream = dma_stream0 if (args.num_gpus == 1 or l_idx < 30) else dma_stream1
        cap = args.capacity if dev == dev0 else min(args.capacity, args.capacity_gpu1)

        wrapper = DeepSeekColossusMoEWrapper(
            layer_idx=l_idx,
            moe_module=layer.mlp,
            cfg=cfg,
            device=dev,
            capacity=cap,
            handles=handles,
            weight_map=weight_map,
            dma_stream=dma_stream,
            enable_hetero=args.enable_hetero,
            cpu_token_threshold=args.cpu_token_threshold,
            adetr_ratio=args.adetr_ratio,
            col_pool=col_pools.get(str(dev)),
        )

        if args.warm_slots:
            # Pre-warm slots with first C experts
            wrapper.warm_up_slots(list(range(cap)))

        layer.mlp = wrapper
        colossus_wrappers.append(wrapper)

    torch.cuda.synchronize(dev0)
    if args.num_gpus > 1:
        torch.cuda.synchronize(dev1)
    print(f"  Installed {len(colossus_wrappers)} COLOSSUS wrappers in {time.time() - t0:.2f}s.")
    print(f"    GPU 0 Allocated (with C={args.capacity} slots): {torch.cuda.memory_allocated(dev0) / (1024**3):.2f} GB")
    if args.num_gpus > 1:
        print(f"    GPU 1 Allocated (with C={args.capacity_gpu1} slots): {torch.cuda.memory_allocated(dev1) / (1024**3):.2f} GB")

    if args.num_gpus > 1:
        # Install P2P Bridge on all layers 30..59 so hidden_states, position_ids, attention_mask are on dev1
        print(f"\n[6] Installing NVLink P2P Hooks on Layers 30..59...")
        def gpu1_pre_hook(module, args, kwargs):
            new_args = [
                a.to(dev1, non_blocking=True) if isinstance(a, torch.Tensor) and a.device != dev1 else a
                for a in args
            ]
            new_kwargs = {
                k: (v.to(dev1, non_blocking=True) if isinstance(v, torch.Tensor) and v.device != dev1 else v)
                for k, v in kwargs.items()
            }
            return tuple(new_args), new_kwargs

        for l_idx in range(30, cfg.num_hidden_layers):
            model.model.layers[l_idx].register_forward_pre_hook(gpu1_pre_hook, with_kwargs=True)
    else:
        print(f"\n[6] Single GPU serving on {p0.name} (No cross-GPU NVLink hooks needed)")
    model.eval()



    # ─────────────────────────────────────────────────────────────────────────
    # Run End-to-End Generation Benchmark
    # ─────────────────────────────────────────────────────────────────────────
    prompt = args.prompt
    print(f"\n[7] Starting Generation Benchmark:")
    print(f"  Prompt: {repr(prompt)}")
    print(f"  Max New Tokens: {args.max_new_tokens}")

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(dev0)
    prompt_len = input_ids.shape[1]
    print(f"  Prompt Length: {prompt_len} tokens")

    # Reset cache metrics before generation
    for w in colossus_wrappers:
        w.hits = 0
        w.misses = 0
        w.cpu_dispatches = 0
        w.adetr_col_dispatches = 0
        w.dma_bytes = 0
    for p in col_pools.values():
        p.reset_stats()

    generated_ids = input_ids.clone()

    # 1. Prefill Phase
    print(f"\n  --- Prefill Phase ({prompt_len} tokens) ---")
    torch.cuda.synchronize(dev0)
    torch.cuda.synchronize(dev1)
    t_prefill_start = time.perf_counter()

    from transformers.cache_utils import DynamicCache
    past_key_values = DynamicCache() if args.use_cache else None

    with torch.no_grad():
        out = model(input_ids=generated_ids, past_key_values=past_key_values, use_cache=args.use_cache)
        logits = out.logits  # [1, prompt_len, vocab_size] on dev1
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True).to(dev0)
        past_key_values = getattr(out, "past_key_values", None)


    torch.cuda.synchronize(dev0)
    torch.cuda.synchronize(dev1)
    t_prefill = time.perf_counter() - t_prefill_start
    prefill_tps = prompt_len / t_prefill
    print(f"  Prefill Time : {t_prefill*1000:.2f} ms ({prefill_tps:.2f} tok/s)")
    sys.stdout.flush()

    generated_ids = torch.cat([generated_ids, next_token], dim=1)

    # 2. Decode Phase (Token-by-Token)
    print(f"\n  --- Decode Phase ({args.max_new_tokens - 1} tokens) [KV-Cache: {args.use_cache}] ---")
    sys.stdout.flush()
    decode_latencies = []
    decode_step_details = []

    last_hits = sum(w.hits for w in colossus_wrappers)
    last_misses = sum(w.misses for w in colossus_wrappers)
    last_cpu = sum(w.cpu_dispatches for w in colossus_wrappers)
    last_dma = sum(w.dma_bytes for w in colossus_wrappers)
    last_pool_hits = sum(p.hits for p in col_pools.values())
    last_pool_miss = sum(p.misses for p in col_pools.values())

    for step in range(args.max_new_tokens - 1):
        torch.cuda.synchronize(dev0)
        torch.cuda.synchronize(dev1)
        t_step_start = time.perf_counter()

        with torch.no_grad():
            if args.use_cache and past_key_values is not None:
                out = model(input_ids=next_token, past_key_values=past_key_values, use_cache=True)
            else:
                out = model(input_ids=generated_ids, use_cache=False)
            logits = out.logits
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True).to(dev0)
            if args.use_cache:
                past_key_values = getattr(out, "past_key_values", None)

        torch.cuda.synchronize(dev0)
        torch.cuda.synchronize(dev1)
        t_step = time.perf_counter() - t_step_start
        decode_latencies.append(t_step)

        # Per-step cache metrics
        cur_hits = sum(w.hits for w in colossus_wrappers)
        cur_misses = sum(w.misses for w in colossus_wrappers)
        cur_cpu = sum(w.cpu_dispatches for w in colossus_wrappers)
        cur_dma = sum(w.dma_bytes for w in colossus_wrappers)
        cur_pool_h = sum(p.hits for p in col_pools.values())
        cur_pool_m = sum(p.misses for p in col_pools.values())

        step_hits = cur_hits - last_hits
        step_misses = cur_misses - last_misses
        step_cpu = cur_cpu - last_cpu
        step_pool_h = cur_pool_h - last_pool_hits
        step_pool_m = cur_pool_m - last_pool_miss
        step_pool_denom = step_pool_h + step_pool_m
        step_pool_rate = (step_pool_h / step_pool_denom * 100.0) if step_pool_denom else 0.0
        step_lookups = step_hits + step_misses + step_cpu
        step_hit_rate = (step_hits / step_lookups * 100.0) if step_lookups > 0 else 0.0
        step_dma_mb = (cur_dma - last_dma) / (1024**2)
        step_dma_saved_mb = step_cpu * 45.0

        last_hits, last_misses, last_cpu, last_dma = cur_hits, cur_misses, cur_cpu, cur_dma
        last_pool_hits, last_pool_miss = cur_pool_h, cur_pool_m

        generated_ids = torch.cat([generated_ids, next_token], dim=1)
        tok_str = tokenizer.decode(next_token[0], skip_special_tokens=False)
        print(f"    Token {step+1:2d}/{args.max_new_tokens-1:2d} | Latency: {t_step*1000:6.1f} ms | Hits: {step_hits:3d}/{step_lookups:3d} ({step_hit_rate:5.1f}%) | CPU Fallback: {step_cpu:2d} ({step_dma_saved_mb:5.1f}MB saved) | Misses: {step_misses:2d} | DMA: {step_dma_mb:5.1f} MB | ColPool: {step_pool_h:4d}h/{step_pool_m:4d}m ({step_pool_rate:4.1f}%) | Tok: {repr(tok_str)}")
        sys.stdout.flush()

        decode_step_details.append({
            "step": step + 1,
            "latency_ms": t_step * 1000,
            "step_hits": step_hits,
            "step_misses": step_misses,
            "step_cpu_dispatches": step_cpu,
            "step_col_pool_hits": step_pool_h,
            "step_col_pool_misses": step_pool_m,
            "step_col_pool_hit_rate_pct": step_pool_rate,
            "step_dma_saved_mb": step_dma_saved_mb,
            "step_hit_rate_pct": step_hit_rate,
            "step_dma_mb": step_dma_mb,
            "token": tok_str,
        })




    # ─────────────────────────────────────────────────────────────────────────
    # Summary Metrics
    # ─────────────────────────────────────────────────────────────────────────
    total_decode_time = sum(decode_latencies)
    avg_decode_lat = total_decode_time / len(decode_latencies) if decode_latencies else 0.0
    decode_tps = 1.0 / avg_decode_lat if avg_decode_lat > 0 else 0.0

    total_hits = sum(w.hits for w in colossus_wrappers)
    total_misses = sum(w.misses for w in colossus_wrappers)
    total_cpu = sum(w.cpu_dispatches for w in colossus_wrappers)
    total_adetr = sum(getattr(w, "adetr_col_dispatches", 0) for w in colossus_wrappers)
    total_pool_h = sum(p.hits for p in col_pools.values())
    total_pool_m = sum(p.misses for p in col_pools.values())
    total_pool_denom = total_pool_h + total_pool_m
    pool_hit_rate = (total_pool_h / total_pool_denom * 100.0) if total_pool_denom else 0.0
    pool_gb = sum(len(p.entries) * p.per_block_bytes for p in col_pools.values()) / (1024 ** 3)
    total_lookups = total_hits + total_misses + total_cpu
    hit_rate = (total_hits / total_lookups * 100.0) if total_lookups > 0 else 0.0
    total_dma_mb = sum(w.dma_bytes for w in colossus_wrappers) / (1024**2)
    total_dma_saved_mb = total_cpu * 45.0 + total_adetr * 45.0 * (1.0 - args.adetr_ratio)

    peak_hbm0 = torch.cuda.max_memory_allocated(dev0) / (1024**3)
    peak_hbm1 = torch.cuda.max_memory_allocated(dev1) / (1024**3)

    gen_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    print("\n" + "=" * 80)
    print("  COLOSSUS SERVING BENCHMARK RESULTS")
    print("=" * 80)
    print(f"  Model                     : DeepSeek-Coder-V2-Instruct (236B MoE)")
    print(f"  Uncompressed Weights      : 471.5 GB BF16")
    if args.num_gpus == 1:
        print(f"  Hardware Footprint        : 1x NVIDIA H100 NVL (Consolidating 8-GPU Cluster onto 1 GPU!)")
    else:
        print(f"  Hardware Footprint        : 2x NVIDIA H100 NVL (Consolidating 8-GPU Cluster)")
    print(f"  Dynamic Slot Capacity C   : {args.capacity} slots / layer ({(args.capacity / cfg.n_routed_experts)*100:.1f}% expert residency)")
    print(f"  Routed Expert Reduction   : {(1.0 - args.capacity / cfg.n_routed_experts)*100:.1f}% reduction in routed expert GPU memory")
    print(f"  Compute-to-Data Engine    : {'Enabled (Threshold <= ' + str(args.cpu_token_threshold) + ')' if args.enable_hetero else 'Disabled'}")
    print(f"  ADETR Column Split        : {'Disabled (whole-expert 45MB/miss)' if args.adetr_ratio >= 1.0 else f'Enabled ratio={args.adetr_ratio} (~{45.0*args.adetr_ratio:.1f}MB/miss, {total_adetr:,} col-dispatches)'}" )
    if args.col_pool_gb > 0:
        print(f"  GlobalColumnPool          : block={args.col_block} cols, budget={args.col_pool_gb}GB, "
              f"blocks={total_pool_h + total_pool_m:,} lookups ({pool_hit_rate:.1f}% hit), resident={pool_gb:.2f}GB")
    print("-" * 80)
    print(f"  Prefill Latency           : {t_prefill*1000:7.1f} ms ({prefill_tps:5.1f} tok/s for {prompt_len} tokens)")
    print(f"  Decode Latency (Avg)      : {avg_decode_lat*1000:7.1f} ms / token")
    print(f"  Decode Throughput         : {decode_tps:7.2f} tokens / sec")
    print(f"  Cache Hits                : {total_hits:,} ({hit_rate:.1f}%)")
    print(f"  CPU Cold Fallbacks        : {total_cpu:,} ({total_dma_saved_mb:,.1f} MB PCIe DMA avoided!)")
    print(f"  Cache Misses (Cold DMA)   : {total_misses:,}")
    print(f"  Total PCIe DMA Transferred: {total_dma_mb:7.1f} MB")
    print(f"  Peak VRAM GPU 0           : {peak_hbm0:7.2f} GB / 93.1 GB")
    if args.num_gpus > 1:
        print(f"  Peak VRAM GPU 1           : {peak_hbm1:7.2f} GB / 93.1 GB")
    print("-" * 80)
    print(f"  Generated Text Output:")
    print(f"  {repr(gen_text)}")
    print("=" * 80)

    # Save to JSON
    results = {
        "model": "DeepSeek-Coder-V2-Instruct",
        "parameters": "236B",
        "routed_experts": 160,
        "active_experts": 6,
        "shared_experts": 2,
        "num_gpus": args.num_gpus,
        "gpus": [p0.name] if args.num_gpus == 1 else [p0.name, p1.name],
        "capacity_slots": args.capacity,
        "residency_reduction_pct": (1.0 - args.capacity / cfg.n_routed_experts) * 100.0,
        "prompt": prompt,
        "prompt_tokens": prompt_len,
        "generated_tokens": args.max_new_tokens,
        "prefill_ms": t_prefill * 1000,
        "prefill_tps": prefill_tps,
        "decode_avg_ms": avg_decode_lat * 1000,
        "decode_tps": decode_tps,
        "decode_latencies_ms": [l * 1000 for l in decode_latencies],
        "decode_step_details": decode_step_details,
        "total_hits": total_hits,
        "total_misses": total_misses,
        "total_cpu_dispatches": total_cpu,
        "total_adetr_col_dispatches": total_adetr,
        "col_pool": {
            "block_cols": args.col_block,
            "budget_gb": args.col_pool_gb,
            "block_hits": total_pool_h,
            "block_misses": total_pool_m,
            "block_hit_rate_pct": pool_hit_rate,
            "resident_gb": pool_gb,
        },
        "total_dma_saved_mb": total_dma_saved_mb,
        "hit_rate_pct": hit_rate,
        "total_dma_mb": total_dma_mb,
        "enable_hetero": args.enable_hetero,
        "cpu_token_threshold": args.cpu_token_threshold,
        "adetr_ratio": args.adetr_ratio,
        "peak_vram_gpu0_gb": peak_hbm0,
        "peak_vram_gpu1_gb": peak_hbm1 if args.num_gpus > 1 else None,
        "generated_text": gen_text,
    }

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved benchmark results to: {args.output_json}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--prompt", type=str, default="def quicksort(arr):")
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--capacity", type=int, default=12)
    parser.add_argument("--capacity_gpu1", type=int, default=7)
    parser.add_argument("--warm_slots", action="store_true", default=False)
    parser.add_argument("--enable_hetero", action="store_true", default=True)
    parser.add_argument("--no_hetero", dest="enable_hetero", action="store_false")
    parser.add_argument("--cpu_token_threshold", type=int, default=1)
    parser.add_argument("--adetr_ratio", type=float, default=1.0,
                        help="Cold column fraction DMA'd per miss (1.0=whole-expert, 0.5=ADETR-50%%, 0.25, 0.1)")
    parser.add_argument("--col_pool_gb", type=float, default=0.0,
                        help="Experiment C: GlobalColumnPool VRAM budget in GB (0=disabled). GPU-only column-LFU.")
    parser.add_argument("--col_block", type=int, default=128,
                        help="Column-block width (intermediate-dim cols) for pool transfers/GEMM slices")

    parser.add_argument("--use_cache", action="store_true", default=True)
    parser.add_argument("--no_cache", dest="use_cache", action="store_false")
    parser.add_argument("--output_json", type=str, default="/home/palakm/MoEServingSim/aditya/deepseek_serving_colossus.json")
    args = parser.parse_args()


    serve_deepseek(args)
