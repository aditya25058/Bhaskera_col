#!/usr/bin/env python3
"""
serve_deepseek_colossus.py
==========================
End-to-End Serving Engine for DeepSeek-Coder-V2 (236B MoE, 160 Experts)
Consolidating 8 GPUs onto 2x NVIDIA H100 NVL via COLOSSUS.

Architecture:
- Host CPU DDR5 (503 GB RAM): Holds the 445 GB cold expert pool (pinned).
- Dual H100 NVL (188 GB HBM3):
  - GPU 0: Layers 0-29 non-MoE + shared experts + C dynamic slots.
  - GPU 1: Layers 30-59 non-MoE + shared experts + C dynamic slots + LM Head.
- Dynamic Expert Caching: Top-6 routed experts fetched over PCIe Gen5 (51.6 GB/s).
"""

import os
import sys
import time
from collections import OrderedDict
from typing import Dict, List, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer

MODEL_PATH = "/home/palakm/MoEServingSim/aditya/models/DeepSeek-Coder-V2-Instruct"

print("=" * 80)
print("  COLOSSUS PRODUCTION SERVING: DeepSeek-Coder-V2 (236B) on 2x H100 NVL")
print("=" * 80)

assert torch.cuda.is_available(), "CUDA is required!"
assert torch.cuda.device_count() >= 2, f"Expected 2 GPUs, found {torch.cuda.device_count()}"

dev0 = torch.device("cuda:0")
dev1 = torch.device("cuda:1")

print(f"[1] Active Accelerators:")
print(f"  GPU 0: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB)")
print(f"  GPU 1: {torch.cuda.get_device_name(1)} ({torch.cuda.get_device_properties(1).total_memory / (1024**3):.1f} GB)")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load Tokenizer & Config
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[2] Loading Tokenizer & Architecture Config from {MODEL_PATH}...")
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
cfg = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
print(f"  Loaded in {time.time() - t0:.2f}s | Vocab: {len(tokenizer):,} | Layers: {cfg.num_hidden_layers} | Experts: {cfg.n_routed_experts} (top-{cfg.num_experts_per_tok})")

# ─────────────────────────────────────────────────────────────────────────────
# 2. COLOSSUS Dynamic Slot Wrapper for DeepSeek-V2 MoE
# ─────────────────────────────────────────────────────────────────────────────
class DeepSeekColossusMoEWrapper(nn.Module):
    """
    Wraps DeepseekV2MoE:
    - Shared experts: permanently resident on GPU.
    - 160 Routed experts: stored in pinned host memory, dynamically slotted into C slots.
    """
    def __init__(self, moe_module: nn.Module, device: torch.device, capacity: int = 12, hot_ratio: float = 0.25):
        super().__init__()
        self.moe = moe_module
        self.device = device
        self.capacity = capacity
        self.hot_ratio = hot_ratio
        
        self.gate = moe_module.gate.to(device)
        self.shared_experts = moe_module.shared_experts.to(device)
        
        # Pinned host weights dictionary
        self.host_weights: Dict[int, Dict[str, torch.Tensor]] = {}
        for idx, expert in enumerate(moe_module.experts):
            self.host_weights[idx] = {
                "gate": expert.gate_proj.weight.data.pin_memory(),
                "up":   expert.up_proj.weight.data.pin_memory(),
                "down": expert.down_proj.weight.data.pin_memory(),
            }
        
        # Allocate C dynamic slots on GPU
        self.slots: List[nn.Module] = [
            type(moe_module.experts[0])(cfg, intermediate_size=cfg.moe_intermediate_size).to(device).to(torch.bfloat16)
            for _ in range(capacity)
        ]
        
        self.slot_to_expert: Dict[int, int] = {}
        self.expert_to_slot: Dict[int, int] = {}
        self.slot_lru: List[int] = list(range(capacity))
        
        self.hits = 0
        self.misses = 0
        self.total_tokens = 0
        self.dma_bytes = 0
        self.dma_stream = torch.cuda.Stream(device=device)

    def _load_expert_to_slot(self, expert_id: int, slot_idx: int):
        slot_mod = self.slots[slot_idx]
        host = self.host_weights[expert_id]
        
        with torch.cuda.stream(self.dma_stream):
            slot_mod.gate_proj.weight.copy_(host["gate"], non_blocking=True)
            slot_mod.up_proj.weight.copy_(host["up"], non_blocking=True)
            slot_mod.down_proj.weight.copy_(host["down"], non_blocking=True)
        
        # Update tracking
        if slot_idx in self.slot_to_expert:
            old_exp = self.slot_to_expert[slot_idx]
            self.expert_to_slot.pop(old_exp, None)
            
        self.slot_to_expert[slot_idx] = expert_id
        self.expert_to_slot[expert_id] = slot_idx
        self.dma_bytes += (host["gate"].nbytes + host["up"].nbytes + host["down"].nbytes)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        identity = hidden_states
        orig_shape = hidden_states.shape
        self.total_tokens += hidden_states.shape[0] * hidden_states.shape[1]
        
        # 1. Gate routing (returns topk_idx, topk_weight, aux_loss)
        topk_indices, topk_weights, _ = self.gate(hidden_states)
        
        needed_experts = topk_indices.unique().tolist()
        
        # 2. Dynamic cache resolution
        for exp_id in needed_experts:
            if exp_id in self.expert_to_slot:
                self.hits += 1
                slot = self.expert_to_slot[exp_id]
                self.slot_lru.remove(slot)
                self.slot_lru.append(slot)
            else:
                self.misses += 1
                evict_slot = self.slot_lru.pop(0)
                self._load_expert_to_slot(exp_id, evict_slot)
                self.slot_lru.append(evict_slot)
                
        # Sync DMA stream before compute
        torch.cuda.current_stream(self.device).wait_stream(self.dma_stream)
        
        # 3. Compute shared experts (always resident)
        out = self.shared_experts(identity)
        
        # 4. Compute routed experts via active slots
        flat_x = hidden_states.view(-1, hidden_states.shape[-1])
        flat_idx = topk_indices.view(-1, cfg.num_experts_per_tok)
        flat_w = topk_weights.view(-1, cfg.num_experts_per_tok)
        
        moe_out = torch.zeros_like(flat_x)
        for i in range(cfg.num_experts_per_tok):
            exp_ids = flat_idx[:, i]
            weights = flat_w[:, i].unsqueeze(-1)
            for e_id in exp_ids.unique().tolist():
                mask = (exp_ids == e_id)
                if mask.any():
                    slot_idx = self.expert_to_slot[e_id]
                    slot_out = self.slots[slot_idx](flat_x[mask])
                    moe_out[mask] += slot_out * weights[mask]
                    
        return out + moe_out.view(*orig_shape)

print("\nCOLOSSUS DeepSeek-V2 serving module defined successfully.")
