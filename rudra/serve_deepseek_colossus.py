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
import mmap
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


_VENV_SITE = "/home/palakm/MoEServingSim/aditya/venv/lib/python3.10/site-packages"


def _load_nvcomp():
    """Official nvidia.nvcomp bindings (namespace-shadow fix). Returns module or None."""
    try:
        import nvidia
        _nvd = os.path.join(_VENV_SITE, "nvidia")
        if _nvd not in list(nvidia.__path__):
            nvidia.__path__.append(_nvd)
        import nvidia.nvcomp as nvcomp
        return nvcomp
    except Exception:
        return None


class ShardMap:
    """Phase 1c-R: whole-file aligned mmap + parsed safetensors header, one mapping per shard.

    Tensor views are zero-copy slices of the mapping (no CPU copy). Registering the
    whole mapping once per shard amortizes cudaHostRegister over all its tensors
    (vs 19k per-tensor registrations that put mlock+TLB-shootdown on la critique).
    """

    _cache = {}

    def __init__(self, path):
        self.path = path
        self.f = open(path, "rb")
        self.header_len = int.from_bytes(self.f.read(8), "little")
        self.header = json.loads(self.f.read(self.header_len))
        self.data_start = 8 + self.header_len
        # ACCESS_COPY: writeable view, no disk I/O for reads (torch needs writeable buffer)
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_COPY)
        self.u8 = torch.frombuffer(self.mm, dtype=torch.uint8)

    @classmethod
    def get(cls, path):
        if path not in cls._cache:
            cls._cache[path] = ShardMap(path)
        return cls._cache[path]

    def view_tensor(self, key):
        info = self.header[key]
        b, e = info["data_offsets"]
        s = self.data_start + b
        return self.u8[s:s + (e - b)].view(torch.bfloat16).reshape(info["shape"])


class DeepSeekColossusMoEWrapper(nn.Module):
    """
    Dynamic Slot Residency MoE Wrapper:
    - Permanently resident: Gate, Shared Experts on GPU HBM.
    - Dynamic pool: C dynamic slots allocated on GPU HBM.
    - Cold experts: Streamed over PCIe Gen5 on demand from mmapped host handles.
    - Phase 1b: optional pinned staging (class-shared, ~90MB host) so DMA runs
      pinned->HBM at full link rate instead of mmap-source staged ~6GB/s.
    """
    # Phase 1b: class-shared pinned staging triple (~90MB host for I=1536/H=5120).
    # mmap (pageable) source -> cudaMemcpyAsync runs driver-staged at ~6GB/s;
    # pinned source -> true async DMA at link rate. One triple shared by all layers
    # (layers execute sequentially in decode; prefill reuses across layers in order).
    _pinned_staging = None
    _pinned_staging_key = None

    @classmethod
    def _get_pinned_staging(cls, inter: int, hidden: int, dtype: torch.dtype):
        key = (inter, hidden, str(dtype))
        if cls._pinned_staging is None or cls._pinned_staging_key != key:
            g = torch.empty((inter, hidden), dtype=dtype, device="cpu", pin_memory=True)
            u = torch.empty((inter, hidden), dtype=dtype, device="cpu", pin_memory=True)
            d = torch.empty((hidden, inter), dtype=dtype, device="cpu", pin_memory=True)
            cls._pinned_staging = (g, u, d)
            cls._pinned_staging_key = key
        return cls._pinned_staging

    # Phase 1c: cudaHostRegister zero-copy (no CPU work, full-rate async DMA).
    # safetensors tensors are mmap-backed (pageable) -> driver-staged ~6GB/s.
    # Registering them once pins the pages -> direct DMA at link rate.
    # Proven viable on H100 (probe job 1836: register_err 0, unregister_err 0).
    _cudart = None
    _cudart_failed = False
    # Phase 1c-R: shard-level registration ledger (class-shared: one entry per shard)
    _shard_registered = set()
    _shard_reg_seconds = 0.0
    _shard_reg_gb = 0.0

    @classmethod
    def _get_cudart(cls):
        if cls._cudart is None and not cls._cudart_failed:
            try:
                import ctypes
                import ctypes.util
                lib = ctypes.util.find_library("cudart")
                cls._cudart = ctypes.CDLL(lib) if lib else None
                if cls._cudart is None:
                    cls._cudart_failed = True
            except Exception:
                cls._cudart_failed = True
        return cls._cudart

    # Phase 2 (--ans): shared pinned staging + GPU scratch for ANS path.
    # Staging (host): comp_stg 16MB + lo_stg 24MB pinned. Scratch (per device):
    # lo",(per-proj 7.9MB) + comp segs + hi (per-proj) + single nvcomp Codec.
    # Layers execute sequentially so one shared set per device is race-free.
    _ans_staging = None
    _ans_scratch = {}
    _ans_codec = {}
    _ans_index = None
    _ans_blobs = {}

    @classmethod
    def _get_ans_staging(cls):
        if cls._ans_staging is None:
            cls._ans_staging = (
                torch.empty(16 * 1024 * 1024, dtype=torch.uint8, device="cpu", pin_memory=True),
                torch.empty(24 * 1024 * 1024, dtype=torch.uint8, device="cpu", pin_memory=True),
            )
        return cls._ans_staging

    @classmethod
    def _get_ans_scratch(cls, device, hidden, inter):
        key = str(device)
        if key not in cls._ans_scratch:
            dt = torch.bfloat16
            hb = inter * hidden  # bytes per full projection (bf16)
            hb2 = hb // 2        # bytes per lo/hi half
            cls._ans_scratch[key] = {
                "lo": [torch.empty((inter, hidden), dtype=torch.uint8, device=device),
                       torch.empty((inter, hidden), dtype=torch.uint8, device=device),
                       torch.empty((hidden, inter), dtype=torch.uint8, device=device)],
                "comp": torch.empty(16 * 1024 * 1024, dtype=torch.uint8, device=device),
                "hi": [torch.empty((inter, hidden), dtype=torch.uint8, device=device),
                       torch.empty((inter, hidden), dtype=torch.uint8, device=device),
                       torch.empty((hidden, inter), dtype=torch.uint8, device=device)],
            }
            nvcomp = _load_nvcomp()
            cls._ans_codec[key] = nvcomp.Codec(algorithm="ans") if nvcomp else None
        return cls._ans_scratch[key], cls._ans_codec.get(key)

    @classmethod
    def _get_ans_index(cls, store_dir):
        if cls._ans_index is None:
            with open(os.path.join(store_dir, "index.json")) as f:
                cls._ans_index = json.load(f)
        return cls._ans_index

    @classmethod
    def _get_ans_blobmap(cls, store_dir):
        if store_dir not in cls._ans_blobs:
            import mmap as _mmap
            maps = {}
            for fn in sorted(os.listdir(store_dir)):
                if fn.endswith(".ansh"):
                    path = os.path.join(store_dir, fn)
                    fh = open(path, "rb")
                    mm = _mmap.mmap(fh.fileno(), 0, access=_mmap.ACCESS_COPY)
                    maps[fn] = (fh, mm, torch.frombuffer(mm, dtype=torch.uint8))
            cls._ans_blobs[store_dir] = maps
        return cls._ans_blobs[store_dir]

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
        zssr_prefetch: bool = False,
        prefetch_topk: int = 8,
        coalesced_dma: bool = False,
        pinned_staging: bool = False,
        host_register: bool = False,
        model_dir: str = None,
        shard_register: bool = False,
        ans_store: str = None,
        col_measure: bool = False,
        col_topk_max: int = 512,
        col_pred_experts: int = 8,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.cfg = cfg
        self.device = device
        self.capacity = capacity
        self.handles = handles
        self.weight_map = weight_map
        self.dma_stream = dma_stream
        # Phase 1 (default OFF): ZSSR prefetch + coalesced DMA. Prediction moves
        # data only — the native router keeps the mathematical decision (exactness).
        self.zssr_enabled = bool(zssr_prefetch)
        self.prefetch_topk = int(prefetch_topk)
        self.coalesced = bool(coalesced_dma)
        self.pinned_staging = bool(pinned_staging)
        self.host_register = bool(host_register)
        self.model_dir = model_dir
        self.shard_register = bool(shard_register)
        self.ans_store = ans_store
        self.ans_bytes_m = 0
        self.ans_decomp_ms = 0.0
        self.ans_dispatches = 0
        self._ans_dec_pending = []
        # Phase 1A: column-prediction measurement (NO movement change; execution untouched)
        self.col_measure = bool(col_measure)
        self.col_topk_max = int(col_topk_max)
        self.col_pred_experts = int(col_pred_experts)
        self._h_pre = None            # stashed by layer-entrance hook (pre-attention hidden)
        self._col_w = {}              # expert_id -> (gate_cpu_f32, up_cpu_f32), small LRU
        self.col_pred_bytes = 0
        self.col_actual_bytes = 0
        self.col_hit_at = {}          # K -> intersection count
        self.col_pred_at = {}         # K -> predicted count
        self.col_actual_at = {}       # K -> actual count
        self._hostreg_ok = {}
        self._hostreg_failed = set()
        self.hostreg_pinned_gb = 0.0
        self.top_k = int(getattr(cfg, "num_experts_per_tok", 6))
        self.n_routed = int(getattr(cfg, "n_routed_experts", 160))

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


        # Metrics
        self.hits = 0
        self.misses = 0
        self.dma_bytes = 0
        # Phase 1 causal telemetry (standardized schema; demand = exposed by construction)
        self.zssr_predictions = 0
        self.zssr_correct = 0
        self.prefetch_bytes_total = 0
        self.prefetch_useful_bytes = 0
        self.demand_bytes_m = 0      # measured (event-drained) demand payload
        self.demand_dma_ms = 0.0     # measured (event-drained) demand transfer time
        self.prefetch_dma_ms = 0.0   # measured (event-drained) prefetch transfer time
        self._dma_pending = []       # (kind, ev_start, ev_end, bytes)
        self._prefetched = {}        # expert_id -> prefetched bytes (unverified)

    def warm_up_slots(self, initial_experts: List[int]):
        """Pre-populates dynamic slots with initial experts."""
        for slot_idx, exp_id in enumerate(initial_experts[:self.capacity]):
            self._load_expert_to_slot(exp_id, slot_idx)

    def _dma_begin(self):
        ev_s = torch.cuda.Event(enable_timing=True)
        ev_s.record(self.dma_stream)
        return ev_s

    def _dma_end(self, ev_s, kind: str, nbytes: int):
        ev_e = torch.cuda.Event(enable_timing=True)
        ev_e.record(self.dma_stream)
        self._dma_pending.append((kind, ev_s, ev_e, int(nbytes)))

    def _drain_dma(self, sync: bool = False):
        """Non-blocking drain of completed DMA event pairs into measured ledgers."""
        if not self._dma_pending:
            return
        if self.device.type != "cuda":
            self._dma_pending.clear()
            return
        if sync:
            torch.cuda.synchronize(self.device)
        remaining = []
        for kind, ev_s, ev_e, nbytes in self._dma_pending:
            if sync or ev_e.query():
                ms = ev_s.elapsed_time(ev_e) / 1000.0
                if kind == "demand":
                    self.demand_dma_ms += ms
                    self.demand_bytes_m += nbytes
                else:
                    self.prefetch_dma_ms += ms
                    self.prefetch_bytes_total += nbytes
            else:
                remaining.append((kind, ev_s, ev_e, nbytes))
        self._dma_pending = remaining

    def _ensure_host_registered(self, shard: str, key: str, t: torch.Tensor) -> bool:
        """Pin mmap-backed tensor pages once for zero-copy DMA. Returns True if direct DMA is safe."""
        if not self.host_register:
            return False
        tag = (shard, key)
        if tag in self._hostreg_ok:
            return True
        if tag in self._hostreg_failed:
            return False
        try:
            import ctypes
            cu = self._get_cudart()
            if cu is None:
                self._hostreg_failed.add(tag)
                return False
            cu.cudaHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
            err = cu.cudaHostRegister(ctypes.c_void_p(t.data_ptr()), ctypes.c_size_t(t.nbytes), ctypes.c_uint(0))
            if int(err) == 0:
                self._hostreg_ok[tag] = True
                self.hostreg_pinned_gb += t.nbytes / (1024 ** 3)
                return True
            self._hostreg_failed.add(tag)
        except Exception:
            self._hostreg_failed.add(tag)
        return False

    @classmethod
    def register_shard(cls, model_dir: str, shard: str) -> float:
        """Register one whole shard mapping once. Returns seconds spent (0 if already done)."""
        if shard in cls._shard_registered or model_dir is None:
            return 0.0
        t0 = time.perf_counter()
        sm = ShardMap.get(os.path.join(model_dir, shard))
        cu = cls._get_cudart()
        if cu is not None:
            import ctypes
            cu.cudaHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
            err = cu.cudaHostRegister(ctypes.c_void_p(sm.u8.data_ptr()),
                                      ctypes.c_size_t(sm.u8.nbytes), ctypes.c_uint(0))
            if int(err) != 0:
                return 0.0
        dt = time.perf_counter() - t0
        cls._shard_registered.add(shard)
        cls._shard_reg_seconds += dt
        cls._shard_reg_gb += sm.u8.nbytes / (1024 ** 3)
        return dt

    def _shard_view(self, shard: str, key: str):
        """Zero-copy BF16 view from registered shard mapping, or None (fallback)."""
        if not self.shard_register or self.model_dir is None:
            return None
        if shard not in self._shard_registered:
            return None
        try:
            return ShardMap.get(os.path.join(self.model_dir, shard)).view_tensor(key)
        except Exception:
            return None

    def _ans_codec_for(self, device):
        if str(device) not in self._ans_codec or self._ans_codec[str(device)] is None:
            nvcomp = _load_nvcomp()
            if nvcomp is None:
                return None
            self._ans_scratch[str(device)] = True  # marker; scratch alloc'd per call
            self._ans_codec[str(device)] = nvcomp.Codec(algorithm="ans")
        return self._ans_codec[str(device)]

    def _ans_stage_expert(self, expert_id: int):
        """DMA lo-strided + comp blob for one expert. Returns stage dict (DMAs async)."""
        idx = self._get_ans_index(self.ans_store)
        blobs = self._get_ans_blobmap(self.ans_store)
        pfx = f"model.layers.{self.layer_idx}.mlp.experts.{expert_id}"
        stage = {"expert": expert_id, "lo": [], "comp": [], "hi_len": [], "nbytes": 0, "shapes": []}
        for proj, pname in (("gate", "gate_proj"), ("up", "up_proj"), ("down", "down_proj")):
            key = f"{pfx}.{pname}.weight"
            meta = idx[key]
            # lo half: raw byte span (arbitrary file offset) -> even bytes. No dtype
            # view: safetensors data offsets are not 2-byte aligned in general.
            shard = self.weight_map[key]
            sm = ShardMap.get(os.path.join(self.model_dir, shard))
            info = sm.header[key]
            b, e = info["data_offsets"]
            s = sm.data_start + b
            span = sm.u8[s:s + (e - b)]
            I0, H0 = info["shape"][0], info["shape"][1]
            # strided view DMA'd directly (async); .contiguous() would be a sync CPU gather.
            lo = span[0::2].reshape(I0, H0)
            lo_g = lo.to(self.device, non_blocking=True)
            # comp blob slice from blob-file mapping (contiguous)
            _fh, _mm, bu8 = blobs[meta["blob"]]
            cb = bu8[meta["offset"]:meta["offset"] + meta["comp_len"]]
            comp_g = cb.to(self.device, non_blocking=True)
            stage["lo"].append(lo_g)
            stage["comp"].append((comp_g, meta["hi_len"]))
            stage["hi_len"].append(meta["hi_len"])
            stage["nbytes"] += lo_g.nbytes + comp_g.nbytes
        return stage

    def _ans_decode_batch(self, stages):
        """ONE nvcomp batched ANS decode for a layer's staged misses. Returns list of hi blobs."""
        nvcomp = _load_nvcomp()
        codec = self._ans_codec_for(self.device)
        arrs = []
        for st in stages:
            for comp_g, _hi_len in st["comp"]:
                arrs.append(nvcomp.as_array(comp_g))
        return codec.decode(arrs)

    def ans_self_check(self) -> bool:
        """Reconstruct layer expert-0 via ANS path; torch.equal vs safe_open. Structural gate."""
        import torch.utils.dlpack as dlpack
        pfx = f"model.layers.{self.layer_idx}.mlp.experts.0"
        st = self._ans_stage_expert(0)
        torch.cuda.synchronize(self.device)
        dec = self._ans_decode_batch([st])
        his = []
        for d in dec:
            his.append(bytes(d) if isinstance(d, (bytes, bytearray, memoryview))
                       else dlpack.from_dlpack(d).cpu().numpy().tobytes())
        # Per-projection interleave + compare (exact byte reconstruction)
        ok = True
        for pi, pname in enumerate(("gate_proj", "up_proj", "down_proj")):
            key = f"{pfx}.{pname}.weight"
            ref = self.handles[self.weight_map[key]].get_tensor(key)
            sm = ShardMap.get(os.path.join(self.model_dir, self.weight_map[key]))
            info = sm.header[key]
            b, e = info["data_offsets"]
            span = sm.u8[sm.data_start + b:sm.data_start + e]
            I0, H0 = info["shape"][0], info["shape"][1]
            lo = span[0::2].reshape(I0, H0).contiguous()
            hi_t = torch.frombuffer(bytearray(his[pi]), dtype=torch.uint8).reshape(I0, H0)
            rec = torch.empty(I0, H0 * 2, dtype=torch.uint8)
            rec[:, 0::2] = lo
            rec[:, 1::2] = hi_t
            ok = ok and torch.equal(rec.view(torch.bfloat16).reshape(ref.shape).cpu(), ref.cpu())
        return bool(ok)

    def _ans_interleave(self, slot_idx: int, lo_list, hi_list):
        """Assemble exact BF16 expert weights in slot from lo/hi uint8 halves (GPU D2D)."""
        slot = self.slots[slot_idx]
        projs = (slot.gate_proj.weight, slot.up_proj.weight, slot.down_proj.weight)
        for w, lo, hi in zip(projs, lo_list, hi_list):
            w8 = w.view(torch.uint8).reshape(w.shape[0], -1)
            w8[:, 0::2].copy_(lo, non_blocking=True)
            w8[:, 1::2].copy_(hi, non_blocking=True)

    def _drain_ans(self, sync: bool = False):
        """Collect completed ANS-decode timings without blocking (unless sync)."""
        if not self._ans_dec_pending:
            return
        if sync and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        rest = []
        for ev0, ev1 in self._ans_dec_pending:
            if sync or (self.device.type == "cuda" and ev1.query()):
                self.ans_decomp_ms += ev0.elapsed_time(ev1) / 1000.0
            else:
                rest.append((ev0, ev1))
        self._ans_dec_pending = rest

    def _load_experts_ans(self, exp_ids):
        """Batched ANS miss path: stage all, ONE nvcomp decode, interleave all. Exact.

        Zero CPU roundtrips: hi stays on GPU via DLPack views; lo DMA'd strided
        (async); decode timed with non-blocking events drained next forward.
        """
        import torch.utils.dlpack as dlpack
        self._drain_ans()
        staged, slots = [], []
        for exp_id in exp_ids:
            self.misses += 1
            slot_idx = self.slot_lru.pop(0)
            staged.append(self._ans_stage_expert(exp_id))
            slots.append(slot_idx)
            self.ans_bytes_m += staged[-1]["nbytes"]
        if not staged:
            return
        self.dma_bytes += sum(s["nbytes"] for s in staged)
        if self.device.type == "cuda":
            ev0 = torch.cuda.Event(enable_timing=True)
            ev1 = torch.cuda.Event(enable_timing=True)
            ev0.record(torch.cuda.current_stream(self.device))
        dec = self._ans_decode_batch(staged)
        if self.device.type == "cuda":
            ev1.record(torch.cuda.current_stream(self.device))
            self._ans_dec_pending.append((ev0, ev1))
        self.ans_dispatches += len(staged)
        flat = []
        for d in dec:
            if isinstance(d, (bytes, bytearray, memoryview)):
                flat.append(d)
            else:
                flat.append(dlpack.from_dlpack(d))  # GPU tensor, zero copy
        pos = 0
        for exp_id, slot_idx, st in zip(exp_ids, slots, staged):
            his = []
            for pi, hi_len in enumerate(st["hi_len"]):
                hb = flat[pos]
                pos += 1
                if isinstance(hb, (bytes, bytearray)):
                    assert len(hb) == hi_len, (len(hb), hi_len)
                    I, H = st["lo"][pi].shape[0], st["lo"][pi].shape[1]
                    his.append(torch.frombuffer(bytearray(hb), dtype=torch.uint8).reshape(I, H).to(self.device))
                else:
                    I, H = st["lo"][pi].shape[0], st["lo"][pi].shape[1]
                    assert hb.numel() == hi_len, (hb.numel(), hi_len)
                    his.append(hb.reshape(I, H))
            self._ans_interleave(slot_idx, st["lo"], his)
            if slot_idx in self.slot_to_expert:
                self.expert_to_slot.pop(self.slot_to_expert[slot_idx], None)
            self.slot_to_expert[slot_idx] = exp_id
            self.expert_to_slot[exp_id] = slot_idx
            self.slot_lru.append(slot_idx)

    def _load_expert_to_slot(self, expert_id: int, slot_idx: int, kind: str = "demand",
                             record: bool = True):
        slot_mod = self.slots[slot_idx]
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
        nbytes = int(t_gate.nbytes + t_up.nbytes + t_down.nbytes)

        # Phase 1c-R: prefer registered whole-shard views (zero-copy, link-rate).
        v_gate = self._shard_view(shard_gate, k_gate)
        v_up = self._shard_view(shard_up, k_up)
        v_down = self._shard_view(shard_down, k_down)
        if v_gate is not None and v_up is not None and v_down is not None:
            t_gate, t_up, t_down = v_gate, v_up, v_down

        ev_s = self._dma_begin() if (record and self.device.type == "cuda") else None
        with torch.no_grad():
            if self.pinned_staging and self.device.type == "cuda" and not self.host_register:
                # Phase 1b (killed default): mmap -> pinned staging, then link-rate DMA.
                st_g, st_u, st_d = self._get_pinned_staging(
                    t_gate.shape[0], t_gate.shape[1], t_gate.dtype)
                st_g.copy_(t_gate)
                st_u.copy_(t_up)
                st_d.copy_(t_down)
                src_g, src_u, src_d = st_g, st_u, st_d
            else:
                # Phase 1c: zero-copy direct DMA. Fast iff pages are HostRegistered
                # (done below); otherwise driver-staged ~6GB/s.
                if self.host_register and self.device.type == "cuda":
                    self._ensure_host_registered(shard_gate, k_gate, t_gate)
                    self._ensure_host_registered(shard_up, k_up, t_up)
                    self._ensure_host_registered(shard_down, k_down, t_down)
                src_g, src_u, src_d = t_gate, t_up, t_down
            with torch.cuda.stream(self.dma_stream):
                slot_mod.gate_proj.weight.copy_(src_g, non_blocking=True)
                slot_mod.up_proj.weight.copy_(src_u, non_blocking=True)
                slot_mod.down_proj.weight.copy_(src_d, non_blocking=True)

        if slot_idx in self.slot_to_expert:
            old_exp = self.slot_to_expert[slot_idx]
            self.expert_to_slot.pop(old_exp, None)

        self.slot_to_expert[slot_idx] = expert_id
        self.expert_to_slot[expert_id] = slot_idx
        self.dma_bytes += nbytes
        if ev_s is not None:
            self._dma_end(ev_s, kind, nbytes)
        return nbytes

    def _col_weights(self, expert_id: int):
        """Lazy CPU fp32 gate/up for column-energy scoring (cap 16/layer, ~500MB max)."""
        if expert_id in self._col_w:
            return self._col_w[expert_id]
        pfx = f"model.layers.{self.layer_idx}.mlp.experts.{expert_id}"
        g = self.handles[self.weight_map[f"{pfx}.gate_proj.weight"]].get_tensor(f"{pfx}.gate_proj.weight")
        u = self.handles[self.weight_map[f"{pfx}.up_proj.weight"]].get_tensor(f"{pfx}.up_proj.weight")
        out = (g.float(), u.float())
        if len(self._col_w) >= 16:
            self._col_w.pop(next(iter(self._col_w)))
        self._col_w[expert_id] = out
        del g, u
        return out

    @staticmethod
    def _col_energy_topk(h_vec, gate_w, up_w, k: int):
        import torch.nn.functional as F
        with torch.no_grad():
            e = (F.silu(h_vec @ gate_w.T) * (h_vec @ up_w.T)).pow(2).squeeze(0)
            return torch.topk(e, k=min(k, e.numel())).indices.tolist()

    @torch.no_grad()
    def col_measure_step(self, h_post, topk_actual):
        """Phase 1A: predicted (h_pre, energy-ranked) vs actual (h_post, energy-ranked) columns.

        No movement, no slot changes. Ground truth = top-K energy columns using the
        TRUE post-attention hidden state for each NATIVELY routed expert.
        """
        import torch.nn.functional as F
        if self._h_pre is None:
            return
        KMAX = self.col_topk_max
        KS = (128, 256, 512)
        h_pre = self._h_pre.detach().float().reshape(-1, self.gate.weight.shape[1])[-1:]
        h_true = h_post.detach().float().reshape(-1, self.gate.weight.shape[1])[-1:]
        col_bytes = 3 * int(getattr(self.cfg, "hidden_size", 5120)) * 2  # per column (BF16)
        # Expert probe from pre-attention stream
        logits = (h_pre @ self.gate.weight.detach().float().t()).squeeze(0)
        pred_experts = torch.topk(logits, k=min(self.col_pred_experts, logits.numel())).indices.tolist()
        actual_experts = topk_actual.unique().tolist() if hasattr(topk_actual, "unique") else list(topk_actual)
        for exp_id in pred_experts:
            g, u = self._col_weights(exp_id)
            pred_cols = self._col_energy_topk(h_pre, g, u, KMAX)
            self.col_pred_bytes += len(pred_cols) * col_bytes
            for K in KS:
                if K > len(pred_cols):
                    continue
                ps = set(pred_cols[:K])
                self.col_pred_at[K] = self.col_pred_at.get(K, 0) + len(ps)
                if exp_id in actual_experts:
                    ga, ua = self._col_weights(exp_id)
                    act_cols = set(self._col_energy_topk(h_true, ga, ua, KMAX)[:K])
                    self.col_actual_at[K] = self.col_actual_at.get(K, 0) + len(act_cols)
                    self.col_hit_at[K] = self.col_hit_at.get(K, 0) + len(ps & act_cols)
        for exp_id in actual_experts:
            if exp_id not in pred_experts:
                ga, ua = self._col_weights(exp_id)
                act_cols = self._col_energy_topk(h_true, ga, ua, KMAX)
                for K in KS:
                    if K <= len(act_cols):
                        self.col_actual_at[K] = self.col_actual_at.get(K, 0) + K

    @torch.no_grad()
    def zssr_prefetch(self, hidden_in: torch.Tensor):
        """Phase 1 probe (called at layer entrance, pre-attention).

        Predicts top-(K+margin) experts from the residual stream and issues one
        grouped async DMA per layer on dma_stream. Prediction moves DATA only;
        the native router in forward() keeps the mathematical decision (exact).
        Returns predicted expert list (possibly empty when disabled).
        """
        if not self.zssr_enabled:
            return []
        try:
            h = hidden_in[:, -1, :].to(dtype=torch.float32, device=self.gate.weight.device)
            scores = torch.softmax(h @ self.gate.weight.detach().float().t(), dim=-1)
            k = min(self.prefetch_topk, self.n_routed)
            pred = torch.topk(scores, k=k, dim=-1).indices.view(-1).tolist()
        except Exception:
            return []
        new_ids = [e for e in dict.fromkeys(pred) if e not in self.expert_to_slot and e not in self._prefetched]
        if not new_ids:
            return pred
        if self.coalesced:
            ev_s = self._dma_begin() if self.device.type == "cuda" else None
        for exp_id in new_ids:
            slot_idx = self.slot_lru.pop(0)
            nbytes = self._load_expert_to_slot(exp_id, slot_idx, kind="prefetch",
                                               record=not self.coalesced)
            self.slot_lru.append(slot_idx)
            self._prefetched[exp_id] = self._prefetched.get(exp_id, 0) + nbytes
        if self.coalesced and self.device.type == "cuda":
            self._dma_end(ev_s, "prefetch", sum(self._prefetched.get(e, 0) for e in new_ids))
        self.zssr_predictions += len(new_ids)
        return pred

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        identity = hidden_states
        orig_shape = hidden_states.shape
        self._drain_dma()  # collect completed transfers only; never blocks

        # 1. Gate routing (returns topk_idx, topk_weight, aux_loss)
        topk_indices, topk_weights, _ = self.gate(hidden_states)
        needed_experts = topk_indices.unique().tolist()

        # Phase 1A: column P/R measurement (no movement, no slot changes)
        if self.col_measure:
            try:
                self.col_measure_step(hidden_states, topk_indices)
            except Exception as e:
                if not getattr(self, "_col_warn_done", False):
                    self._col_warn_done = True
                    print(f"  [warn] col_measure L{self.layer_idx}: {type(e).__name__}: {e}")

        # Phase 1 verify: reconcile ZSSR prediction against ground-truth routing.
        if self._prefetched:
            actual = set(needed_experts)
            for exp_id in list(self._prefetched.keys()):
                if exp_id in actual:
                    self.zssr_correct += 1
                    self.prefetch_useful_bytes += self._prefetched.pop(exp_id)

        # 2. Compute shared experts (permanently resident)
        shared_out = self.shared_experts(identity)

        # 3. Dynamic Slot Management & Expert Streaming
        # Fast path for single-token decode (needed_experts <= capacity): batch-stream all missing experts
        if len(needed_experts) <= self.capacity:
            # First, update LRU for resident hits to protect them from eviction!
            for exp_id in needed_experts:
                if exp_id in self.expert_to_slot:
                    self.hits += 1
                    slot_idx = self.expert_to_slot[exp_id]
                    self.slot_lru.remove(slot_idx)
                    self.slot_lru.append(slot_idx)

            missing_experts = [e for e in needed_experts if e not in self.expert_to_slot]
            if missing_experts:
                if self.ans_store:
                    self._load_experts_ans(missing_experts)
                else:
                    for exp_id in missing_experts:
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

            if i in self.expert_to_slot:
                slot_idx = self.expert_to_slot[i]
                if len(needed_experts) > self.capacity:
                    self.hits += 1
                    self.slot_lru.remove(slot_idx)
                    self.slot_lru.append(slot_idx)
            elif self.ans_store:
                # On-demand ANS load for multi-token prefill where needed > capacity
                self._load_experts_ans([i])
                slot_idx = self.expert_to_slot[i]
            else:
                # On-demand load for multi-token prefill where needed_experts > capacity
                self.misses += 1
                slot_idx = self.slot_lru.pop(0)
                self._load_expert_to_slot(i, slot_idx)
                self.slot_lru.append(slot_idx)
                torch.cuda.current_stream(self.device).wait_stream(self.dma_stream)

            expert = self.slots[slot_idx]
            tokens_for_this = sorted_tokens[start_idx:end_idx]
            outputs.append(expert(tokens_for_this))
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
            zssr_prefetch=args.zssr_prefetch,
            prefetch_topk=args.prefetch_topk,
            coalesced_dma=args.coalesced_dma,
            pinned_staging=args.pinned_staging,
            host_register=args.host_register,
            model_dir=MODEL_PATH,
            shard_register=args.shard_register,
            ans_store=args.ans_store,
            col_measure=args.col_measure,
            col_topk_max=args.col_topk_max,
            col_pred_experts=args.col_pred_experts,
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

    if args.shard_register:
        # Phase 1c-R warmup (UNTIMED deployment cost): map + register every shard
        # holding expert tensors exactly once. Steady-state decode then pays ~0
        # registration. Reports seconds + GB pinned; decode metrics exclude this.
        print(f"\n[6.5] Shard-level HostRegister warmup (one-time, untimed)...")
        expert_shards = sorted({s for k, s in weight_map.items() if ".mlp.experts." in k})
        t_w0 = time.perf_counter()
        for sh in expert_shards:
            DeepSeekColossusMoEWrapper.register_shard(MODEL_PATH, sh)
        dt_w = time.perf_counter() - t_w0
        print(f"  Registered {len(DeepSeekColossusMoEWrapper._shard_registered)} shards "
              f"({DeepSeekColossusMoEWrapper._shard_reg_gb:.1f} GB) in {dt_w:.1f}s "
              f"({dt_w / max(1, len(DeepSeekColossusMoEWrapper._shard_registered)):.2f}s/shard)")
        sys.stdout.flush()

    if args.zssr_prefetch or args.col_measure:
        # Layer-entrance hooks: ALWAYS stash pre-attention hidden (1A measurement);
        # prefetch DMA only when --zssr-prefetch. Registered AFTER P2P hooks.
        print(f"  Installing layer-entrance hooks (stash h_pre"
              f"{f' + ZSSR prefetch top-{args.prefetch_topk}' if args.zssr_prefetch else ''}"
              f"{' + column-P/R measurement' if args.col_measure else ''}) on layers 1..59...")
        def _make_zssr_hook(wrapper):
            def _hook(module, hook_args):
                try:
                    hidden_in = hook_args[0]
                    if isinstance(hidden_in, torch.Tensor):
                        wrapper._h_pre = hidden_in.detach()
                        if wrapper.zssr_enabled:
                            wrapper.zssr_prefetch(hidden_in)
                except Exception as e:
                    print(f"  [warn] entrance hook L{wrapper.layer_idx}: {e}")
                return None
            return _hook
        for l_idx in range(1, cfg.num_hidden_layers):
            model.model.layers[l_idx].register_forward_pre_hook(
                _make_zssr_hook(colossus_wrappers[l_idx - 1]))

    if args.ans_store:
        # Phase 2 structural gate: reconstruct layer-1 expert-0 via ANS path.
        print(f"  ANS self-check (layer 1 expert 0 vs safe_open, torch.equal)...")
        ok = colossus_wrappers[0].ans_self_check()
        print(f"  ANS self-check: {'PASS' if ok else 'FAIL'}")
        assert ok, "ANS reconstruction mismatch — aborting before timed run."
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
        w.dma_bytes = 0
        w.zssr_predictions = 0
        w.zssr_correct = 0
        w.prefetch_bytes_total = 0
        w.prefetch_useful_bytes = 0
        w.demand_bytes_m = 0
        w.demand_dma_ms = 0.0
        w.prefetch_dma_ms = 0.0
        w._dma_pending = []
        w._prefetched = {}
        w.ans_bytes_m = 0
        w.ans_decomp_ms = 0.0
        w.ans_dispatches = 0
        w._ans_dec_pending = []
        w._h_pre = None
        w._col_w = {}
        w.col_pred_bytes = 0
        w.col_actual_bytes = 0
        w.col_hit_at = {}
        w.col_pred_at = {}
        w.col_actual_at = {}

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
    last_dma = sum(w.dma_bytes for w in colossus_wrappers)
    last_zp = sum(w.zssr_predictions for w in colossus_wrappers)
    last_zc = sum(w.zssr_correct for w in colossus_wrappers)
    last_pf = sum(w.prefetch_bytes_total for w in colossus_wrappers)
    last_pfu = sum(w.prefetch_useful_bytes for w in colossus_wrappers)
    last_dms = sum(w.demand_dma_ms for w in colossus_wrappers)
    last_dby = sum(w.demand_bytes_m for w in colossus_wrappers)

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

        # Per-step cache metrics (Phase 1 causal set: DMA bytes vs exposed DMA)
        cur_hits = sum(w.hits for w in colossus_wrappers)
        cur_misses = sum(w.misses for w in colossus_wrappers)
        cur_dma = sum(w.dma_bytes for w in colossus_wrappers)
        cur_zp = sum(w.zssr_predictions for w in colossus_wrappers)
        cur_zc = sum(w.zssr_correct for w in colossus_wrappers)
        cur_pf = sum(w.prefetch_bytes_total for w in colossus_wrappers)
        cur_pfu = sum(w.prefetch_useful_bytes for w in colossus_wrappers)
        cur_dms = sum(w.demand_dma_ms for w in colossus_wrappers)
        cur_dby = sum(w.demand_bytes_m for w in colossus_wrappers)

        step_hits = cur_hits - last_hits
        step_misses = cur_misses - last_misses
        step_lookups = step_hits + step_misses
        step_hit_rate = (step_hits / step_lookups * 100.0) if step_lookups > 0 else 0.0
        step_dma_mb = (cur_dma - last_dma) / (1024**2)
        step_zp = cur_zp - last_zp
        step_zc = cur_zc - last_zc
        step_recall = (step_zc / step_zp * 100.0) if step_zp else 0.0
        step_pfu_mb = (cur_pfu - last_pfu) / (1024**2)
        step_dby_mb = (cur_dby - last_dby) / (1024**2)
        step_dms = (cur_dms - last_dms) * 1000.0
        step_exposed = step_dby_mb  # demand loads issue post-verify: exposed by construction
        step_useful = step_pfu_mb   # prefetched bytes actually consumed: overlapped by construction
        step_overlap = (step_useful / (step_useful + step_exposed) * 100.0) if (step_useful + step_exposed) else 0.0

        last_hits, last_misses, last_dma = cur_hits, cur_misses, cur_dma
        last_zp, last_zc, last_pf, last_pfu, last_dms, last_dby = cur_zp, cur_zc, cur_pf, cur_pfu, cur_dms, cur_dby

        generated_ids = torch.cat([generated_ids, next_token], dim=1)
        tok_str = tokenizer.decode(next_token[0], skip_special_tokens=False)
        print(f"    Token {step+1:2d}/{args.max_new_tokens-1:2d} | Latency: {t_step*1000:6.1f} ms | Hits: {step_hits:3d}/{step_lookups:3d} ({step_hit_rate:5.1f}%) | Misses: {step_misses:2d} | DMA: {step_dma_mb:5.1f} MB (exposed {step_exposed:5.1f}) | ZSSR: {step_zc:3d}/{step_zp:3d} ({step_recall:4.1f}%) overlapped {step_useful:5.1f}MB ({step_overlap:4.1f}%) | Tok: {repr(tok_str)}")
        sys.stdout.flush()

        decode_step_details.append({
            "step": step + 1,
            "latency_ms": t_step * 1000,
            "step_hits": step_hits,
            "step_misses": step_misses,
            "step_hit_rate_pct": step_hit_rate,
            "step_dma_mb": step_dma_mb,
            "step_exposed_dma_mb": step_exposed,
            "step_zssr_predictions": step_zp,
            "step_zssr_correct": step_zc,
            "step_zssr_recall_pct": step_recall,
            "step_overlapped_mb": step_useful,
            "step_overlap_pct": step_overlap,
            "step_demand_dma_ms": step_dms,
            "token": tok_str,
        })




    # ─────────────────────────────────────────────────────────────────────────
    # Summary Metrics
    # ─────────────────────────────────────────────────────────────────────────
    total_decode_time = sum(decode_latencies)
    avg_decode_lat = total_decode_time / len(decode_latencies) if decode_latencies else 0.0
    decode_tps = 1.0 / avg_decode_lat if avg_decode_lat > 0 else 0.0

    # Final sync + drain so event-measured ledgers are complete
    torch.cuda.synchronize(dev0)
    if args.num_gpus > 1:
        torch.cuda.synchronize(dev1)
    for w in colossus_wrappers:
        w._drain_dma(sync=True)
        if hasattr(w, "_drain_ans"):
            w._drain_ans(sync=True)

    total_hits = sum(w.hits for w in colossus_wrappers)
    total_misses = sum(w.misses for w in colossus_wrappers)
    total_lookups = total_hits + total_misses
    hit_rate = (total_hits / total_lookups * 100.0) if total_lookups > 0 else 0.0
    total_dma_mb = sum(w.dma_bytes for w in colossus_wrappers) / (1024**2)
    total_ans_mb = sum(w.ans_bytes_m for w in colossus_wrappers) / (1024**2)
    total_ans_n = sum(w.ans_dispatches for w in colossus_wrappers)
    total_ans_dms = sum(w.ans_decomp_ms for w in colossus_wrappers) * 1000.0
    ans_ratio = (total_ans_mb / total_dma_mb) if total_dma_mb else 0.0
    # Phase 1A column P/R aggregation (micro-averaged over all layer-steps)
    col_pr = {}
    for K in (128, 256, 512):
        ph = sum(w.col_hit_at.get(K, 0) for w in colossus_wrappers)
        pp = sum(w.col_pred_at.get(K, 0) for w in colossus_wrappers)
        pa = sum(w.col_actual_at.get(K, 0) for w in colossus_wrappers)
        col_pr[K] = {"precision": (ph / pp if pp else 0.0),
                     "recall": (ph / pa if pa else 0.0),
                     "pred_cols": pp, "actual_cols": pa}
    col_pred_mb = sum(w.col_pred_bytes for w in colossus_wrappers) / (1024**2)
    # Phase 1 causal set
    total_zp = sum(w.zssr_predictions for w in colossus_wrappers)
    total_zc = sum(w.zssr_correct for w in colossus_wrappers)
    total_pf = sum(w.prefetch_bytes_total for w in colossus_wrappers) / (1024**2)
    total_pfu = sum(w.prefetch_useful_bytes for w in colossus_wrappers) / (1024**2)
    total_dby = sum(w.demand_bytes_m for w in colossus_wrappers) / (1024**2)
    total_dms = sum(w.demand_dma_ms for w in colossus_wrappers) * 1000.0
    total_pms = sum(w.prefetch_dma_ms for w in colossus_wrappers) * 1000.0
    total_wasted = total_pf - total_pfu
    recall = (total_zc / total_zp * 100.0) if total_zp else 0.0
    overlap = (total_pfu / (total_pfu + total_dby) * 100.0) if (total_pfu + total_dby) else 0.0
    eff_gbps = (sum(w.demand_bytes_m for w in colossus_wrappers) / (1024**3)) / (sum(w.demand_dma_ms for w in colossus_wrappers) + 1e-12)
    ref_text = "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n"

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
    print("-" * 80)
    print(f"  Prefill Latency           : {t_prefill*1000:7.1f} ms ({prefill_tps:5.1f} tok/s for {prompt_len} tokens)")
    print(f"  Decode Latency (Avg)      : {avg_decode_lat*1000:7.1f} ms / token")
    print(f"  Decode Throughput         : {decode_tps:7.2f} tokens / sec")
    print(f"  Cache Hits                : {total_hits:,} ({hit_rate:.1f}%)")
    print(f"  Cache Misses (Cold DMA)   : {total_misses:,}")
    print(f"  Total PCIe DMA Transferred: {total_dma_mb:7.1f} MB")
    if args.ans_store:
        print(f"  ANS Wire Compression      : {total_ans_n:,} dispatches, {total_ans_mb:,.1f}MB "
              f"(ratio {ans_ratio:.3f} vs 45MB whole), decomp {total_ans_dms:.2f}ms")
    if args.col_measure:
        print(f"  Column P/R (pred h_pre vs actual h_post):")
        for K in (128, 256, 512):
            m = col_pr[K]
            print(f"    K={K:3d}: P={m['precision'] * 100:5.1f}% R={m['recall'] * 100:5.1f}% "
                  f"(pred {m['pred_cols']:,} actual {m['actual_cols']:,} cols; {col_pred_mb:,.1f}MB predicted)")
    print(f"  ZSSR Prefetch             : {'on (top-%d%s)' % (args.prefetch_topk, ', coalesced' if args.coalesced_dma else '') if args.zssr_prefetch else 'off'} | "
          f"pred {total_zp:,} correct {total_zc:,} (recall {recall:.1f}%) | prefetch {total_pf:,.1f}MB useful {total_pfu:,.1f}MB wasted {total_wasted:,.1f}MB")
    print(f"  DMA Causality             : demand {total_dby:,.1f}MB in {total_dms:.2f}ms (eff {eff_gbps:.1f} GB/s, EXPOSED) | "
          f"prefetch {total_pms:.2f}ms (OVERLAPPED) | overlap {overlap:.1f}%")
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
        "hit_rate_pct": hit_rate,
        "total_dma_mb": total_dma_mb,
        # Phase 1 standardized causal telemetry
        "zssr_prefetch": bool(args.zssr_prefetch),
        "prefetch_topk": int(args.prefetch_topk),
        "coalesced_dma": bool(args.coalesced_dma),
        "pinned_staging": bool(args.pinned_staging),
        "host_register": bool(args.host_register),
        "hostreg_pinned_gb": sum(w.hostreg_pinned_gb for w in colossus_wrappers),
        "shard_register": bool(args.shard_register),
        "shard_reg_shards": len(DeepSeekColossusMoEWrapper._shard_registered),
        "shard_reg_gb": DeepSeekColossusMoEWrapper._shard_reg_gb,
        "shard_reg_seconds": DeepSeekColossusMoEWrapper._shard_reg_seconds,
        "ans_store": args.ans_store,
        "ans_dispatches": total_ans_n,
        "ans_mb": total_ans_mb,
        "ans_ratio": ans_ratio,
        "ans_decomp_ms": total_ans_dms,
        "col_measure": bool(args.col_measure),
        "col_pr": {str(K): col_pr[K] for K in col_pr},
        "col_pred_mb": col_pred_mb,
        "zssr_predictions": total_zp,
        "zssr_correct": total_zc,
        "zssr_recall_pct": recall,
        "prefetch_mb": total_pf,
        "useful_prefetch_mb": total_pfu,
        "wasted_prefetch_mb": total_wasted,
        "demand_mb": total_dby,
        "demand_dma_ms": total_dms,
        "prefetch_dma_ms": total_pms,
        "overlap_pct": overlap,
        "effective_gbps": eff_gbps,
        "exact_vs_baseline": (gen_text == ref_text) if (prompt == "def quicksort(arr):" and args.max_new_tokens == 16) else None,
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
    # Phase 1 (all default OFF for clean ablations: baseline vs +prefetch vs +coalescing)
    parser.add_argument("--zssr-prefetch", dest="zssr_prefetch", action="store_true", default=False,
                        help="Pre-attention ZSSR probe + grouped async DMA (prediction moves data only)")
    parser.add_argument("--prefetch_topk", type=int, default=8,
                        help="Top-K experts to prefetch per layer entrance (K=6 routing + margin)")
    parser.add_argument("--coalesced-dma", dest="coalesced_dma", action="store_true", default=False,
                        help="Single timing group per layer prefetch burst (few large DMAs)")
    parser.add_argument("--pinned_staging", action="store_true", default=False,
                        help="Phase 1b (killed default): stage mmap weights through shared pinned host buffer")
    parser.add_argument("--no_pinned_staging", dest="pinned_staging", action="store_false",
                        help="Opt out: DMA directly from mmap handles (~6GB/s staged)")
    parser.add_argument("--host_register", action="store_true", default=False,
                        help="Phase 1c: cudaHostRegister mmap tensors once for zero-copy link-rate DMA")
    parser.add_argument("--shard_register", action="store_true", default=False,
                        help="Phase 1c-R: one-time whole-shard HostRegister warmup (untimed), then zero-copy DMA")
    parser.add_argument("--ans_store", type=str, default=None,
                        help="Phase 2: ANS hi-byte store dir (0.70 wire ratio, exact, GPU-decoded)")
    parser.add_argument("--col_measure", action="store_true", default=False,
                        help="Phase 1A: measure ZSSR column P/R vs ground truth (no movement change)")
    parser.add_argument("--col_topk_max", type=int, default=512)
    parser.add_argument("--col_pred_experts", type=int, default=8)


    parser.add_argument("--use_cache", action="store_true", default=True)
    parser.add_argument("--no_cache", dest="use_cache", action="store_false")
    parser.add_argument("--output_json", type=str, default="/home/palakm/MoEServingSim/aditya/deepseek_serving_colossus.json")
    args = parser.parse_args()


    serve_deepseek(args)
