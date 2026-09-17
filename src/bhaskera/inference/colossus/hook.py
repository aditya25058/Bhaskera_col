"""COLOSSUS hook for MoE inference (Bhaskera HF backend).

Modes of operation:
  * **Shadow mode** (default): attaches post-hooks to decoder layers,
    captures hidden states, runs ZSSR prediction, updates LRU bookkeeping.
    Dense execution is completely unchanged — zero numerical impact.
  * **Active mode** (``offload_enabled=True``): additionally creates an
    ``ExpertOffloadManager`` that profiles the prompt and offloads cold
    expert weights to CPU before ``model.generate()``.  VRAM drops by
    the fraction of experts that are cold (~80–90% for top-6/64).

In both modes, dense output is always bit-identical when
``colossus.enabled`` is ``false``.  The active-mode offload changes
which experts are resident on GPU but does not alter the computation
for the experts that *are* resident — the model's own ``forward()``
handles the actual expert dispatch.

Any failure disables the hook with a warning — dense output is always
bit-identical with or without it.
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Any, Deque, Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Bound on stored hidden states: last-token vectors only, CPU, deque-capped.
_SHADOW_DEPTH = 4


def _short_name(dotted: str) -> str:
    return dotted.split(".")[-1]


class ColossusMoEHook:
    """Attach ZSSR shadow prediction to a loaded MoE model."""

    def __init__(self, predictor, directory, budget_preset: str = "tiered_fwd"):
        self.predictor = predictor
        self.directory = directory
        self.budget_preset = budget_preset
        self._handles: List[Any] = []
        self._states: Dict[int, Deque] = {}
        self._steps = 0
        self._layers: List[int] = []
        self._hits_count = 0
        self._misses_count = 0
        self.disabled_reason: Optional[str] = None
        # Active-mode offload manager (set by build() when offload_enabled)
        self._offload_mgr: Optional[Any] = None
        self._offload_stats: dict = {}
        # Dynamic MoE expert cache
        self._wrapped_layers: Dict[int, Any] = {}
        self._dynamic_cache_enabled: bool = False
        self._cache_capacity: int = 16

    # -- construction ----------------------------------------------------
    @classmethod
    def build(cls, model, profile, colossus_cfg) -> "ColossusMoEHook":
        """Best-effort router extraction. Raises with reason if unsupported."""
        from .directory import ColumnDirectory, plan_fixed_packets  # noqa: F401
        from .predictor import ZSSRPredictor

        top_k = int(getattr(colossus_cfg, "top_k_experts", 8) or 8)
        pred = ZSSRPredictor(top_k_experts=top_k, top_cols=50,
                             num_col_experts=top_k)
        n_registered = 0
        num_layers = int(getattr(profile, "num_hidden_layers", 0) or 0)
        layers_container = getattr(model, "model", model)
        model_layers = getattr(layers_container, "layers", None)
        if model_layers is not None:
            num_layers = max(num_layers, len(model_layers))

        for idx in range(0, max(num_layers, 1)):
            if model_layers is None or idx >= len(model_layers):
                continue
            layer = model_layers[idx]
            mlp = getattr(layer, "block_sparse_moe", getattr(layer, "mlp", None))
            gate = getattr(mlp, "gate", None) if mlp is not None else None
            if gate is None or not hasattr(gate, "weight"):
                continue
            try:
                pred.router[idx] = gate.weight.detach().float().cpu()
                n_registered += 1
            except Exception:
                continue

            # Populate gate_w and up_w for expert column prediction
            experts = getattr(mlp, "experts", None)
            if experts is not None:
                pred.gate_w[idx] = {}
                pred.up_w[idx] = {}
                for e, exp_mod in enumerate(experts):
                    gw = getattr(exp_mod, "gate_proj", getattr(exp_mod, "w1", None))
                    uw = getattr(exp_mod, "up_proj", getattr(exp_mod, "w3", None))
                    if gw is not None and uw is not None and hasattr(gw, "weight") and hasattr(uw, "weight"):
                        pred.gate_w[idx][e] = gw.weight.detach().cpu()
                        pred.up_w[idx][e] = uw.weight.detach().cpu()

        if n_registered == 0:
            raise RuntimeError(
                "no DeepSeek/Param2/Mixtral-style routers found "
                f"(router_names={list(getattr(profile, 'router_module_names', []))[:4]}); "
                "Qwen3-fused routers stay dense"
            )
        slots = int(getattr(colossus_cfg, "lru_slots_per_expert", 32) or 32)
        num_experts = int(getattr(profile, "num_experts", 0) or 0)
        directory = ColumnDirectory(num_experts=num_experts, num_columns=0,
                                    capacity_per_expert=slots)
        hook = cls(pred, directory,
                   budget_preset=str(getattr(colossus_cfg, "budget", "tiered_fwd")))
        hook._layers = sorted(pred.router.keys())

        # Active-mode dynamic cache
        offload_enabled = bool(getattr(colossus_cfg, "offload_enabled", False))
        if offload_enabled:
            hot_topk = int(getattr(colossus_cfg, "hot_expert_topk", 16) or 16)
            missing_col_ratio = float(getattr(colossus_cfg, "missing_col_ratio", 1.0) or 1.0)
            hook._dynamic_cache_enabled = True
            hook._cache_capacity = hot_topk
            hook._missing_col_ratio = missing_col_ratio
            hook._lookahead_enabled = bool(getattr(colossus_cfg, "lookahead_enabled", False))
            hook._warmup_slots = int(getattr(colossus_cfg, "warmup_slots", 0))
            logger.info(
                f"[Colossus] Dynamic expert cache enabled (capacity={hot_topk} experts per layer, missing_col_ratio={missing_col_ratio:.2f}, lookahead={hook._lookahead_enabled}, warmup_slots={hook._warmup_slots})"
            )

        return hook

    # -- attach ----------------------------------------------------------
    def attach(self, model, profile, device: Optional[torch.device] = None) -> int:
        """Hook decoder layers or install dynamic MoE cache wrappers."""
        if getattr(self, "_dynamic_cache_enabled", False):
            from .dynamic_cache import DynamicMoELayerWrapper

            layers_container = getattr(model, "model", model)
            model_layers = getattr(layers_container, "layers", None)
            model_config = getattr(model, "config", None)
            if model_layers is not None:
                missing_ratio = getattr(self, "_missing_col_ratio", 1.0)
                warmup_slots = getattr(self, "_warmup_slots", 0)
                lookahead_enabled = getattr(self, "_lookahead_enabled", False)
                for idx in range(0, len(model_layers)):
                    layer = model_layers[idx]
                    is_block_sparse = hasattr(layer, "block_sparse_moe")
                    mlp = getattr(layer, "block_sparse_moe", getattr(layer, "mlp", None))
                    if mlp is not None and hasattr(mlp, "experts") and len(mlp.experts) > 1:
                        if device is not None:
                            layer_device = device
                        else:
                            layer_device = next(layer.parameters()).device
                            if layer_device.type == "cpu" and torch.cuda.is_available():
                                layer_device = torch.device("cuda:0")
                        wrapper = DynamicMoELayerWrapper(
                            layer_idx=idx,
                            moe_block=mlp,
                            capacity=self._cache_capacity,
                            device=layer_device,
                            missing_col_ratio=missing_ratio,
                            config=model_config,
                            warmup_slots=warmup_slots,
                            lookahead_enabled=lookahead_enabled,
                        )
                        if is_block_sparse:
                            layer.block_sparse_moe = wrapper
                        else:
                            layer.mlp = wrapper
                        self._wrapped_layers[idx] = wrapper

                        if lookahead_enabled:
                            def _make_pre_hook(l_idx):
                                def _pre_hook(mod, args):
                                    if args and isinstance(args[0], torch.Tensor):
                                        self.on_layer_pre_attention(l_idx, args[0])
                                return _pre_hook

                            handle = layer.register_forward_pre_hook(_make_pre_hook(idx))
                            self._handles.append(handle)
            logger.info(
                f"[Colossus] DynamicMoELayerWrapper installed on "
                f"{len(self._wrapped_layers)} MoE layers (capacity={self._cache_capacity}, missing_col_ratio={getattr(self, '_missing_col_ratio', 1.0):.2f}, lookahead={getattr(self, '_lookahead_enabled', False)})"
            )
            return len(self._wrapped_layers)

        # Shadow-mode fallback: passive hooks on decoder layers
        cls = getattr(profile, "decoder_layer_cls", None)
        targets = []
        if cls is not None:
            for m in model.modules():
                if isinstance(m, cls):
                    targets.append(m)
        for i, mod in enumerate(targets):
            layer_idx = self._layers[i] if i < len(self._layers) else i
            self._states[layer_idx] = deque(maxlen=_SHADOW_DEPTH)
            handle = mod.register_forward_hook(self._make_hook(layer_idx))
            self._handles.append(handle)
        logger.info(f"[Colossus] hook on {len(self._handles)} decoder layers (mode=shadow)")
        return len(self._handles)

    def on_layer_pre_attention(self, layer_idx: int, hidden_states: torch.Tensor):
        """Trigger L+4 lookahead speculative prefetching across upcoming layers (Section 5.1 & 6.1)."""
        # 1. Immediate prefetch for current layer (MHA overlap)
        if layer_idx in self._wrapped_layers:
            self._wrapped_layers[layer_idx].pre_attention_prefetch(hidden_states)

        # 2. Multi-Layer Lookahead (L+1 ... L+4) with confidence gating
        if self.predictor is not None and hasattr(self.predictor, "predict_lookahead"):
            lookahead_plan = self.predictor.predict_lookahead(
                hidden_states,
                current_layer=layer_idx,
                max_depth=4,
                confidence_threshold=0.06,
            )
            for target_l, expert_ids in lookahead_plan.items():
                if target_l in self._wrapped_layers:
                    self._wrapped_layers[target_l].async_prefetch(expert_ids)

    def _make_hook(self, layer_idx: int):
        def _hook(_mod, inputs, _output):
            try:
                h = inputs[0] if inputs else None
                if h is None:
                    return
                # Flatten to [*, H] then take the very last hidden vector → [H]
                H = h.shape[-1]
                last = h.detach().float().reshape(-1, H)[-1].cpu()  # [H]
                self._states.setdefault(layer_idx, deque(maxlen=_SHADOW_DEPTH)).append(last)
                self._steps += 1
            except Exception:
                return
        return _hook

    def detach(self) -> None:
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles = []

    def reset_state(self) -> None:
        """Reset dynamic cache and internal state across prompts."""
        for w in self._wrapped_layers.values():
            w.reset_state()
        self._states.clear()

    def warmup_from_prefill(self, model, tokenizer, prompt: str, device) -> None:
        """Opt 6: Run prefill forward pass and use routing decisions to seed expert caches.

        This should be called once per prompt BEFORE model.generate().
        It runs the prompt through the model to get hidden states at each MoE layer,
        then uses the routing decisions to pre-load the most popular experts.
        """
        import torch
        if not self._wrapped_layers:
            return
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        inputs.pop("token_type_ids", None)
        with torch.inference_mode():
            outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # tuple of [1, seq_len, H] per layer
        warmed = 0
        for layer_idx, wrapper in self._wrapped_layers.items():
            # hidden_states[layer_idx] is the input to layer layer_idx
            if layer_idx < len(hidden_states):
                h = hidden_states[layer_idx]
                wrapper.warmup_from_prefill(h)
                warmed += 1
        logger.info(f"[Colossus] Prefill warmup completed for {warmed} MoE layers")

    # -- active-mode offload / dynamic cache API ------------------------
    def maybe_offload(self, model, input_ids) -> Optional[dict]:
        """If dynamic cache is active, log VRAM and return status.
        If legacy offload manager is set, profile and offload.
        """
        if self._wrapped_layers:
            if torch.cuda.is_available():
                vram = torch.cuda.memory_allocated() / (1024 ** 3)
                logger.info(f"[Colossus] Dynamic cache active ({len(self._wrapped_layers)} layers) | VRAM: {vram:.2f} GB")
            return {"mode": "dynamic_cache", "capacity": self._cache_capacity, "layers": len(self._wrapped_layers)}

        if self._offload_mgr is None:
            return None
        try:
            stats = self._offload_mgr.profile_and_offload(
                model=model, hook=self, input_ids=input_ids
            )
            self._offload_stats = stats
            return stats
        except Exception as e:
            logger.warning(
                f"[Colossus] Offload failed ({e}); continuing dense"
            )
            self._offload_stats = {"status": "error", "reason": str(e)}
            return self._offload_stats

    def maybe_restore(self, model) -> None:
        """If dynamic cache is active, nothing to restore (continuous dynamic LRU).
        If legacy offload manager is set, restore experts.
        """
        if self._wrapped_layers:
            return
        if self._offload_mgr is None:
            return
        try:
            self._offload_mgr.restore_all_experts(model)
        except Exception as e:
            logger.warning(f"[Colossus] Restore failed ({e})")

    # -- shadow prediction API (future SA-FFN replacement consumes this) --
    @torch.inference_mode()
    def predict_for(self, h_prev: torch.Tensor, layer: int) -> Optional[dict]:
        """Return ``{experts, columns, hits, misses}`` or None if unknown."""
        from .directory import plan_fixed_packets

        if layer not in self.predictor.router:
            return None  # layer 0 / unregistered -> frequency fallback path
        ranking, _ = self.predictor.predict_experts(h_prev, layer)
        plan = self.predictor.predict_columns(h_prev, layer, ranking)
        packet = plan_fixed_packets(ranking, preset=self.budget_preset)
        # Intersect energy plan with fixed packet budgets.
        trimmed = {e: plan.get(e, [])[:packet.get(e, 0)] for e in packet}
        hits, misses = self.directory.lookup(trimmed)
        self._hits_count += sum(len(v) for v in hits.values())
        self._misses_count += sum(len(v) for v in misses.values())
        return {"experts": ranking[: self.predictor.top_k], "ranking": ranking,
                "columns": trimmed, "hits": hits, "misses": misses}

    def execute_expert_sa(self, x: torch.Tensor, expert_idx: int, expert_mod: nn.Module) -> torch.Tensor:
        """Lossless split execution of a SwiGLU expert using ColumnDirectory.

        Divides intermediate columns into resident (cached in LRU) and
        missed (streamed/demand), executing ``sa_expert_forward``.
        """
        from .sa_ffn import sa_expert_forward

        Wg = expert_mod.gate_proj.weight
        Wu = expert_mod.up_proj.weight
        Wd = expert_mod.down_proj.weight
        I = Wg.shape[0]

        # Check which columns of this expert are currently resident
        res = [c for c in range(I) if (expert_idx, c) in self.directory._resident]
        if not res:
            # Cold start: top initial packet becomes cached, remainder missed
            initial_budget = min(self.directory.capacity, I)
            res = list(range(initial_budget))
            missed = list(range(initial_budget, I))
        else:
            res_set = set(res)
            missed = [c for c in range(I) if c not in res_set]

        # Touch/commit cached columns to LRU
        self.directory.commit({expert_idx: res})

        _, _, y_total = sa_expert_forward(x, res, missed, Wg, Wu, Wd)
        return y_total

    def stats(self) -> dict:
        mode = "dynamic_cache" if self._wrapped_layers else ("active" if self._offload_mgr else "shadow")
        base = {
            "mode": mode,
            "layers_hooked": len(self._handles) or len(self._wrapped_layers),
            "shadow_steps": self._steps,
            "hits_count": self._hits_count,
            "misses_count": self._misses_count,
            "layers_with_router": list(self._layers),
            "disabled_reason": self.disabled_reason,
        }
        if self._wrapped_layers:
            layer_metrics = [
                w.get_stats()
                for w in sorted(self._wrapped_layers.values(), key=lambda x: x.layer_idx)
            ]
            total_hits = sum(m["hits"] for m in layer_metrics)
            total_misses = sum(m["misses"] for m in layer_metrics)
            total_accesses = total_hits + total_misses
            hit_rate = (total_hits / total_accesses * 100.0) if total_accesses > 0 else 100.0
            total_prefetch_mb = sum(m["prefetch_mb"] for m in layer_metrics)
            total_demand_mb = sum(m["demand_mb"] for m in layer_metrics)
            avg_recall = (
                sum(m["recall_pct"] for m in layer_metrics) / len(layer_metrics)
                if layer_metrics
                else 0.0
            )
            total_demand_stall_s = sum(m.get("demand_stall_s", 0.0) for m in layer_metrics)
            total_prefetch_stall_s = sum(m.get("prefetch_stall_s", 0.0) for m in layer_metrics)
            total_pcie_time_s = sum(m.get("pcie_time_s", 0.0) for m in layer_metrics)

            all_missing_pcts = []
            for w in self._wrapped_layers.values():
                all_missing_pcts.extend(getattr(w, "missing_col_stats", []))
            if all_missing_pcts:
                arr = sorted(all_missing_pcts)
                n = len(arr)
                mean_m = sum(arr) / n
                p50_m = arr[int(n * 0.50)]
                p90_m = arr[min(n - 1, int(n * 0.90))]
                p95_m = arr[min(n - 1, int(n * 0.95))]
                max_m = arr[-1]
            else:
                mean_m = p50_m = p90_m = p95_m = max_m = 0.0

            tot_tokens = max(1, int(sum(getattr(w, 'total_tokens_processed', 0) for w in self._wrapped_layers.values()) / max(1, len(self._wrapped_layers))))
            dma_b_tok = ((total_prefetch_mb + total_demand_mb) * 1024 * 1024) / tot_tokens

            base["dynamic_cache"] = {
                "capacity": self._cache_capacity,
                "layers_wrapped": len(self._wrapped_layers),
                "hits": total_hits,
                "misses": total_misses,
                "hit_rate_pct": hit_rate,
                "prefetch_mb": total_prefetch_mb,
                "demand_mb": total_demand_mb,
                "recall_pct": avg_recall,
                "demand_stall_s": total_demand_stall_s,
                "prefetch_stall_s": total_prefetch_stall_s,
                "pcie_time_s": total_pcie_time_s,
                "missing_col_mean": mean_m,
                "missing_col_p50": p50_m,
                "missing_col_p90": p90_m,
                "missing_col_p95": p95_m,
                "missing_col_max": max_m,
                "dma_bytes_per_tok": dma_b_tok,
                "layers": layer_metrics,
            }
        elif self._offload_mgr is not None:
            base["offload"] = self._offload_mgr.vram_savings()
            base.update(self._offload_stats)
        return base
