"""Shadow-mode COLOSSUS hook for MoE inference (Bhaskera HF backend).

Rationale: :meth:`InferenceEngine.generate` delegates to HF
``model.generate()`` (C++/CUDA loop), so per-layer pre-attention
interception is only possible via module forward hooks. This hook runs
**alongside** dense execution and never alters numerics:

* attaches to each decoder-layer module (from ``ModelProfile``) and
  records the pre-layer hidden state (last token, detached to CPU,
  bounded deque — no VRAM growth);
* builds a :class:`ZSSRPredictor` from router weights where the
  architecture exposes them (DeepSeek-style ``mlp.gate`` today;
  Qwen3-fused / Param2-custom routers log a reason and stay dense);
* exposes :meth:`predict_for` (used by the future SA-FFN replacement)
  and :meth:`stats` (layers hooked, shadow steps, mode).

Any failure disables the hook with a warning — dense output is always
bit-identical with or without it. Full per-token FFN replacement
(``sa_expert_forward`` wired into expert modules) is arch-specific work
tracked per model family, not attempted generically here.
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

        for idx in range(1, max(num_layers, 1)):
            if model_layers is None or idx >= len(model_layers):
                continue
            layer = model_layers[idx]
            mlp = getattr(layer, "mlp", None)
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
                    if hasattr(exp_mod, "gate_proj") and hasattr(exp_mod, "up_proj"):
                        pred.gate_w[idx][e] = exp_mod.gate_proj.weight.detach().float().cpu()
                        pred.up_w[idx][e] = exp_mod.up_proj.weight.detach().float().cpu()

        if n_registered == 0:
            raise RuntimeError(
                "no DeepSeek/Param2-style mlp.gate routers found "
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
        return hook

    # -- attach ----------------------------------------------------------
    def attach(self, model, profile) -> int:
        """Hook decoder layers; returns number of layers hooked."""
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
        logger.info(f"[Colossus] shadow hook on {len(self._handles)} decoder layers")
        return len(self._handles)

    def _make_hook(self, layer_idx: int):
        def _hook(_mod, inputs, _output):
            try:
                h = inputs[0] if inputs else None
                if h is None:
                    return
                last = h[:, -1, :].detach().float().cpu()
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
        trimmed = {e: plan.get(e, [])[: packet.get(e, 0)] for e in packet}
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
        return {"mode": "shadow", "layers_hooked": len(self._handles),
                "shadow_steps": self._steps,
                "hits_count": self._hits_count,
                "misses_count": self._misses_count,
                "layers_with_router": list(self._layers),
                "disabled_reason": self.disabled_reason}
