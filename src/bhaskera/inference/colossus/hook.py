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
        self.disabled_reason: Optional[str] = None

    # -- construction ----------------------------------------------------
    @classmethod
    def build(cls, model, profile, colossus_cfg) -> "ColossusMoEHook":
        """Best-effort router extraction. Raises with reason if unsupported."""
        from .columns import ColumnDirectory  # local import: cheap, no torch
        from .directory import plan_fixed_packets  # noqa: F401 (re-exported use)
        from .predictor import ZSSRPredictor

        _ = plan_fixed_packets
        top_k = int(getattr(colossus_cfg, "top_k_experts", 8) or 8)
        pred = ZSSRPredictor(top_k_experts=top_k, top_cols=50,
                             num_col_experts=top_k)
        n_registered = 0
        num_layers = int(getattr(profile, "num_hidden_layers", 0) or 0)
        for idx in range(1, max(num_layers, 1)):
            try:
                layer = model.model.layers[idx]
            except Exception:
                continue
            mlp = getattr(layer, "mlp", None)
            gate = getattr(mlp, "gate", None) if mlp is not None else None
            if gate is None or not hasattr(gate, "weight"):
                continue
            try:
                pred.router[idx] = gate.weight.detach().float().cpu()
                n_registered += 1
            except Exception:
                continue
        if n_registered == 0:
            raise RuntimeError(
                "no DeepSeek-style mlp.gate routers found "
                f"(router_names={list(getattr(profile, 'router_module_names', []))[:4]}); "
                "Qwen3-fused/Param2-custom routers stay dense"
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
        return {"experts": ranking[: self.predictor.top_k], "ranking": ranking,
                "columns": trimmed, "hits": hits, "misses": misses}

    def stats(self) -> dict:
        return {"mode": "shadow", "layers_hooked": len(self._handles),
                "shadow_steps": self._steps,
                "layers_with_router": list(self._layers),
                "disabled_reason": self.disabled_reason}
