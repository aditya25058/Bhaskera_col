"""ZSSR predictor vendored for Bhaskera (from ``zssr_standalone``).

Pipeline per (prev-layer hidden ``h_{L-1}`` -> curr layer ``L``):

1. Expert stage (zero-shot, zero trainable params):
   ``logits = h_{L-1} @ W_L.T`` -> Top-8 ranking. Layer 0 has no
   ``h_{-1}`` and must use the Markov/frequency fallback instead.
2. Column stage (resident INT8/INT4 SwiGLU surrogate):
   for each predicted expert ``e``:
   ``energy = (silu(h @ Wg_e.T) * (h @ Wu_e.T))^2`` -> top-k columns.

``torch`` is imported lazily so CPU-only hosts can still build configs.
"""
from __future__ import annotations


def quantize_int8(tensor):
    """Symmetric per-tensor INT8 fake-quant (lossless at V2 budgets)."""
    import torch

    scale = tensor.abs().max() / 127.0
    q = torch.clamp(torch.round(tensor / scale), -128, 127)
    return q * scale


class ZSSRPredictor:
    """Holds router + gate/up weights; pure inference, no training.

    Defaults target Qwen3-30B-A3B (128 experts, top-8, 48 MoE layers);
    override ``top_k_experts``/``num_col_experts``/``top_cols`` for
    DeepSeek-V2-Lite (64 experts, top-6) or Param2 (64+2, top-6).
    """

    def __init__(self, top_k_experts: int = 8, top_cols: int = 50,
                 num_col_experts: int = 8, device: str = "cpu"):
        self.top_k = top_k_experts
        self.top_cols = top_cols
        self.num_col_experts = num_col_experts
        self.device = device
        self.router = {}  # layer -> [E,H] float32 cpu
        self.gate_w = {}  # layer -> {e: [I,H]}
        self.up_w = {}    # layer -> {e: [I,H]}

    def load_from_hf_model(self, model, layers=None, quantize: bool = True):
        """Extract router + expert gate/up weights from a loaded HF model.

        Expects DeepSeek-style ``layer.mlp.gate`` + ``layer.mlp.experts[i]
        .gate_proj/.up_proj``. Qwen3 fused ``gate_up_proj`` checkpoints must
        be split by the caller before registering here.
        """
        if layers is None:
            n = len(model.model.layers)
            layers = range(1, n)
        for l_idx in layers:
            layer = model.model.layers[l_idx]
            assert hasattr(layer.mlp, "gate"), f"layer {l_idx} has no mlp.gate"
            self.router[l_idx] = layer.mlp.gate.weight.detach().float().cpu()
            self.gate_w[l_idx] = {}
            self.up_w[l_idx] = {}
            for e, exp_mod in enumerate(layer.mlp.experts):
                g = exp_mod.gate_proj.weight.detach().float()
                u = exp_mod.up_proj.weight.detach().float()
                if quantize:
                    g = quantize_int8(g)
                    u = quantize_int8(u)
                self.gate_w[l_idx][e] = g.cpu()
                self.up_w[l_idx][e] = u.cpu()

    def predict_experts(self, h_prev, layer: int):
        """``h_prev``: [H] or [1,H]. Returns (ranking, logits)."""
        import torch

        with torch.no_grad():
            W = self.router[layer]
            if h_prev.dim() == 1:
                h_prev = h_prev.unsqueeze(0)
            logits = (h_prev.float() @ W.T).squeeze(0)
            ranking = torch.argsort(logits, descending=True).tolist()
            return ranking, logits

    def predict_columns(self, h_prev, layer: int, spec_ranking):
        """SwiGLU-surrogate top-k columns for top-N experts.

        Returns ``{expert_id: [col ids]}``.
        """
        import torch
        import torch.nn.functional as F

        with torch.no_grad():
            if h_prev.dim() == 1:
                h_prev = h_prev.unsqueeze(0)
            h = h_prev.float()
            plan = {}
            for exp in spec_ranking[: self.num_col_experts]:
                Wg = self.gate_w[layer][exp]
                Wu = self.up_w[layer][exp]
                g = F.silu(h @ Wg.T)
                u = h @ Wu.T
                energy = (g * u).pow(2).squeeze(0)
                plan[exp] = torch.topk(energy, k=self.top_cols).indices.tolist()
            return plan

    def predict(self, h_prev, layer: int):
        """Full per-layer prediction: experts + columns."""
        ranking, logits = self.predict_experts(h_prev, layer)
        cols = self.predict_columns(h_prev, layer, ranking)
        return {"experts_top8": ranking[: self.top_k], "ranking": ranking,
                "logits": logits, "columns": cols}
