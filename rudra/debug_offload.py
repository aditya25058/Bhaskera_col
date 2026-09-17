"""Debug: trace the exact types from profile_prompt to offload_cold_experts."""
import sys
sys.path.insert(0, "src")

import torch
import torch.nn as nn

from bhaskera.inference.colossus.predictor import ZSSRPredictor
from bhaskera.inference.colossus.offload import ExpertOffloadManager

# Simulate what happens on Param2: h_prev is [1, H] (batch=1, last token)
H, E = 2048, 64
pred = ZSSRPredictor(top_k_experts=8)
pred.router[1] = torch.randn(E, H)

h_prev = torch.randn(1, H)  # [1, H] — what the hook captures
ranking, logits = pred.predict_experts(h_prev, 1)

print(f"h_prev.shape: {h_prev.shape}")
print(f"logits.shape: {logits.shape}")
print(f"type(ranking): {type(ranking)}")
print(f"len(ranking): {len(ranking)}")
print(f"ranking[:3]: {ranking[:3]}")
print(f"type(ranking[0]): {type(ranking[0])}")

hot = ranking[:8]
print(f"\nhot (top-8): {hot}")
print(f"type(hot[0]): {type(hot[0])}")

try:
    hot_set = set(hot)
    print(f"set(hot) works: {hot_set}")
except TypeError as e:
    print(f"set(hot) FAILED: {e}")
    print(f"hot contains: {[type(x) for x in hot]}")

# Now test with the actual offload manager flow
hot_map = {1: hot}
print(f"\nhot_map: { {k: v[:3] for k,v in hot_map.items()} }")

class FakeExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(128, 64, bias=False)
        self.up_proj = nn.Linear(128, 64, bias=False)
        self.down_proj = nn.Linear(64, 128, bias=False)

class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        layers = nn.ModuleList()
        for _ in range(2):
            layer = nn.Module()
            mlp = nn.Module()
            experts = nn.ModuleList([FakeExpert() for _ in range(8)])
            mlp.experts = experts
            layer.mlp = mlp
            layers.append(layer)
        self.model.layers = layers

model = FakeModel()
mgr = ExpertOffloadManager(hot_expert_topk=8, device="cpu")

try:
    result = mgr.offload_cold_experts(model, hot_map)
    print(f"\noffload result: {result}")
except Exception as e:
    print(f"\noffload FAILED: {e}")
    import traceback
    traceback.print_exc()
