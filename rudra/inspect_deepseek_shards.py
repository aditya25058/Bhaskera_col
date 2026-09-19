import json

idx_path = "/home/palakm/MoEServingSim/aditya/models/DeepSeek-Coder-V2-Instruct/model.safetensors.index.json"
with open(idx_path, "r") as f:
    idx = json.load(f)

wm = idx["weight_map"]
print(f"Total tensor keys in index: {len(wm)}")

# Sample keys
sample_routed = [k for k in wm if "layers.1.mlp.experts." in k]
sample_shared = [k for k in wm if "layers.1.mlp.shared_experts." in k]
sample_attn = [k for k in wm if "layers.1.self_attn." in k]

print(f"Sample routed expert keys in layer 1 ({len(sample_routed)} total): {sample_routed[:4]}")
print(f"Sample shared expert keys in layer 1 ({len(sample_shared)} total): {sample_shared}")
print(f"Sample attention keys in layer 1 ({len(sample_attn)} total): {sample_attn[:4]}")

# Count routed vs non-routed tensors
total_routed = len([k for k in wm if ".experts." in k])
total_other = len(wm) - total_routed
print(f"Total routed expert tensors: {total_routed}")
print(f"Total non-routed tensors: {total_other}")
