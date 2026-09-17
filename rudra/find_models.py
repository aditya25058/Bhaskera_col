import os

search_paths = [
    "/home/bapic_iiitd/.cache/huggingface/hub",
    "/home/bapic_iiitd/2_group",
    "/home/bapic_iiitd",
]

found_models = {}

for sp in search_paths:
    if not os.path.exists(sp):
        continue
    for root, dirs, files in os.walk(sp):
        # check if this dir has model weights
        weights = [f for f in files if f.endswith(".safetensors") or f.endswith(".bin") or f == "model.safetensors.index.json"]
        if weights:
            total_sz = sum(os.path.getsize(os.path.join(root, f)) for f in files if f.endswith(".safetensors") or f.endswith(".bin"))
            sz_gb = total_sz / (1024 ** 3)
            if sz_gb > 0.5: # only real models
                found_models[root] = sz_gb
        # don't recurse into .git or uv
        dirs[:] = [d for d in dirs if d not in [".git", "uv", ".cache"] or d == "huggingface"]

print("=== FOUND MODELS ON SERVER ===")
for path, sz in sorted(found_models.items(), key=lambda x: x[1], reverse=True):
    print(f"  {sz:6.2f} GB : {path}")
