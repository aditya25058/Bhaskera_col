from transformers import AutoConfig

print("--- Qwen2-57B-A14B ---")
cfg57 = AutoConfig.from_pretrained("Qwen/Qwen2-57B-A14B", trust_remote_code=True)
for k, v in cfg57.__dict__.items():
    if any(term in k for term in ["expert", "moe", "shared", "top"]):
        print(f"  {k}: {v}")

print("\n--- Qwen1.5-MoE-A2.7B ---")
cfg2 = AutoConfig.from_pretrained("Qwen/Qwen1.5-MoE-A2.7B", trust_remote_code=True)
for k, v in cfg2.__dict__.items():
    if any(term in k for term in ["expert", "moe", "shared", "top"]):
        print(f"  {k}: {v}")
