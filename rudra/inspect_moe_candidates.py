from transformers import AutoConfig

candidates = [
    "Qwen/Qwen1.5-MoE-A2.7B",
    "allenai/OLMoE-1B-7B-0924",
    "bharatgenai/Param2-17B-A2.4B-Thinking",
    "deepseek-ai/deepseek-moe-16b-base",
    "Qwen/Qwen2-57B-A14B",
    "mistralai/Mixtral-8x7B-v0.1",
]

for m in candidates:
    try:
        cfg = AutoConfig.from_pretrained(m, trust_remote_code=True)
        h = getattr(cfg, "hidden_size", None)
        i = getattr(cfg, "moe_intermediate_size", getattr(cfg, "intermediate_size", None))
        num_e = getattr(cfg, "n_routed_experts", getattr(cfg, "num_local_experts", getattr(cfg, "num_experts", None)))
        top_k = getattr(cfg, "num_experts_per_tok", getattr(cfg, "top_k", None))
        exp_bytes = (h * i * 3 * 2) if (h and i) else 0
        exp_mb = exp_bytes / (1024**2)
        print(f"{m:38s} | H={h:4d} | I={i:5d} | E={num_e:2d} | top_k={top_k} | Expert={exp_mb:6.1f} MB")
    except Exception as e:
        print(f"{m:38s} | Error: {e}")
