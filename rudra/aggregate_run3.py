import json
import torch
from transformers import AutoTokenizer

col_tokens = torch.load('/home/bapic_iiitd/2_group/tokens_mixtral_colossus.pt')
ref_tokens = torch.load('/home/bapic_iiitd/2_group/tokens_mixtral_2gpu.pt')
tokenizer = AutoTokenizer.from_pretrained('/home/bapic_iiitd/2_group/models/Mixtral-8x7B-v0.1')

prompt_results = []
all_exact = True

for i, (c, r) in enumerate(zip(col_tokens, ref_tokens)):
    c_list = c.tolist() if isinstance(c, torch.Tensor) else c
    r_list = r.tolist() if isinstance(r, torch.Tensor) else r
    r_sub = r_list[:len(c_list)]
    eq = torch.equal(torch.tensor(c_list), torch.tensor(r_sub))
    all_exact = all_exact and eq
    c_text = tokenizer.decode(c_list, skip_special_tokens=True)
    r_text = tokenizer.decode(r_sub, skip_special_tokens=True)
    prompt_results.append({
        "prompt_idx": i + 1,
        "tokens_generated": len(c_list),
        "torch_equal": bool(eq),
        "token_ids": c_list,
        "decoded_text": c_text,
        "text_identical": bool(c_text == r_text)
    })

summary = {
    "experiment": "Mixtral-8x7B Physical 3-Run Experimental Sequence",
    "model": "mistralai/Mixtral-8x7B-v0.1 (86.99 GB BF16)",
    "device": "NVIDIA A100 80GB PCIe",
    "total_physical_hbm_gb": 79.15,
    "run1_1gpu_dense": {
        "n_gpus": 1,
        "result": "HARD_OOM",
        "allocated_at_failure_gib": 78.64,
        "tps": 0.0,
        "exact": None
    },
    "run2_2gpu_dense": {
        "n_gpus": 2,
        "result": "FEASIBLE",
        "peak_hbm_per_gpu_gb": 43.51,
        "tps": 14.50,
        "ms_per_tok": 68.97,
        "exact": "Reference"
    },
    "run3_1gpu_colossus": {
        "n_gpus": 1,
        "result": "FEASIBLE_EXACT",
        "capacity_slots": 4,
        "peak_hbm_gb": 55.51,
        "hbm_headroom_gb": 23.64,
        "non_moe_hbm_gb": 2.99,
        "moe_resident_hbm_gb": 52.50,
        "tps": 0.09,
        "ms_per_tok": 11158.27,
        "cache_hit_rate_pct": 45.72,
        "all_prompts_torch_equal": all_exact,
        "prompt_details": prompt_results
    }
}

with open('/home/bapic_iiitd/2_group/run3_metrics_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("Aggregated summary successfully written to /home/bapic_iiitd/2_group/run3_metrics_summary.json")
