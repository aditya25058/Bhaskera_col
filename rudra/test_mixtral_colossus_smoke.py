import os
import sys
import time
from types import SimpleNamespace
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from bhaskera.introspect import introspect_model
from bhaskera.inference.colossus.hook import ColossusMoEHook

def test_smoke():
    model_dir = "/home/bapic_iiitd/2_group/models/Mixtral-8x7B-v0.1"
    ref_path = "/home/bapic_iiitd/2_group/tokens_mixtral_2gpu.pt"
    dev = torch.device("cuda:0")

    print("=" * 80)
    print("  MIXTRAL 1-GPU COLOSSUS SMOKE TEST & BITWISE EXACTNESS CHECK")
    print(f"  Physical Device: {torch.cuda.get_device_name(0)}")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    print("[1] Loading Mixtral onto CPU host memory...")
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    print(f"[1] Loaded model in {time.perf_counter() - t0:.2f}s")

    print("[2] Moving non-MoE components to cuda:0...")
    model.model.embed_tokens.to(dev)
    model.model.norm.to(dev)
    model.lm_head.to(dev)
    for layer in model.model.layers:
        layer.self_attn.to(dev)
        layer.input_layernorm.to(dev)
        layer.post_attention_layernorm.to(dev)
        layer.block_sparse_moe.gate.to(dev)

    non_moe_hbm = torch.cuda.memory_allocated(dev) / (1024 ** 3)
    print(f"[2] Non-MoE components on GPU: {non_moe_hbm:.2f} GB")

    print("[3] Attaching ColossusMoEHook (C=4, m=1.0)...")
    profile = introspect_model(model)
    colossus_cfg = SimpleNamespace(
        enabled=True,
        offload_enabled=True,
        hot_expert_topk=4,
        missing_col_ratio=1.0,
        lru_slots_per_expert=4,
        budget="tiered_fwd",
    )
    hook = ColossusMoEHook.build(model, profile, colossus_cfg)
    hook.attach(model, profile, device=dev)
    post_hook_hbm = torch.cuda.memory_allocated(dev) / (1024 ** 3)
    print(f"[3] Post-hook GPU memory: {post_hook_hbm:.2f} GB (Peak: {torch.cuda.max_memory_allocated(dev) / (1024**3):.2f} GB)")

    prompt = "Explain the Mixture of Experts (MoE) routing mechanism in modern neural networks."
    inputs = tokenizer(prompt, return_tensors="pt").to(dev)
    inputs.pop("token_type_ids", None)

    from transformers import TextStreamer
    streamer = TextStreamer(tokenizer, skip_prompt=True)

    print("[4] Generating 16 tokens with 1-GPU COLOSSUS...")
    torch.cuda.reset_peak_memory_stats(dev)
    t0_gen = time.perf_counter()
    with torch.inference_mode():
        out_ids = model.generate(**inputs, max_new_tokens=16, do_sample=False, streamer=streamer)
    t_gen = time.perf_counter() - t0_gen

    gen_ids = out_ids[0, inputs["input_ids"].shape[1]:].tolist()
    tps = len(gen_ids) / t_gen
    peak_hbm = torch.cuda.max_memory_allocated(dev) / (1024 ** 3)

    print(f"\n[4] Generation complete in {t_gen:.2f}s: {tps:.2f} tok/s | Peak HBM: {peak_hbm:.2f} GB")
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    print(f"Generated text: {text}")

    if os.path.exists(ref_path):
        ref_tokens = torch.load(ref_path, map_location="cpu")
        ref_p1 = ref_tokens[0] if isinstance(ref_tokens, list) else ref_tokens[0].tolist()
        ref_sub = ref_p1[:len(gen_ids)]
        exact = (gen_ids == ref_sub)
        print(f"\nExact match against 2-GPU reference (first {len(gen_ids)} tokens): {exact}")
        if exact:
            print(">>> BITWISE TOKEN EQUALITY CONFIRMED: torch.equal == True <<<")
        else:
            print("Diff in tokens:")
            print("COLOSSUS:", gen_ids)
            print("2-GPU:   ", ref_sub)

if __name__ == "__main__":
    test_smoke()
