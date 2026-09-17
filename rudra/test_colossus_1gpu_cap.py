#!/usr/bin/env python3
"""
Test 1-GPU COLOSSUS under 24-GB Cap.
Target: Load model on CPU, build hook, attach to cuda:0, generate 1 prompt.
"""
import time
import torch
from types import SimpleNamespace
from transformers import AutoModelForCausalLM, AutoTokenizer
from bhaskera.introspect import introspect_model
from bhaskera.inference.colossus.hook import ColossusMoEHook
from bhaskera.inference.colossus.dynamic_cache import DynamicMoELayerWrapper

model_dir = "/home/bapic_iiitd/.cache/huggingface/hub/models--bharatgenai--Param2-17B-A2.4B-Thinking/snapshots/1d5b7897cf1eec5ad5159f216e757e71b751c979"
device = torch.device("cuda:0")

# Set 24GB cap
total_mem = torch.cuda.get_device_properties(0).total_memory
fraction = min(1.0, (24.0 * (1024**3)) / total_mem)
torch.cuda.set_per_process_memory_fraction(fraction, 0)
print(f"Set 24GB cap: fraction = {fraction:.4f}")

# 1. Load model on CPU
t0 = time.time()
print("Loading model on CPU...")
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    trust_remote_code=True,
)
model.eval()
print(f"Loaded on CPU in {time.time() - t0:.2f}s")

# 2. Build hook while model.layers[*].mlp is intact!
profile = introspect_model(model)
colossus_cfg = SimpleNamespace(
    enabled=True,
    offload_enabled=True,
    hot_expert_topk=8,
    missing_col_ratio=0.70,
    lru_slots_per_expert=8,
    budget="tiered_fwd",
)
hook = ColossusMoEHook.build(model, profile, colossus_cfg)
print("Hook built successfully!")

# 3. Move base components to cuda:0 and wrap MoE layers
layers_container = getattr(model, "model", model)
model_layers = layers_container.layers

layers_container.word_embeddings.to(device)
layers_container.rotary_emb.to(device)
layers_container.norm.to(device)
model.lm_head.to(device)
model_layers[0].to(device)

wrapped_layers = {}
for idx in range(1, len(model_layers)):
    layer = model_layers[idx]
    mlp = layer.mlp
    wrapper = DynamicMoELayerWrapper(
        layer_idx=idx,
        moe_block=mlp,
        capacity=8,
        device=device,
        missing_col_ratio=0.70,
    )
    layer.mlp = wrapper
    wrapped_layers[idx] = wrapper
    # Move attention and input layernorms to cuda:0
    for name, child in layer.named_children():
        if name != "mlp":
            child.to(device)

hook._wrapped_layers = wrapped_layers
hook._dynamic_cache_enabled = True
hook._cache_capacity = 8
hook._missing_col_ratio = 0.70

for idx in range(1, len(model_layers)):
    def _make_pre(l_idx):
        def _pre(m, a):
            if a and isinstance(a[0], torch.Tensor):
                hook.on_layer_pre_attention(l_idx, a[0])
        return _pre
    model_layers[idx].register_forward_pre_hook(_make_pre(idx))

torch.cuda.empty_cache()
peak_hbm = torch.cuda.max_memory_allocated(0) / (1024**3)
print(f"Attached to cuda:0! Peak HBM = {peak_hbm:.2f} GB (< 24.0 GB!)")

# 4. Run test generation
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
prompt = "What is the capital of India?"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
inputs.pop("token_type_ids", None)

t0_gen = time.time()
with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=32, do_sample=False)
t_gen = time.time() - t0_gen
peak_gen = torch.cuda.max_memory_allocated(0) / (1024**3)
text = tokenizer.decode(out[0], skip_special_tokens=True)
print(f"Generated 32 tokens in {t_gen:.2f}s ({32/t_gen:.2f} tok/s)!")
print(f"Peak HBM during generation = {peak_gen:.2f} GB (< 24.0 GB!)")
print(f"Output: {text[:100]}...")
