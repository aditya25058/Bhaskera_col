import time, torch
from types import SimpleNamespace
from transformers import AutoModelForCausalLM, AutoTokenizer
from bhaskera.introspect import introspect_model
from bhaskera.inference.colossus.hook import ColossusMoEHook

dev = torch.device('cuda:0')
model_dir = '/home/bapic_iiitd/2_group/models/Mixtral-8x7B-v0.1'

print('[1] Loading model...')
t0 = time.perf_counter()
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)
print(f'Model loaded: {time.perf_counter()-t0:.2f}s')

print('[2] Moving non-MoE to GPU...')
t1 = time.perf_counter()
model.model.embed_tokens.to(dev)
model.model.norm.to(dev)
model.lm_head.to(dev)
for layer in model.model.layers:
    layer.self_attn.to(dev)
    layer.input_layernorm.to(dev)
    layer.post_attention_layernorm.to(dev)
    layer.block_sparse_moe.gate.to(dev)
print(f'Non-MoE moved: {time.perf_counter()-t1:.2f}s | HBM: {torch.cuda.memory_allocated(dev)/(1024**3):.2f} GB')

print('[3] Attaching hook...')
t2 = time.perf_counter()
profile = introspect_model(model)
colossus_cfg = SimpleNamespace(
    enabled=True,
    offload_enabled=True,
    hot_expert_topk=4,
    missing_col_ratio=1.0,
    lru_slots_per_expert=4,
    budget='tiered_fwd',
)
hook = ColossusMoEHook.build(model, profile, colossus_cfg)
hook.attach(model, profile, device=dev)
print(f'Hook attached: {time.perf_counter()-t2:.2f}s | HBM: {torch.cuda.memory_allocated(dev)/(1024**3):.2f} GB')

prompt = 'Explain Mixture of Experts.'
inputs = tokenizer(prompt, return_tensors='pt').to(dev)
inputs.pop('token_type_ids', None)

print('[4] Running single forward pass (Prefill)...')
t3 = time.perf_counter()
with torch.inference_mode():
    out = model(**inputs)
print(f'Prefill forward done in: {time.perf_counter()-t3:.2f}s | Peak HBM: {torch.cuda.max_memory_allocated(dev)/(1024**3):.2f} GB')

print('[5] Running 1 decode step (seq_len=1)...')
next_token = torch.tensor([[out.logits[0, -1].argmax().item()]], device=dev)
t4 = time.perf_counter()
with torch.inference_mode():
    out2 = model(next_token)
print(f'Decode forward done in: {time.perf_counter()-t4:.2f}s')

print('ALL SUCCESSFUL!')
