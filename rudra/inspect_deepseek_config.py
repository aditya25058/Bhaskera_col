#!/usr/bin/env python3
import json
from transformers import AutoConfig

MODEL_PATH = "/home/palakm/MoEServingSim/aditya/models/DeepSeek-Coder-V2-Instruct"
cfg = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)

print("Architecture:", cfg.architectures)
print("num_hidden_layers:", cfg.num_hidden_layers)
print("hidden_size:", cfg.hidden_size)
print("intermediate_size:", cfg.intermediate_size)
print("moe_intermediate_size:", cfg.moe_intermediate_size)
print("first_k_dense_replace:", getattr(cfg, "first_k_dense_replace", None))
print("moe_layer_freq:", getattr(cfg, "moe_layer_freq", None))
print("n_routed_experts:", getattr(cfg, "n_routed_experts", None))
print("num_experts_per_tok:", getattr(cfg, "num_experts_per_tok", None))
print("n_shared_experts:", getattr(cfg, "n_shared_experts", None))
print("norm_topk_prob:", getattr(cfg, "norm_topk_prob", None))
print("routed_scaling_factor:", getattr(cfg, "routed_scaling_factor", None))

import inspect
import torch
from transformers import AutoModelForCausalLM
with torch.device("meta"):
    meta_model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)

print("m.model submodules:", [k for k, _ in meta_model.model.named_children()])
print("layer 0 submodules:", [k for k, _ in meta_model.model.layers[0].named_children()])
print("layer 0 self_attn submodules:", [k for k, _ in meta_model.model.layers[0].self_attn.named_children()])
attn = meta_model.model.layers[0].self_attn
print("attn attributes related to rotary:", [k for k in dir(attn) if "rotary" in k or "rope" in k or "dim" in k])
print("rotary_emb init signature:", inspect.signature(type(rot).__init__))






