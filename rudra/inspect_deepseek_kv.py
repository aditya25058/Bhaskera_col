#!/usr/bin/env python3
"""
inspect_deepseek_kv.py
======================
Inspects KV-cache structure and past_key_value mechanics in DeepSeek-V2 Attention.
"""

import inspect
import torch
from transformers import AutoConfig, AutoModelForCausalLM

MODEL_PATH = "/home/palakm/MoEServingSim/aditya/models/DeepSeek-Coder-V2-Instruct"
cfg = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)

with torch.device("meta"):
    meta_model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)

layer0 = meta_model.model.layers[0]
attn = layer0.self_attn

for i in range(cfg.num_hidden_layers):
    attn = meta_model.model.layers[i].self_attn
    assert hasattr(attn, "layer_idx") and attn.layer_idx == i, f"Layer {i} layer_idx issue: {getattr(attn, 'layer_idx', None)}"
print(f"All {cfg.num_hidden_layers} attention layers have valid layer_idx!")


