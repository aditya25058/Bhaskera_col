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

from transformers.cache_utils import DynamicCache

def get_usable_length(self, *args, **kwargs):
    layer_idx = 0
    if len(args) > 1 and isinstance(args[1], int):
        layer_idx = args[1]
    elif "layer_idx" in kwargs:
        layer_idx = kwargs["layer_idx"]
    return self.get_seq_length(layer_idx)

DynamicCache.get_usable_length = get_usable_length

dc = DynamicCache()
inp = torch.randint(0, 1000, (1, 6), device="meta")
out = meta_model(input_ids=inp, past_key_values=dc, use_cache=True)
print("Prefill out logits shape:", out.logits.shape, "seq_len:", dc.get_seq_length(0))

# Step with 1 token
step_inp = torch.randint(0, 1000, (1, 1), device="meta")
step_out = meta_model(input_ids=step_inp, past_key_values=dc, use_cache=True)
print("Step out logits shape:", step_out.logits.shape, "seq_len:", dc.get_seq_length(0))









