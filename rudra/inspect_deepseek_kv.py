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

from transformers.cache_utils import DynamicCache, Cache
dc = DynamicCache()
print("isinstance(dc, Cache):", isinstance(dc, Cache))




