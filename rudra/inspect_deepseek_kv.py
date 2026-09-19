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

from transformers.modeling_attn_mask_utils import AttentionMaskConverter
_orig_to_causal_4d = AttentionMaskConverter.to_causal_4d

def patched_to_causal_4d(self, batch_size, query_length, key_value_length, dtype, device="cpu"):
    mask = _orig_to_causal_4d(self, batch_size, query_length, key_value_length, dtype, device)
    if mask is None:
        mask = torch.zeros((batch_size, 1, query_length, key_value_length), dtype=dtype, device=device)
    return mask

AttentionMaskConverter.to_causal_4d = patched_to_causal_4d

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
layer0 = meta_model.model.layers[0]
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask


# 1. Prefill (6 tokens)
hidden_states = torch.randn(1, 6, cfg.hidden_size, device="meta")
mask = _prepare_4d_causal_attention_mask(None, (1, 6), hidden_states, 0)
out_l0, _, _ = layer0.self_attn(hidden_states=hidden_states, attention_mask=mask, past_key_value=dc, use_cache=True)
print("Prefill layer 0 attn done! seq_len:", dc.get_seq_length(0))

# 2. Decode step (1 token)
hidden_step = torch.randn(1, 1, cfg.hidden_size, device="meta")
mask_step = _prepare_4d_causal_attention_mask(None, (1, 1), hidden_step, dc.get_seq_length(0))
print("mask_step shape:", mask_step.shape)
out_l0_step, _, _ = layer0.self_attn(hidden_states=hidden_step, attention_mask=mask_step, past_key_value=dc, use_cache=True)
print("Decode step layer 0 attn done! seq_len:", dc.get_seq_length(0))










