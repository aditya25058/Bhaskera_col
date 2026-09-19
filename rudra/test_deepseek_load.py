#!/usr/bin/env python3
"""
test_deepseek_load.py
=====================
Verifies loading the downloaded DeepSeek-Coder-V2 model files and tokenizer
on Server 192.168.3.214 in /home/palakm/MoEServingSim/aditya.
"""

import time
import torch
from transformers import AutoTokenizer, AutoConfig

MODEL_PATH = "/home/palakm/MoEServingSim/aditya/models/DeepSeek-Coder-V2-Instruct"

print("=" * 70)
print("  VERIFYING DOWNLOADED DEEPSEEK-CODER-V2 (236B) ON 2x H100 NVL")
print("=" * 70)

t0 = time.time()
print(f"Loading tokenizer from: {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
print(f"Tokenizer loaded in {time.time() - t0:.2f}s | Vocab size: {len(tokenizer)}")

t1 = time.time()
print(f"Loading configuration from: {MODEL_PATH}")
cfg = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
print(f"Config loaded in {time.time() - t1:.2f}s")
print(f"  Architectures : {cfg.architectures}")
print(f"  Hidden Size   : {cfg.hidden_size}")
print(f"  Routed Experts: {cfg.n_routed_experts} (top-{cfg.num_experts_per_tok} active)")
print(f"  Shared Experts: {cfg.n_shared_experts}")
print(f"  Layers        : {cfg.num_hidden_layers}")

# Test prompt tokenization
prompt = "def quicksort(arr):\n    \"\"\"Implement an in-place quicksort algorithm in Python.\"\"\"\n"
tokens = tokenizer.encode(prompt, return_tensors="pt")
print(f"\nPrompt: {repr(prompt)}")
print(f"Token count: {tokens.shape[-1]}")
print(f"Decoded: {repr(tokenizer.decode(tokens[0]))}")
print("Tokenizer and Model Configuration: PASS")
print("=" * 70)
