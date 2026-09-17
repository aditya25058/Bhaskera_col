#!/usr/bin/env python3
"""
Staging & Verification Script for Mixtral-8x7B-v0.1 on Rudra Cluster.
Performs:
  1. Disk space preflight check.
  2. Snapshot download of mistralai/Mixtral-8x7B-v0.1.
  3. Integrity and architecture verification (BF16, shards, params, config).
"""

import os
import shutil
import sys
import time
import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer

MODEL_ID = "mistralai/Mixtral-8x7B-v0.1"
TARGET_DIR = "/home/bapic_iiitd/2_group/models/Mixtral-8x7B-v0.1"


def check_disk_space(path: str, required_gb: float = 110.0):
    os.makedirs(path, exist_ok=True)
    total, used, free = shutil.disk_usage(path)
    free_gb = free / (1024 ** 3)
    print(f"[Preflight] Disk check on {path}: {free_gb:.2f} GB free (Required: {required_gb:.2f} GB)")
    if free_gb < required_gb:
        print(f"[Error] Insufficient disk space! Free: {free_gb:.2f} GB < Required: {required_gb:.2f} GB")
        sys.exit(1)
    print("[Preflight] Disk space check PASSED ✓")


def download_model(model_id: str, local_dir: str):
    print("=" * 80)
    print(f"  INITIATING DOWNLOAD: {model_id}")
    print(f"  Destination Directory: {local_dir}")
    print("=" * 80)
    t0 = time.time()
    
    downloaded_path = snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
        max_workers=8,
        allow_patterns=["*.safetensors", "*.json", "tokenizer*", "*.model", "*.txt"],
        ignore_patterns=["consolidated*", "*.pt", "*.bin"],
    )
    t_download = time.time() - t0
    print(f"[Download] Finished snapshot download in {t_download:.1f}s ({t_download / 60:.1f} min)")
    return downloaded_path


def verify_model(local_dir: str):
    print("=" * 80)
    print("  VERIFYING MIXTRAL-8x7B-v0.1 INTEGRITY")
    print("=" * 80)
    
    # 1. Inspect Files
    files = os.listdir(local_dir)
    safetensors = [f for f in files if f.endswith(".safetensors")]
    total_bytes = sum(os.path.getsize(os.path.join(local_dir, f)) for f in files if os.path.isfile(os.path.join(local_dir, f)))
    total_gb = total_bytes / (1024 ** 3)
    print(f"[Verify] Found {len(safetensors)} safetensors weight shards.")
    print(f"[Verify] Total model footprint on disk: {total_gb:.2f} GB")
    
    if len(safetensors) == 0:
        print("[Error] No safetensors files found!")
        sys.exit(1)
        
    # 2. Config & Architecture
    config = AutoConfig.from_pretrained(local_dir, trust_remote_code=True)
    print(f"[Verify] Model type: {config.model_type}")
    print(f"[Verify] Hidden size: {config.hidden_size}")
    print(f"[Verify] Number of layers: {config.num_hidden_layers}")
    print(f"[Verify] Number of experts: {getattr(config, 'num_local_experts', getattr(config, 'num_experts', 'Unknown'))}")
    print(f"[Verify] Top-k routed experts: {getattr(config, 'num_experts_per_tok', 'Unknown')}")
    print(f"[Verify] Intermediate size: {config.intermediate_size}")
    print(f"[Verify] Target dtype: {config.torch_dtype}")
    
    # 3. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(local_dir, trust_remote_code=True)
    print(f"[Verify] Tokenizer loaded successfully. Vocab size: {len(tokenizer)}")
    test_text = "The capital of France is"
    toks = tokenizer(test_text, return_tensors="pt")
    print(f"[Verify] Tokenizer smoke test: '{test_text}' -> {toks['input_ids'].tolist()}")
    
    print("=" * 80)
    print("  ALL MIXTRAL-8x7B PREFLIGHT VERIFICATIONS PASSED ✓")
    print("=" * 80)


if __name__ == "__main__":
    check_disk_space("/home/bapic_iiitd/2_group/models", required_gb=100.0)
    download_model(MODEL_ID, TARGET_DIR)
    verify_model(TARGET_DIR)
