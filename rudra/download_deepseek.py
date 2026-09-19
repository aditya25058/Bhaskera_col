#!/usr/bin/env python3
"""
download_deepseek.py
====================
Robust, multi-threaded background downloader for DeepSeek-Coder-V2-Instruct (471.5 GB)
Target: /home/palakm/MoEServingSim/aditya/models/DeepSeek-Coder-V2-Instruct
"""

import os
import sys
import time
from huggingface_hub import snapshot_download

REPO_ID = "deepseek-ai/DeepSeek-Coder-V2-Instruct"
LOCAL_DIR = "/home/palakm/MoEServingSim/aditya/models/DeepSeek-Coder-V2-Instruct"

os.makedirs(LOCAL_DIR, exist_ok=True)

print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting multi-threaded download for {REPO_ID}...")
print(f"Destination: {LOCAL_DIR}")

max_retries = 20
retry_delay = 15

for attempt in range(1, max_retries + 1):
    try:
        t0 = time.time()
        snapshot_download(
            repo_id=REPO_ID,
            local_dir=LOCAL_DIR,
            max_workers=16,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        elapsed = time.time() - t0
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Download completed successfully in {elapsed:.1f}s!")
        break
    except Exception as e:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Attempt {attempt}/{max_retries} encountered error: {e}")
        if attempt < max_retries:
            print(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
        else:
            print("Maximum retries reached. Download failed.")
            sys.exit(1)
