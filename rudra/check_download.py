#!/usr/bin/env python3
"""
check_download.py
=================
Live progress monitor for DeepSeek-Coder-V2 download in /home/palakm/MoEServingSim/aditya/models
"""

import os
import time

TARGET_DIR = "/home/palakm/MoEServingSim/aditya/models/DeepSeek-Coder-V2-Instruct"
TARGET_GB = 471.5

def get_dir_size_bytes(path):
    total = 0
    for root, dirs, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            try:
                total += os.path.getsize(fp)
            except Exception:
                pass
    return total

print("Measuring current download progress...")
s1 = get_dir_size_bytes(TARGET_DIR)
t1 = time.time()
time.sleep(3)
s2 = get_dir_size_bytes(TARGET_DIR)
t2 = time.time()

curr_gb = s2 / (1024**3)
dt = t2 - t1
speed_mb = (s2 - s1) / dt / (1024 * 1024) if dt > 0 else 0.0
pct = (curr_gb / TARGET_GB) * 100.0
rem_gb = max(0.0, TARGET_GB - curr_gb)
eta_min = (rem_gb * 1024) / speed_mb / 60.0 if speed_mb > 0 else 0.0

print("=" * 60)
print(f"  DEEPSEEK-CODER-V2 DOWNLOAD STATUS")
print("=" * 60)
print(f"Downloaded   : {curr_gb:6.2f} GB / {TARGET_GB:.1f} GB ({pct:5.1f}%)")
print(f"Current Speed: {speed_mb:6.2f} MB/s")
print(f"Remaining    : {rem_gb:6.2f} GB")
if speed_mb > 0:
    print(f"Estimated ETA: {eta_min:6.1f} minutes ({eta_min/60.0:.2f} hours)")
else:
    print(f"Estimated ETA: Calculating (waiting for next shard write)...")
print("=" * 60)
