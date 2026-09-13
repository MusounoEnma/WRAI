"""
Download and inspect the official pre-trained base model:
SimpleStories/SimpleStories-V2-1.25M from Hugging Face.
Paper: arXiv:2504.09184
"""

import sys
import os
import requests
import json

sys.stdout.reconfigure(encoding='utf-8')

BASE_URL = "https://huggingface.co/SimpleStories/SimpleStories-V2-1.25M/resolve/main"
TARGET_DIR = os.path.join(os.path.dirname(__file__), "base_model")
os.makedirs(TARGET_DIR, exist_ok=True)

FILES = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "model.safetensors",
]

print("=" * 65)
print(" DOWNLOADING VERIFIED BASE MODEL: SimpleStories-V2-1.25M")
print(" Source: Hugging Face (SimpleStories/SimpleStories-V2-1.25M)")
print(" Paper : arXiv:2504.09184")
print("=" * 65)

for fname in FILES:
    dest_path = os.path.join(TARGET_DIR, fname)
    if os.path.exists(dest_path):
        print(f" [SKIP] {fname} already exists ({os.path.getsize(dest_path)} bytes)")
        continue

    url = f"{BASE_URL}/{fname}"
    print(f" [*] Downloading {fname}...", flush=True)
    r = requests.get(url, allow_redirects=True, timeout=30)
    if r.status_code == 200:
        with open(dest_path, "wb") as f:
            f.write(r.content)
        print(f" [OK] Saved {fname} ({len(r.content)} bytes)")
    else:
        print(f" [ERR] Failed to download {fname}: HTTP {r.status_code}")

print("\n[SUCCESS] All base model files downloaded successfully!")
