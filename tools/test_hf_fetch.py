#!/usr/bin/env python3
"""
Test fetching Orca-Math & Camel-Math reasoning datasets from HuggingFace
"""
import json
import urllib.request
import os
HF_TOKEN = os.environ.get("HF_TOKEN", "")

def fetch_hf_rows(dataset_name, config="default", split="train", limit=5):
    encoded_name = urllib.parse.quote(dataset_name, safe='')
    url = f"https://datasets-server.huggingface.co/rows?dataset={encoded_name}&config={config}&split={split}&offset=0&length={limit}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {HF_TOKEN}"})
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            rows = data.get("rows", [])
            print(f"[OK] Fetched {len(rows)} rows from {dataset_name}!")
            return [r.get("row", {}) for r in rows]
    except Exception as e:
        print(f"[ERROR] Failed {dataset_name}: {e}")
        return []

print("=== TESTING MATH REASONING DATASETS ===")
orca_rows = fetch_hf_rows("microsoft/orca-math-word-problems-200k")
if orca_rows:
    print(f"Sample Orca Math Q: {orca_rows[0].get('question')[:80]}...")
    print(f"Sample Orca Math A: {orca_rows[0].get('answer')[:80]}...\n")

camel_rows = fetch_hf_rows("camel-ai/math")
if camel_rows:
    print(f"Sample Camel Math Q: {camel_rows[0].get('message_1')[:80]}...")
    print(f"Sample Camel Math A: {camel_rows[0].get('message_2')[:80]}...\n")
