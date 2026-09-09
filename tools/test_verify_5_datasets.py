#!/usr/bin/env python3
"""
Verify all HuggingFace dataset URLs and row structures
"""
import json
import urllib.request
import os

HF_TOKEN = os.environ.get("HF_TOKEN", "")

datasets_to_check = [
    ("FreedomIntelligence/alpaca-gpt4-indonesian", "default"),
    ("cahya/alpaca-id-cleaned", "default"),
    ("iamtarun/python_code_instructions_18k_alpaca", "default"),
    ("microsoft/orca-math-word-problems-200k", "default"),
    ("juppy/alpaca-indonesian", "default")
]

print("=== VERIFYING HUGGINGFACE DATASET ACCESSIBILITY ===")
for ds_name, config in datasets_to_check:
    encoded_name = urllib.parse.quote(ds_name, safe='')
    url = f"https://datasets-server.huggingface.co/rows?dataset={encoded_name}&config={config}&split=train&offset=0&length=2"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {HF_TOKEN}"})
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            rows = data.get("rows", [])
            if rows:
                row_keys = list(rows[0].get("row", {}).keys())
                print(f"[ACCESSIBLE] {ds_name} -> Keys: {row_keys}")
            else:
                print(f"[EMPTY] {ds_name} returned 0 rows.")
    except Exception as e:
        print(f"[ERROR/404] {ds_name}: {e}")
