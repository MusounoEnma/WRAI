#!/usr/bin/env python3
"""
Test direct HF Rows API to inspect FreedomIntelligence dataset keys
"""
import json
import urllib.request
import os

HF_TOKEN = os.environ.get("HF_TOKEN", "")

def inspect_hf_keys(dataset_name):
    encoded_name = urllib.parse.quote(dataset_name, safe='')
    url = f"https://datasets-server.huggingface.co/rows?dataset={encoded_name}&config=default&split=train&offset=0&length=2"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {HF_TOKEN}"})
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            rows = data.get("rows", [])
            if rows:
                row_keys = list(rows[0].get("row", {}).keys())
                print(f"[OK] {dataset_name} keys: {row_keys}")
                print(f"     Sample row: {rows[0].get('row')}\n")
            else:
                print(f"[NOTE] {dataset_name} returned 0 rows.")
    except Exception as e:
        print(f"[ERROR] {dataset_name}: {e}")

print("=== TESTING DIRECT HF ROWS API KEYS ===")
inspect_hf_keys("FreedomIntelligence/alpaca-gpt4-indonesian")
inspect_hf_keys("cahya/alpaca-id-cleaned")
inspect_hf_keys("iamtarun/python_code_instructions_18k_alpaca")
inspect_hf_keys("microsoft/orca-math-word-problems-200k")
