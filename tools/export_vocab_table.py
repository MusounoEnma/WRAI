#!/usr/bin/env python3
"""
Exports the 32,000 pruned token string table to binary format:
Format:
  uint32_t num_tokens (32000)
  For each token:
    uint16_t str_len
    char str_bytes[str_len] (UTF-8)
"""
import json
import struct
from transformers import AutoTokenizer

VOCAB_MAP_PATH = "models/pruned_vocab_map_v14_4.json"
OUTPUT_VOCAB_BIN = "models/wrai_v14_4_vocab.bin"

print("[*] Loading Qwen Tokenizer...", flush=True)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", trust_remote_code=True)

with open(VOCAB_MAP_PATH, "r", encoding="utf-8") as f:
    vocab_map = json.load(f)

pruned_to_teacher = vocab_map["pruned_id_to_teacher_id"]
num_tokens = len(pruned_to_teacher)
print(f"[OK] Pruned Vocab Size: {num_tokens}", flush=True)

with open(OUTPUT_VOCAB_BIN, "wb") as f:
    f.write(struct.pack("<I", num_tokens))
    for pid, tid in enumerate(pruned_to_teacher):
        # Decode single token ID
        token_str = tokenizer.decode([tid], skip_special_tokens=False)
        token_bytes = token_str.encode("utf-8")
        str_len = len(token_bytes)
        f.write(struct.pack("<H", str_len))
        f.write(token_bytes)

print(f"[OK 100% SUCCESS] Exported Token Table to: {OUTPUT_VOCAB_BIN}", flush=True)
