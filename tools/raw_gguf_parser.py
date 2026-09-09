#!/usr/bin/env python3
"""
Raw GGUF Binary Reader & Tokenizer Vocabulary Extractor
Reads metadata directly from GGUF binary header without requiring GGML tensor type registration.
Handles type 36 (IQ2_S / BitNet ternary) seamlessly.
"""

import json
import os
import struct
import sys

def read_string(f):
    length_bytes = f.read(8)
    if not length_bytes or len(length_bytes) < 8:
        return ""
    length = struct.unpack("<Q", length_bytes)[0]
    return f.read(length).decode("utf-8", errors="replace")

def read_val(f, val_type):
    if val_type == 0:   # UINT8
        return struct.unpack("<B", f.read(1))[0]
    elif val_type == 1: # INT8
        return struct.unpack("<b", f.read(1))[0]
    elif val_type == 2: # UINT16
        return struct.unpack("<H", f.read(2))[0]
    elif val_type == 3: # INT16
        return struct.unpack("<h", f.read(2))[0]
    elif val_type == 4: # UINT32
        return struct.unpack("<I", f.read(4))[0]
    elif val_type == 5: # INT32
        return struct.unpack("<i", f.read(4))[0]
    elif val_type == 6: # FLOAT32
        return struct.unpack("<f", f.read(4))[0]
    elif val_type == 7: # BOOL
        return struct.unpack("<B", f.read(1))[0] != 0
    elif val_type == 8: # STRING
        return read_string(f)
    elif val_type == 9: # ARRAY
        item_type = struct.unpack("<I", f.read(4))[0]
        array_len = struct.unpack("<Q", f.read(8))[0]
        return [read_val(f, item_type) for _ in range(array_len)]
    elif val_type == 10: # UINT64
        return struct.unpack("<Q", f.read(8))[0]
    elif val_type == 11: # INT64
        return struct.unpack("<q", f.read(8))[0]
    elif val_type == 12: # FLOAT64
        return struct.unpack("<d", f.read(8))[0]
    else:
        raise ValueError(f"Unknown GGUF value type: {val_type}")

def parse_gguf_metadata(file_path):
    print(f"[*] Raw Parsing GGUF file: {os.path.abspath(file_path)}")
    with open(file_path, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            raise ValueError(f"Not a valid GGUF file: magic={magic}")
        version = struct.unpack("<I", f.read(4))[0]
        tensor_count = struct.unpack("<Q", f.read(8))[0]
        kv_count = struct.unpack("<Q", f.read(8))[0]

        print(f"  -> GGUF Version     : {version}")
        print(f"  -> Tensor Count     : {tensor_count}")
        print(f"  -> Metadata KV Pairs: {kv_count}")

        metadata = {}
        for i in range(kv_count):
            key = read_string(f)
            val_type = struct.unpack("<I", f.read(4))[0]
            val = read_val(f, val_type)
            metadata[key] = val

        return metadata

if __name__ == "__main__":
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    gguf_path = os.path.join(models_dir, "bitnet_b1_58_2b_4t.gguf")
    meta = parse_gguf_metadata(gguf_path)
    tokens = meta.get("tokenizer.ggml.tokens", [])
    print(f"\n[OK] Extracted {len(tokens):,} vocabulary tokens from Microsoft BitNet GGUF!")
    print(f"Sample tokens: {tokens[:20]}")
