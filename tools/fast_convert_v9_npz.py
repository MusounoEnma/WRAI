#!/usr/bin/env python3
"""
Fast JSON -> NPZ converter for WRAI v9 Revolution
"""
import json
import numpy as np
import os
import time

models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
bin_path = os.path.join(models_dir, "wrai_v9_best.bin")
json_path = os.path.join(models_dir, "wrai_v9_best.json")
npz_path = os.path.join(models_dir, "wrai_v9_best.npz")

print("==========================================================")
print("  FAST CONVERTING WRAI v9 JSON TO NPZ BINARY CACHE        ")
print("==========================================================")

t0 = time.perf_counter()
print(f"[*] Reading {json_path} (618 MB)...")
with open(json_path, "r", encoding="utf-8") as f:
    meta = json.load(f)

print("[*] Extracting matrices...")
npz_data = {
    "W_emb": np.array(meta["W_emb"], dtype=np.float32),
    "W_out": np.array(meta["W_out"], dtype=np.float32),
    "b_out": np.array(meta["b_out"], dtype=np.float32),
    "sc1_real": np.array(meta["spectral_conv1"]["W_freq_real"], dtype=np.float32),
    "sc1_imag": np.array(meta["spectral_conv1"]["W_freq_imag"], dtype=np.float32),
    "sc2_real": np.array(meta["spectral_conv2"]["W_freq_real"], dtype=np.float32),
    "sc2_imag": np.array(meta["spectral_conv2"]["W_freq_imag"], dtype=np.float32),
}

gru_w = meta["gru_weights"]
for l in range(meta.get("num_layers", 4)):
    npz_data[f"W_ih_l{l}"] = np.array(gru_w[f"W_ih_l{l}"], dtype=np.float32)
    npz_data[f"b_ih_l{l}"] = np.array(gru_w[f"b_ih_l{l}"], dtype=np.float32)
    npz_data[f"W_hh_l{l}"] = np.array(gru_w[f"W_hh_l{l}"], dtype=np.float32)
    npz_data[f"b_hh_l{l}"] = np.array(gru_w[f"b_hh_l{l}"], dtype=np.float32)

meta_info = {
    "vocabulary": meta["vocabulary"],
    "id_to_word": meta["id_to_word"],
    "vocab_size": meta["vocab_size"],
    "num_layers": meta.get("num_layers", 4),
    "hidden_dim": meta.get("hidden_dim", 512),
    "model_type": meta.get("model_type", "ZeroGEMMWaveletAIV9")
}

print("[*] Saving meta_vocab.json and wrai_v9_best.npz...")
with open(os.path.join(models_dir, "meta_vocab.json"), "w", encoding="utf-8") as f_meta:
    json.dump(meta_info, f_meta, ensure_ascii=False)

np.savez_compressed(npz_path, **npz_data)
t1 = time.perf_counter()
print(f"[SUCCESS] WRAI v9 NPZ Binary Cache Created in {(t1-t0):.2f}s! ({os.path.getsize(npz_path)/(1024*1024):.2f} MB)")
