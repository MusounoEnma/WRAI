#!/usr/bin/env python3
"""
Fast NPZ Caching & Honest Test for WRAI v8 Model
"""
import json
import numpy as np
import os
import time

models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
bin_path = os.path.join(models_dir, "wrai_nextgen_v8_best.bin")
json_path = os.path.join(models_dir, "wrai_nextgen_v8_best.json")
npz_path = os.path.join(models_dir, "wrai_nextgen_v8_best.npz")

print("[*] Checking model files...")
if not os.path.exists(npz_path):
    print(f"[*] Converting 618 MB JSON ({json_path}) to fast NPZ binary format for instant loading...")
    t0 = time.perf_counter()
    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    
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

    # Save meta info
    meta_info = {
        "vocabulary": meta["vocabulary"],
        "id_to_word": meta["id_to_word"],
        "vocab_size": meta["vocab_size"],
        "num_layers": meta.get("num_layers", 4),
        "hidden_dim": meta.get("hidden_dim", 512),
        "model_type": meta.get("model_type", "ZeroGEMMWaveletAI")
    }
    with open(os.path.join(models_dir, "meta_vocab.json"), "w", encoding="utf-8") as f_meta:
        json.dump(meta_info, f_meta, ensure_ascii=False)

    np.savez_compressed(npz_path, **npz_data)
    t1 = time.perf_counter()
    print(f"[OK] Saved NPZ binary cache in {(t1-t0):.2f}s! (Size: {os.path.getsize(npz_path)/(1024*1024):.2f} MB)")

# Now update engine to load NPZ cache instantly!
from wrai_pure_generative_engine import WRAIPureGenerativeEngine
t0 = time.perf_counter()
engine = WRAIPureGenerativeEngine()
t1 = time.perf_counter()
print(f"[*] Engine Ready in {(t1-t0):.2f}s!")

test_prompts = [
    "halo kawan",
    "siapa kamu",
    "berikan tips cara hidup sehat",
    "apa rumus luas lingkaran",
    "buatlah fungsi python sederhana untuk menghitung jumlah list",
    "hello my friend"
]

print("\n--- HONEST EVALUATION OF WRAI v8 (88K DATASET MODEL) ---")
for p in test_prompts:
    t_start = time.perf_counter()
    resp = engine.generate(p, max_tokens=30, temperature=0.1)
    t_end = time.perf_counter()
    print(f"Prompt  : {p}")
    print(f"Response: {resp}")
    print(f"Latency : {(t_end - t_start)*1000.0:.2f} ms\n")
