#!/usr/bin/env python3
"""
Test Live WRAI v8 Generative Output Honestly
"""
import time
from wrai_pure_generative_engine import WRAIPureGenerativeEngine

print("==========================================================")
print("  LOADING WRAI v8 ZERO-GEMM WAVELET AI (88K DATASET MODEL)")
print("==========================================================")

t0 = time.perf_counter()
engine = WRAIPureGenerativeEngine()
t1 = time.perf_counter()
print(f"[*] Engine loaded in {(t1-t0):.2f} seconds!\n")

test_prompts = [
    "halo kawan",
    "siapa kamu",
    "berikan tips cara hidup sehat",
    "apa rumus luas lingkaran",
    "buatlah fungsi python sederhana untuk menghitung jumlah list",
    "what is the capital of Indonesia"
]

print("--- EVALUASI JUJUR INFERENSI MODUL WRAI v8 ---")
for p in test_prompts:
    t_start = time.perf_counter()
    response = engine.generate(p, max_tokens=30, temperature=0.1)
    t_end = time.perf_counter()
    latency_ms = (t_end - t_start) * 1000.0
    print(f"Prompt  : '{p}'")
    print(f"Response: '{response}'")
    print(f"Latency : {latency_ms:.2f} ms\n")
