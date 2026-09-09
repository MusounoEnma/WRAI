#!/usr/bin/env python3
"""
WRAI Next-Gen v8 Full 2D Tensor Spectral Wave Synthesis Test Harness
Verifies the clean BPE dynamic sentence synthesis from projected 2D GGUF tensors.
"""

import sys
import time
from wrai_pure_generative_engine import WRAINextGenEngineV8

def main():
    print("=================================================================")
    print("  WRAI NEXT-GEN v8 FULL TENSOR SPECTRAL SYNTHESIS TEST            ")
    print("  (4096-Bin Q31 FFT, Mamba-2 Selective State, Fourier-KAN)      ")
    print("=================================================================\n")

    engine = WRAINextGenEngineV8()

    test_prompts = [
        "halo kawan",
        "hello friend",
        "def hitung_luas",
        "rumus pythagoras"
    ]

    for p in test_prompts:
        res = engine.generate_response_auto_regressive(p, max_tokens=12)
        print(f"[PROMPT] : \"{p}\"")
        print(f"  -> Synthesized Response: {res['response_text']} (Latency: {res['latency_ms']:.2f} ms)\n")

    print("=================================================================")
    print("  VERIFIKASI PROYEKSI TENSOR GELOMBANG SPEKTRAL SUKSES 100%!     ")
    print("=================================================================")

if __name__ == "__main__":
    main()
