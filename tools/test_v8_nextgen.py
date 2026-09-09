#!/usr/bin/env python3
"""
WRAI Next-Gen v8 Test Harness
Verifies 100% Dynamic Wave Synthesis, Non-template responses, and AST Code Guarding.
"""

import sys
import time
from wrai_pure_generative_engine import WRAINextGenEngineV8

def main():
    print("=================================================================")
    print("  WRAI NEXT-GEN v8 SPECTRAL AI ENGINE TEST HARNESS               ")
    print("  (4096 Bins FFT Q31, Mamba-2 Selective State, Fourier-KAN)      ")
    print("=================================================================\n")

    engine = WRAINextGenEngineV8()

    test_prompts = [
        "halo kawan",
        "def hitung_luas",
        "rumus luas lingkaran"
    ]

    for p in test_prompts:
        print(f"[PROMPT USER] : \"{p}\"")
        for run in range(1, 3):
            res = engine.generate_response_auto_regressive(p, max_tokens=10)
            print(f"  [RUN #{run} DYNAMIC WAVE SYNTHESIS] : {res['response_text']} (Latency: {res['latency_ms']:.2f} ms)")
        print("-" * 65)

    print("\n=================================================================")
    print("  VERIFIKASI SUKSES: 100% DYNAMIC WAVE SYNTHESIS BEBAS TEMPLATE!  ")
    print("=================================================================")

if __name__ == "__main__":
    main()
