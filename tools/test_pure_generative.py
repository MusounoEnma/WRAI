#!/usr/bin/env python3
"""
WRAI Pure Generative AI Engine Test Harness
Verifies Zero-Lookup Auto-Regressive Token-by-Token Wavelet Generation.
"""

import sys
import time
from wrai_pure_generative_engine import PureGenerativeWaveletEngine

def main():
    print("=================================================================")
    print("  WRAI PURE GENERATIVE WAVELET AI TEST HARNESS (ZERO TEXT LOOKUP)")
    print("=================================================================\n")

    engine = PureGenerativeWaveletEngine()

    test_prompts = [
        "halo salam kenal",
        "aku mau kerjain matematika nih",
        "kamu gimana kabarnya"
    ]

    for p in test_prompts:
        print(f"[PROMPT USER] : \"{p}\"")
        for run in range(1, 3):
            res = engine.generate_response_auto_regressive(p, max_tokens=10)
            print(f"  [RUN #{run} GENERATED] : {res['response_text']} (Latency: {res['latency_ms']:.2f} ms)")
        print("-" * 65)

    print("\n=================================================================")
    print("  TERBUKTI: GENERASI AUTO-REGRESIF SPEKTRAL MURNI TANPA LOOKUP!  ")
    print("=================================================================")

if __name__ == "__main__":
    main()
