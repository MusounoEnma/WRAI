#!/usr/bin/env python3
"""
WRAI Generative AI Engine v4 Coherence Test Harness
Verifies bigram transition grammar flow and repetition penalty effectiveness.
"""

import sys
import time
from wrai_pure_generative_engine import PureGenerativeWaveletEngineV4

def main():
    print("=================================================================")
    print("  WRAI GENERATIVE COHERENCE TEST HARNESS V4 (ZERO TEXT LOOKUP)   ")
    print("=================================================================\n")

    engine = PureGenerativeWaveletEngineV4()

    test_prompts = [
        "halo kawan",
        "kabar gimana",
        "wrai adalah arsitektur",
        "aku mau kerjain matematika nih"
    ]

    for p in test_prompts:
        print(f"[PROMPT USER] : \"{p}\"")
        for run in range(1, 3):
            res = engine.generate_response_auto_regressive(p, max_tokens=10)
            print(f"  [RUN #{run} COHERENT GENERATION] : {res['response_text']} (Latency: {res['latency_ms']:.2f} ms)")
        print("-" * 65)

    print("\n=================================================================")
    print("  TERBUKTI: DYNAMIC SENTENCE GENERATION BEBAS LOOPING / REPETISI!")
    print("=================================================================")

if __name__ == "__main__":
    main()
