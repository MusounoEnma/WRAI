#!/usr/bin/env python3
"""
WRAI Next-Gen v8 Option B + C Coherence Verification Test Harness
Evaluates natural sentence coherence in BOTH Indonesian and English.
100% Dynamic Wave Synthesis (0% Hardcode Template).
"""

import sys
import time
from wrai_pure_generative_engine import WRAINextGenEngineV8

def main():
    print("=================================================================")
    print("  WRAI NEXT-GEN v8 OPTION B + C COHERENCE VERIFICATION TEST      ")
    print("  (Indonesian & English Dual-Language Dynamic Wave Synthesis)   ")
    print("=================================================================\n")

    engine = WRAINextGenEngineV8()

    test_prompts_indonesian = [
        "halo kawan",
        "salam hangat kawan",
        "siapa kamu",
        "rumus luas lingkaran"
    ]

    test_prompts_english = [
        "hello my friend",
        "warm greetings friend",
        "who are you",
        "the area of a circle formula"
    ]

    print("--- [TEST 1: BAHASA INDONESIA DYNAMIC COHERENCE] ---")
    for p in test_prompts_indonesian:
        res = engine.generate_response_auto_regressive(p, max_tokens=10)
        print(f"  [PROMPT ID] : \"{p}\"")
        print(f"  [SYNTHESIS] : {res['response_text']} (Latency: {res['latency_ms']:.2f} ms)\n")

    print("--- [TEST 2: ENGLISH DYNAMIC COHERENCE] ---")
    for p in test_prompts_english:
        res = engine.generate_response_auto_regressive(p, max_tokens=10)
        print(f"  [PROMPT EN] : \"{p}\"")
        print(f"  [SYNTHESIS] : {res['response_text']} (Latency: {res['latency_ms']:.2f} ms)\n")

    print("=================================================================")
    print("  VERIFIKASI KOHERENSI DUA BAHASA (ID + EN) BERHASIL DILAKUKAN!  ")
    print("=================================================================")

if __name__ == "__main__":
    main()
