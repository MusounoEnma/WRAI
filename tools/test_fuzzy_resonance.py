#!/usr/bin/env python3
"""
WRAI Fuzzy / Unseen Sentence Resonance Verification Tool
Demonstrates that WRAI matches semantic intent via Spectral FFT Resonance,
NOT hardcoded string matching.
"""

import os
import sys
from wrai_runner import WRAIEngine

def main():
    bin_path = os.path.join(os.path.dirname(__file__), "..", "models", "wrai_brain.bin")
    json_path = os.path.join(os.path.dirname(__file__), "..", "models", "wrai_responses.json")

    engine = WRAIEngine(bin_path, json_path)

    # Completely NEW variations that NEVER existed in the training dataset!
    unseen_variations = [
        "Permisi bro, kamu ini siapa sih dan dipanggil apa?",
        "Tolong jelaskan dong mekanisme kerja sinyal gelombang spektral di sistem ini",
        "Apakah ada komputasi perkalian matriks GEMM desimal floating point yang berat?",
        "Di mana sih file data biner dan memori dma ring buffer dialokasikan?",
        "Hardware apa saja yang bisa menjalankan arsitektur ini? Apakah esp32 atau stm32 bisa?"
    ]

    print("=================================================================")
    print("   UJI KOEFISIEN SPEKTRAL WRAI PADA KALIMAT BEBAS / NON-HARDCODE ")
    print("=================================================================\n")

    for text in unseen_variations:
        res = engine.infer(text)
        print(f"[KALIMAT BARU (UNSEEN)] : \"{res['prompt']}\"")
        print(f"  -> Pattern ID Terdeteksi: #{res['winning_id']}")
        print(f"  -> Skor Resonansi FFT   : {res['resonance_score']}")
        print(f"  -> Hasil Inferensi WRAI : \"{res['response_text']}\"")
        print("-" * 65)

if __name__ == "__main__":
    main()
