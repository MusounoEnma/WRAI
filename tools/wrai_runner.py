#!/usr/bin/env python3
"""
WRAI Standalone Interactive AI REPL & Inference Engine
Runs interactive WRAI non-transformer AI session in real-time.
No external dependencies required.
"""

import json
import math
import os
import struct
import sys
import time

LUT_SIZE = 512
FFT_SIZE = 512
SPECTRAL_BINS = 256
MAGIC_HEADER = 0x57524149

def fnv1a(s: str) -> int:
    h = 2166136261
    for char in s.encode('utf-8'):
        h ^= char
        h = (h * 16777619) & 0xFFFFFFFF
    return h

def float_to_q15(v: float) -> int:
    val = round(v * 32768.0)
    return max(-32768, min(32767, val))

def q15_mul(a: int, b: int) -> int:
    return (a * b) >> 15

SIN_LUT = [float_to_q15(math.sin(2.0 * math.pi * i / LUT_SIZE)) for i in range(LUT_SIZE)]
COS_LUT = [float_to_q15(math.cos(2.0 * math.pi * i / LUT_SIZE)) for i in range(LUT_SIZE)]

def bit_reversal(r, i, n):
    j = 0
    for idx in range(n - 1):
        if idx < j:
            r[idx], r[j] = r[j], r[idx]
            i[idx], i[j] = i[j], i[idx]
        k = n >> 1
        while k <= j:
            j -= k
            k >>= 1
        j += k

def fft_radix2_q15(r, i, n):
    bit_reversal(r, i, n)
    length = 2
    while length <= n:
        half = length >> 1
        step = LUT_SIZE // length
        for k in range(0, n, length):
            for j in range(half):
                lut_idx = (j * step) & (LUT_SIZE - 1)
                cos_v = COS_LUT[lut_idx]
                sin_v = SIN_LUT[lut_idx]

                u_r = r[k + j] >> 1
                u_i = i[k + j] >> 1
                v_r = r[k + j + half] >> 1
                v_i = i[k + j + half] >> 1

                tw_r = q15_mul(v_r, cos_v) + q15_mul(v_i, sin_v)
                tw_i = q15_mul(v_i, cos_v) - q15_mul(v_r, sin_v)

                r[k + j] = u_r + tw_r
                i[k + j] = u_i + tw_i
                r[k + j + half] = u_r - tw_r
                i[k + j + half] = u_i - tw_i
        length <<= 1

class WRAIEngine:
    def __init__(self, model_bin_path: str, responses_json_path: str):
        self.model_bin_path = model_bin_path
        self.responses_json_path = responses_json_path

        with open(responses_json_path, "r", encoding="utf-8") as f:
            self.responses_map = json.load(f)

        # Validate header
        with open(model_bin_path, "rb") as f:
            header_bytes = f.read(64)
            magic, ver, fft_sz, spec_bins, quant_b, num_pat, pat_bytes = struct.unpack("<I H H H H I I", header_bytes[:20])
            if magic != MAGIC_HEADER:
                raise ValueError("Invalid WRAI binary header magic!")
            self.num_patterns = num_pat
            self.pattern_bytes = pat_bytes

    def infer(self, prompt_text: str):
        t0 = time.perf_counter()

        words = prompt_text.lower().replace("?", "").replace("!", "").replace(",", "").split()
        r = [0] * FFT_SIZE
        i = [0] * FFT_SIZE

        if words:
            num_words = len(words)
            scale = 0
            temp = num_words
            while temp > 1:
                scale += 1
                temp >>= 1

            for n in range(FFT_SIZE):
                acc = 0
                for w in words:
                    h = fnv1a(w)
                    freq = (h % (SPECTRAL_BINS - 2)) + 1
                    phase = (h >> 8) % LUT_SIZE
                    lut_idx = (freq * n + phase) & (LUT_SIZE - 1)
                    acc += SIN_LUT[lut_idx]
                r[n] = acc >> scale

        # Execute Radix-2 FFT
        t_fft_start = time.perf_counter()
        fft_radix2_q15(r, i, FFT_SIZE)
        t_fft_end = time.perf_counter()

        # Compute Magnitudes
        mags = []
        for m in range(SPECTRAL_BINS):
            r_abs = abs(r[m])
            i_abs = abs(i[m])
            mag = max(r_abs, i_abs) + ((3 * min(r_abs, i_abs)) >> 3)
            mags.append(mag)

        # Stream binary file & Resonance Matching
        best_score = -2147483647
        winning_id = 0

        with open(self.model_bin_path, "rb") as f:
            f.seek(64) # skip header
            for _ in range(self.num_patterns):
                pkt_data = f.read(self.pattern_bytes)
                p_id, _ = struct.unpack("<HH", pkt_data[:4])
                coeffs = struct.unpack(f"<{SPECTRAL_BINS}h", pkt_data[4:])

                score = 0
                for m in range(SPECTRAL_BINS):
                    score += q15_mul(mags[m], coeffs[m])

                if score > best_score:
                    best_score = score
                    winning_id = p_id

        t1 = time.perf_counter()

        resp_info = self.responses_map.get(str(winning_id), {"response": "Resonansi tidak ditemukan."})

        return {
            "prompt": prompt_text,
            "winning_id": winning_id,
            "resonance_score": best_score,
            "response_text": resp_info["response"],
            "fft_time_ms": (t_fft_end - t_fft_start) * 1000.0,
            "total_latency_ms": (t1 - t0) * 1000.0
        }

def main():
    print("=================================================================")
    print("      WRAI (Wavelet-Resonance AI) Bare-Metal Inference Engine    ")
    print("=================================================================")
    print("  Zero-GEMM | Integer Q15/Q31 Fixed-Point | Radix-2 Cooley-Tukey")
    print("=================================================================\n")

    bin_path = os.path.join(os.path.dirname(__file__), "..", "models", "wrai_brain.bin")
    json_path = os.path.join(os.path.dirname(__file__), "..", "models", "wrai_responses.json")

    if not os.path.exists(bin_path):
        print("Error: Model biner belum dibuat. Jalankan `python tools/train_wrai_model.py` terlebih dahulu.")
        sys.exit(1)

    engine = WRAIEngine(bin_path, json_path)

    sample_prompts = [
        "Siapa kamu?",
        "Bagaimana cara kerja WRAI?",
        "Apakah WRAI menggunakan GEMM perkalian matriks?",
        "Di mana model disimpan?",
        "Apa target hardware dari WRAI?"
    ]

    print("--- MENJALANKAN DEMO INFERENSI WRAI AUTOMATIS ---")
    for prompt in sample_prompts:
        res = engine.infer(prompt)
        print(f"\n[USER INPUT] : '{res['prompt']}'")
        print(f"[RESONANCE]  : Pattern ID #{res['winning_id']} | Skor Resonansi: {res['resonance_score']}")
        print(f"[LATENCY]    : FFT: {res['fft_time_ms']:.3f} ms | Total Pipeline: {res['total_latency_ms']:.3f} ms")
        print(f"[WRAI OUTPUT]: \"{res['response_text']}\"")
        print("-" * 65)

    print("\n=================================================================")
    print("  WRAI RUNTIME AI SIAP DUGUNAKAN PADA HARDWARE EMBEDDED / AVX1!  ")
    print("=================================================================")

if __name__ == "__main__":
    main()
