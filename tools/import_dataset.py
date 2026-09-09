#!/usr/bin/env python3
"""
WRAI Dataset Importer & Knowledge Model Synthesizer Tool
Converts custom JSON / CSV / Text knowledge datasets into WRAI Wavelet Spectral Models (.bin).
No third-party dependencies required.
"""

import csv
import json
import math
import os
import struct
import sys

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

def compute_spectral_signature(prompt_text: str):
    words = prompt_text.lower().replace("?", "").replace("!", "").replace(",", "").split()
    r = [0] * FFT_SIZE
    i = [0] * FFT_SIZE

    if not words:
        return [0] * SPECTRAL_BINS

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

    fft_radix2_q15(r, i, FFT_SIZE)

    mags = []
    for m in range(SPECTRAL_BINS):
        r_abs = abs(r[m])
        i_abs = abs(i[m])
        mag = max(r_abs, i_abs) + ((3 * min(r_abs, i_abs)) >> 3)
        mags.append(mag)
    return mags

def import_knowledge_items(items_list, bin_output_path, json_output_path):
    os.makedirs(os.path.dirname(os.path.abspath(bin_output_path)), exist_ok=True)
    
    num_patterns = len(items_list)
    pattern_bytes_len = 4 + (SPECTRAL_BINS * 2)  # 516 bytes

    header_bytes = struct.pack(
        "<I H H H H I I 44s",
        MAGIC_HEADER,
        0x0100,
        FFT_SIZE,
        SPECTRAL_BINS,
        16, # Q15
        num_patterns,
        pattern_bytes_len,
        b"\x00" * 44
    )

    responses_dict = {}

    with open(bin_output_path, "wb") as f_bin:
        f_bin.write(header_bytes)

        for idx, item in enumerate(items_list, start=1):
            keywords = item.get("keywords") or item.get("question") or item.get("input", "")
            resp = item.get("response") or item.get("answer") or item.get("output", "")

            mags = compute_spectral_signature(keywords)
            pkt_header = struct.pack("<HH", idx, 0)
            pkt_coeffs = struct.pack(f"<{SPECTRAL_BINS}h", *mags)
            f_bin.write(pkt_header + pkt_coeffs)

            responses_dict[str(idx)] = {
                "keywords": keywords,
                "response": resp
            }
            print(f"  [+] Ingested Pattern #{idx}: '{keywords}' -> Peak Mag: {max(mags)}")

    with open(json_output_path, "w", encoding="utf-8") as f_json:
        json.dump(responses_dict, f_json, indent=2, ensure_ascii=False)

    print(f"\n[SUCCESS] WRAI Knowledge Model Successfully Exported:")
    print(f"  -> Total Patterns Ingested : {num_patterns}")
    print(f"  -> Model Binary File       : {os.path.abspath(bin_output_path)} ({os.path.getsize(bin_output_path)} Bytes)")
    print(f"  -> Response Decoder Map    : {os.path.abspath(json_output_path)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python import_dataset.py <dataset.json|dataset.csv>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_bin = os.path.join(os.path.dirname(__file__), "..", "models", "wrai_brain.bin")
    output_json = os.path.join(os.path.dirname(__file__), "..", "models", "wrai_responses.json")

    items = []
    if input_file.endswith(".json"):
        with open(input_file, "r", encoding="utf-8") as f:
            items = json.load(f)
    elif input_file.endswith(".csv"):
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                items.append(row)

    import_knowledge_items(items, output_bin, output_json)
