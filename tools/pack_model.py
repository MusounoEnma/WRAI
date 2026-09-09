#!/usr/bin/env python3
"""
WRAI Binary Model Pack Generator
Packs synthetic spectral coefficient patterns into standard WRAI .bin model file format.
No external third-party dependencies required.
"""

import math
import os
import struct
import sys

MAGIC_HEADER = 0x57524149  # "WRAI"
FFT_SIZE = 512
SPECTRAL_BINS = 256
QUANT_BITS = 16

def float_to_q15(val: float) -> int:
    scaled = round(val * 32768.0)
    if scaled > 32767:
        return 32767
    if scaled < -32768:
        return -32768
    return scaled

def create_synthetic_pattern(pattern_id: int, dominant_bin: int) -> bytes:
    coeffs = []
    for bin_idx in range(SPECTRAL_BINS):
        # Synthetic Gaussian-like energy peak around dominant_bin
        dist = abs(bin_idx - dominant_bin)
        val = math.exp(-0.1 * (dist ** 2))
        coeffs.append(float_to_q15(val))

    # Format: uint16_t pattern_id, uint16_t reserved, 256 * int16_t coeffs
    header_part = struct.pack("<HH", pattern_id, 0)
    coeffs_part = struct.pack(f"<{SPECTRAL_BINS}h", *coeffs)
    return header_part + coeffs_part

def generate_binary_model(output_path: str, num_patterns: int = 10):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    pattern_bytes_len = 4 + (SPECTRAL_BINS * 2)  # 516 bytes

    # 64-byte Header
    header_format = "<I H H H H I I 44s"
    reserved_44 = b"\x00" * 44

    header_bytes = struct.pack(
        header_format,
        MAGIC_HEADER,
        0x0100,  # v1.0
        FFT_SIZE,
        SPECTRAL_BINS,
        QUANT_BITS,
        num_patterns,
        pattern_bytes_len,
        reserved_44
    )

    with open(output_path, "wb") as f:
        f.write(header_bytes)
        for p_id in range(1, num_patterns + 1):
            dom_bin = (p_id * 20) % SPECTRAL_BINS
            packet = create_synthetic_pattern(p_id, dom_bin)
            f.write(packet)

    print(f"[OK] Created WRAI binary model ({num_patterns} patterns) at: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    out_file = os.path.join(os.path.dirname(__file__), "..", "tests", "model_sample.bin")
    generate_binary_model(out_file)
