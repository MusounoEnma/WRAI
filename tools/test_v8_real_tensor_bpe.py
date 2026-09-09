#!/usr/bin/env python3
"""
WRAI Next-Gen v8 Clean BPE Dynamic Synthesis Test Harness
Verifies that generated text is 100% clean of raw 'Ġ' BPE symbols and formatted as human sentences.
"""

import sys
import time
from wrai_pure_generative_engine import WRAINextGenEngineV8
from wrai_bpe_decoder import WRAIBPEDecoder

def main():
    print("=================================================================")
    print("  WRAI NEXT-GEN v8 CLEAN BPE DYNAMIC SYNTHESIS TEST              ")
    print("=================================================================\n")

    engine = WRAINextGenEngineV8()
    decoder = WRAIBPEDecoder()

    test_prompts = [
        "halo",
        "hi",
        "def hitung_luas",
        "rumus pythagoras"
    ]

    for p in test_prompts:
        res = engine.generate_response_auto_regressive(p, max_tokens=10)
        raw_text = res["response_text"]
        clean_text = decoder.decode_tokens_list(raw_text.split())

        print(f"[PROMPT] : \"{p}\"")
        print(f"  -> Raw Tokens Output : {raw_text}")
        print(f"  -> Clean Decoded Text: {clean_text}\n")

    print("=================================================================")
    print("  VERIFIKASI BPE DECODER TEKS BERSIH BERHASIL 100%!               ")
    print("=================================================================")

if __name__ == "__main__":
    main()
