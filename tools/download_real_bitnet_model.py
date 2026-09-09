#!/usr/bin/env python3
"""
WRAI Real Model Weight Downloader (Official Microsoft BitNet b1.58 HuggingFace Repo)
Target: microsoft/bitnet-b1.58-2B-4T-gguf or microsoft/bitnet-b1.58-2B-4T
"""

import os
import sys
import urllib.request

def download_file(url, target_path):
    print(f"[*] Downloading from {url} ...")
    print(f"[*] Target location: {os.path.abspath(target_path)}")
    
    def report_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = (downloaded / total_size) * 100.0
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(f"\r  [PROGRESS] {mb_downloaded:.1f} MB / {mb_total:.1f} MB ({percent:.1f}%)")
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, target_path, reporthook=report_progress)
        print(f"\n[OK] Download finished: {target_path}")
        return True
    except Exception as e:
        print(f"\n[ERROR] Download failed: {e}")
        return False

def main():
    print("=================================================================")
    print("  WRAI REAL BITNET b1.58 MODEL WEIGHT DOWNLOADER               ")
    print("  Source: microsoft/bitnet-b1.58-2B-4T-gguf                      ")
    print("=================================================================\n")

    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(models_dir, exist_ok=True)

    # 1. Download official GGUF model file (~1.2 GB)
    gguf_url = "https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf/resolve/main/ggml-model-i2_s.gguf"
    gguf_target = os.path.join(models_dir, "bitnet_b1_58_2b_4t.gguf")

    # Fallback/Alternative GGUF direct URL
    alt_gguf_url = "https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf/resolve/main/ggml-model-q4_0.gguf"

    print("[STEP 1] Downloading Official Microsoft BitNet b1.58 GGUF Weights...")
    success = download_file(gguf_url, gguf_target)
    if not success:
        print("\n[*] Trying alternative BitNet GGUF URL...")
        download_file(alt_gguf_url, gguf_target)

if __name__ == "__main__":
    main()
