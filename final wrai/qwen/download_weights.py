"""
WRAI-X (0.6B) Weight Downloader & Setup Helper
Host weights on Hugging Face Model Hub and download automatically to this folder.
"""

import os
import sys
import argparse

DEFAULT_REPO = "YourUsername/wrai-x-06b"  # Ganti dengan Hugging Face repository Anda
FILES_TO_DOWNLOAD = [
    "wrai_x_06b_int8.bin",
    "wrai_x_06b_transplanted.pt"
]

def main():
    parser = argparse.ArgumentParser(description="Download WRAI-X weights from Hugging Face")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="Hugging Face repository ID (e.g. username/repo-name)")
    args = parser.parse_args()

    target_dir = os.path.dirname(os.path.abspath(__file__))
    print("=" * 65)
    print(" 🌊 WRAI-X (0.6B) WEIGHT DOWNLOAD HELPER")
    print("=" * 65)
    print(f" Target Directory: {target_dir}")
    print(f" Source Repo     : https://huggingface.co/{args.repo}")
    print("=" * 65)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[!] huggingface_hub belum terinstall. Menginstall via pip...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
        from huggingface_hub import hf_hub_download

    for filename in FILES_TO_DOWNLOAD:
        local_path = os.path.join(target_dir, filename)
        if os.path.exists(local_path):
            print(f"[OK] {filename} sudah ada ({os.path.getsize(local_path) / (1024*1024):.2f} MB). Lewati.")
            continue

        print(f"[*] Mendownload {filename}...")
        try:
            downloaded = hf_hub_download(
                repo_id=args.repo,
                filename=filename,
                local_dir=target_dir,
                local_dir_use_symlinks=False
            )
            print(f"[OK] {filename} berhasil diunduh!")
        except Exception as e:
            print(f"[ERROR] Gagal mendownload {filename}: {e}")
            print("Silakan pastikan repo Hugging Face sudah dibuat dan file sudah diunggah.")

    print("\n[SELESAI] Semua bobot siap digunakan!")

if __name__ == "__main__":
    main()
