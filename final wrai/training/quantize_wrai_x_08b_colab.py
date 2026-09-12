"""
=============================================================================
 🚀 WRAI-X (0.8B) DIRECT QUANTIZER & NATIVE C BINARY PACKER
=============================================================================
 Spesifikasi Target:
  - Dimensi: D=1024, FFN=3072, 28 Layers, 16 Heads, Head Dim=128, Vocab=151936
  - Kuantisasi INT8 Row-wise (Symmetric)
  - Ukuran Output Binary: ~540 MB (Sangat ramping, RAM 600 MB di laptop AMD A8)
  - Ekspor Otomatis Tokenizer Binary (wrai_x_vocab.bin)
=============================================================================
"""

import os
import gc
import sys
import time
import math
import struct
import numpy as np
import torch
from transformers import AutoTokenizer

QUANT_MODE = "int8"
DRIVE_DIR = "/content/drive/MyDrive/WRAI_X_08B_Models"
LOCAL_FALLBACK_DIR = "."

OUTPUT_BIN_NAME = f"wrai_x_08b_{QUANT_MODE}.bin"
OUTPUT_VOCAB_NAME = "wrai_x_vocab.bin"
SOURCE_MODEL_NAME = "Qwen/Qwen3-0.8B"

# Header WRAI-X
MAGIC_HEADER = 0x57524149  # "WRAI"
VERSION = 170              # WRAI-X (v17.0)
NUM_LAYERS = 28
HIDDEN_DIM = 1024
FFN_INTERMEDIATE_DIM = 3072
NUM_HEADS = 16
HEAD_DIM = 128
WAVELET_LEVELS = 4
MAX_SEQ_LEN = 512
VOCAB_SIZE = 151936

def quantize_rowwise_int8(tensor):
    if torch.is_tensor(tensor):
        arr = tensor.detach().cpu().to(torch.float32).numpy()
    else:
        arr = np.asarray(tensor, dtype=np.float32)

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    max_abs = np.max(np.abs(arr), axis=1, keepdims=True)
    scales = np.where(max_abs < 1e-8, 1.0, max_abs / 127.0).astype(np.float32)
    q_arr = np.clip(np.round(arr / scales), -128, 127).astype(np.int8)
    return scales.flatten(), q_arr

def export_tokenizer_vocab_bin(output_path):
    print(f"[*] Mengekspor Tokenizer Vocabulary ke {output_path}...", flush=True)
    tok = AutoTokenizer.from_pretrained(SOURCE_MODEL_NAME, trust_remote_code=True)
    vocab = tok.get_vocab()
    num_tokens = len(vocab)
    if num_tokens < VOCAB_SIZE:
        num_tokens = VOCAB_SIZE

    id_to_token = {token_id: token_str for token_str, token_id in vocab.items()}

    with open(output_path, "wb") as f:
        # Header: uint32 vocab_size, uint32 max_token_len
        f.write(struct.pack("<II", num_tokens, 128))
        for tid in range(num_tokens):
            token_str = id_to_token.get(tid, f"<token_{tid}>")
            raw_bytes = token_str.encode("utf-8", errors="replace")
            if len(raw_bytes) > 255: raw_bytes = raw_bytes[:255]
            f.write(struct.pack("<B", len(raw_bytes)))
            f.write(raw_bytes)
    print(f"[OK] Tokenizer binary selesai diekspor! ({os.path.getsize(output_path)/1e6:.2f} MB)\n", flush=True)

def pack_wrai_x_checkpoint(checkpoint_pt_path, output_bin_path):
    print(f"[*] Memuat checkpoint: {checkpoint_pt_path}...")
    sd = torch.load(checkpoint_pt_path, map_location="cpu")

    print(f"[*] Menulis binary WRAI-X ke: {output_bin_path}...")
    with open(output_bin_path, "wb") as f:
        # 64-Byte Header
        # magic(4B), ver(2B), layers(2B), hidden(2B), ffn(2B), heads(2B), head_dim(2B),
        # levels(2B), max_seq(2B), vocab(4B), quant_mode(4B: 1 for int8), reserved(36B)
        header = struct.pack(
            "<IH HHHHHHH II 36s",
            MAGIC_HEADER,
            VERSION,
            NUM_LAYERS,
            HIDDEN_DIM,
            FFN_INTERMEDIATE_DIM,
            NUM_HEADS,
            HEAD_DIM,
            WAVELET_LEVELS,
            MAX_SEQ_LEN,
            VOCAB_SIZE,
            1, # INT8
            b"\x00" * 36
        )
        f.write(header)

        # 1. Embeddings
        print("  -> Menulis Embeddings...")
        embed_w = sd["embed.weight"]
        s_emb, q_emb = quantize_rowwise_int8(embed_w)
        f.write(s_emb.tobytes())
        f.write(q_emb.tobytes())

        # 2. Per-Layer Weights
        for l in range(NUM_LAYERS):
            if (l + 1) % 7 == 0 or l == 0:
                print(f"  -> Mengemas Layer {l+1}/{NUM_LAYERS}...")

            # Norms (FP32)
            f.write(sd[f"layers.{l}.rms_ret.weight"].float().numpy().tobytes())
            f.write(sd[f"layers.{l}.rms_ffn.weight"].float().numpy().tobytes())

            # Memory State RetNet (INT8)
            for w_name in ["w_q", "w_k", "w_v", "w_out"]:
                s_w, q_w = quantize_rowwise_int8(sd[f"layers.{l}.{w_name}.weight"])
                f.write(s_w.tobytes())
                f.write(q_w.tobytes())

            # Decays (FP32)
            f.write(sd[f"layers.{l}.decay_m"].float().numpy().tobytes())
            f.write(sd[f"layers.{l}.decay_r"].float().numpy().tobytes())

            # Haar Bridge (FP32 gains & gate)
            f.write(sd[f"layers.{l}.haar_bridge.low_gain"].float().numpy().tobytes())
            f.write(sd[f"layers.{l}.haar_bridge.mid_gain"].float().numpy().tobytes())
            f.write(sd[f"layers.{l}.haar_bridge.high_gain"].float().numpy().tobytes())
            f.write(sd[f"layers.{l}.haar_bridge.gate_w"].float().numpy().tobytes())
            f.write(sd[f"layers.{l}.haar_bridge.gate_b"].float().numpy().tobytes())

            # Reasoning State RetNet (INT8)
            for w_name in ["w_qr", "w_kr", "w_vr", "w_out_r"]:
                s_w, q_w = quantize_rowwise_int8(sd[f"layers.{l}.{w_name}.weight"])
                f.write(s_w.tobytes())
                f.write(q_w.tobytes())

            # Thinking Gate (FP32)
            f.write(sd[f"layers.{l}.think_gate.weight"].float().numpy().tobytes())
            f.write(sd[f"layers.{l}.think_gate.bias"].float().numpy().tobytes())

            # HDC Scratchpad Projections & Gate
            s_hk, q_hk = quantize_rowwise_int8(sd[f"layers.{l}.hdc.proj_key.weight"])
            s_hv, q_hv = quantize_rowwise_int8(sd[f"layers.{l}.hdc.proj_val.weight"])
            f.write(s_hk.tobytes()); f.write(q_hk.tobytes())
            f.write(s_hv.tobytes()); f.write(q_hv.tobytes())
            f.write(sd[f"layers.{l}.hdc.gate_hdc.weight"].float().numpy().tobytes())
            f.write(sd[f"layers.{l}.hdc.gate_hdc.bias"].float().numpy().tobytes())

            # SwiGLU FFN (INT8)
            for ffn_name in ["w_gate", "w_up", "w_down"]:
                s_ffn, q_ffn = quantize_rowwise_int8(sd[f"layers.{l}.ffn.{ffn_name}.weight"])
                f.write(s_ffn.tobytes())
                f.write(q_ffn.tobytes())

        # Final Norm
        f.write(sd["ln_final.weight"].float().numpy().tobytes())

    file_size_mb = os.path.getsize(output_bin_path) / 1e6
    print(f"\n[OK SUCCESS] WRAI-X Binary Berhasil Dibuat: {output_bin_path} ({file_size_mb:.1f} MB)!")

if __name__ == "__main__":
    export_tokenizer_vocab_bin(OUTPUT_VOCAB_NAME)
    ckpt = "wrai_x_08b_transplanted.pt"
    if os.path.exists(ckpt):
        pack_wrai_x_checkpoint(ckpt, OUTPUT_BIN_NAME)
    else:
        print(f"[*] Info: Checkpoint {ckpt} akan di-pack setelah proses training selesai.")
