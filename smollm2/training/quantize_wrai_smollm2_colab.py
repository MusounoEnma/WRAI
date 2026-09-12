"""
=============================================================================
 🚀 WRAI-Smol: SmolLM2 Quantizer & Zero-Heap C Binary Packer
=============================================================================
 Supports:
   - SmolLM2-135M (Default): D=576, FFN=1536, 30 Layers, 9 Heads, Head Dim=64, Vocab=49152
   - SmolLM2-360M (Option):  D=960, FFN=2560, 32 Layers, 15 Heads, Head Dim=64, Vocab=49152

 Output:
   - wrai_smollm2_135m_int8.bin (~140 MB)
   - wrai_smollm2_vocab.bin (~0.5 MB)
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

MODEL_VARIANT = os.getenv("SMOLLM2_VARIANT", "135M")

if MODEL_VARIANT == "360M":
    SOURCE_MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"
    NUM_LAYERS = 32
    HIDDEN_DIM = 960
    FFN_INTERMEDIATE_DIM = 2560
    NUM_HEADS = 15
    HEAD_DIM = 64
    OUTPUT_BIN_NAME = "wrai_smollm2_360m_int8.bin"
else:
    SOURCE_MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"
    NUM_LAYERS = 30
    HIDDEN_DIM = 576
    FFN_INTERMEDIATE_DIM = 1536
    NUM_HEADS = 9
    HEAD_DIM = 64
    OUTPUT_BIN_NAME = "wrai_smollm2_135m_int8.bin"

OUTPUT_VOCAB_NAME = "wrai_smollm2_vocab.bin"
MAGIC_HEADER = 0x57524149  # "WRAI"
VERSION = 171              # WRAI-Smol (v17.1)
WAVELET_LEVELS = 4
MAX_SEQ_LEN = 2048
VOCAB_SIZE = 49152

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
    print(f"[*] Exporting SmolLM2 Tokenizer Vocabulary to {output_path}...", flush=True)
    tok = AutoTokenizer.from_pretrained(SOURCE_MODEL_NAME)
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
            if len(raw_bytes) > 255:
                raw_bytes = raw_bytes[:255]
            f.write(struct.pack("<B", len(raw_bytes)))
            f.write(raw_bytes)
    print(f"[OK] Tokenizer binary exported! ({os.path.getsize(output_path)/1e6:.2f} MB)\n", flush=True)

def pack_wrai_smollm2_checkpoint(checkpoint_pt_path, output_bin_path):
    print(f"[*] Loading PyTorch checkpoint: {checkpoint_pt_path}...")
    sd = torch.load(checkpoint_pt_path, map_location="cpu")

    print(f"[*] Packing INT8 binary to: {output_bin_path}...")
    with open(output_bin_path, "wb") as f:
        # 64-Byte Standard WRAI Header
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
            1,  # 1 = INT8 Symmetric Row-wise
            b"\x00" * 36
        )
        f.write(header)

        # Helper to write FP32 1D vectors (RMSNorm, bias, gates)
        def write_fp32_tensor(t):
            arr = t.detach().cpu().to(torch.float32).numpy().flatten()
            f.write(arr.tobytes())

        # Helper to write Quantized 2D matrices
        def write_q_matrix(t):
            scales, q_bytes = quantize_rowwise_int8(t)
            f.write(scales.tobytes())
            f.write(q_bytes.tobytes())

        # 1. Global Embeddings & Final Norm & LM Head
        print("  • Serializing embeddings & final head...")
        write_q_matrix(sd["embed_tokens.weight"])
        write_fp32_tensor(sd["norm.weight"])
        write_q_matrix(sd["lm_head.weight"])

        # 2. Sequential Layer Blocks
        for i in range(NUM_LAYERS):
            if (i + 1) % 5 == 0 or i == NUM_LAYERS - 1:
                print(f"  • Serializing Layer {i+1}/{NUM_LAYERS}...")

            pfx = f"layers.{i}."

            # Layer Norms (RMSNorms in FP32)
            write_fp32_tensor(sd[pfx + "input_layernorm.weight"])
            write_fp32_tensor(sd[pfx + "post_attention_layernorm.weight"])

            # Retention Mt Projections (INT8)
            write_q_matrix(sd[pfx + "retention.q_proj.weight"])
            write_q_matrix(sd[pfx + "retention.k_proj.weight"])
            write_q_matrix(sd[pfx + "retention.v_proj.weight"])
            write_q_matrix(sd[pfx + "retention.out_proj.weight"])

            # Retention Rt Projections (INT8)
            write_q_matrix(sd[pfx + "retention.q_r_proj.weight"])
            write_q_matrix(sd[pfx + "retention.k_r_proj.weight"])
            write_q_matrix(sd[pfx + "retention.v_r_proj.weight"])
            write_q_matrix(sd[pfx + "retention.out_r_proj.weight"])

            # Retention Hyperparameters (FP32)
            write_fp32_tensor(sd[pfx + "retention.decay_m"])
            write_fp32_tensor(sd[pfx + "retention.decay_r"])
            write_fp32_tensor(sd[pfx + "retention.gn_m.weight"])
            write_fp32_tensor(sd[pfx + "retention.gn_m.bias"])
            write_fp32_tensor(sd[pfx + "retention.gn_r.weight"])
            write_fp32_tensor(sd[pfx + "retention.gn_r.bias"])

            # Wavelet DWT Gains & Gates (FP32)
            write_fp32_tensor(sd[pfx + "retention.dwt.low_gain"])
            write_fp32_tensor(sd[pfx + "retention.dwt.mid_gain"])
            write_fp32_tensor(sd[pfx + "retention.dwt.high_gain"])
            write_fp32_tensor(sd[pfx + "retention.dwt.gate_w"])
            write_fp32_tensor(sd[pfx + "retention.dwt.gate_b"])
            write_fp32_tensor(sd[pfx + "retention.alpha"])

            # Frozen SwiGLU FFN (INT8)
            write_q_matrix(sd[pfx + "mlp.w_gate.weight"])
            write_q_matrix(sd[pfx + "mlp.w_up.weight"])
            write_q_matrix(sd[pfx + "mlp.w_down.weight"])

    total_bytes = os.path.getsize(output_bin_path)
    print("\n" + "=" * 80)
    print(f" [SUCCESS] WRAI-Smol INT8 Binary Created!")
    print(f"    • File Path : {output_bin_path}")
    print(f"    • File Size : {total_bytes:,} bytes ({total_bytes / 1e6:.2f} MB)")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    export_tokenizer_vocab_bin(OUTPUT_VOCAB_NAME)
    # If checkpoint exists, pack it
    pt_path = f"wrai_smollm2_{MODEL_VARIANT.lower()}_transplanted.pt"
    if os.path.exists(pt_path):
        pack_wrai_smollm2_checkpoint(pt_path, OUTPUT_BIN_NAME)
    else:
        print(f"[*] Note: Run training first to create {pt_path} before packing binary.")
