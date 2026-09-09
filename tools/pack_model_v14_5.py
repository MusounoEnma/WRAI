#!/usr/bin/env python3
"""
=============================================================================
 WRAI v14.5 MODEL QUANTIZER & BINARY PACKER (INT8 SYMMETRIC)
=============================================================================
 Serializes:
  - Header: Magic (0x57524149), Version (145), VocabSize (32000), HiddenDim (1024),
            NumLayers (12), WaveletLevels (4), FFNDim (1536), Epoch, Loss.
  - Pruned Vocab Mapping Table (32000 x 4 bytes)
  - Embedding Matrix (Scale + 32000 x 1024 Int8)
  - Positional Encoding (512 x 1024 Float32)
  - Dual Spectral Layers (Gate, ApproxGain, DetailGains Float32)
  - 12 Hybrid Layers:
      - RMSNorm Weight (1024 Floats)
      - GRU W_ih (Scale + 3072x1024 Int8) + W_hh (Scale + 3072x1024 Int8) + Biases
      - RMSNorm FFN Weight (1024 Floats)
      - SwiGLU W_gate (Scale + 1536x1024 Int8)
      - SwiGLU W_up (Scale + 1536x1024 Int8)
      - SwiGLU W_down (Scale + 1024x1536 Int8)
  - Final RMSNorm Weight (1024 Floats)
=============================================================================
"""

import os
import struct
import json
import numpy as np
import torch

CHECKPOINT_PATH = "models_v14_5/wrai_v14_5_best.pt"
META_PATH = "models_v14_5/wrai_v14_5_best_meta.json"
VOCAB_MAP_PATH = "models/pruned_vocab_map_v14_4.json"
OUTPUT_BIN_PATH = "models/wrai_v14_5.bin"

def quantize_int8(tensor: torch.Tensor):
    arr = tensor.float().numpy()
    max_abs = np.max(np.abs(arr))
    if max_abs < 1e-8:
        scale = 1.0
        q_arr = np.zeros_like(arr, dtype=np.int8)
    else:
        scale = max_abs / 127.0
        q_arr = np.clip(np.round(arr / scale), -128, 127).astype(np.int8)
    return scale, q_arr

def main():
    print("=================================================================")
    print("      WRAI v14.5 INT8 BINARY PACKER (SWIGLU + RMSNORM)           ")
    print("=================================================================")

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"[WARN] Checkpoint not found at: {CHECKPOINT_PATH}. (Run after Colab training finishes)")
        return

    state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu")
    with open(META_PATH, "r") as f:
        meta = json.load(f)

    with open(VOCAB_MAP_PATH, "r", encoding="utf-8") as f:
        vocab_map = json.load(f)
    pruned_to_teacher = vocab_map["pruned_id_to_teacher_id"]

    magic = 0x57524149  # "WRAI"
    version = 145       # v14.5
    vocab_size = meta.get("vocab_size", 32000)
    hidden_dim = meta.get("hidden_dim", 1024)
    num_layers = meta.get("layers", 12)
    wavelet_levels = meta.get("wavelet_levels", 4)
    ffn_dim = meta.get("ffn_intermediate_dim", 1536)
    epoch = meta.get("epoch", 1)
    best_loss = meta.get("best_val_loss", 0.0)

    print(f"[*] Packing Model Version: v{version/10:.1f} | Layers: {num_layers} | Hidden: {hidden_dim} | FFN: {ffn_dim}")

    with open(OUTPUT_BIN_PATH, "wb") as f:
        # 1. Header (40 bytes)
        header = struct.pack("<IIIIIIIIff", 
                             magic, version, vocab_size, hidden_dim, 
                             num_layers, wavelet_levels, ffn_dim, epoch, 
                             best_loss, 0.0)
        f.write(header)

        # 2. Vocab Mapping Table (32000 x 4 bytes)
        f.write(struct.pack(f"<{vocab_size}i", *pruned_to_teacher))

        # 3. Embedding Matrix
        scale_emb, q_emb = quantize_int8(state_dict["embed.weight"])
        f.write(struct.pack("<f", scale_emb))
        f.write(q_emb.tobytes())

        # 4. Positional Encoding
        pe = state_dict["pos_encoder.pe"].squeeze(0).float().numpy()
        f.write(pe.tobytes())

        # 5. Dual Wavelet Spectral Layers
        for prefix in ["spectral1", "spectral2"]:
            gate = state_dict[f"{prefix}.gate.weight"].float().numpy()
            approx_gain = state_dict[f"{prefix}.approx_gain"].float().numpy()
            f.write(gate.tobytes())
            f.write(approx_gain.tobytes())
            for l in range(wavelet_levels):
                d_gain = state_dict[f"{prefix}.detail_gains.{l}"].float().numpy()
                f.write(d_gain.tobytes())

        # 6. 12 Hybrid Layers (RMSNorm + ResGRU + SwiGLU FFN)
        for l in range(num_layers):
            # A. RMSNorm GRU
            rms_gru_w = state_dict[f"layers.{l}.rms_gru.weight"].float().numpy()
            f.write(rms_gru_w.tobytes())

            # B. GRU Weights
            s_ih, q_ih = quantize_int8(state_dict[f"layers.{l}.gru.weight_ih_l0"])
            s_hh, q_hh = quantize_int8(state_dict[f"layers.{l}.gru.weight_hh_l0"])
            b_ih = state_dict[f"layers.{l}.gru.bias_ih_l0"].float().numpy()
            b_hh = state_dict[f"layers.{l}.gru.bias_hh_l0"].float().numpy()
            
            f.write(struct.pack("<f", s_ih))
            f.write(q_ih.tobytes())
            f.write(struct.pack("<f", s_hh))
            f.write(q_hh.tobytes())
            f.write(b_ih.tobytes())
            f.write(b_hh.tobytes())

            # C. RMSNorm FFN
            rms_ffn_w = state_dict[f"layers.{l}.rms_ffn.weight"].float().numpy()
            f.write(rms_ffn_w.tobytes())

            # D. SwiGLU Weights
            s_gate, q_gate = quantize_int8(state_dict[f"layers.{l}.ffn.w_gate.weight"])
            s_up, q_up     = quantize_int8(state_dict[f"layers.{l}.ffn.w_up.weight"])
            s_down, q_down = quantize_int8(state_dict[f"layers.{l}.ffn.w_down.weight"])

            f.write(struct.pack("<f", s_gate))
            f.write(q_gate.tobytes())
            f.write(struct.pack("<f", s_up))
            f.write(q_up.tobytes())
            f.write(struct.pack("<f", s_down))
            f.write(q_down.tobytes())

        # 7. Final RMSNorm
        ln_final = state_dict["ln_final.weight"].float().numpy()
        f.write(ln_final.tobytes())

    file_size_mb = os.path.getsize(OUTPUT_BIN_PATH) / (1024 * 1024)
    print(f"[OK 100% SUCCESS] Model Packed: {OUTPUT_BIN_PATH} ({file_size_mb:.2f} MB)")

if __name__ == "__main__":
    main()
