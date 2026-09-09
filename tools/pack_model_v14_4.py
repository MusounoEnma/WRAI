#!/usr/bin/env python3
"""
=================================================================================
  WRAI v14.4 INT8 QUANTIZER & BINARY PACKER (PT -> BINARY .BIN EXPORTER)
=================================================================================
Script ini membaca checkpoint PyTorch `models/wrai_v14_4_best.pt` (Epoch 8, Val Loss 2.25)
dan file `models/pruned_vocab_map_v14_4.json`, lalu men-quantize bobot model menjadi
Int8 Fixed-Point [-128, 127] serta membungkusnya ke dalam file biner tunggal:
`models/wrai_v14_4.bin` (~108 MB) untuk dieksekusi oleh C Native Engine (wrai.exe).
"""

import json
import math
import os
import struct
import torch
import torch.nn as nn

MODEL_PT_PATH = "models/wrai_v14_4_best.pt"
VOCAB_MAP_PATH = "models/pruned_vocab_map_v14_4.json"
OUTPUT_BIN_PATH = "models/wrai_v14_4.bin"

MAGIC_HEADER = 0x57524149  # ASCII "WRAI"
VERSION = 144              # v14.4
PRUNED_VOCAB_SIZE = 32000
HIDDEN_DIM = 1024
NUM_LAYERS = 12
WAVELET_LEVELS = 4

def quantize_tensor_to_int8(tensor):
    """Symmetric Int8 Quantization to range [-127, 127] with float scale factor."""
    tensor_float = tensor.float()
    max_val = torch.max(torch.abs(tensor_float)).item()
    if max_val == 0.0:
        scale = 1.0
    else:
        scale = max_val / 127.0
    
    quantized = torch.clamp(torch.round(tensor_float / scale), -127, 127).to(torch.int8)
    return quantized, scale

def pack_v14_4_model():
    print("=================================================================", flush=True)
    print("  WRAI v14.4 INT8 QUANTIZER & BINARY PACKER (EXPORTER)          ", flush=True)
    print("=================================================================\n", flush=True)

    if not os.path.exists(MODEL_PT_PATH):
        print(f"[ERROR] File checkpoint {MODEL_PT_PATH} tidak ditemukan!", flush=True)
        return
    if not os.path.exists(VOCAB_MAP_PATH):
        print(f"[ERROR] File vocab map {VOCAB_MAP_PATH} tidak ditemukan!", flush=True)
        return

    print(f"[*] Loading PyTorch Checkpoint: {MODEL_PT_PATH}...", flush=True)
    ckpt = torch.load(MODEL_PT_PATH, map_location="cpu")
    state_dict = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    epoch = ckpt.get("epoch", "N/A") if isinstance(ckpt, dict) else "N/A"
    val_loss = ckpt.get("best_val_loss", "N/A") if isinstance(ckpt, dict) else "N/A"
    val_loss_str = f"{val_loss:.4f}" if isinstance(val_loss, (float, int)) else str(val_loss)
    print(f"[OK] Checkpoint Loaded! Epoch: {epoch} | Val Loss: {val_loss_str}", flush=True)

    print(f"[*] Loading Pruned Vocab Map: {VOCAB_MAP_PATH}...", flush=True)
    with open(VOCAB_MAP_PATH, "r", encoding="utf-8") as f:
        vocab_data = json.load(f)
    pruned_id_to_teacher_id = vocab_data["pruned_id_to_teacher_id"]
    print(f"[OK] Vocab Map Loaded ({len(pruned_id_to_teacher_id):,} tokens)", flush=True)

    print(f"\n[*] Starting Int8 Quantization & Binary Packing to: {OUTPUT_BIN_PATH}...", flush=True)
    
    with open(OUTPUT_BIN_PATH, "wb") as f:
        # 1. Header & Sizing Metadata (32 bytes)
        # Format: Magic(4B), Version(4B), VocabSize(4B), HiddenDim(4B), NumLayers(4B), WaveletLevels(4B), Epoch(4B), ValLoss(4B float)
        val_loss_float = float(val_loss) if isinstance(val_loss, (float, int)) else 0.0
        epoch_int = int(epoch) if isinstance(epoch, int) else 0
        
        header = struct.pack(
            "<IIIIIIIf",
            MAGIC_HEADER,
            VERSION,
            PRUNED_VOCAB_SIZE,
            HIDDEN_DIM,
            NUM_LAYERS,
            WAVELET_LEVELS,
            epoch_int,
            val_loss_float
        )
        f.write(header)

        # 2. Vocabulary Mapping Table (32,000 x 4 bytes int32)
        print("  [+] Packing Vocabulary Mapping Table...", flush=True)
        for tid in pruned_id_to_teacher_id:
            f.write(struct.pack("<i", int(tid)))

        # 3. Quantize Embedding Weight (32,000 x 1024 Int8 + 4B Scale)
        print("  [+] Quantizing Embedding Weights (32000 x 1024)...", flush=True)
        embed_weight = state_dict["embedding.weight"]
        q_embed, scale_embed = quantize_tensor_to_int8(embed_weight)
        f.write(struct.pack("<f", scale_embed))
        f.write(q_embed.numpy().tobytes())

        # 4. Positional Encoding Buffer (1024 x 4B float)
        print("  [+] Packing Positional Encoding Buffer...", flush=True)
        pos_pe = state_dict["pos_encoder.pe"].squeeze(0).float().numpy()
        f.write(pos_pe.tobytes())

        # 5. Layer Norm & Spectral Parameters
        print("  [+] Packing LayerNorm & Spectral Gains...", flush=True)
        for prefix in ["spectral1", "spectral2"]:
            gate = state_dict[f"{prefix}.gate"].float().numpy()
            f.write(gate.tobytes())
            approx_gain = state_dict[f"{prefix}.approx_gain"].float().numpy()
            f.write(approx_gain.tobytes())
            for l in range(WAVELET_LEVELS):
                detail_gain = state_dict[f"{prefix}.detail_gains.{l}"].float().numpy()
                f.write(detail_gain.tobytes())

        # 6. 12-Layer ResGRU Weights (Int8 Quantized)
        print("  [+] Quantizing 12-Layer ResGRU Weights...", flush=True)
        for l in range(NUM_LAYERS):
            ln_w = state_dict[f"gru_stack.layers.{l}.ln.weight"].float().numpy()
            ln_b = state_dict[f"gru_stack.layers.{l}.ln.bias"].float().numpy()
            f.write(ln_w.tobytes())
            f.write(ln_b.tobytes())

            w_ih = state_dict[f"gru_stack.layers.{l}.gru.weight_ih_l0"]
            w_hh = state_dict[f"gru_stack.layers.{l}.gru.weight_hh_l0"]
            b_ih = state_dict[f"gru_stack.layers.{l}.gru.bias_ih_l0"].float().numpy()
            b_hh = state_dict[f"gru_stack.layers.{l}.gru.bias_hh_l0"].float().numpy()

            q_w_ih, s_w_ih = quantize_tensor_to_int8(w_ih)
            q_w_hh, s_w_hh = quantize_tensor_to_int8(w_hh)

            f.write(struct.pack("<f", s_w_ih))
            f.write(q_w_ih.numpy().tobytes())
            f.write(struct.pack("<f", s_w_hh))
            f.write(q_w_hh.numpy().tobytes())
            f.write(b_ih.tobytes())
            f.write(b_hh.tobytes())

        # 7. Final LayerNorm Weights
        print("  [+] Packing Final LayerNorm Weights...", flush=True)
        ln_f_w = state_dict["ln_final.weight"].float().numpy()
        ln_f_b = state_dict["ln_final.bias"].float().numpy()
        f.write(ln_f_w.tobytes())
        f.write(ln_f_b.tobytes())

    out_size = os.path.getsize(OUTPUT_BIN_PATH) / (1024 * 1024)
    print(f"\n[OK 100% SUCCESS] Binary Model Successfully Exported to: {OUTPUT_BIN_PATH} ({out_size:.1f} MB)", flush=True)

if __name__ == "__main__":
    pack_v14_4_model()
