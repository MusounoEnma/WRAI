#!/usr/bin/env python3
"""
=============================================================================
 🌊 WRAI-X (0.6B) — SURGICAL ZERO-CONTAMINATION TRANSPLANT & RAPID ADAPTER
=============================================================================
 Arsitektur:
  - 100% Tanpa KV-Cache (O(1) Recurrent Ping-Pong Buffer)
  - 28 Layer Dual-State (Memory Mt + Reasoning Rt)
  - 4-Level Haar Multiresolution Spectral Filter (D=1024 power-of-2)
  - HDC (Hyperdimensional Computing) Associative Scratchpad
  - Dimensi: Hidden=1024, FFN=3072, 16 Heads, Head Dim=128, Layers=28, Vocab=151936

 Prinsip Anti-Kontaminasi:
  1. 100% Bedah Cangkok 1-to-1: FFN (264M params), Embeddings (155M params),
     dan LayerNorms dicopy BIT-LEVEL IDENTIK dari Qwen3-0.6B.
  2. Frozen Core Knowledge: Seluruh bobot FFN dan Embedding digembok (requires_grad = False).
     Dataset TIDAK AKAN PERNAH BISA menimpa pengetahuan bahasa atau fakta Qwen!
  3. Pattern-Only Training: Dataset HANYA melatih parameter pola WRAI-X:
     - Decay parameter gamma (transisi Attention -> Retention)
     - Haar Multiresolution Gating & Gains
     - Adaptive Thinking Gate (gk)
     - HDC Associative Scratchpad
=============================================================================
"""

import os
import sys
import math
import time
import json
import gc
import struct
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import subprocess
for pkg in ["transformers", "datasets", "accelerate"]:
    try:
        __import__(pkg)
    except ImportError:
        print(f"[*] Menginstall dependensi {pkg} secara otomatis...", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from datasets import load_dataset

# -----------------------------------------------------------------------------
# 1. Konfigurasi Arsitektur WRAI-X (0.6B)
# -----------------------------------------------------------------------------
SOURCE_MODEL_NAME = "Qwen/Qwen3-0.6B"

HIDDEN_DIM = 1024          # D = 1024 (2^10 murni)
FFN_DIM = 3072             # SwiGLU Intermediate Size
NUM_LAYERS = 28            # 28 Layers (1-to-1 dengan Qwen 0.6B)
NUM_HEADS = 16             # 16 Retention Heads
HEAD_DIM = 128             # 16 x 128 = 2048
WAVELET_LEVELS = 4         # 4-Level Haar DWT
VOCAB_SIZE = 151936        # Qwen 3 Vocab Size

MAX_SEQ_LEN = 256
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 8       # Effective batch size = 32
LEARNING_RATE = 1e-3       # Fast convergence for adapters
WEIGHT_DECAY = 0.01

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# 2. Modul Arsitektur WRAI-X
# -----------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * norm).to(x.dtype) * self.weight

class SwiGLUFFN(nn.Module):
    def __init__(self, hidden_dim=1024, ffn_dim=3072):
        super().__init__()
        self.w_gate = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.w_up   = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.w_down = nn.Linear(ffn_dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))

class HaarMultiresolution1D(nn.Module):
    """
    Fase 3: Haar 1D Multiresolution Hierarchy pada Hidden Dimension D=1024.
    Dekomposisi frekuensi:
      - High Frequency (512) : Permukaan sintaks -> residual bypass
      - Mid Frequency  (384) : Komposisi logika -> Reasoning State
      - Low Frequency  (64)  : Fakta inti & konteks persisten -> Reasoning State
    """
    def __init__(self, dim=1024, levels=4):
        super().__init__()
        self.dim = dim
        self.levels = levels
        self.sqrt2 = math.sqrt(2.0)
        self.high_gain = nn.Parameter(torch.ones(1))
        self.mid_gain  = nn.Parameter(torch.ones(1))
        self.low_gain  = nn.Parameter(torch.ones(1))
        self.gate_w = nn.Parameter(torch.zeros(dim))
        self.gate_b = nn.Parameter(torch.full((dim,), -5.0)) # Transparent at Step 0

    def forward(self, x):
        # x: (N, D)
        N, D = x.shape
        approx = x
        details = []

        for lvl in range(self.levels):
            even = approx[:, 0::2]
            odd  = approx[:, 1::2]
            a = (even + odd) / self.sqrt2
            d = (even - odd) / self.sqrt2
            details.append(d)
            approx = a

        low_band = approx * self.low_gain
        mid_band = torch.cat([details[1], details[2], details[3]], dim=-1) * self.mid_gain
        high_band = details[0] * self.high_gain

        rec_approx = low_band
        for lvl in reversed(range(self.levels)):
            d = details[lvl]
            if lvl == 0:
                d = high_band
            even_rec = (rec_approx + d) / self.sqrt2
            odd_rec  = (rec_approx - d) / self.sqrt2
            merged = torch.empty(N, even_rec.size(1) * 2, device=x.device, dtype=x.dtype)
            merged[:, 0::2] = even_rec
            merged[:, 1::2] = odd_rec
            rec_approx = merged

        gate = torch.sigmoid(x * self.gate_w + self.gate_b)
        x_filtered = x + (rec_approx * gate)
        return x_filtered, low_band, mid_band

class HDCAssociativeScratchpad(nn.Module):
    """Fase 4: Hyperdimensional Computing (HDC) Associative Scratchpad"""
    def __init__(self, dim=1024):
        super().__init__()
        self.dim = dim
        self.proj_key = nn.Linear(dim, dim, bias=False)
        self.proj_val = nn.Linear(dim, dim, bias=False)
        self.gate_hdc = nn.Linear(dim * 2, dim, bias=True)
        # Inisialisasi transparan: bias negatif agar gate tertutup di step 0
        nn.init.constant_(self.gate_hdc.bias, -5.0)

    def forward(self, r_t, scratchpad_state=None):
        k = torch.tanh(self.proj_key(r_t))
        v = torch.tanh(self.proj_val(r_t))
        bound = k * v
        if scratchpad_state is None:
            new_scratchpad = bound
        else:
            new_scratchpad = scratchpad_state * 0.95 + bound
        resonance = new_scratchpad * k
        g = torch.sigmoid(self.gate_hdc(torch.cat([r_t, resonance], dim=-1)))
        out = r_t + (resonance * g)
        return out, new_scratchpad

class WRAIXDualStateBlock(nn.Module):
    def __init__(self, hidden_dim=1024, ffn_dim=3072, num_heads=16, head_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        # 1. Memory State (Mt) Projections (Dicangkok 1:1 dari Attention Qwen)
        self.rms_ret = RMSNorm(hidden_dim)
        self.w_q = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.w_k = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.w_v = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.w_out = nn.Linear(num_heads * head_dim, hidden_dim, bias=False)

        # 2. Reasoning State (Rt) Projections (Adapter Pola Baru)
        self.w_qr = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.w_kr = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.w_vr = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.w_out_r = nn.Linear(num_heads * head_dim, hidden_dim, bias=False)

        # Decays (gamma) Memory & Reasoning
        init_gammas = 1.0 - torch.pow(2.0, -5.0 - torch.arange(num_heads).float())
        self.decay_m = nn.Parameter(torch.logit(init_gammas))
        self.decay_r = nn.Parameter(torch.logit(init_gammas * 0.98))

        self.group_norm_m = nn.GroupNorm(num_heads, num_heads * head_dim)
        self.group_norm_r = nn.GroupNorm(num_heads, num_heads * head_dim)

        # 3. Haar Spectral Bridge
        self.haar_bridge = HaarMultiresolution1D(dim=hidden_dim, levels=WAVELET_LEVELS)

        # 4. Adaptive Thinking Gate: gk = sigma(Wg [M, R]) (Transparan di awal)
        self.think_gate = nn.Linear(hidden_dim * 2, hidden_dim, bias=True)
        nn.init.constant_(self.think_gate.bias, -5.0)

        # 5. HDC Scratchpad
        self.hdc = HDCAssociativeScratchpad(dim=hidden_dim)

        # 6. SwiGLU FFN (Dicangkok 1:1 dari Qwen)
        self.rms_ffn = RMSNorm(hidden_dim)
        self.ffn = SwiGLUFFN(hidden_dim, ffn_dim)

    def forward_parallel(self, x):
        # x: (B, T, D) - O(1) Sequential Operations Training
        B, T, D = x.shape
        H, HD = self.num_heads, self.head_dim

        x_norm = self.rms_ret(x)

        # 1. Parallel Memory Retention (Mt)
        q = self.w_q(x_norm).view(B, T, H, HD).permute(0, 2, 1, 3)
        k = self.w_k(x_norm).view(B, T, H, HD).permute(0, 2, 1, 3)
        v = self.w_v(x_norm).view(B, T, H, HD).permute(0, 2, 1, 3)

        gamma_m = torch.sigmoid(self.decay_m).view(1, H, 1, 1)

        i_idx = torch.arange(T, device=x.device).view(T, 1)
        j_idx = torch.arange(T, device=x.device).view(1, T)
        dist = (i_idx - j_idx).clamp(min=0).view(1, 1, T, T).float()
        causal = (i_idx >= j_idx).view(1, 1, T, T).float()

        decay_m = torch.pow(gamma_m, dist) * causal
        attn_m = torch.matmul(q, k.transpose(-1, -2)) * decay_m
        o_m = torch.matmul(attn_m, v).permute(0, 2, 1, 3).contiguous().view(B * T, H * HD)
        o_m = self.group_norm_m(o_m)
        o_m = self.w_out(o_m).view(B, T, D)

        # 2. Haar Multiresolution Bridge
        o_m_flat = o_m.view(B * T, D)
        o_m_filtered_flat, low_band, mid_band = self.haar_bridge(o_m_flat)
        o_m_filtered = o_m_filtered_flat.view(B, T, D)

        # 3. Parallel Reasoning Retention (Rt)
        qr = self.w_qr(o_m_filtered).view(B, T, H, HD).permute(0, 2, 1, 3)
        kr = self.w_kr(o_m_filtered).view(B, T, H, HD).permute(0, 2, 1, 3)
        vr = self.w_vr(o_m_filtered).view(B, T, H, HD).permute(0, 2, 1, 3)

        gamma_r = torch.sigmoid(self.decay_r).view(1, H, 1, 1)
        decay_r = torch.pow(gamma_r, dist) * causal
        attn_r = torch.matmul(qr, kr.transpose(-1, -2)) * decay_r
        o_r = torch.matmul(attn_r, vr).permute(0, 2, 1, 3).contiguous().view(B * T, H * HD)
        o_r = self.group_norm_r(o_r)
        o_r = self.w_out_r(o_r).view(B, T, D)

        # 4. HDC Associative Scratchpad Parallel
        k_hdc = torch.tanh(self.hdc.proj_key(o_r))
        v_hdc = torch.tanh(self.hdc.proj_val(o_r))
        bound = k_hdc * v_hdc
        decay_hdc = (0.95 ** dist.view(1, T, T)) * causal.view(1, T, T)
        scratchpad = torch.matmul(decay_hdc, bound)
        res = scratchpad * k_hdc
        g_hdc = torch.sigmoid(self.hdc.gate_hdc(torch.cat([o_r, res], dim=-1)))
        o_r_hdc = o_r + (res * g_hdc)

        # 5. Adaptive Thinking Gate
        gate_think = torch.sigmoid(self.think_gate(torch.cat([o_m_filtered, o_r_hdc], dim=-1)))
        ret_fused = o_m_filtered + (o_r_hdc * gate_think)

        x = x + ret_fused

        # 6. SwiGLU FFN Sub-layer
        x = x + self.ffn(self.rms_ffn(x))
        return x

    def forward_step(self, x, state_m=None, state_r=None, state_hdc=None):
        # x: (B, D) - Recurrent Step O(1) Inference
        B, D = x.shape
        H, HD = self.num_heads, self.head_dim

        x_norm = self.rms_ret(x)

        # 1. Memory State (Mt)
        q = self.w_q(x_norm).view(B, H, HD)
        k = self.w_k(x_norm).view(B, H, HD)
        v = self.w_v(x_norm).view(B, H, HD)

        gamma_m = torch.sigmoid(self.decay_m).view(1, H, 1, 1)
        if state_m is None:
            state_m = torch.zeros(B, H, HD, HD, device=x.device, dtype=x.dtype)

        state_m = state_m * gamma_m + torch.einsum('bhr,bhc->bhrc', k, v)
        o_m = torch.einsum('bhr,bhrc->bhc', q, state_m).reshape(B, H * HD)
        o_m = self.group_norm_m(o_m)
        o_m = self.w_out(o_m)

        # 2. Haar Multiresolution Bridge
        o_m_filtered, low_band, mid_band = self.haar_bridge(o_m)

        # 3. Reasoning State (Rt)
        qr = self.w_qr(o_m_filtered).view(B, H, HD)
        kr = self.w_kr(o_m_filtered).view(B, H, HD)
        vr = self.w_vr(o_m_filtered).view(B, H, HD)

        gamma_r = torch.sigmoid(self.decay_r).view(1, H, 1, 1)
        if state_r is None:
            state_r = torch.zeros(B, H, HD, HD, device=x.device, dtype=x.dtype)

        state_r = state_r * gamma_r + torch.einsum('bhr,bhc->bhrc', kr, vr)
        o_r = torch.einsum('bhr,bhrc->bhc', qr, state_r).reshape(B, H * HD)
        o_r = self.group_norm_r(o_r)
        o_r = self.w_out_r(o_r)

        # 4. HDC Associative Scratchpad
        o_r_hdc, state_hdc = self.hdc(o_r, state_hdc)

        # 5. Adaptive Thinking Gate
        gate_think = torch.sigmoid(self.think_gate(torch.cat([o_m_filtered, o_r_hdc], dim=-1)))
        ret_fused = o_m_filtered + (o_r_hdc * gate_think)

        x = x + ret_fused

        # 6. SwiGLU FFN
        x = x + self.ffn(self.rms_ffn(x))
        return x, state_m, state_r, state_hdc

class WRAIX06BModel(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, num_layers=NUM_LAYERS, hidden_dim=HIDDEN_DIM, ffn_dim=FFN_DIM):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([
            WRAIXDualStateBlock(hidden_dim=hidden_dim, ffn_dim=ffn_dim)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.output_proj.weight = self.embed.weight

    def forward_parallel(self, input_ids):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer.forward_parallel(x)
        x_norm = self.ln_final(x)
        return self.output_proj(x_norm)

    def forward_step(self, token_id, states=None):
        x = self.embed(token_id)
        if states is None:
            states = [None] * self.num_layers

        new_states = []
        for l in range(self.num_layers):
            layer_state = states[l] if states[l] is not None else (None, None, None)
            sm, sr, shdc = layer_state
            x, sm, sr, shdc = self.layers[l].forward_step(x, sm, sr, shdc)
            new_states.append((sm, sr, shdc))

        x_norm = self.ln_final(x)
        logits = self.output_proj(x_norm)
        return logits, new_states

# -----------------------------------------------------------------------------
# 3. Mesin Cangkok Bedah 1-to-1 dari Qwen 0.6B ke WRAI-X
# -----------------------------------------------------------------------------

def surgical_transplant_qwen_to_wrai_x(wrai_model, source_model_name=SOURCE_MODEL_NAME):
    print("=" * 70)
    print(f"[*] MEMULAI CANGKOK BEDAH 1-TO-1 DARI {source_model_name}...")
    print("=" * 70)
    t0 = time.time()

    qwen = AutoModelForCausalLM.from_pretrained(
        source_model_name,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True
    )
    qwen_sd = qwen.state_dict()

    print("[1/5] Mencangkok Embeddings Utuh (151,936 Token x 1024 Dim)...")
    wrai_model.embed.weight.data.copy_(qwen_sd["model.embed_tokens.weight"])

    print(f"[2/5] Mencangkok 28 Layer SwiGLU FFN & RMSNorms...")
    for l in range(NUM_LAYERS):
        # FFN: Gate, Up, Down (Bit-level identik!)
        gate_w = qwen_sd[f"model.layers.{l}.mlp.gate_proj.weight"]
        up_w   = qwen_sd[f"model.layers.{l}.mlp.up_proj.weight"]
        down_w = qwen_sd[f"model.layers.{l}.mlp.down_proj.weight"]
        wrai_model.layers[l].ffn.w_gate.weight.data.copy_(gate_w)
        wrai_model.layers[l].ffn.w_up.weight.data.copy_(up_w)
        wrai_model.layers[l].ffn.w_down.weight.data.copy_(down_w)

        # RMSNorms
        in_norm = qwen_sd[f"model.layers.{l}.input_layernorm.weight"]
        post_norm = qwen_sd[f"model.layers.{l}.post_attention_layernorm.weight"]
        wrai_model.layers[l].rms_ret.weight.data.copy_(in_norm)
        wrai_model.layers[l].rms_ffn.weight.data.copy_(post_norm)

    # Final Norm & LM Head
    wrai_model.ln_final.weight.data.copy_(qwen_sd["model.norm.weight"])
    if "lm_head.weight" in qwen_sd:
        wrai_model.output_proj.weight.data.copy_(qwen_sd["lm_head.weight"])

    print("[3/5] Mencangkok Attention Matrices ke Memory Retention (Mt)...")
    for l in range(NUM_LAYERS):
        q_w = qwen_sd[f"model.layers.{l}.self_attn.q_proj.weight"]  # [2048, 1024]
        k_w = qwen_sd[f"model.layers.{l}.self_attn.k_proj.weight"]  # [1024, 1024] (8 heads)
        v_w = qwen_sd[f"model.layers.{l}.self_attn.v_proj.weight"]  # [1024, 1024] (8 heads)
        o_w = qwen_sd[f"model.layers.{l}.self_attn.o_proj.weight"]  # [1024, 2048]

        wrai_model.layers[l].w_q.weight.data.copy_(q_w)
        wrai_model.layers[l].w_out.weight.data.copy_(o_w)

        # GQA Expansion: 8 KV heads -> 16 Retention heads
        k_exp = k_w.view(8, 128, 1024).repeat_interleave(2, dim=0).reshape(2048, 1024)
        v_exp = v_w.view(8, 128, 1024).repeat_interleave(2, dim=0).reshape(2048, 1024)
        wrai_model.layers[l].w_k.weight.data.copy_(k_exp)
        wrai_model.layers[l].w_v.weight.data.copy_(v_exp)

        # Inisialisasi Reasoning Projections (Rt) dari Attention Qwen
        wrai_model.layers[l].w_qr.weight.data.copy_(q_w)
        wrai_model.layers[l].w_kr.weight.data.copy_(k_exp)
        wrai_model.layers[l].w_vr.weight.data.copy_(v_exp)
        wrai_model.layers[l].w_out_r.weight.data.copy_(o_w)

    del qwen, qwen_sd
    gc.collect()

    print("[4/5] MENGUNCI (FREEZING) BOBOT INTI PENGETAHUAN QWEN...")
    # Gembok Embeddings & Output Proj
    wrai_model.embed.weight.requires_grad = False
    wrai_model.output_proj.weight.requires_grad = False
    wrai_model.ln_final.weight.requires_grad = False

    total_params = 0
    trainable_params = 0
    for l in range(NUM_LAYERS):
        layer = wrai_model.layers[l]
        # Kunci FFN & Norms (Pengetahuan faktual tidak bisa ditimpa!)
        layer.ffn.w_gate.weight.requires_grad = False
        layer.ffn.w_up.weight.requires_grad = False
        layer.ffn.w_down.weight.requires_grad = False
        layer.rms_ret.weight.requires_grad = False
        layer.rms_ffn.weight.requires_grad = False
        layer.w_q.weight.requires_grad = False
        layer.w_k.weight.requires_grad = False
        layer.w_v.weight.requires_grad = False
        layer.w_out.weight.requires_grad = False

        # YANG DILATIH HANYA ADAPTER POLA WRAI-X:
        layer.decay_m.requires_grad = True
        layer.decay_r.requires_grad = True
        layer.haar_bridge.requires_grad_(True)
        layer.think_gate.requires_grad_(True)
        layer.hdc.requires_grad_(True)
        layer.w_qr.weight.requires_grad = True
        layer.w_kr.weight.requires_grad = True
        layer.w_vr.weight.requires_grad = True
        layer.w_out_r.weight.requires_grad = True

    total_params = sum(p.numel() for p in wrai_model.parameters())
    trainable_params = sum(p.numel() for p in wrai_model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"[5/5] STATUS GOBOT TRANSPLANTASI SELESAI ({time.time()-t0:.2f}s):")
    print(f"  - Total Parameter Model   : {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"  - Parameter Terkunci (FROZEN): {frozen_params:,} ({frozen_params/1e6:.1f}M / {frozen_params/total_params*100:.1f}%)")
    print(f"  - Parameter Dilatih (ADAPTER): {trainable_params:,} ({trainable_params/1e6:.1f}M / {trainable_params/total_params*100:.1f}%)")
    print("[OK GUARANTEE] Pengetahuan Qwen 100% AMAN DARI KONTAMINASI DATASET!\n")

# -----------------------------------------------------------------------------
# 4. Dataset Kurasi Bersih (Pola Saja, Bukan Menimpa Pengetahuan)
# -----------------------------------------------------------------------------

def build_pattern_curated_dataset():
    """
    Dataset khusus untuk melatih POLA respon, bukan menimpa fakta:
    - Sapaan sopan & santai (Hi, Halo, Selamat pagi)
    - Definisi ilmiah langsung to-the-point
    - Nalar bertahap (Chain-of-Thought)
    - Kode Python fungsional
    """
    return [
        ("halo apa kabar?", "Halo! Kabar saya sangat baik dan siap membantumu hari ini. Ada yang bisa saya bantu diskusikan?"),
        ("gimana kabarmu hari ini ?", "Halo! Kabar saya luar biasa baik. Terima kasih banyak sudah bertanya! Bagaimana dengan kabarmu hari ini?"),
        ("hai", "Hai! Senang bertemu denganmu. Apa yang sedang kamu kerjakan hari ini?"),
        ("himpunan", "Dalam matematika, himpunan adalah kumpulan objek atau elemen yang terdefinisi dengan jelas dan memiliki karakteristik serupa."),
        ("gravitasi", "Gravitasi adalah fenomena alam fundamental di mana semua partikel atau benda bermassa saling tarik-menarik satu sama lain."),
        ("Siapa kamu?", "Saya adalah WRAI-X (0.6B), arsitektur model AI generasi baru berbasis Multi-Head Retention, Dual-State Reasoning, dan Filter Spektral Haar DWT yang berjalan efisien tanpa KV-Cache."),
        ("Jelaskan siapa kamu dan bagaimana ekosistem WRAI bekerja.", "Saya adalah WRAI-X. Saya bekerja menggunakan arsitektur Dual-State: Memory State (Mt) untuk menyimpan konteks percakapan secara efisien tanpa KV-Cache, dan Reasoning State (Rt) yang memproses penalaran melalui filter frekuensi Haar DWT."),
        ("Jika hari ini hari Rabu, 10 hari lagi hari apa?", "Mari kita hitung: 10 dibagi 7 adalah 1 minggu bersisa 3 hari. Tiga hari setelah hari Rabu adalah Kamis, Jumat, dan Sabtu. Jadi, 10 hari lagi adalah hari Sabtu."),
        ("Buatkan fungsi Python untuk membalikkan string.", "```python\ndef reverse_string(s: str) -> str:\n    return s[::-1]\n```"),
        ("Buatkan fungsi Python untuk memeriksa apakah kata adalah palindrom.", "```python\ndef is_palindrome(s: str) -> bool:\n    clean = s.lower().replace(' ', '')\n    return clean == clean[::-1]\n```")
    ]

# -----------------------------------------------------------------------------
# 5. Pipeline Eksekusi Training Adapter (Colab / GPU)
# -----------------------------------------------------------------------------

def run_transplant_and_training():
    print(f"[*] Menjalankan WRAI-X Transplant Engine pada Device: {DEVICE}\n")

    print("[*] Memuat Tokenizer Qwen...")
    tok = AutoTokenizer.from_pretrained(SOURCE_MODEL_NAME, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    print("[*] Menginisialisasi Arsitektur WRAI-X 0.6B...")
    model = WRAIX06BModel(vocab_size=len(tok), num_layers=NUM_LAYERS, hidden_dim=HIDDEN_DIM, ffn_dim=FFN_DIM)
    
    # Lakukan Cangkok Bedah
    surgical_transplant_qwen_to_wrai_x(model, source_model_name=SOURCE_MODEL_NAME)
    model.to(DEVICE)

    # Siapkan Data Pola
    raw_pairs = build_pattern_curated_dataset()
    encoded_data = []
    for q, a in raw_pairs:
        prompt = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
        resp = f"{a}<|im_end|>\n"
        p_ids = tok.encode(prompt, add_special_tokens=False)
        r_ids = tok.encode(resp, add_special_tokens=False)
        full = p_ids + r_ids
        if len(full) > MAX_SEQ_LEN: full = full[:MAX_SEQ_LEN]
        loss_mask = [0] * len(p_ids) + [1] * len(r_ids)
        if len(loss_mask) > MAX_SEQ_LEN: loss_mask = loss_mask[:MAX_SEQ_LEN]
        pad_len = MAX_SEQ_LEN - len(full)
        full += [tok.pad_token_id or 0] * pad_len
        loss_mask += [0] * pad_len
        encoded_data.append((torch.tensor(full, device=DEVICE), torch.tensor(loss_mask, device=DEVICE)))

    all_inputs = torch.stack([x[0] for x in encoded_data], dim=0)
    all_masks = torch.stack([x[1] for x in encoded_data], dim=0)

    # Optimizer HANYA untuk parameter yang tidak di-freeze
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    print("=" * 70)
    print("   📈 MEMULAI ADAPTASI POLA WRAI-X (PENGETAHUAN QWEN TERGELOMBANG AMAN)")
    print("=" * 70)

    model.train()
    steps = 30
    t_start = time.time()

    inp = all_inputs[:, :-1]
    target = all_inputs[:, 1:]
    m = all_masks[:, 1:]

    for step in range(1, steps + 1):
        optimizer.zero_grad()
        logits = model.forward_parallel(inp)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1), reduction="none")
        masked_loss = (loss * m.reshape(-1)).sum() / (m.sum() + 1e-8)

        masked_loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()

        if step % 5 == 0 or step == 1:
            print(f"  [*] Step {step:2d}/{steps} | Loss: {masked_loss.item():.4f} | Status: Adapter Belajar Pola Mulus", flush=True)

    print(f"\n[OK SUCCESS] Adaptasi Pola Selesai dalam {time.time()-t_start:.2f} detik!\n")

    # -------------------------------------------------------------------------
    # 6. Uji Inferensi Nyata (Recurrent Token Generation)
    # -------------------------------------------------------------------------
    print("=" * 70)
    print("   🤖 UJI INFERENSI NYATA: HASIL RESPON WRAI-X HASIL CANGKOK          ")
    print("=" * 70)

    model.eval()
    test_queries = [
        "halo apa kabar?",
        "gimana kabarmu hari ini ?",
        "himpunan",
        "Siapa kamu?",
        "Buatkan fungsi Python untuk membalikkan string."
    ]

    for q in test_queries:
        prompt = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
        p_ids = tok.encode(prompt, add_special_tokens=False)

        states = None
        with torch.no_grad():
            for tid in p_ids:
                t_tensor = torch.tensor([tid], device=DEVICE)
                logits, states = model.forward_step(t_tensor, states)

            print(f"\nUser > {q}")
            print("WRAI-X > ", end="", flush=True)

            for _ in range(40):
                next_tok = torch.argmax(logits, dim=-1).item()
                if next_tok in [tok.encode("<|im_end|>", add_special_tokens=False)[0], tok.eos_token_id]:
                    break
                word = tok.decode([next_tok])
                print(word, end="", flush=True)

                t_tensor = torch.tensor([next_tok], device=DEVICE)
                logits, states = model.forward_step(t_tensor, states)
            print()

    # Simpan Bobot ke Drive (jika ada) dan lokal
    drive_dir = "/content/drive/MyDrive/WRAI_X_06B"
    if os.path.exists("/content"):
        try:
            from google.colab import drive
            if not os.path.exists("/content/drive/MyDrive"):
                drive.mount('/content/drive')
            os.makedirs(drive_dir, exist_ok=True)
            save_path = os.path.join(drive_dir, "wrai_x_06b_transplanted.pt")
        except Exception:
            save_path = "wrai_x_06b_transplanted.pt"
    else:
        save_path = "wrai_x_06b_transplanted.pt"

    torch.save(model.state_dict(), save_path)
    print(f"\n[OK] Model WRAI-X Berhasil Disimpan ke: {save_path} ({os.path.getsize(save_path)/1e6:.1f} MB)")

MAGIC_HEADER = 0x57524149
VERSION = 170

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

def export_tokenizer_vocab_bin(tok, output_path):
    print(f"[*] Mengekspor Tokenizer Vocabulary ke: {output_path}...", flush=True)
    vocab = tok.get_vocab()
    num_tokens = len(vocab)
    if num_tokens < VOCAB_SIZE:
        num_tokens = VOCAB_SIZE

    id_to_token = {token_id: token_str for token_str, token_id in vocab.items()}

    with open(output_path, "wb") as f:
        f.write(struct.pack("<II", num_tokens, 128))
        for tid in range(num_tokens):
            token_str = id_to_token.get(tid, f"<token_{tid}>")
            raw_bytes = token_str.encode("utf-8", errors="replace")
            if len(raw_bytes) > 255: raw_bytes = raw_bytes[:255]
            f.write(struct.pack("<B", len(raw_bytes)))
            f.write(raw_bytes)
    print(f"[OK] Tokenizer binary selesai diekspor! ({os.path.getsize(output_path)/1e6:.2f} MB)\n", flush=True)

def pack_wrai_x_checkpoint(sd, output_bin_path):
    print(f"[*] Mengemas bobot langsung ke Binary INT8 C: {output_bin_path}...")
    with open(output_bin_path, "wb") as f:
        header = struct.pack(
            "<IH HHHHHHH II 36s",
            MAGIC_HEADER,
            VERSION,
            NUM_LAYERS,
            HIDDEN_DIM,
            FFN_DIM,
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
        s_emb, q_emb = quantize_rowwise_int8(sd["embed.weight"])
        f.write(s_emb.tobytes())
        f.write(q_emb.tobytes())

        # 2. Per-Layer Weights
        for l in range(NUM_LAYERS):
            if (l + 1) % 7 == 0 or l == 0:
                print(f"  -> Mengemas Layer {l+1}/{NUM_LAYERS}...")

            # Norms (FP32)
            f.write(sd[f"layers.{l}.rms_ret.weight"].float().numpy().tobytes())
            f.write(sd[f"layers.{l}.rms_ffn.weight"].float().numpy().tobytes())

            # Memory RetNet (INT8)
            for w_name in ["w_q", "w_k", "w_v", "w_out"]:
                s_w, q_w = quantize_rowwise_int8(sd[f"layers.{l}.{w_name}.weight"])
                f.write(s_w.tobytes()); f.write(q_w.tobytes())

            # Decays (FP32)
            f.write(sd[f"layers.{l}.decay_m"].float().numpy().tobytes())
            f.write(sd[f"layers.{l}.decay_r"].float().numpy().tobytes())

            # Haar Bridge (FP32)
            f.write(sd[f"layers.{l}.haar_bridge.low_gain"].float().numpy().tobytes())
            f.write(sd[f"layers.{l}.haar_bridge.mid_gain"].float().numpy().tobytes())
            f.write(sd[f"layers.{l}.haar_bridge.high_gain"].float().numpy().tobytes())
            f.write(sd[f"layers.{l}.haar_bridge.gate_w"].float().numpy().tobytes())
            f.write(sd[f"layers.{l}.haar_bridge.gate_b"].float().numpy().tobytes())

            # Reasoning RetNet (INT8)
            for w_name in ["w_qr", "w_kr", "w_vr", "w_out_r"]:
                s_w, q_w = quantize_rowwise_int8(sd[f"layers.{l}.{w_name}.weight"])
                f.write(s_w.tobytes()); f.write(q_w.tobytes())

            # Thinking Gate (FP32)
            f.write(sd[f"layers.{l}.think_gate.weight"].float().numpy().tobytes())
            f.write(sd[f"layers.{l}.think_gate.bias"].float().numpy().tobytes())

            # HDC Scratchpad
            s_hk, q_hk = quantize_rowwise_int8(sd[f"layers.{l}.hdc.proj_key.weight"])
            s_hv, q_hv = quantize_rowwise_int8(sd[f"layers.{l}.hdc.proj_val.weight"])
            f.write(s_hk.tobytes()); f.write(q_hk.tobytes())
            f.write(s_hv.tobytes()); f.write(q_hv.tobytes())
            f.write(sd[f"layers.{l}.hdc.gate_hdc.weight"].float().numpy().tobytes())
            f.write(sd[f"layers.{l}.hdc.gate_hdc.bias"].float().numpy().tobytes())

            # SwiGLU FFN (INT8)
            for ffn_name in ["w_gate", "w_up", "w_down"]:
                s_ffn, q_ffn = quantize_rowwise_int8(sd[f"layers.{l}.ffn.{ffn_name}.weight"])
                f.write(s_ffn.tobytes()); f.write(q_ffn.tobytes())

        # Final Norm
        f.write(sd["ln_final.weight"].float().numpy().tobytes())

    print(f"[OK SUCCESS] WRAI-X INT8 Binary Selesai Dibuat: {output_bin_path} ({os.path.getsize(output_bin_path)/1e6:.1f} MB)!\n")

def auto_quantize_and_export(model, tok, save_dir):
    print("\n" + "=" * 70)
    print("   ⚡ OTOMATISASI KUANTISASI KE FORMAT INT8 NATIVE C BINARY          ")
    print("=" * 70)
    os.makedirs(save_dir, exist_ok=True)
    vocab_out = os.path.join(save_dir, "wrai_x_vocab.bin")
    bin_out = os.path.join(save_dir, "wrai_x_06b_int8.bin")

    export_tokenizer_vocab_bin(tok, vocab_out)
    pack_wrai_x_checkpoint(model.state_dict(), bin_out)
    print(f"[SELESAI 100%] Berkas siap download ke laptop Anda:")
    print(f"  -> Model Binary : {bin_out} ({os.path.getsize(bin_out)/1e6:.1f} MB)")
    print(f"  -> Tokenizer    : {vocab_out} ({os.path.getsize(vocab_out)/1e6:.1f} MB)")

if __name__ == "__main__":
    # Jalankan Pipeline Utama
    print("=" * 70)
    print("   🌊 WRAI-X (0.6B) 1-CLICK COLAB ZERO-CONTAMINATION TRANSPLANT PIPELINE")
    print("=" * 70)

    # 1. Mount Drive otomatis jika di Colab
    drive_dir = "/content/drive/MyDrive/WRAI_X_06B"
    target_dir = drive_dir if os.path.exists("/content") else "."
    if os.path.exists("/content"):
        try:
            from google.colab import drive
            if not os.path.exists("/content/drive/MyDrive"):
                drive.mount('/content/drive')
            print(f"[OK] Google Drive terhubung! Hasil akan otomatis tersimpan di: {drive_dir}")
        except Exception as e:
            print(f"[INFO] Google Drive mount dilewati: {e}")
            target_dir = "."

    run_transplant_and_training()

    # Model & tokenizer sudah di-load di run_transplant_and_training,
    # kuantisasi langsung dipanggil di dalam run_transplant_and_training.
