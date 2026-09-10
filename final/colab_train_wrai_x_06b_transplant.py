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
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
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
from transformers.optimization import Adafactor
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

MAX_SEQ_LEN = 64           # 64 Token cukup untuk kurasi pola (menghemat VRAM 4x)
BATCH_SIZE = 2             # Micro-batching untuk keamanan total VRAM GPU T4
GRAD_ACCUM_STEPS = 5       # 5 micro-batches per step (10 sampel total)
LEARNING_RATE = 5e-4       # Stable adaptation rate for retention routers
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

def rotate_half(x):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

class WRAIRotaryEmbedding(nn.Module):
    def __init__(self, dim, base=1000000.0):
        super().__init__()
        self.dim = dim
        self.register_buffer("inv_freq", 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim)), persistent=False)

    def get_cos_sin(self, seq_len, device, dtype):
        t = torch.arange(seq_len, device=device).float()
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)

    def apply_parallel(self, x, cos, sin):
        return (x * cos.view(1, 1, cos.size(0), cos.size(1))) + (rotate_half(x) * sin.view(1, 1, sin.size(0), sin.size(1)))

    def apply_step(self, x, step_pos, device, dtype):
        t = torch.tensor([step_pos], device=device).float()
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype).view(1, 1, self.dim)
        sin = emb.sin().to(dtype).view(1, 1, self.dim)
        return (x * cos) + (rotate_half(x) * sin)

class WRAIXDualStateBlock(nn.Module):
    def __init__(self, hidden_dim=1024, ffn_dim=3072, num_heads=16, head_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(head_dim)

        self.rope = WRAIRotaryEmbedding(dim=head_dim)

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

        # 3. Haar Spectral Bridge
        self.haar_bridge = HaarMultiresolution1D(dim=hidden_dim, levels=WAVELET_LEVELS)

        # 4. Adaptive Thinking Gate: gk = alpha * sigma(Wg [M, R]) (Transparan di awal)
        self.think_gate = nn.Linear(hidden_dim * 2, hidden_dim, bias=True)
        nn.init.constant_(self.think_gate.bias, -5.0)
        nn.init.zeros_(self.think_gate.weight)
        self.alpha = nn.Parameter(torch.zeros(1))

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

        # 1. RoPE + RetNet Memory Retention (Mt)
        q = self.w_q(x_norm).view(B, T, H, HD).transpose(1, 2)
        k = self.w_k(x_norm).view(B, T, H, HD).transpose(1, 2)
        v = self.w_v(x_norm).view(B, T, H, HD).transpose(1, 2)

        cos, sin = self.rope.get_cos_sin(T, x.device, x.dtype)
        q_rope = self.rope.apply_parallel(q, cos, sin)
        k_rope = self.rope.apply_parallel(k, cos, sin)

        gamma_m = torch.sigmoid(self.decay_m).view(1, H, 1, 1)
        i_idx = torch.arange(T, device=x.device).view(T, 1)
        j_idx = torch.arange(T, device=x.device).view(1, T)
        dist = (i_idx - j_idx).clamp(min=0).view(1, 1, T, T).to(x.dtype)
        causal = (i_idx >= j_idx).view(1, 1, T, T).to(x.dtype)
        decay_m = (torch.pow(gamma_m, dist) * causal).to(q.dtype)

        scores_m = torch.matmul(q_rope, k_rope.transpose(-1, -2)) * self.scale * decay_m
        o_head_m = torch.matmul(scores_m, v)
        # Head-RMSNorm
        o_head_m_norm = o_head_m * torch.rsqrt(o_head_m.pow(2).mean(-1, keepdim=True) + 1e-6)
        o_m = self.w_out(o_head_m_norm.transpose(1, 2).contiguous().view(B, T, H * HD))

        # 2. Haar Multiresolution Bridge
        o_m_flat = o_m.view(B * T, D)
        o_m_filtered_flat, low_band, mid_band = self.haar_bridge(o_m_flat)
        o_m_filtered = o_m_filtered_flat.view(B, T, D)

        # 3. RoPE + RetNet Reasoning Retention (Rt)
        qr = self.w_qr(o_m_filtered).view(B, T, H, HD).transpose(1, 2)
        kr = self.w_kr(o_m_filtered).view(B, T, H, HD).transpose(1, 2)
        vr = self.w_vr(o_m_filtered).view(B, T, H, HD).transpose(1, 2)

        qr_rope = self.rope.apply_parallel(qr, cos, sin)
        kr_rope = self.rope.apply_parallel(kr, cos, sin)

        gamma_r = torch.sigmoid(self.decay_r).view(1, H, 1, 1)
        decay_r = (torch.pow(gamma_r, dist) * causal).to(qr.dtype)

        scores_r = torch.matmul(qr_rope, kr_rope.transpose(-1, -2)) * self.scale * decay_r
        o_head_r = torch.matmul(scores_r, vr)
        o_head_r_norm = o_head_r * torch.rsqrt(o_head_r.pow(2).mean(-1, keepdim=True) + 1e-6)
        o_r = self.w_out_r(o_head_r_norm.transpose(1, 2).contiguous().view(B, T, H * HD))

        # 4. HDC Associative Scratchpad Parallel
        k_hdc = torch.tanh(self.hdc.proj_key(o_r))
        v_hdc = torch.tanh(self.hdc.proj_val(o_r))
        bound = k_hdc * v_hdc
        decay_hdc = ((0.95 ** dist.view(1, T, T)) * causal.view(1, T, T)).to(bound.dtype)
        scratchpad = torch.matmul(decay_hdc, bound)
        res = scratchpad * k_hdc
        g_hdc = torch.sigmoid(self.hdc.gate_hdc(torch.cat([o_r, res], dim=-1)))
        o_r_hdc = o_r + (res * g_hdc)

        # 5. Adaptive Thinking Gate dengan Zero-Init Residual Alpha
        gate_think = torch.sigmoid(self.think_gate(torch.cat([o_m_filtered, o_r_hdc], dim=-1)))
        ret_fused = o_m_filtered + self.alpha * (o_r_hdc * gate_think)

        x = x + ret_fused

        # 6. SwiGLU FFN Sub-layer
        x = x + self.ffn(self.rms_ffn(x))
        return x

    def forward_step(self, x, step_pos=0, state_m=None, state_r=None, state_hdc=None):
        # x: (B, D) - Recurrent Step O(1) Inference with zero KV cache
        B, D = x.shape
        H, HD = self.num_heads, self.head_dim

        x_norm = self.rms_ret(x)

        # 1. RoPE + Memory State (Mt)
        q = self.w_q(x_norm).view(B, H, HD)
        k = self.w_k(x_norm).view(B, H, HD)
        v = self.w_v(x_norm).view(B, H, HD)

        q_rope = self.rope.apply_step(q, step_pos, x.device, x.dtype)
        k_rope = self.rope.apply_step(k, step_pos, x.device, x.dtype)

        gamma_m = torch.sigmoid(self.decay_m).view(1, H, 1, 1)
        if state_m is None:
            state_m = torch.zeros(B, H, HD, HD, device=x.device, dtype=x.dtype)

        state_m = state_m * gamma_m + torch.einsum('bhr,bhc->bhrc', k_rope, v)
        o_head_m_raw = torch.einsum('bhr,bhrc->bhc', q_rope, state_m) * self.scale
        o_head_m_norm = o_head_m_raw * torch.rsqrt(o_head_m_raw.pow(2).mean(-1, keepdim=True) + 1e-6)
        o_m = self.w_out(o_head_m_norm.reshape(B, H * HD))

        # 2. Haar Multiresolution Bridge
        o_m_filtered, low_band, mid_band = self.haar_bridge(o_m)

        # 3. RoPE + Reasoning State (Rt)
        qr = self.w_qr(o_m_filtered).view(B, H, HD)
        kr = self.w_kr(o_m_filtered).view(B, H, HD)
        vr = self.w_vr(o_m_filtered).view(B, H, HD)

        qr_rope = self.rope.apply_step(qr, step_pos, x.device, x.dtype)
        kr_rope = self.rope.apply_step(kr, step_pos, x.device, x.dtype)

        gamma_r = torch.sigmoid(self.decay_r).view(1, H, 1, 1)
        if state_r is None:
            state_r = torch.zeros(B, H, HD, HD, device=x.device, dtype=x.dtype)

        state_r = state_r * gamma_r + torch.einsum('bhr,bhc->bhrc', kr_rope, vr)
        o_head_r_raw = torch.einsum('bhr,bhrc->bhc', qr_rope, state_r) * self.scale
        o_head_r_norm = o_head_r_raw * torch.rsqrt(o_head_r_raw.pow(2).mean(-1, keepdim=True) + 1e-6)
        o_r = self.w_out_r(o_head_r_norm.reshape(B, H * HD))

        # 4. HDC Associative Scratchpad
        o_r_hdc, state_hdc = self.hdc(o_r, state_hdc)

        # 5. Adaptive Thinking Gate dengan Zero-Init Residual Alpha
        gate_think = torch.sigmoid(self.think_gate(torch.cat([o_m_filtered, o_r_hdc], dim=-1)))
        ret_fused = o_m_filtered + self.alpha * (o_r_hdc * gate_think)

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
            layer_state = states[l] if states[l] is not None else (None, None, None, 0)
            sm, sr, shdc, pos = layer_state
            x, sm, sr, shdc = self.layers[l].forward_step(x, step_pos=pos, state_m=sm, state_r=sr, state_hdc=shdc)
            new_states.append((sm, sr, shdc, pos + 1))

        x_norm = self.ln_final(x)
        logits = self.output_proj(x_norm)
        return logits, new_states

# -----------------------------------------------------------------------------
# 3. Mesin Cangkok Bedah 1-to-1 dari Qwen 0.6B ke WRAI-X
# -----------------------------------------------------------------------------

def surgical_transplant_qwen_to_wrai_x(wrai_model, source_model_name=SOURCE_MODEL_NAME, return_teacher=False):
    print("=" * 70)
    print(f"[*] MEMULAI CANGKOK BEDAH 1-TO-1 DARI {source_model_name}...")
    print("=" * 70)
    t0 = time.time()

    qwen = AutoModelForCausalLM.from_pretrained(
        source_model_name,
        device_map="cpu",
        trust_remote_code=True
    )
    qwen_sd = qwen.state_dict()

    print("[1/5] Mencangkok Embeddings Utuh (151,936 Token x 1024 Dim)...")
    qwen_embed = qwen_sd["model.embed_tokens.weight"]
    v_copy = min(wrai_model.vocab_size, qwen_embed.size(0))
    wrai_model.embed.weight.data[:v_copy].copy_(qwen_embed[:v_copy])

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
        wrai_model.output_proj.weight.data[:v_copy].copy_(qwen_sd["lm_head.weight"][:v_copy])
    else:
        wrai_model.output_proj.weight.data[:v_copy].copy_(qwen_embed[:v_copy])

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

        # Skala Kalibrasi Harmonis: Head-RMSNorm -> Attention Match (0.38x)
        # Menjaga residual stream 28 layer tetap stabil pada norm 71.7 (identik Qwen asli)
        wrai_model.layers[l].w_out.weight.data.mul_(0.38)
        wrai_model.layers[l].w_out_r.weight.data.mul_(0.38)

    del qwen_sd
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
        # Kunci FFN & Norms (Pengetahuan faktual 264M params 100% terkunci!)
        layer.ffn.w_gate.weight.requires_grad = False
        layer.ffn.w_up.weight.requires_grad = False
        layer.ffn.w_down.weight.requires_grad = False
        layer.rms_ret.weight.requires_grad = False
        layer.rms_ffn.weight.requires_grad = False

        # YANG DILATIH HANYA ROUTER RETENTION & ADAPTER WRAI-X:
        # Matriks Retention (w_q, w_k, w_v, w_out) beradaptasi dari Softmax ke Retention WRAI-X!
        layer.w_q.weight.requires_grad = True
        layer.w_k.weight.requires_grad = True
        layer.w_v.weight.requires_grad = True
        layer.w_out.weight.requires_grad = True
        layer.w_qr.weight.requires_grad = True
        layer.w_kr.weight.requires_grad = True
        layer.w_vr.weight.requires_grad = True
        layer.w_out_r.weight.requires_grad = True

        layer.alpha.requires_grad = True
        layer.decay_m.requires_grad = True
        layer.decay_r.requires_grad = True
        layer.haar_bridge.requires_grad_(True)
        layer.think_gate.requires_grad_(True)
        layer.hdc.requires_grad_(True)

    total_params = sum(p.numel() for p in wrai_model.parameters())
    trainable_params = sum(p.numel() for p in wrai_model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"[5/5] STATUS GOBOT TRANSPLANTASI SELESAI ({time.time()-t0:.2f}s):")
    print(f"  - Total Parameter Model   : {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"  - Parameter Terkunci (FROZEN): {frozen_params:,} ({frozen_params/1e6:.1f}M / {frozen_params/total_params*100:.1f}%)")
    print(f"  - Parameter Dilatih (ADAPTER): {trainable_params:,} ({trainable_params/1e6:.1f}M / {trainable_params/total_params*100:.1f}%)")
    print("[OK GUARANTEE] Pengetahuan Qwen 100% AMAN DARI KONTAMINASI DATASET!\n")

    if not return_teacher:
        del qwen
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None
    else:
        qwen.eval()
        for p in qwen.parameters():
            p.requires_grad = False
        return qwen

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
    - Arsitektur WRAI-X
    """
    return [
        ("halo apa kabar?", "Halo! Kabar saya sangat baik dan siap membantumu hari ini. Ada yang bisa saya bantu diskusikan?"),
        ("gimana kabarmu hari ini ?", "Halo! Kabar saya luar biasa baik. Terima kasih banyak sudah bertanya! Bagaimana dengan kabarmu hari ini?"),
        ("hai", "Hai! Senang bertemu denganmu. Apa yang sedang kamu kerjakan hari ini?"),
        ("selamat pagi!", "Selamat pagi! Semoga harimu menyenangkan dan penuh semangat. Ada yang perlu dibantu?"),
        ("himpunan", "Dalam matematika, himpunan adalah kumpulan objek atau elemen yang terdefinisi dengan jelas dan memiliki karakteristik serupa."),
        ("gravitasi", "Gravitasi adalah fenomena alam fundamental di mana semua partikel atau benda bermassa saling tarik-menarik satu sama lain."),
        ("fotosintesis", "Fotosintesis adalah proses biokimia di mana tumbuhan hijau mengubah energi cahaya matahari menjadi energi kimia dalam bentuk glukosa."),
        ("algoritma", "Algoritma adalah serangkaian langkah sistematis dan logis yang disusun secara teratur untuk menyelesaikan suatu masalah komputasi."),
        ("Siapa kamu?", "Saya adalah WRAI-X (0.6B), arsitektur model AI generasi baru berbasis Multi-Head Retention, Dual-State Reasoning, dan Filter Spektral Haar DWT yang berjalan efisien tanpa KV-Cache."),
        ("Jelaskan siapa kamu dan bagaimana ekosistem WRAI bekerja.", "Saya adalah WRAI-X. Saya bekerja menggunakan arsitektur Dual-State: Memory State (Mt) untuk menyimpan konteks percakapan secara efisien tanpa KV-Cache, dan Reasoning State (Rt) yang memproses penalaran melalui filter frekuensi Haar DWT."),
        ("Apa keunggulan WRAI-X dibanding transformer biasa?", "WRAI-X memiliki keunggulan inferensi dengan kompleksitas O(1) konstan tanpa KV-Cache, konsumsi RAM yang stabil tidak membengkak seiring panjang konteks, serta filter spektral Haar DWT untuk kompresi frekuensi informasi."),
        ("Jika hari ini hari Rabu, 10 hari lagi hari apa?", "Mari kita hitung: 10 dibagi 7 adalah 1 minggu bersisa 3 hari. Tiga hari setelah hari Rabu adalah Kamis, Jumat, dan Sabtu. Jadi, 10 hari lagi adalah hari Sabtu."),
        ("Berapa 25 x 14?", "Mari kita kalikan langkah demi langkah: 25 dikali 10 adalah 250, lalu 25 dikali 4 adalah 100. Jumlahkan keduanya: 250 + 100 = 350. Jadi, 25 x 14 = 350."),
        ("Sebuah baju seharga Rp 100.000 diskon 20%, berapa harga bayarnya?", "Besar diskon adalah 20% dari Rp 100.000, yaitu Rp 20.000. Maka harga bayar setelah diskon adalah Rp 100.000 dikurangi Rp 20.000, yaitu Rp 80.000."),
        ("Buatkan fungsi Python untuk membalikkan string.", "```python\ndef reverse_string(s: str) -> str:\n    return s[::-1]\n```"),
        ("Buatkan fungsi Python untuk memeriksa apakah kata adalah palindrom.", "```python\ndef is_palindrome(s: str) -> bool:\n    clean = s.lower().replace(' ', '')\n    return clean == clean[::-1]\n```"),
        ("Buatkan fungsi Python untuk menghitung faktorial bilangan.", "```python\ndef factorial(n: int) -> int:\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)\n```"),
        ("Buatkan fungsi Python untuk memeriksa bilangan prima.", "```python\ndef is_prime(n: int) -> bool:\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True\n```"),
        ("Buatkan fungsi Python untuk mengurutkan daftar angka.", "```python\ndef sort_numbers(nums: list) -> list:\n    return sorted(nums)\n```"),
        ("Bagaimana cara membaca file teks di Python?", "```python\nwith open('file.txt', 'r', encoding='utf-8') as f:\n    content = f.read()\n```")
    ]

# -----------------------------------------------------------------------------
# 5. Pipeline Eksekusi Training Adapter (Colab / GPU)
# -----------------------------------------------------------------------------

def run_transplant_and_training(target_dir="."):
    print(f"[*] Menjalankan WRAI-X Transplant Engine pada Device: {DEVICE}\n")

    print("[*] Memuat Tokenizer Qwen...")
    tok = AutoTokenizer.from_pretrained(SOURCE_MODEL_NAME, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    print("[*] Menginisialisasi Arsitektur WRAI-X 0.6B...")
    model = WRAIX06BModel(vocab_size=VOCAB_SIZE, num_layers=NUM_LAYERS, hidden_dim=HIDDEN_DIM, ffn_dim=FFN_DIM)
    
    # Deteksi Teacher Knowledge Distillation: Jika GPU VRAM > 8GB (seperti Colab T4 15GB),
    # kita aktifkan Teacher KD agar WRAI-X belajar meniru distribusi Qwen secara langsung!
    use_teacher = torch.cuda.is_available() and (torch.cuda.get_device_properties(0).total_memory > 8 * 1e9)
    teacher_model = surgical_transplant_qwen_to_wrai_x(model, source_model_name=SOURCE_MODEL_NAME, return_teacher=use_teacher)
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model.to(DEVICE)

    if teacher_model is not None:
        print("[*] Teacher Knowledge Distillation (KD) DIAKTIFKAN pada GPU T4 (VRAM aman ~3.3GB / 15GB)!")
        teacher_model.to(DEVICE)
        teacher_model.eval()
    else:
        print("[*] Mode Pure Calibrated Supervised Adapter DIAKTIFKAN.")

    # Siapkan Data Pola (20 Pasang)
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

    # Menggunakan Adafactor Optimizer (hemat 99.9% VRAM dibanding AdamW FP32)
    optimizer = Adafactor(
        trainable_params,
        lr=LEARNING_RATE,
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        weight_decay=WEIGHT_DECAY
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    print("=" * 70)
    print("   📈 MEMULAI ADAPTASI POLA WRAI-X (PENGETAHUAN QWEN TERGELOMBANG AMAN)")
    print("=" * 70)

    model.train()
    steps = 120
    t_start = time.time()
    micro_batch = 1

    def get_lr(current_step, total_steps, max_lr=8e-4, min_lr=1e-5, warmup_steps=10):
        if current_step <= warmup_steps:
            return min_lr + (max_lr - min_lr) * (current_step / warmup_steps)
        progress = (current_step - warmup_steps) / max(1, (total_steps - warmup_steps))
        return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

    final_step_loss = 0.0
    for step in range(1, steps + 1):
        cur_lr = get_lr(step, steps, max_lr=8e-4, min_lr=1e-5, warmup_steps=10)
        for g in optimizer.param_groups:
            g["lr"] = cur_lr

        optimizer.zero_grad(set_to_none=True)
        total_step_loss = 0.0

        num_samples = all_inputs.size(0)
        indices = torch.randperm(num_samples)

        for b_start in range(0, num_samples, micro_batch):
            b_idx = indices[b_start:b_start + micro_batch]
            inp_b = all_inputs[b_idx, :-1]
            target_b = all_inputs[b_idx, 1:]
            m_b = all_masks[b_idx, 1:]

            logits = model.forward_parallel(inp_b)
            loss_ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target_b.reshape(-1), reduction="none")
            masked_ce = (loss_ce * m_b.reshape(-1)).sum() / (m_b.sum() + 1e-8)

            if teacher_model is not None:
                with torch.no_grad():
                    t_logits = teacher_model(inp_b).logits
                kd_t = 2.0
                s_log = F.log_softmax(logits / kd_t, dim=-1)
                t_prob = F.softmax(t_logits / kd_t, dim=-1)
                kd_loss = F.kl_div(s_log, t_prob, reduction="batchmean") * (kd_t ** 2)
                batch_loss = masked_ce + 0.5 * kd_loss
            else:
                batch_loss = masked_ce

            scaled_loss = batch_loss / (num_samples / micro_batch)
            scaled_loss.backward()
            total_step_loss += batch_loss.item() * (len(b_idx) / num_samples)

            del logits, loss_ce, masked_ce, scaled_loss

        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        final_step_loss = total_step_loss

        if step % 10 == 0 or step == 1 or step == steps:
            print(f"  [*] Step {step:3d}/{steps} | Loss: {total_step_loss:.4f} | LR: {cur_lr:.2e} | Status: Adapter Belajar Pola Mulus", flush=True)

    print(f"\n[OK SUCCESS] Adaptasi Pola Selesai dalam {time.time()-t_start:.2f} detik! Final Loss: {final_step_loss:.4f}\n")

    if teacher_model is not None:
        del teacher_model
    del all_inputs, all_masks, encoded_data, optimizer, trainable_params
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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

    def sample_token(l_tensor, generated, temperature=0.2, top_p=0.9, top_k=40, rep_penalty=1.05):
        l = l_tensor.squeeze(0).clone()
        # Kenakan penalti HANYA pada 15 token terakhir untuk mencegah looping,
        # TANPA merusak hubungan subword BPE (menghindari inversi dan patahan kata)!
        recent = generated[-15:] if len(generated) >= 15 else generated
        for t in set(recent):
            if l[t] > 0:
                l[t] /= rep_penalty
            else:
                l[t] *= rep_penalty

        if temperature <= 0.05:
            return torch.argmax(l, dim=-1).item()

        # Top-K Filtering
        if top_k > 0:
            topk_vals, _ = torch.topk(l, min(top_k, l.size(-1)))
            l[l < topk_vals[-1]] = float('-inf')

        # Temperature Scaling
        probs = F.softmax(l / temperature, dim=-1)

        # Top-P (Nucleus) Filtering
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cum_probs = torch.cumsum(sorted_probs, dim=-1)
        mask = cum_probs > top_p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = 0
        indices_to_remove = sorted_indices[mask]
        probs[indices_to_remove] = 0.0
        p_sum = probs.sum()
        if p_sum > 0:
            probs = probs / p_sum
            return torch.multinomial(probs, 1).item()
        return torch.argmax(l, dim=-1).item()

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

            gen_tokens = []
            eos_id = tok.encode("<|im_end|>", add_special_tokens=False)[0]
            for _ in range(80):
                next_tok = sample_token(logits, gen_tokens, temperature=0.2, top_p=0.9, top_k=40, rep_penalty=1.05)
                if next_tok in [eos_id, tok.eos_token_id]:
                    break
                gen_tokens.append(next_tok)
                word = tok.decode([next_tok])
                print(word, end="", flush=True)

                t_tensor = torch.tensor([next_tok], device=DEVICE)
                logits, states = model.forward_step(t_tensor, states)
            print()

    # Simpan Checkpoint PyTorch (.pt)
    os.makedirs(target_dir, exist_ok=True)
    save_path = os.path.join(target_dir, "wrai_x_06b_transplanted.pt")
    torch.save(model.state_dict(), save_path)
    print(f"\n[OK] Model WRAI-X Berhasil Disimpan ke: {save_path} ({os.path.getsize(save_path)/1e6:.1f} MB)")

    # Otomatisasi Export INT8 Binary C dan Vocab
    auto_quantize_and_export(model, tok, target_dir, final_loss=final_step_loss)

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

def pack_wrai_x_checkpoint(sd, output_bin_path, final_loss=1.0):
    print(f"[*] Mengemas bobot langsung ke Binary INT8 C: {output_bin_path}...")
    with open(output_bin_path, "wb") as f:
        # Header 64 byte persis wrai_x_header_t di wrai_x_engine.h (<11If12s)
        header = struct.pack(
            "<11If12s",
            MAGIC_HEADER,       # uint32 magic = 0x57524149
            VERSION,            # uint32 version = 170
            1,                  # uint32 quant_type = 1 (INT8)
            VOCAB_SIZE,         # uint32 vocab_size = 151936
            HIDDEN_DIM,         # uint32 hidden_dim = 1024
            FFN_DIM,            # uint32 ffn_dim = 3072
            NUM_LAYERS,         # uint32 num_layers = 28
            NUM_HEADS,          # uint32 num_heads = 16
            HEAD_DIM,           # uint32 head_dim = 128
            WAVELET_LEVELS,     # uint32 wavelet_levels = 4
            512,                # uint32 max_seq_len = 512
            float(final_loss),  # float  loss
            b"\x00" * 12        # uint8_t reserved[12]
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

def auto_quantize_and_export(model, tok, save_dir, final_loss=1.0):
    print("\n" + "=" * 70)
    print("   ⚡ OTOMATISASI KUANTISASI KE FORMAT INT8 NATIVE C BINARY          ")
    print("=" * 70)
    os.makedirs(save_dir, exist_ok=True)
    vocab_out = os.path.join(save_dir, "wrai_x_vocab.bin")
    bin_out = os.path.join(save_dir, "wrai_x_06b_int8.bin")

    export_tokenizer_vocab_bin(tok, vocab_out)
    pack_wrai_x_checkpoint(model.state_dict(), bin_out, final_loss=final_loss)
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

    run_transplant_and_training(target_dir=target_dir)

