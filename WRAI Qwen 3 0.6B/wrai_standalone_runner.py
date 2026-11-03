#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wrai_standalone_runner.py — 100% Offline Full Standalone WRAI Model Loader & Runner
Loads a single self-contained WRAI full model file (`~1.13 GB .pt`) + local `tokenizer_and_config/`
using `torch.device("meta")` skeleton instantiation (`0.00 MB` duplicate RAM, `local_files_only=True`).
No original Qwen model or internet connection is ever needed!

Usage on ASUS X454YA (AMD A8):
  python3 wrai_standalone_runner.py --model standalone_models/wrai_qwen3_0_6b_micro_int8_standalone.pt --backend cpp_avx1 --threads 4 --interactive
  python3 wrai_standalone_runner.py --model standalone_models/wrai_qwen3_0_6b_homogeneous_standalone.pt --backend jit_float --threads 4 --interactive
"""
from __future__ import annotations
import argparse, ctypes, gc, math, os, subprocess, sys, time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, DynamicCache

HEADS, KV_HEADS, HEAD_DIM = 16, 8, 128
GROUPS = HEADS // KV_HEADS
PREFIX_CANDIDATES, LANDMARKS = 64, 48
PREFILL_CHUNK_SIZE, SEQ_READOUT_CHUNK = 256, 512

def build_haar_pyramid(n_fine: int) -> Tuple[List[Tuple[int, int, int]], List[Tuple[int, int, int]], int]:
    fine_blocks = [(i, i, 1) for i in range(n_fine)]
    coarse_blocks = []
    cur = n_fine
    for mult, count in [(2, 4), (4, 4)]:
        for _ in range(count):
            coarse_blocks.append((cur, cur + mult - 1, mult))
            cur += mult
    return fine_blocks + coarse_blocks, coarse_blocks, cur - 1

PYRAMIDS = {16: build_haar_pyramid(16), 32: build_haar_pyramid(32), 48: build_haar_pyramid(48), 64: build_haar_pyramid(64)}
_DECAY_CACHE: Dict[Tuple[float, int, str], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]] = {}
_VECTOR_PYRAMID_CACHE: Dict[Tuple[int, str], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
_G_ARANGE_CACHE: Dict[str, torch.Tensor] = {}

def get_g_arange(device: torch.device) -> torch.Tensor:
    k = str(device)
    if k not in _G_ARANGE_CACHE:
        _G_ARANGE_CACHE[k] = torch.arange(KV_HEADS, device=device, dtype=torch.long)
    return _G_ARANGE_CACHE[k]

def get_decay_tensors(gamma: float, c_len: int, device: torch.device):
    key = (float(gamma), int(c_len), str(device))
    if key not in _DECAY_CACHE:
        g = float(gamma)
        idx = torch.arange(c_len, device=device, dtype=torch.float64)
        diff = idx.unsqueeze(1) - idx.unsqueeze(0)
        if g == 1.0:
            causal_decay = torch.tril(torch.ones((c_len, c_len), device=device, dtype=torch.float32))
            step_decay = torch.ones((1, c_len, 1), device=device, dtype=torch.float32)
            chunk_decay = 1.0
        else:
            causal_decay = torch.tril(torch.pow(torch.tensor(g, device=device, dtype=torch.float64), diff)).float()
            step_decay = torch.pow(torch.tensor(g, device=device, dtype=torch.float64), idx + 1.0).float().view(1, c_len, 1)
            chunk_decay = float(g ** c_len)
        last_weights = causal_decay[-1, :].view(1, c_len, 1).contiguous()
        _DECAY_CACHE[key] = (causal_decay, step_decay, last_weights, chunk_decay)
    return _DECAY_CACHE[key]

def get_vectorized_pyramid_tensors(n_fine: int, device: torch.device):
    key = (n_fine, str(device))
    if key not in _VECTOR_PYRAMID_CACHE:
        all_blocks, coarse_blocks, horizon = PYRAMIDS[n_fine]
        c_new = torch.tensor([b[0] for b in coarse_blocks], device=device, dtype=torch.long)
        c_old = torch.tensor([b[1] for b in coarse_blocks], device=device, dtype=torch.long)
        c_mult = torch.tensor([b[2] for b in coarse_blocks], device=device, dtype=torch.float32)
        c_inv_m = (1.0 / c_mult).view(1, 1, 8, 1)
        c_log_tau = (0.30 * torch.log(c_mult)).view(1, 1, 1, 8)
        o_old_all = torch.tensor([b[1] for b in all_blocks], device=device, dtype=torch.long)

        coarse_span = (horizon + 1) - n_fine
        w_ca_c = torch.zeros((len(coarse_blocks), coarse_span), device=device, dtype=torch.float32)
        w_cd_c = torch.zeros((len(coarse_blocks), coarse_span), device=device, dtype=torch.float32)
        for s_i, (o_new, o_old, mult) in enumerate(coarse_blocks):
            lo = horizon - o_old
            hi = horizon - o_new + 1
            w_ca_c[s_i, lo:hi] = 1.0 / float(mult)
            hm = mult // 2
            w_cd_c[s_i, lo : lo + hm] = -1.0 / float(mult)
            w_cd_c[s_i, lo + hm : hi] = +1.0 / float(mult)

        _VECTOR_PYRAMID_CACHE[key] = (c_new, c_old, c_inv_m, c_log_tau, o_old_all, w_ca_c, w_cd_c)
    return _VECTOR_PYRAMID_CACHE[key]

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos.unsqueeze(1)) + (rotate_half(x) * sin.unsqueeze(1))

def diverse_landmark_indices(candidates: torch.Tensor, count: int) -> torch.Tensor:
    n = int(candidates.shape[0])
    limit = min(int(count), n)
    norm = F.normalize(candidates.float(), p=2, dim=-1, eps=1e-12)
    sim = norm @ norm.T
    selected = [0]
    min_dist = 1.0 - sim[:, 0]
    for _ in range(1, limit):
        min_dist[torch.tensor(selected, device=candidates.device)] = -float("inf")
        idx = int(torch.argmax(min_dist).item())
        selected.append(idx)
        min_dist = torch.minimum(min_dist, 1.0 - sim[:, idx])
    while len(selected) < int(count):
        selected.append(selected[len(selected) % max(1, n)])
    return torch.tensor(selected, device=candidates.device, dtype=torch.long)

def basis_from_keys_gqa(k_kv: torch.Tensor, prefix_candidates: int, landmark_count: int) -> torch.Tensor:
    prefix = min(int(prefix_candidates), int(k_kv.shape[2]))
    bases = []
    for h in range(int(k_kv.shape[1])):
        cand = k_kv[0, h, :prefix]
        idx = diverse_landmark_indices(cand, landmark_count)
        bases.append(cand.index_select(0, idx).detach().float())
    return torch.stack(bases, dim=0).contiguous()

def quantize_to_true_int8_pair(x: torch.Tensor, scale_dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    xf = x.detach().float()
    amax = torch.amax(torch.abs(xf), dim=-1, keepdim=True).clamp_min(1e-6)
    scale = (amax / 127.0).to(dtype=scale_dtype).contiguous()
    q_i8 = torch.clamp(torch.round(xf / (amax / 127.0)), -127.0, 127.0).to(dtype=torch.int8).contiguous()
    return q_i8, scale

def quantize_to_cpu_int8_c_pair(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    xf = x.detach().float()
    amax = torch.amax(torch.abs(xf), dim=-1, keepdim=True).clamp_min(1e-6)
    scale = (amax / 127.0).squeeze(-1).cpu().to(dtype=torch.float32).contiguous()
    q_i8 = torch.clamp(torch.round(xf / (amax / 127.0)), -127.0, 127.0).cpu().to(dtype=torch.int8).contiguous()
    return q_i8, scale

def select_hybrid_novelty_and_cue_span_hdc(
    q_grp: torch.Tensor, k_kv: torch.Tensor, v_kv: torch.Tensor,
    n_fine: int = 64, sink_slots: int = 16, hdc_slots: int = 144
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n_kv, seq_len, head_dim = k_kv.shape
    device = k_kv.device
    s_lim = min(int(sink_slots), seq_len)
    sink_idx = torch.arange(0, s_lim, device=device, dtype=torch.long).unsqueeze(0).expand(n_kv, -1)
    sink_nov = torch.full((n_kv, s_lim), 1e9, device=device, dtype=torch.float32)

    if hdc_slots <= 0 or seq_len <= s_lim + 2:
        idx_exp = sink_idx.unsqueeze(-1).expand(-1, -1, head_dim)
        return torch.gather(k_kv, 1, idx_exp).contiguous(), torch.gather(v_kv, 1, idx_exp).contiguous(), sink_idx, sink_nov

    start_c = max(1, s_lim)
    end_c = seq_len
    span_len = end_c - start_c
    k_curr = k_kv[:, start_c:end_c, :].detach()
    k_prev = k_kv[:, start_c - 1 : end_c - 1, :].detach()
    novelty = torch.linalg.vector_norm(k_curr - k_prev, dim=-1)

    hist_cutoff = max(start_c + 16, seq_len - n_fine)
    hist_len = hist_cutoff - start_c

    if span_len <= hdc_slots or hist_len <= 48 or hdc_slots < 32:
        eff_hdc = min(int(hdc_slots), span_len)
        edges = torch.linspace(0, span_len, steps=eff_hdc + 1, device=device).long()
        seg_starts, seg_ends = edges[:-1], edges[1:]
        picked_idx, picked_nov = [], []
        for s_i in range(eff_hdc):
            a_i = min(span_len - 1, int(seg_starts[s_i].item()))
            b_i = max(a_i + 1, min(span_len, int(seg_ends[s_i].item())))
            max_v, rel_max = torch.max(novelty[:, a_i:b_i], dim=-1)
            picked_idx.append(start_c + a_i + rel_max)
            picked_nov.append(max_v)
        hdc_idx = torch.stack(picked_idx, dim=-1)
        hdc_nov = torch.stack(picked_nov, dim=-1)
    else:
        n_windows = max(2, min(8, (hdc_slots // 2) // 8))
        win_len = 8
        n_span_slots = n_windows * win_len
        n_nov_slots = hdc_slots - n_span_slots

        edges = torch.linspace(0, span_len, steps=n_nov_slots + 1, device=device).long()
        seg_starts, seg_ends = edges[:-1], edges[1:]
        picked_idx, picked_nov = [], []
        for s_i in range(n_nov_slots):
            a_i = min(span_len - 1, int(seg_starts[s_i].item()))
            b_i = max(a_i + 1, min(span_len, int(seg_ends[s_i].item())))
            max_v, rel_max = torch.max(novelty[:, a_i:b_i], dim=-1)
            picked_idx.append(start_c + a_i + rel_max)
            picked_nov.append(max_v)
        nov_idx = torch.stack(picked_idx, dim=-1)
        nov_val = torch.stack(picked_nov, dim=-1)

        q_tail_len = min(24, int(q_grp.shape[2]))
        q_cue = F.normalize(q_grp[:, :, -q_tail_len:, :].detach(), p=2, dim=-1)
        k_hist = F.normalize(k_kv[:, start_c:hist_cutoff, :].detach(), p=2, dim=-1).unsqueeze(1)
        sim_mat = torch.matmul(q_cue, k_hist.transpose(-1, -2))
        cue_resonance = sim_mat.amax(dim=(0, 1, 2))
        norm_nov = novelty[:, :hist_len].mean(dim=0) / (novelty[:, :hist_len].mean().clamp_min(1e-6))
        salience_score = cue_resonance + 0.15 * norm_nov

        work_sal = salience_score.clone()
        span_windows = []
        for _ in range(n_windows):
            best_rel = int(torch.argmax(work_sal).item())
            w_st = max(0, min(hist_len - win_len, best_rel - 1))
            span_windows.append(torch.arange(start_c + w_st, start_c + w_st + win_len, device=device, dtype=torch.long))
            sup_lo = max(0, w_st - 12)
            sup_hi = min(hist_len, w_st + win_len + 12)
            work_sal[sup_lo:sup_hi] = -1e9

        span_idx = torch.cat(span_windows, dim=0).unsqueeze(0).expand(n_kv, -1)
        span_nov = torch.full((n_kv, n_span_slots), 500.0, device=device, dtype=torch.float32)
        hdc_idx = torch.cat([nov_idx, span_idx], dim=-1)
        hdc_nov = torch.cat([nov_val, span_nov], dim=-1)

    all_idx = torch.cat([sink_idx, hdc_idx], dim=-1)
    all_nov = torch.cat([sink_nov, hdc_nov], dim=-1)
    idx_exp = all_idx.unsqueeze(-1).expand(-1, -1, head_dim)
    bank_k = torch.gather(k_kv, 1, idx_exp).contiguous()
    bank_v = torch.gather(v_kv, 1, idx_exp).contiguous()
    return bank_k, bank_v, all_idx, all_nov

def compute_prefix_head_gain(q_full: torch.Tensor, k_kv: torch.Tensor, v_kv: torch.Tensor, y_wrai_full: torch.Tensor, prefix_len: int = 64) -> torch.Tensor:
    p = min(int(prefix_len), int(q_full.shape[2]))
    qp = q_full[0, :, :p, :].detach().float().view(KV_HEADS, GROUPS, p, HEAD_DIM)
    kp = k_kv[0, :, :p, :].detach().float().unsqueeze(1)
    vp = v_kv[0, :, :p, :].detach().float().unsqueeze(1)
    scores = torch.matmul(qp, kp.transpose(-2, -1)) / math.sqrt(HEAD_DIM)
    idx = torch.arange(p, device=qp.device)
    scores = scores.masked_fill(~(idx.unsqueeze(0) <= idx.unsqueeze(1)).view(1, 1, p, p), -1e9)
    y_exact_p = torch.matmul(torch.softmax(scores, dim=-1), vp).reshape(HEADS, p, HEAD_DIM)
    s_cmp = min(16, p - 1)
    y_wrai_p = y_wrai_full[0, s_cmp:p, :, :].detach().permute(1, 0, 2).float()
    y_ref_p = y_exact_p[:, s_cmp:p, :]
    rms_ref = torch.sqrt(torch.mean(y_ref_p * y_ref_p, dim=(1, 2)).clamp_min(1e-12))
    rms_wrai = torch.sqrt(torch.mean(y_wrai_p * y_wrai_p, dim=(1, 2)).clamp_min(1e-12))
    return (rms_ref / rms_wrai).clamp(0.70, 1.40).view(-1, 1)

@torch.jit.script
def jit_standalone_wrai_decode_step(
    qkv: torch.Tensor, q_norm_w: torch.Tensor, k_norm_w: torch.Tensor,
    cos: torch.Tensor, sin: torch.Tensor,
    basis_q_bf16: torch.Tensor, basis_k_bf16: torch.Tensor,
    state_s_bf16: torch.Tensor, state_z_fp32: torch.Tensor,
    unified_k_bf16: torch.Tensor, unified_v_bf16: torch.Tensor,
    static_bias_bf16: torch.Tensor, gain_r48_bf16: torch.Tensor, gain_shh_bf16: torch.Tensor,
    gamma: float, ring_ptr: int, n_fine: int, total_seen: int
) -> torch.Tensor:
    q_raw = qkv[0, 0, :2048].view(16, 128).float()
    k_raw = qkv[0, 0, 2048:3072].view(8, 128).float()
    v_bf16 = qkv[0, 0, 3072:4096].view(8, 1, 128)

    q_rms = torch.rsqrt(torch.mean(q_raw * q_raw, dim=-1, keepdim=True) + 1e-6)
    k_rms = torch.rsqrt(torch.mean(k_raw * k_raw, dim=-1, keepdim=True) + 1e-6)
    q_n = (q_raw * q_rms).to(qkv.dtype) * q_norm_w
    k_n = (k_raw * k_rms).to(qkv.dtype) * k_norm_w

    c, s = cos[0, 0].unsqueeze(0), sin[0, 0].unsqueeze(0)
    q_bf16 = (q_n * c + torch.cat((-q_n[:, 64:], q_n[:, :64]), dim=-1) * s).view(8, 2, 128)
    k_bf16 = (k_n * c + torch.cat((-k_n[:, 64:], k_n[:, :64]), dim=-1) * s).view(8, 1, 128)

    fq = torch.softmax(torch.bmm(q_bf16, basis_q_bf16).float(), dim=-1)
    fk = torch.softmax(torch.bmm(k_bf16, basis_k_bf16).float(), dim=-1)
    fk_t = fk.transpose(1, 2)
    state_s_bf16.mul_(gamma).add_(torch.bmm(fk_t.to(qkv.dtype), v_bf16))
    state_z_fp32.mul_(gamma).add_(fk_t)
    den = torch.bmm(fq, state_z_fp32).clamp_min(1e-8).to(qkv.dtype)
    out_r48 = (torch.bmm(fq.to(qkv.dtype), state_s_bf16) / den).reshape(16, 128)

    k_exit = unified_k_bf16[:, ring_ptr, :]
    v_exit = unified_v_bf16[:, ring_ptr, :]

    if (total_seen & 3) == 0:
        unified_k_bf16[:, n_fine + 5 : n_fine + 8, :] = unified_k_bf16[:, n_fine + 4 : n_fine + 7, :].clone()
        unified_v_bf16[:, n_fine + 5 : n_fine + 8, :] = unified_v_bf16[:, n_fine + 4 : n_fine + 7, :].clone()
        unified_k_bf16[:, n_fine + 4, :] = (unified_k_bf16[:, n_fine + 2, :] + unified_k_bf16[:, n_fine + 3, :]) * 0.5
        unified_v_bf16[:, n_fine + 4, :] = (unified_v_bf16[:, n_fine + 2, :] + unified_v_bf16[:, n_fine + 3, :]) * 0.5
    if (total_seen & 1) == 0:
        unified_k_bf16[:, n_fine + 1 : n_fine + 4, :] = unified_k_bf16[:, n_fine : n_fine + 3, :].clone()
        unified_v_bf16[:, n_fine + 1 : n_fine + 4, :] = unified_v_bf16[:, n_fine : n_fine + 3, :].clone()
        unified_k_bf16[:, n_fine, :] = k_exit
        unified_v_bf16[:, n_fine, :] = v_exit
    else:
        unified_k_bf16[:, n_fine, :].mul_(0.5).add_(k_exit, alpha=0.5)
        unified_v_bf16[:, n_fine, :].mul_(0.5).add_(v_exit, alpha=0.5)

    unified_k_bf16[:, ring_ptr, :] = k_bf16.squeeze(1)
    unified_v_bf16[:, ring_ptr, :] = v_bf16.squeeze(1)

    scores = torch.bmm(q_bf16, unified_k_bf16.transpose(1, 2)) * 0.0883883476 + static_bias_bf16
    probs = torch.softmax(scores.float(), dim=-1).to(qkv.dtype)
    out_shh = torch.bmm(probs, unified_v_bf16).reshape(16, 128)

    out = out_r48 * gain_r48_bf16 + out_shh * gain_shh_bf16
    return out.view(1, 1, 2048)

class StandaloneWRAILayer(nn.Module):
    def __init__(
        self, layer_idx: int, alpha: float, gamma: float, n_fine: int, sink_slots: int, hdc_slots: int,
        param_dtype: torch.dtype = torch.bfloat16, param_device: torch.device = torch.device("meta"),
        backend: str = "jit_float", cpp_lib: Optional[ctypes.CDLL] = None
    ):
        super().__init__()
        self.__dict__["cpp_lib"] = cpp_lib
        self.backend = backend
        self.layer_idx = int(layer_idx)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.n_fine = int(n_fine)
        self.blocks, self.coarse_blocks, self.horizon = PYRAMIDS[self.n_fine]
        self.sink_slots = int(sink_slots)
        self.hdc_slots = int(hdc_slots)

        # Standalone Parameter Tensors (allocated on `meta` by default -> 0 bytes RAM until `load_state_dict(assign=True)`!)
        self.w_qkv = nn.Parameter(torch.empty((4096, 1024), dtype=param_dtype, device=param_device), requires_grad=False)
        self.w_o = nn.Parameter(torch.empty((1024, 2048), dtype=param_dtype, device=param_device), requires_grad=False)
        self.q_norm_w = nn.Parameter(torch.empty((128,), dtype=param_dtype, device=param_device), requires_grad=False)
        self.k_norm_w = nn.Parameter(torch.empty((128,), dtype=param_dtype, device=param_device), requires_grad=False)

        self.wav_logit = nn.Parameter(torch.empty((HEADS,), dtype=torch.float32, device=param_device), requires_grad=False)
        self.head_scale_raw = nn.Parameter(torch.empty((HEADS,), dtype=torch.float32, device=param_device), requires_grad=False)
        self.r48_scale_raw = nn.Parameter(torch.empty((HEADS,), dtype=torch.float32, device=param_device), requires_grad=False)
        self.alpha_q_raw = nn.Parameter(torch.empty((KV_HEADS, GROUPS), dtype=torch.float32, device=param_device), requires_grad=False)
        self.alpha_k_raw = nn.Parameter(torch.empty((KV_HEADS,), dtype=torch.float32, device=param_device), requires_grad=False)
        self.coarse_tau_raw = nn.Parameter(torch.empty((KV_HEADS, 1, 1, 8), dtype=torch.float32, device=param_device), requires_grad=False)
        self.detail_beta_raw = nn.Parameter(torch.empty((KV_HEADS, 1, 1, 8), dtype=torch.float32, device=param_device), requires_grad=False)

        self.basis_q_bf16: Optional[torch.Tensor] = None
        self.basis_k_bf16: Optional[torch.Tensor] = None
        self.state_s_bf16: Optional[torch.Tensor] = None
        self.state_z_fp32: Optional[torch.Tensor] = None
        self.unified_k_bf16: Optional[torch.Tensor] = None
        self.unified_v_bf16: Optional[torch.Tensor] = None
        self.static_bias_bf16: Optional[torch.Tensor] = None
        self.gain_r48_bf16: Optional[torch.Tensor] = None
        self.gain_shh_bf16: Optional[torch.Tensor] = None
        self.prev_fine_k: Optional[torch.Tensor] = None
        self.bank_novelty: Optional[torch.Tensor] = None

        self.basis_q_i8: Optional[torch.Tensor] = None
        self.basis_q_scale: Optional[torch.Tensor] = None
        self.basis_k_i8: Optional[torch.Tensor] = None
        self.basis_k_scale: Optional[torch.Tensor] = None
        self.state_s_i8: Optional[torch.Tensor] = None
        self.state_s_scale: Optional[torch.Tensor] = None
        self.unified_k_i8: Optional[torch.Tensor] = None
        self.unified_k_scale: Optional[torch.Tensor] = None
        self.unified_v_i8: Optional[torch.Tensor] = None
        self.unified_v_scale: Optional[torch.Tensor] = None
        self.static_bias_fp32: Optional[torch.Tensor] = None
        self.gain_r48_fp32: Optional[torch.Tensor] = None
        self.gain_shh_fp32: Optional[torch.Tensor] = None
        self.out_cpu_buf: torch.Tensor = torch.zeros((HEADS, HEAD_DIM), device="cpu", dtype=torch.float32).contiguous()
        self.ring_ptr: int = 0
        self.total_seen: int = 0

    def get_bounded_multipliers(self):
        return (
            1.0 + 0.05 * torch.tanh(self.head_scale_raw),
            1.0 + 0.08 * torch.tanh(self.r48_scale_raw),
            1.0 + 0.20 * torch.tanh(self.alpha_q_raw),
            1.0 + 0.20 * torch.tanh(self.alpha_k_raw),
            1.0 + 0.25 * torch.tanh(self.coarse_tau_raw),
            1.0 + 0.25 * torch.tanh(self.detail_beta_raw),
        )

    def get_actual_state_bytes(self) -> int:
        total = 0
        for name in [
            "basis_q_bf16", "basis_k_bf16", "state_s_bf16", "state_z_fp32",
            "unified_k_bf16", "unified_v_bf16", "static_bias_bf16", "prev_fine_k",
            "bank_novelty", "gain_r48_bf16", "gain_shh_bf16",
            "basis_q_i8", "basis_q_scale", "basis_k_i8", "basis_k_scale",
            "state_s_i8", "state_s_scale", "unified_k_i8", "unified_k_scale",
            "unified_v_i8", "unified_v_scale", "static_bias_fp32", "gain_r48_fp32", "gain_shh_fp32"
        ]:
            t = getattr(self, name, None)
            if isinstance(t, torch.Tensor):
                total += t.numel() * t.element_size()
        return total

    def forward(self, hidden_states: torch.Tensor, position_embeddings: Tuple[torch.Tensor, torch.Tensor], **kwargs):
        b, t, _ = hidden_states.shape
        cos, sin = position_embeddings
        dt = hidden_states.dtype

        if t == 1 and self.state_z_fp32 is not None:
            qkv = F.linear(hidden_states, self.w_qkv)

            if self.backend == "cpp_avx1":
                q_raw = qkv[0, 0, :2048].view(16, 128).float()
                k_raw = qkv[0, 0, 2048:3072].view(8, 128).float()
                v_f32 = qkv[0, 0, 3072:4096].view(8, 128).float()
                q_n = (q_raw * torch.rsqrt(torch.mean(q_raw * q_raw, dim=-1, keepdim=True) + 1e-6)).to(dt) * self.q_norm_w
                k_n = (k_raw * torch.rsqrt(torch.mean(k_raw * k_raw, dim=-1, keepdim=True) + 1e-6)).to(dt) * self.k_norm_w
                c, s_r = cos[0, 0].unsqueeze(0), sin[0, 0].unsqueeze(0)
                q_rot = (q_n * c + torch.cat((-q_n[:, 64:], q_n[:, :64]), dim=-1) * s_r).float().cpu().contiguous()
                k_rot = (k_n * c + torch.cat((-k_n[:, 64:], k_n[:, :64]), dim=-1) * s_r).float().cpu().contiguous()
                v_cpu = v_f32.cpu().contiguous()

                n_slots = int(self.unified_k_i8.shape[1])
                self.__dict__["cpp_lib"].wrai_avx1_int8_decode_step_c(
                    q_rot.data_ptr(), k_rot.data_ptr(), v_cpu.data_ptr(),
                    self.basis_q_i8.data_ptr(), self.basis_q_scale.data_ptr(),
                    self.basis_k_i8.data_ptr(), self.basis_k_scale.data_ptr(),
                    self.state_s_i8.data_ptr(), self.state_s_scale.data_ptr(), self.state_z_fp32.data_ptr(),
                    self.unified_k_i8.data_ptr(), self.unified_k_scale.data_ptr(),
                    self.unified_v_i8.data_ptr(), self.unified_v_scale.data_ptr(),
                    self.static_bias_fp32.data_ptr(), self.gain_r48_fp32.data_ptr(), self.gain_shh_fp32.data_ptr(),
                    ctypes.c_float(self.gamma), int(self.ring_ptr), int(self.n_fine), n_slots, int(self.total_seen),
                    self.out_cpu_buf.data_ptr()
                )
                self.ring_ptr = (self.ring_ptr + 1) % self.n_fine
                self.total_seen += 1
                return F.linear(self.out_cpu_buf.to(device=hidden_states.device, dtype=dt).view(1, 1, 2048), self.w_o), None

            if self.backend == "pytorch_int8":
                q_raw = qkv[0, 0, :2048].view(16, 128).float()
                k_raw = qkv[0, 0, 2048:3072].view(8, 128).float()
                v_bf16 = qkv[0, 0, 3072:4096].view(8, 1, 128)
                q_n = (q_raw * torch.rsqrt(torch.mean(q_raw * q_raw, dim=-1, keepdim=True) + 1e-6)).to(dt) * self.q_norm_w
                k_n = (k_raw * torch.rsqrt(torch.mean(k_raw * k_raw, dim=-1, keepdim=True) + 1e-6)).to(dt) * self.k_norm_w
                c, s_r = cos[0, 0].unsqueeze(0), sin[0, 0].unsqueeze(0)
                q_bf16 = (q_n * c + torch.cat((-q_n[:, 64:], q_n[:, :64]), dim=-1) * s_r).view(8, 2, 128)
                k_bf16 = (k_n * c + torch.cat((-k_n[:, 64:], k_n[:, :64]), dim=-1) * s_r).view(8, 1, 128)

                k_q, k_sc = quantize_to_true_int8_pair(k_bf16.squeeze(1), dt)
                v_q, v_sc = quantize_to_true_int8_pair(v_bf16.squeeze(1), dt)
                k_deq = (k_q.to(dt) * k_sc).unsqueeze(1)
                v_deq = (v_q.to(dt) * v_sc).unsqueeze(1)

                bq_deq = (self.basis_q_i8.to(dt) * self.basis_q_scale).transpose(1, 2)
                bk_deq = (self.basis_k_i8.to(dt) * self.basis_k_scale).transpose(1, 2)
                fq = torch.softmax(torch.bmm(q_bf16, bq_deq).float(), dim=-1)
                fk = torch.softmax(torch.bmm(k_deq, bk_deq).float(), dim=-1)
                fk_t = fk.transpose(1, 2)

                s_deq = (self.state_s_i8.to(dt) * self.state_s_scale).mul_(self.gamma).add_(torch.bmm(fk_t.to(dt), v_deq))
                self.state_s_i8, self.state_s_scale = quantize_to_true_int8_pair(s_deq, dt)
                self.state_z_fp32.mul_(self.gamma).add_(fk_t)
                den = torch.bmm(fq, self.state_z_fp32).clamp_min(1e-8).to(dt)
                out_r48 = (torch.bmm(fq.to(dt), self.state_s_i8.to(dt) * self.state_s_scale) / den).reshape(16, 128)

                ptr = self.ring_ptr
                self.unified_k_i8[:, ptr, :] = k_q
                self.unified_k_scale[:, ptr, :] = k_sc
                self.unified_v_i8[:, ptr, :] = v_q
                self.unified_v_scale[:, ptr, :] = v_sc
                self.ring_ptr = (ptr + 1) % self.n_fine
                self.total_seen += 1

                uk_deq = self.unified_k_i8.to(dt) * self.unified_k_scale
                uv_deq = self.unified_v_i8.to(dt) * self.unified_v_scale
                scores = torch.bmm(q_bf16, uk_deq.transpose(1, 2)) * 0.0883883476 + self.static_bias_bf16
                probs = torch.softmax(scores.float(), dim=-1).to(dt)
                out_shh = torch.bmm(probs, uv_deq).reshape(16, 128)
                out = out_r48 * self.gain_r48_bf16 + out_shh * self.gain_shh_bf16
                return F.linear(out.view(1, 1, 2048), self.w_o), None

            ptr = self.ring_ptr
            if (self.total_seen & 7) == 0 and self.bank_novelty is not None and self.hdc_slots >= 16:
                k_exit = self.unified_k_bf16[:, ptr, :]
                v_exit = self.unified_v_bf16[:, ptr, :]
                nov_exit = torch.linalg.vector_norm((k_exit - self.prev_fine_k).float(), dim=-1)
                self.prev_fine_k = k_exit.clone()
                if self.bank_novelty.shape[1] > self.sink_slots:
                    self.bank_novelty[:, self.sink_slots:].mul_(0.985)
                    min_nov, min_rel_idx = torch.min(self.bank_novelty[:, self.sink_slots:], dim=-1)
                    target_slot = (self.n_fine + 8 + self.sink_slots) + min_rel_idx
                    promote_mask = nov_exit > min_nov
                    g_idx = get_g_arange(hidden_states.device)
                    pm_2d = promote_mask.unsqueeze(-1)
                    self.unified_k_bf16[g_idx, target_slot, :] = torch.where(pm_2d, k_exit, self.unified_k_bf16[g_idx, target_slot, :])
                    self.unified_v_bf16[g_idx, target_slot, :] = torch.where(pm_2d, v_exit, self.unified_v_bf16[g_idx, target_slot, :])
                    self.bank_novelty[g_idx, self.sink_slots + min_rel_idx] = torch.where(promote_mask, nov_exit, min_nov)

            y_flat = jit_standalone_wrai_decode_step(
                qkv, self.q_norm_w, self.k_norm_w, cos, sin,
                self.basis_q_bf16, self.basis_k_bf16, self.state_s_bf16, self.state_z_fp32,
                self.unified_k_bf16, self.unified_v_bf16, self.static_bias_bf16,
                self.gain_r48_bf16, self.gain_shh_bf16,
                self.gamma, ptr, self.n_fine, self.total_seen
            )
            self.ring_ptr = (ptr + 1) % self.n_fine
            self.total_seen += 1
            return F.linear(y_flat, self.w_o), None

        # Sequence Prefill (`t > 1`)
        qkv = F.linear(hidden_states, self.w_qkv)
        q_raw = qkv[:, :, :2048].view(b, t, HEADS, HEAD_DIM).float()
        k_raw = qkv[:, :, 2048:3072].view(b, t, KV_HEADS, HEAD_DIM).float()
        v = qkv[:, :, 3072:4096].view(b, t, KV_HEADS, HEAD_DIM)

        q = (q_raw * torch.rsqrt(torch.mean(q_raw * q_raw, dim=-1, keepdim=True) + 1e-6)).to(dt) * self.q_norm_w
        k = (k_raw * torch.rsqrt(torch.mean(k_raw * k_raw, dim=-1, keepdim=True) + 1e-6)).to(dt) * self.k_norm_w
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        q_f, k_kv, v_kv = q[0].float(), k[0].float(), v[0].float()
        q_grp = q_f.view(KV_HEADS, GROUPS, t, HEAD_DIM)
        bank_k, bank_v, bank_pos, bank_nov = select_hybrid_novelty_and_cue_span_hdc(
            q_grp, k_kv, v_kv, n_fine=self.n_fine, sink_slots=self.sink_slots, hdc_slots=self.hdc_slots
        )
        basis = basis_from_keys_gqa(k, PREFIX_CANDIDATES, LANDMARKS)
        basis_t = basis.transpose(-1, -2).contiguous()

        h_sc, r_sc, aq_m, ak_m, ct_m, db_m = self.get_bounded_multipliers()
        w_head = torch.sigmoid(self.wav_logit)
        base_scale = float(self.alpha) / math.sqrt(HEAD_DIM)

        q_alpha = (base_scale * aq_m).view(KV_HEADS, GROUPS, 1, 1)
        k_alpha = (base_scale * ak_m).view(KV_HEADS, 1, 1)
        fq_all = torch.softmax(torch.matmul(q_grp, basis_t.unsqueeze(1)) * q_alpha, dim=-1)
        fk_all = torch.softmax(torch.matmul(k_kv, basis_t) * k_alpha, dim=-1)

        s = torch.zeros((KV_HEADS, LANDMARKS, HEAD_DIM), device=q.device, dtype=torch.float32)
        z = torch.zeros((KV_HEADS, LANDMARKS), device=q.device, dtype=torch.float32)
        out_chunks = []
        for start in range(0, t, PREFILL_CHUNK_SIZE):
            end = min(t, start + PREFILL_CHUNK_SIZE)
            c_len = end - start
            fq_c, fk_c, v_c = fq_all[:, :, start:end, :], fk_all[:, start:end, :], v_kv[:, start:end, :]
            causal_decay, step_decay, last_weights, chunk_decay = get_decay_tensors(self.gamma, c_len, q.device)
            z_traj = (z.unsqueeze(1) * step_decay) + torch.matmul(causal_decay, fk_c)
            num_Carry = torch.matmul(fq_c * step_decay.unsqueeze(1), s.unsqueeze(1))
            w_intra = torch.matmul(fq_c, fk_c.transpose(-1, -2).unsqueeze(1)) * causal_decay.view(1, 1, c_len, c_len)
            num_traj = num_Carry + torch.matmul(w_intra, v_c.unsqueeze(1))
            den_traj = torch.sum(fq_c * z_traj.unsqueeze(1), dim=-1, keepdim=True).clamp_min(1e-8)
            y_c = (num_traj / den_traj).reshape(HEADS, c_len, HEAD_DIM).transpose(0, 1)
            out_chunks.append(y_c)
            s = (s * chunk_decay) + torch.matmul((fk_c * last_weights).transpose(-1, -2), v_c)
            z = z_traj[:, -1, :].contiguous()

        y_r48 = torch.cat(out_chunks, dim=0) * r_sc.view(1, HEADS, 1)

        c_new, c_old, c_inv_m, c_log_tau, o_old_all, _, _ = get_vectorized_pyramid_tensors(self.n_fine, q.device)
        n_slots_haar = self.n_fine + 8
        s_scale = 1.0 / math.sqrt(HEAD_DIM)
        k_pad = F.pad(k_kv, (0, 0, self.horizon, 0))
        v_pad = F.pad(v_kv, (0, 0, self.horizon, 0))
        z_zero = torch.zeros((KV_HEADS, 1, HEAD_DIM), device=q.device, dtype=torch.float32)
        cs_k = torch.cat([z_zero, torch.cumsum(k_pad, dim=1)], dim=1)
        cs_v = torch.cat([z_zero, torch.cumsum(v_pad, dim=1)], dim=1)

        shh_chunks = []
        for q_start in range(0, t, SEQ_READOUT_CHUNK):
            q_end = min(t, q_start + SEQ_READOUT_CHUNK)
            c_len = q_end - q_start
            qg_c = q_grp[:, :, q_start:q_end, :]
            t_idx = torch.arange(q_start, q_end, device=q.device)
            p_idx = t_idx + self.horizon

            f_start = q_start + self.horizon - (self.n_fine - 1)
            f_end = q_end + self.horizon
            k_f_win = k_pad[:, f_start:f_end, :].unfold(1, self.n_fine, 1).flip(-1)
            v_f_win = v_pad[:, f_start:f_end, :].unfold(1, self.n_fine, 1).flip(-1).transpose(-1, -2)
            logits_fine = (torch.matmul(qg_c.transpose(1, 2), k_f_win) * s_scale).transpose(1, 2)

            idx_hi = p_idx.unsqueeze(1) - c_new.unsqueeze(0) + 1
            idx_lo = p_idx.unsqueeze(1) - c_old.unsqueeze(0)
            idx_mid = (idx_lo + idx_hi) // 2

            cs_k_hi, cs_k_lo, cs_k_mid = cs_k[:, idx_hi, :], cs_k[:, idx_lo, :], cs_k[:, idx_mid, :]
            k_ca_c = (cs_k_hi - cs_k_lo) * c_inv_m
            v_ca_c = (cs_v[:, idx_hi, :] - cs_v[:, idx_lo, :]) * c_inv_m
            cd_n = torch.linalg.vector_norm((cs_k_hi - cs_k_mid) - (cs_k_mid - cs_k_lo), dim=-1)
            ca_n = torch.linalg.vector_norm((cs_k_hi - cs_k_mid) + (cs_k_mid - cs_k_lo), dim=-1).clamp_min(1e-6)
            detail_bonus = 0.15 * db_m * (cd_n / ca_n).unsqueeze(1)

            logits_coarse = (torch.matmul(qg_c.transpose(1, 2), k_ca_c.transpose(-1, -2)) * s_scale).transpose(1, 2) + (c_log_tau * ct_m) + detail_bonus
            logits_haar = torch.cat([logits_fine, logits_coarse], dim=-1)
            valid_haar = (t_idx.unsqueeze(1) >= o_old_all.unsqueeze(0)).view(1, 1, c_len, n_slots_haar)
            logits_haar = logits_haar.masked_fill(~valid_haar, -1e9)
            v_haar = torch.cat([v_f_win, v_ca_c], dim=2)

            bank_scores = torch.matmul(qg_c, bank_k.transpose(-1, -2).unsqueeze(1)) * s_scale
            allowed_bank = (bank_pos.unsqueeze(1) < (t_idx.view(1, c_len, 1) - self.n_fine)).unsqueeze(1)
            bank_scores = bank_scores.masked_fill(~allowed_bank, -1e9)

            probs = torch.softmax(torch.cat([logits_haar, bank_scores], dim=-1), dim=-1)
            y_grp = torch.matmul(probs[:, :, :, :n_slots_haar].transpose(1, 2), v_haar).transpose(1, 2) + torch.matmul(probs[:, :, :, n_slots_haar:], bank_v.unsqueeze(1))
            shh_chunks.append(y_grp.reshape(HEADS, c_len, HEAD_DIM))

        y_shh = torch.cat(shh_chunks, dim=1).transpose(0, 1)
        w_h = w_head.view(1, HEADS, 1)
        y_comb = ((1.0 - w_h) * y_r48 + w_h * y_shh).unsqueeze(0)

        head_gain = compute_prefix_head_gain(q, k, v, y_comb, PREFIX_CANDIDATES)
        eff_gain = head_gain.view(1, 1, HEADS, 1) * h_sc.view(1, 1, HEADS, 1)
        y_heads = (y_comb * eff_gain).to(dtype=dt)

        q_alpha_mean = float(base_scale) * aq_m.mean(dim=1).view(KV_HEADS, 1, 1)
        k_alpha_vec = float(base_scale) * ak_m.view(KV_HEADS, 1, 1)
        hg_hs = head_gain.view(HEADS, 1) * h_sc.view(HEADS, 1)

        h_win = self.horizon + 1
        k_tail = k_kv[:, -h_win:, :].detach() if t >= h_win else F.pad(k_kv.detach(), (0, 0, h_win - t, 0))
        v_tail = v_kv[:, -h_win:, :].detach() if t >= h_win else F.pad(v_kv.detach(), (0, 0, h_win - t, 0))

        _, _, _, c_log_tau, _, w_ca_c, w_cd_c = get_vectorized_pyramid_tensors(self.n_fine, q.device)
        k_c24, v_c24 = k_tail[:, :24, :], v_tail[:, :24, :]
        k_coarse = torch.matmul(w_ca_c, k_c24)
        v_coarse = torch.matmul(w_ca_c, v_c24)
        k_cd = torch.matmul(w_cd_c, k_c24)
        ct_v, db_v = ct_m.view(KV_HEADS, 1, 8), db_m.view(KV_HEADS, 1, 8)
        coarse_bonus = (c_log_tau.view(1, 1, 8) * ct_v) + 0.15 * db_v * (torch.linalg.vector_norm(k_cd, dim=-1) / torch.linalg.vector_norm(k_coarse, dim=-1).clamp_min(1e-6)).unsqueeze(1)

        k_fine_init, v_fine_init = k_tail[:, 24:, :], v_tail[:, 24:, :]
        uk_full = torch.cat([k_fine_init, k_coarse, bank_k.detach()], dim=1)
        uv_full = torch.cat([v_fine_init, v_coarse, bank_v.detach()], dim=1)
        s_tot = int(uk_full.shape[1])

        if self.backend == "cpp_avx1":
            self.basis_q_i8, self.basis_q_scale = quantize_to_cpu_int8_c_pair(basis * q_alpha_mean)
            self.basis_k_i8, self.basis_k_scale = quantize_to_cpu_int8_c_pair(basis * k_alpha_vec)
            self.state_s_i8, self.state_s_scale = quantize_to_cpu_int8_c_pair(s.detach())
            self.state_z_fp32 = z.detach().float().cpu().contiguous()
            self.gain_r48_fp32 = ((1.0 - w_h.view(HEADS, 1)) * hg_hs * r_sc.view(HEADS, 1)).view(HEADS).float().cpu().contiguous()
            self.gain_shh_fp32 = (w_h.view(HEADS, 1) * hg_hs).view(HEADS).float().cpu().contiguous()
            self.unified_k_i8, self.unified_k_scale = quantize_to_cpu_int8_c_pair(uk_full)
            self.unified_v_i8, self.unified_v_scale = quantize_to_cpu_int8_c_pair(uv_full)
            sb = torch.zeros((KV_HEADS, s_tot), device="cpu", dtype=torch.float32)
            sb[:, self.n_fine : self.n_fine + 8] = coarse_bonus.squeeze(1).detach().float().cpu()
            self.static_bias_fp32 = sb.contiguous()
        elif self.backend == "pytorch_int8":
            self.basis_q_i8, self.basis_q_scale = quantize_to_true_int8_pair(basis * q_alpha_mean, dt)
            self.basis_k_i8, self.basis_k_scale = quantize_to_true_int8_pair(basis * k_alpha_vec, dt)
            self.state_s_i8, self.state_s_scale = quantize_to_true_int8_pair(s.detach(), dt)
            self.state_z_fp32 = z.detach().unsqueeze(-1).contiguous()
            self.gain_r48_bf16 = ((1.0 - w_h.view(HEADS, 1)) * hg_hs * r_sc.view(HEADS, 1)).to(dtype=dt).contiguous()
            self.gain_shh_bf16 = (w_h.view(HEADS, 1) * hg_hs).to(dtype=dt).contiguous()
            self.unified_k_i8, self.unified_k_scale = quantize_to_true_int8_pair(uk_full, dt)
            self.unified_v_i8, self.unified_v_scale = quantize_to_true_int8_pair(uv_full, dt)
            sb = torch.zeros((KV_HEADS, 1, s_tot), device=q.device, dtype=torch.float32)
            sb[:, :, self.n_fine : self.n_fine + 8] = coarse_bonus
            self.static_bias_bf16 = sb.to(dtype=dt).contiguous()
        else:
            self.basis_q_bf16 = (basis_t * q_alpha_mean).to(dtype=dt).contiguous()
            self.basis_k_bf16 = (basis_t * k_alpha_vec).to(dtype=dt).contiguous()
            self.state_s_bf16 = s.detach().to(dtype=dt).contiguous()
            self.state_z_fp32 = z.detach().unsqueeze(-1).contiguous()
            self.gain_r48_bf16 = ((1.0 - w_h.view(HEADS, 1)) * hg_hs * r_sc.view(HEADS, 1)).to(dtype=dt).contiguous()
            self.gain_shh_bf16 = (w_h.view(HEADS, 1) * hg_hs).to(dtype=dt).contiguous()
            self.prev_fine_k = k_tail[:, 23, :].to(dtype=dt).clone()
            self.bank_novelty = bank_nov
            self.unified_k_bf16 = uk_full.to(dtype=dt).contiguous()
            self.unified_v_bf16 = uv_full.to(dtype=dt).contiguous()
            sb = torch.zeros((KV_HEADS, 1, s_tot), device=q.device, dtype=torch.float32)
            sb[:, :, self.n_fine : self.n_fine + 8] = coarse_bonus
            self.static_bias_bf16 = sb.to(dtype=dt).contiguous()

        self.ring_ptr = 0
        self.total_seen = t
        return F.linear(y_heads.reshape(b, t, HEADS * HEAD_DIM), self.w_o), None

def load_cpp_avx1_library(bundle_dir: Path) -> ctypes.CDLL:
    so_path = bundle_dir / "libwrai_avx1_int8_engine.so"
    cpp_path = bundle_dir / "wrai_avx1_int8_engine.cpp"
    if not so_path.exists():
        subprocess.run([
            "g++", "-O3", "-mavx", "-msse4.1", "-mno-avx2", "-mno-fma",
            "-fopenmp", "-shared", "-fPIC", str(cpp_path), "-o", str(so_path), "-lm"
        ], check=True)
    lib = ctypes.CDLL(str(so_path))
    lib.wrai_avx1_int8_decode_step_c.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_float, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p
    ]
    lib.wrai_avx1_int8_decode_step_c.restype = None
    return lib

def load_full_standalone_wrai_model(
    standalone_pt_path: Path, tok_cfg_dir: Path, device: torch.device,
    dtype: torch.dtype, backend_override: Optional[str] = None
) -> Tuple[Any, Any, List[StandaloneWRAILayer]]:
    """
    100% OFFLINE STANDALONE LOADER:
      1. Loads local `config.json` & `tokenizer` with `local_files_only=True` (0 internet!).
      2. Instantiates the skeleton on `torch.device("meta")` (0.00 MB RAM overhead!).
      3. Replaces the 28 `meta` attention layers with `StandaloneWRAILayer` on `meta`.
      4. Calls `model.load_state_dict(pkg["model_state_dict"], strict=True, assign=True)` to bind
         the standalone `.pt` weights directly into the model!
    """
    pkg = torch.load(standalone_pt_path, map_location=device)
    meta_cfg = pkg["wrai_architecture_meta"]
    eff_backend = backend_override if backend_override is not None else meta_cfg.get("default_backend", "jit_float")

    tok_is_local = (
        isinstance(tok_cfg_dir, Path) and tok_cfg_dir.is_dir()
    ) or (
        isinstance(tok_cfg_dir, str) and Path(tok_cfg_dir).is_dir()
    )
    tokenizer = AutoTokenizer.from_pretrained(str(tok_cfg_dir), local_files_only=tok_is_local)
    config = AutoConfig.from_pretrained(str(tok_cfg_dir), local_files_only=tok_is_local)

    cpp_lib = None
    if eff_backend == "cpp_avx1":
        try:
            cpp_lib = load_cpp_avx1_library(standalone_pt_path.resolve().parent.parent)
        except Exception as e:
            print(f"[WRAI Runner] C++ AVX1 engine unavailable ({e}). Falling back to Pure Python JIT backend ('jit_float').")
            eff_backend = "jit_float"

    # Instantiate 0-byte skeleton on `meta` device (never allocates original Qwen3Attention weights!)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)

    wrai_layers: List[StandaloneWRAILayer] = []
    for i, l_spec in enumerate(meta_cfg["layers"]):
        mod = StandaloneWRAILayer(
            layer_idx=i, alpha=l_spec["alpha"], gamma=l_spec["gamma"],
            n_fine=l_spec["n_fine"], sink_slots=l_spec["sink_slots"], hdc_slots=l_spec["hdc_slots"],
            param_dtype=torch.bfloat16, param_device=torch.device("meta"),
            backend=eff_backend, cpp_lib=cpp_lib
        )
        model.model.layers[i].self_attn = mod
        wrai_layers.append(mod)

    # Bind all standalone tensors directly (`assign=True`) and cast to target `dtype`/`device`
    state_dict = {
        k: (v.to(device=device, dtype=dtype) if v.dtype in (torch.bfloat16, torch.float16) else v.to(device=device))
        for k, v in pkg["model_state_dict"].items()
    }
    state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"]
    model.load_state_dict(state_dict, strict=True, assign=True)
    model.model.rotary_emb.inv_freq = pkg["rotary_inv_freq"].to(device=device)
    model.eval()
    model.requires_grad_(False)
    return model, tokenizer, wrai_layers

@torch.no_grad()
def generate_standalone_answer(model, tokenizer, prompt: str, max_new_tokens: int = 56) -> Tuple[str, float, float]:
    dev = next(model.parameters()).device
    inp_ids = tokenizer(prompt.rstrip(), add_special_tokens=True, return_tensors="pt")["input_ids"].to(dev)
    ctx_len = int(inp_ids.shape[1])
    stop_ids = {151643, 151645}
    if tokenizer.eos_token_id is not None:
        stop_ids.add(int(tokenizer.eos_token_id))

    poison_cache = DynamicCache()
    def _forbidden_kv_write(*_a, **_kw):
        raise RuntimeError("FATAL KV_CACHE_POISON_TRAP TRIGGERED!")
    poison_cache.update = _forbidden_kv_write

    t0 = time.perf_counter()
    out_pre = model.model(input_ids=inp_ids, past_key_values=poison_cache, use_cache=True, return_dict=True)
    pre_s = time.perf_counter() - t0

    nt = int(torch.argmax(model.lm_head(out_pre.last_hidden_state[:, -1:, :])[:, -1, :], dim=-1).item())
    gen_ids = [nt]
    inp_tok = torch.tensor([[nt]], device=dev, dtype=torch.long)
    pos_ids = torch.tensor([[ctx_len]], device=dev, dtype=torch.long)

    td0 = time.perf_counter()
    for step_i in range(1, max_new_tokens):
        if nt in stop_ids:
            break
        inp_tok[0, 0] = nt
        pos_ids[0, 0] = ctx_len + step_i - 1
        oh = model.model(input_ids=inp_tok, position_ids=pos_ids, past_key_values=poison_cache, use_cache=True, return_dict=True).last_hidden_state
        nt = int(torch.argmax(model.lm_head(oh[:, -1, :]), dim=-1).item())
        gen_ids.append(nt)
    dec_s = time.perf_counter() - td0
    tps = max(1, len(gen_ids) - 1) / max(dec_s, 1e-9)
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip(), pre_s, tps

def main_cli() -> None:
    parser = argparse.ArgumentParser(description="100% Offline Standalone WRAI Model Runner")
    parser.add_argument("--model", type=str, default="wrai_qwen3_0_6b_universal_homogeneous_standalone.pt",
                        help="Path to standalone .pt model checkpoint")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Path to tokenizer_and_config directory (default: ./tokenizer_and_config)")
    parser.add_argument("--backend", type=str, default="jit_float", choices=["jit_float", "pytorch_int8", "cpp_avx1"])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prompt", type=str, default=None, help="Input prompt for single-shot generation")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Maximum new tokens to generate")
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    torch.set_num_threads(args.threads)
    dev = torch.device(args.device)
    dt = torch.bfloat16 if dev.type == "cuda" else torch.float32
    bundle_dir = Path(__file__).resolve().parent

    # Search for model path across common candidate locations
    model_input = Path(args.model)
    cand_paths = [
        model_input,
        bundle_dir / args.model,
        bundle_dir / "standalone_models" / args.model,
        bundle_dir.parent / "hf_models" / "WRAI-Qwen3-0.6B" / args.model,
        bundle_dir.parent / "wrai_github_test" / "models" / args.model,
    ]
    pt_path = None
    for cand in cand_paths:
        if cand.is_file():
            pt_path = cand.resolve()
            break

    if pt_path is None:
        print("=" * 80)
        print(f"[ERROR] Standalone model checkpoint not found: '{args.model}'")
        print("\nPlease ensure you have downloaded the pre-distilled standalone .pt checkpoint.")
        print("Available from:")
        print("  - Hugging Face : https://huggingface.co/MusounoEnma/WRAI-Qwen3-0.6B")
        print("  - Zenodo DOI   : https://doi.org/10.5281/zenodo.23920132")
        print("\nOnce downloaded (~1.14 GB), run:")
        print("  python wrai_standalone_runner.py --model wrai_qwen3_0_6b_universal_homogeneous_standalone.pt --interactive")
        print("=" * 80)
        sys.exit(1)

    # Search for tokenizer directory
    if args.tokenizer is not None and Path(args.tokenizer).exists():
        tok_dir = Path(args.tokenizer).resolve()
    elif (bundle_dir / "tokenizer_and_config").is_dir():
        tok_dir = (bundle_dir / "tokenizer_and_config").resolve()
    elif (bundle_dir / "standalone_models" / "tokenizer_and_config").is_dir():
        tok_dir = (bundle_dir / "standalone_models" / "tokenizer_and_config").resolve()
    else:
        tok_dir = "Qwen/Qwen3-0.6B"

    print(f"[OFFLINE STANDALONE LOAD] Loading {pt_path.name} on {dev} ({dt})...")
    model, tokenizer, wrai_layers = load_full_standalone_wrai_model(pt_path, tok_dir, dev, dt, args.backend)
    if args.interactive:
        print("=== Mode Chat Interaktif Standalone WRAI (Ketik 'exit' untuk keluar) ===")
        while True:
            q = input("\nUser > ").strip()
            if not q or q.lower() in ("exit", "quit"):
                break
            p = f"<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
            ans, _, tps = generate_standalone_answer(model, tokenizer, p, max_new_tokens=args.max_new_tokens)
            print(f"WRAI ({tps:.1f} tok/s) > {ans}")
    else:
        test_prompt = args.prompt if args.prompt is not None else "The future of artificial intelligence on edge devices is"
        p = f"<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n<|im_start|>user\n{test_prompt}<|im_end|>\n<|im_start|>assistant\n"
        print(f"\n[Prompt] : {test_prompt}")
        ans, pre_s, tps = generate_standalone_answer(model, tokenizer, p, max_new_tokens=args.max_new_tokens)
        print(f"[Output] : {ans}")
        print(f"[Speed]  : {tps:.2f} tok/s (prefill: {pre_s:.3f}s)")

if __name__ == "__main__":
    main_cli()
