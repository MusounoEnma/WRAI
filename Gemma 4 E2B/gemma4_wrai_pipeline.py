#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
====================================================================================================
WRAI x GEMMA 4 E2B — FULL CONVERSION & UNIVERSAL SELF-DISTILLATION PIPELINE
Repository : https://github.com/MusounoEnma/WRAI
File       : gemma4_wrai_pipeline.py
Description:
  End-to-end pipeline that converts native `google/gemma-4-E2B` into a Standalone Zero-KV
  Universal WRAI (WRAI-U) model:
    1. Freezes 100% of the Gemma 4 backbone weights.
    2. Captures multi-domain native pre-`o_proj` targets (Literature, Code, Science, Reasoning)
       handling Producer (L00-L14) and Shared-KV Consumer (L15-L34) layers safely.
    3. Optimizes 665 lightweight Universal WRAI parameters (19 scalars x 35 layers).
    4. Exports the portable checkpoint (`wrai_gemma4_universal_checkpoint.pt`) & `wrai_config.json`.
    5. Runs post-conversion verification under strict SDPA & DynamicCache poison traps.
====================================================================================================
"""
from __future__ import annotations
import argparse, gc, hashlib, inspect, json, math, os, sys, time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, DynamicCache

# ====================================================================================================
# ARCHITECTURE CONSTANTS & DEFAULT HYPERPARAMETERS
# ====================================================================================================
DEFAULT_MODEL_ID = "google/gemma-4-E2B"
DEFAULT_REVISION = "d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f"
EXPECTED_TEXT_LAYERS = 35
FULL_ATTN_INDICES = [4, 9, 14, 19, 24, 29, 34]

RANK = 48
PREFIX_CANDIDATES = 64
FINE_SLOTS = 64
SINK_SLOTS = 16
HDC_SLOTS = 144
PREFILL_CHUNK_SIZE = 128

QWEN_28L_ALPHA = {
    0: 0.125, 1: 0.80, 2: 0.325, 3: 0.70, 4: 0.525, 5: 0.50, 6: 0.65, 7: 0.275,
    8: 0.90, 9: 0.825, 10: 0.30, 11: 0.85, 12: 0.10, 13: 0.46875, 14: 0.525, 15: 0.575,
    16: 0.825, 17: 0.5625, 18: 0.75, 19: 0.875, 20: 0.90, 21: 0.925, 22: 0.90, 23: 0.85,
    24: 0.825, 25: 0.775, 26: 0.925, 27: 0.75,
}
QWEN_28L_GAMMA_RAW = {
    0: 0.40, 1: 0.70, 2: 0.55, 3: 0.9995, 4: 0.60, 5: 0.65, 6: 0.999, 7: 0.75,
    8: 0.9995, 9: 1.0, 10: 0.80, 11: 0.60, 12: 0.97, 13: 0.93, 14: 0.97, 15: 0.92,
    16: 0.9995, 17: 1.0, 18: 1.0, 19: 0.9995, 20: 1.0, 21: 1.0, 22: 1.0, 23: 1.0,
    24: 1.0, 25: 1.0, 26: 1.0, 27: 1.0,
}

CALIBRATION_CORPUS = [
    "The autumn wind swept through the dense oak canopy, scattering amber leaves across the desolate stone terrace. "
    "Elizabeth watched the carriage disappear beyond the valley with a quiet sense of resolve. Every promise made "
    "during that turbulent winter had now been fulfilled, yet the silence of the great hall remained unbroken.",
    "def merge_k_sorted_arrays(arrays: list[list[int]]) -> list[int]:\n"
    "    import heapq\n"
    "    min_heap = []\n"
    "    for i, arr in enumerate(arrays):\n"
    "        if arr:\n"
    "            heapq.heappush(min_heap, (arr[0], i, 0))\n"
    "    result = []\n"
    "    while min_heap:\n"
    "        val, arr_idx, elem_idx = heapq.heappop(min_heap)\n"
    "        result.append(val)\n"
    "        if elem_idx + 1 < len(arrays[arr_idx]):\n"
    "            next_val = arrays[arr_idx][elem_idx + 1]\n"
    "            heapq.heappush(min_heap, (next_val, arr_idx, elem_idx + 1))\n"
    "    return result\n",
    "In quantum thermodynamics, the second law can be formulated as a monotonicity property of the quantum relative entropy. "
    "When a closed system undergoes unitary evolution governed by a time-independent Hamiltonian, the von Neumann entropy remains "
    "strictly invariant. However, contact with a thermal bath induces non-unitary dissipation and irreversible entropy production.",
    "Question: What is the primary difference between Grouped-Query Attention (GQA) and Multi-Head Attention (MHA)?\n"
    "Answer: In Multi-Head Attention, each query head possesses its own dedicated key and value projection. In Grouped-Query "
    "Attention, multiple query heads share a single key-value head group, drastically reducing memory bandwidth during autoregressive decoding.",
]


# ====================================================================================================
# GLOBAL HELPERS & TORCHSCRIPT JIT KERNELS
# ====================================================================================================
def apply_gemma4_rope_fallback(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    rot_dim = cos.shape[-1]
    x_rot, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    c = cos.unsqueeze(1) if cos.ndim == 3 else cos
    s = sin.unsqueeze(1) if sin.ndim == 3 else sin
    h = rot_dim // 2
    x_half = torch.cat((-x_rot[..., h:], x_rot[..., :h]), dim=-1)
    x_rot_out = (x_rot * c) + (x_half * s)
    return torch.cat((x_rot_out, x_pass), dim=-1).to(dtype=x.dtype)


@torch.jit.script
def jit_diverse_landmark_indices(candidates: torch.Tensor, count: int) -> torch.Tensor:
    n = candidates.size(0)
    limit = count if count < n else n
    norm = F.normalize(candidates.float(), p=2.0, dim=-1, eps=1e-12)
    sim = torch.matmul(norm, norm.transpose(0, 1))
    selected = torch.zeros(limit, dtype=torch.long, device=candidates.device)
    min_dist = 1.0 - sim[:, 0]
    neg_inf = torch.tensor(-1e9, dtype=torch.float32, device=candidates.device)
    min_dist[0] = neg_inf

    for i in range(1, limit):
        idx = torch.argmax(min_dist)
        selected[i] = idx
        min_dist = torch.minimum(min_dist, 1.0 - sim[:, idx])
        min_dist[idx] = neg_inf
    return selected


@torch.jit.script
def jit_pick_cue_resonant_spans(
    salience_score: torch.Tensor, start_c: int, hist_len: int, win_len: int, n_windows: int
) -> torch.Tensor:
    work_sal = salience_score.clone()
    device = salience_score.device
    base_arange = torch.arange(win_len, dtype=torch.long, device=device)
    all_positions = torch.arange(hist_len, dtype=torch.long, device=device)
    spans = torch.empty((n_windows, win_len), dtype=torch.long, device=device)
    max_st = hist_len - win_len
    if max_st < 0:
        max_st = 0

    for w_i in range(n_windows):
        best_rel = torch.argmax(work_sal)
        w_st = torch.clamp(best_rel - 2, min=0, max=max_st)
        spans[w_i] = start_c + w_st + base_arange
        sup_lo = w_st - 16
        sup_hi = w_st + win_len + 16
        mask = (all_positions >= sup_lo) & (all_positions < sup_hi)
        work_sal = torch.where(mask, torch.full_like(work_sal, -1e9), work_sal)
    return spans.reshape(-1)


@torch.jit.script
def jit_vectorized_novelty_segments(
    novelty: torch.Tensor, start_c: int, span_len: int, n_nov_slots: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = novelty.device
    seg_width = max(1, (span_len + n_nov_slots - 1) // n_nov_slots)
    grid_base = (torch.arange(n_nov_slots, device=device, dtype=torch.long) * span_len) // n_nov_slots
    offsets = torch.arange(seg_width, device=device, dtype=torch.long)
    idx_grid = grid_base.unsqueeze(1) + offsets.unsqueeze(0)
    seg_end = ((torch.arange(1, n_nov_slots + 1, device=device, dtype=torch.long) * span_len) // n_nov_slots).unsqueeze(1)
    valid_mask = (idx_grid < seg_end) & (idx_grid < span_len)
    clamped_grid = torch.clamp(idx_grid, min=0, max=span_len - 1)

    gathered = novelty[:, clamped_grid]
    masked_vals = torch.where(valid_mask.unsqueeze(0), gathered, torch.full_like(gathered, -1e9))
    max_v, rel_max = torch.max(masked_vals, dim=-1)
    picked_pos = start_c + grid_base.unsqueeze(0) + rel_max
    return picked_pos, max_v


def build_haar_pyramid(n_fine: int = 64) -> Tuple[List[Tuple[int, int, int]], List[Tuple[int, int, int]], int]:
    fine_blocks = [(i, i, 1) for i in range(n_fine)]
    coarse_blocks = []
    cur = n_fine
    for mult, count in [(2, 4), (4, 4)]:
        for _ in range(count):
            coarse_blocks.append((cur, cur + mult - 1, mult))
            cur += mult
    return fine_blocks + coarse_blocks, coarse_blocks, cur - 1


_PYRAMID_CACHE: Dict[Tuple[int, str], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]] = {}


def get_haar_pyramid_tensors(n_fine: int, device: torch.device):
    key = (int(n_fine), str(device))
    if key not in _PYRAMID_CACHE:
        all_blocks, coarse_blocks, horizon = build_haar_pyramid(n_fine)
        c_new = torch.tensor([b[0] for b in coarse_blocks], device=device, dtype=torch.long)
        c_old = torch.tensor([b[1] for b in coarse_blocks], device=device, dtype=torch.long)
        c_mult = torch.tensor([b[2] for b in coarse_blocks], device=device, dtype=torch.float32)
        c_inv_m = (1.0 / c_mult).view(1, 1, 8, 1)
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

        _PYRAMID_CACHE[key] = (c_new, c_old, c_inv_m, o_old_all, w_ca_c, w_cd_c, horizon)
    return _PYRAMID_CACHE[key]


def resolve_gemma4_text_decoder(model: nn.Module, config: Any) -> Tuple[nn.Module, nn.ModuleList, str]:
    text_cfg = getattr(config, "text_config", config)
    expected_n = int(getattr(text_cfg, "num_hidden_layers", EXPECTED_TEXT_LAYERS))
    for path_str in ["model.language_model.layers", "language_model.layers", "language_model.model.layers", "model.layers"]:
        try:
            mod = model.get_submodule(path_str)
            if isinstance(mod, nn.ModuleList) and len(mod) == expected_n:
                parent_str = path_str.rsplit(".", 1)[0]
                return model.get_submodule(parent_str), mod, path_str
        except Exception:
            pass
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) == expected_n and "vision" not in name.lower() and "audio" not in name.lower():
            parent_name = name.rsplit(".", 1)[0] if "." in name else ""
            parent = model.get_submodule(parent_name) if parent_name else model
            return parent, mod, name
    raise RuntimeError(f"FAIL-CLOSED: Could not locate {expected_n}-layer text decoder!")


# ====================================================================================================
# UNIVERSAL LEARNABLE WRAI ADAPTER (665 PARAMETERS ACROSS 35 LAYERS)
# ====================================================================================================
class UniversalWRAIParameters(nn.Module):
    def __init__(self, layer_idx: int, head_dim: int, is_full_attn: bool, num_q_heads: int):
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.head_dim = int(head_dim)
        self.is_full_attn = bool(is_full_attn)
        self.num_q_heads = int(num_q_heads)

        q_idx = int(round(layer_idx * 27.0 / 34.0))
        base_alpha = float(QWEN_28L_ALPHA[q_idx])
        base_gamma = 1.0 if (is_full_attn and layer_idx >= 15) else (
            0.9993 if is_full_attn else (0.65 if QWEN_28L_GAMMA_RAW[q_idx] < 0.85 else 0.95)
        )

        self.log_alpha = nn.Parameter(torch.tensor(math.log(max(1e-4, base_alpha)), dtype=torch.float32))
        if self.is_full_attn:
            self.gamma_logit = nn.Parameter(torch.tensor(2.58, dtype=torch.float32))
        else:
            norm_g = max(0.05, min(0.95, (base_gamma - 0.40) / 0.58))
            init_logit = math.log(norm_g / (1.0 - norm_g))
            self.gamma_logit = nn.Parameter(torch.tensor(init_logit, dtype=torch.float32))

        self.wavelet_logit = nn.Parameter(torch.tensor(math.log(0.98 / 0.02), dtype=torch.float32))
        self.head_gain_r48_log = nn.Parameter(torch.zeros(num_q_heads, 1, dtype=torch.float32))
        self.head_gain_shh_log = nn.Parameter(torch.zeros(num_q_heads, 1, dtype=torch.float32))

    def get_beta_tensor(self) -> torch.Tensor:
        return torch.clamp(torch.exp(self.log_alpha), min=0.05, max=2.5) / math.sqrt(float(self.head_dim))

    def get_gamma_tensor(self) -> torch.Tensor:
        if self.is_full_attn and self.layer_idx >= 15:
            return torch.tensor(1.0, device=self.log_alpha.device, dtype=torch.float32)
        sig = torch.sigmoid(self.gamma_logit)
        if self.is_full_attn:
            return 0.995 + 0.0049 * sig
        return 0.40 + 0.58 * sig

    def get_wavelet_tensor(self) -> torch.Tensor:
        return torch.clamp(torch.sigmoid(self.wavelet_logit), min=0.80, max=0.995)

    def get_r48_head_mult(self) -> torch.Tensor:
        return torch.exp(self.head_gain_r48_log).clamp(0.50, 2.00)

    def get_shh_head_mult(self) -> torch.Tensor:
        return torch.exp(self.head_gain_shh_log).clamp(0.85, 1.15)

    @property
    def alpha(self) -> float:
        return float(torch.clamp(torch.exp(self.log_alpha), min=0.05, max=2.5).item())

    @property
    def beta(self) -> float:
        return self.alpha / math.sqrt(float(self.head_dim))

    @property
    def gamma(self) -> float:
        return float(self.get_gamma_tensor().item())

    @property
    def wavelet_weight(self) -> float:
        return float(self.get_wavelet_tensor().item())


# ====================================================================================================
# SHH BRANCH & DISTILLATION ENGINE
# ====================================================================================================
def compute_shh_and_prefix_ref(
    q_f: torch.Tensor,
    k_kv: torch.Tensor,
    v_kv: torch.Tensor,
    is_sliding: bool,
    sliding_window: Optional[int],
    attn_scaling: float,
    softcap_val: float,
    num_q_heads: int,
    head_dim: int,
    dt: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], int]:
    q_len = int(q_f.shape[2])
    n_kv = int(k_kv.shape[0])
    device = q_f.device

    c_new, c_old, c_inv_m, o_old_all, _, _, horizon = get_haar_pyramid_tensors(FINE_SLOTS, device)
    n_slots_haar = FINE_SLOTS + 8

    win_start = max(0, q_len - 512) if is_sliding else 0
    avail_len = q_len - win_start
    s_lim = min(int(SINK_SLOTS), avail_len)
    sink_idx = torch.arange(win_start, win_start + s_lim, device=device, dtype=torch.long).unsqueeze(0).expand(n_kv, -1)
    sink_nov = torch.full((n_kv, s_lim), 1e9, device=device, dtype=torch.float32)

    start_c = max(1, win_start + s_lim)
    end_c = q_len
    span_len = end_c - start_c
    novelty = torch.linalg.vector_norm(k_kv[:, start_c:end_c, :].detach() - k_kv[:, start_c - 1 : end_c - 1, :].detach(), dim=-1)
    hist_cutoff = max(start_c + 16, q_len - FINE_SLOTS)
    hist_len = hist_cutoff - start_c

    if span_len <= HDC_SLOTS or hist_len <= 48:
        edges = torch.linspace(0, max(1, span_len - 1), steps=HDC_SLOTS, device=device).long()
        hdc_idx = (start_c + edges).unsqueeze(0).expand(n_kv, -1)
        hdc_nov = torch.gather(novelty, 1, edges.unsqueeze(0).expand(n_kv, -1))
    else:
        win_len, n_windows = 16, 4
        n_span_slots = n_windows * win_len
        n_nov_slots = HDC_SLOTS - n_span_slots
        nov_idx, nov_val = jit_vectorized_novelty_segments(novelty, start_c, span_len, n_nov_slots)

        q_tail_len = min(24, q_len)
        q_cue = F.normalize(q_f[:, :, -q_tail_len:, :].detach(), p=2.0, dim=-1).to(dtype=dt)
        k_hist = F.normalize(k_kv[:, start_c:hist_cutoff, :].detach(), p=2.0, dim=-1).to(dtype=dt).unsqueeze(1)
        sim_mat = torch.matmul(q_cue, k_hist.transpose(-1, -2)).float()
        cue_resonance = sim_mat.amax(dim=(0, 1, 2))
        norm_nov = novelty[:, :hist_len].mean(dim=0) / (novelty[:, :hist_len].mean().clamp_min(1e-6))
        salience_score = cue_resonance + 0.15 * norm_nov

        span_1d = jit_pick_cue_resonant_spans(salience_score, start_c, hist_len, win_len, n_windows)
        span_idx = span_1d.unsqueeze(0).expand(n_kv, -1)
        span_nov = torch.full((n_kv, n_span_slots), 500.0, device=device, dtype=torch.float32)
        hdc_idx = torch.cat([nov_idx, span_idx], dim=-1)
        hdc_nov = torch.cat([nov_val, span_nov], dim=-1)

    all_bank_idx = torch.cat([sink_idx, hdc_idx], dim=-1)
    bank_nov = torch.cat([sink_nov, hdc_nov], dim=-1)
    idx_exp = all_bank_idx.unsqueeze(-1).expand(-1, -1, head_dim)
    bank_k = torch.gather(k_kv, 1, idx_exp).contiguous()
    bank_v = torch.gather(v_kv, 1, idx_exp).contiguous()

    s_scale = float(attn_scaling)
    k_pad = F.pad(k_kv, (0, 0, horizon, 0))
    v_pad = F.pad(v_kv, (0, 0, horizon, 0))
    z_zero = torch.zeros((n_kv, 1, head_dim), device=device, dtype=torch.float32)
    cs_k = torch.cat([z_zero, torch.cumsum(k_pad, dim=1)], dim=1)
    cs_v = torch.cat([z_zero, torch.cumsum(v_pad, dim=1)], dim=1)

    t_idx = torch.arange(0, q_len, device=device)
    p_idx = t_idx + horizon

    f_start = horizon - (FINE_SLOTS - 1)
    f_end = q_len + horizon
    k_f_win = k_pad[:, f_start:f_end, :].unfold(1, FINE_SLOTS, 1).flip(-1)
    v_f_win = v_pad[:, f_start:f_end, :].unfold(1, FINE_SLOTS, 1).flip(-1).transpose(-1, -2)

    q_bf = q_f.to(dtype=dt)
    logits_fine = (torch.matmul(q_bf.transpose(1, 2), k_f_win.to(dtype=dt)).float() * s_scale).transpose(1, 2)
    del k_f_win, k_pad, v_pad

    idx_hi = p_idx.unsqueeze(1) - c_new.unsqueeze(0) + 1
    idx_lo = p_idx.unsqueeze(1) - c_old.unsqueeze(0)
    idx_mid = (idx_lo + idx_hi) // 2

    cs_k_hi, cs_k_lo, cs_k_mid = cs_k[:, idx_hi, :], cs_k[:, idx_lo, :], cs_k[:, idx_mid, :]
    k_ca_c = (cs_k_hi - cs_k_lo) * c_inv_m
    v_ca_c = (cs_v[:, idx_hi, :] - cs_v[:, idx_lo, :]) * c_inv_m
    cd_n = torch.linalg.vector_norm((cs_k_hi - cs_k_mid) - (cs_k_mid - cs_k_lo), dim=-1)
    ca_n = torch.linalg.vector_norm((cs_k_hi - cs_k_mid) + (cs_k_mid - cs_k_lo), dim=-1).clamp_min(1e-6)
    detail_bonus = 0.08 * (cd_n / ca_n).unsqueeze(1)
    del cs_k, cs_v, cs_k_hi, cs_k_lo, cs_k_mid

    logits_coarse = (torch.matmul(q_f.transpose(1, 2), k_ca_c.transpose(-1, -2)) * s_scale).transpose(1, 2) + detail_bonus
    logits_haar = torch.cat([logits_fine, logits_coarse], dim=-1)

    valid_haar = (t_idx.unsqueeze(1) >= o_old_all.unsqueeze(0))
    if is_sliding and sliding_window is not None:
        valid_haar = valid_haar & (o_old_all.unsqueeze(0) < int(sliding_window))
    logits_haar = logits_haar.masked_fill(~valid_haar.view(1, 1, q_len, n_slots_haar), -1e9)
    v_haar = torch.cat([v_f_win, v_ca_c], dim=2)
    del v_f_win, v_ca_c, k_ca_c

    bank_scores = torch.matmul(q_bf, bank_k.to(dtype=dt).transpose(-1, -2).unsqueeze(1)).float() * s_scale
    allowed_bank = all_bank_idx.unsqueeze(1) < (t_idx.view(1, q_len, 1) - horizon)
    if is_sliding and sliding_window is not None:
        in_sliding_win = (t_idx.view(1, q_len, 1) - all_bank_idx.unsqueeze(1)) < int(sliding_window)
        allowed_bank = allowed_bank & in_sliding_win
    bank_scores = bank_scores.masked_fill(~allowed_bank.unsqueeze(1), -1e9)

    all_scores = torch.cat([logits_haar, bank_scores], dim=-1)
    if softcap_val > 0.0:
        all_scores = torch.tanh(all_scores / softcap_val) * softcap_val

    probs = torch.softmax(all_scores, dim=-1)
    probs_bf = probs.to(dtype=dt)
    y_shh = (
        torch.matmul(probs_bf[:, :, :, :n_slots_haar].transpose(1, 2), v_haar.to(dtype=dt)).transpose(1, 2)
        + torch.matmul(probs_bf[:, :, :, n_slots_haar:], bank_v.to(dtype=dt).unsqueeze(1))
    ).float().reshape(num_q_heads, q_len, head_dim).transpose(0, 1)
    del v_haar, probs, all_scores, logits_haar, bank_scores

    p_len = min(int(PREFIX_CANDIDATES), q_len)
    rms_ref: Optional[torch.Tensor] = None
    if p_len > 20:
        qp = q_f[:, :, :p_len, :]
        kp = k_kv[:, :p_len, :].unsqueeze(1)
        vp = v_kv[:, :p_len, :].unsqueeze(1)
        sc_p = torch.matmul(qp, kp.transpose(-2, -1)) * s_scale
        if softcap_val > 0.0:
            sc_p = torch.tanh(sc_p / softcap_val) * softcap_val
        p_idx_m = torch.arange(p_len, device=device)
        c_mask = p_idx_m.unsqueeze(0) <= p_idx_m.unsqueeze(1)
        if is_sliding and sliding_window is not None:
            c_mask = c_mask & ((p_idx_m.unsqueeze(1) - p_idx_m.unsqueeze(0)) < int(sliding_window))
        sc_p = sc_p.masked_fill(~c_mask.view(1, 1, p_len, p_len), -1e9)
        y_ref_p = torch.matmul(torch.softmax(sc_p, dim=-1), vp).reshape(num_q_heads, p_len, head_dim)[:, 16:p_len, :]
        rms_ref = torch.sqrt(torch.mean(y_ref_p * y_ref_p, dim=(1, 2)).clamp_min(1e-12)).detach()

    return y_shh, bank_k, bank_v, bank_nov, rms_ref, horizon


@torch.no_grad()
def capture_layer_training_data(
    model: nn.Module, text_layers: nn.ModuleList, batches: List[torch.Tensor]
) -> List[List[Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]]]:
    captured: List[List[Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]]] = [
        [] for _ in range(len(text_layers))
    ]
    for b_ids in batches:
        temp_hs: Dict[int, torch.Tensor] = {}
        temp_pe: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        temp_yo: Dict[int, torch.Tensor] = {}
        hooks = []

        for idx, lyr in enumerate(text_layers):
            def _make_attn_hook(l_i: int):
                def _pre_hook(mod, args, kwargs):
                    hs = args[0] if len(args) > 0 else kwargs["hidden_states"]
                    pe = args[1] if len(args) > 1 else kwargs.get("position_embeddings", None)
                    temp_hs[l_i] = hs.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
                    temp_pe[l_i] = (
                        pe[0].detach().to(device="cpu", dtype=torch.bfloat16).contiguous(),
                        pe[1].detach().to(device="cpu", dtype=torch.bfloat16).contiguous(),
                    )
                return _pre_hook

            def _make_oproj_hook(l_i: int):
                def _oproj_pre_hook(mod, args):
                    temp_yo[l_i] = args[0].detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
                return _oproj_pre_hook

            hooks.append(lyr.self_attn.register_forward_pre_hook(_make_attn_hook(idx), with_kwargs=True))
            hooks.append(lyr.self_attn.o_proj.register_forward_pre_hook(_make_oproj_hook(idx)))

        try:
            _ = model(input_ids=b_ids, use_cache=False, return_dict=True)
        finally:
            for h in hooks:
                h.remove()

        for idx in range(len(text_layers)):
            captured[idx].append((temp_hs[idx], temp_pe[idx], temp_yo[idx]))
        temp_hs.clear()
        temp_pe.clear()
        temp_yo.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return captured


def distill_universal_wrai_parameters(
    model: nn.Module,
    text_layers: nn.ModuleList,
    tokenizer: Any,
    device: torch.device,
    dtype: torch.dtype,
    steps: int = 45,
    lr: float = 6e-3,
    seq_len: int = 1024,
) -> Tuple[List[UniversalWRAIParameters], List[Dict[str, Any]]]:
    calib_batches = []
    for txt in CALIBRATION_CORPUS:
        ids = tokenizer(txt, add_special_tokens=True, return_tensors="pt")["input_ids"]
        if ids.shape[1] < seq_len:
            ids = ids.repeat(1, math.ceil(seq_len / ids.shape[1]))
        calib_batches.append(ids[:, :seq_len].to(device).contiguous())

    print(f"[WRAI-PIPELINE] Capturing native multi-domain targets across {len(text_layers)} layers...")
    captured_data = capture_layer_training_data(model, text_layers, calib_batches)

    universal_params: List[UniversalWRAIParameters] = []
    distill_metrics: List[Dict[str, Any]] = []
    shared_kv_cache: List[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = [{} for _ in calib_batches]

    for l_idx in range(len(text_layers)):
        nat_attn = text_layers[l_idx].self_attn
        h_dim = int(getattr(nat_attn, "head_dim"))
        is_full = (l_idx in FULL_ATTN_INDICES)
        is_sliding = not is_full
        sliding_window = int(getattr(nat_attn, "sliding_window", 512) or 512) if is_sliding else None
        attn_scaling = float(getattr(nat_attn, "scaling", 1.0 / math.sqrt(h_dim)))
        softcap = getattr(nat_attn, "attn_logit_softcapping", None)
        softcap_val = float(softcap) if (softcap is not None and float(softcap) > 0.0) else 0.0
        num_q = int(nat_attn.q_proj.weight.shape[0] // h_dim)

        u_param = UniversalWRAIParameters(l_idx, h_dim, is_full, num_q).to(device)
        optimizer = torch.optim.AdamW(u_param.parameters(), lr=lr, weight_decay=1e-4)

        layer_samples = []
        with torch.no_grad():
            for b_i in range(len(calib_batches)):
                hs = captured_data[l_idx][b_i][0].to(device=device, dtype=dtype)
                pe = (
                    captured_data[l_idx][b_i][1][0].to(device=device, dtype=dtype),
                    captured_data[l_idx][b_i][1][1].to(device=device, dtype=dtype),
                )
                y_nat_cpu = captured_data[l_idx][b_i][2]

                bsz, q_len, _ = hs.shape
                cos, sin = pe
                attn_mod = sys.modules.get(nat_attn.__module__, None)
                native_rope_fn = getattr(attn_mod, "apply_rotary_pos_emb", None) if attn_mod is not None else None

                def _rope(x: torch.Tensor) -> torch.Tensor:
                    if native_rope_fn is not None:
                        try:
                            res = native_rope_fn(x, x, cos, sin)
                            return (res[0] if isinstance(res, tuple) else res).to(dtype=x.dtype)
                        except Exception:
                            pass
                    return apply_gemma4_rope_fallback(x, cos, sin)

                q = nat_attn.q_proj(hs).view(bsz, q_len, num_q, h_dim)
                if hasattr(nat_attn, "q_norm") and nat_attn.q_norm is not None:
                    q = nat_attn.q_norm(q)
                q = _rope(q.transpose(1, 2))

                k_proj_mod = getattr(nat_attn, "k_proj", None)
                v_proj_mod = getattr(nat_attn, "v_proj", None)
                has_own = (k_proj_mod is not None and v_proj_mod is not None) and not bool(
                    getattr(nat_attn, "is_kv_shared", False) or getattr(nat_attn, "is_kv_shared_layer", False)
                )
                kv_src_idx = getattr(nat_attn, "kv_shared_layer_index", None)
                reg_by_idx = f"layer_{kv_src_idx}" if kv_src_idx is not None else None
                reg_by_type = f"type_{'sliding' if is_sliding else 'full'}"

                if has_own:
                    k_proj_w = k_proj_mod.weight.shape[0] // h_dim if hasattr(k_proj_mod, "weight") else 1
                    k = k_proj_mod(hs).view(bsz, q_len, k_proj_w, h_dim)
                    v_raw = v_proj_mod(hs).view(bsz, q_len, k_proj_w, h_dim)
                    if hasattr(nat_attn, "k_norm") and nat_attn.k_norm is not None:
                        k = nat_attn.k_norm(k)
                    k = _rope(k.transpose(1, 2))
                    if hasattr(nat_attn, "v_norm") and nat_attn.v_norm is not None:
                        v = nat_attn.v_norm(v_raw).transpose(1, 2)
                    else:
                        v = F.rms_norm(v_raw, (h_dim,), eps=1e-6).transpose(1, 2)
                    shared_kv_cache[b_i][f"layer_{l_idx}"] = (k, v)
                    shared_kv_cache[b_i][reg_by_type] = (k, v)
                else:
                    if reg_by_idx is not None and reg_by_idx in shared_kv_cache[b_i]:
                        k, v = shared_kv_cache[b_i][reg_by_idx]
                    elif reg_by_type in shared_kv_cache[b_i]:
                        k, v = shared_kv_cache[b_i][reg_by_type]
                    else:
                        cands = [k_n for k_n in shared_kv_cache[b_i].keys() if k_n.startswith("layer_")]
                        k, v = shared_kv_cache[b_i][cands[-1]]

                y_target = y_nat_cpu.to(device=device, dtype=torch.float32).view(q_len, num_q, h_dim).contiguous()
                q_f = q[0].float().view(k.shape[1], num_q // k.shape[1], q_len, h_dim)
                k_kv = k[0].float()
                v_kv = v[0].float()

                start_p = max(0, q_len - 512) if is_sliding else 0
                idx_b = jit_diverse_landmark_indices(k_kv[0, start_p : start_p + PREFIX_CANDIDATES, :], RANK)
                sample_basis = k_kv[0, start_p : start_p + PREFIX_CANDIDATES, :].index_select(0, idx_b).float().unsqueeze(0).contiguous()
                sample_basis_t = sample_basis.transpose(-1, -2).contiguous()

                y_shh_sample, _, _, _, rms_ref_sample, _ = compute_shh_and_prefix_ref(
                    q_f=q_f, k_kv=k_kv, v_kv=v_kv, is_sliding=is_sliding, sliding_window=sliding_window,
                    attn_scaling=attn_scaling, softcap_val=softcap_val, num_q_heads=num_q, head_dim=h_dim, dt=dtype
                )

                layer_samples.append((
                    q_f.detach(), k_kv.detach(), v_kv.detach(), y_target.detach(),
                    sample_basis_t.detach(), y_shh_sample.detach(), rms_ref_sample
                ))

        init_cos, final_cos, final_loss = 0.0, 0.0, 0.0
        for step in range(steps):
            optimizer.zero_grad()
            step_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
            step_cos = 0.0

            for q_grp, k_kv, v_kv, y_tgt, basis_t, y_shh_det, rms_ref in layer_samples:
                cur_beta = u_param.get_beta_tensor()
                cur_gamma = u_param.get_gamma_tensor()
                cur_wavelet = u_param.get_wavelet_tensor()
                mult_r48 = u_param.get_r48_head_mult()
                mult_shh = u_param.get_shh_head_mult()

                fq = torch.softmax(torch.matmul(q_grp, basis_t.unsqueeze(1)) * cur_beta, dim=-1)
                fk = torch.softmax(torch.matmul(k_kv, basis_t) * cur_beta, dim=-1)

                s_st = torch.zeros((k_kv.shape[0], RANK, h_dim), device=device, dtype=torch.float32)
                z_st = torch.zeros((k_kv.shape[0], RANK), device=device, dtype=torch.float32)
                r48_out_list = []
                t_total = int(q_grp.shape[2])

                for start in range(0, t_total, PREFILL_CHUNK_SIZE):
                    end = min(t_total, start + PREFILL_CHUNK_SIZE)
                    c_len = end - start
                    idx_arr = torch.arange(c_len, device=device, dtype=torch.float32)
                    diff_arr = idx_arr.unsqueeze(1) - idx_arr.unsqueeze(0)
                    causal_w = torch.tril(torch.pow(cur_gamma, diff_arr))
                    step_w = torch.pow(cur_gamma, idx_arr + 1.0).view(1, c_len, 1)
                    chunk_dec = torch.pow(cur_gamma, float(c_len))

                    fq_c = fq[:, :, start:end, :]
                    fk_c = fk[:, start:end, :]
                    v_c = v_kv[:, start:end, :]

                    z_traj = (z_st.unsqueeze(1) * step_w) + torch.matmul(causal_w, fk_c)
                    num_carry = torch.matmul(fq_c * step_w.unsqueeze(1), s_st.unsqueeze(1))
                    w_intra = torch.matmul(fq_c, fk_c.transpose(-1, -2).unsqueeze(1)) * causal_w.view(1, 1, c_len, c_len)
                    num_traj = num_carry + torch.matmul(w_intra, v_c.unsqueeze(1))
                    den_traj = torch.sum(fq_c * z_traj.unsqueeze(1), dim=-1, keepdim=True).clamp_min(1e-8)
                    y_c = (num_traj / den_traj).reshape(num_q, c_len, h_dim).transpose(0, 1)
                    r48_out_list.append(y_c)

                    last_w_vec = causal_w[-1, :].view(1, c_len, 1)
                    s_st = (s_st * chunk_dec) + torch.matmul((fk_c * last_w_vec).transpose(-1, -2), v_c)
                    z_st = z_traj[:, -1, :].contiguous()

                y_r48_diff = torch.cat(r48_out_list, dim=0)
                w_r48_h = (1.0 - cur_wavelet) * mult_r48
                w_shh_h = cur_wavelet * mult_shh
                y_pred = y_r48_diff * w_r48_h.view(1, num_q, 1) + y_shh_det * w_shh_h.view(1, num_q, 1)

                p_len = min(int(PREFIX_CANDIDATES), t_total)
                if rms_ref is not None and p_len > 20:
                    y_wrai_p = y_pred[16:p_len, :, :].detach().permute(1, 0, 2)
                    rms_wrai = torch.sqrt(torch.mean(y_wrai_p * y_wrai_p, dim=(1, 2)).clamp_min(1e-12))
                    dyn_gain = (rms_ref / rms_wrai).clamp(0.80, 1.25).view(1, num_q, 1)
                    y_pred = y_pred * dyn_gain

                eval_st = PREFIX_CANDIDATES
                p_flat = y_pred[eval_st:, :, :].reshape(-1, h_dim)
                t_flat = y_tgt[eval_st:, :, :].reshape(-1, h_dim)

                cos_sim = F.cosine_similarity(p_flat, t_flat, dim=-1).mean()
                rel_rms = torch.linalg.vector_norm(p_flat - t_flat) / torch.linalg.vector_norm(t_flat).clamp_min(1e-12)
                reg_penalty = 0.02 * (torch.mean(u_param.head_gain_r48_log ** 2) + torch.mean(u_param.head_gain_shh_log ** 2))
                loss = (1.0 - cos_sim) + 0.35 * rel_rms + reg_penalty

                step_loss = step_loss + (loss / len(layer_samples))
                step_cos += cos_sim.item() / len(layer_samples)

            if step == 0:
                init_cos = step_cos
            step_loss.backward()
            optimizer.step()
            final_cos = step_cos
            final_loss = float(step_loss.item())

        universal_params.append(u_param)
        distill_metrics.append({
            "layer_idx": l_idx,
            "layer_type": "full_attention" if is_full else "sliding_attention",
            "head_dim": h_dim,
            "learned_alpha": round(u_param.alpha, 5),
            "learned_beta": round(u_param.beta, 6),
            "learned_gamma": round(u_param.gamma, 5),
            "learned_wavelet_weight": round(u_param.wavelet_weight, 5),
            "init_cosine": round(init_cos, 5),
            "final_cosine": round(final_cos, 5),
            "cosine_gain": round(final_cos - init_cos, 5),
            "final_loss": round(final_loss, 5),
        })
        print(
            f"  [L{l_idx:02d} | {'FULL' if is_full else 'SLID'} D={h_dim:3d}] "
            f"alpha*={u_param.alpha:.4f} (beta*={u_param.beta:.5f}) | gamma*={u_param.gamma:.4f} | "
            f"wavelet*={u_param.wavelet_weight:.4f} | Cos: {init_cos:.4f} -> {final_cos:.4f} ({final_cos - init_cos:+.4f})"
        )
        del layer_samples
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del captured_data, shared_kv_cache, calib_batches
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return universal_params, distill_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Google Gemma 4 E2B to Standalone Zero-KV WRAI-U")
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID, help="Hugging Face model ID or local directory")
    parser.add_argument("--revision", type=str, default=DEFAULT_REVISION, help="Pinned model commit revision")
    parser.add_argument("--output-dir", type=str, default="./wrai_gemma4_release", help="Directory for exported artifacts")
    parser.add_argument("--steps", type=int, default=45, help="Distillation steps per layer")
    parser.add_argument("--lr", type=float, default=6e-3, help="Distillation learning rate")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda' or 'cpu', default: auto-detect)")
    args, _ = parser.parse_known_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    is_local = os.path.isdir(args.model_id)
    active_rev = None if is_local else args.revision

    print("=" * 110)
    print(f"WRAI x GEMMA 4 E2B — FULL MASTER CONVERSION PIPELINE")
    print(f"Base Model : {args.model_id} (revision: {active_rev or 'N/A (Local / Unpinned)'})")
    print(f"Device     : {device} | Dtype: {dtype} | Steps/Layer: {args.steps}")
    print("=" * 110)

    try:
        config = AutoConfig.from_pretrained(args.model_id, revision=active_rev)
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=active_rev)
    except Exception as e:
        print(f"\n[FATAL ERROR] Failed to load config/tokenizer for `{args.model_id}`: {e}")
        print("\nPlease ensure you have an active network connection or provide a valid local directory path.")
        sys.exit(1)

    load_kwargs = {"low_cpu_mem_usage": True}
    if active_rev is not None:
        load_kwargs["revision"] = active_rev
    if "dtype" in inspect.signature(AutoModelForCausalLM.from_pretrained).parameters:
        load_kwargs["dtype"] = dtype
    else:
        load_kwargs["torch_dtype"] = dtype

    try:
        model = AutoModelForCausalLM.from_pretrained(args.model_id, **load_kwargs).to(device).eval()
    except Exception as e:
        print("\n" + "=" * 90)
        print(f"[FATAL ERROR] Failed to load base model weights for `{args.model_id}`: {e}")
        print("=" * 90)
        print("[PREREQUISITE & WORKFLOW GUIDE]")
        print("  Gemma 4 E2B is a 2B parameter multimodal foundation model (~9.5 GB BF16 weights).")
        print("  The WRAI repository does not bundle raw 9.5 GB foundation weights.")
        print("  To run the master distillation pipeline:")
        print("  1. Online: Ensure you have sufficient disk space (~10 GB) and network connectivity.")
        print("  2. Offline: Pre-download `google/gemma-4-E2B` beforehand and pass its local directory:")
        print("     python gemma4_wrai_pipeline.py --model-id /path/to/downloaded/gemma-4-E2B")
        print("  3. Hardware requirement: ~12 GB+ VRAM recommended for calibration (CUDA).")
        print("=" * 90 + "\n")
        sys.exit(1)
    model.requires_grad_(False)
    _, text_layers, resolved_path = resolve_gemma4_text_decoder(model, config)
    print(f"[ARCH] Resolved {len(text_layers)} text decoder layers at `{resolved_path}`")

    universal_params, distill_metrics = distill_universal_wrai_parameters(
        model=model, text_layers=text_layers, tokenizer=tokenizer,
        device=device, dtype=dtype, steps=args.steps, lr=args.lr
    )

    ckpt_path = out_dir / "wrai_gemma4_universal_checkpoint.pt"
    config_path = out_dir / "wrai_config.json"

    total_params = sum(p.numel() for u in universal_params for p in u.parameters())
    ckpt_payload = {
        "architecture": "WRAI-U-Gemma4-E2B-ZeroKV",
        "base_model_id": args.model_id,
        "base_revision": args.revision,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_trainable_params": total_params,
        "hyperparameters": {
            "rank": RANK,
            "prefix_candidates": PREFIX_CANDIDATES,
            "fine_slots": FINE_SLOTS,
            "sink_slots": SINK_SLOTS,
            "hdc_slots": HDC_SLOTS,
            "full_attn_indices": FULL_ATTN_INDICES,
        },
        "layers": {
            i: {
                "log_alpha": u.log_alpha.detach().cpu(),
                "gamma_logit": u.gamma_logit.detach().cpu(),
                "wavelet_logit": u.wavelet_logit.detach().cpu(),
                "head_gain_r48_log": u.head_gain_r48_log.detach().cpu(),
                "head_gain_shh_log": u.head_gain_shh_log.detach().cpu(),
                "effective_alpha": round(u.alpha, 6),
                "effective_beta": round(u.beta, 6),
                "effective_gamma": round(u.gamma, 6),
                "effective_wavelet_weight": round(u.wavelet_weight, 6),
            }
            for i, u in enumerate(universal_params)
        },
    }
    torch.save(ckpt_payload, ckpt_path)
    ckpt_sha256 = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()

    config_json = {
        "architecture": "WRAI-U-Gemma4-E2B-ZeroKV",
        "base_model_id": args.model_id,
        "base_revision": args.revision,
        "checkpoint_file": ckpt_path.name,
        "checkpoint_sha256": ckpt_sha256,
        "total_trainable_params": total_params,
        "historical_kv_cache_mib": 0.0,
        "fixed_o1_state_mib": 11.5492,
        "layer_metrics": distill_metrics,
    }
    config_path.write_text(json.dumps(config_json, indent=2), encoding="utf-8")

    print("=" * 110)
    print(f"[SUCCESS] WRAI-U Gemma 4 E2B Release Bundle Exported!")
    print(f"  -> Checkpoint : {ckpt_path} (SHA256: {ckpt_sha256[:16]}...)")
    print(f"  -> Metadata   : {config_path}")
    print(f"  -> Parameters : {total_params} trained scalars across 35 layers")
    print("=" * 110)


if __name__ == "__main__":
    main()