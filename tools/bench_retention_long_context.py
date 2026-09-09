import sys
import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.stdout.reconfigure(encoding='utf-8')

# -----------------------------------------------------------------------------
# 1. Layer Definitions
# -----------------------------------------------------------------------------

# GRU Single Layer Baseline
class GRULayerWrapper(nn.Module):
    def __init__(self, hidden_dim=896):
        super().__init__()
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

    def forward(self, x, h=None):
        return self.gru(x, h)

# Retention Vanilla (From Section 1)
class RetentionVanilla(nn.Module):
    def __init__(self, hidden_dim=896, num_heads=14, head_dim=64):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.w_q = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.w_k = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.w_v = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.w_out = nn.Linear(num_heads * head_dim, hidden_dim, bias=False)
        self.decay_logit = nn.Parameter(torch.zeros(num_heads))

    def forward(self, x, state=None):
        B, T, _ = x.shape
        H, D = self.num_heads, self.head_dim
        q = self.w_q(x).view(B, T, H, D)
        k = self.w_k(x).view(B, T, H, D)
        v = self.w_v(x).view(B, T, H, D)
        gamma = torch.sigmoid(self.decay_logit).view(1, H, 1, 1)

        if state is None:
            state = torch.zeros(B, H, D, D, device=x.device, dtype=x.dtype)

        outs = []
        for t in range(T):
            kt = k[:, t].unsqueeze(-1)
            vt = v[:, t].unsqueeze(-2)
            state = gamma * state + torch.matmul(kt, vt)
            qt = q[:, t].unsqueeze(-2)
            ot = torch.matmul(qt, state).squeeze(-2)
            outs.append(ot)

        out = torch.stack(outs, dim=1).reshape(B, T, H * D)
        return self.w_out(out), state

# Retention LongContext (Section 6a: Multi-Scale Decay Init + GroupNorm)
class RetentionLayerLongContext(nn.Module):
    def __init__(self, hidden_dim=896, num_heads=14, head_dim=64):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.w_q = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.w_k = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.w_v = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.w_out = nn.Linear(num_heads * head_dim, hidden_dim, bias=False)

        # Multi-scale decay init: spread explicitly from fast decay to slow decay
        init_gammas = 1.0 - torch.pow(2.0, -5.0 - torch.arange(num_heads).float())
        self.decay_logit = nn.Parameter(torch.logit(init_gammas))

        # GroupNorm per head before final output projection
        self.group_norm = nn.GroupNorm(num_heads, num_heads * head_dim)

    def forward(self, x, state=None):
        B, T, _ = x.shape
        H, D = self.num_heads, self.head_dim
        q = self.w_q(x).view(B, T, H, D)
        k = self.w_k(x).view(B, T, H, D)
        v = self.w_v(x).view(B, T, H, D)
        gamma = torch.sigmoid(self.decay_logit).view(1, H, 1, 1)

        if state is None:
            state = torch.zeros(B, H, D, D, device=x.device, dtype=x.dtype)

        outs = []
        for t in range(T):
            kt = k[:, t].unsqueeze(-1)
            vt = v[:, t].unsqueeze(-2)
            state = gamma * state + torch.matmul(kt, vt)
            qt = q[:, t].unsqueeze(-2)
            ot = torch.matmul(qt, state).squeeze(-2)
            outs.append(ot)

        out = torch.stack(outs, dim=1).reshape(B, T, H * D)
        # GroupNorm across [B, H*D, T]
        out_normed = self.group_norm(out.transpose(1, 2)).transpose(1, 2)
        return self.w_out(out_normed), state

# -----------------------------------------------------------------------------
# 2. Benchmark 6b: TPS & Magnitude Drift Across Sequence Lengths
# -----------------------------------------------------------------------------

def run_sequence_length_benchmark():
    print("=================================================================", flush=True)
    print("   🔬 SECTION 6b: 3-WAY SEQUENCE LENGTH & STABILITY BENCHMARK    ", flush=True)
    print("   Layers: GRU vs Retention-Vanilla vs Retention-LongContext     ", flush=True)
    print("   Sequence Lengths: 256, 512, 1024, 2048                        ", flush=True)
    print("=================================================================\n", flush=True)

    torch.set_num_threads(4)
    device = torch.device("cpu")
    hidden_dim = 896
    num_heads = 14
    head_dim = 64

    gru = GRULayerWrapper(hidden_dim=hidden_dim).to(device).eval()
    ret_vanilla = RetentionVanilla(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim).to(device).eval()
    ret_long = RetentionLayerLongContext(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim).to(device).eval()

    seq_lengths = [256, 512, 1024, 2048]

    results = []

    for seq_len in seq_lengths:
        torch.manual_seed(42)
        x = torch.randn(1, seq_len, hidden_dim, device=device)

        # Warmup
        with torch.no_grad():
            _ = gru(x)
            _ = ret_vanilla(x)
            _ = ret_long(x)

        # 1. GRU
        t0 = time.perf_counter()
        with torch.no_grad():
            out_gru, _ = gru(x)
        elapsed_gru = time.perf_counter() - t0
        tps_gru = seq_len / elapsed_gru
        mag_gru_max = out_gru.abs().max().item()
        mag_gru_std = out_gru.std().item()

        # 2. Retention Vanilla
        t0 = time.perf_counter()
        with torch.no_grad():
            out_v, _ = ret_vanilla(x)
        elapsed_v = time.perf_counter() - t0
        tps_v = seq_len / elapsed_v
        mag_v_max = out_v.abs().max().item()
        mag_v_std = out_v.std().item()

        # 3. Retention LongContext
        t0 = time.perf_counter()
        with torch.no_grad():
            out_l, _ = ret_long(x)
        elapsed_l = time.perf_counter() - t0
        tps_l = seq_len / elapsed_l
        mag_l_max = out_l.abs().max().item()
        mag_l_std = out_l.std().item()

        results.append({
            "seq_len": seq_len,
            "tps_gru": tps_gru, "mag_gru_max": mag_gru_max, "mag_gru_std": mag_gru_std,
            "tps_v": tps_v, "mag_v_max": mag_v_max, "mag_v_std": mag_v_std,
            "tps_l": tps_l, "mag_l_max": mag_l_max, "mag_l_std": mag_l_std,
        })

    print("------------------------------------------------------------------------------------------------------", flush=True)
    print(" SeqLen │    GRU (TPS / MaxMag)    │ Retention-Vanilla (TPS / MaxMag) │ Retention-LongCtx (TPS / MaxMag)", flush=True)
    print("------------------------------------------------------------------------------------------------------", flush=True)
    for r in results:
        print(f" {r['seq_len']:6d} │ {r['tps_gru']:6.1f} TPS  (Max: {r['mag_gru_max']:5.2f}) │ {r['tps_v']:6.1f} TPS  (Max: {r['mag_v_max']:7.2f}) │ {r['tps_l']:6.1f} TPS  (Max: {r['mag_l_max']:5.2f})", flush=True)
    print("------------------------------------------------------------------------------------------------------\n", flush=True)

    print("[*] Magnitude Standard Deviation across Lengths (Stability Analysis):", flush=True)
    for r in results:
        print(f"    T={r['seq_len']:4d} -> Vanilla Std: {r['mag_v_std']:7.4f} | LongContext (GroupNorm) Std: {r['mag_l_std']:7.4f}", flush=True)

    print("\n=================================================================", flush=True)

if __name__ == "__main__":
    run_sequence_length_benchmark()
