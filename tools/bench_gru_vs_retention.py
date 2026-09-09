import sys
import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Fix Windows console encoding
sys.stdout.reconfigure(encoding='utf-8')

class RetentionLayer(nn.Module):
    def __init__(self, hidden_dim=896, num_heads=14, head_dim=64):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hidden_dim = hidden_dim

        self.w_q = nn.Linear(hidden_dim, num_heads * self.head_dim, bias=False)
        self.w_k = nn.Linear(hidden_dim, num_heads * self.head_dim, bias=False)
        self.w_v = nn.Linear(hidden_dim, num_heads * self.head_dim, bias=False)
        self.w_out = nn.Linear(num_heads * self.head_dim, hidden_dim, bias=False)

        # learnable decay per head in (0, 1)
        self.decay_logit = nn.Parameter(torch.zeros(num_heads))

    def forward_step(self, x_t, state):
        # x_t: [1, hidden_dim]
        # state: [1, num_heads, head_dim, head_dim]
        H, D = self.num_heads, self.head_dim
        q_t = self.w_q(x_t).view(1, H, 1, D)
        k_t = self.w_k(x_t).view(1, H, D, 1)
        v_t = self.w_v(x_t).view(1, H, 1, D)
        gamma = torch.sigmoid(self.decay_logit).view(1, H, 1, 1)

        # 1. State update: S_t = gamma * S_{t-1} + k_t @ v_t
        next_state = gamma * state + torch.matmul(k_t, v_t)
        # 2. Output read: o_t = q_t @ S_t
        o_t = torch.matmul(q_t, next_state).squeeze(2).reshape(1, H * D)
        out = self.w_out(o_t)
        return out, next_state

def run_isolated_microbenchmark(hidden_dim=896, num_heads=14, head_dim=64, num_iterations=2000):
    print("=================================================================", flush=True)
    print("   ISOLATED CPU MICROBENCHMARK: SINGLE-LAYER GRU VS RETENTION    ", flush=True)
    print(f"   Hidden Dim: {hidden_dim} | Heads: {num_heads} | Head Dim: {head_dim}", flush=True)
    print(f"   Iterations: {num_iterations:,} tokens forward steps", flush=True)
    print("=================================================================\n", flush=True)

    torch.set_num_threads(1) # Single-core CPU baseline for pure deterministic throughput
    device = torch.device("cpu")

    # 1. GRU Single Layer Setup
    gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True).to(device)
    gru.eval()
    h_gru = torch.zeros(1, 1, hidden_dim, device=device)
    x_dummy = torch.randn(1, 1, hidden_dim, device=device)

    # Warmup GRU
    for _ in range(100):
        with torch.no_grad():
            _, h_gru = gru(x_dummy, h_gru)

    # Benchmark GRU
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_iterations):
            _, h_gru = gru(x_dummy, h_gru)
    elapsed_gru = time.perf_counter() - t0
    tps_gru = num_iterations / elapsed_gru

    # 2. Retention Single Layer Setup
    ret = RetentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim).to(device)
    ret.eval()
    state_ret = torch.zeros(1, num_heads, head_dim, head_dim, device=device)
    x_t = torch.randn(1, hidden_dim, device=device)

    # Warmup Retention
    for _ in range(100):
        with torch.no_grad():
            _, state_ret = ret.forward_step(x_t, state_ret)

    # Benchmark Retention
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_iterations):
            _, state_ret = ret.forward_step(x_t, state_ret)
    elapsed_ret = time.perf_counter() - t0
    tps_ret = num_iterations / elapsed_ret

    ratio = (tps_ret / tps_gru) * 100.0

    print(f"[*] GRU Single Layer Forward:", flush=True)
    print(f"    - Elapsed Time: {elapsed_gru:.4f}s", flush=True)
    print(f"    - Throughput:   {tps_gru:,.1f} TPS (Tokens Per Second)\n", flush=True)

    print(f"[*] Retention Single Layer Forward:", flush=True)
    print(f"    - Elapsed Time: {elapsed_ret:.4f}s", flush=True)
    print(f"    - Throughput:   {tps_ret:,.1f} TPS (Tokens Per Second)\n", flush=True)

    print(f"[*] Comparison Ratio:", flush=True)
    print(f"    - TPS_retention / TPS_gru_baseline: {ratio:.2f}%\n", flush=True)

    if ratio >= 80.0:
        print("[DECISION GATE] [PASSED] (>= 80%): Retention throughput is extremely competitive!", flush=True)
    elif ratio >= 70.0:
        print("[DECISION GATE] [BORDERLINE] (70-80%): Tradeoff area.", flush=True)
    else:
        print("[DECISION GATE] [FAILED] (< 70%): GRU is faster on raw single-layer execution.", flush=True)
    print("=================================================================", flush=True)

if __name__ == "__main__":
    run_isolated_microbenchmark()
