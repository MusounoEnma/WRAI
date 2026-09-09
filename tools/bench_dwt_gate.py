import sys
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.stdout.reconfigure(encoding='utf-8')

# 1. Original HaarDWT1D with Linear Gate (O(H^2))
class HaarDWT1D_LinearGate(nn.Module):
    def __init__(self, hidden_dim=896, levels=4):
        super().__init__()
        self.levels = levels
        self.hidden_dim = hidden_dim
        self.sqrt2 = math.sqrt(2.0)
        self.detail_gains = nn.Parameter(torch.ones(levels))
        self.approx_gain = nn.Parameter(torch.tensor(1.0))
        self.gate = nn.Linear(hidden_dim, hidden_dim, bias=False)
        nn.init.zeros_(self.gate.weight)

    def dwt_step(self, x):
        even = x[..., 0::2]
        odd = x[..., 1::2]
        approx = (even + odd) / self.sqrt2
        detail = (even - odd) / self.sqrt2
        return approx, detail

    def idwt_step(self, approx, detail):
        even = (approx + detail) / self.sqrt2
        odd = (approx - detail) / self.sqrt2
        B, T, D_half = approx.shape
        out = torch.empty(B, T, D_half * 2, device=approx.device, dtype=approx.dtype)
        out[..., 0::2] = even
        out[..., 1::2] = odd
        return out

    def forward(self, x):
        curr_approx = x
        details = []
        for l in range(self.levels):
            curr_approx, d = self.dwt_step(curr_approx)
            details.append(d * self.detail_gains[l])
        curr_approx = curr_approx * self.approx_gain
        for l in reversed(range(self.levels)):
            curr_approx = self.idwt_step(curr_approx, details[l])
        gate = torch.sigmoid(self.gate(x))
        return x + (curr_approx * gate)

# 2. Optimized HaarDWT1D with Elementwise Gate (O(H))
class HaarDWT1D_ElementwiseGate(nn.Module):
    def __init__(self, hidden_dim=896, levels=4):
        super().__init__()
        self.levels = levels
        self.hidden_dim = hidden_dim
        self.sqrt2 = math.sqrt(2.0)
        self.detail_gains = nn.Parameter(torch.ones(levels))
        self.approx_gain = nn.Parameter(torch.tensor(1.0))
        self.gate_weight = nn.Parameter(torch.zeros(hidden_dim))
        self.gate_bias = nn.Parameter(torch.full((hidden_dim,), -4.0))

    def dwt_step(self, x):
        even = x[..., 0::2]
        odd = x[..., 1::2]
        approx = (even + odd) / self.sqrt2
        detail = (even - odd) / self.sqrt2
        return approx, detail

    def idwt_step(self, approx, detail):
        even = (approx + detail) / self.sqrt2
        odd = (approx - detail) / self.sqrt2
        B, T, D_half = approx.shape
        out = torch.empty(B, T, D_half * 2, device=approx.device, dtype=approx.dtype)
        out[..., 0::2] = even
        out[..., 1::2] = odd
        return out

    def forward(self, x):
        curr_approx = x
        details = []
        for l in range(self.levels):
            curr_approx, d = self.dwt_step(curr_approx)
            details.append(d * self.detail_gains[l])
        curr_approx = curr_approx * self.approx_gain
        for l in reversed(range(self.levels)):
            curr_approx = self.idwt_step(curr_approx, details[l])
        gate = torch.sigmoid(x * self.gate_weight + self.gate_bias)
        return x + (curr_approx * gate)

def run_dwt_microbenchmark(hidden_dim=896, num_iterations=10000):
    print("=================================================================", flush=True)
    print("   ISOLATED CPU MICROBENCHMARK: DWT LINEAR GATE VS ELEMENTWISE  ", flush=True)
    print(f"   Hidden Dim: {hidden_dim} | Levels: 4 | Iterations: {num_iterations:,}", flush=True)
    print("=================================================================\n", flush=True)

    torch.set_num_threads(1)
    device = torch.device("cpu")

    dwt_linear = HaarDWT1D_LinearGate(hidden_dim=hidden_dim).to(device)
    dwt_elem = HaarDWT1D_ElementwiseGate(hidden_dim=hidden_dim).to(device)
    dwt_linear.eval()
    dwt_elem.eval()

    x = torch.randn(1, 1, hidden_dim, device=device)

    # Warmup
    for _ in range(500):
        with torch.no_grad():
            _ = dwt_linear(x)
            _ = dwt_elem(x)

    # 1. Benchmark Linear Gate
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = dwt_linear(x)
    elapsed_lin = time.perf_counter() - t0
    tps_lin = num_iterations / elapsed_lin

    # 2. Benchmark Elementwise Gate
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = dwt_elem(x)
    elapsed_elem = time.perf_counter() - t0
    tps_elem = num_iterations / elapsed_elem

    speedup = (tps_elem / tps_lin) * 100.0

    print(f"[*] DWT with Linear Gate (O(H^2)):", flush=True)
    print(f"    - Elapsed Time: {elapsed_lin:.4f}s", flush=True)
    print(f"    - Throughput:   {tps_lin:,.1f} TPS\n", flush=True)

    print(f"[*] DWT with Elementwise Gate (O(H)):", flush=True)
    print(f"    - Elapsed Time: {elapsed_elem:.4f}s", flush=True)
    print(f"    - Throughput:   {tps_elem:,.1f} TPS\n", flush=True)

    print(f"[*] Speedup Ratio:", flush=True)
    print(f"    - TPS_elementwise / TPS_linear: {speedup:.2f}% ({speedup/100.0:.2f}x faster!)\n", flush=True)
    print("=================================================================", flush=True)

if __name__ == "__main__":
    run_dwt_microbenchmark()
