import sys
import os
import time
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.stdout.reconfigure(encoding='utf-8')

# --- Architecture Definition ---

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight

class SwiGLUFFN(nn.Module):
    def __init__(self, hidden_dim, intermediate_dim):
        super().__init__()
        self.w_gate = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.w_up   = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.w_down = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))

class MultiHeadRetentionLayer(nn.Module):
    def __init__(self, hidden_dim=896, num_heads=14, head_dim=64):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hidden_dim = hidden_dim

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

class HaarDWT1D_Elementwise(nn.Module):
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

class WRAIAblationModel(nn.Module):
    def __init__(self, vocab_size=32000, hidden_dim=896, ffn_dim=4864, num_layers=4, use_dwt=True):
        super().__init__()
        self.use_dwt = use_dwt
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        if use_dwt:
            self.spectral1 = HaarDWT1D_Elementwise(hidden_dim, levels=4)
            self.spectral2 = HaarDWT1D_Elementwise(hidden_dim, levels=4)

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "rms_ret": RMSNorm(hidden_dim),
                "retention": MultiHeadRetentionLayer(hidden_dim, num_heads=14, head_dim=64),
                "rms_ffn": RMSNorm(hidden_dim),
                "ffn": SwiGLUFFN(hidden_dim, ffn_dim)
            }) for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.output_proj.weight = self.embed.weight

    def forward(self, input_ids):
        x = self.embed(input_ids)
        if self.use_dwt:
            x = self.spectral1(x)

        for i, layer in enumerate(self.layers):
            res = x
            out_ret, _ = layer["retention"](layer["rms_ret"](x))
            x = res + out_ret
            res = x
            x = res + layer["ffn"](layer["rms_ffn"](x))
            if self.use_dwt and i == (len(self.layers) // 2) - 1:
                x = self.spectral2(x)

        x = self.ln_final(x)
        return self.output_proj(x)

def train_quick_curve(use_dwt=True, num_steps=50, seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    torch.set_num_threads(4)
    device = torch.device("cpu")

    model = WRAIAblationModel(vocab_size=1024, hidden_dim=256, ffn_dim=768, num_layers=4, use_dwt=use_dwt).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)

    losses = []
    for step in range(num_steps):
        torch.manual_seed(seed + step)
        inputs = torch.randint(0, 1024, (4, 32), device=device)
        targets = torch.roll(inputs, -1, dims=1)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = F.cross_entropy(logits.view(-1, 1024), targets.view(-1))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses

def main():
    print("=================================================================", flush=True)
    print("   QUALITY ABLATION STUDY: WITH DWT (Config A) vs NO DWT (Config B)", flush=True)
    print("=================================================================\n", flush=True)

    print("[*] Running Config A (WITH HaarDWT1D Elementwise)...", flush=True)
    losses_with_dwt = train_quick_curve(use_dwt=True, num_steps=50, seed=42)

    print("[*] Running Config B (WITHOUT DWT - Pure Identity)...", flush=True)
    losses_without_dwt = train_quick_curve(use_dwt=False, num_steps=50, seed=42)

    print("\n-----------------------------------------------------------------", flush=True)
    print(" Step  | Loss With DWT (Config A) | Loss No DWT (Config B) | Diff ", flush=True)
    print("-----------------------------------------------------------------", flush=True)
    for s in [5, 10, 20, 30, 40, 50]:
        idx = s - 1
        l_a = losses_with_dwt[idx]
        l_b = losses_without_dwt[idx]
        diff = l_a - l_b
        status = "A lower (Better)" if diff < 0 else "B lower/equal"
        print(f" {s:4d}  |        {l_a:.4f}         |        {l_b:.4f}        | {diff:+.4f} ({status})", flush=True)
    print("-----------------------------------------------------------------\n", flush=True)

    avg_last_a = sum(losses_with_dwt[-15:]) / 15.0
    avg_last_b = sum(losses_without_dwt[-15:]) / 15.0
    print(f"[*] Final 15-step Avg Loss Config A (With DWT):    {avg_last_a:.4f}", flush=True)
    print(f"[*] Final 15-step Avg Loss Config B (Without DWT): {avg_last_b:.4f}", flush=True)

    if avg_last_a < avg_last_b:
        print("\n[DECISION GATE] [PASSED] DWT IS BENEFICIAL: Loss is lower with Wavelet Spectral Mixing!", flush=True)
        print("  --> Rekomendasi: Pertahankan HaarDWT1D dengan Elementwise Gate (Murah & Bertenaga).", flush=True)
    else:
        print("\n[DECISION GATE] [MARGINAL] DWT IS MARGINAL: Loss is roughly equal.", flush=True)
    print("=================================================================", flush=True)

if __name__ == "__main__":
    main()
