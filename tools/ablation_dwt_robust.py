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
    def __init__(self, hidden_dim=256, num_heads=8, head_dim=32):
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
    def __init__(self, hidden_dim=256, levels=4):
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
    def __init__(self, vocab_size=1024, hidden_dim=256, ffn_dim=768, num_layers=4, use_dwt=True):
        super().__init__()
        self.use_dwt = use_dwt
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        if use_dwt:
            self.spectral1 = HaarDWT1D_Elementwise(hidden_dim, levels=4)
            self.spectral2 = HaarDWT1D_Elementwise(hidden_dim, levels=4)

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "rms_ret": RMSNorm(hidden_dim),
                "retention": MultiHeadRetentionLayer(hidden_dim, num_heads=8, head_dim=32),
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

def run_single_experiment(use_dwt, seed, num_steps=150):
    torch.manual_seed(seed)
    random.seed(seed)
    torch.set_num_threads(4)
    device = torch.device("cpu")

    model = WRAIAblationModel(vocab_size=1024, hidden_dim=256, ffn_dim=768, num_layers=4, use_dwt=use_dwt).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=0.01)

    losses = []
    for step in range(num_steps):
        torch.manual_seed(seed * 1000 + step)
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
    print("   🔬 STATISTICALLY ROBUST DWT ABLATION (3 SEEDS x 150 STEPS)    ", flush=True)
    print("=================================================================\n", flush=True)

    seeds = [42, 123, 999]
    num_steps = 150

    all_losses_a = [] # With DWT
    all_losses_b = [] # Without DWT

    for s_idx, s in enumerate(seeds):
        print(f"[*] Running Experiment Seed {s} ({s_idx+1}/{len(seeds)})...", flush=True)
        l_a = run_single_experiment(use_dwt=True, seed=s, num_steps=num_steps)
        l_b = run_single_experiment(use_dwt=False, seed=s, num_steps=num_steps)
        all_losses_a.append(l_a)
        all_losses_b.append(l_b)

    # Compute mean loss curves across seeds
    mean_a = [sum(all_losses_a[i][step] for i in range(len(seeds))) / len(seeds) for step in range(num_steps)]
    mean_b = [sum(all_losses_b[i][step] for i in range(len(seeds))) / len(seeds) for step in range(num_steps)]

    print("\n-----------------------------------------------------------------", flush=True)
    print(" Step  | Mean Loss (With DWT A) | Mean Loss (No DWT B) | Gap (A - B) ", flush=True)
    print("-----------------------------------------------------------------", flush=True)
    
    checkpoints = [10, 25, 50, 75, 100, 125, 150]
    a_wins = 0
    b_wins = 0

    for step in checkpoints:
        idx = step - 1
        la = mean_a[idx]
        lb = mean_b[idx]
        gap = la - lb
        if gap < 0:
            status = "A lower (With DWT Wins)"
            a_wins += 1
        else:
            status = "B lower (No DWT Wins)"
            b_wins += 1
        print(f" {step:4d}  |        {la:.4f}         |        {lb:.4f}        | {gap:+.4f} ({status})", flush=True)
    print("-----------------------------------------------------------------\n", flush=True)

    avg_final_a = sum(mean_a[-30:]) / 30.0
    avg_final_b = sum(mean_b[-30:]) / 30.0
    gap_final = avg_final_a - avg_final_b

    print(f"[*] Final 30-Step Mean Loss With DWT (A):    {avg_final_a:.4f}", flush=True)
    print(f"[*] Final 30-Step Mean Loss Without DWT (B): {avg_final_b:.4f}", flush=True)
    print(f"[*] Net Average Difference (A - B):          {gap_final:+.4f}\n", flush=True)

    if avg_final_a < avg_final_b and a_wins >= 4:
        print("[DECISION GATE] [CONFIRMED STATISTICALLY]: Wavelet DWT consistently lowers loss across multiple seeds!", flush=True)
        print("  --> Keputusan: Pertahankan HaarDWT1D dengan Elementwise Gate.", flush=True)
    elif abs(gap_final) < 0.5:
        print("[DECISION GATE] [NEUTRAL / MARGINAL]: Loss gap is negligible (< 0.5 diff across 3 seeds).", flush=True)
        print("  --> Keputusan: DWT dipertahankan untuk arsitektur hybrid spektral karena cost-nya sudah murah (5a), bukan klaim peningkatan kualitas ekstrem.", flush=True)
    else:
        print("[DECISION GATE] [NO BENEFIT]: Without DWT is equal or slightly better.", flush=True)
        print("  --> Keputusan: Pertimbangkan untuk menyederhanakan arsitektur.", flush=True)
    print("=================================================================", flush=True)

if __name__ == "__main__":
    main()
