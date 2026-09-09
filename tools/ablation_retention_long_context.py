import sys
import os
import time
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.stdout.reconfigure(encoding='utf-8')

# --- Layer & Model Definitions ---

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

class RetentionVanilla(nn.Module):
    def __init__(self, hidden_dim=256, num_heads=8, head_dim=32):
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

class RetentionLongContext(nn.Module):
    def __init__(self, hidden_dim=256, num_heads=8, head_dim=32):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.w_q = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.w_k = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.w_v = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.w_out = nn.Linear(num_heads * head_dim, hidden_dim, bias=False)

        # Multi-scale decay init
        init_gammas = 1.0 - torch.pow(2.0, -5.0 - torch.arange(num_heads).float())
        self.decay_logit = nn.Parameter(torch.logit(init_gammas))
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
        out_normed = self.group_norm(out.transpose(1, 2)).transpose(1, 2)
        return self.w_out(out_normed), state

class GRULayerBlock(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

    def forward(self, x, state=None):
        out, next_h = self.gru(x, state)
        return out, next_h

class TestModel(nn.Module):
    def __init__(self, layer_type="long_context", vocab_size=1024, hidden_dim=256, ffn_dim=768, num_layers=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            if layer_type == "gru":
                rec = GRULayerBlock(hidden_dim)
            elif layer_type == "vanilla":
                rec = RetentionVanilla(hidden_dim, num_heads=8, head_dim=32)
            elif layer_type == "long_context":
                rec = RetentionLongContext(hidden_dim, num_heads=8, head_dim=32)
            
            self.layers.append(nn.ModuleDict({
                "rms_rec": RMSNorm(hidden_dim),
                "rec": rec,
                "rms_ffn": RMSNorm(hidden_dim),
                "ffn": SwiGLUFFN(hidden_dim, ffn_dim)
            }))
            
        self.ln_final = RMSNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.output_proj.weight = self.embed.weight

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for layer in self.layers:
            res = x
            out_rec, _ = layer["rec"](layer["rms_rec"](x))
            x = res + out_rec
            res = x
            x = res + layer["ffn"](layer["rms_ffn"](x))
        x = self.ln_final(x)
        return self.output_proj(x)

def run_train_curve(layer_type, seq_len=64, num_steps=60, seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    torch.set_num_threads(4)
    device = torch.device("cpu")

    model = TestModel(layer_type=layer_type, vocab_size=1024, hidden_dim=256, ffn_dim=768, num_layers=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)

    losses = []
    for step in range(num_steps):
        torch.manual_seed(seed * 500 + step)
        inputs = torch.randint(0, 1024, (2, seq_len), device=device)
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
    print("   🔬 SECTION 6c: 3-WAY QUALITY ABLATION (GRU vs VANILLA vs LONG-CTX)", flush=True)
    print("=================================================================\n", flush=True)

    print("[*] Running GRU Baseline...", flush=True)
    loss_gru = run_train_curve("gru", seq_len=64, num_steps=60, seed=42)

    print("[*] Running Retention-Vanilla...", flush=True)
    loss_vanilla = run_train_curve("vanilla", seq_len=64, num_steps=60, seed=42)

    print("[*] Running Retention-LongContext (Multi-Scale Decay + GroupNorm)...", flush=True)
    loss_long = run_train_curve("long_context", seq_len=64, num_steps=60, seed=42)

    print("\n----------------------------------------------------------------------------------", flush=True)
    print(" Step │   GRU Loss   │ Retention-Vanilla Loss │ Retention-LongContext Loss │ Status", flush=True)
    print("----------------------------------------------------------------------------------", flush=True)
    for s in [10, 20, 30, 40, 50, 60]:
        idx = s - 1
        lg = loss_gru[idx]
        lv = loss_vanilla[idx]
        ll = loss_long[idx]
        best = "LongCtx Wins" if ll <= lv and ll <= lg else ("Vanilla Wins" if lv <= lg else "GRU Wins")
        print(f" {s:4d} │   {lg:8.4f}   │       {lv:8.4f}         │         {ll:8.4f}           │ {best}", flush=True)
    print("----------------------------------------------------------------------------------\n", flush=True)

    avg_g = sum(loss_gru[-20:]) / 20.0
    avg_v = sum(loss_vanilla[-20:]) / 20.0
    avg_l = sum(loss_long[-20:]) / 20.0

    print(f"[*] Final 20-Step Avg Loss GRU:                  {avg_g:.4f}", flush=True)
    print(f"[*] Final 20-Step Avg Loss Retention-Vanilla:    {avg_v:.4f}", flush=True)
    print(f"[*] Final 20-Step Avg Loss Retention-LongContext: {avg_l:.4f}\n", flush=True)

    if avg_l <= avg_v:
        print("[DECISION GATE] [PASSED] Retention-LongContext outperforms or matches Vanilla with bounded magnitude!", flush=True)
        print("  --> REKOMENDASI FINAL WRAI v15: Gunakan RetentionLayerLongContext (Multi-Scale Decay + GroupNorm)!")
    print("=================================================================", flush=True)

if __name__ == "__main__":
    main()
