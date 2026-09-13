"""
WRAI-Micro (1.25M) Architectural Transplantation & Training Script
Model: SimpleStories-V2-1.25M -> WRAI Dual-State Linear Recurrence (Zero KV-Cache)
Target: ESP32 DevKit 38-Pin Microcontroller (Internal SRAM, Zero PSRAM)
"""

import os
import sys
import math
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import PreTrainedTokenizerFast

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "base_model")
TOKEN_FILE = os.path.join(BASE_DIR, "dataset_tokens.pt")

print("=" * 65, flush=True)
print(" 🧬 WRAI-MICRO (1.25M) ARCHITECTURAL TRANSMUTATION", flush=True)
print(" Target Silicon: ESP32 DevKit (240MHz, 320KB RAM, 4MB Flash)", flush=True)
print("=" * 65, flush=True)

# --- 1. Model Components ---
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * norm).to(x.dtype) * self.weight

class WRAIDualStateRetention(nn.Module):
    def __init__(self, d_model=128, n_heads=4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads # 32
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.r_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Decay parameter per head
        init_gammas = torch.tensor([0.96, 0.92, 0.88, 0.80], dtype=torch.float32)
        self.decay_param = nn.Parameter(torch.logit(init_gammas))
        self.gn = nn.GroupNorm(num_groups=n_heads, num_channels=d_model, affine=True)

    def forward_parallel(self, x):
        B, T, D = x.shape
        H, HD = self.n_heads, self.head_dim

        q = self.q_proj(x).view(B, T, H, HD).transpose(1, 2) # [B, H, T, HD]
        k = self.k_proj(x).view(B, T, H, HD).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, HD).transpose(1, 2)
        r = self.r_proj(x) # [B, T, D]

        # Exact proven causal retention decay matrix
        gamma = torch.sigmoid(self.decay_param).view(1, H, 1, 1)
        i_idx = torch.arange(T, device=x.device).view(T, 1)
        j_idx = torch.arange(T, device=x.device).view(1, T)
        dist = (i_idx - j_idx).clamp(min=0).view(1, 1, T, T).to(x.dtype)
        causal = (i_idx >= j_idx).view(1, 1, T, T).to(x.dtype)
        decay = torch.pow(gamma, dist) * causal

        scores = torch.matmul(q * self.scale, k.transpose(-1, -2)) * decay # [B, H, T, T]
        y_heads = torch.matmul(scores, v).transpose(1, 2).contiguous() # [B, T, H, HD]

        # GroupNorm & Gating
        y_flat = y_heads.view(B * T, H * HD)
        y_gn = self.gn(y_flat).view(B, T, D)
        out = self.out_proj(y_gn * F.silu(r))
        return out

class WRAISwiGLUFFN(nn.Module):
    def __init__(self, d_model=128, intermediate_size=341):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.up_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class WRAIMicroBlock(nn.Module):
    def __init__(self, d_model=128, n_heads=4, intermediate_size=341):
        super().__init__()
        self.input_layernorm = RMSNorm(d_model)
        self.retention = WRAIDualStateRetention(d_model, n_heads)
        self.post_attention_layernorm = RMSNorm(d_model)
        self.mlp = WRAISwiGLUFFN(d_model, intermediate_size)

    def forward(self, x):
        norm_x = self.input_layernorm(x)
        x = x + self.retention.forward_parallel(norm_x)
        norm_post = self.post_attention_layernorm(x)
        x = x + self.mlp(norm_post)
        return x

class WRAIMicroModel(nn.Module):
    def __init__(self, vocab_size=4023, d_model=128, n_layers=4, n_heads=4, intermediate_size=341):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embed_tokens = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            WRAIMicroBlock(d_model, n_heads, intermediate_size) for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)

    def forward(self, input_ids):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = F.linear(x, self.embed_tokens.weight)
        return logits

# --- 2. Load Base Weights and Transplant ---
print("\n[*] Loading SimpleStories-V2-1.25M weights from safetensors...", flush=True)
base_tensors = load_file(os.path.join(MODEL_DIR, "model.safetensors"))

model = WRAIMicroModel(vocab_size=4023, d_model=128, n_layers=4, n_heads=4, intermediate_size=341)

with torch.no_grad():
    base_embed = base_tensors["model.embed_tokens.weight"] # [4019, 128]
    model.embed_tokens.weight[:base_embed.shape[0]].copy_(base_embed)
    model.norm.weight.copy_(base_tensors["model.norm.weight"])

    for l in range(4):
        model.layers[l].input_layernorm.weight.copy_(base_tensors[f"model.layers.{l}.input_layernorm.weight"])
        model.layers[l].post_attention_layernorm.weight.copy_(base_tensors[f"model.layers.{l}.post_attention_layernorm.weight"])

        model.layers[l].mlp.gate_proj.weight.copy_(base_tensors[f"model.layers.{l}.mlp.gate_proj.weight"])
        model.layers[l].mlp.up_proj.weight.copy_(base_tensors[f"model.layers.{l}.mlp.up_proj.weight"])
        model.layers[l].mlp.down_proj.weight.copy_(base_tensors[f"model.layers.{l}.mlp.down_proj.weight"])

        q_w = base_tensors[f"model.layers.{l}.self_attn.q_proj.weight"]
        k_w = base_tensors[f"model.layers.{l}.self_attn.k_proj.weight"]
        v_w = base_tensors[f"model.layers.{l}.self_attn.v_proj.weight"]
        o_w = base_tensors[f"model.layers.{l}.self_attn.o_proj.weight"]

        model.layers[l].retention.q_proj.weight.copy_(q_w)
        if k_w.shape[0] == 64:
            k_w = k_w.repeat(2, 1)
            v_w = v_w.repeat(2, 1)
        model.layers[l].retention.k_proj.weight.copy_(k_w)
        model.layers[l].retention.v_proj.weight.copy_(v_w)
        model.layers[l].retention.out_proj.weight.copy_(o_w)
        nn.init.orthogonal_(model.layers[l].retention.r_proj.weight, gain=0.8)

print(" [OK] Pre-trained weights successfully transplanted!", flush=True)

# Freeze base knowledge layers
trainable_params = 0
frozen_params = 0
for name, p in model.named_parameters():
    if "retention" in name:
        p.requires_grad = True
        trainable_params += p.numel()
    else:
        p.requires_grad = False
        frozen_params += p.numel()

print(f" Frozen Knowledge Parameters : {frozen_params:,} (100% from HF)", flush=True)
print(f" Trainable WRAI Retention   : {trainable_params:,}", flush=True)
print("=" * 65, flush=True)

# --- 3. Fast Training on Dataset ---
dataset = torch.load(TOKEN_FILE) # [N, 64]
print(f"[*] Training on {len(dataset)} hybrid chat/logic sequences...", flush=True)

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=3e-3,
    weight_decay=0.01
)

batch_size = 32
num_epochs = 25
total_batches = math.ceil(len(dataset) / batch_size)

model.train()
start_t = time.time()

for epoch in range(1, num_epochs + 1):
    perm = torch.randperm(len(dataset))
    epoch_loss = 0.0
    for b in range(total_batches):
        idx = perm[b * batch_size : (b + 1) * batch_size]
        batch = dataset[idx]

        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        logits = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, model.vocab_size), targets.reshape(-1), ignore_index=0)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / total_batches
    if epoch % 5 == 0 or epoch == 1:
        elapsed = time.time() - start_t
        print(f" Epoch [{epoch:2d}/{num_epochs:2d}] | Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s", flush=True)

# Save checkpoint
save_path = os.path.join(BASE_DIR, "wrai_micro_1.25m_transplanted.pt")
torch.save(model.state_dict(), save_path)
print(f"\n[SUCCESS] Model successfully trained & saved to: {save_path}", flush=True)

# Quick inference validation
tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=os.path.join(MODEL_DIR, "tokenizer.json"),
    bos_token="<s>",
    eos_token="</s>"
)

print("\n--- Test Prompt Generation ---", flush=True)
prompt = "User: Who are you?\nAssistant:"
tokens = tokenizer.encode(prompt, add_special_tokens=False)
inp = torch.tensor([tokens], dtype=torch.long)

model.eval()
with torch.no_grad():
    curr = inp
    generated = list(tokens)
    for _ in range(25):
        out = model(curr)
        next_tok = torch.argmax(out[:, -1, :], dim=-1).item()
        if next_tok in [0, 2]: # pad or eos
            break
        generated.append(next_tok)
        curr = torch.cat([curr, torch.tensor([[next_tok]])], dim=1)

print("Generated Response:", flush=True)
print(tokenizer.decode(generated), flush=True)
print("=" * 65, flush=True)
