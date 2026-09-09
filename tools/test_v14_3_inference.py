#!/usr/bin/env python3
"""
=================================================================================
  WRAI v14.3.1 INTERACTIVE INFERENCE TESTER (EPOCH 12 EVALUATION)
=================================================================================
Jalankan script ini di Colab (atau Lokal) untuk menguji kualitas generasi teks 
WRAI v14.3.1 pada checkpoint Epoch 12 (`wrai_v14_3_latest.pt`).
"""

import json
import math
import os
import re
import sys
import torch
import torch.nn as nn

try:
    from tokenizers import Tokenizer
except ImportError:
    os.system("pip install -q tokenizers")
    from tokenizers import Tokenizer

MAX_VOCAB_SIZE = 32000
MAX_SEQ_LEN = 128
HIDDEN_DIM = 1024
NUM_LAYERS = 12
WAVELET_LEVELS = 4

DRIVE_SAVE_DIR = "/content/drive/MyDrive/WRAI_Models"
TOKENIZER_FILENAME = "tokenizer_v14.json"

class SinusoidalWavePositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class HaarDWTSpectralLayer(nn.Module):
    def __init__(self, hidden_dim, num_levels=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_levels = num_levels
        self.detail_gains = nn.ParameterList([
            nn.Parameter(torch.ones(hidden_dim // (2 ** (l + 1))))
            for l in range(num_levels)
        ])
        self.approx_gain = nn.Parameter(torch.ones(hidden_dim // (2 ** num_levels)))
        self.gate = nn.Parameter(torch.zeros(hidden_dim))

    def _haar_forward(self, x):
        details = []
        cur = x
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        for l in range(self.num_levels):
            even = cur[..., 0::2]
            odd = cur[..., 1::2]
            approx = (even + odd) * inv_sqrt2
            detail = (even - odd) * inv_sqrt2
            details.append(detail)
            cur = approx
        return details, cur

    def _haar_inverse(self, details, approx):
        cur = approx
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        for l in reversed(range(self.num_levels)):
            detail = details[l]
            even = (cur + detail) * inv_sqrt2
            odd = (cur - detail) * inv_sqrt2
            n = even.shape[-1]
            out = torch.empty(*even.shape[:-1], n * 2, device=even.device, dtype=even.dtype)
            out[..., 0::2] = even
            out[..., 1::2] = odd
            cur = out
        return cur

    def forward(self, x):
        x_float = x.float()
        details, approx = self._haar_forward(x_float)
        details = [d * g for d, g in zip(details, self.detail_gains)]
        approx = approx * self.approx_gain
        out = self._haar_inverse(details, approx)
        gate = torch.sigmoid(self.gate)
        out = out * gate + x_float * (1.0 - gate)
        return out.to(x.dtype)

class ResGRULayer(nn.Module):
    def __init__(self, hidden_dim, dropout=0.0):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x_norm = self.ln(x)
        out, _ = self.gru(x_norm)
        out = self.dropout(out)
        return residual + out

class ResGRUStack(nn.Module):
    def __init__(self, hidden_dim, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            ResGRULayer(hidden_dim, dropout=dropout if l < num_layers - 1 else 0.0)
            for l in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class WRAIv14_3(nn.Module):
    def __init__(self, vocab_size=32000, hidden_dim=1024, num_layers=12, dropout=0.1, wavelet_levels=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encoder = SinusoidalWavePositionalEncoding(hidden_dim)
        self.spectral1 = HaarDWTSpectralLayer(hidden_dim, num_levels=wavelet_levels)
        self.gru_stack = ResGRUStack(hidden_dim, num_layers, dropout=dropout)
        self.spectral2 = HaarDWTSpectralLayer(hidden_dim, num_levels=wavelet_levels)
        self.ln_final = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.fc.weight = self.embedding.weight
        self.embed_scale = hidden_dim ** 0.5

    def forward(self, x):
        emb = self.embedding(x) * self.embed_scale
        emb = self.pos_encoder(emb)
        feat = self.spectral1(emb)
        out = self.gru_stack(feat)
        feat2 = self.spectral2(out)
        out = self.ln_final(feat2 + out)
        logits = self.fc(out)
        return logits

def generate_text_v14_3(model, tokenizer, prompt, max_new_tokens=40, temperature=0.7, top_k=40, device="cuda"):
    model.eval()
    tokens = tokenizer.encode(prompt).ids
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    
    generated = list(tokens)
    eos_id = tokenizer.token_to_id("<EOS>")
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            if input_ids.size(1) > MAX_SEQ_LEN:
                curr_input = input_ids[:, -MAX_SEQ_LEN:]
            else:
                curr_input = input_ids
                
            logits = model(curr_input)
            next_logits = logits[0, -1, :] / max(temperature, 1e-5)
            
            if top_k > 0:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[-1]] = -float('Inf')
                
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            if next_token == eos_id:
                break
                
            generated.append(next_token)
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=device)], dim=1)
            
    return tokenizer.decode(generated)

def main():
    print("=================================================================", flush=True)
    print("  WRAI v14.3.1 INFERENCE EVALUATOR (EPOCH 12 CHECKPOINT)        ", flush=True)
    print("=================================================================\n", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Inference Device: {device}", flush=True)

    # 1. Load Tokenizer
    tok_path_drive = os.path.join(DRIVE_SAVE_DIR, TOKENIZER_FILENAME)
    tok_path = tok_path_drive if os.path.exists(tok_path_drive) else TOKENIZER_FILENAME
    
    if not os.path.exists(tok_path):
        print(f"[ERROR] Tokenizer file not found at {tok_path}!", flush=True)
        return

    tokenizer = Tokenizer.from_file(tok_path)
    vocab_size = len(tokenizer.get_vocab())
    print(f"[OK] Tokenizer Loaded from {tok_path} (Vocab: {vocab_size:,})", flush=True)

    # 2. Load Model
    model = WRAIv14_3(vocab_size=vocab_size, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, wavelet_levels=WAVELET_LEVELS).to(device)
    
    candidate_pt = [
        os.path.join(DRIVE_SAVE_DIR, "wrai_v14_3_latest.pt"),
        os.path.join(DRIVE_SAVE_DIR, "wrai_v14_3_best.pt"),
        "wrai_v14_3_latest.pt",
        "wrai_v14_3_best.pt"
    ]
    
    target_pt = None
    for p in candidate_pt:
        if os.path.exists(p):
            target_pt = p
            break
            
    if not target_pt:
        print("[ERROR] Checkpoint .pt file not found!", flush=True)
        return

    print(f"[*] Loading Checkpoint Weights from: {target_pt}...", flush=True)
    ckpt = torch.load(target_pt, map_location=device)
    model.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt)
    epoch = ckpt.get("epoch", "Unknown") if isinstance(ckpt, dict) else "Unknown"
    val_loss = ckpt.get("best_val_loss", "Unknown") if isinstance(ckpt, dict) else "Unknown"
    print(f"[OK 100% SUCCESS] Model Loaded! Checkpoint Epoch: {epoch} | Val Loss: {val_loss}\n", flush=True)

    # 3. Test Prompts across Multi-Domain
    test_prompts = [
        ("<ID>", "<ID> Halo, jelaskan apa itu"),
        ("<EN>", "<EN> Artificial Intelligence is"),
        ("<PY>", "<PY> def calculate_sum(a, b):"),
        ("<MATH>", "<MATH> If x + 5 = 12, then x is"),
        ("<ID>", "<ID> Indonesia adalah negara"),
    ]

    print("=================================================================", flush=True)
    print("                RESULTS OF WRAI v14.3.1 EVALUATION               ", flush=True)
    print("=================================================================\n", flush=True)

    for domain, prompt in test_prompts:
        output_text = generate_text_v14_3(model, tokenizer, prompt, max_new_tokens=35, temperature=0.65, top_k=30, device=device)
        print(f"PROMPT [{domain}]: {prompt}")
        print(f"OUTPUT   : {output_text}\n" + "-"*65)

if __name__ == "__main__":
    main()
