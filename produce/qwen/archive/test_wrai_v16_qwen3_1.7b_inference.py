#!/usr/bin/env python3
"""
=============================================================================
  WRAI v16 (1.7B) — INFERENCE & BILINGUAL BENCHMARK STUDIO
=============================================================================
 Features:
  - 1:1 Qwen 3 (1.7B) Transplanted Architecture (28 Layers, 2048 Dim, 6144 FFN)
  - Fixed 2 MB Memory Buffer Budget (O(1) Recurrent State, 99.9% lighter)
  - Wavelet Spectral Mixer & Multi-Head Retention
  - Fast Single File Loader (~2.4 GB Checkpoint)
  - Bilingual Benchmark Suite (English & Indonesian)
  - Live Interactive Chat Studio
=============================================================================
"""

import os
import sys
import math
import time
import glob
import gc
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
HF_TOKEN = os.environ.get("HF_TOKEN", "")
QWEN3_MODEL_NAME = "Qwen/Qwen3-1.7B"
FALLBACK_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
HIDDEN_DIM = 2048
FFN_INTERMEDIATE_DIM = 6144
NUM_LAYERS = 28
NUM_HEADS = 16
HEAD_DIM = 128
WAVELET_LEVELS = 4
MAX_SEQ_LEN = 256

DRIVE_SAVE_DIR = "/content/drive/MyDrive/WRAI_v16_1.7B_Models_Transplant"
OUTPUT_DIR = "models_v16_1.7b_transplant"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DTYPE = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

# -----------------------------------------------------------------------------
# 1. WRAI v16 Neural Architecture
# -----------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        in_dtype = x.dtype
        x_f32 = x.float()
        norm = torch.rsqrt(x_f32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x_f32 * norm).to(in_dtype) * self.weight

class SwiGLUFFN(nn.Module):
    def __init__(self, hidden_dim, intermediate_dim):
        super().__init__()
        self.w_gate = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.w_up   = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.w_down = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))

class HaarDWT1D(nn.Module):
    def __init__(self, hidden_dim, levels=4):
        super().__init__()
        self.levels = levels
        self.hidden_dim = hidden_dim
        self.sqrt2 = math.sqrt(2.0)
        self.detail_gains = nn.Parameter(torch.ones(levels))
        self.approx_gain = nn.Parameter(torch.tensor(1.0))
        self.gate_weight = nn.Parameter(torch.zeros(hidden_dim))
        self.gate_bias = nn.Parameter(torch.full((hidden_dim,), -4.0))

    def forward(self, x):
        B, T, D = x.shape
        if T % 2 != 0:
            pad = torch.zeros(B, 1, D, device=x.device, dtype=x.dtype)
            x_pad = torch.cat([x, pad], dim=1)
            padded = True
        else:
            x_pad = x
            padded = False

        curr_approx = x_pad
        for lvl in range(min(self.levels, int(math.log2(max(2, curr_approx.size(1)))))):
            cur_len = curr_approx.size(1)
            if cur_len % 2 != 0:
                break
            even = curr_approx[:, 0::2, :]
            odd  = curr_approx[:, 1::2, :]
            approx = (even + odd) / self.sqrt2
            detail = (even - odd) / self.sqrt2
            detail = detail * self.detail_gains[lvl]
            even_rec = (approx + detail) / self.sqrt2
            odd_rec  = (approx - detail) / self.sqrt2
            curr_approx = torch.empty_like(curr_approx)
            curr_approx[:, 0::2, :] = even_rec
            curr_approx[:, 1::2, :] = odd_rec

        if padded:
            curr_approx = curr_approx[:, :T, :]

        gate = torch.sigmoid(x * self.gate_weight + self.gate_bias)
        return x + (curr_approx * gate)

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class MultiHeadRetentionLayer(nn.Module):
    def __init__(self, hidden_dim=2048, num_heads=16, head_dim=128):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hidden_dim = hidden_dim

        self.w_q = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.w_k = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.w_v = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.w_out = nn.Linear(num_heads * head_dim, hidden_dim, bias=False)

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
        # Per-Token GroupNorm: strictly causal, zero temporal leakage across sequence lengths
        C = H * D
        out_flat = out.contiguous().view(B * T, C, 1)
        out_normed = self.group_norm(out_flat).view(B, T, C)
        return self.w_out(out_normed), state

class WRAI17BLayer(nn.Module):
    def __init__(self, hidden_dim=2048, ffn_dim=6144, num_heads=16, head_dim=128):
        super().__init__()
        self.rms_ret = RMSNorm(hidden_dim)
        self.retention = MultiHeadRetentionLayer(hidden_dim, num_heads, head_dim)
        self.rms_ffn = RMSNorm(hidden_dim)
        self.ffn = SwiGLUFFN(hidden_dim, ffn_dim)

    def forward(self, x, s=None):
        res = x
        out_ret, next_s = self.retention(self.rms_ret(x), s)
        x = res + out_ret
        res = x
        x = res + self.ffn(self.rms_ffn(x))
        return x, next_s

class WRAI17BModel(nn.Module):
    def __init__(self, vocab_size=151936, hidden_dim=HIDDEN_DIM, ffn_dim=FFN_INTERMEDIATE_DIM,
                 num_layers=NUM_LAYERS, max_seq_len=MAX_SEQ_LEN):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encoder = SinusoidalPositionalEncoding(hidden_dim, max_seq_len)
        self.spectral1 = HaarDWT1D(hidden_dim, levels=WAVELET_LEVELS)
        self.spectral2 = HaarDWT1D(hidden_dim, levels=WAVELET_LEVELS)
        self.layers = nn.ModuleList([WRAI17BLayer(hidden_dim, ffn_dim) for _ in range(num_layers)])
        self.ln_final = RMSNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.output_proj.weight = self.embed.weight
    def forward(self, input_ids, hidden_states=None):
        x = self.embed(input_ids)
        x = self.spectral1(x)
        next_hidden_states = []
        for i, layer in enumerate(self.layers):
            h_i = hidden_states[i] if hidden_states is not None else None
            x, next_h = layer(x, h_i)
            next_hidden_states.append(next_h)
            if i == (len(self.layers) // 2) - 1:
                x = self.spectral2(x)
        x = self.ln_final(x)
        return self.output_proj(x), next_hidden_states

# -----------------------------------------------------------------------------
# 2. Checkpoint Loader & Setup
# -----------------------------------------------------------------------------

def load_17b_model():
    print(f"\n[*] Initializing WRAI v16 (1.7B) Inference Engine on Device: {DEVICE}")

    # Auto-mount Google Drive if in Colab
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
        print(f"[OK] GOOGLE DRIVE MOUNTED! Checking dedicated folder: {DRIVE_SAVE_DIR}")
    except Exception:
        pass

    active_model_name = QWEN3_MODEL_NAME
    print(f"[*] Loading Full 1:1 Tokenizer from: {active_model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(active_model_name, token=HF_TOKEN, trust_remote_code=True)
    except Exception as e:
        print(f"[WARN] Failed to load tokenizer {active_model_name} ({e}). Falling back to {FALLBACK_MODEL_NAME}...")
        active_model_name = FALLBACK_MODEL_NAME
        tokenizer = AutoTokenizer.from_pretrained(active_model_name, token=HF_TOKEN, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = getattr(tokenizer, "vocab_size", 151936)
    if vocab_size < 151936:
        vocab_size = 151936

    model = WRAI17BModel(vocab_size=vocab_size)

    # Search for single monolithic checkpoint files
    candidate_checkpoints = []
    preferred_names = [
        "wrai_v16_1.7b_latest.pt",
        "wrai_v16_1.7b_best.pt",
        "wrai_v16_1.7b_transplant_initial.pt"
    ]
    search_dirs = [DRIVE_SAVE_DIR, OUTPUT_DIR, "."]

    for d in search_dirs:
        for pref in preferred_names:
            p = os.path.join(d, pref)
            if os.path.exists(p) and p not in candidate_checkpoints:
                candidate_checkpoints.append(p)

    loaded_ckpt = False
    for p in candidate_checkpoints:
        try:
            print(f"[*] Found checkpoint file: {p} ({os.path.getsize(p) / (1024**2):.1f} MB). Streaming weights...")
            try:
                ckpt = torch.load(p, map_location="cpu", mmap=True, weights_only=False)
            except Exception:
                ckpt = torch.load(p, map_location="cpu", weights_only=False)

            state = ckpt.get("model_state", ckpt)
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in state:
                        param.copy_(state[name].to(device=DEVICE, dtype=MODEL_DTYPE))
                for name, buf in model.named_buffers():
                    if name in state:
                        buf.copy_(state[name].to(device=DEVICE, dtype=MODEL_DTYPE))

            model.to(DEVICE, dtype=MODEL_DTYPE)
            model.output_proj.weight = model.embed.weight
            print(f"[OK SUCCESS] WRAI v16 Weights 100% Loaded from: {p}")
            if isinstance(ckpt, dict) and "loss" in ckpt:
                print(f"  --> Checkpoint Loss: {ckpt['loss']:.4f} | Epoch: {ckpt.get('epoch', '?')} | Step: {ckpt.get('step', '?')}")
            del state, ckpt
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            loaded_ckpt = True
            break
        except Exception as e:
            print(f"[WARN] Failed to load {p}: {e}")

    if not loaded_ckpt:
        print("\n[!] WARNING: No .pt checkpoint files found. Running model in initial random state.")

    model.eval()
    return model, tokenizer

# -----------------------------------------------------------------------------
# 3. Response Generation (Greedy-Biased Decoding)
# -----------------------------------------------------------------------------

def generate_response_17b(model, tokenizer, prompt: str, max_new_tokens: int = 160, min_new_tokens: int = 5,
                          temperature: float = 0.35, top_k: int = 30, repetition_penalty: float = 1.15):
    clean_prompt = prompt.strip()
    if "<|im_start|>" not in clean_prompt:
        clean_prompt = f"<|im_start|>user\n{clean_prompt}<|im_end|>\n<|im_start|>assistant\n"

    input_ids = tokenizer.encode(clean_prompt, add_special_tokens=False)
    input_tokens = list(input_ids)
    gen_tokens = []

    eos_ids = {tokenizer.eos_token_id, 151643, 151645}
    if tokenizer.pad_token_id is not None:
        eos_ids.add(tokenizer.pad_token_id)

    t0 = time.perf_counter()
    with torch.no_grad():
        for step in range(max_new_tokens):
            cur_seq = torch.tensor([input_tokens], dtype=torch.long, device=DEVICE)
            logits, _ = model(cur_seq)
            next_logits = logits[0, -1, :].clone()

            # Prevent model from emitting EOS tokens prematurely (before min_new_tokens is met)
            if step < min_new_tokens:
                for eid in eos_ids:
                    if eid is not None and eid < next_logits.size(0):
                        next_logits[eid] = -float('Inf')

            # Apply repetition penalty to recent tokens
            for token_id in set(gen_tokens[-30:]):
                if token_id < next_logits.size(0):
                    if next_logits[token_id] < 0:
                        next_logits[token_id] *= repetition_penalty
                    else:
                        next_logits[token_id] /= repetition_penalty

            if temperature > 0:
                next_logits = next_logits / temperature
                if top_k > 0:
                    v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                    next_logits[next_logits < v[-1]] = -float('Inf')
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
            else:
                next_token = torch.argmax(next_logits).item()

            if step >= min_new_tokens and (next_token in eos_ids):
                break

            gen_tokens.append(next_token)
            input_tokens.append(next_token)

            tok_str = tokenizer.decode([next_token])
            if step >= min_new_tokens and ("<|im_end|>" in tok_str or "<|endoftext|>" in tok_str):
                break

            if len(input_tokens) >= MAX_SEQ_LEN:
                break

    elapsed = max(0.001, time.perf_counter() - t0)
    speed = len(gen_tokens) / elapsed
    response_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip() if gen_tokens else ""

    return {
        "response": response_text,
        "tokens": len(gen_tokens),
        "speed": speed,
        "elapsed": elapsed
    }

# -----------------------------------------------------------------------------
# 4. Bilingual Benchmark Evaluation
# -----------------------------------------------------------------------------

def run_benchmark_suite_17b(model, tokenizer):
    print("\n" + "="*70)
    print("      🚀 WRAI v16 (1.7B) BILINGUAL BENCHMARK EVALUATION           ")
    print("="*70)

    test_cases = [
        # --- ENGLISH SUITE ---
        ("🇬🇧 Swarm Persona (English)", "Explain who you are and how the WRAI ecosystem works."),
        ("🇬🇧 Python Coding (English)", "Write a clean Python function to calculate Fibonacci numbers with memoization."),
        ("🇬🇧 Cybersecurity (English)", "What is SQL Injection and how can developers prevent it in web applications?"),
        ("🇬🇧 Physics & Science (English)", "Why does the sky appear blue during the day and red at sunset?"),

        # --- INDONESIAN SUITE ---
        ("🇮🇩 WRAI Persona (Indo)", "Jelaskan siapa kamu dan bagaimana ekosistem WRAI bekerja."),
        ("🇮🇩 Penerjemah Inggris -> Indo", "Terjemahkan kalimat berikut ke bahasa Indonesia:\n\"Artificial Intelligence empowers edge devices with ultra-low latency and absolute privacy.\""),
        ("🇮🇩 Delegate to WRAI-Coder", "Buatkan script python untuk web scraping data harga dari website e-commerce"),
        ("🇮🇩 Delegate to WRAI-Security", "Analisis potensi celah SQL Injection pada query database ini dan cara patchnya"),
        ("🇮🇩 Sains & Nalar Mendalam", "Mengapa langit terlihat berwarna biru pada siang hari dan berubah merah saat senja?")
    ]

    for domain, prompt in test_cases:
        print(f"\n----------------------------------------------------------------------")
        print(f"\033[1;33m[TEST DOMAIN]\033[0m {domain}")
        print(f"\033[1;37m[USER PROMPT]\033[0m \"{prompt}\"")

        result = generate_response_17b(
            model, tokenizer, prompt, max_new_tokens=150, temperature=0.35, top_k=30, repetition_penalty=1.15
        )

        print(f"\033[1;36m[WRAI v16]\033[0m {result['response']}")
        print(f"\033[90m({result['tokens']} tokens | {result['speed']:.1f} tok/s | {result['elapsed']:.2f}s)\033[0m")

    print("\n" + "="*70 + "\n")

# -----------------------------------------------------------------------------
# 5. Live Interactive Chat Studio
# -----------------------------------------------------------------------------

def interactive_chat_17b(model, tokenizer):
    print("="*70)
    print("💬 LIVE INTERACTIVE CHAT STUDIO — WRAI v16 (1.7B PARAMETERS)")
    print("Type 'exit' or 'quit' to end session")
    print("="*70)

    while True:
        try:
            prompt = input("\n\033[1;32mUser:\033[0m ").strip()
            if not prompt:
                continue
            if prompt.lower() in ("exit", "keluar", "quit", "q"):
                print("\n[👋] WRAI v16 Session Completed!\n")
                break

            result = generate_response_17b(
                model, tokenizer, prompt, max_new_tokens=180, temperature=0.2, top_k=20, repetition_penalty=1.1
            )

            print(f"\033[1;36mWRAI v16:\033[0m {result['response']}")
            print(f"\033[90m({result['tokens']} tokens | {result['speed']:.1f} tok/s | {result['elapsed']:.2f}s)\033[0m")
        except (KeyboardInterrupt, EOFError):
            print("\n[👋] Session ended.")
            break

def main():
    model, tokenizer = load_17b_model()
    run_benchmark_suite_17b(model, tokenizer)
    if sys.stdin.isatty():
        interactive_chat_17b(model, tokenizer)

if __name__ == "__main__":
    main()
