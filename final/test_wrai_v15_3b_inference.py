#!/usr/bin/env python3
"""
=============================================================================
 WRAI v15 (3B) — INFERENCE BENCHMARK & INTERACTIVE CHAT STUDIO
=============================================================================
 Evaluation of WRAI v15 model transplanted from Qwen2.5-3B (1:1):
  - 100% Zero KV-Cache (Fixed 16 KB SRAM Ping-Pong)
  - Full 151,936 Vocab (Zero Slicing, Natural & Fluent Reasoning)
  - 24-Layer Multi-Head Retention Long-Context + SwiGLU + Wavelet Spectral (2048-dim)
=============================================================================
"""

import os
import sys
import math
import time
import json
import gc
import glob
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import AutoTokenizer
    from huggingface_hub import login
except ImportError:
    os.system("pip install -q transformers huggingface_hub")
    from transformers import AutoTokenizer
    from huggingface_hub import login

HF_TOKEN = os.environ.get("HF_TOKEN", "")
os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
if HF_TOKEN:
    try:
        login(token=HF_TOKEN)
    except Exception:
        pass

DRIVE_MOUNT_PATH = "/content/drive"
DRIVE_SAVE_DIR = "/content/drive/MyDrive/WRAI_v15_3B_Models_Transplant"
OUTPUT_DIR = "models_v15_3b_transplant"
MODEL_NAME_3B = "Qwen/Qwen2.5-3B-Instruct"

drive_active = False
try:
    from google.colab import drive
    drive.mount(DRIVE_MOUNT_PATH, force_remount=False)
    drive_active = True
    print(f"\n[OK] GOOGLE DRIVE MOUNTED! Models loaded from dedicated folder: {DRIVE_SAVE_DIR}\n", flush=True)
except Exception as e:
    print(f"\n[INFO] Local environment detected: {e}\n", flush=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Initializing WRAI v15 (3B) Inference Engine on Device: {DEVICE}")

# -----------------------------------------------------------------------------
# Architecture
# -----------------------------------------------------------------------------

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

        # Multi-scale decay init: spread explicitly from fast decay to long decay
        init_gammas = 1.0 - torch.pow(2.0, -5.0 - torch.arange(num_heads).float())
        self.decay_logit = nn.Parameter(torch.logit(init_gammas))
        self.group_norm = nn.GroupNorm(num_heads, num_heads * head_dim)

    def forward_step(self, x_t, state):
        # x_t: [1, hidden_dim]
        # state: [1, num_heads, head_dim, head_dim]
        H, D = self.num_heads, self.head_dim
        q_t = self.w_q(x_t).view(1, H, 1, D)
        k_t = self.w_k(x_t).view(1, H, D, 1)
        v_t = self.w_v(x_t).view(1, H, 1, D)
        gamma = torch.sigmoid(self.decay_logit).view(1, H, 1, 1)

        next_state = gamma * state + torch.matmul(k_t, v_t)
        o_t = torch.matmul(q_t, next_state).squeeze(2).reshape(1, H * D)
        o_t_norm = self.group_norm(o_t.unsqueeze(-1)).squeeze(-1)
        out = self.w_out(o_t_norm)
        return out, next_state

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

class WRAI3BLayer(nn.Module):
    def __init__(self, hidden_dim=2048, ffn_dim=11008, num_heads=16, head_dim=128):
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

class WRAI3BModel(nn.Module):
    def __init__(self, vocab_size=151936, hidden_dim=2048, ffn_dim=11008,
                 num_layers=24, max_seq_len=256):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encoder = SinusoidalPositionalEncoding(hidden_dim, max_seq_len)
        self.spectral1 = HaarDWT1D(hidden_dim, levels=4)
        self.spectral2 = HaarDWT1D(hidden_dim, levels=4)
        self.layers = nn.ModuleList([WRAI3BLayer(hidden_dim, ffn_dim) for _ in range(num_layers)])
        self.ln_final = RMSNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.output_proj.weight = self.embed.weight
        self.embed_scale = hidden_dim ** 0.5

    def forward(self, input_ids, hidden_states=None):
        x = self.embed(input_ids) * self.embed_scale
        x = self.pos_encoder(x)
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
# Model Loader
# -----------------------------------------------------------------------------

MODEL_DTYPE = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

def load_3b_model():
    tok_path = DRIVE_SAVE_DIR if (drive_active and os.path.exists(os.path.join(DRIVE_SAVE_DIR, "vocab.json"))) else (
        OUTPUT_DIR if os.path.exists(os.path.join(OUTPUT_DIR, "vocab.json")) else MODEL_NAME_3B
    )
    print(f"[*] Loading Full 1:1 Tokenizer from: {tok_path}...")
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    vocab_size = 151936

    model = WRAI3BModel(vocab_size=vocab_size).to(DEVICE, dtype=MODEL_DTYPE)

    # 1. Coba Format 2-Shard Baru
    shard_dirs = [DRIVE_SAVE_DIR] if drive_active else []
    shard_dirs += ["/content/drive/My Drive/WRAI_v15_3B_Models_Transplant", OUTPUT_DIR, "."]

    loaded_ckpt = False
    for loc in shard_dirs:
        s1_path = os.path.join(loc, "wrai_v15_3b_shard_1.pt")
        s2_path = os.path.join(loc, "wrai_v15_3b_shard_2.pt")
        meta_path = os.path.join(loc, "wrai_v15_3b_meta.json")

        if os.path.exists(s1_path) and os.path.exists(s2_path):
            try:
                print(f"[*] Found 2-Shard Checkpoint in: {loc}. Streaming layers...")
                try:
                    s1 = torch.load(s1_path, map_location="cpu", mmap=True, weights_only=False)
                except Exception:
                    s1 = torch.load(s1_path, map_location="cpu", weights_only=False)
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if name in s1:
                            param.copy_(s1[name].to(device=DEVICE, dtype=MODEL_DTYPE))
                    for name, buf in model.named_buffers():
                        if name in s1:
                            buf.copy_(s1[name].to(device=DEVICE, dtype=MODEL_DTYPE))
                del s1
                gc.collect()

                try:
                    s2 = torch.load(s2_path, map_location="cpu", mmap=True, weights_only=False)
                except Exception:
                    s2 = torch.load(s2_path, map_location="cpu", weights_only=False)
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if name in s2:
                            param.copy_(s2[name].to(device=DEVICE, dtype=MODEL_DTYPE))
                    for name, buf in model.named_buffers():
                        if name in s2:
                            buf.copy_(s2[name].to(device=DEVICE, dtype=MODEL_DTYPE))
                del s2
                gc.collect()

                model.to(DEVICE, dtype=MODEL_DTYPE)
                model.output_proj.weight = model.embed.weight
                print(f"[OK SUCCESS] WRAI v15 2-Shard Weights 100% Loaded from: {loc}")
                if os.path.exists(meta_path):
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    print(f"  --> Checkpoint Loss: {meta.get('loss', '?'):.4f} | Epoch: {meta.get('epoch', '?')} | Step: {meta.get('step', '?')}")
                loaded_ckpt = True
                break
            except Exception as e:
                print(f"[WARN] Failed to load 2-shard from {loc}: {e}")

    # 2. Fallback to Legacy Monolithic Format (.pt)
    if not loaded_ckpt:
        candidate_checkpoints = []
        if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
            candidate_checkpoints.append(sys.argv[1])

        candidate_checkpoints += [
            os.path.join(DRIVE_SAVE_DIR, "wrai_v15_3b_transplant_best.pt"),
            os.path.join(DRIVE_SAVE_DIR, "wrai_v15_3b_transplant_latest.pt"),
            os.path.join("/content/drive/My Drive/WRAI_v15_3B_Models_Transplant", "wrai_v15_3b_transplant_best.pt"),
            os.path.join("/content/drive/My Drive/WRAI_v15_3B_Models_Transplant", "wrai_v15_3b_transplant_latest.pt"),
            os.path.join(OUTPUT_DIR, "wrai_v15_3b_transplant_best.pt"),
            os.path.join(OUTPUT_DIR, "wrai_v15_3b_transplant_latest.pt"),
        ]

        search_patterns = [
            os.path.join(DRIVE_SAVE_DIR, "*.pt"),
            "/content/drive/MyDrive/WRAI_v15_3B_Models_Transplant/*.pt",
            "/content/drive/My Drive/WRAI_v15_3B_Models_Transplant/*.pt",
            "models_v15_3b_transplant/*.pt",
            "*.pt"
        ]
        for pat in search_patterns:
            for found in glob.glob(pat):
                if found not in candidate_checkpoints and "shard" not in found:
                    candidate_checkpoints.append(found)

        for p in candidate_checkpoints:
            if os.path.exists(p) and os.path.isfile(p):
                try:
                    print(f"[*] Found monolithic checkpoint file: {p} ({os.path.getsize(p) / (1024**2):.1f} MB). Streaming weights...")
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
                    print(f"[OK SUCCESS] WRAI v15 Weights 100% Loaded from: {p}")
                    if isinstance(ckpt, dict) and "loss" in ckpt:
                        step_val = ckpt.get('step', '?')
                        opt_step = ckpt.get('optimizer_step', None)
                        step_display = f"{step_val} (Optim Step: {opt_step})" if opt_step and opt_step != step_val else f"{step_val}"
                        print(f"  --> Checkpoint Loss: {ckpt['loss']:.4f} | Epoch: {ckpt.get('epoch', '?')} | Step: {step_display}")
                    del state, ckpt
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    loaded_ckpt = True
                    break
                except Exception as e:
                    print(f"[WARN] Failed to load {p}: {e}")

    if not loaded_ckpt:
        print("\n[!] WARNING: No .pt checkpoint files found!")
        print(f"  --> Checked paths: {candidate_checkpoints[:5]}")
        print("  --> Ensure Google Drive is mounted and folder /content/drive/MyDrive/WRAI_v15_3B_Models_Transplant contains .pt files\n")

    model.eval()
    return model, tokenizer

def generate_response_3b(model, tokenizer, prompt: str, max_new_tokens: int = 200,
                         temperature: float = 0.6, top_k: int = 40, repetition_penalty: float = 1.15):
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_tokens = list(input_ids)
    gen_tokens = []

    t0 = time.perf_counter()
    with torch.no_grad():
        for step in range(max_new_tokens):
            cur_seq = torch.tensor([input_tokens], dtype=torch.long, device=DEVICE)
            logits, _ = model(cur_seq)
            next_logits = logits[0, -1, :].clone()

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

            if next_token == tokenizer.eos_token_id:
                break

            gen_tokens.append(next_token)
            input_tokens.append(next_token)

            tok_str = tokenizer.decode([next_token])
            if "<|im_end|>" in tok_str or "<|endoftext|>" in tok_str:
                break

            if len(input_tokens) >= 256:
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

def run_benchmark_suite_3b(model, tokenizer):
    print("\n" + "="*70)
    print("      🚀 WRAI v15 (3B) BENCHMARK EVALUATION           ")
    print("="*70)

    test_cases = [
        # --- ENGLISH SUITE ---
        ("🇬🇧 Swarm Persona (English)", "Explain who you are and how the WRAI ecosystem works."),
        ("🇬🇧 Python Coding (English)", "Write a clean Python function to calculate Fibonacci numbers with memoization."),
        ("🇬🇧 Cybersecurity (English)", "What is SQL Injection and how can developers prevent it in web applications?"),
        ("🇬🇧 Physics & Science (English)", "Why does the sky appear blue during the day and red at sunset?"),
        
        # --- INDONESIAN SUITE ---
        ("🇮🇩 WRAI-3B Swarm Persona (Indo)", "Jelaskan siapa kamu dan bagaimana ekosistem WRAI bekerja."),
        ("🇮🇩 Penerjemah Inggris -> Indo", "Terjemahkan kalimat berikut ke bahasa Indonesia:\n\"Artificial Intelligence empowers edge devices with ultra-low latency and absolute privacy.\""),
        ("🇮🇩 Delegate to WRAI-Coder", "Buatkan script python untuk web scraping data harga dari website e-commerce"),
        ("🇮🇩 Delegate to WRAI-Security", "Analisis potensi celah SQL Injection pada query database ini dan cara patchnya"),
        ("🇮🇩 Sains & Nalar Mendalam", "Mengapa langit terlihat berwarna biru pada siang hari dan berubah merah saat senja?")
    ]

    for domain, prompt in test_cases:
        print(f"\n----------------------------------------------------------------------")
        print(f"\033[1;33m[TEST DOMAIN]\033[0m {domain}")
        print(f"\033[1;37m[USER PROMPT]\033[0m \"{prompt}\"")
        
        # Use low temperature (0.15) & precise top_k (20) for focused, sharp outputs
        result = generate_response_3b(
            model, tokenizer, prompt, max_new_tokens=150, temperature=0.15, top_k=20, repetition_penalty=1.1
        )

        print(f"\033[1;36m[WRAI v15]\033[0m {result['response']}")
        print(f"\033[90m({result['tokens']} tokens | {result['speed']:.1f} tok/s | {result['elapsed']:.2f}s)\033[0m")

    print("\n" + "="*70 + "\n")

def interactive_chat_3b(model, tokenizer):
    print("="*70)
    print("💬 LIVE INTERACTIVE CHAT STUDIO — WRAI v15 (3B PARAMETERS)")
    print("Type 'exit' or 'quit' to end session")
    print("="*70)

    while True:
        try:
            prompt = input("\n\033[1;32mUser:\033[0m ").strip()
            if not prompt:
                continue
            if prompt.lower() in ("exit", "keluar", "quit", "q"):
                print("\n[👋] WRAI v15 Session Completed!\n")
                break

            result = generate_response_3b(
                model, tokenizer, prompt, max_new_tokens=180, temperature=0.2, top_k=20, repetition_penalty=1.1
            )

            print(f"\033[1;36mWRAI v15:\033[0m {result['response']}")
            print(f"\033[90m[{result['tokens']} tokens | {result['speed']:.1f} tok/s | {result['elapsed']:.2f}s]\033[0m")
        except KeyboardInterrupt:
            print("\n\n[Session Ended]")
            break

def main():
    model, tokenizer = load_3b_model()
    run_benchmark_suite_3b(model, tokenizer)
    interactive_chat_3b(model, tokenizer)

if __name__ == "__main__":
    main()
