#!/usr/bin/env python3
"""
=============================================================================
  🇬🇧 WRAI-X (0.6B) — ENGLISH BENCHMARK & LIVE CHAT STUDIO
=============================================================================
 Architecture:
  - 100% Zero KV-Cache (Constant O(1) Memory Budget)
  - 28 Layers Dual-State (Memory Mt + Reasoning Rt)
  - 4-Level Haar Multiresolution Spectral Filter
  - Native ChatML Tokenizer Alignment with Qwen-0.6B

 Features:
  - Comprehensive English Benchmark Suite (Science, Math, Code, Logic, Knowledge)
  - Fast Checkpoint Loader (Loads wrai_x_06b_transplanted.pt in ~2 seconds)
  - Interactive Live Chat Studio with Streaming Tokens
  - Clean Standard Decoding (No destructive penalties)
=============================================================================
"""

import os
import sys
import math
import time
import json
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer

# Auto-mount Google Drive if running in Google Colab
try:
    from google.colab import drive
    if not os.path.exists('/content/drive/MyDrive'):
        print("[*] Menghubungkan Google Drive untuk memuat model checkpoint...")
        drive.mount('/content/drive')
except ImportError:
    pass

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
SOURCE_MODEL_NAME = "Qwen/Qwen3-0.6B"

HIDDEN_DIM = 1024
FFN_DIM = 3072
NUM_LAYERS = 28
NUM_HEADS = 16
HEAD_DIM = 128
WAVELET_LEVELS = 4
VOCAB_SIZE = 151936

DRIVE_CHECKPOINT = "/content/drive/MyDrive/WRAI_X_06B/wrai_x_06b_transplanted.pt"
LOCAL_CHECKPOINT = "wrai_x_06b_transplanted.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# 1. Architecture Modules (1-to-1 WRAI-X 0.6B)
# -----------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * norm).to(x.dtype) * self.weight

class SwiGLUFFN(nn.Module):
    def __init__(self, hidden_dim=1024, ffn_dim=3072):
        super().__init__()
        self.w_gate = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.w_up   = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.w_down = nn.Linear(ffn_dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))

class HaarMultiresolution1D(nn.Module):
    def __init__(self, dim=1024, levels=4):
        super().__init__()
        self.dim = dim
        self.levels = levels
        self.sqrt2 = math.sqrt(2.0)
        self.high_gain = nn.Parameter(torch.ones(1))
        self.mid_gain  = nn.Parameter(torch.ones(1))
        self.low_gain  = nn.Parameter(torch.ones(1))
        self.gate_w = nn.Parameter(torch.zeros(dim))
        self.gate_b = nn.Parameter(torch.full((dim,), -5.0))

    def forward(self, x):
        N, D = x.shape
        approx = x
        details = []

        for lvl in range(self.levels):
            even = approx[:, 0::2]
            odd  = approx[:, 1::2]
            a = (even + odd) / self.sqrt2
            d = (even - odd) / self.sqrt2
            details.append(d)
            approx = a

        low_band = approx * self.low_gain
        mid_band = torch.cat([details[1], details[2], details[3]], dim=-1) * self.mid_gain
        high_band = details[0] * self.high_gain

        rec_approx = low_band
        for lvl in reversed(range(self.levels)):
            d = details[lvl]
            if lvl == 0:
                d = high_band
            even_rec = (rec_approx + d) / self.sqrt2
            odd_rec  = (rec_approx - d) / self.sqrt2
            rec_approx = torch.empty(N, 2 * even_rec.shape[1], device=x.device, dtype=x.dtype)
            rec_approx[:, 0::2] = even_rec
            rec_approx[:, 1::2] = odd_rec

        g = torch.sigmoid(x * self.gate_w + self.gate_b)
        out = x * (1.0 - g) + rec_approx * g
        return out, low_band, mid_band

class HDCAssociativeScratchpad(nn.Module):
    def __init__(self, dim=1024):
        super().__init__()
        self.dim = dim
        self.proj_key = nn.Linear(dim, dim, bias=False)
        self.proj_val = nn.Linear(dim, dim, bias=False)
        self.gate_hdc = nn.Linear(dim * 2, dim, bias=True)
        nn.init.constant_(self.gate_hdc.bias, -5.0)
        nn.init.zeros_(self.gate_hdc.weight)

    def forward(self, r_t, scratchpad=None):
        B, D = r_t.shape
        k = torch.tanh(self.proj_key(r_t))
        v = torch.tanh(self.proj_val(r_t))
        bound = k * v

        if scratchpad is None:
            new_scratchpad = bound
        else:
            new_scratchpad = (0.95 * scratchpad) + bound

        resonance = new_scratchpad * k
        g = torch.sigmoid(self.gate_hdc(torch.cat([r_t, resonance], dim=-1)))
        out = r_t + (resonance * g)
        return out, new_scratchpad

def rotate_half(x):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

class WRAIRotaryEmbedding(nn.Module):
    def __init__(self, dim, base=1000000.0):
        super().__init__()
        self.dim = dim
        self.register_buffer("inv_freq", 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim)), persistent=False)

    def apply_step(self, x, step_pos, device, dtype):
        t = torch.tensor([step_pos], device=device).float()
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype).view(1, 1, self.dim)
        sin = emb.sin().to(dtype).view(1, 1, self.dim)
        return (x * cos) + (rotate_half(x) * sin)

class WRAIXDualStateBlock(nn.Module):
    def __init__(self, hidden_dim=1024, ffn_dim=3072, num_heads=16, head_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(head_dim)

        self.rope = WRAIRotaryEmbedding(dim=head_dim)

        self.rms_ret = RMSNorm(hidden_dim)
        self.w_q = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.w_k = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.w_v = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.w_out = nn.Linear(num_heads * head_dim, hidden_dim, bias=False)
        self.gn_m = nn.GroupNorm(num_groups=num_heads, num_channels=num_heads * head_dim, affine=True)

        self.w_qr = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.w_kr = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.w_vr = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.w_out_r = nn.Linear(num_heads * head_dim, hidden_dim, bias=False)
        self.gn_r = nn.GroupNorm(num_groups=num_heads, num_channels=num_heads * head_dim, affine=True)

        init_gammas = 1.0 - torch.exp(-torch.linspace(1.5, 5.0, num_heads))
        self.decay_m = nn.Parameter(torch.logit(init_gammas))
        self.decay_r = nn.Parameter(torch.logit(init_gammas * 0.98))

        self.haar_bridge = HaarMultiresolution1D(dim=hidden_dim, levels=WAVELET_LEVELS)

        self.think_gate = nn.Linear(hidden_dim * 2, hidden_dim, bias=True)
        nn.init.constant_(self.think_gate.bias, -5.0)
        nn.init.zeros_(self.think_gate.weight)
        self.alpha = nn.Parameter(torch.zeros(1))

        self.hdc = HDCAssociativeScratchpad(dim=hidden_dim)

        self.rms_ffn = RMSNorm(hidden_dim)
        self.ffn = SwiGLUFFN(hidden_dim, ffn_dim)

    def forward_step(self, x, step_pos=0, state_m=None, state_r=None, state_hdc=None):
        B, D = x.shape
        H, HD = self.num_heads, self.head_dim

        x_norm = self.rms_ret(x)

        # 1. RoPE + Memory State (Mt)
        q = self.w_q(x_norm).view(B, H, HD)
        k = self.w_k(x_norm).view(B, H, HD)
        v = self.w_v(x_norm).view(B, H, HD)

        q_rope = self.rope.apply_step(q, step_pos, x.device, x.dtype)
        k_rope = self.rope.apply_step(k, step_pos, x.device, x.dtype)

        gamma_m = torch.sigmoid(self.decay_m).view(1, H, 1, 1)
        if state_m is None:
            state_m = torch.zeros(B, H, HD, HD, device=x.device, dtype=x.dtype)

        state_m = state_m * gamma_m + torch.einsum('bhr,bhc->bhrc', k_rope * self.scale, v)
        o_head_m = torch.einsum('bhr,bhrc->bhc', q_rope, state_m).reshape(B, H * HD)
        o_m = self.w_out(self.gn_m(o_head_m))

        # 2. Haar Multiresolution Bridge
        o_m_filtered, low_band, mid_band = self.haar_bridge(o_m)

        # 3. RoPE + Reasoning State (Rt)
        qr = self.w_qr(o_m_filtered).view(B, H, HD)
        kr = self.w_kr(o_m_filtered).view(B, H, HD)
        vr = self.w_vr(o_m_filtered).view(B, H, HD)

        qr_rope = self.rope.apply_step(qr, step_pos, x.device, x.dtype)
        kr_rope = self.rope.apply_step(kr, step_pos, x.device, x.dtype)

        gamma_r = torch.sigmoid(self.decay_r).view(1, H, 1, 1)
        if state_r is None:
            state_r = torch.zeros(B, H, HD, HD, device=x.device, dtype=x.dtype)

        state_r = state_r * gamma_r + torch.einsum('bhr,bhc->bhrc', kr_rope * self.scale, vr)
        o_head_r = torch.einsum('bhr,bhrc->bhc', qr_rope, state_r).reshape(B, H * HD)
        o_r = self.w_out_r(self.gn_r(o_head_r))

        # 4. HDC Associative Scratchpad
        o_r_hdc, state_hdc = self.hdc(o_r, state_hdc)

        # 5. Adaptive Thinking Gate
        gate_think = torch.sigmoid(self.think_gate(torch.cat([o_m_filtered, o_r_hdc], dim=-1)))
        ret_fused = o_m_filtered + self.alpha * (o_r_hdc * gate_think)

        x = x + ret_fused
        x = x + self.ffn(self.rms_ffn(x))
        return x, state_m, state_r, state_hdc

class WRAIX06BModel(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, num_layers=NUM_LAYERS, hidden_dim=HIDDEN_DIM, ffn_dim=FFN_DIM):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([
            WRAIXDualStateBlock(hidden_dim=hidden_dim, ffn_dim=ffn_dim)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.output_proj.weight = self.embed.weight

    def forward_step(self, token_id, states=None):
        x = self.embed(token_id)
        if states is None:
            states = [None] * self.num_layers

        new_states = []
        for l in range(self.num_layers):
            layer_state = states[l] if states[l] is not None else (None, None, None, 0)
            sm, sr, shdc, pos = layer_state
            x, sm, sr, shdc = self.layers[l].forward_step(
                x, step_pos=pos, state_m=sm, state_r=sr, state_hdc=shdc
            )
            new_states.append((sm, sr, shdc, pos + 1))

        x_norm = self.ln_final(x)
        logits = self.output_proj(x_norm)
        return logits, new_states

# -----------------------------------------------------------------------------
# 2. Decoding & Sampling Engine (Clean Standard Decoding)
# -----------------------------------------------------------------------------

def sample_token(l_tensor, generated, temperature=0.1, top_p=0.90, top_k=40, rep_penalty=1.05):
    l = l_tensor.squeeze(0).clone()

    # 1. Anti-stutter: Prevent identical token twice in a row
    if len(generated) >= 2 and generated[-1] == generated[-2]:
        l[generated[-1]] = float('-inf')

    # 2. Standard Multiplicative Repetition Penalty
    if len(generated) > 0 and rep_penalty != 1.0:
        recent = set(generated[-20:])
        for t in recent:
            if l[t] > 0:
                l[t] /= rep_penalty
            else:
                l[t] *= rep_penalty

    # 3. Deterministic Greedy decoding if temperature is low
    if temperature <= 0.05:
        return torch.argmax(l, dim=-1).item()

    # 4. Top-K Filtering
    if top_k > 0:
        topk_vals, _ = torch.topk(l, min(top_k, l.size(-1)))
        l[l < topk_vals[-1]] = float('-inf')

    # 5. Temperature Scaling & Softmax
    probs = F.softmax(l / temperature, dim=-1)

    # 6. Top-P (Nucleus) Filtering
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cum_probs = torch.cumsum(sorted_probs, dim=-1)
    mask = cum_probs > top_p
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = 0
    indices_to_remove = sorted_indices[mask]
    probs[indices_to_remove] = 0.0
    p_sum = probs.sum()
    if p_sum > 0:
        probs = probs / p_sum
        return torch.multinomial(probs, 1).item()
    return torch.argmax(l, dim=-1).item()

def generate_english_response(model, tok, user_query, max_tokens=150, temperature=0.1, rep_penalty=1.05):
    prompt = f"<|im_start|>system\nYou are a helpful, harmless, and honest AI assistant.<|im_end|>\n<|im_start|>user\n{user_query}<|im_end|>\n<|im_start|>assistant\n"
    p_ids = tok.encode(prompt, add_special_tokens=False)

    stop_ids = {tok.eos_token_id, 151643, 151644, 151645}
    for st in ["<|im_end|>", "<|im_start|>", "<|endoftext|>", "<|end|>"]:
        sid = tok.convert_tokens_to_ids(st)
        if isinstance(sid, int) and sid > 0:
            stop_ids.add(sid)

    states = None
    gen_tokens = []

    with torch.no_grad():
        # Ingest prompt tokens into constant O(1) recurrent states
        for tid in p_ids:
            t_tensor = torch.tensor([tid], device=DEVICE)
            logits, states = model.forward_step(t_tensor, states)

        # Autoregressive generation
        for _ in range(max_tokens):
            next_tok = sample_token(logits, gen_tokens, temperature=temperature, top_p=0.90, top_k=40, rep_penalty=rep_penalty)
            if next_tok in stop_ids:
                break
            gen_tokens.append(next_tok)
            word = tok.decode([next_tok])
            print(word, end="", flush=True)

            t_tensor = torch.tensor([next_tok], device=DEVICE)
            logits, states = model.forward_step(t_tensor, states)
    print()
    return tok.decode(gen_tokens)

# -----------------------------------------------------------------------------
# 3. Model Loader
# -----------------------------------------------------------------------------

def load_model(checkpoint_path=None):
    print("=" * 70)
    print("   🚀 WRAI-X (0.6B) ZERO KV-CACHE ENGLISH BENCHMARK STUDIO")
    print("=" * 70)
    print(f"[*] Execution Device: {DEVICE}")

    print("[*] Loading Qwen Tokenizer...")
    tok = AutoTokenizer.from_pretrained(SOURCE_MODEL_NAME, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("[*] Initializing WRAI-X 0.6B Architecture...")
    model = WRAIX06BModel(vocab_size=VOCAB_SIZE, num_layers=NUM_LAYERS, hidden_dim=HIDDEN_DIM, ffn_dim=FFN_DIM)

    # Locate Checkpoint
    ckpt_file = None
    candidates = [
        checkpoint_path,
        DRIVE_CHECKPOINT,
        LOCAL_CHECKPOINT,
        os.path.join(".", "wrai_x_06b_transplanted.pt")
    ]
    for c in candidates:
        if c and os.path.exists(c):
            ckpt_file = c
            break

    if ckpt_file:
        print(f"[OK] Found Checkpoint: {ckpt_file} ({os.path.getsize(ckpt_file)/1e6:.1f} MB)")
        print("[*] Loading Weights into Memory...")
        sd = torch.load(ckpt_file, map_location="cpu")
        model.load_state_dict(sd, strict=False)
        # Pastikan pointer tied weights terpasang 100% sempurna (untuk checkpoint ramping 1.19 GB)
        for l in range(NUM_LAYERS):
            model.layers[l].w_qr.weight = model.layers[l].w_q.weight
            model.layers[l].w_kr.weight = model.layers[l].w_k.weight
            model.layers[l].w_vr.weight = model.layers[l].w_v.weight
            model.layers[l].w_out_r.weight = model.layers[l].w_out.weight
        model.output_proj.weight = model.embed.weight
        del sd
        print(f"[OK SUCCESS] Weights loaded in {time.time()-t0:.2f}s!")
    else:
        print(f"[!] Warning: No checkpoint found at {DRIVE_CHECKPOINT} or locally.")
        print("[*] Performing Direct In-Memory Surgery Transplant from Qwen...")
        from transformers import AutoModelForCausalLM
        qwen = AutoModelForCausalLM.from_pretrained(SOURCE_MODEL_NAME, device_map="cpu", trust_remote_code=True)
        qsd = qwen.state_dict()
        model.embed.weight.data.copy_(qsd["model.embed_tokens.weight"][:VOCAB_SIZE])
        for l in range(NUM_LAYERS):
            model.layers[l].ffn.w_gate.weight.data.copy_(qsd[f"model.layers.{l}.mlp.gate_proj.weight"])
            model.layers[l].ffn.w_up.weight.data.copy_(qsd[f"model.layers.{l}.mlp.up_proj.weight"])
            model.layers[l].ffn.w_down.weight.data.copy_(qsd[f"model.layers.{l}.mlp.down_proj.weight"])
            model.layers[l].rms_ret.weight.data.copy_(qsd[f"model.layers.{l}.input_layernorm.weight"])
            model.layers[l].rms_ffn.weight.data.copy_(qsd[f"model.layers.{l}.post_attention_layernorm.weight"])
            model.layers[l].w_q.weight.data.copy_(qsd[f"model.layers.{l}.self_attn.q_proj.weight"])
            model.layers[l].w_out.weight.data.copy_(qsd[f"model.layers.{l}.self_attn.o_proj.weight"])
            kw = qsd[f"model.layers.{l}.self_attn.k_proj.weight"].view(8, 128, 1024).repeat_interleave(2, dim=0).reshape(2048, 1024)
            vw = qsd[f"model.layers.{l}.self_attn.v_proj.weight"].view(8, 128, 1024).repeat_interleave(2, dim=0).reshape(2048, 1024)
            model.layers[l].w_k.weight.data.copy_(kw)
            model.layers[l].w_v.weight.data.copy_(vw)
            model.layers[l].w_qr.weight = model.layers[l].w_q.weight
            model.layers[l].w_kr.weight = model.layers[l].w_k.weight
            model.layers[l].w_vr.weight = model.layers[l].w_v.weight
            model.layers[l].w_out_r.weight = model.layers[l].w_out.weight
        model.ln_final.weight.data.copy_(qsd["model.norm.weight"])
        del qwen, qsd
        gc.collect()
        print("[OK] Direct In-Memory Surgery Finished!")

    model.to(DEVICE)
    model.eval()
    return model, tok

# -----------------------------------------------------------------------------
# 4. Comprehensive English Benchmark Suite
# -----------------------------------------------------------------------------

ENGLISH_BENCHMARK_SUITE = [
    # 1. Science & Biology
    ("Science", "What is photosynthesis?"),
    ("Science", "Explain the concept of gravity."),
    ("Science", "What is DNA and what is its role in living organisms?"),
    ("Science", "What causes day and night on Earth?"),
    ("Science", "Why is the ocean blue?"),

    # 2. Computer Science & Systems
    ("Computer Science", "Explain the difference between a compiler and an interpreter."),
    ("Computer Science", "What is the role of RAM in a computer?"),
    ("Computer Science", "What is an algorithm?"),
    ("Computer Science", "Explain what an operating system does."),

    # 3. Programming (Python & C)
    ("Programming C", "Write a simple C program to print Hello World."),
    ("Programming Python", "Write a Python function to check if a word is a palindrome."),
    ("Programming Python", "Write a Python function to compute the factorial of a number."),
    ("Programming C", "Write a C function to swap two integers using pointers."),

    # 4. Math & Step-by-Step Chain-of-Thought
    ("Math Reasoning", "What is 12 multiplied by 15? Explain the steps."),
    ("Math Reasoning", "If today is Wednesday, what day will it be in 10 days?"),
    ("Math Reasoning", "Solve the equation: 2x + 6 = 14. What is the value of x?"),
    ("Math Reasoning", "What is 25% of 200,000?"),

    # 5. General Knowledge & Architecture
    ("General Knowledge", "What is the capital of France and what is it famous for?"),
    ("General Knowledge", "Who wrote Romeo and Juliet?"),
    ("Architecture", "Who are you and how does the WRAI-X architecture achieve Zero KV-Cache?")
]

def run_english_benchmark(model, tok, temperature=0.1):
    print("\n" + "=" * 70)
    print("   📊 RUNNING COMPREHENSIVE ENGLISH BENCHMARK (18 TASKS)")
    print("=" * 70)

    for i, (category, query) in enumerate(ENGLISH_BENCHMARK_SUITE, 1):
        print(f"\n[{i:02d}/18] [{category.upper()}]")
        print(f"User   > {query}")
        print("WRAI-X > ", end="", flush=True)
        t0 = time.time()
        generate_english_response(model, tok, query, max_tokens=220, temperature=temperature)
        elapsed = time.time() - t0
        print(f"         [Generation Time: {elapsed:.2f}s]")

    print("\n" + "=" * 70)
    print("   [OK] ENGLISH BENCHMARK COMPLETED SUCCESSFULLY!")
    print("=" * 70)

# -----------------------------------------------------------------------------
# 5. Interactive Chat Mode
# -----------------------------------------------------------------------------

def start_interactive_chat(model, tok):
    print("\n" + "=" * 70)
    print("   💬 WRAI-X (0.6B) LIVE ENGLISH CHAT STUDIO")
    print("   (Type your message and press Enter. Type 'exit' or 'quit' to stop)")
    print("=" * 70)

    while True:
        try:
            query = input("\nYou > ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit", "q"]:
                print("Exiting chat studio. Goodbye!")
                break

            print("WRAI-X > ", end="", flush=True)
            generate_english_response(model, tok, query, max_tokens=180, temperature=0.1)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat studio. Goodbye!")
            break

# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="WRAI-X 0.6B English Benchmark Studio")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to transplanted checkpoint (.pt)")
    parser.add_argument("--interactive", action="store_true", help="Start interactive live chat mode")
    parser.add_argument("--temp", type=float, default=0.1, help="Sampling temperature (default: 0.1)")
    args, _ = parser.parse_known_args()

    model, tok = load_model(args.ckpt)

    if args.interactive:
        start_interactive_chat(model, tok)
    else:
        run_english_benchmark(model, tok, temperature=args.temp)
        # Offer interactive chat after benchmark (works in terminal, Colab, or Jupyter)
        try:
            choice = input("\nWould you like to start interactive chat? [y/N]: ").strip().lower()
            if choice in ["y", "yes"]:
                start_interactive_chat(model, tok)
        except (EOFError, KeyboardInterrupt):
            pass
