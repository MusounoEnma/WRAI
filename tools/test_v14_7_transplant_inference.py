#!/usr/bin/env python3
"""
=============================================================================
 WRAI v14.7 TRANSPLANT — INFERENCE BENCHMARK & INTERACTIVE CHAT STUDIO
=============================================================================
 Evaluasi model WRAI hasil transplantasi bobot Qwen:
  - 100% 0% KV-Cache (Fixed 8 KB SRAM)
  - Wavelet Spectral + 12 Layer ResGRU + SwiGLU
  - Multi-Agent Pipeline & Translation Verification
=============================================================================
"""

import os
import sys
import math
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import AutoTokenizer
except ImportError:
    os.system("pip install -q transformers")
    from transformers import AutoTokenizer

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DRIVE_MOUNT_PATH = "/content/drive"
DRIVE_SAVE_DIR = "/content/drive/MyDrive/WRAI_Models_Transplant"
OUTPUT_DIR = "models_v14_7_transplant"
VOCAB_MAP_FILENAME = "vocab_map_v14_7_transplant.json"
TEACHER_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

drive_active = False
try:
    from google.colab import drive
    drive.mount(DRIVE_MOUNT_PATH, force_remount=False)
    drive_active = True
    print(f"\n[OK] GOOGLE DRIVE MOUNTED! Models loaded from dedicated folder: {DRIVE_SAVE_DIR}\n", flush=True)
except Exception as e:
    print(f"\n[INFO] Local environment detected (Drive mount skipped): {e}\n", flush=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Initializing WRAI v14.7 Transplant Inference Engine on Device: {DEVICE}")

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
        self.gate = nn.Linear(hidden_dim, hidden_dim, bias=False)

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

class WRAIv14_7Layer(nn.Module):
    def __init__(self, hidden_dim, ffn_dim):
        super().__init__()
        self.rms_gru = RMSNorm(hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.rms_ffn = RMSNorm(hidden_dim)
        self.ffn = SwiGLUFFN(hidden_dim, ffn_dim)

    def forward(self, x, h=None):
        res = x
        out_gru, next_h = self.gru(self.rms_gru(x), h)
        x = res + out_gru
        res = x
        x = res + self.ffn(self.rms_ffn(x))
        return x, next_h

class WRAIv14_7Model(nn.Module):
    def __init__(self, vocab_size, hidden_dim=896, ffn_dim=4864,
                 num_layers=12, max_seq_len=256):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encoder = SinusoidalPositionalEncoding(hidden_dim, max_seq_len)
        self.spectral1 = HaarDWT1D(hidden_dim, levels=4)
        self.spectral2 = HaarDWT1D(hidden_dim, levels=4)
        self.layers = nn.ModuleList([WRAIv14_7Layer(hidden_dim, ffn_dim) for _ in range(num_layers)])
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

def load_transplant_model():
    print("[*] Loading Tokenizer Base: Qwen/Qwen2.5-3B-Instruct...")
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_NAME, trust_remote_code=True)

    vocab_paths = [
        os.path.join(DRIVE_SAVE_DIR, VOCAB_MAP_FILENAME),
        os.path.join(OUTPUT_DIR, VOCAB_MAP_FILENAME),
        VOCAB_MAP_FILENAME
    ]
    vocab_data = None
    for vp in vocab_paths:
        if os.path.exists(vp):
            with open(vp, "r", encoding="utf-8") as f:
                vocab_data = json.load(f)
            print(f"[OK] Vocab Map loaded from: {vp}")
            break

    if not vocab_data:
        raise FileNotFoundError("Vocab map transplant tidak ditemukan!")

    teacher_to_pruned = {int(k): v for k, v in vocab_data["teacher_id_to_pruned_id"].items()}
    pruned_to_teacher = {int(k): v for k, v in vocab_data["pruned_id_to_teacher_id"].items()}
    tag_to_id = vocab_data.get("tag_to_id", {"<ID>": 31997, "<EN>": 31998})
    id_to_tag = {v: k for k, v in tag_to_id.items()}
    total_vocab_size = vocab_data["total_vocab_size"]

    model = WRAIv14_7Model(vocab_size=total_vocab_size).to(DEVICE)

    candidate_checkpoints = [
        os.path.join(DRIVE_SAVE_DIR, "wrai_v14_7_transplant_best.pt"),
        os.path.join(DRIVE_SAVE_DIR, "wrai_v14_7_transplant_latest.pt"),
        os.path.join(OUTPUT_DIR, "wrai_v14_7_transplant_best.pt"),
        os.path.join(OUTPUT_DIR, "wrai_v14_7_transplant_latest.pt"),
    ]
    loaded_ckpt = False
    for p in candidate_checkpoints:
        if os.path.exists(p):
            try:
                ckpt = torch.load(p, map_location=DEVICE)
                state = ckpt.get("model_state", ckpt)
                if "embed.weight" in state:
                    ckpt_vocab = state["embed.weight"].size(0)
                    if ckpt_vocab != model.vocab_size:
                        print(f"  [ADAPT] Matching model vocab size to checkpoint: {ckpt_vocab}")
                        model = WRAIv14_7Model(vocab_size=ckpt_vocab).to(DEVICE)
                model.load_state_dict(state, strict=False)
                model.output_proj.weight = model.embed.weight
                print(f"[OK SUCCESS] WRAI Weights Loaded from: {p}")
                if isinstance(ckpt, dict) and "loss" in ckpt:
                    print(f"  --> Checkpoint Loss: {ckpt['loss']:.4f} | Epoch: {ckpt.get('epoch', '?')} | Step: {ckpt.get('step', '?')}")
                loaded_ckpt = True
                break
            except Exception as e:
                print(f"[WARN] Gagal memuat {p}: {e}")

    if not loaded_ckpt:
        print("[!] PERINGATAN: Tidak ada file .pt ditemukan. Menjalankan model dalam mode inisialisasi.")

    model.eval()
    return model, tokenizer, teacher_to_pruned, pruned_to_teacher, tag_to_id, id_to_tag

def generate_response(model, tokenizer, teacher_to_pruned, pruned_to_teacher, id_to_tag,
                      prompt: str, max_new_tokens: int = 150, temperature: float = 0.6,
                      top_k: int = 40, repetition_penalty: float = 1.15):
    q_ids = tokenizer.encode(prompt, add_special_tokens=False)
    q_pruned = [teacher_to_pruned.get(t, 1) for t in q_ids]

    input_tokens = list(q_pruned)
    gen_tokens = []
    detected_tag = None

    t0 = time.perf_counter()
    with torch.no_grad():
        for step in range(max_new_tokens):
            cur_seq = torch.tensor([input_tokens], dtype=torch.long, device=DEVICE)
            logits, _ = model(cur_seq)
            next_logits = logits[0, -1, :].clone()

            for token_id in set(gen_tokens[-20:]):
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

            if step == 0:
                tag_ids = list(id_to_tag.keys())
                if next_token in id_to_tag:
                    best_tag_id = next_token
                else:
                    tag_sublogits = next_logits[tag_ids]
                    best_tag_id = tag_ids[torch.argmax(tag_sublogits).item()]

                detected_tag = id_to_tag.get(best_tag_id, "<ID>")
                input_tokens.append(best_tag_id)
                continue

            if next_token == 0:
                break
            
            gen_tokens.append(next_token)
            input_tokens.append(next_token)

            if next_token in pruned_to_teacher:
                teacher_tid = pruned_to_teacher[next_token]
                if teacher_tid is not None:
                    tok_str = tokenizer.decode([int(teacher_tid)])
                    if "<|im_end|>" in tok_str or "<|endoftext|>" in tok_str:
                        break

            if len(input_tokens) >= 256:
                break

    elapsed = max(0.001, time.perf_counter() - t0)
    speed = len(gen_tokens) / elapsed

    teacher_tokens = []
    for t in gen_tokens:
        if t in pruned_to_teacher:
            tid = pruned_to_teacher[t]
            if tid is not None:
                teacher_tokens.append(int(tid))

    response_text = tokenizer.decode(teacher_tokens, skip_special_tokens=True).strip() if teacher_tokens else ""

    return {
        "tag": detected_tag or "Auto",
        "response": response_text,
        "tokens": len(gen_tokens),
        "speed": speed,
        "elapsed": elapsed
    }

def run_benchmark_suite(model, tokenizer, teacher_to_pruned, pruned_to_teacher, id_to_tag):
    print("\n" + "="*70)
    print("      🚀 WRAI v14.7 TRANSPLANT BENCHMARK EVALUATION                  ")
    print("="*70)

    test_cases = [
        ("🤖 WRAI Swarm Persona", "Jelaskan siapa kamu dan bagaimana ekosistem WRAI bekerja."),
        ("🌐 Penerjemah Inggris -> Indo", "Terjemahkan kalimat berikut ke bahasa Indonesia:\n\"Artificial Intelligence empowers edge devices with ultra-low latency.\""),
        ("💻 Delegate to WRAI-Coder", "Buatkan script python untuk web scraping data harga"),
        ("🔒 Delegate to WRAI-Security", "Analisis potensi celah SQL Injection pada query database ini"),
        ("🔄 Multi-Agent Pipeline (3 Models)", "Cari berita celah zero-day terbaru di internet lalu analisis dan buatkan script patch perbaikannya di Python."),
        ("🇮🇩 Sains & Nalar Umum", "Mengapa langit terlihat berwarna biru pada siang hari?")
    ]

    for domain, prompt in test_cases:
        print(f"\n----------------------------------------------------------------------")
        print(f"\033[1;33m[TEST DOMAIN]\033[0m {domain}")
        print(f"\033[1;37m[USER PROMPT]\033[0m \"{prompt}\"")
        
        result = generate_response(
            model, tokenizer, teacher_to_pruned, pruned_to_teacher, id_to_tag,
            prompt, max_new_tokens=120, temperature=0.6, top_k=40
        )

        print(f"\033[1;35m[INTERNAL ROUTING]\033[0m {result['tag']}")
        print(f"\033[1;36m[WRAI v14.7]\033[0m {result['response']}")
        print(f"\033[90m({result['tokens']} tokens | {result['speed']:.1f} tok/s | {result['elapsed']:.2f}s)\033[0m")

    print("\n" + "="*70 + "\n")

def interactive_chat(model, tokenizer, teacher_to_pruned, pruned_to_teacher, id_to_tag):
    print("="*70)
    print("💬 LIVE INTERACTIVE CHAT STUDIO — WRAI v14.7 TRANSPLANT")
    print("Ketik 'exit' atau 'keluar' untuk selesai")
    print("="*70)

    while True:
        try:
            prompt = input("\n\033[1;32mUser:\033[0m ").strip()
            if not prompt:
                continue
            if prompt.lower() in ("exit", "keluar", "quit", "q"):
                print("\n[👋] Terima kasih telah menguji WRAI v14.7 Transplant!\n")
                break

            result = generate_response(
                model, tokenizer, teacher_to_pruned, pruned_to_teacher, id_to_tag,
                prompt, max_new_tokens=150, temperature=0.65, top_k=40
            )

            print(f"\033[1;36mWRAI:\033[0m {result['response']}")
            print(f"\033[90m[Mode: {result['tag']} | {result['tokens']} tokens | {result['speed']:.1f} tok/s]\033[0m")
        except KeyboardInterrupt:
            print("\n\n[Session Ended]")
            break

def main():
    model, tokenizer, teacher_to_pruned, pruned_to_teacher, tag_to_id, id_to_tag = load_transplant_model()
    run_benchmark_suite(model, tokenizer, teacher_to_pruned, pruned_to_teacher, id_to_tag)
    interactive_chat(model, tokenizer, teacher_to_pruned, pruned_to_teacher, id_to_tag)

if __name__ == "__main__":
    main()
