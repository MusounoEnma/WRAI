#!/usr/bin/env python3
"""
=============================================================================
 WRAI v14.7 — QWEN-TO-WRAI WEIGHT TRANSPLANT & RAPID ADAPTATION TRAINER
=============================================================================
 Architecture:
  - 100% Zero KV-Cache (Fixed 8 KB SRAM Ping-Pong Buffer)
  - 12-Layer ResGRU + SwiGLU FFN + 4-Level Haar DWT Spectral Mixer
  - Dimensions: Hidden=896, FFN=4864, Layers=12, Vocab=32000 (~245M Parameters)
  - Initialization: Mature Weight Transplant from Qwen2.5-0.5B-Instruct
  - Distillation: Shift-1 Clean Alignment from Qwen2.5-3B-Instruct
=============================================================================
"""

import os
import sys
import math
import time
import json
import random
import shutil
import gc
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader

try:
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
    from huggingface_hub import login
except ImportError:
    os.system("pip install -q datasets huggingface_hub accelerate bitsandbytes")
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
    from huggingface_hub import login

# -----------------------------------------------------------------------------
# 1. Configuration & Hyperparameters
# -----------------------------------------------------------------------------
HF_TOKEN = os.environ.get("HF_TOKEN", "")
os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN

TEACHER_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
TRANSPLANT_SOURCE_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
ROUTE_TAGS = ["<ID>", "<EN>"]

# --- Kuota Dataset Multi-Domain ---
N_ID = 40000                 # Alpaca GPT-4 Indonesian
N_EN = 40000                 # Alpaca Cleaned English
N_TRANS = 20000              # OPUS-100 Parallel EN-ID Translation
N_TOOL = 11500               # Hermes Function Calling v1
N_PERSONA = 5000             # WRAI Swarm Router & Pipeline Bank
SHUFFLE_BUFFER_SIZE = 2500
VOCAB_CAP = 32000
MAX_CHARS_RESPONSE = 800

# --- Hyperparameters Arsitektur WRAI Transplant ---
HIDDEN_DIM = 896             # Sesuai Qwen 0.5B (896 / 16 = 56 untuk Haar DWT & AVX)
FFN_INTERMEDIATE_DIM = 4864  # Sesuai Qwen 0.5B SwiGLU FFN
NUM_LAYERS = 12              # 12 Layer ResGRU + SwiGLU
WAVELET_LEVELS = 4           # 4-Level Haar DWT Spectral
MAX_SEQ_LEN = 256
BATCH_SIZE = 8               # Batch size per step (Aman VRAM 15GB GPU)
GRAD_ACCUM_STEPS = 8         # Effective batch size = 64 (8 x 8 = 64)
LEARNING_RATE = 2.5e-4       # Warm-start fine-tuning LR
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 2               # 2 Epoch adaptasi kilat (~25 menit)
SAVE_EVERY_STEPS = 1000      # Auto-save ke Google Drive setiap 1000 steps

DISTILL_TEMP = 2.0
ALPHA_DISTILL = 0.7
ALPHA_CE = 0.3
PAD_ID = 0
UNK_ID = 1

DRIVE_MOUNT_PATH = "/content/drive"
DRIVE_SAVE_DIR = "/content/drive/MyDrive/WRAI_Models_Transplant"
OUTPUT_DIR = "models_v14_7_transplant"
VOCAB_MAP_FILENAME = "vocab_map_v14_7_transplant.json"
os.makedirs(OUTPUT_DIR, exist_ok=True)

drive_active = False
try:
    from google.colab import drive
    drive.mount(DRIVE_MOUNT_PATH, force_remount=False)
    os.makedirs(DRIVE_SAVE_DIR, exist_ok=True)
    drive_active = True
    print(f"\n[OK] GOOGLE DRIVE MOUNTED! Models will auto-save to dedicated folder: {DRIVE_SAVE_DIR}\n", flush=True)
except Exception as e:
    print(f"\n[INFO] Local environment detected (Drive mount skipped): {e}\n", flush=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Initializing WRAI Transplant Engine on Device: {DEVICE}")

# -----------------------------------------------------------------------------
# 2. Neural Architecture WRAI (0% KV-Cache, Wavelet Spectral, SwiGLU, ResGRU)
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
    def __init__(self, vocab_size, hidden_dim=HIDDEN_DIM, ffn_dim=FFN_INTERMEDIATE_DIM,
                 num_layers=NUM_LAYERS, max_seq_len=MAX_SEQ_LEN):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encoder = SinusoidalPositionalEncoding(hidden_dim, max_seq_len)
        self.spectral1 = HaarDWT1D(hidden_dim, levels=WAVELET_LEVELS)
        self.spectral2 = HaarDWT1D(hidden_dim, levels=WAVELET_LEVELS)
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
# 3. Qwen-to-WRAI Weight Transplantation Engine
# -----------------------------------------------------------------------------

def transplant_weights_from_qwen(student_model, pruned_to_teacher, source_model_name=TRANSPLANT_SOURCE_NAME):
    """
    Mencangkok bobot pre-trained Qwen2.5 ke dalam model WRAI:
    - 32k Token Embeddings langsung dicangkok
    - 12 Layer SwiGLU FFN (w_gate, w_up, w_down) dicangkok 1-to-1
    - RMSNorm Scaling dicangkok 1-to-1
    - ResGRU Recurrent Gates di-prime dari Attention matrices
    """
    print(f"\n[*] MEMULAI TRANSPLANTASI BOBOT DARI {source_model_name} KE WRAI...", flush=True)
    t0 = time.perf_counter()

    qwen_source = AutoModelForCausalLM.from_pretrained(
        source_model_name,
        device_map="cpu",
        trust_remote_code=True
    )
    qwen_sd = qwen_source.state_dict()

    # 1. Slicing Embeddings
    print("  --> [1/4] Mencangkok 32.000 Token Embeddings...", flush=True)
    qwen_embed = qwen_sd["model.embed_tokens.weight"] # [151936, 896]
    with torch.no_grad():
        for pruned_id, teacher_id in pruned_to_teacher.items():
            if pruned_id < student_model.vocab_size and teacher_id < qwen_embed.size(0):
                student_model.embed.weight.data[pruned_id] = qwen_embed[teacher_id]

    # 2. Transplant SwiGLU FFN Layers & RMSNorms
    print("  --> [2/4] Mencangkok 12 Layer SwiGLU FFN & RMSNorm...", flush=True)
    total_qwen_layers = qwen_source.config.num_hidden_layers
    step_stride = total_qwen_layers // len(student_model.layers)

    with torch.no_grad():
        for i, student_layer in enumerate(student_model.layers):
            src_idx = min(i * step_stride, total_qwen_layers - 1)
            
            # SwiGLU FFN
            gate_w = qwen_sd[f"model.layers.{src_idx}.mlp.gate_proj.weight"]
            up_w   = qwen_sd[f"model.layers.{src_idx}.mlp.up_proj.weight"]
            down_w = qwen_sd[f"model.layers.{src_idx}.mlp.down_proj.weight"]
            student_layer.ffn.w_gate.weight.data.copy_(gate_w)
            student_layer.ffn.w_up.weight.data.copy_(up_w)
            student_layer.ffn.w_down.weight.data.copy_(down_w)

            # RMSNorms
            in_norm = qwen_sd[f"model.layers.{src_idx}.input_layernorm.weight"]
            post_norm = qwen_sd[f"model.layers.{src_idx}.post_attention_layernorm.weight"]
            student_layer.rms_gru.weight.data.copy_(in_norm)
            student_layer.rms_ffn.weight.data.copy_(post_norm)

        # Final Norm
        final_norm = qwen_sd["model.norm.weight"]
        student_model.ln_final.weight.data.copy_(final_norm)

    # 3. Mimetic Priming GRU Gates dari Attention Matrices
    print("  --> [3/4] Melakukan Mimetic Priming pada ResGRU Recurrent Gates...", flush=True)
    with torch.no_grad():
        for i, student_layer in enumerate(student_model.layers):
            src_idx = min(i * step_stride, total_qwen_layers - 1)
            q_w = qwen_sd[f"model.layers.{src_idx}.self_attn.q_proj.weight"] # [896, 896]
            k_w = qwen_sd[f"model.layers.{src_idx}.self_attn.k_proj.weight"] # [128, 896]
            v_w = qwen_sd[f"model.layers.{src_idx}.self_attn.v_proj.weight"] # [128, 896]

            # Replicate k, v to 896
            k_expanded = k_w.repeat(896 // k_w.size(0), 1)
            v_expanded = v_w.repeat(896 // v_w.size(0), 1)

            # GRU weight_ih has shape [3 * 896, 896] -> (reset, update, new)
            w_ih_combined = torch.cat([q_w * 0.5, k_expanded * 0.5, v_expanded * 0.5], dim=0)
            student_layer.gru.weight_ih_l0.data.copy_(w_ih_combined)
            nn.init.orthogonal_(student_layer.gru.weight_hh_l0.data)

    del qwen_source, qwen_sd
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[OK SUCCESS] Transplantasi Selesai 100% dalam {(time.perf_counter()-t0):.2f}s!\n", flush=True)

# -----------------------------------------------------------------------------
# 4. Multi-Domain & Swarm Data Streaming Sources
# -----------------------------------------------------------------------------

def build_wrai_persona_dataset():
    """Koleksi dialog WRAI Master: Identitas, Swarm Colony Router & Multi-Agent Pipeline"""
    qa_list = [
        # --- 1. WRAI Central Router Persona & Swarm Meta-Awareness (ID) ---
        ("<ID>", "Halo", "Halo! Saya WRAI Master Router. Ada yang bisa saya bantu atau arahkan ke spesialis AI hari ini?"),
        ("<ID>", "Halo apa kabar?", "Halo! Kabar saya sangat baik dan siap mengorkestrasi tugas Anda. Ada yang ingin Anda diskusikan?"),
        ("<ID>", "Hai", "Hai! Selamat datang di ekosistem WRAI. Apa yang ingin Anda tanyakan atau delegasikan hari ini?"),
        ("<ID>", "Siapa kamu?", "Saya adalah WRAI (Wavelet Recurrent AI), model router sentral dan orkestrator koloni AI edge yang ringan, cepat, dan efisien."),
        ("<ID>", "Siapa namamu?", "Nama saya WRAI Master Router. Saya bertugas mengelola percakapan dan mendelegasikan tugas khusus ke koloni model spesialis WRAI."),
        ("<ID>", "Jelaskan siapa kamu dan bagaimana ekosistem WRAI bekerja.", "Saya adalah WRAI Master Router. Saya memimpin koloni model edge AI spesialis. Untuk percakapan dan pertanyaan umum, saya merespons langsung. Namun ketika Anda meminta tugas khusus, saya secara otomatis memanggil agen spesialis kami:\n1. WRAI-Coder (Pemrograman & Algoritma Python/C)\n2. WRAI-Security (Keamanan Siber & Analisis Kerentanan)\n3. WRAI-Browser (Penelusuran Web & Berita Realtime)\n4. WRAI-Math (Kalkulasi Simbolik & Matematika Murni)"),
        ("<ID>", "Apa kelebihan arsitektur WRAI?", "WRAI dibangun dengan arsitektur hybrid Wavelet Spectral Transform + ResGRU + SwiGLU yang berjalan tanpa KV-Cache (0% KV-Cache), berbobot hanya ~150MB int8, dan dapat di-swap dalam 145ms antar model spesialis pada perangkat edge."),
        ("<ID>", "Terima kasih banyak", "Sama-sama! Senang bisa membantu Anda. Beritahu saya jika Anda butuh bantuan spesialis kami lainnya."),
        ("<ID>", "Selamat pagi", "Selamat pagi! WRAI Swarm siap membantu produktivitas Anda hari ini."),
        ("<ID>", "Selamat malam", "Selamat malam! Ada tugas atau riset yang ingin diselesaikan sebelum beristirahat?"),

        # --- 2. Swarm Routing to WRAI-Coder (Programming, Python, C, Debugging) ---
        ("<EN>", "Buatkan script python untuk web scraping data harga", "<tool_call>\n{\"name\": \"route_to_agent\", \"arguments\": {\"agent\": \"wrai_coder\", \"task\": \"Buatkan script python untuk web scraping data harga\"}}\n</tool_call>"),
        ("<EN>", "Tulis fungsi binary search di Python yang efisien", "<tool_call>\n{\"name\": \"route_to_agent\", \"arguments\": {\"agent\": \"wrai_coder\", \"task\": \"Tulis fungsi binary search di Python yang efisien\"}}\n</tool_call>"),
        ("<EN>", "Write a fast matrix multiplication algorithm in C with AVX", "<tool_call>\n{\"name\": \"route_to_agent\", \"arguments\": {\"agent\": \"wrai_coder\", \"task\": \"Write a fast matrix multiplication algorithm in C with AVX\"}}\n</tool_call>"),
        ("<EN>", "Debug kode python ini yang mengalami IndexError: list index out of range", "<tool_call>\n{\"name\": \"route_to_agent\", \"arguments\": {\"agent\": \"wrai_coder\", \"task\": \"Debug kode python IndexError list index out of range\"}}\n</tool_call>"),
        ("<EN>", "Bagaimana cara membuat REST API dengan FastAPI di Python?", "<tool_call>\n{\"name\": \"route_to_agent\", \"arguments\": {\"agent\": \"wrai_coder\", \"task\": \"Tutorial dan template pembuatan REST API dengan FastAPI di Python\"}}\n</tool_call>"),
        ("<EN>", "Write a Python script to parse JSON and export to SQLite", "<tool_call>\n{\"name\": \"route_to_agent\", \"arguments\": {\"agent\": \"wrai_coder\", \"task\": \"Write a Python script to parse JSON and export to SQLite\"}}\n</tool_call>"),

        # --- 3. Swarm Routing to WRAI-Security (Cyber Security, Pentest, Exploits, Audits) ---
        ("<EN>", "Analisis potensi celah SQL Injection pada query database ini", "<tool_call>\n{\"name\": \"route_to_agent\", \"arguments\": {\"agent\": \"wrai_security\", \"task\": \"Analisis potensi celah SQL Injection dan remedi parameterization\"}}\n</tool_call>"),
        ("<EN>", "Bagaimana cara kerja serangan buffer overflow dan cara memitigasinya?", "<tool_call>\n{\"name\": \"route_to_agent\", \"arguments\": {\"agent\": \"wrai_security\", \"task\": \"Penjelasan teknis serangan buffer overflow dan mitigasi ASLR/DEP/Canary\"}}\n</tool_call>"),
        ("<EN>", "Inspect this suspicious bash reverse shell payload", "<tool_call>\n{\"name\": \"route_to_agent\", \"arguments\": {\"agent\": \"wrai_security\", \"task\": \"Inspect and deconstruct suspicious bash reverse shell payload\"}}\n</tool_call>"),
        ("<EN>", "Audit konfigurasi firewall iptables dan deteksi port yang terbuka", "<tool_call>\n{\"name\": \"route_to_agent\", \"arguments\": {\"agent\": \"wrai_security\", \"task\": \"Audit konfigurasi firewall iptables dan rekomendasi hardening port\"}}\n</tool_call>"),

        # --- 4. Swarm Routing to WRAI-Browser (Live Web Search, News, Weather) ---
        ("<EN>", "Cari berita teknologi AI dan model terbaru hari ini di internet.", "<tool_call>\n{\"name\": \"route_to_agent\", \"arguments\": {\"agent\": \"wrai_browser\", \"query\": \"berita teknologi AI dan model terbaru hari ini\"}}\n</tool_call>"),
        ("<EN>", "Search the web for the latest artificial intelligence news today.", "<tool_call>\n{\"name\": \"route_to_agent\", \"arguments\": {\"agent\": \"wrai_browser\", \"query\": \"latest artificial intelligence news today\"}}\n</tool_call>"),
        ("<EN>", "Cari informasi harga emas dan kurs dollar rupiah hari ini.", "<tool_call>\n{\"name\": \"route_to_agent\", \"arguments\": {\"agent\": \"wrai_browser\", \"query\": \"harga emas antam dan kurs dollar rupiah hari ini\"}}\n</tool_call>"),
        ("<EN>", "Check the current weather forecast in Jakarta.", "<tool_call>\n{\"name\": \"route_to_agent\", \"arguments\": {\"agent\": \"wrai_browser\", \"query\": \"current weather forecast Jakarta today\"}}\n</tool_call>"),

        # --- 5. Swarm Routing to WRAI-Math (Complex Math, Calculus, Equations) ---
        ("<EN>", "Hitung turunan pertama dari f(x) = x^3 * e^(2x)", "<tool_call>\n{\"name\": \"route_to_agent\", \"arguments\": {\"agent\": \"wrai_math\", \"expression\": \"derivative(x^3 * exp(2*x), x)\"}}\n</tool_call>"),
        ("<EN>", "Calculate the integral of 1 / (1 + x^2) from 0 to 1", "<tool_call>\n{\"name\": \"route_to_agent\", \"arguments\": {\"agent\": \"wrai_math\", \"expression\": \"integrate(1 / (1 + x^2), (x, 0, 1))\"}}\n</tool_call>"),
        ("<EN>", "Selesaikan persamaan kuadrat 2x^2 + 5x - 3 = 0", "<tool_call>\n{\"name\": \"route_to_agent\", \"arguments\": {\"agent\": \"wrai_math\", \"expression\": \"solve(2*x^2 + 5*x - 3, x)\"}}\n</tool_call>"),

        # --- 6. Multi-Agent Pipeline & Chained Routing ---
        ("<EN>", "Cari berita celah zero-day terbaru di internet lalu analisis dan buatkan script patch perbaikannya di Python.", "<tool_call>\n{\"name\": \"route_pipeline\", \"arguments\": {\"steps\": [{\"step\": 1, \"agent\": \"wrai_browser\", \"task\": \"Cari berita celah zero-day terbaru\"}, {\"step\": 2, \"agent\": \"wrai_security\", \"task\": \"Analisis akar penyebab exploit dan vektor serangan\"}, {\"step\": 3, \"agent\": \"wrai_coder\", \"task\": \"Tulis script Python patch perbaikan dan verifikasi\"}]}}\n</tool_call>"),
        ("<EN>", "Audit kode C ini untuk celah buffer overflow lalu tulis ulang kodenya secara aman dengan sanitasi input.", "<tool_call>\n{\"name\": \"route_pipeline\", \"arguments\": {\"steps\": [{\"step\": 1, \"agent\": \"wrai_security\", \"task\": \"Audit kode C untuk celah buffer overflow dan memory safety\"}, {\"step\": 2, \"agent\": \"wrai_coder\", \"task\": \"Refactor dan tulis ulang fungsi C menggunakan bounded buffer dan safe string functions\"}]}}\n</tool_call>"),
        ("<EN>", "Cari data historis harga saham IHSG 5 hari terakhir di web lalu hitung rata-rata dan standar deviasinya.", "<tool_call>\n{\"name\": \"route_pipeline\", \"arguments\": {\"steps\": [{\"step\": 1, \"agent\": \"wrai_browser\", \"task\": \"Ambil data historis harga penutupan IHSG 5 hari terakhir\"}, {\"step\": 2, \"agent\": \"wrai_math\", \"task\": \"Kalkulasi statistik mean dan standard deviation dari data harga\"}]}}\n</tool_call>"),
        ("<EN>", "Buatkan sistem autentikasi JWT di FastAPI lalu audit keamanannya terhadap token tampering dan signature stripping.", "<tool_call>\n{\"name\": \"route_pipeline\", \"arguments\": {\"steps\": [{\"step\": 1, \"agent\": \"wrai_coder\", \"task\": \"Implementasi endpoint autentikasi JWT di FastAPI Python\"}, {\"step\": 2, \"agent\": \"wrai_security\", \"task\": \"Lakukan security audit dan uji ketahanan terhadap token tampering\"}]}}\n</tool_call>"),

        # --- 7. Direct Factual & Conversational Knowledge (ID & EN) ---
        ("<ID>", "Mengapa langit terlihat berwarna biru pada siang hari?", "Langit berwarna biru pada siang hari karena fenomena hamburan Rayleigh. Partikel gas di atmosfer bumi menghamburkan cahaya matahari dengan panjang gelombang pendek (warna biru dan violet) ke segala penjuru jauh lebih kuat dibandingkan cahaya bergelombang panjang seperti merah."),
        ("<ID>", "Jelaskan secara singkat bagaimana proses fotosintesis terjadi pada tumbuhan.", "Fotosintesis adalah proses biokimia di mana tumbuhan hijau mengubah air (H2O) dan karbon dioksida (CO2) menjadi glukosa dan oksigen (O2) dengan memanfaatkan energi cahaya matahari yang diserap oleh pigmen klorofil di dalam kloroplas."),
        ("<ID>", "Berikan 3 tips utama untuk menjaga kesehatan tubuh setiap hari.", "Berikut 3 tips utama menjaga kesehatan tubuh setiap hari:\n1. Mengonsumsi makanan bergizi seimbang dan cukup minum air putih (minimal 2 liter sehari).\n2. Melakukan aktivitas fisik atau olahraga ringan minimal 30 menit setiap hari.\n3. Memastikan istirahat dan tidur yang cukup (7-8 jam per malam) untuk regenerasi sel tubuh."),
        ("<ID>", "Apa ibu kota Indonesia?", "Ibu kota Indonesia saat ini adalah Jakarta, dan sedang dalam proses transisi pembangunan ke Ibu Kota Nusantara (IKN) di Kalimantan Timur."),
        ("<EN>", "Who are you?", "I am WRAI (Wavelet Recurrent AI), the master router and orchestrator of an edge AI swarm."),
        ("<EN>", "What are the core advantages of small language models on edge devices?", "The primary advantages of running small models on edge devices are:\n1. 100% Data Privacy (inference stays on local hardware).\n2. Ultra-Low Latency (instant token streaming without cloud roundtrips).\n3. Zero Server API Costs and full offline resilience.")
    ]
    return qa_list

def stream_wrai_persona(n=N_PERSONA):
    qa_list = build_wrai_persona_dataset()
    count = 0
    while count < n:
        random.shuffle(qa_list)
        for tag, q, a in qa_list:
            yield (tag, q, a + "<|im_end|>")
            count += 1
            if count >= n:
                return

def _iter_sharegpt_pairs(convo):
    pairs = []
    turns = convo.get("conversations", [])
    last_human = None
    for turn in turns:
        role = turn.get("from", "") or turn.get("role", "")
        text = turn.get("value", "") or turn.get("content", "")
        if role in ("human", "user"):
            last_human = text
        elif role in ("gpt", "assistant") and last_human is not None:
            pairs.append((last_human, text))
            last_human = None
    return pairs

def stream_alpaca_id(n=N_ID):
    ds = load_dataset("FreedomIntelligence/alpaca-gpt4-indonesian", split="train", streaming=True)
    ds = ds.shuffle(seed=42, buffer_size=SHUFFLE_BUFFER_SIZE)
    count = 0
    for row in ds:
        for q, a in _iter_sharegpt_pairs(row):
            q, a = q.strip(), a.strip()
            if len(a) <= MAX_CHARS_RESPONSE and len(q) > 0 and len(a) > 0:
                yield ("<ID>", q, a + "<|im_end|>")
                count += 1
                if count >= n:
                    return

def stream_alpaca_en(n=N_EN):
    ds = load_dataset("yahma/alpaca-cleaned", split="train", streaming=True)
    ds = ds.shuffle(seed=42, buffer_size=SHUFFLE_BUFFER_SIZE)
    count = 0
    for row in ds:
        if "conversations" in row:
            for q, a in _iter_sharegpt_pairs(row):
                q, a = q.strip(), a.strip()
                if len(a) <= MAX_CHARS_RESPONSE and len(q) > 0 and len(a) > 0:
                    yield ("<EN>", q, a + "<|im_end|>")
                    count += 1
                    if count >= n:
                        return
        elif "instruction" in row:
            q = row["instruction"].strip()
            extra = row.get("input", "")
            if extra:
                q = f"{q}\n{extra.strip()}"
            a = row["output"].strip()
            if len(a) <= MAX_CHARS_RESPONSE and len(q) > 0 and len(a) > 0:
                yield ("<EN>", q, a + "<|im_end|>")
                count += 1
                if count >= n:
                    return

def stream_opus_translation(n=N_TRANS):
    try:
        ds = load_dataset("Helsinki-NLP/opus-100", "en-id", split="train", streaming=True)
    except Exception:
        try:
            ds = load_dataset("Helsinki-NLP/tatoeba_mt", "eng-ind", split="train", streaming=True)
        except Exception:
            return
    ds = ds.shuffle(seed=42, buffer_size=SHUFFLE_BUFFER_SIZE)
    count = 0
    for row in ds:
        trans = row.get("translation", {})
        en = trans.get("en", "") or trans.get("eng", "")
        id_txt = trans.get("id", "") or trans.get("ind", "")
        en, id_txt = en.strip(), id_txt.strip()
        if not en or not id_txt or len(en) > MAX_CHARS_RESPONSE or len(id_txt) > MAX_CHARS_RESPONSE:
            continue
        
        if random.random() < 0.5:
            q = f"Terjemahkan kalimat berikut ke bahasa Indonesia:\n\"{en}\""
            a = id_txt
            yield ("<ID>", q, a + "<|im_end|>")
        else:
            q = f"Translate the following sentence into English:\n\"{id_txt}\""
            a = en
            yield ("<EN>", q, a + "<|im_end|>")
        count += 1
        if count >= n:
            return

def stream_hermes_tool(n=N_TOOL):
    ds = load_dataset("NousResearch/hermes-function-calling-v1",
                       "func_calling_singleturn", split="train", streaming=True)
    ds = ds.shuffle(seed=42, buffer_size=SHUFFLE_BUFFER_SIZE)
    count = 0
    for convo in ds:
        for q, a in _iter_sharegpt_pairs(convo):
            q, a = q.strip(), a.strip()
            if "<tool_call>" in a and len(a) <= MAX_CHARS_RESPONSE:
                yield ("<EN>", q, a + "<|im_end|>")
                count += 1
                if count >= n:
                    return

def round_robin(*generators):
    iters = [iter(g) for g in generators]
    while iters:
        for it in list(iters):
            try:
                yield next(it)
            except StopIteration:
                iters.remove(it)

def make_combined_stream():
    return round_robin(
        stream_wrai_persona(N_PERSONA),
        stream_alpaca_id(N_ID),
        stream_alpaca_en(N_EN),
        stream_opus_translation(N_TRANS),
        stream_hermes_tool(N_TOOL)
    )

# -----------------------------------------------------------------------------
# 5. Clean Vocab Map Setup
# -----------------------------------------------------------------------------

def get_or_build_vocab_map(tokenizer, vocab_cap=VOCAB_CAP):
    candidate_paths = []
    if drive_active:
        candidate_paths.append(os.path.join(DRIVE_SAVE_DIR, VOCAB_MAP_FILENAME))
    candidate_paths.append(os.path.join(OUTPUT_DIR, VOCAB_MAP_FILENAME))
    candidate_paths.append(VOCAB_MAP_FILENAME)

    for path in candidate_paths:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                saved = json.load(f)
            teacher_to_pruned = {int(k): v for k, v in saved["teacher_id_to_pruned_id"].items()}
            pruned_to_teacher = {int(k): v for k, v in saved["pruned_id_to_teacher_id"].items()}
            tag_to_id = saved["tag_to_id"]
            total_vocab_size = saved["total_vocab_size"]
            V = saved.get("V", len(pruned_to_teacher))
            print(f"[OK] Existing Vocab Map loaded from: {path} (total_vocab_size={total_vocab_size})", flush=True)
            if drive_active:
                try:
                    drive_path = os.path.join(DRIVE_SAVE_DIR, VOCAB_MAP_FILENAME)
                    if not os.path.exists(drive_path):
                        shutil.copyfile(path, drive_path)
                        print(f"[OK] Synced vocab map to Google Drive: {drive_path}", flush=True)
                except Exception:
                    pass
            return teacher_to_pruned, pruned_to_teacher, tag_to_id, total_vocab_size, V

    print("[*] Pass 1: Scanning corpus to build clean Pruned Vocab Map...", flush=True)
    t0 = time.perf_counter()
    counter = Counter()
    n_seen = 0
    for tag, q, a in make_combined_stream():
        text = f"{q} {a}"
        tok_ids = tokenizer.encode(text, add_special_tokens=False)
        counter.update(tok_ids)
        n_seen += 1
        if n_seen >= 10000:
            break

    most_common = counter.most_common(vocab_cap - 5)
    teacher_to_pruned = {tok_id: rank + 2 for rank, (tok_id, _) in enumerate(most_common)}
    pruned_to_teacher = {v: k for k, v in teacher_to_pruned.items()}
    V = len(most_common)

    tag_to_id = {tag: V + 2 + i for i, tag in enumerate(ROUTE_TAGS)}
    total_vocab_size = V + 2 + len(ROUTE_TAGS)
    print(f"[OK] Vocab scanned from {n_seen} samples in {(time.perf_counter()-t0):.1f}s | V={V}, total={total_vocab_size}")

    save_obj = {
        "teacher_id_to_pruned_id": {str(k): v for k, v in teacher_to_pruned.items()},
        "pruned_id_to_teacher_id": {str(k): v for k, v in pruned_to_teacher.items()},
        "tag_to_id": tag_to_id,
        "pad_id": PAD_ID, "unk_id": UNK_ID,
        "total_vocab_size": total_vocab_size,
        "V": V,
    }
    local_path = os.path.join(OUTPUT_DIR, VOCAB_MAP_FILENAME)
    with open(local_path, "w", encoding="utf-8") as f:
        json.dump(save_obj, f, ensure_ascii=False, indent=2)

    if drive_active:
        try:
            drive_path = os.path.join(DRIVE_SAVE_DIR, VOCAB_MAP_FILENAME)
            shutil.copyfile(local_path, drive_path)
            print(f"[OK] Vocab map saved permanently to Drive: {drive_path}", flush=True)
        except Exception as e:
            print(f"[WARN] Drive save note: {e}", flush=True)

    return teacher_to_pruned, pruned_to_teacher, tag_to_id, total_vocab_size, V

# -----------------------------------------------------------------------------
# 6. Shift-1 Clean Distillation Streaming Dataset
# -----------------------------------------------------------------------------

class WRAIStreamDataset(IterableDataset):
    def __init__(self, tokenizer, teacher_to_pruned, tag_to_id, teacher_placeholder_id,
                 total_vocab_size, max_len=MAX_SEQ_LEN, shuffle_buffer=SHUFFLE_BUFFER_SIZE):
        self.tokenizer = tokenizer
        self.teacher_to_pruned = teacher_to_pruned
        self.tag_to_id = tag_to_id
        self.teacher_placeholder_id = teacher_placeholder_id
        self.total_vocab_size = total_vocab_size
        self.max_len = max_len
        self.shuffle_buffer = shuffle_buffer

    def _encode_one(self, tag, query, response):
        q_ids = self.tokenizer.encode(query, add_special_tokens=False)
        r_ids = self.tokenizer.encode(response, add_special_tokens=False)
        tag_id = self.tag_to_id[tag]

        q_pruned = [self.teacher_to_pruned.get(t, UNK_ID) for t in q_ids]
        r_pruned = [self.teacher_to_pruned.get(t, UNK_ID) for t in r_ids]

        full = q_pruned + [tag_id] + r_pruned
        full_teacher = q_ids + r_ids
        loss_mask = [0] * len(q_pruned) + [1] * (1 + len(r_pruned))
        distill_mask = [0] * len(q_pruned) + [0] + [1] * len(r_pruned)

        if len(full) > self.max_len + 1:
            full = full[: self.max_len + 1]
            loss_mask = loss_mask[: self.max_len + 1]
            distill_mask = distill_mask[: self.max_len + 1]
        else:
            pad_n = (self.max_len + 1) - len(full)
            full += [PAD_ID] * pad_n
            loss_mask += [0] * pad_n
            distill_mask += [0] * pad_n

        if len(full_teacher) > self.max_len + 1:
            full_teacher = full_teacher[: self.max_len + 1]
        else:
            pad_t = (self.max_len + 1) - len(full_teacher)
            full_teacher += [self.teacher_placeholder_id] * pad_t

        full = torch.tensor(full, dtype=torch.long)
        full_teacher = torch.tensor(full_teacher, dtype=torch.long)
        loss_mask = torch.tensor(loss_mask, dtype=torch.float)
        distill_mask = torch.tensor(distill_mask, dtype=torch.float)
        return {
            "input": full[:-1],
            "target": full[1:],
            "teacher_input": full_teacher[:-1],
            "loss_mask": loss_mask[1:],
            "distill_mask": distill_mask[1:],
        }

    def __iter__(self):
        buf = []
        for tag, q, a in make_combined_stream():
            buf.append(self._encode_one(tag, q, a))
            if len(buf) >= self.shuffle_buffer:
                random.shuffle(buf)
                while buf:
                    yield buf.pop()
        random.shuffle(buf)
        while buf:
            yield buf.pop()

# -----------------------------------------------------------------------------
# 7. Checkpoint Persistence & Auto-Resume
# -----------------------------------------------------------------------------

def save_checkpoint(model, optimizer, lr_scheduler, epoch, step, loss, prefix_name):
    raw_model = getattr(model, "_orig_mod", model)
    ckpt = {
        "model_state": raw_model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": lr_scheduler.state_dict() if lr_scheduler else None,
        "epoch": epoch,
        "step": step,
        "loss": float(loss),
        "arch": {
            "hidden_dim": HIDDEN_DIM,
            "ffn_dim": FFN_INTERMEDIATE_DIM,
            "num_layers": NUM_LAYERS,
            "wavelet_levels": WAVELET_LEVELS,
            "max_seq_len": MAX_SEQ_LEN,
            "vocab_size": raw_model.vocab_size
        }
    }
    local_pt = os.path.join(OUTPUT_DIR, f"{prefix_name}.pt")
    local_meta = os.path.join(OUTPUT_DIR, f"{prefix_name}_meta.json")
    torch.save(ckpt, local_pt)
    with open(local_meta, "w", encoding="utf-8") as f:
        json.dump({"epoch": epoch, "step": step, "loss": float(loss)}, f, indent=2)

    if drive_active:
        try:
            drive_pt = os.path.join(DRIVE_SAVE_DIR, f"{prefix_name}.pt")
            drive_meta = os.path.join(DRIVE_SAVE_DIR, f"{prefix_name}_meta.json")
            shutil.copyfile(local_pt, drive_pt)
            shutil.copyfile(local_meta, drive_meta)
            print(f"  [AUTO-SAVED TO DRIVE] -> {drive_pt} (Loss: {loss:.4f})", flush=True)
        except Exception as e:
            print(f"  [WARN] Drive checkpoint save note: {e}", flush=True)

def try_load_checkpoint(model, optimizer):
    raw_model = getattr(model, "_orig_mod", model)
    candidate_files = [
        "wrai_v14_7_transplant_latest.pt",
        "wrai_v14_7_transplant_best.pt"
    ]
    for pt_name in candidate_files:
        check_paths = []
        if drive_active:
            check_paths.append(os.path.join(DRIVE_SAVE_DIR, pt_name))
        check_paths.append(os.path.join(OUTPUT_DIR, pt_name))
        check_paths.append(pt_name)

        for check_path in check_paths:
            if os.path.exists(check_path):
                try:
                    print(f"[*] AUTO-RESUME DETECTED! Loading checkpoint from: {check_path}...", flush=True)
                    ckpt = torch.load(check_path, map_location="cpu")
                    state = ckpt.get("model_state", ckpt)

                    if "embed.weight" in state and state["embed.weight"].shape != raw_model.embed.weight.shape:
                        old_v = state["embed.weight"].size(0)
                        new_v = raw_model.embed.weight.size(0)
                        print(f"  [ADAPT] Adjusting embedding table size: Checkpoint={old_v} -> Model={new_v}", flush=True)
                        with torch.no_grad():
                            min_v = min(old_v, new_v)
                            raw_model.embed.weight.data[:min_v].copy_(state["embed.weight"][:min_v])
                        del state["embed.weight"]
                        if "output_proj.weight" in state:
                            del state["output_proj.weight"]

                    raw_model.load_state_dict(state, strict=False)
                    raw_model.output_proj.weight = raw_model.embed.weight

                    if optimizer and "optimizer_state" in ckpt and ckpt["optimizer_state"]:
                        try:
                            optimizer.load_state_dict(ckpt["optimizer_state"])
                        except Exception as e_opt:
                            print(f"  [NOTE] Optimizer re-initialized for momentum stability: {e_opt}", flush=True)

                    start_epoch = ckpt.get("epoch", 1)
                    start_step = ckpt.get("step", 0)
                    saved_loss = ckpt.get("loss", float("inf"))
                    sched_state = ckpt.get("scheduler_state", None)
                    print(f"[OK SUCCESS] Resumed from Epoch {start_epoch}, Step {start_step}, Loss {saved_loss:.4f}!\n", flush=True)
                    return start_epoch, start_step, saved_loss, sched_state, True
                except Exception as e:
                    print(f"[WARN] Failed to load checkpoint {check_path}: {e}", flush=True)
    return 1, 0, float("inf"), None, False

# -----------------------------------------------------------------------------
# 8. Main Clean Training Loop
# -----------------------------------------------------------------------------

def main():
    print("=================================================================")
    print("   WRAI v14.7 QWEN WEIGHT TRANSPLANT & RAPID ADAPTATION TRAINER   ")
    print("=================================================================")

    if HF_TOKEN:
        try:
            login(token=HF_TOKEN)
        except Exception:
            pass

    print(f"[*] Loading Teacher: {TEACHER_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_NAME, trust_remote_code=True)

    teacher = None
    try:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        teacher = AutoModelForCausalLM.from_pretrained(
            TEACHER_MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        print("[OK] Teacher Model Loaded in 4-bit NF4 (< 2.5 GB VRAM)!\n")
    except Exception as e:
        print(f"[!] 4-bit BitsAndBytes skipped ({e}). Loading Teacher in native FP16 (Auto Device Map)...")
        teacher = AutoModelForCausalLM.from_pretrained(
            TEACHER_MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        print("[OK] Teacher Model Loaded in native FP16 (< 6.0 GB VRAM)!\n")

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # 1. Vocab Map setup
    teacher_to_pruned, pruned_to_teacher, tag_to_id, total_vocab_size, V = get_or_build_vocab_map(tokenizer, VOCAB_CAP)
    teacher_placeholder_id = tokenizer.eos_token_id or tokenizer.pad_token_id or 0

    content_teacher_ids = torch.tensor(
        [pruned_to_teacher[r] for r in range(2, V + 2)], dtype=torch.long, device=DEVICE
    )

    # 2. Student Model Initialization & Weight Transplantation / Auto-Resume
    student = WRAIv14_7Model(vocab_size=total_vocab_size, max_seq_len=MAX_SEQ_LEN).to(DEVICE)
    total_params = sum(p.numel() for p in student.parameters())
    print(f"[OK] Student WRAI Initialized! Total Params: {total_params:,} (~{total_params/1e6:.1f}M, Vocab: {total_vocab_size})")

    optimizer = torch.optim.AdamW(student.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-5)

    # 3. Check for existing checkpoint in Google Drive / local output dir
    start_epoch, global_step, best_loss, sched_state, resumed = try_load_checkpoint(student, optimizer)

    if resumed:
        print("[OK] Resumed from existing checkpoint! Skipping initial Qwen transplant.", flush=True)
        if sched_state:
            try:
                lr_scheduler.load_state_dict(sched_state)
            except Exception:
                pass
    else:
        print("[*] No checkpoint found. Performing initial Qwen weight transplant...", flush=True)
        transplant_weights_from_qwen(student, pruned_to_teacher, TRANSPLANT_SOURCE_NAME)

    dataset = WRAIStreamDataset(tokenizer, teacher_to_pruned, tag_to_id, teacher_placeholder_id, total_vocab_size, max_len=MAX_SEQ_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE.type == 'cuda'))
    total_samples = N_ID + N_EN + N_TRANS + N_TOOL + N_PERSONA
    total_steps_est = total_samples // BATCH_SIZE

    print(f"\n[*] Total Dataset Corpus: {total_samples:,} samples | Steps/Epoch: ~{total_steps_est:,}")
    print(f"[*] Batch Configuration: {BATCH_SIZE} x {GRAD_ACCUM_STEPS} = {BATCH_SIZE*GRAD_ACCUM_STEPS} (Effective Batch)")
    print(f"[*] GPU VRAM: {torch.cuda.memory_allocated() / (1024**2):.1f} MB Allocated | {torch.cuda.memory_reserved() / (1024**2):.1f} MB Reserved\n")

    running_loss = best_loss if (resumed and best_loss < float("inf")) else None

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        student.train()
        total_epoch_loss = 0.0
        num_batches = 0
        optimizer.zero_grad()
        epoch_t0 = time.perf_counter()

        for step, batch in enumerate(loader):
            # Fast-forward if resuming in the middle of current epoch
            if resumed and epoch == start_epoch and step < (global_step % total_steps_est):
                continue

            inputs = batch["input"].to(DEVICE)
            targets = batch["target"].to(DEVICE)
            teacher_inputs = batch["teacher_input"].to(DEVICE)
            loss_mask = batch["loss_mask"].to(DEVICE)
            distill_mask = batch["distill_mask"].to(DEVICE)

            with torch.no_grad():
                teacher_outputs = teacher(teacher_inputs)
                teacher_content_logits = teacher_outputs.logits.index_select(-1, content_teacher_ids)
                teacher_probs = F.softmax(teacher_content_logits / DISTILL_TEMP, dim=-1)
                del teacher_outputs, teacher_content_logits

            with torch.amp.autocast('cuda', enabled=(DEVICE.type == 'cuda')):
                student_logits, _ = student(inputs)
                student_content_logits = student_logits[:, :, 2 : 2 + V]
                student_log_probs = F.log_softmax(student_content_logits / DISTILL_TEMP, dim=-1)

                # Distillation Loss (Shift-1 Alignment on natural Teacher logits)
                distill_per_tok = F.kl_div(
                    student_log_probs[:, 1:, :],
                    teacher_probs[:, :-1, :],
                    reduction="none"
                ).sum(-1)
                mask_d = distill_mask[:, 1:]
                denom_d = mask_d.sum().clamp(min=1.0)
                loss_distill = (distill_per_tok * mask_d).sum() / denom_d * (DISTILL_TEMP ** 2)

                # Hard Cross-Entropy Loss (on route tag & response tokens)
                ce_per_tok = F.cross_entropy(
                    student_logits.reshape(-1, total_vocab_size),
                    targets.reshape(-1),
                    reduction="none",
                    label_smoothing=0.05
                ).view_as(targets)
                denom_c = loss_mask.sum().clamp(min=1.0)
                loss_ce = (ce_per_tok * loss_mask).sum() / denom_c

                loss_total = (ALPHA_DISTILL * loss_distill) + (ALPHA_CE * loss_ce)
                loss_step = loss_total / GRAD_ACCUM_STEPS

            scaler.scale(loss_step).backward()

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1

            loss_val = loss_total.item()
            total_epoch_loss += loss_val
            num_batches += 1
            if running_loss is None:
                running_loss = loss_val
            else:
                running_loss = 0.95 * running_loss + 0.05 * loss_val

            # Progress Logging
            if (step + 1) % 50 == 0 or (step + 1) == total_steps_est:
                lr_curr = optimizer.param_groups[0]['lr']
                elapsed = time.perf_counter() - epoch_t0
                vram_used = torch.cuda.memory_allocated() / (1024**3)
                print(f"  [Epoch {epoch:02d}/{NUM_EPOCHS:02d} | Step {step+1:04d}/{total_steps_est:04d}] "
                      f"Loss: {running_loss:.4f} (Distill: {loss_distill.item():.4f}, CE: {loss_ce.item():.4f}) | "
                      f"LR: {lr_curr:.6f} | VRAM: {vram_used:.2f} GB ({elapsed:.1f}s)", flush=True)

            # Auto-Save every SAVE_EVERY_STEPS to Drive
            if (step + 1) % SAVE_EVERY_STEPS == 0:
                print(f"\n  [AUTO-SAVE] Saving Step {step+1} Checkpoint to Google Drive...", flush=True)
                save_checkpoint(student, optimizer, lr_scheduler, epoch, global_step, running_loss, "wrai_v14_7_transplant_latest")
                if running_loss < best_loss:
                    best_loss = running_loss
                    save_checkpoint(student, optimizer, lr_scheduler, epoch, global_step, best_loss, "wrai_v14_7_transplant_best")
                print("  [AUTO-SAVE COMPLETE] Checkpoint synced.\n", flush=True)

            if step >= total_steps_est:
                break

        avg_epoch_loss = total_epoch_loss / max(num_batches, 1)
        lr_scheduler.step()
        print(f"\n[EPOCH {epoch:02d} FINISHED] Average Epoch Loss: {avg_epoch_loss:.4f}\n", flush=True)
        save_checkpoint(student, optimizer, lr_scheduler, epoch, global_step, avg_epoch_loss, f"wrai_v14_7_transplant_epoch_{epoch}")

    print("\n=================================================================")
    print("   🎉 WRAI TRANSPLANT RAPID ADAPTATION TRAINING COMPLETE!        ")
    print("=================================================================")

if __name__ == "__main__":
    main()
