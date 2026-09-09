#!/usr/bin/env python3
"""
=================================================================================
  WRAI v15.0 LOGIT DISTILLATION ENGINE (TEACHER-STUDENT KNOWLEDGE TRANSFER)
=================================================================================
Upload script ini ke Google Colab (Runtime: GPU T4/A100).

TUJUAN v15.0:
1. Menyuntikkan triliunan pengetahuan raksasa dari Teacher LLM (SmolLM2-360M / Qwen2.5-0.5B)
   ke dalam Student Model WRAI v14.3.1 (108.4M Parameters).
2. Menggunakan KL-Divergence Loss + Softmax Temperature Scaling (T=2.0).
3. WRAI v14.3.1 mewarisi daya nalar, trivia dunia, dan kecerdasan bahasa tingkat tinggi
   secara permanen ke dalam file bobot `wrai_v15_distilled_best.pt`.
"""

import copy
import json
import math
import os
import random
import re
import shutil
import time
import urllib.request
import urllib.parse

try:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
except ImportError:
    os.system("pip install -q torch torchvision torchaudio numpy")
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim

try:
    from datasets import load_dataset
except ImportError:
    os.system("pip install -q datasets huggingface_hub tokenizers transformers accelerate")
    from datasets import load_dataset

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    os.system("pip install -q transformers")
    from transformers import AutoModelForCausalLM, AutoTokenizer

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

HF_TOKEN = ""
os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN

DRIVE_MOUNT_PATH = "/content/drive"
DRIVE_SAVE_DIR = "/content/drive/MyDrive/WRAI_Models"
drive_active = False

try:
    from google.colab import drive
    drive.mount(DRIVE_MOUNT_PATH, force_remount=False)
    os.makedirs(DRIVE_SAVE_DIR, exist_ok=True)
    drive_active = True
    print(f"\n[OK] GOOGLE DRIVE SUCCESSFULLY MOUNTED! Distilled Models will save to: {DRIVE_SAVE_DIR}\n", flush=True)
except Exception as e:
    print(f"\n[IMPORTANT NOTICE] Google Drive Auto-Mount skipped: {e}")

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
        with torch.amp.autocast('cuda', enabled=False):
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
        nn.init.normal_(self.embedding.weight, mean=0.0, std=hidden_dim ** -0.5)
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

class DistillationLoss(nn.Module):
    """
    Combined Cross-Entropy + KL-Divergence Loss for Logit Distillation.
    """
    def __init__(self, temperature=2.0, alpha=0.4, ignore_index=0):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, label_smoothing=0.05)
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits, teacher_logits, targets):
        loss_ce = self.ce_loss(student_logits.view(-1, student_logits.size(-1)), targets.view(-1))
        
        # Softmax with temperature scaling
        p_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        p_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        loss_kl = self.kl_loss(p_student, p_teacher) * (self.temperature ** 2)
        total_loss = self.alpha * loss_ce + (1.0 - self.alpha) * loss_kl
        return total_loss, loss_ce, loss_kl

def clean_text_sanitizer(raw_text):
    if not isinstance(raw_text, str): return ""
    text = re.sub(r'[\r\n\t]+', ' ', raw_text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def fetch_hf_dataset_sanitized(dataset_name, tag_prefix, max_samples=15000, config="default"):
    print(f"[*] Streaming Distillation Corpus ({dataset_name}, Tag: {tag_prefix}, Target: {max_samples:,})...", flush=True)
    samples = []
    try:
        ds = load_dataset(dataset_name, config if config != "default" else None, split="train", streaming=True)
        for item in ds:
            txt = clean_text_sanitizer(item.get("text", "") or item.get("content", ""))
            if txt and len(txt) >= 30:
                samples.append(f"{tag_prefix} {txt[:750]}")
                if len(samples) >= max_samples: break
        print(f"[OK] Streamed {len(samples):,} samples from {dataset_name}!", flush=True)
        return samples
    except Exception as e:
        print(f"[NOTE] Streaming note: {e}", flush=True)
        return []

def main():
    print("=================================================================", flush=True)
    print("  WRAI v15.0 LOGIT DISTILLATION ENGINE (QWEN/SMOLLM -> WRAI)    ", flush=True)
    print("=================================================================\n", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Distillation Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)

    # 1. Load Teacher Model (HuggingFaceTB/SmolLM2-360M-Instruct or Qwen2.5-0.5B-Instruct)
    TEACHER_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"
    print(f"[*] Loading Pretrained Teacher Model ({TEACHER_NAME})...", flush=True)
    teacher_tokenizer = AutoTokenizer.from_pretrained(TEACHER_NAME)
    teacher_model = AutoModelForCausalLM.from_pretrained(TEACHER_NAME, torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32).to(device)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False
    print(f"[OK 100% SUCCESS] Teacher Model Loaded & Frozen in GPU Memory!", flush=True)

    # 2. Load Student Model (WRAI v14.3.1 - 108.4M)
    print("\n[*] Initializing Student Model (WRAI v14.3.1 - 108.4M)...", flush=True)
    tok_path_drive = os.path.join(DRIVE_SAVE_DIR, TOKENIZER_FILENAME)
    tok_path = tok_path_drive if os.path.exists(tok_path_drive) else TOKENIZER_FILENAME
    
    if not os.path.exists(tok_path):
        print(f"[ERROR] Student Tokenizer file not found at {tok_path}!", flush=True)
        return
        
    student_tokenizer = Tokenizer.from_file(tok_path)
    vocab_size = len(student_tokenizer.get_vocab())
    student_model = WRAIv14_3(vocab_size=vocab_size, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, wavelet_levels=WAVELET_LEVELS).to(device)

    # Load pretrained student weights from Drive
    candidate_pt = [
        os.path.join(DRIVE_SAVE_DIR, "wrai_v14_3_latest.pt"),
        os.path.join(DRIVE_SAVE_DIR, "wrai_v14_3_best.pt")
    ]
    target_pt = None
    for p in candidate_pt:
        if os.path.exists(p):
            target_pt = p
            break
            
    if target_pt:
        print(f"[*] Loading Student Pretrained Weights from: {target_pt}...", flush=True)
        ckpt = torch.load(target_pt, map_location=device)
        student_model.load_state_dict(ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt)
        print(f"[OK 100% SUCCESS] Student Model Initialized with Pretrained Weights!", flush=True)

    # 3. Data Preparation
    corpus = []
    corpus.extend(fetch_hf_dataset_sanitized("HuggingFaceTB/smollm-corpus", "<EN>", 15000, "cosmopedia-v2"))
    corpus.extend(fetch_hf_dataset_sanitized("FreedomIntelligence/alpaca-gpt4-indonesian", "<ID>", 10000))
    corpus.extend(fetch_hf_dataset_sanitized("iamtarun/python_code_instructions_18k_alpaca", "<PY>", 5000))
    random.seed(42)
    random.shuffle(corpus)

    padded_x, padded_y = [], []
    for txt in corpus:
        full_text = f"{txt} <EOS>"
        tokens = student_tokenizer.encode(full_text).ids[:MAX_SEQ_LEN + 1]
        x_seq = tokens[:-1] + [0] * (MAX_SEQ_LEN - len(tokens[:-1]))
        y_seq = tokens[1:]  + [0] * (MAX_SEQ_LEN - len(tokens[1:]))
        padded_x.append(x_seq)
        padded_y.append(y_seq)

    train_tensor_x = torch.tensor(padded_x, dtype=torch.long)
    train_tensor_y = torch.tensor(padded_y, dtype=torch.long)
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_tensor_x, train_tensor_y), batch_size=32, shuffle=True)

    # 4. Distillation Training Loop
    optimizer = optim.AdamW(student_model.parameters(), lr=3e-4, weight_decay=0.01)
    distill_loss_fn = DistillationLoss(temperature=2.0, alpha=0.3, ignore_index=0)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    distill_epochs = 10
    print(f"\n[*] STARTING LOGIT DISTILLATION (Horizon: {distill_epochs} Epochs)...", flush=True)
    t0 = time.perf_counter()

    for epoch in range(1, distill_epochs + 1):
        student_model.train()
        total_loss, total_ce, total_kl = 0.0, 0.0, 0.0
        epoch_t0 = time.perf_counter()

        for step, (bx, by) in enumerate(train_loader):
            bx, by = bx.to(device), by.to(device)
            
            # Generate Teacher Logits in float16
            with torch.no_grad():
                teacher_outputs = teacher_model(bx)
                teacher_logits = teacher_outputs.logits
                # Resize teacher logits if needed to match student vocab_size
                if teacher_logits.size(-1) != vocab_size:
                    teacher_logits = teacher_logits[:, :, :vocab_size]

            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                student_logits = student_model(bx)
                loss, loss_ce, loss_kl = distill_loss_fn(student_logits, teacher_logits.float(), by)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_ce += loss_ce.item()
            total_kl += loss_kl.item()

            if (step + 1) % 50 == 0 or (step + 1) == len(train_loader):
                elapsed = time.perf_counter() - epoch_t0
                print(f"  [Distill Epoch {epoch:2d} | Step {step+1:4d}/{len(train_loader)}] Total Loss: {total_loss/(step+1):.4f} (CE: {total_ce/(step+1):.4f} | KL: {total_kl/(step+1):.4f}) [{elapsed:.1f}s]", flush=True)

        # Auto-Save Distilled Checkpoint
        save_path = os.path.join(DRIVE_SAVE_DIR, "wrai_v15_distilled_best.pt") if drive_active else "wrai_v15_distilled_best.pt"
        torch.save({
            "model_state": student_model.state_dict(),
            "epoch": epoch,
            "distill_loss": total_loss / len(train_loader)
        }, save_path)
        print(f"\n  ==> [DISTILLED MODEL AUTO-SAVED TO DRIVE] -> {save_path}\n", flush=True)

    t1 = time.perf_counter()
    print(f"\n[OK 100% SUCCESS] WRAI v15.0 LOGIT DISTILLATION COMPLETE in {(t1-t0)/60:.1f} minutes!", flush=True)

if __name__ == "__main__":
    main()
