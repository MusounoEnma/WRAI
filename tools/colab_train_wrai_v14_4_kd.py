#!/usr/bin/env python3
"""
=================================================================================
  WRAI v14.4 KNOWLEDGE-DISTILLATION ENGINE (TEACHER: Qwen2.5-1.5B-Instruct)
  [FRESH RETRAIN VERSION WITH AGENT DOMAIN & PERFECTED EOS & SAFETY FIXES]
=================================================================================
Upload script ini ke Google Colab (Runtime: GPU T4/A100).

PERBAIKAN TERPADU ENTIRE PIPELINE:
1. [NEW DOMAIN] Menambahkan domain <AGENT> (hypervariance/function-calling-sharegpt).
2. [EXPLICIT EOS FIX] Sinyal stop EOS (<|im_end|>) dijamin masuk top-32k vocab & di-append ke tiap sampel training.
3. [KL-LOSS FLATTEN FIX] Tensor F.kl_div di-flatten ke (-1, 32000) agar loss stabil (< 5.0, tidak meledak ke 2000).
4. [MID-EPOCH AUTO-SAVE] Auto-save ke Google Drive setiap 1.500 step (~45 menit).
5. [FAST 90s VAL EVAL] Validation eval dibatasi max 2.000 sampel agar cepat (90 detik) & anti Colab hang/disconnect.
6. [HUMAN READABLE VOCAB] Map vocab menyimpan teks string token manusia untuk kemudahan debugging & C Native.
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
from collections import Counter

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
    os.system("pip install -q datasets huggingface_hub accelerate")
    from datasets import load_dataset

try:
    import sympy.printing
    from transformers import AutoModelForCausalLM, AutoTokenizer
except (ImportError, AttributeError):
    os.system("pip install -q --upgrade sympy transformers accelerate")
    import sympy.printing
    from transformers import AutoModelForCausalLM, AutoTokenizer

PRUNED_VOCAB_SIZE = 32000   # Ukuran vocab student setelah pruning (id 0 = OOV/PAD)
MAX_SEQ_LEN = 128

HIDDEN_DIM = 1024
NUM_LAYERS = 12
WAVELET_LEVELS = 4

TEACHER_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
KD_TEMPERATURE = 2.0
KD_ALPHA = 0.7  # Bobot KD-loss; (1-KD_ALPHA) untuk hard-label CE

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
    print(f"\n[OK] GOOGLE DRIVE SUCCESSFULLY MOUNTED! Models will auto-save & auto-resume from: {DRIVE_SAVE_DIR}\n", flush=True)
except Exception as e:
    print(f"\n[IMPORTANT NOTICE] Google Drive Auto-Mount skipped: {e}")
    print("[TIP] To enable Auto-Resume from Drive, click the Folder icon on Colab left sidebar -> Click 'Mount Drive' button!\n", flush=True)

VOCAB_MAP_FILENAME = "pruned_vocab_map_v14_4.json"


# ============================= MODEL ARCHITECTURE =============================

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
        assert hidden_dim % (2 ** num_levels) == 0
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


class WRAIv14_4(nn.Module):
    def __init__(self, vocab_size, hidden_dim=1024, num_layers=12, dropout=0.1, wavelet_levels=4):
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


class WarmupStableDecayLR:
    def __init__(self, optimizer, warmup_steps, stable_steps, decay_steps, min_lr=1e-5, base_lr_override=None):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.stable_steps = stable_steps
        self.decay_steps = decay_steps
        self.min_lr = min_lr
        if base_lr_override is not None:
            self.base_lrs = [base_lr_override for _ in optimizer.param_groups]
        else:
            self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_step = 0

    def state_dict(self):
        return {
            "warmup_steps": self.warmup_steps, "stable_steps": self.stable_steps,
            "decay_steps": self.decay_steps, "min_lr": self.min_lr,
            "base_lrs": self.base_lrs, "current_step": self.current_step,
        }

    def load_state_dict(self, sd):
        self.warmup_steps = sd["warmup_steps"]
        self.stable_steps = sd["stable_steps"]
        self.decay_steps = sd["decay_steps"]
        self.min_lr = sd["min_lr"]
        self.base_lrs = sd["base_lrs"]
        self.current_step = sd["current_step"]

    def step(self):
        self.current_step += 1
        for i, group in enumerate(self.optimizer.param_groups):
            base_lr = self.base_lrs[i]
            if self.current_step <= self.warmup_steps:
                lr = base_lr * (self.current_step / float(max(1, self.warmup_steps)))
            elif self.current_step <= self.warmup_steps + self.stable_steps:
                lr = base_lr
            else:
                progress = (self.current_step - self.warmup_steps - self.stable_steps) / float(max(1, self.decay_steps))
                progress = min(1.0, progress)
                lr = self.min_lr + (base_lr - self.min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))
            group['lr'] = lr


# ============================= DATA PIPELINE =============================

def clean_text_sanitizer(raw_text):
    if not isinstance(raw_text, str):
        return ""
    text = re.sub(r'[\r\n\t]+', ' ', raw_text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_clean_text_from_row(item, tag_prefix):
    text = ""
    if "text" in item and isinstance(item["text"], str):
        text = clean_text_sanitizer(item["text"])
    elif "content" in item and isinstance(item["content"], str):
        text = clean_text_sanitizer(item["content"])
    elif "conversations" in item and isinstance(item["conversations"], list):
        parts = []
        for turn in item["conversations"]:
            if isinstance(turn, dict):
                val = clean_text_sanitizer(str(turn.get("value", "") or turn.get("content", "")))
                if val:
                    parts.append(val)
        text = " ".join(parts)

    if not text:
        q = clean_text_sanitizer(str(item.get("question", "") or item.get("instruction", "") or item.get("prompt", "") or item.get("sql_prompt", "") or item.get("sql_context", "")))
        a = clean_text_sanitizer(str(item.get("answer", "") or item.get("output", "") or item.get("response", "") or item.get("sql", "")))
        if q and a:
            text = f"{q} {a}"

    if text and len(text) >= 30:
        return f"{tag_prefix} {text[:750]}"
    return None


def fetch_hf_dataset_sanitized(dataset_name, tag_prefix, max_samples=25000, config="default"):
    print(f"[*] Streaming ({dataset_name}, Tag: {tag_prefix}, Target: {max_samples:,})...", flush=True)
    samples = []
    try:
        ds = load_dataset(dataset_name, config if config != "default" else None, split="train", streaming=True)
        for item in ds:
            txt = extract_clean_text_from_row(item, tag_prefix)
            if txt:
                samples.append(txt)
                if len(samples) >= max_samples:
                    break
        print(f"[OK] Streamed {len(samples):,} samples from {dataset_name}", flush=True)
        return samples
    except Exception as e:
        print(f"[NOTE] Streaming fallback note for {dataset_name}: {e}", flush=True)

    try:
        offset = 0
        batch_size = 100
        encoded_name = urllib.parse.quote(dataset_name, safe='')
        while len(samples) < max_samples and offset < 200000:
            url = f"https://datasets-server.huggingface.co/rows?dataset={encoded_name}&config={config}&split=train&offset={offset}&length={batch_size}"
            req = urllib.request.Request(url, headers={"Authorization": f"Bearer {HF_TOKEN}"})
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                rows = data.get("rows", [])
                if not rows:
                    break
                for r in rows:
                    txt = extract_clean_text_from_row(r.get("row", {}), tag_prefix)
                    if txt:
                        samples.append(txt)
                offset += batch_size
        print(f"[OK] Streamed {len(samples):,} samples via API from {dataset_name}!", flush=True)
    except Exception as e:
        print(f"[WARNING] API fetch failed for {dataset_name}: {e}", flush=True)
    return samples


def build_v14_4_corpus():
    """Target ~460k sampel total melingkupi <EN>, <ID>, <PY>, <MATH>, dan <AGENT>."""
    all_texts = []
    all_texts.extend(fetch_hf_dataset_sanitized("HuggingFaceTB/smollm-corpus", "<EN>", 150000, "cosmopedia-v2"))
    all_texts.extend(fetch_hf_dataset_sanitized("HuggingFaceTB/smollm-corpus", "<EN>", 100000, "fineweb-edu-dedup"))
    all_texts.extend(fetch_hf_dataset_sanitized("FreedomIntelligence/alpaca-gpt4-indonesian", "<ID>", 45000))
    all_texts.extend(fetch_hf_dataset_sanitized("cahya/alpaca-id-cleaned", "<ID>", 40000))
    all_texts.extend(fetch_hf_dataset_sanitized("iamtarun/python_code_instructions_18k_alpaca", "<PY>", 18000))
    all_texts.extend(fetch_hf_dataset_sanitized("gretelai/synthetic_text_to_sql", "<PY>", 25000))
    all_texts.extend(fetch_hf_dataset_sanitized("microsoft/orca-math-word-problems-200k", "<MATH>", 40000))
    # Domain AI-Agent & Function Calling
    all_texts.extend(fetch_hf_dataset_sanitized("hypervariance/function-calling-sharegpt", "<AGENT>", 45000))
    random.seed(42)
    random.shuffle(all_texts)
    return all_texts


# ============================= VOCAB PRUNING =============================

def get_or_build_pruned_vocab(corpus_texts, teacher_tokenizer):
    candidate_paths = []
    if drive_active:
        candidate_paths.append(os.path.join(DRIVE_SAVE_DIR, VOCAB_MAP_FILENAME))
    candidate_paths.append(VOCAB_MAP_FILENAME)

    for path in candidate_paths:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                saved = json.load(f)
            pruned_id_to_teacher_id = torch.tensor(saved["pruned_id_to_teacher_id"], dtype=torch.long)
            teacher_id_to_pruned_id = {int(k): v for k, v in saved["teacher_id_to_pruned_id"].items()}
            print(f"[OK] Pruned vocab map LAMA dimuat ulang dari: {path} (vocab id konsisten).", flush=True)
            return teacher_id_to_pruned_id, pruned_id_to_teacher_id

    print("[*] Tidak ada pruned vocab map tersimpan -> membangun BARU dari 460k corpus...", flush=True)
    t0 = time.perf_counter()
    freq_counter = Counter()
    eos_tid = teacher_tokenizer.eos_token_id if teacher_tokenizer.eos_token_id is not None else 151645
    
    batch_size = 1000
    for i in range(0, len(corpus_texts), batch_size):
        batch = corpus_texts[i:i + batch_size]
        enc = teacher_tokenizer(batch, truncation=True, max_length=MAX_SEQ_LEN)
        for ids in enc["input_ids"]:
            # [FIX EOS] Append EOS token eksplisit di hitungan frekuensi
            ids_with_eos = list(ids) + [eos_tid]
            freq_counter.update(ids_with_eos)
    print(f"[OK] Frekuensi token dihitung dari {len(corpus_texts):,} sampel dalam {(time.perf_counter()-t0):.1f}s.", flush=True)

    most_common = [tid for tid, _ in freq_counter.most_common(PRUNED_VOCAB_SIZE - 5)]
    
    # Kunci Special Tokens (PAD=0, EOS=1) di posisi terdepan
    pad_tid = teacher_tokenizer.pad_token_id if teacher_tokenizer.pad_token_id is not None else eos_tid
    reserved_tids = [pad_tid, eos_tid]
    
    final_vocab_list = []
    for rtid in reserved_tids:
        if rtid not in final_vocab_list:
            final_vocab_list.append(rtid)
            
    for tid in most_common:
        if tid not in final_vocab_list and len(final_vocab_list) < PRUNED_VOCAB_SIZE:
            final_vocab_list.append(tid)

    teacher_id_to_pruned_id = {tid: i for i, tid in enumerate(final_vocab_list)}
    pruned_id_to_teacher_id_list = list(final_vocab_list)
    while len(pruned_id_to_teacher_id_list) < PRUNED_VOCAB_SIZE:
        pruned_id_to_teacher_id_list.append(pad_tid)
    pruned_id_to_teacher_id = torch.tensor(pruned_id_to_teacher_id_list, dtype=torch.long)

    # Human-readable token text mapping untuk debugging & C Native decoding
    pruned_id_to_token_text = {}
    for pid, tid in enumerate(pruned_id_to_teacher_id_list):
        try:
            tok_str = teacher_tokenizer.decode([tid])
            if not tok_str:
                tok_str = teacher_tokenizer.convert_ids_to_tokens(tid)
        except Exception:
            tok_str = f"<ID_{tid}>"
        pruned_id_to_token_text[str(pid)] = tok_str

    save_obj = {
        "pruned_id_to_teacher_id": pruned_id_to_teacher_id_list,
        "teacher_id_to_pruned_id": {str(k): v for k, v in teacher_id_to_pruned_id.items()},
        "pruned_id_to_token_text": pruned_id_to_token_text,
        "teacher_model": TEACHER_MODEL_NAME,
        "pruned_vocab_size": PRUNED_VOCAB_SIZE,
    }
    with open(VOCAB_MAP_FILENAME, "w", encoding="utf-8") as f:
        json.dump(save_obj, f, ensure_ascii=False, indent=2)
    if drive_active:
        try:
            shutil.copyfile(VOCAB_MAP_FILENAME, os.path.join(DRIVE_SAVE_DIR, VOCAB_MAP_FILENAME))
            print(f"[OK] Pruned vocab map disimpan permanen ke Drive: {DRIVE_SAVE_DIR}/{VOCAB_MAP_FILENAME}", flush=True)
        except Exception as e:
            print(f"[WARNING] Gagal simpan vocab map ke Drive: {e}", flush=True)

    return teacher_id_to_pruned_id, pruned_id_to_teacher_id


# ============================= CHECKPOINT I/O =============================

def export_model_v14_4(model, optimizer, wsd_scheduler, epoch, best_val_loss, training_phase, prefix_name="wrai_v14_4_best"):
    raw_model = getattr(model, "_orig_mod", model)
    checkpoint_state = {
        "model_state": raw_model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": wsd_scheduler.state_dict(),
        "epoch": epoch,
        "best_val_loss": float(best_val_loss),
        "training_phase": training_phase,
    }
    pt_path = f"{prefix_name}.pt"
    meta_path = f"{prefix_name}_meta.json"
    torch.save(checkpoint_state, pt_path)

    total_params = sum(p.numel() for p in raw_model.parameters())
    meta_json = {
        "version": "14.4_KD_Qwen2.5-1.5B-Instruct",
        "model_type": "WRAIv14_4", "vocab_size": PRUNED_VOCAB_SIZE,
        "hidden_dim": HIDDEN_DIM, "num_layers": NUM_LAYERS, "total_params": total_params,
        "training_epochs": epoch, "final_best_val_loss": float(best_val_loss),
        "training_phase": training_phase, "teacher_model": TEACHER_MODEL_NAME,
        "kd_alpha": KD_ALPHA, "kd_temperature": KD_TEMPERATURE,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_json, f, ensure_ascii=False)

    pt_size = os.path.getsize(pt_path) / (1024 * 1024)
    print(f"[EXPORTED] {pt_path} ({pt_size:.1f} MB) + {meta_path}", flush=True)
    if drive_active:
        try:
            for fname in [pt_path, meta_path]:
                shutil.copyfile(fname, os.path.join(DRIVE_SAVE_DIR, fname))
            if os.path.exists(VOCAB_MAP_FILENAME):
                shutil.copyfile(VOCAB_MAP_FILENAME, os.path.join(DRIVE_SAVE_DIR, VOCAB_MAP_FILENAME))
            print(f"[AUTO-SAVED TO DRIVE] -> {DRIVE_SAVE_DIR}/{prefix_name}.*", flush=True)
        except Exception as e:
            print(f"[WARNING] Drive save note: {e}", flush=True)


def try_resume_v14_4_checkpoint(model, optimizer):
    raw_model = getattr(model, "_orig_mod", model)
    candidate_files = [
        ("wrai_v14_4_latest.pt", "wrai_v14_4_latest_meta.json"),
        ("wrai_v14_4_best.pt", "wrai_v14_4_best_meta.json"),
    ]
    
    selected_pt = None
    max_found_epoch = -1
    
    for pt_name, _ in candidate_files:
        check_pt = os.path.join(DRIVE_SAVE_DIR, pt_name) if drive_active and os.path.exists(os.path.join(DRIVE_SAVE_DIR, pt_name)) else pt_name
        if os.path.exists(check_pt):
            try:
                ckpt = torch.load(check_pt, map_location="cpu")
                ep = ckpt.get("epoch", 1) if isinstance(ckpt, dict) else 1
                if ep > max_found_epoch:
                    max_found_epoch = ep
                    selected_pt = check_pt
            except Exception:
                pass

    if selected_pt and os.path.exists(selected_pt):
        try:
            print(f"[*] CONTINUOUS RESUME DETECTED! Loading state from: {selected_pt}...", flush=True)
            ckpt = torch.load(selected_pt, map_location="cpu")
            raw_model.load_state_dict(ckpt["model_state"], strict=True)
            start_epoch = ckpt.get("epoch", 1) + 1
            best_val_loss = ckpt.get("best_val_loss", float('inf'))
            optimizer_state = ckpt.get("optimizer_state", None)
            scheduler_state = ckpt.get("scheduler_state", None)
            training_phase = ckpt.get("training_phase", "stable")
            print(f"[OK 100% SUCCESS] RESUMED AT EPOCH {start_epoch} (Best Val Loss: {best_val_loss:.4f}, Phase: {training_phase})!", flush=True)
            return True, start_epoch, best_val_loss, optimizer_state, scheduler_state, training_phase
        except Exception as e:
            print(f"[NOTE] Checkpoint resume note: {e}.", flush=True)

    return False, 1, float('inf'), None, None, "stable"


# ============================= MAIN =============================

def main():
    print("=================================================================", flush=True)
    print("  WRAI v14.4 KNOWLEDGE-DISTILLATION ENGINE (QWEN -> WRAI)       ", flush=True)
    print("=================================================================\n", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)

    print("\n[*] Loading Teacher Model (Qwen2.5-1.5B-Instruct, frozen, fp16)...", flush=True)
    teacher_tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_NAME)
    if teacher_tokenizer.pad_token is None:
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
    teacher_model = AutoModelForCausalLM.from_pretrained(TEACHER_MODEL_NAME, torch_dtype=torch.float16).to(device)
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False
    print(f"[OK] Teacher {TEACHER_MODEL_NAME} loaded & frozen. Native vocab: {len(teacher_tokenizer):,}", flush=True)

    corpus_texts = build_v14_4_corpus()
    print(f"[*] TOTAL CORPUS PREPARED: {len(corpus_texts):,} sampel", flush=True)

    teacher_id_to_pruned_id, pruned_id_to_teacher_id = get_or_build_pruned_vocab(corpus_texts, teacher_tokenizer)
    pruned_id_to_teacher_id = pruned_id_to_teacher_id.to(device)

    model = WRAIv14_4(vocab_size=PRUNED_VOCAB_SIZE, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS,
                       dropout=0.1, wavelet_levels=WAVELET_LEVELS).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1)

    resumed, start_epoch, best_val_loss, optimizer_state, scheduler_state, training_phase = \
        try_resume_v14_4_checkpoint(model, optimizer)
    if resumed and optimizer_state is not None:
        try:
            optimizer.load_state_dict(optimizer_state)
            print("[OK] Optimizer state berhasil di-restore.", flush=True)
        except Exception as e:
            print(f"[WARNING] Gagal restore optimizer state: {e}", flush=True)

    print("[*] Tokenizing corpus dengan teacher tokenizer + pruning ke student id-space...", flush=True)
    
    eos_token_id = teacher_tokenizer.eos_token_id if teacher_tokenizer.eos_token_id is not None else 151645
    total_samples = len(corpus_texts)
    Teacher_all = torch.full((total_samples, MAX_SEQ_LEN), teacher_tokenizer.pad_token_id, dtype=torch.long)
    X_all = torch.zeros((total_samples, MAX_SEQ_LEN), dtype=torch.long)
    Y_all = torch.zeros((total_samples, MAX_SEQ_LEN), dtype=torch.long)

    batch_size_tok = 1000
    for i in range(0, total_samples, batch_size_tok):
        batch = corpus_texts[i:i + batch_size_tok]
        enc = teacher_tokenizer(batch, truncation=True, max_length=MAX_SEQ_LEN)
        for idx, ids in enumerate(enc["input_ids"]):
            row_idx = i + idx
            if not ids or ids[-1] != eos_token_id:
                ids = list(ids) + [eos_token_id]
            ids = ids[:MAX_SEQ_LEN + 1]
            pruned_ids = [teacher_id_to_pruned_id.get(tid, 0) for tid in ids]
            
            seq_len = min(len(ids), MAX_SEQ_LEN)
            Teacher_all[row_idx, :seq_len] = torch.tensor(ids[:seq_len], dtype=torch.long)
            
            x_len = min(len(pruned_ids[:-1]), MAX_SEQ_LEN)
            X_all[row_idx, :x_len] = torch.tensor(pruned_ids[:x_len], dtype=torch.long)
            
            y_len = min(len(pruned_ids[1:]), MAX_SEQ_LEN)
            Y_all[row_idx, :y_len] = torch.tensor(pruned_ids[1:1+y_len], dtype=torch.long)

    val_size = int(total_samples * 0.10)
    train_size = total_samples - val_size

    Teacher_train, Teacher_val = Teacher_all[:train_size], Teacher_all[train_size:]
    X_train, X_val = X_all[:train_size], X_all[train_size:]
    Y_train, Y_val = Y_all[:train_size], Y_all[train_size:]
    print(f"[*] Split: {train_size:,} Train | {val_size:,} Validation samples", flush=True)

    batch_size = 32
    grad_accum_steps = 16
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Teacher_train, X_train, Y_train), batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Teacher_val, X_val, Y_val), batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[*] WRAI v14.4 Student Initialized | Total Params (Tied): {total_params:,} (~{total_params/1e6:.1f}M)", flush=True)

    torch.backends.cudnn.benchmark = True
    if hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True

    max_epochs = 1000
    total_steps_per_epoch = len(train_loader)
    START_DECAY_NOW = False

    if resumed and scheduler_state is not None and training_phase == "decay" and not START_DECAY_NOW:
        wsd_scheduler = WarmupStableDecayLR(optimizer, warmup_steps=0, stable_steps=0, decay_steps=1)
        wsd_scheduler.load_state_dict(scheduler_state)
        print(f"[*] Fase DECAY dilanjutkan dari step {wsd_scheduler.current_step:,}.", flush=True)
    else:
        target_horizon_epochs = 15
        horizon_steps = target_horizon_epochs * total_steps_per_epoch
        warmup_steps = int(horizon_steps * 0.05)
        stable_steps = int(horizon_steps * 0.70)
        decay_steps = horizon_steps - warmup_steps - stable_steps
        wsd_scheduler = WarmupStableDecayLR(optimizer, warmup_steps=warmup_steps, stable_steps=stable_steps, decay_steps=decay_steps, min_lr=1e-5)
        if resumed and scheduler_state is not None:
            wsd_scheduler.load_state_dict(scheduler_state)
            print(f"[*] Fase STABLE dilanjutkan dari step {wsd_scheduler.current_step:,}.", flush=True)
        training_phase = "stable"

    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    print(f"[*] Starting WRAI v14.4 Training from Epoch {start_epoch} (Phase={training_phase})...", flush=True)
    t0 = time.perf_counter()
    patience, patience_counter = 30, 0

    for epoch in range(start_epoch, max_epochs + 1):
        model.train()
        train_loss, train_batches = 0.0, 0
        optimizer.zero_grad()
        epoch_t0 = time.perf_counter()

        for step, (bt, bx, by) in enumerate(train_loader):
            bt, bx, by = bt.to(device), bx.to(device), by.to(device)

            with torch.no_grad():
                teacher_logits_full = teacher_model(input_ids=bt).logits
                teacher_logits_pruned = teacher_logits_full[..., pruned_id_to_teacher_id]
                teacher_probs = F.softmax(teacher_logits_pruned.float() / KD_TEMPERATURE, dim=-1)

            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                student_logits = model(bx)
                student_log_probs = F.log_softmax(student_logits / KD_TEMPERATURE, dim=-1)
                
                # [FIX KL LOSS FLATTEN] Reshape ke 2D (-1, 32000) agar batchmean membagi per-token!
                kd_loss = F.kl_div(
                    student_log_probs.view(-1, PRUNED_VOCAB_SIZE),
                    teacher_probs.view(-1, PRUNED_VOCAB_SIZE),
                    reduction='batchmean'
                ) * (KD_TEMPERATURE ** 2)
                
                hard_loss = F.cross_entropy(student_logits.view(-1, PRUNED_VOCAB_SIZE), by.view(-1),
                                             ignore_index=0, label_smoothing=0.05)
                loss_hard_total = KD_ALPHA * kd_loss + (1 - KD_ALPHA) * hard_loss
                loss = loss_hard_total / grad_accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == total_steps_per_epoch:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            wsd_scheduler.step()
            train_loss += loss_hard_total.item()
            train_batches += 1

            if (step + 1) % 100 == 0 or (step + 1) == total_steps_per_epoch:
                current_avg_loss = train_loss / train_batches
                elapsed_epoch = time.perf_counter() - epoch_t0
                print(f"  [Epoch {epoch:2d} | Step {step+1:5d}/{total_steps_per_epoch}] Loss: {current_avg_loss:.4f} "
                      f"(KD:{kd_loss.item():.4f} Hard:{hard_loss.item():.4f}) ({elapsed_epoch:.1f}s)", flush=True)

            # [MID-EPOCH AUTO-SAVE TO DRIVE EVERY 1,500 STEPS (~45 MIN)]
            if (step + 1) % 1500 == 0:
                current_avg_loss = train_loss / train_batches
                print(f"\n  [MID-EPOCH SAVE] Auto-saving step {step+1}/{total_steps_per_epoch} checkpoint to Drive...", flush=True)
                export_model_v14_4(model, optimizer, wsd_scheduler, epoch, current_avg_loss, training_phase, "wrai_v14_4_latest")
                print("  [MID-EPOCH SAVE COMPLETE] Safe to disconnect or continue!\n", flush=True)

        avg_train_loss = train_loss / max(1, train_batches)

        # [FAST 90s VALIDATION EVALUATION]
        if epoch % 1 == 0:
            model.eval()
            val_loss, val_batches, correct, total = 0.0, 0, 0, 0
            MAX_VAL_SAMPLES = 2000
            current_val_count = 0
            
            with torch.no_grad():
                for bt_val, bx_val, by_val in val_loader:
                    bt_val, bx_val, by_val = bt_val.to(device), bx_val.to(device), by_val.to(device)
                    teacher_logits_full = teacher_model(input_ids=bt_val).logits
                    teacher_logits_pruned = teacher_logits_full[..., pruned_id_to_teacher_id]
                    teacher_probs_v = F.softmax(teacher_logits_pruned.float() / KD_TEMPERATURE, dim=-1)
                    with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                        v_logits = model(bx_val)
                        v_log_probs = F.log_softmax(v_logits / KD_TEMPERATURE, dim=-1)
                        v_kd = F.kl_div(
                            v_log_probs.view(-1, PRUNED_VOCAB_SIZE),
                            teacher_probs_v.view(-1, PRUNED_VOCAB_SIZE),
                            reduction='batchmean'
                        ) * (KD_TEMPERATURE ** 2)
                        v_hard = F.cross_entropy(v_logits.view(-1, PRUNED_VOCAB_SIZE), by_val.view(-1), ignore_index=0)
                        v_loss = KD_ALPHA * v_kd + (1 - KD_ALPHA) * v_hard
                    val_loss += v_loss.item()
                    val_batches += 1
                    preds = torch.argmax(v_logits, dim=-1)
                    mask = (by_val != 0)
                    correct += ((preds == by_val) & mask).sum().item()
                    total += mask.sum().item()
                    
                    current_val_count += bx_val.size(0)
                    if current_val_count >= MAX_VAL_SAMPLES:
                        break

            avg_val_loss = val_loss / max(1, val_batches)
            val_acc = (correct / max(1, total)) * 100.0
            lr = optimizer.param_groups[0]['lr']
            print(f"\n  ==> Epoch {epoch:4d}/{max_epochs} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | Acc: {val_acc:.2f}% | LR: {lr:.6f}\n", flush=True)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                export_model_v14_4(model, optimizer, wsd_scheduler, epoch, best_val_loss, training_phase, "wrai_v14_4_best")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n[*] EARLY STOPPING at Epoch {epoch}! Best Val Loss: {best_val_loss:.4f}", flush=True)
                    break

            export_model_v14_4(model, optimizer, wsd_scheduler, epoch, avg_val_loss, training_phase, "wrai_v14_4_latest")

        if device.type == 'cuda':
            torch.cuda.empty_cache()

    t1 = time.perf_counter()
    print(f"\n[OK] WRAI v14.4 TRAINING COMPLETE in {(t1 - t0):.1f}s!", flush=True)
    export_model_v14_4(model, optimizer, wsd_scheduler, epoch, best_val_loss, training_phase, "wrai_v14_4_final")
    print(f"\n[ALL DONE] WRAI v14.4 model saved to Google Drive & Local!", flush=True)


if __name__ == "__main__":
    main()
