#!/usr/bin/env python3
"""
=================================================================================
  WRAI v14.3.1 RESGRU + HAAR-DWT + TIED-EMBEDDING ENGINE (STABLE EMBED SCALING)
=================================================================================
Upload script ini ke Google Colab (Runtime: GPU T4/A100).

PERUBAHAN ARSITEKTUR v14.3.1 (vs v14.2):
1. [WEIGHT TYING & EMBED SCALING] fc.weight = embedding.weight -> 141.1M -> ~108.4M parameter.
   Menghapus 32.7M parameter redundan + GPT-2 style embed scaling (std ~ 1/sqrt(hidden_dim)).
2. [ResGRU] GRU 12-layer ditumpuk MANUAL dengan residual connection + Pre-LayerNorm di tiap layer.
3. [Haar DWT Spectral] ZeroGEMMSpectralConvLayer diganti HaarDWTSpectralLayer 4-Level O(H).
4. LR peak diturunkan 0.003 -> 1e-3.
5. Tokenizer persistence (tokenizer_v14.json) + optimizer/scheduler state resume.
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
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, AddedToken
except ImportError:
    os.system("pip install -q tokenizers")
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, AddedToken

MAX_VOCAB_SIZE = 32000
MAX_SEQ_LEN = 128

HIDDEN_DIM = 1024
NUM_LAYERS = 12
WAVELET_LEVELS = 4  # hidden_dim harus habis dibagi 2**WAVELET_LEVELS (1024 / 16 = 64, OK)

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
    """
    Multi-level Haar Discrete Wavelet Transform di axis channel (hidden_dim).
    Pengganti FFT-based ZeroGEMMSpectralConvLayer: O(H) bukan O(H log H),
    dan memberi dekomposisi multi-resolusi SUNGGUHAN (approx + detail di
    beberapa skala), bukan filter frekuensi resolusi tunggal.

    JUJUR: ini channel-mixer (mencampur antar-dimensi hidden per posisi
    token secara independen), BUKAN token-mixer. Tidak menggantikan peran
    attention/context-mixing antar-posisi -- itu tetap tugas ResGRU.
    """
    def __init__(self, hidden_dim, num_levels=4):
        super().__init__()
        assert hidden_dim % (2 ** num_levels) == 0, \
            f"hidden_dim ({hidden_dim}) harus habis dibagi 2**num_levels ({2**num_levels})"
        self.hidden_dim = hidden_dim
        self.num_levels = num_levels
        self.detail_gains = nn.ParameterList([
            nn.Parameter(torch.ones(hidden_dim // (2 ** (l + 1))))
            for l in range(num_levels)
        ])
        self.approx_gain = nn.Parameter(torch.ones(hidden_dim // (2 ** num_levels)))
        # Gate GLU-style per-channel (depthwise, O(H) -- bukan linear penuh O(H^2))
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
    """Satu layer GRU dengan Pre-LayerNorm + residual (skip) connection."""
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
        self.num_layers = num_layers

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
        # [WEIGHT TYING] fc.weight berbagi memori dengan embedding.weight.
        # fc.bias TETAP terpisah (standar, bias tidak di-tie).
        self.fc.weight = self.embedding.weight

        # [FIX v14.3.1] nn.Embedding default init std=1 -> kalau di-tie langsung
        # ke fc, logit output meledak (std(logit) ~ sqrt(hidden_dim) ~ 32).
        # Recipe standar tied-embedding (GPT-2 style): init embedding KECIL
        # (std ~ 1/sqrt(hidden_dim)), lalu SKALAKAN NAIK lagi di sisi input
        # (x sqrt(hidden_dim)) supaya representasi yang masuk ke GRU tetap
        # skala normal, sementara matriks mentah yang dipakai fc tetap kecil.
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
    print(f"[*] Real-Time HTTP Streaming Dataset ({dataset_name}, Tag: {tag_prefix}, Target: {max_samples:,})...", flush=True)
    samples = []
    try:
        ds = load_dataset(dataset_name, config if config != "default" else None, split="train", streaming=True)
        for item in ds:
            txt = extract_clean_text_from_row(item, tag_prefix)
            if txt:
                samples.append(txt)
                if len(samples) >= max_samples:
                    break
        print(f"[OK] Streamed {len(samples):,} clean samples from {dataset_name} (Disk Used: 0 MB)!", flush=True)
        return samples
    except Exception as e:
        print(f"[NOTE] Streaming fallback note for {dataset_name}: {e}", flush=True)

    try:
        offset = 0
        batch_size = 100
        encoded_name = urllib.parse.quote(dataset_name, safe='')
        while len(samples) < max_samples and offset < 40000:
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


def build_v14_sanitized_corpus():
    all_texts = []
    all_texts.extend(fetch_hf_dataset_sanitized("HuggingFaceTB/smollm-corpus", "<EN>", 20000, "cosmopedia-v2"))
    all_texts.extend(fetch_hf_dataset_sanitized("HuggingFaceTB/smollm-corpus", "<EN>", 15000, "fineweb-edu-dedup"))
    all_texts.extend(fetch_hf_dataset_sanitized("FreedomIntelligence/alpaca-gpt4-indonesian", "<ID>", 15000))
    all_texts.extend(fetch_hf_dataset_sanitized("cahya/alpaca-id-cleaned", "<ID>", 10000))
    all_texts.extend(fetch_hf_dataset_sanitized("iamtarun/python_code_instructions_18k_alpaca", "<PY>", 8000))
    all_texts.extend(fetch_hf_dataset_sanitized("gretelai/synthetic_text_to_sql", "<PY>", 4000))
    all_texts.extend(fetch_hf_dataset_sanitized("microsoft/orca-math-word-problems-200k", "<MATH>", 8000))
    random.seed(42)
    random.shuffle(all_texts)
    return all_texts


def get_or_create_tokenizer(corpus_texts):
    SPECIAL_VOCAB = ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<ID>", "<EN>", "<PY>", "<MATH>"]
    candidate_paths = []
    if drive_active:
        candidate_paths.append(os.path.join(DRIVE_SAVE_DIR, TOKENIZER_FILENAME))
    candidate_paths.append(TOKENIZER_FILENAME)

    for path in candidate_paths:
        if os.path.exists(path):
            tokenizer_fast = Tokenizer.from_file(path)
            print(f"[OK] Tokenizer LAMA dimuat ulang dari: {path} (vocab id identik).", flush=True)
            return tokenizer_fast, False

    print("[*] Tidak ada tokenizer tersimpan -> melatih BARU (training dari epoch 1).", flush=True)
    t_tok0 = time.perf_counter()
    tokenizer_fast = Tokenizer(models.BPE(unk_token="<UNK>"))
    tokenizer_fast.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(vocab_size=MAX_VOCAB_SIZE, special_tokens=SPECIAL_VOCAB)
    tokenizer_fast.add_special_tokens([AddedToken(tok, single_word=True, lstrip=False, rstrip=False) for tok in SPECIAL_VOCAB])

    temp_corpus_file = "temp_v14_sanitized_corpus.txt"
    with open(temp_corpus_file, "w", encoding="utf-8") as f_tmp:
        for txt in corpus_texts:
            f_tmp.write(f"{txt}\n")
    tokenizer_fast.train(files=[temp_corpus_file], trainer=trainer)
    t_tok1 = time.perf_counter()
    print(f"[OK] Tokenizer BARU selesai dalam {(t_tok1 - t_tok0):.2f}s!", flush=True)

    tokenizer_fast.save(TOKENIZER_FILENAME)
    if drive_active:
        try:
            shutil.copyfile(TOKENIZER_FILENAME, os.path.join(DRIVE_SAVE_DIR, TOKENIZER_FILENAME))
            print(f"[OK] Tokenizer disimpan permanen ke Drive: {DRIVE_SAVE_DIR}/{TOKENIZER_FILENAME}", flush=True)
        except Exception as e:
            print(f"[WARNING] Gagal simpan tokenizer ke Drive: {e}", flush=True)
    return tokenizer_fast, True


def export_model_v14_3(model, optimizer, wsd_scheduler, vocab_size, epoch, best_val_loss,
                        training_phase, prefix_name="wrai_v14_3_best"):
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
        "version": "14.3.1_ResGRU_HaarDWT_TiedEmbedding",
        "model_type": "WRAIv14_3",
        "vocab_size": vocab_size, "hidden_dim": HIDDEN_DIM, "num_layers": NUM_LAYERS,
        "total_params": total_params,
        "training_epochs": epoch, "final_best_val_loss": float(best_val_loss),
        "training_phase": training_phase,
        "tokenizer_file": TOKENIZER_FILENAME,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_json, f, ensure_ascii=False)

    pt_size = os.path.getsize(pt_path) / (1024 * 1024)
    print(f"[EXPORTED] {pt_path} ({pt_size:.1f} MB) + {meta_path}", flush=True)

    if drive_active:
        try:
            for fname in [pt_path, meta_path]:
                shutil.copyfile(fname, os.path.join(DRIVE_SAVE_DIR, fname))
            # Guarantee tokenizer_v14.json is always present on Google Drive
            if os.path.exists(TOKENIZER_FILENAME):
                shutil.copyfile(TOKENIZER_FILENAME, os.path.join(DRIVE_SAVE_DIR, TOKENIZER_FILENAME))
            print(f"[AUTO-SAVED TO DRIVE] -> {DRIVE_SAVE_DIR}/{prefix_name}.*", flush=True)
        except Exception as e:
            print(f"[WARNING] Drive save note: {e}", flush=True)


def try_resume_v14_3_checkpoint(model, optimizer):
    raw_model = getattr(model, "_orig_mod", model)
    candidate_files = [
        ("wrai_v14_3_latest.pt", "wrai_v14_3_latest_meta.json"),
        ("wrai_v14_3_best.pt", "wrai_v14_3_best_meta.json"),
    ]
    
    selected_pt, selected_meta = None, None
    max_found_epoch = -1
    
    for pt_name, meta_name in candidate_files:
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


def main():
    print("=================================================================", flush=True)
    print("  WRAI v14.3.1 RESGRU + HAAR-DWT + TIED-EMBEDDING ENGINE         ", flush=True)
    print("=================================================================\n", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)

    corpus_texts = build_v14_sanitized_corpus()
    print(f"[*] TOTAL SANITIZED PRE-TRAINING CORPUS PREPARED: {len(corpus_texts):,} clean text samples", flush=True)

    model = WRAIv14_3(vocab_size=MAX_VOCAB_SIZE, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS,
                       dropout=0.1, wavelet_levels=WAVELET_LEVELS).to(device)

    # LR PEAK DITURUNKAN: 0.003 -> 1e-3 (RNN dalam lebih stabil di LR lebih rendah)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1)

    resumed, start_epoch, best_val_loss, optimizer_state, scheduler_state, training_phase = \
        try_resume_v14_3_checkpoint(model, optimizer)

    if resumed and optimizer_state is not None:
        try:
            optimizer.load_state_dict(optimizer_state)
            print("[OK] Optimizer state berhasil di-restore.", flush=True)
        except Exception as e:
            print(f"[WARNING] Gagal restore optimizer state: {e}", flush=True)

    print("\n[*] Initializing Tokenizer (Persistent Vocabulary)...", flush=True)
    tokenizer_fast, tokenizer_is_new = get_or_create_tokenizer(corpus_texts)
    vocab_size = len(tokenizer_fast.get_vocab())

    def bpe_encode(text):
        return tokenizer_fast.encode(text).ids

    padded_x, padded_y = [], []
    for txt in corpus_texts:
        full_text = f"{txt} <EOS>"
        tokens = bpe_encode(full_text)[:MAX_SEQ_LEN + 1]
        x_seq = tokens[:-1] + [0] * (MAX_SEQ_LEN - len(tokens[:-1]))
        y_seq = tokens[1:]  + [0] * (MAX_SEQ_LEN - len(tokens[1:]))
        padded_x.append(x_seq)
        padded_y.append(y_seq)

    total_samples = len(padded_x)
    val_size = int(total_samples * 0.10)
    train_size = total_samples - val_size

    X_train = torch.tensor(padded_x[:train_size], dtype=torch.long)
    Y_train = torch.tensor(padded_y[:train_size], dtype=torch.long)
    X_val   = torch.tensor(padded_x[train_size:], dtype=torch.long)
    Y_val   = torch.tensor(padded_y[train_size:], dtype=torch.long)
    print(f"[*] Split: {train_size:,} Train | {val_size:,} Validation samples", flush=True)

    batch_size = 64
    grad_accum_steps = 8
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_val, Y_val), batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[*] WRAI v14.3.1 Initialized | Total Params (Tied): {total_params:,} (~{total_params/1e6:.1f}M)", flush=True)

    torch.backends.cudnn.benchmark = True
    if hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = True

    max_epochs = 1000
    total_steps_per_epoch = len(train_loader)

    START_DECAY_NOW = False  # ubah True SEKALI kalau memang mau memicu fase decay secara sengaja

    if resumed and scheduler_state is not None and training_phase == "decay" and not START_DECAY_NOW:
        wsd_scheduler = WarmupStableDecayLR(optimizer, warmup_steps=0, stable_steps=0, decay_steps=1)
        wsd_scheduler.load_state_dict(scheduler_state)
        print(f"[*] Fase DECAY dilanjutkan dari step {wsd_scheduler.current_step:,}.", flush=True)
    elif resumed and START_DECAY_NOW and training_phase != "decay":
        target_horizon_epochs = 10
        decay_steps = target_horizon_epochs * total_steps_per_epoch
        wsd_scheduler = WarmupStableDecayLR(optimizer, warmup_steps=0, stable_steps=0, decay_steps=decay_steps,
                                             min_lr=1e-5, base_lr_override=3e-4)
        training_phase = "decay"
        print(f"[*] FASE DECAY DIMULAI: LR Annealing 3e-4 -> 1e-5 over {target_horizon_epochs} epochs.", flush=True)
    else:
        # Horizon dipendekkan dari 50 -> 25 epoch (lebih realistis untuk kuota T4 gratis).
        # warmup 5% (~1.25 epoch) | stable 70% (~17.5 epoch) | decay 25% (~6.25 epoch)
        target_horizon_epochs = 25
        horizon_steps = target_horizon_epochs * total_steps_per_epoch
        warmup_steps = int(horizon_steps * 0.05)
        stable_steps = int(horizon_steps * 0.70)
        decay_steps = horizon_steps - warmup_steps - stable_steps
        wsd_scheduler = WarmupStableDecayLR(optimizer, warmup_steps=warmup_steps, stable_steps=stable_steps, decay_steps=decay_steps, min_lr=1e-5)
        if resumed and scheduler_state is not None:
            wsd_scheduler.load_state_dict(scheduler_state)
            print(f"[*] Fase STABLE dilanjutkan dari step {wsd_scheduler.current_step:,}.", flush=True)
        training_phase = "stable"

    criterion_ce = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.05)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    print(f"[*] Starting WRAI v14.3.1 Training from Epoch {start_epoch} (Phase={training_phase})...", flush=True)
    t0 = time.perf_counter()
    patience, patience_counter = 30, 0

    for epoch in range(start_epoch, max_epochs + 1):
        model.train()
        train_loss, train_batches = 0.0, 0
        optimizer.zero_grad()
        epoch_t0 = time.perf_counter()

        for step, (bx, by) in enumerate(train_loader):
            bx, by = bx.to(device), by.to(device)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                logits = model(bx)
                loss_hard = criterion_ce(logits.view(-1, vocab_size), by.view(-1))
                loss = loss_hard / grad_accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == total_steps_per_epoch:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            wsd_scheduler.step()
            train_loss += loss.item() * grad_accum_steps
            train_batches += 1

            if (step + 1) % 100 == 0 or (step + 1) == total_steps_per_epoch:
                current_avg_loss = train_loss / train_batches
                elapsed_epoch = time.perf_counter() - epoch_t0
                print(f"  [Epoch {epoch:2d} | Step {step+1:4d}/{total_steps_per_epoch}] Loss: {current_avg_loss:.4f} ({elapsed_epoch:.1f}s)", flush=True)

        avg_train_loss = train_loss / max(1, train_batches)

        if epoch % 2 == 0 or epoch == 1 or epoch == start_epoch:
            model.eval()
            val_loss, val_batches, correct, total = 0.0, 0, 0, 0
            with torch.no_grad():
                for bx_val, by_val in val_loader:
                    bx_val, by_val = bx_val.to(device), by_val.to(device)
                    with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                        v_logits = model(bx_val)
                        v_loss = criterion_ce(v_logits.view(-1, vocab_size), by_val.view(-1))
                    val_loss += v_loss.item()
                    val_batches += 1
                    preds = torch.argmax(v_logits, dim=-1)
                    mask = (by_val != 0)
                    correct += ((preds == by_val) & mask).sum().item()
                    total += mask.sum().item()

            avg_val_loss = val_loss / max(1, val_batches)
            val_acc = (correct / max(1, total)) * 100.0
            lr = optimizer.param_groups[0]['lr']
            print(f"\n  ==> Epoch {epoch:4d}/{max_epochs} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | Acc: {val_acc:.2f}% | LR: {lr:.6f}\n", flush=True)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                export_model_v14_3(model, optimizer, wsd_scheduler, vocab_size, epoch, best_val_loss, training_phase, "wrai_v14_3_best")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n[*] EARLY STOPPING at Epoch {epoch}! Best Val Loss: {best_val_loss:.4f}", flush=True)
                    break

            export_model_v14_3(model, optimizer, wsd_scheduler, vocab_size, epoch, avg_val_loss, training_phase, "wrai_v14_3_latest")

        if device.type == 'cuda':
            torch.cuda.empty_cache()

    t1 = time.perf_counter()
    print(f"\n[OK] WRAI v14.3.1 TRAINING COMPLETE in {(t1 - t0):.1f}s!", flush=True)
    export_model_v14_3(model, optimizer, wsd_scheduler, vocab_size, epoch, best_val_loss, training_phase, "wrai_v14_3_final")
    print(f"\n[ALL DONE] WRAI v14.3.1 model saved to Google Drive & Local!", flush=True)


if __name__ == "__main__":
    main()
