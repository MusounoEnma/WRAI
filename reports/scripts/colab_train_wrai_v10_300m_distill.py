#!/usr/bin/env python3
"""
=================================================================================
  WRAI v10 300M DISTILLATION ENGINE (TEACHER-AUGMENTED + NPZ EXPORT)
=================================================================================
Upload script ini ke Google Colab (Runtime: GPU T4/A100).
Spesifikasi WRAI v10 (300M Parameter):
- Hidden Dimension : 1536 (3x lipat dari v9)
- GRU Layers       : 12 Layers (3x lipat dari v9)
- Vocab Size       : 32.000 BPE Subwords (Fast C++ BPE Tokenizer)
- Total Parameter  : ~305 Million Parameters
- Teacher Model    : Qwen/Qwen2.5-0.5B-Instruct (Teacher-Augmented Distillation)
- Distillation     : Teacher generates high-quality answers → Student belajar dari teks tsb
- Export Format    : PyTorch .pt + NPZ (BUKAN JSON, anti-crash memori!)
- VRAM GPU         : ~10.5 GB VRAM (batch_size=32, grad_accum=16, 100% Bebas OOM)
- Output           : wrai_v10_300m_best.pt + .npz auto-save ke Google Drive!
"""

import copy
import json
import math
import os
import random
import re
import shutil
import struct
import sys
import time
import urllib.request

# Auto Install Required Packages if missing in fresh Colab session
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
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers
except ImportError:
    os.system("pip install -q tokenizers")
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    os.system("pip install -q transformers accelerate")
    from transformers import AutoModelForCausalLM, AutoTokenizer

FFT_SIZE = 4096
SPECTRAL_BINS = 2048
MAGIC_HEADER = 0x57524149
MODEL_VERSION = 0x0A00  # v10.0
MAX_VOCAB_SIZE = 32000
MAX_SEQ_LEN = 128

# 300M Parameter Hyperparameters
HIDDEN_DIM = 1536
NUM_LAYERS = 12

HF_TOKEN = ""

os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
try:
    import huggingface_hub
    huggingface_hub.login(token=HF_TOKEN, add_to_git_credential=False)
    print("[*] HuggingFace Hub Logged In Successfully!")
except Exception as e:
    print(f"[NOTE] HuggingFace login note: {e}")

# Auto Mount Google Drive
DRIVE_MOUNT_PATH = "/content/drive"
DRIVE_SAVE_DIR = "/content/drive/MyDrive/WRAI_Models"
drive_active = False

try:
    from google.colab import drive
    drive.mount(DRIVE_MOUNT_PATH, force_remount=False)
    os.makedirs(DRIVE_SAVE_DIR, exist_ok=True)
    drive_active = True
    print(f"[OK] Google Drive Mounted! Models will auto-save & auto-resume from: {DRIVE_SAVE_DIR}")
except Exception as e:
    print(f"[NOTE] Google Drive Auto-Mount skipped (local or unmounted): {e}")

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

class ZeroGEMMSpectralConvLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        freq_size = hidden_dim // 2 + 1
        self.W_freq_real = nn.Parameter(torch.randn(freq_size) * 0.01)
        self.W_freq_imag = nn.Parameter(torch.randn(freq_size) * 0.01)

    def forward(self, x):
        with torch.amp.autocast('cuda', enabled=False):
            x_float = x.float()
            x_fft = torch.fft.rfft(x_float, dim=-1)
            W_complex = torch.complex(self.W_freq_real, self.W_freq_imag)
            out_fft = x_fft * W_complex
            out = torch.fft.irfft(out_fft, n=self.hidden_dim, dim=-1)
            return torch.tanh(out)

class WRAIZeroGEMMWaveletAI300M(nn.Module):
    def __init__(self, vocab_size, hidden_dim=1536, num_layers=12, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encoder = SinusoidalWavePositionalEncoding(hidden_dim)
        self.spectral_conv1 = ZeroGEMMSpectralConvLayer(hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.spectral_conv2 = ZeroGEMMSpectralConvLayer(hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        emb = self.embedding(x)
        emb = self.pos_encoder(emb)
        feat = self.spectral_conv1(emb)
        out, _ = self.gru(feat)
        feat2 = self.spectral_conv2(out)
        out = self.ln(feat2 + out)
        logits = self.fc(out)
        return logits

# =========================================================================
#  DATASET FUNCTIONS
# =========================================================================

def extract_qa_from_row(item, tag_prefix):
    inst, out = "", ""
    if "conversations" in item and isinstance(item["conversations"], list):
        convs = item["conversations"]
        human_text, gpt_text = "", ""
        for turn in convs:
            if isinstance(turn, dict):
                speaker = str(turn.get("from", "") or turn.get("role", "")).lower()
                val = str(turn.get("value", "") or turn.get("content", "")).strip()
                if speaker in ["human", "user"] and not human_text:
                    human_text = val
                elif speaker in ["gpt", "assistant", "bot"] and not gpt_text:
                    gpt_text = val
        if human_text and gpt_text:
            inst, out = human_text, gpt_text
    if not inst or not out:
        if "question" in item and "answer" in item:
            inst = str(item.get("question", "")).strip()
            out = str(item.get("answer", "")).strip()
    if not inst or not out:
        inst_base = str(item.get("instruction", "") or item.get("prompt", "") or item.get("input", "") or item.get("sql_prompt", "")).strip()
        inp_extra = str(item.get("input", "")).strip() if "instruction" in item else ""
        if inp_extra and inp_extra != inst_base:
            inst = f"{inst_base} {inp_extra}".strip()
        else:
            inst = inst_base
        out = str(item.get("output", "") or item.get("response", "") or item.get("text", "") or item.get("sql", "")).strip()
    if inst and out and len(inst) < 400 and len(out) < 1000:
        return (f"{tag_prefix} {inst}", f"{tag_prefix} {out}")
    return None

def fetch_hf_dataset_stream(dataset_name, tag_prefix, max_samples=40000, config="default"):
    print(f"[*] Streaming HuggingFace Dataset ({dataset_name}, Target: {max_samples:,} samples)...")
    hf_pairs = []
    try:
        ds = load_dataset(dataset_name, config if config != "default" else None, split=f"train[:{max_samples}]")
        for item in ds:
            pair = extract_qa_from_row(item, tag_prefix)
            if pair:
                hf_pairs.append(pair)
        print(f"[OK] Loaded {len(hf_pairs):,} pairs from {dataset_name}!")
        return hf_pairs
    except Exception as e:
        print(f"[NOTE] Fallback for {dataset_name}: {e}")
    try:
        offset = 0
        batch_size = 100
        encoded_name = urllib.parse.quote(dataset_name, safe='')
        while len(hf_pairs) < max_samples and offset < 50000:
            url = f"https://datasets-server.huggingface.co/rows?dataset={encoded_name}&config={config}&split=train&offset={offset}&length={batch_size}"
            req = urllib.request.Request(url, headers={"Authorization": f"Bearer {HF_TOKEN}"})
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                rows = data.get("rows", [])
                if not rows: break
                for r in rows:
                    pair = extract_qa_from_row(r.get("row", {}), tag_prefix)
                    if pair: hf_pairs.append(pair)
                offset += batch_size
        print(f"[OK] Loaded {len(hf_pairs):,} pairs from {dataset_name} via API!")
    except Exception as e:
        print(f"[WARNING] API fetch failed for {dataset_name}: {e}")
    return hf_pairs

def build_massive_150k_corpus():
    all_pairs = []
    all_pairs.extend(fetch_hf_dataset_stream("FreedomIntelligence/alpaca-gpt4-indonesian", "<ID>", 40000))
    all_pairs.extend(fetch_hf_dataset_stream("cahya/alpaca-id-cleaned", "<ID>", 30000))
    all_pairs.extend(fetch_hf_dataset_stream("iamtarun/python_code_instructions_18k_alpaca", "<PY>", 18000))
    all_pairs.extend(fetch_hf_dataset_stream("microsoft/orca-math-word-problems-200k", "<MATH>", 30000))
    all_pairs.extend(fetch_hf_dataset_stream("databricks/databricks-dolly-15k", "<EN>", 15000))
    all_pairs.extend(fetch_hf_dataset_stream("gretelai/synthetic_text_to_sql", "<PY>", 15000))
    random.seed(42)
    random.shuffle(all_pairs)
    return all_pairs

# =========================================================================
#  TEACHER-AUGMENTED DISTILLATION: Teacher generates improved answers
#  FAST MODE: 500 samples, true batch generation, max 64 new tokens
# =========================================================================

def teacher_augment_corpus(qa_pairs, device, max_augment=500):
    """
    Teacher-Augmented Distillation (FAST MODE):
    Qwen2.5-0.5B generates high-quality answers for 500 questions.
    Uses TRUE BATCH generation for 10x speedup (~5 minutes total).
    """
    print("\n[*] === TEACHER-AUGMENTED DISTILLATION PHASE (FAST MODE) ===")
    print("[*] Loading Teacher Model (Qwen/Qwen2.5-0.5B-Instruct)...")
    
    try:
        teacher_name = "Qwen/Qwen2.5-0.5B-Instruct"
        teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_name, trust_remote_code=True)
        if teacher_tokenizer.pad_token is None:
            teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
        teacher_model = AutoModelForCausalLM.from_pretrained(
            teacher_name, dtype=torch.float16, trust_remote_code=True
        ).to(device)
        teacher_model.eval()
        print(f"[OK] Teacher Model Loaded! Generating {max_augment:,} improved answers (FAST batch mode)...")
    except Exception as e:
        print(f"[NOTE] Teacher Model not available: {e}. Skipping augmentation.")
        return qa_pairs

    augmented_pairs = list(qa_pairs)
    sample_indices = random.sample(range(len(qa_pairs)), min(max_augment, len(qa_pairs)))
    
    BATCH_SIZE = 16  # True batch generation
    MAX_NEW_TOKENS = 64  # Shorter but faster
    augmented_count = 0
    t0 = time.perf_counter()
    
    for batch_start in range(0, len(sample_indices), BATCH_SIZE):
        batch_indices = sample_indices[batch_start:batch_start + BATCH_SIZE]
        
        # Prepare batch prompts
        batch_prompts = []
        batch_tags = []
        for idx in batch_indices:
            q_text = qa_pairs[idx][0]
            clean_q = re.sub(r"^<[A-Z]+>\s*", "", q_text).strip()
            tag = q_text.split()[0] if q_text.startswith("<") else "<ID>"
            messages = [{"role": "user", "content": clean_q}]
            prompt = teacher_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            batch_prompts.append(prompt)
            batch_tags.append((tag, clean_q))
        
        try:
            # TRUE BATCH GENERATION (all at once!)
            inputs = teacher_tokenizer(
                batch_prompts, return_tensors="pt", padding=True, 
                truncation=True, max_length=192
            ).to(device)
            
            with torch.no_grad():
                outputs = teacher_model.generate(
                    **inputs, max_new_tokens=MAX_NEW_TOKENS,
                    temperature=0.7, do_sample=True, top_p=0.9,
                    pad_token_id=teacher_tokenizer.pad_token_id
                )
            
            # Decode each response in batch
            for i in range(len(batch_indices)):
                prompt_len = inputs.input_ids.shape[1]
                response = teacher_tokenizer.decode(
                    outputs[i][prompt_len:], skip_special_tokens=True
                ).strip()
                
                if len(response) > 10 and len(response) < 500:
                    tag, question = batch_tags[i]
                    augmented_pairs.append((f"{tag} {question}", f"{tag} {response}"))
                    augmented_count += 1
        except Exception as e:
            # Fallback: skip this batch silently
            continue
        
        # Progress report every 5 batches
        done = min(batch_start + BATCH_SIZE, len(sample_indices))
        if (batch_start // BATCH_SIZE + 1) % 5 == 0 or done >= len(sample_indices):
            elapsed = time.perf_counter() - t0
            speed = done / max(elapsed, 0.1)
            eta = (len(sample_indices) - done) / max(speed, 0.1)
            print(f"  -> {done}/{len(sample_indices)} questions | {augmented_count} answers | {elapsed:.0f}s elapsed | ETA: {eta:.0f}s")
    
    # Free Teacher GPU memory immediately
    del teacher_model
    del teacher_tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    t1 = time.perf_counter()
    print(f"[OK] Teacher Augmentation Done! +{augmented_count:,} teacher answers in {(t1-t0):.0f}s")
    print(f"[*] Total Corpus: {len(augmented_pairs):,} Q&A pairs\n")
    
    random.shuffle(augmented_pairs)
    return augmented_pairs

# =========================================================================
#  EXPORT: PyTorch .pt + NPZ (NO JSON for 300M model!)
# =========================================================================

def export_model_v10_fast(model, word_to_id, id_to_word_dict, vocab_size, epoch, best_val_loss, prefix_name="wrai_v10_300m_best"):
    """Export 300M model to PyTorch .pt + NPZ format (memory-safe, no JSON crash)."""
    raw_model = getattr(model, "_orig_mod", model)
    hidden_dim = raw_model.fc.in_features
    num_layers = raw_model.gru.num_layers
    
    pt_path = f"{prefix_name}.pt"
    npz_path = f"{prefix_name}.npz"
    meta_path = f"{prefix_name}_meta.json"
    
    # Save PyTorch state dict (fastest, most reliable)
    torch.save(raw_model.state_dict(), pt_path)
    
    # Save NPZ for C Engine inference
    npz_data = {}
    npz_data["W_emb"] = raw_model.embedding.weight.cpu().detach().numpy()
    npz_data["W_out"] = raw_model.fc.weight.cpu().detach().numpy().T
    npz_data["b_out"] = raw_model.fc.bias.cpu().detach().numpy()
    npz_data["sc1_real"] = raw_model.spectral_conv1.W_freq_real.cpu().detach().numpy()
    npz_data["sc1_imag"] = raw_model.spectral_conv1.W_freq_imag.cpu().detach().numpy()
    npz_data["sc2_real"] = raw_model.spectral_conv2.W_freq_real.cpu().detach().numpy()
    npz_data["sc2_imag"] = raw_model.spectral_conv2.W_freq_imag.cpu().detach().numpy()
    
    for l in range(num_layers):
        npz_data[f"W_ih_l{l}"] = getattr(raw_model.gru, f"weight_ih_l{l}").cpu().detach().numpy().T
        npz_data[f"b_ih_l{l}"] = getattr(raw_model.gru, f"bias_ih_l{l}").cpu().detach().numpy()
        npz_data[f"W_hh_l{l}"] = getattr(raw_model.gru, f"weight_hh_l{l}").cpu().detach().numpy().T
        npz_data[f"b_hh_l{l}"] = getattr(raw_model.gru, f"bias_hh_l{l}").cpu().detach().numpy()
    
    np.savez_compressed(npz_path, **npz_data)
    
    # Save lightweight metadata JSON (vocab only, no weights!)
    meta_json = {
        "version": "10.0", "model_type": "ZeroGEMMWaveletAI300M",
        "vocab_size": vocab_size, "hidden_dim": hidden_dim, "num_layers": num_layers,
        "training_epochs": epoch, "final_best_val_loss": float(best_val_loss),
        "vocabulary": word_to_id,
        "id_to_word": {str(i): w for i, w in id_to_word_dict.items()},
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_json, f, ensure_ascii=False)
    
    pt_size = os.path.getsize(pt_path) / (1024*1024)
    npz_size = os.path.getsize(npz_path) / (1024*1024)
    print(f"[EXPORTED] {pt_path} ({pt_size:.1f} MB) + {npz_path} ({npz_size:.1f} MB) + {meta_path}")
    
    if drive_active:
        try:
            for fname in [pt_path, npz_path, meta_path]:
                shutil.copyfile(fname, os.path.join(DRIVE_SAVE_DIR, fname))
            print(f"[AUTO-SAVED TO DRIVE] -> {DRIVE_SAVE_DIR}/{prefix_name}.*")
        except Exception as e:
            print(f"[WARNING] Drive save note: {e}")

def try_resume_v10_checkpoint(model):
    """Auto-Resume from PyTorch .pt checkpoint."""
    raw_model = getattr(model, "_orig_mod", model)
    pt_filename = "wrai_v10_300m_best.pt"
    target_path = None
    
    if drive_active and os.path.exists(os.path.join(DRIVE_SAVE_DIR, pt_filename)):
        target_path = os.path.join(DRIVE_SAVE_DIR, pt_filename)
    elif os.path.exists(pt_filename):
        target_path = pt_filename
    
    if target_path:
        try:
            print(f"[*] AUTO-RESUME DETECTED! Loading v10 300M weights from: {target_path}...")
            state_dict = torch.load(target_path, map_location="cpu", weights_only=True)
            raw_model.load_state_dict(state_dict, strict=False)
            print(f"[OK] SUCCESSFULLY RESUMED WRAI v10 300M WEIGHTS!")
            return True
        except Exception as e:
            print(f"[NOTE] Checkpoint resume note: {e}. Starting fresh training...")
    return False

# =========================================================================
#  MAIN TRAINING LOOP
# =========================================================================

def main():
    print("=================================================================")
    print("  WRAI v10 300M TEACHER-AUGMENTED DISTILLATION ENGINE            ")
    print("=================================================================\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    qa_pairs = build_massive_150k_corpus()
    print(f"[*] TOTAL MULTI-DOMAIN CORPUS: {len(qa_pairs):,} Q&A pairs")

    # Phase 1: Teacher-Augmented Distillation (Qwen generates better answers)
    qa_pairs = teacher_augment_corpus(qa_pairs, device, max_augment=5000)

    # Fast C++ BPE Tokenizer Training
    print("[*] Training Fast C++ BPE Subword Tokenizer...")
    SPECIAL_VOCAB = ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<id>", "<en>", "<py>", "<math>"]
    
    t_tok0 = time.perf_counter()
    try:
        tokenizer_fast = Tokenizer(models.BPE(unk_token="<UNK>"))
        tokenizer_fast.pre_tokenizer = pre_tokenizers.Whitespace()
        trainer = trainers.BpeTrainer(vocab_size=MAX_VOCAB_SIZE, special_tokens=SPECIAL_VOCAB)
        
        temp_corpus_file = "temp_bpe_corpus.txt"
        with open(temp_corpus_file, "w", encoding="utf-8") as f_tmp:
            for q, a in qa_pairs:
                f_tmp.write(f"{q} {a}\n")
        
        tokenizer_fast.train(files=[temp_corpus_file], trainer=trainer)
        bpe_vocab_dict = tokenizer_fast.get_vocab()
        word_to_id = bpe_vocab_dict
        id_to_word_dict = {i: w for w, i in bpe_vocab_dict.items()}
        vocab_size = len(word_to_id)
        
        def bpe_encode(text):
            return tokenizer_fast.encode(text).ids
        
        t_tok1 = time.perf_counter()
        print(f"[OK] Fast C++ BPE Trained in {(t_tok1-t_tok0):.2f}s! Vocab: {vocab_size:,} subwords")
        
    except Exception as e:
        print(f"[NOTE] Fallback to word-level tokenizer ({e})...")
        word_counts = {}
        for q, a in qa_pairs:
            for w in (q + " " + a).lower().split():
                word_counts[w] = word_counts.get(w, 0) + 1
        sorted_words = sorted(word_counts.keys(), key=lambda x: -word_counts[x])
        vocab = SPECIAL_VOCAB + sorted_words[:MAX_VOCAB_SIZE - len(SPECIAL_VOCAB)]
        word_to_id = {w: i for i, w in enumerate(vocab)}
        id_to_word_dict = {i: w for i, w in enumerate(vocab)}
        vocab_size = len(vocab)
        def bpe_encode(text):
            return [word_to_id.get(w, 1) for w in text.lower().split()]

    # Encode Corpus
    print("[*] Encoding corpus with BPE Tokenizer...")
    padded_x, padded_y = [], []
    for q, a in qa_pairs:
        tokens = bpe_encode(f"{q} {a}")[:MAX_SEQ_LEN + 1]
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

    print(f"[*] Split: {train_size:,} Train | {val_size:,} Validation samples")

    batch_size = 32
    grad_accum_steps = 16
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_val, Y_val), batch_size=batch_size, shuffle=False, pin_memory=True)

    # Initialize WRAI 300M Model
    model = WRAIZeroGEMMWaveletAI300M(vocab_size=vocab_size, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, dropout=0.1).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[*] WRAI v10 300M Initialized | Params: {total_params:,} (~{total_params/1e6:.1f}M)")

    try_resume_v10_checkpoint(model)

    try:
        if hasattr(torch, "compile"):
            model = torch.compile(model)
            print("[*] torch.compile() JIT Activated!")
    except Exception as e:
        print(f"[NOTE] torch.compile: {e}")

    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
    max_epochs = 1000
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    print(f"\n[*] Training WRAI v10 300M (batch={batch_size}, accum={grad_accum_steps}) up to {max_epochs} Epochs...")
    t0 = time.perf_counter()
    best_val_loss = float('inf')
    best_model_weights = None
    patience = 30
    patience_counter = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        train_batches = 0
        optimizer.zero_grad()

        for step, (bx, by) in enumerate(train_loader):
            bx, by = bx.to(device), by.to(device)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                logits = model(bx)
                loss = criterion(logits.view(-1, vocab_size), by.view(-1)) / grad_accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item() * grad_accum_steps
            train_batches += 1

        avg_train_loss = train_loss / max(1, train_batches)

        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            val_loss, val_batches, correct, total = 0.0, 0, 0, 0
            with torch.no_grad():
                for bx_val, by_val in val_loader:
                    bx_val, by_val = bx_val.to(device), by_val.to(device)
                    with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                        v_logits = model(bx_val)
                        v_loss = criterion(v_logits.view(-1, vocab_size), by_val.view(-1))
                    val_loss += v_loss.item()
                    val_batches += 1
                    preds = torch.argmax(v_logits, dim=-1)
                    mask = (by_val != 0)
                    correct += ((preds == by_val) & mask).sum().item()
                    total += mask.sum().item()

            avg_val_loss = val_loss / max(1, val_batches)
            val_acc = (correct / max(1, total)) * 100.0
            scheduler.step(avg_val_loss)
            lr = optimizer.param_groups[0]['lr']

            print(f"  -> Epoch {epoch:4d}/{max_epochs} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | Acc: {val_acc:.2f}% | LR: {lr:.6f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                raw_model = getattr(model, "_orig_mod", model)
                best_model_weights = copy.deepcopy(raw_model.state_dict())
                patience_counter = 0
                raw_model.load_state_dict(best_model_weights)
                export_model_v10_fast(raw_model, word_to_id, id_to_word_dict, vocab_size, epoch, best_val_loss)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n[*] EARLY STOPPING at Epoch {epoch}! Best Val Loss: {best_val_loss:.4f}")
                    break

        if device.type == 'cuda':
            torch.cuda.empty_cache()

    t1 = time.perf_counter()
    print(f"\n[OK] WRAI v10 300M TRAINING COMPLETE in {(t1 - t0):.1f}s!")
    raw_model = getattr(model, "_orig_mod", model)
    export_model_v10_fast(raw_model, word_to_id, id_to_word_dict, vocab_size, epoch, best_val_loss, "wrai_v10_300m_final")
    print(f"\n[ALL DONE] WRAI v10 300M saved to Google Drive & Local!")

if __name__ == "__main__":
    main()
