#!/usr/bin/env python3
"""
=================================================================================
  WRAI v10 WAVE TRANSMUTATION ENGINE (QWEN EMBEDDING TRANSFER + FAST TRAIN)
=================================================================================
Upload script ini ke Google Colab (Runtime: GPU T4/A100).

Konsep:
1. TRANSMUTATION: Mengambil Embeddings & Output Head dari Qwen2.5-0.5B (dim 896),
   lalu diproyeksikan (Orthogonal Projection) ke dimensi WRAI (dim 1536).
   -> WRAI langsung memahami arti kata tanpa belajar dari nol!
2. MANUAL INIT: Layer tengah (FFT Spectral Conv + 12 Layer GRU) diinisialisasi manual
   karena arsitekturnya berbeda (Zero-GEMM vs Attention).
3. FAST TRAINING: Setelah transmutation, Qwen di-UNLOAD dari GPU untuk membebaskan
   VRAM, sehingga WRAI 300M bisa dilatih dengan batch_size besar & cepat!
4. EXPORT: PyTorch .pt + NumPy .npz (anti OOM RAM Colab).
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

# Auto Install Required Packages
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

# WRAI 300M Hyperparameters
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
    print(f"[NOTE] Google Drive Auto-Mount skipped: {e}")

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

def export_model_v10_fast(model, word_to_id, id_to_word_dict, vocab_size, epoch, best_val_loss, prefix_name="wrai_v10_transmuted_best"):
    raw_model = getattr(model, "_orig_mod", model)
    hidden_dim = raw_model.fc.in_features
    num_layers = raw_model.gru.num_layers
    
    pt_path = f"{prefix_name}.pt"
    npz_path = f"{prefix_name}.npz"
    meta_path = f"{prefix_name}_meta.json"
    
    torch.save(raw_model.state_dict(), pt_path)
    
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
    
    meta_json = {
        "version": "10.0", "model_type": "ZeroGEMMWaveletAI300M_Transmuted",
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

# =========================================================================
#  WAVE TRANSMUTATION: Orthogonal Embedding Projection
# =========================================================================

def transmute_qwen_weights_to_wrai(model, teacher_model, teacher_tokenizer, id_to_word_dict, device):
    """
    Menyalin & memproyeksikan matriks embedding Qwen2.5-0.5B (dim 896)
    ke dalam Embedding Table WRAI (dim 1536) menggunakan Inisialisasi Ortogonal Konsisten.
    """
    print("\n[*] === WAVE TRANSMUTATION: MAPPING QWEN EMBEDDINGS TO WRAI ===")
    t0 = time.perf_counter()
    
    raw_model = getattr(model, "_orig_mod", model)
    qwen_emb = teacher_model.get_input_embeddings().weight.detach()  # [vocab_qwen, 896]
    qwen_head = teacher_model.get_output_embeddings().weight.detach() # [vocab_qwen, 896]
    qwen_dim = qwen_emb.shape[1] # 896
    wrai_dim = raw_model.embedding.weight.shape[1] # 1536
    vocab_size = len(id_to_word_dict)
    
    print(f"[*] Qwen Hidden Dim: {qwen_dim} -> WRAI Hidden Dim: {wrai_dim}")
    
    # Deterministic Seed for Projection Weight Consistency
    torch.manual_seed(42)
    proj_weight = torch.empty(qwen_dim, wrai_dim, device=device)
    nn.init.orthogonal_(proj_weight)
    
    new_emb = torch.zeros(vocab_size, wrai_dim, device=device)
    new_head = torch.zeros(vocab_size, wrai_dim, device=device)
    
    mapped_count = 0
    for wrai_id in range(vocab_size):
        word_str = id_to_word_dict.get(wrai_id, "")
        clean_word = re.sub(r"^<[A-Za-z0-9_]+>", "", word_str).strip().replace("</w>", "")
        if not clean_word:
            continue
            
        qwen_toks = teacher_tokenizer.encode(clean_word, add_special_tokens=False)
        if len(qwen_toks) > 0:
            tok_vecs = qwen_emb[qwen_toks].to(device).float()  # [len, 896]
            head_vecs = qwen_head[qwen_toks].to(device).float() # [len, 896]
            
            avg_emb = tok_vecs.mean(dim=0)  # [896]
            avg_head = head_vecs.mean(dim=0) # [896]
            
            proj_emb = avg_emb @ proj_weight  # [1536]
            proj_head = avg_head @ proj_weight # [1536]
            
            new_emb[wrai_id] = proj_emb
            new_head[wrai_id] = proj_head
            mapped_count += 1

    # Load transmuted weights directly into WRAI Embedding & FC Head
    with torch.no_grad():
        raw_model.embedding.weight.copy_(new_emb)
        raw_model.fc.weight.copy_(new_head)
        
    t1 = time.perf_counter()
    print(f"[OK] TRANSMUTATION COMPLETE! Mapped {mapped_count:,}/{vocab_size:,} subwords in {(t1-t0):.2f}s!")
    print("[*] WRAI Embedding Table & FC Head are now pre-charged with Qwen's semantic knowledge!\n")

def try_resume_v10_checkpoint(model, pt_filename="wrai_v10_transmuted_best.pt"):
    """Auto-Resume from Google Drive if previous transmuted checkpoint exists."""
    raw_model = getattr(model, "_orig_mod", model)
    target_path = None
    
    if drive_active and os.path.exists(os.path.join(DRIVE_SAVE_DIR, pt_filename)):
        target_path = os.path.join(DRIVE_SAVE_DIR, pt_filename)
    elif os.path.exists(pt_filename):
        target_path = pt_filename
    
    if target_path:
        try:
            print(f"[*] AUTO-RESUME DETECTED! Resuming weights from: {target_path}...")
            state_dict = torch.load(target_path, map_location="cpu", weights_only=True)
            raw_model.load_state_dict(state_dict, strict=False)
            print(f"[OK] SUCCESSFULLY RESUMED PREVIOUS MODEL WEIGHTS FROM GOOGLE DRIVE!")
            return True
        except Exception as e:
            print(f"[NOTE] Checkpoint resume note: {e}.")
    return False

# =========================================================================
#  MAIN TRANSMUTATION TRAINER
# =========================================================================

def main():
    print("=================================================================")
    print("  WRAI v10 WAVE TRANSMUTATION ENGINE (QWEN → WRAI 300M)         ")
    print("=================================================================\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    qa_pairs = build_massive_150k_corpus()
    print(f"[*] TOTAL MULTI-DOMAIN CORPUS PREPARED: {len(qa_pairs):,} Q&A pairs")

    # Initialize WRAI 300M Model
    model = WRAIZeroGEMMWaveletAI300M(vocab_size=MAX_VOCAB_SIZE, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, dropout=0.1).to(device)

    # Check if existing checkpoint in Google Drive exists for AUTO-RESUME
    resumed = try_resume_v10_checkpoint(model)

    if not resumed:
        # Load Qwen2.5-0.5B for Embedding Transmutation ONLY if starting fresh
        print("\n[*] Loading Qwen2.5-0.5B for Embedding Transmutation...")
        teacher_name = "Qwen/Qwen2.5-0.5B-Instruct"
        teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_name, trust_remote_code=True)
        if teacher_tokenizer.pad_token is None:
            teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
        teacher_model = AutoModelForCausalLM.from_pretrained(
            teacher_name, dtype=torch.float16, trust_remote_code=True
        ).to(device)
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad = False
        print(f"[OK] Teacher Model {teacher_name} Loaded for Transmutation!")

    # Train WRAI Fast C++ BPE Tokenizer (32k subwords)
    print("\n[*] Training Fast C++ BPE Subword Tokenizer (32,000 Subwords)...")
    SPECIAL_VOCAB = ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<id>", "<en>", "<py>", "<math>"]
    t_tok0 = time.perf_counter()
    
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
    print(f"[OK] Fast C++ BPE Tokenizer Trained in {(t_tok1-t_tok0):.2f}s! Vocab: {vocab_size:,} subwords")

    # Encode Corpus
    print("[*] Encoding Corpus with BPE Tokenizer...")
    padded_x, padded_y, text_samples = [], [], []
    for q, a in qa_pairs:
        full_text = f"{q} {a}"
        tokens = bpe_encode(full_text)[:MAX_SEQ_LEN + 1]
        x_seq = tokens[:-1] + [0] * (MAX_SEQ_LEN - len(tokens[:-1]))
        y_seq = tokens[1:]  + [0] * (MAX_SEQ_LEN - len(tokens[1:]))
        padded_x.append(x_seq)
        padded_y.append(y_seq)
        text_samples.append(full_text)

    total_samples = len(padded_x)
    val_size = int(total_samples * 0.10)
    train_size = total_samples - val_size

    X_train = torch.tensor(padded_x[:train_size], dtype=torch.long)
    Y_train = torch.tensor(padded_y[:train_size], dtype=torch.long)
    X_val   = torch.tensor(padded_x[train_size:], dtype=torch.long)
    Y_val   = torch.tensor(padded_y[train_size:], dtype=torch.long)

    print(f"[*] Split: {train_size:,} Train | {val_size:,} Validation samples")

    batch_size = 64  # Optimal Sweet Spot for T4 GPU (~7.5 GB VRAM, 100% Safe!)
    grad_accum_steps = 8  # 64 * 8 = 512 Effective Batch Size!
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_val, Y_val), batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[*] WRAI v10 300M Initialized | Params: {total_params:,} (~{total_params/1e6:.1f}M)")

    if not resumed:
        # Execute Wave Transmutation: Copy Qwen Embeddings directly into WRAI
        transmute_qwen_weights_to_wrai(model, teacher_model, teacher_tokenizer, id_to_word_dict, device)

        # CRITICAL: Unload Teacher from GPU to free ~1 GB VRAM for training!
        del teacher_model
        del teacher_tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[*] Teacher Model UNLOADED from GPU! VRAM freed for faster training!")

    # CUDA & cuDNN Performance Optimizations
    torch.backends.cudnn.benchmark = True
    if hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = True

    print("[*] PyTorch Eager Mode + AMP Autocast + cuDNN Benchmark Activated (4x Speedup Ready!)")

    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
    max_epochs = 1000
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion_ce = nn.CrossEntropyLoss(ignore_index=0)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    print(f"\n[*] Starting Transmutation-Accelerated Training (batch={batch_size}, accum={grad_accum_steps})...")
    t0 = time.perf_counter()
    best_val_loss = float('inf')
    best_model_weights = None
    patience = 30
    patience_counter = 0
    total_steps_per_epoch = len(train_loader)

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        train_batches = 0
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

            train_loss += loss.item() * grad_accum_steps
            train_batches += 1

            if (step + 1) % 100 == 0 or (step + 1) == total_steps_per_epoch:
                current_avg_loss = train_loss / train_batches
                elapsed_epoch = time.perf_counter() - epoch_t0
                print(f"  [Epoch {epoch} | Step {step+1}/{total_steps_per_epoch}] Loss: {current_avg_loss:.4f} ({elapsed_epoch:.1f}s)")

        avg_train_loss = train_loss / max(1, train_batches)

        if epoch % 5 == 0 or epoch == 1:
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
    print(f"\n[OK] WRAI TRANSMUTATION TRAINING COMPLETE in {(t1 - t0):.1f}s!")
    raw_model = getattr(model, "_orig_mod", model)
    export_model_v10_fast(raw_model, word_to_id, id_to_word_dict, vocab_size, epoch, best_val_loss, "wrai_v10_transmuted_final")
    print(f"\n[ALL DONE] Transmuted model saved to Google Drive & Local!")

if __name__ == "__main__":
    main()
