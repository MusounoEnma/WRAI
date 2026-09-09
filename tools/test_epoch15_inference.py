#!/usr/bin/env python3
"""
=================================================================================
  WRAI v14.1 INTERACTIVE INFERENCE TESTER (CLEAN WORD-SPACED DECODER)
=================================================================================
Jalankan script ini di Colab atau Local untuk menguji hasil generasi teks WRAI v14.1 Epoch 15!
"""

import json
import math
import os
import sys

try:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    os.system("pip install -q torch numpy tokenizers")
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

try:
    from tokenizers import Tokenizer, models, pre_tokenizers, AddedToken
except ImportError:
    os.system("pip install -q tokenizers")
    from tokenizers import Tokenizer, models, pre_tokenizers, AddedToken

DRIVE_SAVE_DIR = "/content/drive/MyDrive/WRAI_Models"
PT_NAME = "wrai_v14_scratch_best.pt"
META_NAME = "wrai_v14_scratch_best_meta.json"

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
        with torch.no_grad():
            x_float = x.float()
            x_fft = torch.fft.rfft(x_float, dim=-1)
            W_complex = torch.complex(self.W_freq_real, self.W_freq_imag)
            out_fft = x_fft * W_complex
            out = torch.fft.irfft(out_fft, n=self.hidden_dim, dim=-1)
            return torch.tanh(out)

class WRAIZeroGEMMPureScratchv14(nn.Module):
    def __init__(self, vocab_size=32000, hidden_dim=1024, num_layers=12, dropout=0.0):
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

def find_checkpoint():
    target_pt = None
    target_meta = None
    
    if os.path.exists(os.path.join(DRIVE_SAVE_DIR, PT_NAME)):
        target_pt = os.path.join(DRIVE_SAVE_DIR, PT_NAME)
        target_meta = os.path.join(DRIVE_SAVE_DIR, META_NAME)
    elif os.path.exists(PT_NAME):
        target_pt = PT_NAME
        target_meta = META_NAME
        
    return target_pt, target_meta

def decode_tokens_clean(tokens, id_to_word_dict):
    words = []
    for t in tokens:
        w = id_to_word_dict.get(str(t), id_to_word_dict.get(t, ""))
        if w and w not in ["<PAD>", "<UNK>", "<BOS>"]:
            words.append(w)
    return " ".join(words)

def generate_text_clean(model, tokenizer, id_to_word_dict, prompt_text, max_new_tokens=40, temperature=0.7, top_k=40, device="cpu"):
    model.eval()
    tokens = tokenizer.encode(prompt_text).ids
    input_ids = torch.tensor([tokens], dtype=torch.long).to(device)
    
    eos_id = tokenizer.token_to_id("<EOS>")
    generated = list(tokens)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            if input_ids.size(1) > 128:
                input_ids = input_ids[:, -128:]
                
            logits = model(input_ids)[:, -1, :] / max(1e-5, temperature)
            
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            if next_token == eos_id:
                break
                
            generated.append(next_token)
            input_ids = torch.tensor([generated], dtype=torch.long).to(device)
            
    return decode_tokens_clean(generated, id_to_word_dict)

def main():
    print("=================================================================", flush=True)
    print("  WRAI v14.1 INTERACTIVE INFERENCE TESTER (CLEAN WORD-SPACED)    ", flush=True)
    print("=================================================================\n", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Inference Device: {device}", flush=True)

    pt_path, meta_path = find_checkpoint()
    if not pt_path or not os.path.exists(pt_path):
        print(f"[ERROR] Checkpoint {PT_NAME} tidak ditemukan di Google Drive atau lokal!", flush=True)
        return

    print(f"[*] Loading Checkpoint from: {pt_path}...", flush=True)
    ckpt = torch.load(pt_path, map_location=device)
    
    saved_vocab = None
    id_to_word_dict = {}
    epoch_num = 15
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f_m:
            meta = json.load(f_m)
            saved_vocab = meta.get("vocabulary", None)
            id_to_word_dict = meta.get("id_to_word", {})
            epoch_num = meta.get("training_epochs", 15)

    vocab_size = 32000
    model = WRAIZeroGEMMPureScratchv14(vocab_size=vocab_size, hidden_dim=1024, num_layers=12).to(device)
    
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
        
    print(f"[OK] Checkpoint Epoch {epoch_num} Loaded Successfully!", flush=True)

    # Reconstruct Tokenizer from Meta Vocabulary
    SPECIAL_VOCAB = ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<ID>", "<EN>", "<PY>", "<MATH>"]
    tokenizer_fast = Tokenizer(models.BPE(unk_token="<UNK>"))
    tokenizer_fast.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer_fast.add_special_tokens([AddedToken(tok, single_word=True) for tok in SPECIAL_VOCAB])
    
    if saved_vocab:
        bpe_model = models.BPE(vocab=saved_vocab, merges=[], unk_token="<UNK>")
        tokenizer_fast.model = bpe_model
        print(f"[OK] Tokenizer Vocabulary Restored ({len(saved_vocab):,} subwords)!", flush=True)

    test_prompts = [
        "<ID> Siapa kamu dan apa kemampuanmu?",
        "<ID> Apakah ibu kota Indonesia?",
        "<EN> Explain what artificial intelligence is.",
        "<PY> def sapaan(nama):",
        "<MATH> Berapakah hasil dari 25 ditambah 15?",
    ]

    print(f"\n=================================================================", flush=True)
    print(f"  HASIL INFERENCE WRAI v14.1 CHECKPOINT EPOCH {epoch_num}          ", flush=True)
    print(f"=================================================================\n", flush=True)

    for i, prompt in enumerate(test_prompts, 1):
        print(f"[{i}] PROMPT : {prompt}", flush=True)
        try:
            res = generate_text_clean(model, tokenizer_fast, id_to_word_dict, prompt, max_new_tokens=40, temperature=0.7, device=device)
            print(f"    HASIL  : {res}\n", flush=True)
        except Exception as e:
            print(f"    ERROR  : {e}\n", flush=True)

if __name__ == "__main__":
    main()
