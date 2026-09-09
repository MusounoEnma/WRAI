#!/usr/bin/env python3
"""
=================================================================================
  WRAI v14.4 INTERACTIVE INFERENCE EVALUATOR (DISTILLED MODEL EVALUATION)
=================================================================================
Jalankan script ini di Colab (atau Lokal) untuk menguji kualitas generasi teks 
WRAI v14.4 yang telah di-distilasi dari Teacher Qwen2.5-1.5B-Instruct.
"""

import json
import math
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure sympy and transformers are compatible in Colab environment
try:
    import sympy.printing
    from transformers import AutoTokenizer
except (ImportError, AttributeError):
    os.system("pip install -q --upgrade sympy transformers")
    import sympy.printing
    from transformers import AutoTokenizer

DRIVE_MOUNT_PATH = "/content/drive"
DRIVE_SAVE_DIR = "/content/drive/MyDrive/WRAI_Models"
VOCAB_MAP_FILENAME = "pruned_vocab_map_v14_4.json"

try:
    from google.colab import drive
    drive.mount(DRIVE_MOUNT_PATH, force_remount=False)
    print(f"\n[OK] GOOGLE DRIVE SUCCESSFULLY MOUNTED! Models loaded from: {DRIVE_SAVE_DIR}\n", flush=True)
except Exception as e:
    print(f"\n[NOTICE] Local / Non-Colab execution detected: {e}\n", flush=True)

PRUNED_VOCAB_SIZE = 32000
MAX_SEQ_LEN = 128
HIDDEN_DIM = 1024
NUM_LAYERS = 12
WAVELET_LEVELS = 4

TEACHER_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

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

def generate_text_v14_4(model, teacher_tokenizer, teacher_id_to_pruned_id, pruned_id_to_teacher_id, prompt, max_new_tokens=40, temperature=0.65, top_k=35, repetition_penalty=1.2, device="cuda"):
    model.eval()
    
    teacher_enc = teacher_tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()
    prompt_oov = sum(1 for tid in teacher_enc if tid not in teacher_id_to_pruned_id)
    
    pruned_input = [teacher_id_to_pruned_id.get(tid, 0) for tid in teacher_enc]
    input_ids = torch.tensor([pruned_input], dtype=torch.long, device=device)
    
    generated_pruned_ids = list(pruned_input)
    eos_teacher_id = teacher_tokenizer.eos_token_id if teacher_tokenizer.eos_token_id is not None else 151645
    eos_pruned_id = teacher_id_to_pruned_id.get(eos_teacher_id, None)
    
    generated_oov_count = 0
    with torch.no_grad():
        for _ in range(max_new_tokens):
            if input_ids.size(1) > MAX_SEQ_LEN:
                curr_input = input_ids[:, -MAX_SEQ_LEN:]
            else:
                curr_input = input_ids
                
            logits = model(curr_input)
            next_logits = logits[0, -1, :].clone()
            
            # Apply Repetition Penalty to prevent repetitive loops (area_area_area / 000000)
            if repetition_penalty != 1.0:
                for pid in set(generated_pruned_ids):
                    if pid < next_logits.size(0):
                        if next_logits[pid] < 0:
                            next_logits[pid] *= repetition_penalty
                        else:
                            next_logits[pid] /= repetition_penalty

            next_logits = next_logits / max(temperature, 1e-5)
            
            # 1. Compute raw softmax probabilities and EOS rank BEFORE top_k filtering
            raw_probs = torch.softmax(next_logits, dim=-1)
            eos_prob = raw_probs[eos_pruned_id].item() if eos_pruned_id is not None and eos_pruned_id < raw_probs.size(0) else 0.0
            
            # Find EOS rank in raw logits
            _, sorted_indices = torch.sort(next_logits, descending=True)
            eos_rank = (sorted_indices == eos_pruned_id).nonzero(as_tuple=True)[0].item() + 1 if eos_pruned_id is not None else 99999
            
            # 2. Apply Top-K sampling
            if top_k > 0:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[-1]] = -float('Inf')
                
            probs = torch.softmax(next_logits, dim=-1)
            next_pruned_token = torch.multinomial(probs, num_samples=1).item()
            
            if next_pruned_token == 0:
                generated_oov_count += 1
                
            if eos_pruned_id is not None and next_pruned_token == eos_pruned_id:
                break
                
            next_teacher_token = pruned_id_to_teacher_id[next_pruned_token].item()
            if next_teacher_token == eos_teacher_id:
                break
                
            generated_pruned_ids.append(next_pruned_token)
            input_ids = torch.cat([input_ids, torch.tensor([[next_pruned_token]], device=device)], dim=1)

    generated_teacher_ids = [pruned_id_to_teacher_id[pid].item() for pid in generated_pruned_ids]
    decoded_str = teacher_tokenizer.decode(generated_teacher_ids, skip_special_tokens=True)
    stop_reason = "EOS_TOKEN_STOP" if (len(generated_pruned_ids) - len(pruned_input) < max_new_tokens) else "MAX_TOKENS_LIMIT"
    debug_str = f"DEBUG    : stop_reason={stop_reason} | last_eos_prob={eos_prob*100:.3f}% (rank #{eos_rank}) | prompt_oov={prompt_oov}/{len(teacher_enc)} | generated_oov={generated_oov_count}/{len(generated_pruned_ids)-len(pruned_input)}"
    return decoded_str, debug_str

def main():
    print("=================================================================", flush=True)
    print("  WRAI v14.4 INFERENCE EVALUATOR (DISTILLED MODEL EVALUATION)   ", flush=True)
    print("=================================================================\n", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Inference Device: {device}", flush=True)

    # 1. Load Vocab Map
    candidate_vocab_map = [
        os.path.join(DRIVE_SAVE_DIR, VOCAB_MAP_FILENAME),
        VOCAB_MAP_FILENAME
    ]
    target_vocab = None
    for p in candidate_vocab_map:
        if os.path.exists(p):
            target_vocab = p
            break
            
    if not target_vocab:
        print(f"[ERROR] Pruned Vocab Map file ({VOCAB_MAP_FILENAME}) not found!", flush=True)
        return

    print(f"[*] Loading Pruned Vocab Map from: {target_vocab}...", flush=True)
    with open(target_vocab, "r", encoding="utf-8") as f:
        saved_map = json.load(f)

    teacher_id_to_pruned_id = {int(k): v for k, v in saved_map["teacher_id_to_pruned_id"].items()}
    pruned_id_to_teacher_id = torch.tensor(saved_map["pruned_id_to_teacher_id"], dtype=torch.long)
    print(f"[OK 100% SUCCESS] Vocab Map Loaded (Pruned Vocab Size: {len(pruned_id_to_teacher_id):,})", flush=True)

    # 2. Load Teacher Tokenizer
    print(f"[*] Loading Teacher Tokenizer ({TEACHER_MODEL_NAME})...", flush=True)
    teacher_tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_NAME)
    print(f"[OK] Teacher Tokenizer Loaded!", flush=True)

    # 3. Load Student Model
    model = WRAIv14_4(vocab_size=PRUNED_VOCAB_SIZE, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, wavelet_levels=WAVELET_LEVELS).to(device)

    candidate_pt = [
        os.path.join(DRIVE_SAVE_DIR, "wrai_v14_4_latest.pt"),
        os.path.join(DRIVE_SAVE_DIR, "wrai_v14_4_best.pt"),
        "wrai_v14_4_latest.pt",
        "wrai_v14_4_best.pt"
    ]
    
    target_pt = None
    for p in candidate_pt:
        if os.path.exists(p):
            target_pt = p
            break

    if not target_pt:
        print("[ERROR] Checkpoint .pt file not found!", flush=True)
        return

    print(f"[*] Loading Checkpoint Weights from: {target_pt}...", flush=True)
    ckpt = torch.load(target_pt, map_location=device)
    model.load_state_dict(ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt)
    epoch = ckpt.get("epoch", "Unknown") if isinstance(ckpt, dict) else "Unknown"
    val_loss = ckpt.get("best_val_loss", "Unknown") if isinstance(ckpt, dict) else "Unknown"
    print(f"[OK 100% SUCCESS] Model Loaded! Checkpoint Epoch: {epoch} | Val Loss: {val_loss}\n", flush=True)

    # 4. Multi-Domain Evaluation Prompts (ID, EN, PY, MATH, AGENT)
    test_prompts = [
        ("<ID>", "<ID> Halo, jelaskan apa fungsi dari AI"),
        ("<EN>", "<EN> Artificial Intelligence is defined as"),
        ("<PY>", "<PY> def calculate_area(radius):"),
        ("<MATH>", "<MATH> If 2x + 6 = 14, then x ="),
        ("<AGENT>", "<AGENT> Action: weather_api location=\"Jakarta\" days=3"),
        ("<ID>", "<ID> Indonesia adalah negara kepulauan yang"),
    ]

    print("=================================================================", flush=True)
    print("            JUJUR EVALUATION RESULTS OF WRAI v14.4               ", flush=True)
    print("=================================================================\n", flush=True)

    for domain, prompt in test_prompts:
        output_text, debug_str = generate_text_v14_4(model, teacher_tokenizer, teacher_id_to_pruned_id, pruned_id_to_teacher_id,
                                                     prompt, max_new_tokens=90, temperature=0.65, top_k=35, repetition_penalty=1.2, device=device)
        print(f"PROMPT [{domain}]: {prompt}")
        print(f"OUTPUT   : '{output_text}'")
        print(f"{debug_str}\n" + "-"*65)

if __name__ == "__main__":
    main()
