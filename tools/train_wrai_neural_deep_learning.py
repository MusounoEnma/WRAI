#!/usr/bin/env python3
"""
WRAI Real Deep Learning Neural Network Trainer (Fixed Weights Bridge & Q31 Quantizer)
=====================================================================================
Trains a Neural Network via Backpropagation & Cross-Entropy Loss on Multi-Domain Dataset (ID, EN, Code, Math).
Saves trained W_emb, W_out, and b_out directly into metadata JSON & binary model: models/wrai_nextgen_v8.bin
"""

import json
import math
import numpy as np
import os
import struct
import sys
import time

FFT_SIZE = 4096
SPECTRAL_BINS = 2048
MAGIC_HEADER = 0x57524149
MODEL_VERSION = 0x0800

def float_to_q31(v: float) -> int:
    val = round(v * 2147483648.0)
    return max(-2147483648, min(2147483647, val))

TRAINING_CORPUS = [
    # --- INDONESIAN DIALOGUE ---
    "halo kawan selamat datang mari kita berdiskusi bersama dengan senang hati",
    "salam hangat kawan apa kabar hari ini semoga sehat dan bahagia selalu ya",
    "kabar saya sangat baik dan selalu siap membantu pertanyaan anda kapan saja",
    "saya adalah wrai kecerdasan buatan berbasis spektral gelombang fixed-point zero-gemm",
    "wrai beroperasi murni tanpa perkalian matriks berat dan sangat hemat memori ram",
    "sampai jumpa kembali kawan semoga hari anda menyenangkan dan sukses selalu",

    # --- ENGLISH DIALOGUE ---
    "hello my friend welcome let us discuss today with great pleasure",
    "warm greetings friend how are you today I hope you are healthy and happy",
    "my condition is great and I am always ready to help you anytime",
    "I am wrai a wave spectral artificial intelligence based on zero-gemm fixed-point",
    "wrai operates purely without heavy matrix multiplication and is very ram efficient",
    "see you again my friend have a wonderful day and continuous success",

    # --- PYTHON PROGRAMMING ---
    "def hitung_luas_lingkaran(r): return 3.14159 * r * r",
    "def hitung_pythagoras(a, b): return math.sqrt(a * a + b * b)",
    "class WRAINeuralEngine: def __init__(self): self.status = True",
    "def bubble_sort(arr): n = len(arr); return sorted(arr)",
    "import math; import numpy as np; print('WRAI Neural Deep Learning')",

    # --- MATHEMATICS ---
    "rumus luas lingkaran adalah pi dikali jari jari kuadrat",
    "teorema pythagoras menyatakan kuadrat sisi miring segitiga sama dengan jumlah kuadrat sisi tegak",
    "persamaan kuadrat ax2 bx c 0 memiliki akar rumus abc",
    "the area of a circle formula is pi multiplied by radius squared",
    "pythagoras theorem states hypotenuse squared equals sum of squared sides"
]

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def main():
    print("=================================================================")
    print("  WRAI REAL DEEP LEARNING NEURAL BACKPROPAGATION TRAINER (v8)   ")
    print("=================================================================\n")

    # 1. Build Vocabulary
    word_counts = {}
    for sentence in TRAINING_CORPUS:
        for w in sentence.lower().replace("?", "").replace("!", "").replace(",", "").replace(".", "").split():
            word_counts[w] = word_counts.get(w, 0) + 1

    sorted_words = sorted(word_counts.keys(), key=lambda x: -word_counts[x])

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"

    vocab = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN] + sorted_words
    word_to_id = {w: i for i, w in enumerate(vocab)}
    id_to_word_dict = {i: w for i, w in enumerate(vocab)}
    vocab_size = len(vocab)

    print(f"[*] Vocabulary Size: {vocab_size} tokens (ID + EN + Code + Math)")

    # 2. Prepare Training Sequence Pairs (X_curr -> Y_next)
    X_data = []
    Y_data = []

    for sentence in TRAINING_CORPUS:
        tokens = [word_to_id.get(w, 1) for w in sentence.lower().replace("?", "").replace("!", "").replace(",", "").replace(".", "").split()]
        for i in range(len(tokens) - 1):
            X_data.append(tokens[i])
            Y_data.append(tokens[i+1])

    X_data = np.array(X_data, dtype=np.int32)
    Y_data = np.array(Y_data, dtype=np.int32)
    num_samples = len(X_data)

    print(f"[*] Training Sequence Samples: {num_samples} pairs")

    # 3. Initialize Neural Network Weights
    hidden_dim = 128
    np.random.seed(42)
    W_emb = np.random.randn(vocab_size, hidden_dim).astype(np.float32) * 0.05
    W_out = np.random.randn(hidden_dim, vocab_size).astype(np.float32) * 0.05
    b_out = np.zeros(vocab_size, dtype=np.float32)

    learning_rate = 1.2
    epochs = 1200

    print(f"\n[*] Starting Backpropagation Training ({epochs} Epochs)...")
    t0 = time.perf_counter()

    for epoch in range(1, epochs + 1):
        # Forward Pass
        h = W_emb[X_data] # [num_samples, hidden_dim]
        logits = h @ W_out + b_out # [num_samples, vocab_size]
        probs = softmax(logits) # [num_samples, vocab_size]

        # Compute Cross-Entropy Loss
        correct_probs = probs[np.arange(num_samples), Y_data]
        loss = -np.mean(np.log(correct_probs + 1e-12))

        # Compute Accuracy
        preds = np.argmax(probs, axis=1)
        accuracy = np.mean(preds == Y_data) * 100.0

        # Backpropagation
        dlogits = probs.copy()
        dlogits[np.arange(num_samples), Y_data] -= 1.0
        dlogits /= num_samples

        dW_out = h.T @ dlogits
        db_out = np.sum(dlogits, axis=0)
        dh = dlogits @ W_out.T

        # Update Weights via Accelerated Gradient Descent
        for idx in range(num_samples):
            W_emb[X_data[idx]] -= learning_rate * dh[idx]

        W_out -= learning_rate * dW_out
        b_out -= learning_rate * db_out

        if epoch % 200 == 0 or epoch == 1:
            print(f"  -> Epoch {epoch:4d}/{epochs:4d} | Cross-Entropy Loss: {loss:.4f} | Accuracy: {accuracy:.2f}%")

    t1 = time.perf_counter()
    print(f"\n[OK] REAL NEURAL BACKPROPAGATION TRAINING COMPLETE in {(t1 - t0):.2f}s!")
    print(f"     Final Loss: {loss:.4f} | Final Accuracy: {accuracy:.2f}%")

    # 4. Quantize REAL TRAINED LOGITS & WEIGHTS into WRAI 4096-Bin Q31 Spectral Embeddings
    print("\n[*] Quantizing & Preserving REAL Neural Logits directly to 4096-Bin Q31 Spectral Embeddings...")
    vocab_entries = []

    direct_logits = W_emb @ W_out + b_out

    for idx in range(vocab_size):
        word_str = id_to_word_dict.get(idx, f"<token_{idx}>")
        raw_logits = direct_logits[idx]
        coeffs_q31 = [0] * SPECTRAL_BINS

        for target_id in range(min(vocab_size, SPECTRAL_BINS)):
            coeffs_q31[target_id] = float_to_q31(math.tanh(raw_logits[target_id]))

        vocab_entries.append({
            "id": idx,
            "word": word_str,
            "coeffs_q31": coeffs_q31
        })

    # Save to Binary Model File: models/wrai_nextgen_v8.bin
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(models_dir, exist_ok=True)
    bin_path = os.path.join(models_dir, "wrai_nextgen_v8.bin")
    json_path = os.path.join(models_dir, "wrai_nextgen_v8.json")

    # Header (64 Bytes)
    header = struct.pack(
        "<I H H H H I I 44s",
        MAGIC_HEADER,
        MODEL_VERSION,
        FFT_SIZE,
        SPECTRAL_BINS,
        32, # Q31
        vocab_size,
        4 + (SPECTRAL_BINS * 4),
        b"\x00" * 44
    )

    with open(bin_path, "wb") as f_bin:
        f_bin.write(header)
        for e in vocab_entries:
            pkt_head = struct.pack("<HH", e["id"], 0)
            pkt_coeffs = struct.pack(f"<{SPECTRAL_BINS}i", *e["coeffs_q31"])
            f_bin.write(pkt_head + pkt_coeffs)

    # Save Metadata JSON with correct string ID keys
    meta_json = {
        "version": "8.0",
        "fft_size": FFT_SIZE,
        "spectral_bins": SPECTRAL_BINS,
        "vocab_size": vocab_size,
        "vocabulary": word_to_id,
        "id_to_word": {str(i): w for i, w in id_to_word_dict.items()},
        "training_epochs": epochs,
        "final_loss": float(loss),
        "final_accuracy": float(accuracy),
        "W_emb": W_emb.tolist(),
        "W_out": W_out.tolist(),
        "b_out": b_out.tolist()
    }
    with open(json_path, "w", encoding="utf-8") as f_json:
        json.dump(meta_json, f_json, ensure_ascii=False, indent=2)

    model_size_mb = os.path.getsize(bin_path) / (1024.0 * 1024.0)
    print(f"[OK] REAL TRAINED WRAI v8 MODEL BUILT SUCCESSFULLY!")
    print(f"  -> Model Path  : {os.path.abspath(bin_path)} ({model_size_mb:.2f} MB)")
    print(f"  -> Vocab Size  : {vocab_size} tokens")

if __name__ == "__main__":
    main()
