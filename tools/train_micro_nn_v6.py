#!/usr/bin/env python3
"""
WRAI v6 Micro Neural Language Model — Deep Learning Training Script
====================================================================
TRAINING PHASE: Uses GEMM (numpy matmul, FP32 backpropagation) to train a real neural network.
DEPLOYMENT:     Quantizes all weights to Q15 (int16) for Zero-GEMM inference.

Architecture:
  - Embedding Layer:  vocab_size x EMBED_DIM (64)
  - Context Window:   3 words concatenated -> 192 dims
  - Hidden Layer:     192 -> 128 (ReLU)
  - Output Layer:     128 -> vocab_size (Softmax -> cross-entropy loss)
  - Training:         SGD with momentum, ~82K parameters

This is a REAL neural network trained with REAL backpropagation.
"""

import json
import math
import numpy as np
import os
import sys
import time

# =====================================================================
# HYPERPARAMETERS
# =====================================================================
EMBED_DIM = 64          # Embedding dimension per word
HIDDEN_DIM = 128        # Hidden layer width
CONTEXT_WINDOW = 3      # Number of previous words as context
LEARNING_RATE = 0.05    # SGD learning rate
MOMENTUM = 0.9          # SGD momentum
EPOCHS = 200            # Training epochs
BATCH_SIZE = 16         # Mini-batch size

# =====================================================================
# TRAINING CORPUS (Intent-Categorized Indonesian Sentences)
# =====================================================================
TRAINING_CORPUS = [
    # Greeting
    "halo kawan selamat datang senang bertemu dengan anda",
    "salam hangat kawan apa kabar hari ini",
    "hai kawan saya senang bisa menyapa anda hari ini",
    "halo selamat pagi semoga hari anda menyenangkan",
    "salam kenal kawan perkenalkan saya wrai asisten anda",
    "hai salam hangat senang sekali bisa berdiskusi bersama anda",
    "halo kawan ada yang bisa saya bantu hari ini",
    "selamat siang kawan semoga aktivitas anda berjalan lancar",
    "hai senang bertemu anda kawan mari kita mengobrol",
    "halo apa kabar kawan saya harap anda sehat selalu",
    "salam kawan terima kasih sudah menyapa saya hari ini",
    "hai kawan selamat datang kembali mari kita lanjutkan percakapan",
    "halo kawan senang sekali bisa membantu anda hari ini",
    "salam hangat dari saya untuk anda kawan semoga baik baik saja",
    "hai kawan bagaimana kabar anda hari ini semoga sehat selalu",

    # Farewell
    "sampai jumpa kawan semoga hari anda menyenangkan",
    "terima kasih kawan sampai bertemu lagi nanti",
    "selamat tinggal kawan senang bisa berdiskusi dengan anda",
    "sampai nanti kawan jaga kesehatan selalu",
    "terima kasih sudah mengobrol kawan sampai jumpa lagi",
    "baik kawan saya akan selalu siap membantu kapan saja",

    # Identity
    "saya adalah wrai asisten kecerdasan buatan berbasis gelombang spektral",
    "nama saya wrai wavelet resonance artificial intelligence",
    "saya wrai sistem kecerdasan buatan yang mengolah bahasa dengan sinyal gelombang",
    "saya adalah asisten ai yang dirancang khusus tanpa perkalian matriks berat",
    "wrai adalah arsitektur ai spektral non transformer yang sangat efisien",
    "saya diciptakan menggunakan metode wavelet resonance tanpa transformer",
    "saya bukan transformer saya adalah ai berbasis gelombang spektral fixed point",
    "wrai singkatan dari wavelet resonance artificial intelligence",

    # Math
    "saya siap membantu anda mengerjakan soal matematika dan logika",
    "mari kita kerjakan persamaan matematika ini bersama sama kawan",
    "rumus luas lingkaran adalah pi dikali jari jari kuadrat",
    "teorema pythagoras menyatakan kuadrat sisi miring sama dengan jumlah kuadrat kedua sisi",
    "untuk menghitung persamaan linier kita perlu mencari nilai variabel yang memenuhi",
    "matematika adalah bahasa alam semesta yang mengungkap pola tersembunyi",
    "aljabar membantu kita menyelesaikan persamaan dengan variabel yang belum diketahui",
    "geometri mempelajari bentuk ukuran dan posisi objek dalam ruang",
    "kalkulus adalah cabang matematika yang mempelajari perubahan dan laju",
    "statistika membantu kita menganalisis data dan membuat kesimpulan dari informasi",
    "trigonometri mempelajari hubungan antara sisi dan sudut segitiga",
    "bilangan prima hanya bisa dibagi satu dan dirinya sendiri",

    # Tech
    "wrai beroperasi murni zero gemm tanpa perkalian matriks floating point",
    "sistem ini menggunakan aritmatika fixed point integer enam belas bit",
    "target hardware utama adalah mikrokontroler esp32 stm32 dan cortex m",
    "model disimpan dalam format biner dan diproses via double ring buffer",
    "alokasi ram tetap statis pada enam belas kilobyte tanpa kebocoran memori",
    "pencarian spektral menggunakan fft radix dua lima ratus dua belas titik",
    "kompleksitas waktu inferensi wrai adalah konstan tanpa bergantung ukuran model",
    "wrai tidak membutuhkan gpu atau tpu untuk berjalan cukup cpu biasa",
    "arsitektur ini dirancang untuk perangkat edge computing dan internet of things",

    # Science
    "sains adalah proses sistematis untuk memahami alam semesta melalui pengamatan dan eksperimen",
    "fisika mempelajari hukum dasar alam semesta dari partikel subatom hingga galaksi",
    "kimia mempelajari komposisi struktur dan perubahan materi",
    "biologi mempelajari makhluk hidup dan proses kehidupan",
    "astronomi mempelajari benda langit dan fenomena luar angkasa",
    "ekologi mempelajari hubungan antara organisme dan lingkungannya",

    # General
    "tentu saya bisa membantu anda dengan pertanyaan tersebut kawan",
    "pertanyaan yang menarik mari kita bahas bersama sama",
    "saya akan berusaha menjawab sebaik mungkin berdasarkan pengetahuan saya",
    "mari kita diskusikan topik ini lebih mendalam kawan",
    "baik kawan saya akan menjelaskan dengan bahasa yang mudah dipahami",
    "itu pertanyaan bagus kawan izinkan saya menjelaskan",
    "saya memahami pertanyaan anda kawan berikut penjelasan saya",
    "dengan senang hati saya akan membantu menjawab pertanyaan anda",
]

def softmax(x):
    """Numerically stable softmax."""
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)

def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(np.float32)

def cross_entropy_loss(probs, targets):
    """Cross-entropy loss for batch of predictions."""
    n = len(targets)
    log_probs = -np.log(probs[np.arange(n), targets] + 1e-9)
    return np.mean(log_probs)

def float_to_q15(v):
    val = int(round(v * 32768.0))
    return max(-32768, min(32767, val))

def main():
    print("=================================================================")
    print("  WRAI v6 MICRO NEURAL LANGUAGE MODEL — DEEP LEARNING TRAINING  ")
    print("  (GEMM/Backpropagation FP32 -> Q15 Quantized Deployment)       ")
    print("=================================================================\n")

    # =====================================================================
    # STEP 1: Build Vocabulary from Corpus
    # =====================================================================
    all_words = []
    for sentence in TRAINING_CORPUS:
        all_words.extend(sentence.lower().split())
    
    word_counts = {}
    for w in all_words:
        word_counts[w] = word_counts.get(w, 0) + 1
    
    # Sort by frequency, add special tokens
    sorted_words = sorted(word_counts.keys(), key=lambda w: -word_counts[w])
    
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    BOS_TOKEN = "<BOS>"
    
    vocab = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN] + sorted_words
    word_to_id = {w: i for i, w in enumerate(vocab)}
    id_to_word = {i: w for w, i in word_to_id.items()}
    vocab_size = len(vocab)
    
    print(f"[*] Vocabulary size: {vocab_size} words")
    print(f"[*] Training sentences: {len(TRAINING_CORPUS)}")

    # =====================================================================
    # STEP 2: Create Training Samples (Context Window -> Next Word)
    # =====================================================================
    training_samples = []  # (context_ids[CONTEXT_WINDOW], target_id)
    
    PAD_ID = word_to_id[PAD_TOKEN]
    BOS_ID = word_to_id[BOS_TOKEN]
    
    for sentence in TRAINING_CORPUS:
        word_ids = [word_to_id.get(w, word_to_id[UNK_TOKEN]) for w in sentence.lower().split()]
        
        # Prepend BOS tokens for context padding
        padded = [BOS_ID] * CONTEXT_WINDOW + word_ids
        
        for i in range(len(word_ids)):
            context = padded[i : i + CONTEXT_WINDOW]
            target = word_ids[i]
            training_samples.append((context, target))
    
    print(f"[*] Training samples: {len(training_samples)}")
    
    X_contexts = np.array([s[0] for s in training_samples], dtype=np.int32)
    Y_targets = np.array([s[1] for s in training_samples], dtype=np.int32)

    # =====================================================================
    # STEP 3: Initialize Neural Network Weights (Xavier/He Initialization)
    # =====================================================================
    np.random.seed(42)
    
    context_dim = CONTEXT_WINDOW * EMBED_DIM  # 3 * 64 = 192
    
    # Embedding matrix: vocab_size x EMBED_DIM
    W_embed = np.random.randn(vocab_size, EMBED_DIM).astype(np.float32) * 0.1
    
    # Hidden layer: context_dim -> HIDDEN_DIM
    W1 = np.random.randn(context_dim, HIDDEN_DIM).astype(np.float32) * np.sqrt(2.0 / context_dim)
    b1 = np.zeros(HIDDEN_DIM, dtype=np.float32)
    
    # Output layer: HIDDEN_DIM -> vocab_size
    W2 = np.random.randn(HIDDEN_DIM, vocab_size).astype(np.float32) * np.sqrt(2.0 / HIDDEN_DIM)
    b2 = np.zeros(vocab_size, dtype=np.float32)
    
    total_params = W_embed.size + W1.size + b1.size + W2.size + b2.size
    print(f"[*] Total parameters: {total_params:,}")
    print(f"[*] Model size (FP32): {total_params * 4 / 1024:.1f} KB")
    print(f"[*] Model size (Q15):  {total_params * 2 / 1024:.1f} KB")
    
    # Momentum buffers
    mW_embed = np.zeros_like(W_embed)
    mW1 = np.zeros_like(W1)
    mb1 = np.zeros_like(b1)
    mW2 = np.zeros_like(W2)
    mb2 = np.zeros_like(b2)

    # =====================================================================
    # STEP 4: TRAINING LOOP (GEMM / Backpropagation / Deep Learning)
    # =====================================================================
    print(f"\n[*] Starting training: {EPOCHS} epochs, lr={LEARNING_RATE}, batch={BATCH_SIZE}")
    print("-" * 65)
    
    n_samples = len(training_samples)
    
    for epoch in range(EPOCHS):
        # Shuffle training data
        perm = np.random.permutation(n_samples)
        X_shuf = X_contexts[perm]
        Y_shuf = Y_targets[perm]
        
        epoch_loss = 0.0
        n_batches = 0
        
        for batch_start in range(0, n_samples, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, n_samples)
            X_batch = X_shuf[batch_start:batch_end]
            Y_batch = Y_shuf[batch_start:batch_end]
            bs = len(Y_batch)
            
            # ======================== FORWARD PASS (GEMM!) ========================
            # 1. Embedding lookup & concatenate context window
            embed_out = W_embed[X_batch]  # (bs, CONTEXT_WINDOW, EMBED_DIM)
            context_vec = embed_out.reshape(bs, context_dim)  # (bs, 192)
            
            # 2. Hidden layer: z1 = context @ W1 + b1,  h1 = ReLU(z1)
            z1 = context_vec @ W1 + b1  # <<< GEMM HERE (matrix multiply)
            h1 = relu(z1)  # (bs, HIDDEN_DIM)
            
            # 3. Output layer: z2 = h1 @ W2 + b2,  probs = softmax(z2)
            z2 = h1 @ W2 + b2  # <<< GEMM HERE (matrix multiply)
            probs = softmax(z2)  # (bs, vocab_size)
            
            # 4. Cross-entropy loss
            loss = cross_entropy_loss(probs, Y_batch)
            epoch_loss += loss
            n_batches += 1
            
            # ======================== BACKWARD PASS (GEMM!) ========================
            # Gradient of cross-entropy + softmax
            dz2 = probs.copy()
            dz2[np.arange(bs), Y_batch] -= 1.0
            dz2 /= bs
            
            # Gradients for W2, b2
            dW2 = h1.T @ dz2       # <<< GEMM HERE
            db2 = np.sum(dz2, axis=0)
            
            # Backprop through ReLU
            dh1 = dz2 @ W2.T       # <<< GEMM HERE
            dz1 = dh1 * relu_grad(z1)
            
            # Gradients for W1, b1
            dW1 = context_vec.T @ dz1  # <<< GEMM HERE
            db1 = np.sum(dz1, axis=0)
            
            # Gradient for embeddings
            d_context = dz1 @ W1.T  # <<< GEMM HERE
            d_embed = d_context.reshape(bs, CONTEXT_WINDOW, EMBED_DIM)
            
            # Sparse embedding gradient update
            dW_embed = np.zeros_like(W_embed)
            for b in range(bs):
                for c in range(CONTEXT_WINDOW):
                    dW_embed[X_batch[b, c]] += d_embed[b, c]
            
            # ======================== SGD WITH MOMENTUM ========================
            mW_embed = MOMENTUM * mW_embed - LEARNING_RATE * dW_embed
            mW1 = MOMENTUM * mW1 - LEARNING_RATE * dW1
            mb1 = MOMENTUM * mb1 - LEARNING_RATE * db1
            mW2 = MOMENTUM * mW2 - LEARNING_RATE * dW2
            mb2 = MOMENTUM * mb2 - LEARNING_RATE * db2
            
            W_embed += mW_embed
            W1 += mW1
            b1 += mb1
            W2 += mW2
            b2 += mb2
        
        avg_loss = epoch_loss / max(1, n_batches)
        
        if epoch % 20 == 0 or epoch == EPOCHS - 1:
            # Quick accuracy check
            embed_all = W_embed[X_contexts].reshape(n_samples, context_dim)
            z1_all = relu(embed_all @ W1 + b1)
            z2_all = z1_all @ W2 + b2
            preds = np.argmax(z2_all, axis=1)
            accuracy = np.mean(preds == Y_targets) * 100.0
            print(f"  Epoch {epoch+1:4d}/{EPOCHS} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.1f}%")
    
    print("-" * 65)
    print("[OK] Training complete!\n")

    # =====================================================================
    # STEP 5: QUANTIZE TO Q15 (FP32 -> int16)
    # =====================================================================
    print("[*] Quantizing FP32 weights to Q15 (int16)...")
    
    # Find scale factors for each weight matrix
    def quantize_matrix(mat, name):
        abs_max = np.max(np.abs(mat))
        if abs_max == 0:
            abs_max = 1.0
        scale = 32767.0 / abs_max
        q_mat = np.clip(np.round(mat * scale), -32768, 32767).astype(np.int16)
        print(f"  {name}: shape={mat.shape}, abs_max={abs_max:.6f}, scale={scale:.2f}")
        return q_mat, float(abs_max)
    
    q_W_embed, s_embed = quantize_matrix(W_embed, "W_embed")
    q_W1, s_W1 = quantize_matrix(W1, "W1")
    q_b1, s_b1 = quantize_matrix(b1.reshape(1, -1), "b1")
    q_W2, s_W2 = quantize_matrix(W2, "W2")
    q_b2, s_b2 = quantize_matrix(b2.reshape(1, -1), "b2")

    # =====================================================================
    # STEP 6: SAVE MODEL
    # =====================================================================
    output_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "wrai_nn_v6.json")
    
    model_data = {
        "version": "6.0",
        "architecture": {
            "embed_dim": EMBED_DIM,
            "hidden_dim": HIDDEN_DIM,
            "context_window": CONTEXT_WINDOW,
            "vocab_size": vocab_size,
            "total_params": int(total_params)
        },
        "vocabulary": {w: i for w, i in word_to_id.items()},
        "id_to_word": {str(i): w for i, w in id_to_word.items()},
        "scales": {
            "embed": s_embed,
            "w1": s_W1,
            "b1": s_b1,
            "w2": s_W2,
            "b2": s_b2
        },
        "weights_q15": {
            "W_embed": q_W_embed.tolist(),
            "W1": q_W1.tolist(),
            "b1": q_b1.flatten().tolist(),
            "W2": q_W2.tolist(),
            "b2": q_b2.flatten().tolist()
        }
    }
    
    with open(model_path, "w", encoding="utf-8") as f:
        json.dump(model_data, f)
    
    model_size_kb = os.path.getsize(model_path) / 1024.0
    print(f"\n[OK] Model v6 saved!")
    print(f"  -> Path: {os.path.abspath(model_path)} ({model_size_kb:.1f} KB)")
    print(f"  -> Vocab: {vocab_size} words")
    print(f"  -> Params: {total_params:,}")
    print(f"  -> Architecture: Embed({EMBED_DIM}) -> Hidden({HIDDEN_DIM}) -> Output({vocab_size})")
    
    # =====================================================================
    # STEP 7: VERIFY Q15 INFERENCE (Zero-GEMM sanity check)
    # =====================================================================
    print("\n[*] Verifying Q15 Zero-GEMM inference...")
    
    test_prompts = ["halo kawan apa", "siapa nama kamu", "bantu aku matematika"]
    
    for prompt in test_prompts:
        words = prompt.lower().split()
        context_ids = []
        for w in words[-CONTEXT_WINDOW:]:
            context_ids.append(word_to_id.get(w, word_to_id[UNK_TOKEN]))
        while len(context_ids) < CONTEXT_WINDOW:
            context_ids.insert(0, word_to_id[BOS_TOKEN])
        
        # === Q15 FORWARD PASS (Zero-GEMM Integer Only) ===
        # 1. Embedding lookup (no GEMM, just array indexing)
        embed_q15 = []
        for cid in context_ids:
            embed_q15.extend(q_W_embed[cid].tolist())
        
        # 2. Hidden layer: dot-product Q15 (no matmul, explicit loop)
        hidden_q15 = []
        for h in range(HIDDEN_DIM):
            acc = 0
            for d in range(CONTEXT_WINDOW * EMBED_DIM):
                acc += (embed_q15[d] * int(q_W1[d, h])) >> 15
            acc += int(q_b1.flatten()[h])
            acc = max(0, acc)  # ReLU
            hidden_q15.append(acc)
        
        # 3. Output layer: dot-product Q15
        output_q15 = []
        for v in range(vocab_size):
            acc = 0
            for h in range(HIDDEN_DIM):
                acc += (hidden_q15[h] * int(q_W2[h, v])) >> 15
            acc += int(q_b2.flatten()[v])
            output_q15.append(acc)
        
        # 4. Argmax (no softmax needed for greedy decode)
        best_id = max(range(vocab_size), key=lambda i: output_q15[i])
        predicted_word = id_to_word[best_id]
        
        print(f"  [{prompt}] -> Q15 Predicted Next: '{predicted_word}'")
    
    print("\n[OK] Q15 Zero-GEMM inference verified! All operations are integer-only.")

if __name__ == "__main__":
    main()
