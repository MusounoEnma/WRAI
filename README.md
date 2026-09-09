# 🌊 WRAI: Wavelet Retention Artificial Intelligence
> **Sub-Quadratic Causal Sequence Architecture with 0% KV-Cache & Pure Native C Inference Engine**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![C Standard](https://img.shields.io/badge/C-C99-00599C?logo=c)](src/wrai_v16_engine.c)
[![Architecture](https://img.shields.io/badge/Architecture-RetNet%20%2B%20Haar%20DWT-ff69b4)](docs/)
[![Memory Complexity](https://img.shields.io/badge/Memory-O(1)%20Constant-brightgreen)](include/)
[![Hardware Target](https://img.shields.io/badge/Optimized-AMD%20AVX%201.0%20%26%20Legacy%20x86-orange)](src/)

---

## 🌟 Overview

**WRAI (Wavelet Retention AI)** adalah arsitektur model bahasa generasi baru yang mendobrak keterbatasan Transformer konvensional. Menggabungkan **Multi-Head Retention (RetNet)** dengan **Filter Spektral Haar DWT 1D** dan **SwiGLU FFN**, WRAI menghadirkan inferensi berkecepatan tinggi dengan penggunaan memori konstan $O(1)$ tanpa ketergantungan pada KV-Cache yang membengkak.

Flagship rilis saat ini adalah **WRAI v16 (1.7B)** — model ~1.84 Miliar parameter yang ditransplantasikan secara presisi 1:1 dari `Qwen/Qwen3-1.7B` dan dilatih menggunakan dataset dwibahasa murni (Bahasa Indonesia, English Reasoning, dan Python Engineering).

---

## 💎 Fitur Utama WRAI v16 (1.7B)

1. **🚀 0% KV-Cache ($O(1)$ Constant Memory):**
   * Berbeda dengan Transformer biasa yang menyedot RAM bergiga-giga saat percakapan memanjang, WRAI v16 mengunci seluruh memori konteks dalam recurrent state matrix $S \in \mathbb{R}^{128 \times 128}$ per head.
   * **Total memori konteks hanya ~28 MB RAM tetap**, tidak akan pernah membengkak atau memicu Out-of-Memory (OOM)!
2. **⚡ Pure Native C Inference Engine (`wrai_v16.exe` ~315 KB):**
   * Inferensi murni tanpa Python, tanpa PyTorch, tanpa CUDA, dan tanpa ketergantungan library eksternal.
   * Kernel SIMD **AVX 1.0 32-vector unrolled** yang dirancang khusus untuk berjalan kencang di hardware minimalis (teruji di prosesor laptop lawas AMD A8 Puma+ 2014 RAM 8GB).
3. **🧠 Kecerdasan Dwibahasa & Coding:**
   * Memahami Bahasa Indonesia secara natural, penalaran teknis, dan pembuatan kode Python yang akurat.
4. **📦 Zero-Heap Virtual Memory Mapping:**
   * Binary model INT8 (~1.84 GB) dimap secara virtual (*mmap*), startup instan dalam 0.3 detik.

---

## 📊 Komparasi: WRAI v16 vs Transformer Konvensional

| Metrik Performa | Transformer Standard (Qwen di llama.cpp/PyTorch) | WRAI v16 (Native C Engine) |
| :--- | :--- | :--- |
| **Konsumsi RAM Konteks (KV-Cache)** | Membengkak Drastis ($O(N)$):<br>• 1.000 token $\to$ +115 MB<br>• 8.000 token $\to$ **+1.0 GB RAM**<br>• 32.000 token $\to$ **+3.7 GB RAM (OOM)** | **Terkunci Konstan ($O(1)$):**<br>• 10 token $\to$ **28 MB**<br>• 1.000 token $\to$ **28 MB**<br>• 100.000 token $\to$ **Tetap 28 MB!** |
| **Degradasi Latensi Konteks Panjang** | Semakin panjang chat, generasi makin melambat | **Stabil Konstan**, kecepatan per token tidak pernah drop |
| **Ketergantungan Software** | PyTorch (~6-8 GB instalasi) atau runtime GGML | **1 File Binary Mandiri (`wrai_v16.exe` 315 KB)** |
| **Waktu Cold Boot / Startup** | 5 – 15 detik | **~0.3 detik** (Memory-Mapped I/O) |

---

## 🏛️ Arsitektur WRAI v16

```
                    [ Input Token ]
                          │
                  Embedding Matrix (151936 x 2048)
                          │
             ┌────────────┴────────────┐
             │   28x Retention Blocks   │
             │                         │
             │   1. RMSNorm            │
             │   2. Multi-Head RetNet  │ (16 Heads, D=128, Per-Token GroupNorm)
             │   3. Residual Add       │
             │   4. RMSNorm            │
             │   5. SwiGLU FFN (6144)  │
             │   6. Residual Add       │
             │   7. Haar DWT Filter    │ (Layer 0 & Layer 13 Mid-Stabilizer)
             └────────────┬────────────┘
                          │
                    Final RMSNorm
                          │
             LM Head (Tied Embedding 151936)
                          │
                  [ Next Token Logits ]
```

---

## 🚀 Panduan Memulai Cepat (Quick Start)

### 1. Download Bobot Model & Vocabulary
Letakkan file binary di folder `1.7b models/`:
* `1.7b models/wrai_v16_1.7b_int8.bin` (~1.84 GB)
* `1.7b models/wrai_v16_vocab.bin` (~1.67 MB)

*(File bobot model dapat diunduh melalui Hugging Face Hub / Google Drive rilis resmi).*

### 2. Kompilasi & Jalankan di Windows (AMD / Intel CPU)
Kompilasi satu klik menggunakan GCC MinGW-w64:
```cmd
build_wrai_v16.bat
```

Jalankan chat shell interaktif:
```cmd
wrai_v16.exe
```

---

## 📂 Struktur Direktori

```
WRAI/
├── final/                     # Skrip produksi, training Colab, & evaluasi
│   ├── colab_train_wrai_v16_qwen3_1.7b.py # Trainer 1:1 Qwen 3 Transplant
│   ├── quantize_wrai_v16_colab.py        # INT8 / INT4 Quantizer & Packer
│   ├── test_wrai_v16_qwen3_1.7b_inference.py # Python bilingual inference
│   └── README.md              # Dokumentasi teknis eksperimen Colab
├── include/                   # Header C Native Engine
│   └── wrai_v16_engine.h      # Arsitektur struct, SIMD math, & tokenizer header
├── src/                       # Source Code C Native
│   ├── wrai_v16_engine.c      # AVX 1.0 32-vector GEMV & Recurrent RetNet kernel
│   └── wrai_v16_cli.c         # Interactive Chat Terminal Shell
├── tests/                     # Microbenchmarks & unit tests
│   ├── bench_gemv.c           # AVX GEMV latency benchmark
│   └── test_interactive_prompts.c # Multi-prompt end-to-end verification
├── build_wrai_v16.bat         # 1-Click compiler script untuk Windows
└── .gitignore                 # Exclude large models & binaries
```

---

## 📜 Lisensi & Atribusi
Proyek ini dilisensikan di bawah **MIT License**.  
Bobot dasar ditransplantasikan dari arsitektur Qwen 3 (`Qwen/Qwen3-1.7B`) dengan penghormatan penuh terhadap lisensi upstream Qwen.
