# 🌊 WRAI: Wavelet Retention Artificial Intelligence
> **A High-Performance Sub-Quadratic Neural Framework with Strictly Constant $O(1)$ State Memory, Zero KV-Cache, and Pure Native C Execution Across PC & Embedded Silicon**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Release: v0.2.0](https://img.shields.io/badge/Release-v0.2.0%20Production-brightgreen.svg)](https://github.com/MusounoEnma/WRAI/releases)
[![C Standard](https://img.shields.io/badge/C-C99%20Native%20Zero--Heap-00599C?logo=c)](qwen/engine/)
[![Context Complexity](https://img.shields.io/badge/Context%20RAM-O(1)%20Strictly%20Constant-success)](#-the-zero-kv-cache-mathematical-foundation)
[![Hardware Targets](https://img.shields.io/badge/Hardware-x86__64%20AVX2%20%7C%20ESP32%20Xtensa-orange)](#-the-two-production-tiers)

---

## 📌 Executive Overview

Conventional Transformer Large Language Models (LLMs) suffer from an inherent memory bottleneck: **Linear KV-Cache Growth ($O(T)$)**. As conversation context expands, storing past key-value vectors consumes tens of gigabytes of RAM/VRAM, causing server latency degradation and making execution on edge microcontrollers virtually impossible.

The **WRAI (Wavelet Retention AI)** project presents a proven, mathematically grounded solution: **Architectural Transmutation (Transplantation)**. Instead of retraining multi-billion parameter networks from scratch, WRAI preserves pre-trained linguistic knowledge while surgically replacing quadratic Softmax Attention with **Dual-State Linear Retention** and **Multiresolution Wavelet Spectral Filtering**.

```
Conventional Transformer (Quadratic & Linear Memory Growth):
  Token (t) ──> Q, K, V ──> Softmax(Q · K^T / √d) · V ──> Must store ALL historical vectors in KV-Cache O(T)

WRAI Architecture (Sub-Quadratic & Strictly Constant O(1) Memory):
  Token (t) ──> Q, K, V ──> S_t = γ · S_{t-1} + K_t^T · V_t ──> Merged in-place into fixed matrix O(1)
```

---

## 🏆 The Two Production Tiers

WRAI is deployed across two distinct, fully reproducible production modules:

| Feature / Metric | 🚀 [WRAI-X (0.8B)](qwen/) | ⚡ [WRAI-Micro (1.25M)](esp32/) |
| :--- | :--- | :--- |
| **Primary Domain** | Desktop, Laptop, & Edge Server CPU | Ultra-Low-Power Embedded IoT Silicon |
| **Base Knowledge Backbone** | `Qwen/Qwen3-0.6B` (100% Frozen FFN) | `SimpleStories-V2-1.25M` (100% Frozen FFN) |
| **Total Measured Parameters** | **831,268,848** (~0.83B parameters) | **1,250,560** (1.25M parameters) |
| **Context Memory Complexity** | **Strictly $O(1)$** (Zero KV-Cache) | **Strictly $O(1)$** (Zero KV-Cache) |
| **Recurrent State Buffer** | **29.42 MB** (28 layers $\times$ 16 heads $\times 128 \times 128$) | **64.0 KB** (4 layers $\times$ 4 heads $\times 32 \times 32$) |
| **Dynamic Heap Delta** | **+0.00 MB** ($T=1$ to $T=32\text{K}$) | **0 bytes** ($T=1$ to continuous chat) |
| **Target Silicon & Runtime** | x86_64 CPU (256-bit AVX SIMD + `mmap`) | ESP32 Xtensa Dual-Core LX6 @ 240 MHz |
| **Hardware Requirements** | Standard Consumer PC (No GPU required) | **$3 ESP32 DevKit (0 MB PSRAM, 4MB Flash)** |
| **Measured Throughput** | ~40 – 50 tok/s (Multi-core x86) | **1.7 – 4.2 tok/s** (Direct from SPI Flash) |
| **Official Module Path** | 👉 [`qwen/`](qwen/) | 👉 [`esp32/`](esp32/) |

---

## 🧠 The Zero KV-Cache Mathematical Foundation

### 1. The Associative Law of Linear Retention
In standard Transformers, the non-linear $\text{Softmax}$ operator prevents decoupling past tokens:
$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

WRAI removes the softmax bottleneck and exploits the **associative property of matrix multiplication**:
$$(Q \cdot K^T) \cdot V \equiv Q \cdot (K^T \cdot V)$$

At every step $t$, the outer product of key and value $K_t^T \cdot V_t$ forms a **fixed-size matrix** ($d_k \times d_k$). This matrix is accumulated recursively into an internal recurrent state buffer $S_t$ with exponential decay $\gamma$:
$$S_t = \gamma \cdot S_{t-1} + K_t^T \cdot V_t$$
$$Y_t = \text{GroupNorm}\left(Q_t \cdot S_t\right)$$

Once $K_t$ and $V_t$ are absorbed into $S_t$, **they are immediately discarded from memory**. Historical context is permanently retained without allocating a single additional byte of RAM.

### 2. 4-Level 1D Discrete Haar Wavelet Transform (DWT)
To prevent semantic drift over long sequences, token representations pass through a multi-resolution Haar Wavelet filter:
* **Low-Frequency Band ($L$)**: Captures global semantics, conceptual continuity, and macro intent.
* **High-Frequency Band ($H$)**: Captures local syntax, micro-features, code tokens, and punctuation.
* **Adaptive Gating**: Smoothly balances global context versus fine-grained syntax before entering the Feed-Forward Network.

---

## 📁 Repository Architecture

The repository is cleanly partitioned into modular, self-contained sub-ecosystems:

```text
WRAI/
├── README.md                          # Master Architecture Overview & Portal (You are here)
├── LICENSE                            # Apache License 2.0
├── .gitignore                         # Strict protection against checkpoints & large binaries
│
├── qwen/                              # 🚀 WRAI-X (0.8B) FOUNDATION MODEL MODULE
│   ├── README.md                      # Detailed technical manual & replication guide
│   ├── engine/                        # Pure Native C Inference Engine (Zero KV-Cache)
│   │   ├── build_wrai_x.bat           # Automated MinGW-w64 build script
│   │   ├── run_wrai_x.bat             # Interactive execution launcher
│   │   ├── src/                       # CLI & memory-mapped C kernel
│   │   ├── include/                   # Architectural C headers
│   │   └── audit/                     # Windows NT psapi.h forensic audit tools
│   ├── training/                      # 1-Click Zero-Contamination Colab Transplant
│   │   ├── colab_train_wrai_x_08b_transplant.py
│   │   ├── WRAI_X_08B_COLAB.ipynb
│   │   └── poc_wrai_x_08b.py
│   ├── quantization/                  # INT8 Symmetric Row-wise Exporter
│   │   └── quantize_wrai_x_08b_colab.py
│   ├── inference/                     # Evaluation, Dialect Tests, & State Audits
│   │   ├── test_wrai_x_08b_english.py
│   │   ├── inspect_wrai_x_drive_model.py
│   │   └── verify_zero_kv_cache_deep_dive.py
│   └── weights/                       # Auto-Downloader from Hugging Face Model Hub
│       └── download_weights.py
│
└── esp32/                             # ⚡ WRAI-MICRO (1.25M) EMBEDDED SILICON MODULE
    ├── README.md                      # Bare-metal microcontroller documentation & logs
    ├── firmware/wrai_micro_esp32/     # Arduino IDE Firmware
    │   ├── wrai_micro_esp32.ino       # Continuous-state serial dialogue sketch
    │   ├── wrai_micro_engine.h        # C linear recurrence engine (64KB SRAM)
    │   ├── wrai_micro_vocab.h         # 4,023-token PROGMEM dictionary
    │   └── wrai_micro_weights.h       # INT8 weights PROGMEM array (~1.31 MB)
    ├── training/                      # Transplantation Pipeline
    │   ├── download_base_model.py     # SimpleStories base downloader
    │   ├── prepare_dataset.py         # Subword tokenizer & dataset builder
    │   └── train_transplant_esp32.py  # PyTorch architectural transplant
    └── tools/                         # Verification & Deployment Tools
        ├── export_to_c.py             # INT8 quantizer and C code exporter
        ├── bin_to_c_array.py          # Binary-to-Header C array generator
        ├── benchmark_reality_check.py # Authentic reality-check benchmark
        └── test_generalization_proof.py # Neural dynamics & generalization audit
```

---

## 🚀 Quick Navigation & Getting Started

### 🖥️ For PC / Server CPU Inference: WRAI-X (0.8B)
👉 **Read the complete guide in [`qwen/README.md`](qwen/)**
1. Download pre-compiled INT8 weights (1.35 GB) from [Hugging Face](https://huggingface.co/Musouno-Enma99/WRAI-X-0.8B-Qwen3):
   ```bash
   python qwen/weights/download_weights.py
   ```
2. Compile and launch the native C engine:
   ```bash
   qwen/engine/build_wrai_x.bat
   qwen/engine/run_wrai_x.bat
   ```

### ⚡ For Physical Microcontrollers: WRAI-Micro (1.25M)
👉 **Read the complete guide in [`esp32/README.md`](esp32/)**
1. Open [`esp32/firmware/wrai_micro_esp32/wrai_micro_esp32.ino`](esp32/firmware/wrai_micro_esp32/wrai_micro_esp32.ino) in Arduino IDE.
2. Select **Board: ESP32 Dev Module** and **Partition Scheme: Huge APP (3MB No OTA)**.
3. Flash to your board on `COM3` (or device port), then open Serial Monitor at **115200 baud** to converse with offline AI running directly on your microcontroller!

---

## 📜 Academic Integrity & Citation

WRAI is an open-source research initiative distributed under the **Apache License 2.0**.  
If you reference our sub-quadratic recurrence findings, zero KV-cache proofs, or physical silicon implementations, please cite:

```bibtex
@misc{wrai_2026,
  author = {WRAI Research Team},
  title = {WRAI: Wavelet Retention Artificial Intelligence with Strictly Constant O(1) State Memory and Native C Execution},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/MusounoEnma/WRAI}}
}
```
