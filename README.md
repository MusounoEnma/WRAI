# 🌊 WRAI-X (0.8B): Wavelet Retention AI Engine
> **A Transparent, Zero KV-Cache Recurrent Wavelet Architecture Transplanted from Qwen3-0.6B with a Pure Native C Inference Engine**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![C Standard](https://img.shields.io/badge/C-C99%20Pure%20Native-00599C?logo=c)](final%20wrai/engine/src/wrai_x_engine.c)
[![Architecture](https://img.shields.io/badge/Architecture-Dual--State%20Retention%20%2B%20Haar%20DWT-ff69b4)](wrai-x/)
[![Memory Complexity](https://img.shields.io/badge/RAM%20Scaling-O(1)%20Constant%20(Zero%20KV--Cache)-brightgreen)](final%20wrai/engine/audit/)
[![Hardware Target](https://img.shields.io/badge/Hardware-x86__64%20AVX%20SIMD%20%2B%20OpenMP-orange)](final%20wrai/engine/)

---

## 📌 Introduction & Design Philosophy

The **WRAI-X (0.8B)** project stems from a pragmatic, transparent, and mathematically grounded engineering philosophy: **we do not claim to reinvent the wheel from scratch**. Instead, **we synthesize and transplant proven foundational techniques** into a unified, lightweight, and low-power CPU inference framework.

Conventional Transformers suffer from an inherent memory scaling bottleneck: **Linear KV-Cache Growth ($O(T)$)**, which consumes gigabytes of RAM/VRAM as conversation lengths expand.

To solve this fundamentally without retraining a multi-billion parameter model from scratch at massive computational cost, WRAI-X performs an **Architectural Transmutation (Transplantation)**:
1. **Preserving Pre-trained Knowledge**: Retaining and freezing the SwiGLU Feed-Forward Networks (FFN), RMSNorm layers, and Unembedding Head of **Qwen3-0.6B** (100% frozen).
2. **Replacing Quadratic Attention with Dual-State Retention**: Substituting Multi-Head Attention with **Dual-State Recurrent Retention ($M_t / R_t$)**, locking contextual memory into fixed-size matrices ($128 \times 128$) — achieving **Zero KV-Cache ($O(1)$ Memory)**.
3. **4-Level 1D Discrete Haar Wavelet Spectral Filtering (DWT)**: Decomposing latent representations into low-frequency approximations (global semantics) and high-frequency details (local syntax).
4. **Pure Native C Inference Engine**: Completely independent of Python, PyTorch, or CUDA runtimes, utilizing *zero-heap virtual memory-mapping* (`mmap`) and 256-bit AVX SIMD execution.

---

## 🏛️ Computational Architecture Diagram

```
                                [ Input Token ID ]
                                        │
                         Embedding Matrix [151936 × 1024]
                                        │
           ┌────────────────────────────┴────────────────────────────┐
           │                   28x WRAI-X LAYERS                     │
           │                                                         │
           │  1. RMSNorm (Pre-Retention)                             │
           │                                                         │
           │  2. Dual Recurrent Retention State (In-Place O(1)):     │
           │     • Memory State:    M_t = γ_m · M_{t-1} + K_t^T V_t  │
           │     • Reasoning State: R_t = γ_r · R_{t-1} + K_t^T V_t  │
           │     • RetNet GroupNorm per-head                         │
           │                                                         │
           │  3. 4-Level 1D Discrete Haar Wavelet Transform (DWT):   │
           │     • Approximation (Low-Freq): Global semantic state   │
           │     • Detail (High-Freq): Local syntax / micro-features │
           │                                                         │
           │  4. Residual Addition                                   │
           │                                                         │
           │  5. RMSNorm (Pre-FFN)                                   │
           │                                                         │
           │  6. Frozen Qwen SwiGLU FFN:                             │
           │     • W_gate [1024 → 3072], W_up [1024 → 3072]          │
           │     • SiLU Activation & W_down [3072 → 1024]            │
           │                                                         │
           │  7. Residual Addition                                   │
           └────────────────────────────┬────────────────────────────┘
                                        │
                                  Final RMSNorm
                                        │
                          Unembedding Head [1024 × 151936]
                                        │
                             [ Next Token Logits ]
```

---

## 🔬 Empirical OS Kernel & Hardware Audits (Not a Mockup / Simulation)

WRAI-X performance claims are verified through direct measurement at the **Windows NT Kernel API (`psapi.h`)** level while executing the 1.35 GB binary on physical x86_64 CPU hardware:

### 1. Verification of Memory Mapping & Disk-to-RAM Physical Transfer
| Kernel & Hardware Metric | Measured Value | Forensic Significance |
| :--- | :--- | :--- |
| **Model Binary on Storage** | `wrai_x_08b_int8.bin` | **1,422,927,244 bytes** (~**1.35 GB**) |
| **Windows Virtual Base Address** | `0x0000023c80000000` | Virtual memory range assigned by Windows NT Memory Manager |
| **Physical Working Set (RAM)** | **912.90 MB** | Physical DDR hardware RAM holding active model weights |
| **Hardware Page Faults (MMU)** | **234,259 pages** | Direct physical transfer of 4 KB blocks from SSD to RAM by CPU MMU |
| **SIMD AVX Computation** | **~1.30 GFLOPs / token** | Real quantized matrix dot-products in 256-bit CPU registers |

> **Physical Calculation**: $234,259\text{ pages} \times 4,096\text{ bytes} = 959,524,864\text{ bytes} \approx \mathbf{915\text{ MB}}$.  
> This closely matches the physical RAM Working Set (912.90 MB), providing undeniable forensic proof that weights are genuinely read and paged from disk into physical hardware RAM.

### 2. Empirical Proof of Zero KV-Cache ($O(1)$ Scaling)
Physical RAM consumption of the active inference process measured continuously as sequence length increases:

| Context Length ($T$) | Physical RAM (WRAI-X) | Delta RAM (WRAI-X) | Standard Transformer (KV-Cache) | Cache Status |
| :---: | :---: | :---: | :---: | :---: |
| **$T = 1$** | **918.62 MB** | **+0.00 MB** | 0.44 MB | **0% (Zero KV)** |
| **$T = 16$** | **918.62 MB** | **+0.00 MB** | 7.00 MB | **0% (Zero KV)** |
| **$T = 32$** | **918.62 MB** | **+0.00 MB** | 14.00 MB | **0% (Zero KV)** |
| **$T = 64$** | **918.61 MB** | **-0.01 MB** | 28.00 MB | **0% (Zero KV)** |
| **$T = 128$** | **918.61 MB** | **-0.01 MB** | 56.00 MB | **0% (Zero KV)** |
| **$T = 256$** | **918.61 MB** | **-0.01 MB** | 112.00 MB | **0% (Zero KV)** |
| **$T = 1024$** | **918.61 MB** | **-0.01 MB** | 448.00 MB | **0% (Zero KV)** |

* **Standard Transformer**: Memory grows linearly $O(T)$, quickly triggering Out-of-Memory (OOM) on resource-constrained hardware.
* **WRAI-X**: Memory remains strictly **flat and constant (+0.00 MB)** because state matrices $S_t \in \mathbb{R}^{128 \times 128}$ are updated in-place via recursive decay.

---

## 📁 Repository Structure

```
WRAI/
├── wrai-x/                             # 🚀 WRAI-X (0.8B) CORE WORKSPACE (Modular Base)
│   ├── run_wrai_x.bat                  # 1-Click interactive launcher for Windows
│   ├── README.md                       # Technical documentation & architecture details
│   │
│   ├── qwen/                           # Model Weights & Automated Setup
│   │   ├── wrai_x_vocab.bin            # BPE Vocabulary (151,936 tokens, 1.52 MB, in repo)
│   │   ├── download_weights.py         # Automated downloader from Hugging Face Model Hub
│   │   └── README.md                   # Specifications for .bin (1.35 GB) & .pt (1.66 GB)
│   │
│   ├── engine/                         # Pure Native C Inference Engine
│   │   ├── wrai_x.exe                  # Optimized compiled binary (-O3 -mavx -fopenmp)
│   │   ├── build_wrai_x.bat            # GCC MinGW-w64 build script
│   │   ├── run_wrai_x.bat              # Local execution script
│   │   ├── src/                        # C source files (kernel, CLI, BPE tables)
│   │   ├── include/                    # Architectural C header definitions
│   │   └── audit/                      # OS kernel & memory scaling forensic audit tools
│   │
│   └── training/                       # PyTorch Architectural Transplant Pipeline
│       ├── colab_train_wrai_x_08b_transplant.py # End-to-end transplant training script
│       ├── WRAI_X_08B_COLAB.ipynb      # Interactive Google Colab notebook
│       ├── quantize_wrai_x_08b_colab.py# PyTorch FP32 -> INT8 Row-wise Binary Converter
│       └── poc_wrai_x_08b.py           # Theoretical PyTorch verification suite
│
├── LICENSE                             # Official Apache License 2.0
└── .gitignore                          # Configured protection against >100MB GitHub limit
```

---

## 🚀 Quick Start Guide

### 1. Clone Repository
```bash
git clone https://github.com/MusounoEnma/WRAI.git
cd WRAI/wrai-x
```

### 2. Download Model Weights (1.35 GB INT8)
Official weights are hosted on the **Hugging Face Model Hub**:  
👉 **[https://huggingface.co/Musouno-Enma99/WRAI-X-0.8B-Qwen3](https://huggingface.co/Musouno-Enma99/WRAI-X-0.8B-Qwen3)**

Run the automated download helper:
```bash
cd qwen
python download_weights.py
cd ..
```

### 3. Launch Native C Inference Engine (Windows)
Double-click **`run_wrai_x.bat`** or run via command line:
```cmd
run_wrai_x.bat
```

---

## 🗺️ Continuous Evolution & Roadmap

WRAI-X (0.8B) represents the **Foundation Phase** of this research. The architecture is actively designed for modular evolution:
- [x] **v0.8B Foundation (Current Release)**: Empirical validation of Qwen3-0.6B architectural transmutation with verified $O(1)$ Zero KV-Cache.
- [ ] **Scaling to 1.5B & 3B**: Applying the transplant pipeline to larger base models (such as Qwen2.5-1.5B and Meta Llama-3.2) for deeper logical reasoning.
- [ ] **Universal Multi-Model Engine**: Dynamic tensor-dimension discovery in C (load any arbitrary WRAI binary model without recompilation).
- [ ] **Extended Context Tuning**: Continued distillation on dialogue datasets for enhanced conversational fluency.

---

## 📢 Transparency & Checkpoint Notes

In accordance with our commitment to scientific honesty:
* **Current Checkpoint Status**: The included weights represent an early **Stage-1 Architectural Transplant Proof-of-Concept**. The model successfully adheres to Chain-of-Thought formatting (`<think> ... </think>`) and responsive greeting routines, with the primary objective centered on **proving recurrent numerical stability and eliminating KV-Cache**.
* **Alpha Gate**: The residual integration gate $\alpha$ is intentionally set within a conservative range (~0.01) during initial tuning to guarantee signal stability. Continued training can be conducted using scripts in the `training/` folder on free Google Colab GPUs.

---

## 🙏 Theoretical Foundations & Acknowledgements

This project is built upon foundational research contributions from the global AI community:
1. **Qwen Team (Alibaba Cloud)**: For releasing the outstanding [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) foundation model under the Apache 2.0 license, providing high-quality representations, FFNs, and BPE tokenization.
2. **Microsoft Research (RetNet Authors - Sun et al., 2023)**: For the seminal paper *"Retentive Network: A Successor to Transformer for Large Language Models"*, mathematically formulating recursive linear retention and GroupNorm.
3. **Alfréd Haar (1909) & The Signal Processing Community**: For the Discrete Haar Wavelet Transform (DWT), enabling elegant multi-resolution frequency decomposition.
4. **Pentti Kanerva & The Hyperdimensional Computing (HDC) Community**: For foundational concepts in high-dimensional associative vector representations.
5. **Georgi Gerganov & The Open-Source C/C++ Community (*llama.cpp*)**: For demonstrating that clean, dependency-free C/C++ implementations with memory-mapping (`mmap`) make LLMs accessible on everyday consumer hardware.

---

## 📜 License
This project is licensed under the **Apache License 2.0** — see the [LICENSE](LICENSE) file for details.
