# 🌊 WRAI-X (0.8B-Class / ~0.83B): Wavelet Retention AI Engine
> **A Transparent, Sub-Quadratic Architecture Transplanted from Qwen3-0.6B (Yielding 831M Total Parameters) with Zero KV-Cache and a Pure Native C Inference Engine**

[![Release: v0.1.0](https://img.shields.io/badge/Release-v0.1.0-blue.svg)](https://github.com/MusounoEnma/WRAI/releases)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![C Standard](https://img.shields.io/badge/C-C99%20Pure%20Native-00599C?logo=c)](final%20wrai/engine/src/wrai_x_engine.c)
[![Architecture](https://img.shields.io/badge/Architecture-Dual--State%20Retention%20%2B%20Haar%20DWT-ff69b4)](wrai-x/)
[![Memory Complexity](https://img.shields.io/badge/Context%20RAM-O(1)%20Persistent%20State%20w.r.t.%20Length-brightgreen)](final%20wrai/engine/audit/)
[![Hardware Target](https://img.shields.io/badge/Hardware-x86__64%20AVX%20SIMD%20%2B%20OpenMP-orange)](final%20wrai/engine/)

---

## 📌 Introduction & Design Philosophy

The **WRAI-X (0.8B-class / ~0.83B)** project stems from a pragmatic, transparent, and mathematically grounded engineering philosophy: **we do not claim to reinvent the wheel from scratch**. Instead, **we synthesize and transplant proven foundational techniques** into a unified, lightweight, and low-power CPU inference framework.

Conventional Transformers suffer from an inherent memory scaling bottleneck: **Linear KV-Cache Growth ($O(T)$)**, which consumes gigabytes of RAM/VRAM as conversation lengths expand.

To solve this fundamentally without retraining a multi-billion parameter model from scratch at massive computational cost, WRAI-X performs an **Architectural Transmutation (Transplantation)**:
1. **Preserving Pre-trained Knowledge**: Retaining and freezing the SwiGLU Feed-Forward Networks (FFN), RMSNorm layers, and Unembedding Head of **Qwen3-0.6B** (100% frozen).
2. **Replacing Quadratic Attention with Dual-State Retention**: Substituting Multi-Head Attention with **Dual-State Recurrent Retention ($M_t / R_t$)**, locking contextual memory into fixed-size matrices ($128 \times 128$) — achieving **Zero KV-Cache ($O(1)$ persistent state memory with respect to context length)**.
3. **4-Level 1D Discrete Haar Wavelet Spectral Filtering (DWT)**: Decomposing latent representations into low-frequency approximations (global semantics) and high-frequency details (local syntax).
4. **Pure Native C Inference Engine**: Completely independent of Python, PyTorch, or CUDA runtimes, utilizing *zero-heap virtual memory-mapping* (`mmap`) and 256-bit AVX SIMD execution.
5. **Exact Parameter Accounting**: While the base backbone is Qwen3-0.6B, the addition of dual recurrent retention projections ($W_q, W_k, W_v, W_o$ for both $M_t$ and $R_t$) and HDC matrices results in **831,268,848 unique parameters** (~0.83B), placing the model accurately in the **0.8B-class**.

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
           │  2. Dual Recurrent Retention State (In-Place O(1) w.r.t Context Length): │
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

All performance metrics below were measured directly via **Windows NT Kernel APIs (`psapi.h`)** while executing the actual compiled release binary (`wrai_x.exe`) on physical x86_64 CPU hardware (benchmarked on a low-power AMD A8 Puma+ APU @ 2.0 GHz, 8 GB RAM):

### 1. Verification of Memory Mapping & Disk-to-RAM Physical Transfer
| Kernel & Hardware Metric | Measured Value | Forensic Significance |
| :--- | :--- | :--- |
| **Model Binary on Storage** | `wrai_x_08b_int8.bin` | **1,422,927,244 bytes** (~**1.35 GB**) |
| **Windows Virtual Base Address** | `0x0000023c80000000` | Virtual memory range assigned by Windows NT Memory Manager |
| **Physical Working Set (RAM)** | **912.90 MB** | Physical DDR hardware RAM holding active model weights |
| **Hardware Page Faults (MMU)** | **234,259 pages** | Direct physical transfer of 4 KB blocks from SSD to RAM by CPU MMU |
| **SIMD AVX Computation** | **~1.30 GFLOPs / token** | Real quantized matrix dot-products in 256-bit CPU registers |

> **Physical Calculation**: $234,259\text{ pages} \times 4,096\text{ bytes} = 959,524,864\text{ bytes} \approx \mathbf{915\text{ MB}}$.  
> This matches the physical RAM Working Set (912.90 MB) almost byte-for-byte, providing verifiable forensic proof that weights are genuinely paged from disk into physical hardware RAM by the operating system kernel.

### 2. Empirical Proof of Zero KV-Cache ($O(1)$ Persistent State Memory w.r.t. Context Length)

#### A. Dedicated Recurrent State Buffer Audit (The Core Architectural Proof)
Across the generation lifecycle from short prompt ($T=16$) to long context ($T=8,192$), WRAI-X retains contextual history in-place with zero dynamic heap reallocations:

| Empirical Audit Metric | Short Context ($T = 16$) | Long Context ($T = 8,192$) | Scalability Impact |
| :--- | :---: | :---: | :--- |
| **WRAI-X Persistent Recurrent State** | **29.42 MB** | **29.42 MB** | **$\Delta\text{State} = \mathbf{0.0000\text{ MB}}$ (Strictly Constant)** |
| **Dynamic Heap Calls during Inference** | **0 (`malloc` = 0)** | **0 (`realloc` = 0)** | **Zero heap fragmentation or re-allocation** |
| **Equivalent Transformer KV-Cache (FP16)\*** | **7.00 MB** | **3,584.00 MB (3.50 GB)** | Linear $O(T)$ memory growth ($512\times$ explosion) |
| **WRAI-X Memory Advantage vs. Transformer** | Baseline | **-3,554.58 MB (-99.2%)** | Completely eliminates KV-cache blowout |

#### B. Operating System Working Set (Physical RAM Footprint via Windows NT `psapi.h`)
Physical RAM consumption of the active inference process (`wrai_x.exe`) was continuously sampled via OS performance counters as the context length scaled up to 32,768 ($32\text{K}$) tokens:

| Context Length ($T$) | Physical RAM (WRAI-X C Engine) | Measured RAM Delta | Equivalent Transformer (KV-Cache Only)* | WRAI-X Context Memory Status |
| :---: | :---: | :---: | :---: | :---: |
| **$T = 1$** | **918.62 MB** | **+0.00 MB** | 0.44 MB | $O(1)$ Persistent State |
| **$T = 16$** | **918.62 MB** | **+0.00 MB** | 7.00 MB | $O(1)$ Persistent State |
| **$T = 64$** | **918.61 MB** | **-0.01 MB** | 28.00 MB | $O(1)$ Persistent State |
| **$T = 256$** | **918.61 MB** | **-0.01 MB** | 112.00 MB | $O(1)$ Persistent State |
| **$T = 1,024$** | **918.61 MB** | **-0.01 MB** | 448.00 MB | $O(1)$ Persistent State |
| **$T = 4,096$** | **918.61 MB** | **-0.01 MB** | **1,792.00 MB (1.75 GB)** | **Zero Cache Overhead** |
| **$T = 8,192$** | **918.61 MB** | **-0.01 MB** | **3,584.00 MB (3.50 GB)** | **Zero Cache Overhead** |
| **$T = 32,768$ (32K)** | **918.61 MB** | **-0.01 MB** | **14,336.00 MB (14.0 GB)** | **Eliminates OOM Risk** |

> *\* **Equivalent Transformer Reference Configuration**: Theoretical KV-Cache calculation is based on an identical architectural specification ($L=28$ layers, $H_{kv}=16$ key-value heads, $d_k=128$ head dimension) stored in standard 16-bit precision (FP16):  
> $$\text{Memory per Token} = 2 \times L \times H_{kv} \times d_k \times \text{sizeof(FP16)} = 2 \times 28 \times 16 \times 128 \times 2 = 229,376\text{ bytes} \approx 0.4375\text{ MB/token}$$  
> **WRAI-X Mathematical Complexity**: Memory is strictly **$O(1)$ persistent state memory with respect to context length $T$**. The recurrent state size is fixed at $S_t \in \mathbb{R}^{L \times H \times d_k \times d_k}$ (scaling purely with architectural dimensions $\mathcal{O}(L \cdot H \cdot d_k^2)$, independent of sequence length $T$). All historical context is retained inside fixed-dimension dual recurrent matrices $S_t \in \mathbb{R}^{128 \times 128}$ per head, requiring only **29.42 MB** of total recurrent state buffers throughout the entire lifetime of the process.

> **Benchmark Reproducibility**: All empirical figures above are generated directly by the release binary (`wrai_x.exe`) and can be verified by running `final wrai/engine/audit/verify_hardware_level.c` and `final wrai/engine/audit/audit_c_runtime_kv_cache.c` under the GCC `-O3 -mavx -msse4.2 -fopenmp` optimization profile.

---

## 📁 Repository Structure

```
WRAI/
├── qwen/                               # 🚀 WRAI-X (0.8B-Class / ~0.83B) FOUNDATION RELEASE
│   ├── README.md                       # Comprehensive Qwen documentation & Hugging Face guide
│   ├── engine/                         # Pure Native C Inference Engine (Zero KV-Cache)
│   │   ├── build_wrai_x.bat            # GCC MinGW-w64 build script
│   │   ├── run_wrai_x.bat              # Execution wrapper
│   │   ├── src/wrai_x_cli.c            # CLI with AVX SIMD execution
│   │   ├── include/wrai_x_engine.h     # C header definitions
│   │   └── audit/                      # Windows NT kernel & memory audit tools
│   ├── training/                       # Model Training & Transplantation
│   │   ├── colab_train_wrai_x_08b_transplant.py # Zero-contamination transplant pipeline
│   │   ├── WRAI_X_08B_COLAB.ipynb      # Official Google Colab notebook
│   │   └── poc_wrai_x_08b.py           # Rapid POC verification suite
│   ├── quantization/                   # Model Quantization
│   │   └── quantize_wrai_x_08b_colab.py# PyTorch FP32 -> INT8 Row-wise Binary Converter
│   ├── inference/                      # Verification & Inspection
│   │   ├── test_wrai_x_08b_english.py  # English dialogue & reasoning test
│   │   ├── inspect_wrai_x_drive_model.py # Weight tensor inspector
│   │   └── verify_zero_kv_cache_deep_dive.py # Deep-dive zero KV-cache audit
│   └── weights/                        # Weights Retrieval
│       └── download_weights.py         # Automated downloader from Hugging Face Model Hub
│
├── esp32/                              # ⚡ WRAI-MICRO (1.25M) PHYSICAL SILICON EDGE RELEASE
│   ├── README.md                       # Physical microcontroller documentation & benchmarks
│   ├── firmware/wrai_micro_esp32/      # Arduino IDE Firmware
│   │   ├── wrai_micro_esp32.ino        # Continuous state interactive serial sketch
│   │   ├── wrai_micro_engine.h         # Pure native C linear recurrence engine
│   │   ├── wrai_micro_vocab.h          # 4023-token PROGMEM dictionary
│   │   └── wrai_micro_weights.h        # INT8 weights PROGMEM array (~1.31 MB)
│   ├── training/                       # Transplantation Pipeline
│   │   ├── download_base_model.py      # Base SimpleStories downloader
│   │   ├── prepare_dataset.py          # Tokenizer & hybrid dataset generator
│   │   └── train_transplant_esp32.py   # PyTorch architectural transplantation
│   └── tools/                          # Deployment & Verification
│       ├── export_to_c.py              # INT8 quantizer and C exporter
│       ├── bin_to_c_array.py           # Binary-to-Header C array generator
│       ├── benchmark_reality_check.py  # Authentic reality-check benchmark
│       └── test_generalization_proof.py# Neural dynamics & generalization audit
│
├── LICENSE                             # Official Apache License 2.0
└── .gitignore                          # Configured protection against >100MB GitHub limit
```

---

## 🚀 Quick Start Guide

### 1. Clone Repository
```bash
git clone https://github.com/MusounoEnma/WRAI.git
cd WRAI
```

### 2. Run WRAI-X (0.8B) on PC / Server
Download model weights (~1.35 GB INT8) from **Hugging Face Model Hub**:  
👉 **[https://huggingface.co/Musouno-Enma99/WRAI-X-0.8B-Qwen3](https://huggingface.co/Musouno-Enma99/WRAI-X-0.8B-Qwen3)**

```bash
python qwen/weights/download_weights.py
qwen/engine/build_wrai_x.bat
qwen/engine/run_wrai_x.bat
```

### 3. Run WRAI-Micro (1.25M) on Physical ESP32
Open `esp32/firmware/wrai_micro_esp32/wrai_micro_esp32.ino` in Arduino IDE:
* Board: `ESP32 Dev Module`
* Partition Scheme: `Huge APP (3MB No OTA / 1MB SPIFFS)`
* Upload to device on your serial port (e.g. `COM3`), then open Serial Monitor at `115200 baud`!

---

## 🗺️ Continuous Evolution & Roadmap

WRAI-X (0.8B-class) represents the **Foundation Phase (v0.1.0)** of this research. The architecture is actively designed for modular evolution:
- [x] **v0.1.0 Foundation (Current Release)**: Empirical validation of Qwen3-0.6B architectural transmutation (831M parameters) with verified $O(1)$ persistent state memory (Zero KV-Cache).
- [ ] **Scaling to 1.5B & 3B**: Expanding the transplant pipeline to Qwen2.5-1.5B and Meta Llama-3.2 for deeper logical reasoning.
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
