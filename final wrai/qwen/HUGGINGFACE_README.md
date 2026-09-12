---
license: apache-2.0
language:
- en
- id
tags:
- recurrent
- retnet
- wavelet
- haar-dwt
- zero-kv-cache
- qwen
- architectural-transplant
- cpu-inference
- int8
base_model: Qwen/Qwen3-0.6B
pipeline_tag: text-generation
library_name: c-native
---

# 🌊 WRAI-X (0.8B): Experimental Wavelet-Retention Hybrid

> **⚠️ Research Disclaimer:** This model is an **Experimental Proof-of-Concept (PoC)** designed to explore the feasibility of transplanting Transformer quadratic attention into **Dual-State Recurrent Retention + Discrete Haar Wavelet Transform (DWT)** without training from scratch. It is **not** a production-ready conversational agent, but rather an open research artifact intended for AI systems researchers, students, and low-power edge computing enthusiasts.

* **Hugging Face Model Hub:** [https://huggingface.co/Musouno-Enma99/WRAI-X-0.8B-Qwen3](https://huggingface.co/Musouno-Enma99/WRAI-X-0.8B-Qwen3)
* **GitHub Repository:** [https://github.com/MusounoEnma/WRAI](https://github.com/MusounoEnma/WRAI)
* **Base Pre-trained Brain:** `Qwen/Qwen3-0.6B` (100% Frozen FFN, RMSNorm, and Embeddings)
* **Total Measured Parameters:** **831,268,848 Parameters** (~0.83B)
* **Available Artifact Formats:**
  * `wrai_x_08b_int8.bin` (1.35 GB) — *INT8 symmetric row-wise quantized binary for zero-heap C engine*
  * `wrai_x_08b_transplanted.pt` (1.66 GB) — *PyTorch checkpoint for research, inspection, and continued fine-tuning*

---

## 📌 Motivation & Architecture

Standard Transformer architectures suffer from a fundamental memory scaling bottleneck: the **Linear KV-Cache Growth ($O(T)$)**, which consumes gigabytes of VRAM/RAM as sequence lengths increase.

Rather than training a billion-parameter model from scratch at massive computational expense, the WRAI-X project explores **Architectural Transmutation (Transplantation)**:
1. **Leveraging Pre-Trained Knowledge**: We preserve and freeze the pre-trained SwiGLU FFN knowledge layers, RMSNorms, and embeddings from **Qwen3-0.6B**.
2. **Replacing Quadratic Attention with Dual-State Retention ($M_t / R_t$)**: Inspired by *Microsoft Research's RetNet*, sequence context is compressed into fixed-size state matrices ($128 \times 128$) updated in-place recursively — achieving **Zero KV-Cache ($O(1)$ constant memory)**.
3. **4-Level 1D Discrete Haar Wavelet Transform (DWT)**: Decomposes latent representation signals into low-frequency approximations (global semantics) and high-frequency details (local syntax).
4. **Pure Native C Inference Engine**: Completely eliminates Python and heavy runtimes. Powered by Win32/POSIX virtual memory-mapping (`mmap`) and 256-bit AVX SIMD execution.

---

## 🔬 Honest Assessment & Current Status (No Hype)

We strongly believe in academic integrity and radical transparency regarding model capabilities:

### ✅ Validated & Working Well:
* **Recurrent Numerical Stability**: The decay parameter $\gamma$ and per-head RetNet GroupNorm normalization remain strictly stable without numerical drift or exploding activations over long generation loops.
* **Empirically Proven Zero KV-Cache ($O(1)$ RAM)**: The physical memory footprint of the active process remains completely flat (**+0.00 MB delta**) from token $T=1$ to long context sequences.
* **Low-Power CPU Viability**: Operates smoothly on consumer laptop CPUs (benchmarked on a 2014 AMD A8 Puma+ APU) at ~2 – 3 tokens/second without requiring dedicated GPU hardware.
* **Output Formatting**: Consistently triggers structured Chain-of-Thought thinking blocks (`<think> ... </think>`) and basic greeting routines.

### ⚠️ Current Limitations (Work in Progress):
* **Early-Stage Linguistic Coherence**: This checkpoint represents an initial **Stage-1 Checkpoint** (early adapter routing and gate alignment). Complex reasoning and grammar fluency are still adapting, and the model may produce repetitive loops or unnatural syntax when presented with complex prompts.
* **Conservative Alpha Gate**: The residual transplant gate $\alpha$ is intentionally set to a small range (~0.01) to preserve baseline signal stability during initial distillation.
* **Target Audience**: Intended strictly for **AI architecture researchers, systems engineers, and hobbyists** investigating recurrent linear-time alternatives to attention mechanisms.

---

## 📊 OS Kernel & Hardware Forensics (Empirical Audit)

The following metrics were captured directly via **Windows NT Kernel Memory APIs (`psapi.h`)** while running `wrai_x_08b_int8.bin` on physical x86_64 CPU hardware:

| Kernel & Hardware Metric | Measured Value | Forensic Significance |
| :--- | :--- | :--- |
| **Model Binary on Storage** | **1,422,927,244 bytes** (~1.35 GB) | Physical INT8 row-wise binary |
| **Physical Working Set (RAM)** | **912.90 MB** | Actual hardware DDR memory paged in by OS |
| **Hardware Page Faults (MMU)** | **234,259 pages** | Direct physical transfer of 4 KB blocks from SSD to RAM |
| **SIMD AVX Computation** | **~1.30 GFLOPs / token** | Real quantized matrix dot-products in 256-bit CPU registers |
| **KV-Cache Memory Growth** | **+0.00 MB Flat ($O(1)$)** | Constant ~14.5 MB recurrent state matrix buffer |

$$\text{Paging Calculation: } 234,259 \text{ pages} \times 4,096 \text{ bytes} \approx \mathbf{915 \text{ MB}} \quad (\text{Exact match to 912.90 MB Working Set})$$

---

## 🚀 Getting Started

### Option 1: Native C Inference (Windows / Linux)
1. Clone the project repository:
   ```bash
   git clone https://github.com/MusounoEnma/WRAI.git
   cd WRAI/wrai-x
   ```
2. Place `wrai_x_08b_int8.bin` inside the `qwen/` folder (or run `python qwen/download_weights.py`).
3. Double-click **`run_wrai_x.bat`** (or execute `.\engine\wrai_x.exe`).

### Option 2: PyTorch Experimentation (Google Colab / Python)
The full transplant training script, verification suite, and Google Colab notebook are available in the GitHub repository under `training/`:
* `training/colab_train_wrai_x_08b_transplant.py`
* `training/WRAI_X_08B_COLAB.ipynb`

---

## 🗺️ Continuous Evolution & Roadmap

WRAI-X (0.8B) represents the **Foundation Phase** of this research. The architecture is actively designed for modular evolution:
- [x] **v0.8B Foundation (Active)**: Core Proof-of-Concept on Qwen3-0.6B backbone with verified $O(1)$ Zero KV-Cache.
- [ ] **Scaling to 1.5B & 3B**: Expanding the transplant pipeline to Qwen2.5-1.5B and Meta Llama-3.2 for enhanced logic and coding.
- [ ] **Universal Multi-Model Engine**: Dynamic tensor-dimension auto-discovery in C (load any WRAI model binary seamlessly).
- [ ] **Extended Stream Tuning**: Continued distillation on conversational datasets for enhanced natural fluency.

---

## 🙏 Theoretical Foundations & Acknowledgements

This exploratory work stands on the shoulders of remarkable contributions from the global AI research community:
1. **Qwen Team (Alibaba Cloud)**: For releasing the outstanding [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) foundation model under the Apache 2.0 license, providing robust language embeddings and representations.
2. **Microsoft Research (RetNet - Sun et al., 2023)**: For the seminal paper *"Retentive Network: A Successor to Transformer for Large Language Models"*, which mathematically formulated recursive linear retention and GroupNorm.
3. **Alfréd Haar (1909) & The Signal Processing Community**: For the foundational Discrete Haar Wavelet Transform (DWT), enabling multi-resolution frequency decomposition.
4. **Pentti Kanerva & The Hyperdimensional Computing (HDC) Community**: For foundational concepts in high-dimensional associative vector representations.
5. **Georgi Gerganov & The *llama.cpp* Community**: For demonstrating that clean, dependency-free C/C++ implementations with memory-mapping (`mmap`) make LLMs accessible on everyday consumer hardware.

---

## 📜 License
This model card, the quantized weights, and the accompanying engine code are distributed under the **Apache License 2.0**.
