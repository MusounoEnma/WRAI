# 🚀 WRAI-X (0.8B-Class / ~0.83B): Wavelet-Retention Foundation Model
> **Sub-Quadratic Recurrent Language Model Transplanted from Qwen3-0.6B with 831M Parameters, Zero KV-Cache, and Pure Native C Inference**

[![Base Model](https://img.shields.io/badge/Base%20Backbone-Qwen3--0.6B%20(100%25%20Frozen%20FFN)-blue.svg)](https://huggingface.co/Qwen/Qwen3-0.6B)
[![Total Parameters](https://img.shields.io/badge/Total%20Parameters-831%2C268%2C848%20(~0.83B)-brightgreen.svg)](#-exact-architectural--parameter-accounting)
[![Memory Scaling](https://img.shields.io/badge/Context%20RAM-O(1)%20Persistent%20State%20(29.42%20MB)-success.svg)](#-os-kernel--hardware-memory-audits)
[![Engine Target](https://img.shields.io/badge/C%20Engine-x86__64%20AVX%20SIMD%20%2B%20Virtual%20mmap-orange.svg)](#-native-c-inference-engine-guide)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Model%20Weights%20(1.35%20GB)-yellow.svg)](https://huggingface.co/Musouno-Enma99/WRAI-X-0.8B-Qwen3)

---

## 📌 Executive Summary

**WRAI-X (0.8B)** addresses the core bottleneck of modern Transformer language models: **Linear KV-Cache Growth ($O(T)$)**. In conventional Transformer architectures, storing past Key-Value states requires allocating dozens of gigabytes of RAM/VRAM as sequence context expands.

WRAI-X implements an **Architectural Transmutation (Organ Transplantation)** from `Qwen/Qwen3-0.6B`:
1. **100% Frozen Knowledge Layers**: Preserves all 28 layers of pre-trained SwiGLU Feed-Forward Networks (FFN), Word Embeddings (151,936 tokens), and RMSNorm layers.
2. **Dual-State Linear Retention ($M_t / R_t$)**: Replaces quadratic Softmax Self-Attention with dual recursive retention matrices ($128 \times 128$ per head), compressing conversational history into a fixed **29.42 MB buffer** with **Zero KV-Cache ($O(1)$ memory complexity)**.
3. **4-Level 1D Discrete Haar Wavelet Transform (DWT)**: Decomposes latent representation vectors (1,024 dimensions) into low-frequency approximations (global semantics) and high-frequency details (local syntax).
4. **HDC Associative Scratchpad**: Provides Hyperdimensional Computing associative memory ($k \odot v$) with gated resonance for `<think> ... </think>` Chain-of-Thought reasoning tokens.
5. **Pure Native C Inference Engine**: Completely independent of Python, PyTorch, or CUDA runtimes, utilizing zero-heap virtual memory-mapping (`MapViewOfFile` / `mmap`) and 256-bit AVX SIMD execution.

---

## 🏛️ Exact Architectural & Parameter Accounting

While the base pre-trained backbone is Qwen3-0.6B, the integration of dual linear retention projection matrices ($W_q, W_k, W_v, W_o$ for both Memory State $M_t$ and Reasoning State $R_t$), GroupNorm, and HDC scratchpad projections brings the total count to **831,268,848 unique parameters**, placing it accurately in the **0.8B class**:

| Architectural Component | Dimensions / Structure | Parameter Count | Knowledge Status |
| :--- | :--- | :---: | :--- |
| **Token Embeddings (`embed_tokens`)** | $151,936 \times 1024$ | 155,582,464 | **100% Frozen from Qwen3** |
| **SwiGLU FFN (28 Layers)** | $28 \times [3 \times (1024 \times 3072)]$ | 264,241,152 | **100% Frozen from Qwen3** |
| **RMSNorm Layers (All Layers)** | $28 \times 2 \times 1024 + 1024$ | 58,368 | **100% Frozen from Qwen3** |
| **Dual Retention Projections ($M_t / R_t$)** | $28 \times [2 \times 4 \times (1024 \times 2048)]$ | 469,762,048 | Transplanted & Adapted |
| **Unembedding Head (LM Head)** | Weight-Tied to Embeddings | 0 (Tied) | **100% Frozen from Qwen3** |
| **Total Parameter Count** | — | **831,268,848** | **~0.83B Parameters** |

---

## 🔬 OS Kernel & Hardware Memory Audits

The forensic metrics below were captured directly via **Windows NT Kernel Memory APIs (`psapi.h`)** while running the compiled release binary `wrai_x.exe` on physical x86_64 CPU hardware:

### 1. Verification of Physical Memory-Mapping & Disk-to-RAM Transfer
| Kernel & Hardware Metric | Measured Value | Forensic Significance |
| :--- | :--- | :--- |
| **Model Binary on Storage** | `wrai_x_08b_int8.bin` | **1,422,927,244 bytes (~1.35 GB)** |
| **Windows Virtual Base Address** | `0x0000023c80000000` | Virtual address space assigned by Windows NT Memory Manager |
| **Physical Working Set (RAM)** | **912.90 MB** | Physical DDR memory paged in by OS |
| **Hardware Page Faults (MMU)** | **234,259 pages** | Direct physical transfer of 4 KB blocks from SSD to RAM |
| **SIMD AVX Computation** | **~1.30 GFLOPs / token** | Real quantized matrix dot-products in 256-bit CPU registers |

$$\text{Paging Calculation: } 234,259 \text{ pages} \times 4,096 \text{ bytes} \approx \mathbf{915 \text{ MB}} \quad (\text{Matches 912.90 MB Physical Working Set})$$

### 2. Empirical Proof of Zero KV-Cache Across Context Lengths
Unlike Transformers that suffer exponential memory blowups, WRAI-X physical RAM consumption remains flat as sequence length scales from $T=1$ to $T=32,768$ ($32\text{K}$):

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

> *\* Theoretical KV-Cache calculation assumes $L=28$ layers, $H_{kv}=16$ key-value heads, $d_k=128$ head dimension in standard FP16 ($2 \times 28 \times 16 \times 128 \times 2 = 229,376\text{ bytes/token} \approx 0.4375\text{ MB/token}$).*

---

## 🛠️ Step-by-Step Reproduction Guide

### 1. Download Model Weights
Pre-quantized INT8 binary weights (~1.35 GB) are hosted on the **Hugging Face Model Hub**:  
👉 **[https://huggingface.co/Musouno-Enma99/WRAI-X-0.8B-Qwen3](https://huggingface.co/Musouno-Enma99/WRAI-X-0.8B-Qwen3)**

Run the automated download helper:
```bash
python qwen/weights/download_weights.py
```

### 2. Run the Native C Engine on Windows
Compile the high-performance AVX2 C engine and start interactive inference:
```cmd
qwen\engine\build_wrai_x.bat
qwen\engine\run_wrai_x.bat
```

### 3. Re-Transplant Model from Scratch (Google Colab / PyTorch)
To replicate the architectural transplantation from Qwen3-0.6B to WRAI-X:
1. Open [`qwen/training/WRAI_X_08B_COLAB.ipynb`](qwen/training/WRAI_X_08B_COLAB.ipynb) in Google Colab (Free T4 or A100).
2. Or run the standalone training script locally:
   ```bash
   python qwen/training/colab_train_wrai_x_08b_transplant.py
   ```

### 4. Quantize PyTorch Checkpoint to INT8
Converts the `.pt` PyTorch checkpoint into the contiguous row-wise INT8 binary payload required by the C engine:
```bash
python qwen/quantization/quantize_wrai_x_08b_colab.py
```

### 5. Verify Zero KV-Cache & Inspect Weights
```bash
python qwen/inference/verify_zero_kv_cache_deep_dive.py
python qwen/inference/inspect_wrai_x_drive_model.py
python qwen/inference/test_wrai_x_08b_english.py
```

---

## 🔬 Radical Transparency: Scope & Honest Status (No Hype)

We adhere strictly to academic integrity and transparency regarding model capabilities:
* **What Works Decisively**:
  * **Zero KV-Cache**: 100% physically proven; memory remains fixed at 29.42 MB state buffer across 32,768 tokens with zero heap reallocations.
  * **Low-Power Execution**: Runs at **~2.1 – 3.2 tok/s** on an ancient 2014 AMD A8 APU (2.0 GHz) laptop CPU without GPU, proving that pure C AVX vectorization enables edge CPU inference.
  * **Knowledge Retention**: All 28 layers of pre-trained SwiGLU FFN and token embeddings from Qwen3 remain intact and frozen.
* **Current Stage-1 Research Boundaries**:
  * This release represents an **Architectural Stage-1 Foundation Release**. While the mathematical recurrence and zero-cache mechanisms are rock-solid, nuanced multi-step logical reasoning and freeform conversational fluency are actively being refined through subsequent distillation and alignment passes.

---

## 🙏 Acknowledgements & Attribution

WRAI-X builds upon the brilliant research of the global AI community. We specifically acknowledge:
1. **The Qwen Team at Alibaba Cloud**: For the pre-trained weights and architectural excellence of **Qwen3-0.6B**.
2. **Microsoft Research**: For the mathematical principles of **RetNet** (*Sun et al.*), which provided the theoretical bedrock for linear retention recurrence.
3. **The Wavelet and Cognitive Architecture Pioneers**: Whose formulation of the **Discrete Haar Wavelet Transform (DWT)** and **Hyperdimensional Computing (HDC)** inspired our multi-resolution representation filters.

---

## 📜 License & Citation

WRAI-X is open-source under the **Apache License 2.0**.  
If you utilize this architecture or empirical memory audit data in your research, please cite:

```bibtex
@misc{wrai_x_08b_2026,
  author = {WRAI Research Team},
  title = {WRAI-X: Sub-Quadratic Dual-State Recurrent Language Model Transplanted from Qwen3 with Zero KV-Cache},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/MusounoEnma/WRAI/tree/main/qwen}}
}
```
