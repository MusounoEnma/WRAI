# WRAI: Surgical $O(1)$ Constant-Memory Retrofit for Edge LLMs
### Transforming Linear KV-Cache into Bounded Tri-Memory for Low-Resource Edge Computing

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.9%2B-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![Dynamic KV Cache](https://img.shields.io/badge/Dynamic%20KV--Cache-0.00%20Bytes-success.svg)](#architectural-guarantees--poison-traps)
[![Models](https://img.shields.io/badge/Models-Qwen3--0.6B%20%7C%20Gemma--4--E2B%20%7C%20BananaMind--2M--ESP32-purple.svg)](#supported-model-ecosystem)

---

## Abstract

Standard autoregressive Large Language Models (LLMs) suffer from linear $O(N)$ memory growth and quadratic $O(N^2)$ computational complexity in their Key-Value (KV) cache. On resource-constrained edge hardware—such as budget laptops, single-board computers, and embedded devices with limited physical RAM—scaling context length inevitably triggers severe memory bandwidth bottlenecks, memory swapping, or Out-Of-Memory (OOM) crashes.

**WRAI (Wavelet / Recurrent / Attentive Intelligence)** introduces a **surgical transmutation architecture** that replaces standard self-attention mechanisms with a bounded, strictly constant $O(1)$ **Tri-Memory State**. WRAI requires **zero pretraining from scratch**, preserves native model weights via calibration, and guarantees:

1. **0.00 Bytes Dynamic KV Cache**: Zero memory allocations during autoregressive token generation.
2. **Flat $O(1)$ Decode Throughput**: Consistent generation speed regardless of sequence length.
3. **Hardware-Enforced Integrity**: DynamicCache poison traps prevent any fallback to quadratic attention.

> [!NOTE]
> **Historical Research Prototype**: Looking for the initial exploratory research (SimpleStories-1.25M & early WRAI-X prototypes)? The complete initial research codebase has been permanently preserved in the [`legacy` branch](https://github.com/MusounoEnma/WRAI/tree/legacy).

---

## The WRAI Tri-Memory Architecture

Across all supported model families, WRAI unifies three complementary mathematical structures into a coherent, constant-size memory bank:

```
                            ┌──────────────────────────────────────────────┐
                            │             Input Hidden State               │
                            └──────────────────────┬───────────────────────┘
                                                   │
                         ┌─────────────────────────┴─────────────────────────┐
                         ▼                                                   ▼
            ┌───────────────────────────┐                       ┌───────────────────────────┐
            │   R48 Associative State   │                       │ Haar Wavelet & Cue-Span   │
            │  Low-Rank Recurrence (S, Z)│                      │    Multiscale Ring Buffer │
            └────────────┬──────────────┘                       └─────────────┬─────────────┘
                         │                                                    │
                         │ Y_r48                                              │ Y_shh
                         ▼                                                    ▼
            ┌───────────────────────────┐                       ┌───────────────────────────┐
            │   Head Gain G_r48 Multiplier│                     │  Head Gain G_shh Multiplier│
            └────────────┬──────────────┘                       └─────────────┬─────────────┘
                         │                                                    │
                         └─────────────────────────┬──────────────────────────┘
                                                   │ Dual-Gated Sum
                                                   ▼
                                    ┌─────────────────────────────┐
                                    │    Transmuted Attention     │
                                    │           Output            │
                                    └─────────────────────────────┘
```

1. **R48 Low-Rank Associative Recurrence ($S, Z$)**:
   - Projects keys and queries onto a compact landmark basis ($M=48$).
   - Maintains recurrent associative state matrices updated via depth-dependent decay $\gamma$:
     $$S_t = \gamma S_{t-1} + \phi(k_t)^T v_t, \quad Z_t = \gamma Z_{t-1} + \phi(k_t)^T$$
   - Provides global associative memory retrieval in $O(1)$ time complexity without quadratic scans.

2. **Natural Haar Wavelet Pyramid**:
   - Preserves high-frequency temporal details through a hierarchical ring buffer:
     - **Fine Slots**: Immediate preceding tokens ($N_{\text{fine}}=16$ or $64$).
     - **Coarse $2\times$ Slots**: Paired Haar wavelet averages capturing medium-range transitions.
     - **Coarse $4\times$ Slots**: Quad-token averages capturing broader temporal trends.

3. **Salient Cue-Span & Novelty HDC State**:
   - **Sink Slots ($SINK\_SLOTS=16$)**: Initial tokens and system anchors permanently pinned in memory.
   - **Novelty Bank**: Tracks informational surprises via key vector norm displacement ($\|k_{\text{exit}} - k_{\text{prev}}\|_2$), promoting salient factual and numeric tokens into permanent slots.

4. **Dual-Gated Output Fusion**:
   - Fuses recurrent associative retrieval with multiscale local context using calibrated per-head gain multipliers:
     $$Y = \text{Gain}_{R48} \odot Y_{R48} + \text{Gain}_{SHH} \odot Y_{SHH}$$

---

## Supported Model Ecosystem

| Model Architecture | Layer Configuration | Special Features | Directory |
| :--- | :--- | :--- | :--- |
| **Qwen 3 0.6B** | 28 Homogeneous Layers (GQA 2:1, Head Dim 128) | 100% Pure Python + PyTorch JIT, cross-platform runner, budget CPU optimized | [`WRAI Qwen 3 0.6B/`](./WRAI%20Qwen%203%200.6B/) |
| **Google Gemma 4 E2B** | 35 Heterogeneous Layers (Sliding-Window + Full Attention) | Shared-KV cross-layer registry, Logit softcapping, Stage 17 Built-in Distilled Priors | [`Gemma 4 E2B/`](./Gemma%204%20E2B/) |
| **BananaMind 2 Micro (2.93M)** | 9 Layers (Hidden 192, 6 Heads, Head Dim 32) | Bare-metal C99 on ESP32 (4MB Flash, no PSRAM), MMU Flash mmap, 3.78 TPS, Strictly 0.00 Bytes Dynamic KV | [`BananaMind 2 Micro/`](./BananaMind%202%20Micro/) |

---

## Empirical Benchmarks

### 1. Generation Speed vs Context Scaling (AMD A8-7410 Quad-Core CPU, FP32)

Empirically measured on an entry-level AMD A8-7410 APU (2.2 GHz, 8 GB RAM, AVX1, SSE4.1, NO AVX2, NO FMA) over greedy generation (64 decode steps):

| Context Length | Standard Transformer (Qwen3) | WRAI Standalone (Pure Python JIT) | Speedup Ratio |
| :---: | :---: | :---: | :---: |
| **~33 tokens** | 1.727 tok/s | 1.842 tok/s | 1.07× |
| **~134 tokens** | 1.570 tok/s | 1.859 tok/s | 1.18× |
| **~518 tokens** | 1.235 tok/s | 1.864 tok/s | 1.51× |
| **~1,030 tokens** | 0.899 tok/s | **1.851 tok/s** | **2.06× Faster (+105.9%)** |

> **Key Observation**: While standard Transformers degrade by **-47.9%** due to expanding KV cache overhead, WRAI exhibits a **completely flat $O(1)$ generation rate**.

### 2. Physical Memory Footprint

* **Break-Even Point**: At **~95 tokens**, WRAI becomes strictly lighter than standard KV caching.
* **At 1,030 Tokens**:
  * Standard KV-Cache: **239.31 MiB** (and continues growing linearly with every token).
  * WRAI Tri-Memory: **35.27 MiB** (strictly constant, 6.8× smaller).
  * Total System RAM Saved: **~548.3 MiB** reduction in operating memory.

---

## Architectural Guarantees & Poison Traps

To prove conclusively that quadratic attention is never invoked during runtime, WRAI integrates **three hardware-level poison traps**:

1. **`DynamicCache` Poison Trap**: Calling `.update()` on the KV cache immediately throws `RuntimeError("FATAL KV_CACHE_POISON_TRAP TRIGGERED!")`.
2. **Global SDPA Ban**: `torch.nn.functional.scaled_dot_product_attention` is globally hijacked to fail-closed if accessed during decode.
3. **Module Tree Audit**: Zero standard attention classes (`Qwen3Attention`, `GemmaAttention`) or un-transmuted projection layers exist in the runtime model tree.

---

## Quickstart Guide

### 1. Clone Repository & Setup

```bash
git clone https://github.com/MusounoEnma/WRAI.git
cd WRAI
```

### 2. Choose Your Model

#### A. WRAI Qwen 3 0.6B (100% Pure Python Runner)
```bash
cd "WRAI Qwen 3 0.6B"
pip install -r requirements.txt

# Run Standalone Interactive Inference
python wrai_standalone_runner.py \
  --model standalone_models/wrai_qwen3_0_6b_universal_homogeneous_standalone.pt \
  --backend jit_float \
  --threads 4 \
  --interactive
```

#### B. WRAI Gemma 4 E2B (Zero-Shot Built-in Priors)
```bash
cd "Gemma 4 E2B"
pip install -r requirements.txt

# Run Zero-Shot Interactive Inference (Built-in Stage 17 Priors)
python wrai_gemma4_inference.py --interactive
```

#### C. BananaMind 2 Micro (Bare-Metal ESP32 Microcontroller)
```bash
cd "BananaMind 2 Micro"

# Option 1: 1-Click Flash & Monitor (Precompiled Binaries - No Toolchain Needed)
# Double-click flash_and_monitor.bat or run:
flash_and_monitor.bat

# Option 2: Monitor Only (Connect to Serial Output without Re-flashing)
monitor_only.bat
```

---

## Repository Structure

```text
WRAI/
├── LICENSE                        # Apache 2.0 Open Source License
├── README.md                      # Master architecture and technical overview
├── .gitignore                     # Git exclusion rules
│
├── BananaMind 2 Micro/            # Bare-Metal ESP32 Microcontroller Implementation
│   ├── model/                     # INT8 memory-mapped weights & tokenizer
│   ├── firmware/                  # ESP-IDF & Arduino C99 engine & precompiled binaries
│   ├── audit/                     # Verification logs, benchmark CSVs & hardware audit
│   ├── flash_and_monitor.bat      # 1-Click flash and stream tool
│   ├── monitor_only.bat           # 1-Click lightweight monitor tool
│   ├── serial_monitor.py          # Standalone Python serial console
│   └── README.md                  # BananaMind ESP32 documentation & flash guide
│
├── Gemma 4 E2B/                   # Google Gemma 4 E2B Implementation
│   ├── gemma4_wrai_pipeline.py    # Distillation and calibration pipeline
│   ├── wrai_gemma4_inference.py   # Zero-KV standalone inference engine
│   ├── requirements.txt           # Gemma-specific dependencies
│   ├── LICENSE                    # Apache 2.0 License
│   ├── .gitignore                 # Exclusion rules
│   └── README.md                  # Gemma 4 E2B technical documentation
│
└── WRAI Qwen 3 0.6B/              # Qwen 3 0.6B Implementation
    ├── wrai_end_to_end_master_pipeline.py # Full verification & reproduction pipeline
    ├── wrai_standalone_runner.py  # Pure Python + PyTorch JIT standalone runner
    ├── requirements.txt           # Qwen-specific dependencies
    ├── LICENSE                    # Apache 2.0 License
    ├── .gitignore                 # Exclusion rules
    ├── README.md                  # Qwen 3 0.6B technical documentation
    └── benchmarks/                # Full benchmark suite
        ├── bench_context_scaling.py
        ├── bench_memory_wrai.py
        ├── bench_memory_qwen.py
        ├── wrai_standalone_bench.py
        └── qwen_native_cpu_bench.py
```

---

## Citation & Technical Report

If you utilize WRAI in your research or edge deployments, please cite:

```bibtex
@article{wrai2026surgical,
  title={WRAI: Surgical $O(1)$ Memory Retrofit for Edge LLMs via Tri-Memory Recurrent Transmutation},
  author={WRAI Research Team},
  journal={TechRxiv / Zenodo Preprint},
  year={2026},
  doi={10.5281/zenodo.xxxxxxx}
}
```

## License

This project is licensed under the [Apache 2.0 License](LICENSE).
