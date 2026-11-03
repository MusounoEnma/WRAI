# WRAI: Surgical $O(1)$ Memory Retrofit for Edge LLMs
### Transforming Linear KV-Cache into Constant Tri-Memory for Low-Resource Edge Computing

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.9%2B-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![Hardware](https://img.shields.io/badge/Tested%20On-AMD%20A8--7410%20(AVX1%2FSSE4.1)-purple.svg)](#empirical-evaluation)

---

## Abstract

Autoregressive Large Language Models (LLMs) deployed on edge devices are severely bottlenecked by the quadratic $O(N^2)$ / linear $O(N)$ memory growth of the Key-Value (KV) cache. On resource-constrained hardware (e.g., budget laptops, single-board computers, embedded devices with 4–8 GB RAM and legacy CPUs), increasing the prompt context length rapidly exhausts memory bandwidth and physical RAM, leading to catastrophic slowdowns or Out-Of-Memory (OOM) crashes.

**WRAI (Wavelet / Recurrent / Attentive Intelligence)** introduces a **surgical transmutation architecture** that replaces standard attention layers of pretrained models (`Qwen/Qwen3-0.6B`) with a bounded, constant $O(1)$ **Tri-Memory State**:
1. **Natural Haar Wavelet Pyramid**: Multi-resolution temporal pyramid capturing hierarchical time horizons (fine, coarse 2×, coarse 4×).
2. **R48 Recurrent State ($S, Z$)**: Bounded associative recurrence with depth-dependent decay ($\gamma$).
3. **Cue-Span & Novelty HDC State**: Dynamic consolidation of salient numeric/factual multi-token spans and global novelty spikes.

WRAI requires **zero pretraining from scratch** and operates 100% offline with zero online Hugging Face weight lookups during standalone runtime.

---

## Architectural Comparison

| Dimension | Standard Transformer (Qwen3-0.6B) | WRAI v2.0 (Tri-Memory) |
| :--- | :--- | :--- |
| **KV-Cache Complexity** | $O(N)$ Unbounded Growth | **$O(1)$ Strictly Constant** |
| **Decode Step Complexity** | $O(N)$ Attention Scan | **$O(1)$ Fixed Matrix Multiplication** |
| **State Size @ 1024 Context**| ~239.31 MiB (FP32) | **35.27 MiB (FP32) / 7.64 MiB (INT8)** |
| **Pretraining Cost** | Billions of tokens ($10^5+$ USD) | **Surgical Transfer & Calibration** |
| **Hardware Portability** | Heavy reliance on AVX2 / FMA / GPU | **100% Pure Python + PyTorch JIT (x86, ARM64, Apple Silicon)** |
| **Runtime Dependencies** | Full Hugging Face runtime | **Zero-Dependency Meta Cold Loader** |

---

## Empirical Evaluation (ASUS X454YA / AMD A8-7410 CPU)

All benchmarks below were empirically measured on an ultra-budget **ASUS X454YA** laptop equipped with an **AMD A8-7410 APU (4 Cores @ 2.2 GHz, 8 GB RAM, AVX1, SSE4.1, NO AVX2, NO FMA)** running standard FP32 CPU execution across identical ChatML prompts and greedy decoding (fixed 64 decode steps).

### 1. Decode Speed & Context Scaling ($O(1)$ Flatline vs $O(N)$ Drop)

As the context length increases, standard Qwen's decode speed degrades by **-47.9%** due to expanding KV-cache traversal. In contrast, WRAI maintains a **completely flat $O(1)$ generation rate**, achieving a **+105.9% speedup (2.06× faster)** at 1,024 context tokens.

| Context Length | Qwen Native (FP32) | WRAI Micro (FP32) | Speed Difference | Speedup Ratio |
| :---: | :---: | :---: | :---: | :---: |
| **~33 tokens** | 1.727 tok/s | 1.842 tok/s | +6.6% | 1.07× |
| **~134 tokens** | 1.570 tok/s | 1.859 tok/s | +18.4% | 1.18× |
| **~518 tokens** | 1.235 tok/s | 1.864 tok/s | +50.9% | 1.51× |
| **~1,030 tokens** | 0.899 tok/s | **1.851 tok/s** | **+105.9%** | **2.06× Faster** |

### 2. State Memory Break-Even Point

* **Break-Even Point**: At **~95 tokens**, WRAI's Tri-Memory becomes smaller than standard Qwen KV-cache.
* **At 1,030 Context Tokens**:
  * Qwen KV-Cache: **239.31 MiB** (and continues growing linearly).
  * WRAI Tri-Memory: **35.27 MiB** (strictly constant, 6.8× smaller).
  * System Working Set: WRAI uses **~548.3 MiB less system RAM** (3,684.7 MiB vs 4,233.0 MiB).

---

## Repository Structure

```text
├── wrai_end_to_end_master_pipeline.py # Full Scientific Verification & Master Reproduction Pipeline (5-Step Audit & 6 Gates)
├── wrai_standalone_runner.py          # Standalone runner (meta-device loader, 3 poison traps, interactive CLI, pure Python)
├── tokenizer_and_config/              # Bundled offline tokenizer (11 MB) & config for 100% zero-internet execution
│   ├── chat_template.jinja
│   ├── config.json
│   ├── tokenizer.json
│   └── tokenizer_config.json
├── benchmarks/
│   ├── validate_zero_kv_psutil.py     # Live step-by-step 0-byte dynamic KV-cache & psutil audit
│   ├── bench_context_scaling.py       # Context scaling audit (27 to 1024 tokens)
│   ├── bench_memory_wrai.py           # Real-time psutil continuous lifecycle memory monitor (WRAI)
│   ├── bench_memory_qwen.py           # Real-time psutil continuous lifecycle memory monitor (Qwen)
│   ├── wrai_standalone_bench.py       # Fixed 64-step greedy decode benchmark for WRAI
│   └── qwen_native_cpu_bench.py       # Fixed 64-step greedy decode benchmark for Qwen
├── requirements.txt                   # Minimal Python dependencies
├── .gitignore                         # Git exclusion rules (weights, binaries, logs)
├── LICENSE                            # Apache 2.0 Open Source License
└── README.md                          # Architecture and benchmark documentation
```

---

## Quickstart Guide

### 1. Requirements & Installation

```bash
git clone https://github.com/MusounoEnma/WRAI.git
cd "WRAI/WRAI Qwen 3 0.6B"
pip install -r requirements.txt
```

### 2. Run Standalone Inference (100% Pure Python, Offline, Zero Internet)

The standalone runner loads the surgical WRAI model via `meta` device instantiation, with zero dynamic KV cache and three active poison traps.

1. Download the pre-distilled standalone checkpoint (`~1.14 GB`):
   - **Hugging Face**: [MusounoEnma/WRAI-Qwen3-0.6B](https://huggingface.co/MusounoEnma/WRAI-Qwen3-0.6B)
   - **Zenodo DOI**: [10.5281/zenodo.23920132](https://doi.org/10.5281/zenodo.23920132)
   Place `wrai_qwen3_0_6b_universal_homogeneous_standalone.pt` in this folder.

2. Run interactive chat on CPU (or GPU):
```bash
# Pure Python + PyTorch JIT (100% Cross-Platform: Windows, Linux, macOS, ARM64)
python wrai_standalone_runner.py \
  --model wrai_qwen3_0_6b_universal_homogeneous_standalone.pt \
  --backend jit_float \
  --threads 4 \
  --interactive
```

### 3. Full Master Reproduction & 5-Step Verification (CLI / Server / Colab)

To run the complete verification suite from raw `Qwen/Qwen3-0.6B`, surgical transplant, 16-token Cue-Span HDC verification, multi-seed passkey audit, and 6 Pre-Gemma verification gates:

```bash
python wrai_end_to_end_master_pipeline.py --output-dir ./runs/master_verification --device cuda
```

---

## Architectural Guarantees & Poison Traps

To ensure absolute integrity and prove mathematically that standard quadratic attention is never invoked during inference, the WRAI standalone runner enforces **three hardware-level poison traps**:

1. **`DynamicCache` Poison Trap**: Any attempt to call `.update()` on the KV cache immediately raises a fatal `RuntimeError("FATAL KV_CACHE_POISON_TRAP TRIGGERED!")`.
2. **`scaled_dot_product_attention` Poison Trap**: `torch.nn.functional.scaled_dot_product_attention` is globally hijacked to throw a fatal exception if invoked.
3. **Module Architecture Audit**: An assertion verifies that zero `Qwen3Attention` modules and zero raw projection weights (`.q_proj`, `.k_proj`, `.v_proj`) exist in the runtime model tree.

---

## Citation & Technical Report

If you use or build upon WRAI in your research, please cite:

```bibtex
@article{wrai2026surgical,
  title={WRAI: Surgical $O(1)$ Memory Retrofit for Edge LLMs via Tri-Memory Recurrent Transmutation},
  author={MusounoEnma},
  journal={Zenodo Preprint},
  year={2026},
  doi={10.5281/zenodo.23920132}
}
```

## License

This project is released under the [Apache 2.0 License](LICENSE).
