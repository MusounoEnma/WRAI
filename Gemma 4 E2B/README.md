# WRAI x Google Gemma 4 E2B: Zero-KV Tri-Memory Inference Engine
### Transforming 35-Layer Heterogeneous Transformer Attention into $O(1)$ Constant State

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.9%2B-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/Model-google%2Fgemma--4--E2B-yellow.svg)](https://huggingface.co/google/gemma-4-E2B)

---

## Overview

**WRAI (Wavelet / Recurrent / Attentive Intelligence)** for **Google Gemma 4 E2B** (`google/gemma-4-E2B`) adapts the surgical Tri-Memory architecture to Google's state-of-the-art heterogeneous LLM backbone.

Standard autoregressive generation on Gemma 4 E2B suffers from linear $O(N)$ KV-cache accumulation across its 35 deep text layers. WRAI replaces every native attention block with **Universal Zero-KV WRAI Attention**, completely freezing memory consumption during generation:

* **0.0000 MiB Dynamic KV Cache**: Zero allocations during token generation.
* **11.5492 MiB Strictly Constant $O(1)$ State**: Fixed memory footprint across infinite sequence lengths.
* **Stage 17 Universal Distilled Priors**: Built-in distilled priors $(\alpha^*, \gamma^*, \text{wavelet}^*)$ for all 35 layers, enabling immediate inference without requiring custom weights.
* **Heterogeneous Layer Harmony**: Seamlessly supports Gemma's hybrid mix of Sliding-Window layers (512 tokens) and Full Attention layers (`[4, 9, 14, 19, 24, 29, 34]`), inter-layer Shared-KV registries, and logit softcapping.

---

## Architectural Comparison

| Dimension | Native Gemma 4 E2B (`Transformers`) | WRAI x Gemma 4 E2B |
| :--- | :--- | :--- |
| **Dynamic KV-Cache per Step** | Expanding $O(N)$ allocations | **0.00 Bytes ($O(1)$ Strict Invariance)** |
| **Attention Step Complexity** | $O(N)$ sequence scan | **$O(1)$ Low-Rank Matrix Multiplication** |
| **State Size @ 1024 Context** | ~480 MiB (expanding linearly) | **11.55 MiB (Strictly Constant)** |
| **State Size @ 8192 Context** | ~3.84 GiB (OOM risk on edge) | **11.55 MiB (Completely Flat)** |
| **Layer Architecture** | 35 Mixed Sliding & Full Layers | **35 Transmuted Zero-KV Tri-Memory Blocks** |
| **Weight Modification** | Standard Base Weights | **Zero Base Weight Alteration (Surgical Plug-in)** |

---

## Core Components

1. **R48 Associative Recurrence ($S, Z$)**:
   Bounded low-rank associative memory ($M=48$) driven by depth-calibrated temporal decay $\gamma$ and diverse landmark basis projections. Captures long-range document context with constant step complexity.
2. **Natural Haar Wavelet Pyramid**:
   Hierarchical multi-resolution ring buffer maintaining immediate fine tokens (64 slots) alongside coarse $2\times$ and $4\times$ averaged temporal summaries.
3. **Salient Cue-Span & Novelty HDC State**:
   16 initial sink slots preserving system prompt anchors, combined with a 144-slot novelty bank tracking information surprises ($\|k - k_{prev}\|_2$) to retain critical factual and numerical tokens permanently.
4. **Logit Softcapping & Shared-KV Handling**:
   Preserves Gemma's native logit softcapping (`torch.tanh(scores / softcap) * softcap`) and routes producer/consumer Key-Value tensors across shared attention layers via an internal registry.

---

## Directory Structure

```text
├── gemma4_wrai_pipeline.py    # Master Distillation Pipeline (calibration, loss tracking, and .pt export)
├── wrai_gemma4_inference.py   # High-Performance Zero-KV Standalone Engine (built-in priors + CLI)
├── requirements.txt           # Minimal dependencies
├── .gitignore                 # Exclusion rules
├── LICENSE                    # Apache 2.0 License
└── README.md                  # Comprehensive documentation
```

---

## Quickstart Guide

### 1. Installation

```bash
git clone https://github.com/MusounoEnma/WRAI.git
cd "WRAI/Gemma 4 E2B"
pip install -r requirements.txt
```

### 2. Run Inference Immediately (Zero-Shot Built-in Priors)

WRAI Gemma 4 includes built-in Stage 17 Universal Distilled Priors. You can launch standalone inference directly with zero preparation:

```bash
# Interactive Chat Mode
python wrai_gemma4_inference.py --interactive

# Single Prompt Execution
python wrai_gemma4_inference.py \
  --prompt "Explain quantum computing in simple terms." \
  --max-new-tokens 128
```

### 3. Run Inference with a Distilled Checkpoint

If you have trained or downloaded a calibrated `.pt` checkpoint:

```bash
python wrai_gemma4_inference.py \
  --checkpoint ./wrai_gemma4_release/wrai_gemma4_universal_checkpoint.pt \
  --max-new-tokens 128 \
  --interactive
```

### 4. Running the Distillation Pipeline

To run the calibration and distillation pipeline from scratch using Hugging Face weights:

```bash
python gemma4_wrai_pipeline.py \
  --model-id google/gemma-4-E2B \
  --output-dir ./wrai_gemma4_release \
  --steps 45
```

---

## Citation & Technical Report

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
