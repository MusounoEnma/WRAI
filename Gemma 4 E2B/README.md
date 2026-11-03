# WRAI x Google Gemma 4 E2B: Zero-KV Tri-Memory Inference Engine
### Transforming 35-Layer Heterogeneous Transformer Attention into $O(1)$ Constant State

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.9%2B-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/Model-google%2Fgemma--4--E2B-yellow.svg)](https://huggingface.co/google/gemma-4-E2B)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.23920132-blue.svg)](https://doi.org/10.5281/zenodo.23920132)

---

## Overview

**WRAI (Wavelet / Recurrent / Attentive Intelligence)** for **Google Gemma 4 E2B** (`google/gemma-4-E2B`) adapts the surgical Tri-Memory architecture to Google's state-of-the-art heterogeneous LLM backbone.

Standard autoregressive generation on Gemma 4 E2B suffers from linear $O(N)$ KV-cache accumulation across its 35 deep text layers. WRAI replaces every native attention block with **Universal Zero-KV WRAI Attention**, completely freezing memory consumption during generation:

* **0.0000 MiB Dynamic KV Cache**: Zero allocations during token generation.
* **11.5492 MiB Strictly Constant $O(1)$ State**: Fixed memory footprint across infinite sequence lengths.
* **Stage 17 Universal Distilled Priors**: Built-in and bundled pre-calibrated priors $(\alpha^*, \gamma^*, \text{wavelet}^*)$ for all 35 layers, enabling immediate zero-KV inference.
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
| **Backbone Weight Modification** | Standard Base Weights | **Zero Base Weight Alteration (Surgical Plug-in)** |

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
├── gemma4_wrai_pipeline.py              # Master Distillation Pipeline (calibrates 665 scalars & exports .pt)
├── wrai_gemma4_inference.py             # High-Performance Zero-KV Standalone Engine (CLI + chat)
├── wrai_gemma4_universal_checkpoint.pt  # Bundled Pre-calibrated Stage 17 Checkpoint (48 KB, 665 params)
├── requirements.txt                     # Minimal dependencies (torch, transformers, accelerate)
├── .gitignore                           # Exclusion rules (whitelists checkpoint, excludes raw base weights)
├── LICENSE                              # Apache 2.0 License
└── README.md                            # Comprehensive documentation & reproduction guide
```

---

## Prerequisites & Model Architecture Notes

> [!NOTE]
> **Base Foundation Weights Requirement**
> 
> Gemma 4 E2B is Google's multimodal 2B parameter foundation model (`google/gemma-4-E2B`). The raw foundation weights (`model.safetensors`) comprise **~9.54 GB** in BF16 precision.
> 
> Consistent with open-source LLM engineering standards (e.g. vLLM, SGLang, Hugging Face), this repository does not re-host Google's raw 9.5 GB backbone weights. Instead, WRAI provides:
> 1. The surgical zero-KV transmutation architecture.
> 2. The master distillation pipeline (`gemma4_wrai_pipeline.py`) to calibrate the 665 WRAI parameters.
> 3. The bundled pre-calibrated lightweight checkpoint (`wrai_gemma4_universal_checkpoint.pt`, **~48 KB**).
> 
> When executing either script, the base weights are either loaded from local disk via `--model-id /path/to/local/gemma-4-E2B` or automatically fetched from Hugging Face hub (`google/gemma-4-E2B`).
> 
> **Hardware Recommendation:**
> - GPU: $\ge$ 12 GB VRAM (NVIDIA RTX 3060/4070 or better) recommended for the calibration pipeline.
> - CPU / RAM: $\ge$ 16 GB system RAM for inference.

---

## Complete Reproduction Guide

### 1. Installation

```bash
git clone https://github.com/MusounoEnma/WRAI.git
cd "WRAI/Gemma 4 E2B"
pip install -r requirements.txt
```

---

### 2. Path A: Master Distillation Pipeline (Calibrating from Scratch)

To execute the scientific distillation pipeline from raw `google/gemma-4-E2B` weights and generate a new calibrated checkpoint:

```bash
# Full Master Calibration (45 steps per layer across 4 domains)
python gemma4_wrai_pipeline.py \
  --model-id google/gemma-4-E2B \
  --output-dir ./wrai_gemma4_release \
  --steps 45 \
  --device cuda
```

**What this pipeline executes:**
1. Loads base `google/gemma-4-E2B` weights and freezes 100% of backbone parameters.
2. Captures multi-domain native attention targets across Literature, Code, Science, and Reasoning domains.
3. Optimizes 665 universal WRAI parameters (19 scalars $\times$ 35 layers) via AdamW.
4. Enforces hardware poison traps verifying zero quadratic SDPA access.
5. Exports the calibrated release bundle:
   - `./wrai_gemma4_release/wrai_gemma4_universal_checkpoint.pt` (~48 KB)
   - `./wrai_gemma4_release/wrai_config.json` (metadata & layer metrics)

---

### 3. Path B: Run Zero-KV Inference with Checkpoint

Once a checkpoint is generated (or using the bundled `wrai_gemma4_universal_checkpoint.pt`):

```bash
# Interactive Chat Mode
python wrai_gemma4_inference.py \
  --checkpoint wrai_gemma4_universal_checkpoint.pt \
  --interactive

# Single Prompt Execution
python wrai_gemma4_inference.py \
  --checkpoint wrai_gemma4_universal_checkpoint.pt \
  --prompt "Explain quantum computing in simple terms." \
  --max-new-tokens 128
```

---

### 4. Path C: Built-in Stage 17 Priors & Fast Checkpoint Export

`wrai_gemma4_inference.py` includes hardcoded Stage 17 Universal Distilled Priors. If `--checkpoint` is omitted, the engine automatically uses these built-in priors:

```bash
# Run inference using built-in Stage 17 priors
python wrai_gemma4_inference.py --interactive
```

To export the built-in Stage 17 priors into a new standalone `.pt` file without running training:

```bash
python wrai_gemma4_inference.py --export-checkpoint ./exported_stage17_checkpoint.pt
```

---

### 5. Path D: Air-Gapped / Offline Local Execution

If working in an isolated or air-gapped environment without direct Hugging Face internet access:

1. Pre-download the base model folder (`google/gemma-4-E2B`) to a local path (e.g., `/data/models/gemma-4-E2B`).
2. Run the pipeline or inference by pointing `--model-id` to the local folder:

```bash
# 1. Distill offline from local folder
python gemma4_wrai_pipeline.py \
  --model-id /data/models/gemma-4-E2B \
  --output-dir ./wrai_gemma4_release

# 2. Run inference offline from local folder
python wrai_gemma4_inference.py \
  --model-id /data/models/gemma-4-E2B \
  --checkpoint ./wrai_gemma4_release/wrai_gemma4_universal_checkpoint.pt \
  --interactive
```

---

## Hardware Guarantees & Poison Traps

During inference, WRAI enforces strict hardware invariants:
1. **Zero Dynamic KV Cache**: Allocations in `DynamicCache` are 0.00 Bytes across all 35 layers.
2. **Fail-Closed SDPA Trap**: `torch.nn.functional.scaled_dot_product_attention` is globally hijacked to raise a fatal exception if invoked.
3. **Strict $O(1)$ State Flatline**: Tri-Memory state remains exactly **11.5492 MiB** whether context length is 10 tokens or 100,000 tokens.

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
