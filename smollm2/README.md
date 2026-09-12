# 🌊 WRAI-Smol: SmolLM2 Wavelet-Retention Transplantation Workspace

Welcome to the **WRAI-Smol** working directory. This project adapts the **WRAI (Wavelet Retention AI)** architectural transmutation framework to the **SmolLM2** model family developed by Hugging Face (`HuggingFaceTB/SmolLM2-135M-Instruct` and `SmolLM2-360M-Instruct`).

---

## 🎯 Strategic Objective

1. **Universal Portability Proof**: Demonstrate that the WRAI transplant methodology operates reliably on standard LLaMA-family architectures outside the Alibaba Qwen ecosystem.
2. **Ultra-Edge Footprint**: Leverage the compact $d_k = 64$ head dimension to produce the world's most lightweight recurrent wavelet language model:
   * **State Matrix ($64 \times 64$)**: Only 4,096 elements per head (**$4\times$ less compute** than Qwen's $128 \times 128$).
   * **INT8 Binary File**: ~**140 MB** for the 135M variant, enabling sub-second downloads and zero-heap execution on low-spec hardware.
3. **Reasoning Persistence**: Retain structured Chain-of-Thought (`<think> ... </think>`) reasoning cycles using ChatML prompting format.

---

## 📐 Architectural Comparison: Qwen3 vs. SmolLM2

| Architectural Dimension | WRAI-X (Qwen3-0.6B Base) | WRAI-Smol (135M Base) | WRAI-Smol (360M Base) |
| :--- | :---: | :---: | :---: |
| **Base Model Hub** | `Qwen/Qwen3-0.6B` | `HuggingFaceTB/SmolLM2-135M-Instruct` | `HuggingFaceTB/SmolLM2-360M-Instruct` |
| **Number of Layers ($L$)** | 28 | **30** | 32 |
| **Hidden Dimension ($D$)** | 1024 | **576** | 960 |
| **Attention / Retention Heads ($H$)** | 16 | **9** | 15 |
| **Head Dimension ($d_k$)** | 128 | **64** | **64** |
| **Retention Matrix Dimensions** | $128 \times 128$ (16,384 floats) | **$64 \times 64$ (4,096 floats)** | **$64 \times 64$ (4,096 floats)** |
| **Intermediate FFN Dimension** | 3072 (SwiGLU) | 1536 (SwiGLU) | 2560 (SwiGLU) |
| **Vocabulary Size** | 151,936 tokens | **49,152 tokens** | **49,152 tokens** |
| **Persistent State Buffer** | 29.42 MB | **~8.85 MB (FP16)** | **~19.66 MB (FP16)** |
| **Estimated INT8 Binary Size** | ~1.35 GB | **~140 MB** | **~360 MB** |
| **Colab T4 Training Duration** | ~2.5 hours | **~20 minutes** | **~45 minutes** |

---

## 📁 Directory Structure

```
smollm2/
├── README.md                           # This architecture roadmap & execution guide
├── configs/                            # Model configuration definitions
│   ├── smollm2_135m.json              # 135M hyperparameter profile
│   └── smollm2_360m.json              # 360M hyperparameter profile
├── engine/                             # Native C Engine adapter
│   └── wrai_smollm2_config.h          # C header definitions and SIMD kernel shapes
└── training/                           # PyTorch surgery & distillation pipeline
    ├── colab_train_wrai_smollm2_transplant.py # End-to-end transplant script for Colab
    └── quantize_wrai_smollm2_colab.py # INT8 row-wise binary exporter
```

---

## 🚀 Execution Roadmap

- [x] **Phase 1: Architecture Profiling & Tokenizer Audit**: Verified $64 \times 64$ state matrix shapes and ChatML token compatibility for `<think>` sequences.
- [ ] **Phase 2: Colab Surgery & Distillation**: Run `colab_train_wrai_smollm2_transplant.py` on Google Colab T4 GPU (~20 mins).
- [ ] **Phase 3: INT8 Quantization**: Run `quantize_wrai_smollm2_colab.py` to generate `wrai_smollm2_135m_int8.bin` (~140 MB).
- [ ] **Phase 4: Pure C Engine Verification**: Compile and run `wrai_x_engine` with SmolLM2 configurations on CPU hardware.
