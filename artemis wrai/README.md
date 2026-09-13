# 🏹 WRAI-Artemis: On-Device Vision Agent Engine for Google ARTEMIS

> **Target Hardware**: MediaTek Dimensity 1100 (8 GB LPDDR4X RAM) / Android Edge SoC  
> **Transmutation Backbone**: Google Gemma 3n (Multimodal On-Device)  
> **Agent Specializations**: WhatsApp Multi-turn Replying, Google Maps Navigation, Chrome Web Research  
> **Status**: Workspace Isolated for Google Colab Adaptation (`artemis wrai/`)

---

## 🎯 Project Overview

**WRAI-Artemis** is an on-device, sub-quadratic vision-agent engine designed specifically to power the **[Google ARTEMIS](https://github.com/google/artemis)** mobile automation framework. 

While conventional Transformer models suffer from catastrophic context window blowout and quadratic latency degradation during multi-turn mobile operation (due to accumulating high-resolution UI screenshots in the KV-Cache), WRAI-Artemis operates with **strictly $O(1)$ constant memory** across hundreds of interaction turns.

---

## 🏗️ Architectural Pillars

```text
 ┌─────────────────────────────────────────────────────────────┐
 │                    GOOGLE GEMMA 3n                          │
 │         (GeGLU Feed-Forward Network + RMSNorm)              │
 └──────────────────────────────┬──────────────────────────────┘
                                │
                                ▼
 ┌─────────────────────────────────────────────────────────────┐
 │       5-LAYER SURGICAL VOCABULARY PRUNER (EN + ID)          │
 │  256,000 Multilingual Tokens ──► 32,768 Tokens              │
 │  - 256 UTF-8 Byte Fallbacks (<0x00>-<0xFF>) Locked          │
 │  - Android UI Action Tokens (<click>, <type>, <scroll>)     │
 │  - Normalized Coordinate Bins (<loc_000>-<loc_999>)         │
 │  - Parameter Reduction: >500 Million Parameters Saved!      │
 └──────────────────────────────┬──────────────────────────────┘
                                │
                                ▼
 ┌─────────────────────────────────────────────────────────────┐
 │            WRAI DUAL-STATE RECURRENT RETENTION              │
 │  1. Memory State (Mt): S_t = γ_m · S_{t-1} + K_t^T · V_t    │
 │  2. 4-Level Discrete Haar Wavelet Filter (Global vs Local)  │
 │  3. Reasoning State (Rt): Deep Action Planning              │
 │  4. HDC Scratchpad (S_hdc): Holographic Multi-App Memory    │
 │     b_t = tanh(W_hk · r_t) ⊙ tanh(W_hv · r_t)               │
 └──────────────────────────────┬──────────────────────────────┘
                                │
                                ▼
 ┌─────────────────────────────────────────────────────────────┐
 │             INT8 ROW-WISE QUANTIZED BINARY                  │
 │    Footprint: ~1.2 GB | Execution: Native ARM NEON SIMD     │
 └─────────────────────────────────────────────────────────────┘
```

---

## 📁 Workspace Contents

| File | Description |
| :--- | :--- |
| **[`WRAI_ARTEMIS_COLAB.ipynb`](file:///c:/porto/11MYPORTO/WRAI/artemis%20wrai/WRAI_ARTEMIS_COLAB.ipynb)** | Complete 1-click Google Colab Jupyter Notebook for GPU adaptation. |
| **[`audit_and_prune_vocab.py`](file:///c:/porto/11MYPORTO/WRAI/artemis%20wrai/audit_and_prune_vocab.py)** | 5-Layer Safety Audit Guard for vocabulary slicing with round-trip checks. |
| **[`colab_wrai_artemis_gemma3n.py`](file:///c:/porto/11MYPORTO/WRAI/artemis%20wrai/colab_wrai_artemis_gemma3n.py)** | Core model definitions, retention blocks, HDC scratchpad, and dataset pairs. |

---

## 🛡️ 5-Layer Vocabulary Audit Protocol

1. **Byte Fallback Lock**: Preserves all 256 byte-level fallbacks (`<0x00>` to `<0xFF>`) to guarantee the model never throws `<unk>` errors on unusual text or emojis.
2. **UI Action Whitelist**: Embeds Android UI actions (`<click>`, `<type>`, `<scroll>`, `<open_app>`, `<loc_000>`-`<loc_999>`).
3. **Multi-Domain Empirical Profiling**: Evaluates token occurrence frequency across formal Indonesian/English, colloquial WhatsApp chats, Android XML, and shell commands.
4. **Round-Trip Exact Match**: Validates that $\text{Detokenize}(\text{Tokenize}(T)) == T$ with 100% character-level accuracy.
5. **Inflation Ratio Constraint**: Ensures average subword tokenization length increases by less than 10%.

---

## 🚀 Quick Start (Google Colab)

1. Upload `WRAI_ARTEMIS_COLAB.ipynb`, `audit_and_prune_vocab.py`, and `colab_wrai_artemis_gemma3n.py` to your Google Drive or Colab.
2. Select a GPU Runtime (`Runtime` > `Change runtime type` > `T4` or `A100`).
3. Run all cells to execute vocabulary pruning, WRAI recurrent transmutation, and INT8 export.
