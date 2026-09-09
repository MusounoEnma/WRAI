# Laporan Rencana Pelatihan Deep Reasoning & Coding WRAI v8

---

## 1. Konsep Deep Teacher Distillation (4.000+ Q&A Samples)
Knowledge Distillation diperluas dengan menggabungkan 3 dataset industri raksasa dari HuggingFace:

1. **`cahya/alpaca-id-cleaned` (1.200+ samples):** Instruksi Bahasa Indonesia & Penalaran Pengetahuan Umum.
2. **`iamtarun/python_code_instructions_18k_alpaca` (1.200+ samples):** Sintaksis Koding Python, Pembuatan Fungsi & Struktur Data.
3. **`microsoft/orca-math-word-problems-200k` & `camel-ai/math` (1.200+ samples):** Logika Penalaran Matematika Step-by-Step.

---

## 2. Peningkatan Arsitektur Model: 3-Layer Recurrent State Core

```
 ┌────────────────────────────────────────────────────────┐
 │           COLAB GPU TRAINER (colab_train_wrai.py)      │
 │                                                        │
 │  - 20.000 Epochs BPTT GPU Accelerated                 │
 │  - 3-Layer Stacked Recurrent Network (256-dim State)   │
 │  - Layer 1: Features | Layer 2: Logic | Layer 3: Code  │
 └───────────────────────────┬────────────────────────────┘
                             │ Export Quantized Q31 Weights
                             ▼
 ┌────────────────────────────────────────────────────────┐
 │            STUDENT AI (WRAI v8 Engine Laptop)          │
 │  - File Model Biner: wrai_nextgen_v8.bin (~3.5 MB)     │
 │  - Latensi Inferensi: < 3.5 ms di CPU                  │
 │  - RAM Usage: < 45 MB                                  │
 └────────────────────────────────────────────────────────┘
```

---

## 3. Langkah Eksekusi Praktis di Google Colab

1. **Buka Google Colab:** [colab.research.google.com](https://colab.research.google.com/)
2. **Set Hardware Accelerator:** Pilih `T4 GPU` atau `A100 GPU`.
3. **Run Code Script:** Copy-paste isi file local project: [colab_train_wrai.py](file:///c:/porto/11MYPORTO/WRAI/tools/colab_train_wrai.py).
4. **Waktu Pelatihan:** **20.000 Epochs BPTT** selesai dalam kurun waktu **~1 hingga 2 menit** di GPU Colab.
5. **Download Output Model:** Download `wrai_nextgen_v8.bin` & `wrai_nextgen_v8.json` dan letakkan di folder `models/` project lokal PC.
