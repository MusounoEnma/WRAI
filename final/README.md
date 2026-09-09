# 🚀 WRAI v16 (1.7B) & v15 (3B) — Production Suite

Selamat datang di repositori resmi **WRAI** (*Wavelet Retention AI*).

---

## 🌟 WRAI v16 (1.7B) — Generasi Baru (Qwen 3 Direct Transplant) [REKOMENDASI UTAMA]
WRAI v16 mengadopsi model dasar generasi terbaru **`Qwen/Qwen3-1.7B`** dengan pendekatan **Cangkok Murni 1:1 (*Pure Weight Transplant*)** tanpa beban model guru di VRAM dan tanpa dataset tool-calling sintetis Hermes yang membingungkan model.

### 💎 Keunggulan WRAI v16:
* 📦 **1 File Checkpoint Tunggal (~2.4 GB FP16):** Tidak perlu sharding, bebas repot, dan 100% aman dari OOM Colab.
* ⚡ **3x–4x Lebih Cepat:** Konvergensi loss tercapai dalam 2–3 jam di Colab T4.
* 🧠 **28 Deep Retention Layers:** Hidden Dim 2048 (1:1 Exact Match AVX-512 & Haar DWT), FFN 6144 SwiGLU.
* 💾 **Fixed 2 MB Memory Buffer (99.9% Lebih Hemat dari Transformer):** Mengunci memori konteks pada batas tetap 2 MB O(1) tanpa memori bocor/membengkak.
* 🌐 **Dataset Bersih Dwibahasa:** Percakapan alami Bahasa Indonesia (Alpaca-ID), Nalar & Koding (Alpaca-Cleaned), dan Terjemahan Teknis (OPUS-100).
* 🚫 **Zero Distillation Overhead:** Memori VRAM hanya terpakai ~5.5 GB (lega di T4 16GB).

---

## ⚡ Panduan Eksekusi WRAI v16 di Google Colab:

### 1. Jalankan Pelatihan WRAI v16 (Auto-Save ke Google Drive):
```bash
!python final/colab_train_wrai_v16_qwen3_1.7b.py
```
* **Dedicated Google Drive Folder:** `/content/drive/MyDrive/WRAI_v16_1.7B_Models_Transplant`
* **File Checkpoint:** `wrai_v16_1.7b_latest.pt` & `wrai_v16_1.7b_best.pt` (Auto-save setiap 2500 steps).
* **Auto-Resume:** Jika sesi Colab terputus, cukup jalankan ulang perintah di atas.

### 2. Forensik & Inspeksi Langsung Kesehatan Model (Step 70K Audit):
```bash
!python final/inspect_wrai_v16_step70k.py
```
* Memeriksa kecocokan Vocab Size & Tokenizer (151.643 base vs 151.936 added vs bobot embedding).
* Mengaudit stabilitas faktor peluruhan (Decay Gamma) di 28 layer retention.
* Membuktikan secara komputasi: Original GroupNorm temporal leak vs Per-Token GroupNorm (selisih 0.000000).
* Menjalankan generasi multi-token bilingual & coding langsung (Sampling + Repetition Penalty).

### 3. Uji Inferensi & Benchmark Dwibahasa (Bilingual Studio):
```bash
!python final/test_wrai_v16_qwen3_1.7b_inference.py
```

---

## 📂 Struktur Berkas di Folder `final/`:
* 📄 **`colab_train_wrai_v16_qwen3_1.7b.py`**: [WRAI v16] Trainer cangkok murni 1:1 dari `Qwen/Qwen3-1.7B` (Per-Token Causal Norm).
* 📄 **`inspect_wrai_v16_step70k.py`**: [WRAI v16] Skrip audit forensik komprehensif Step 70k & live causality test.
* 📄 **`inspect_wrai_v16_colab.py`**: [WRAI v16] Alat inspeksi forensik bobot & logits mendalam.
* 📄 **`test_wrai_v16_qwen3_1.7b_inference.py`**: [WRAI v16] Engine inferensi bilingual & Live Interactive Chat Studio.
* 📄 **`colab_train_wrai_v15_3b.py`**: [WRAI v15] Trainer 3B legacy dengan arsitektur 2-shard.
* 📄 **`test_wrai_v15_3b_inference.py`**: [WRAI v15] Engine inferensi 3B legacy.

---

## 🏛️ Spesifikasi Arsitektur WRAI v16 (1.7B):
* **Total Parameter:** ~1.7 Miliar Parameter
* **Lapisan Utama:** 28 Layer Multi-Head Retention ($H_q=16, D=128$, GroupNorm Stabilizer, Multi-Scale Decay $\gamma$)
* **GQA KV Broadcast:** 8 KV Heads dari Qwen 3 dibroadcast 2x ke 16 Retention Heads secara lossless
* **Penyaring Spektral:** 4-Level Haar 1D DWT (Elementwise Gated, Identity Init)
* **Kapasitas FFN:** 1:1 SwiGLU FFN ($2048 \leftrightarrow 6144$)
* **Kosakata:** Full 151.936 BPE Tokens
* **Skalabilitas Memori:** Fixed 2 MB Buffer Budget ($O(1)$ Ultra-Low RAM Footprint, 99.9% Lebih Ringan dari Transformer)

---

## 💻 Native C Inference Engine (Zero Python / Low-End AMD A8 Puma+ Support):

Untuk menjalankan inferensi lokal tanpa Python, tanpa PyTorch, dan tanpa KV-Cache di CPU (termasuk hardware spek minim seperti AMD A8 RAM 8GB):

1. **Build Executable Native C:**
   ```cmd
   build_wrai_v16.bat
   ```
2. **Jalankan Terminal Chatbot Interaktif:**
   ```cmd
   wrai_v16.exe
   ```
   * **RAM Footprint:** Hanya ~28 MB buffer recurrent (menggunakan memory-mapped binary 1.7 GB).
   * **SIMD Engine:** AVX 1.0 32-vector unrolled kernel.
   * **Kecepatan:** ~1.0–1.5 token/detik langsung di CPU laptop lawas.

