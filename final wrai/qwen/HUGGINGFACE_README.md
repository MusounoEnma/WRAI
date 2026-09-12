---
license: apache-2.0
language:
- id
- en
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
base_model: Qwen/Qwen2.5-0.5B
pipeline_tag: text-generation
library_name: c-native
---

# 🌊 WRAI-X (0.8B): Experimental Wavelet-Retention Hybrid

> **⚠️ Status Riset:** Model ini adalah **Proof-of-Concept (PoC) Eksperimental** untuk menguji transplantasi arsitektur Transformer ke Recurrent Retention + Discrete Haar Wavelet. Ini **bukan** model produksi siap pakai, melainkan artefak riset terbuka bagi komunitas kecerdasan buatan dan sistem komputasi berdaya rendah.

* **GitHub Repository:** [https://github.com/YourUsername/WRAI](https://github.com/YourUsername/WRAI) *(Ganti dengan link repo Anda)*
* **Base Pre-trained Brain:** `Qwen/Qwen2.5-0.5B` (Frozen FFN, RMSNorm, & Vocab)
* **Total Parameter Riil:** **831.268.848 Parameter** (~0.83B)
* **Format Bobot Tersedia:**
  * `wrai_x_08b_int8.bin` (1.35 GB) — *Biner INT8 mmap untuk Native C Inference Engine*
  * `wrai_x_08b_transplanted.pt` (1.66 GB) — *PyTorch Checkpoint untuk eksperimen & training lanjutan*

---

## 📌 Apa itu WRAI-X?

Transformer standar memiliki tantangan fundamental: **KV-Cache yang membengkak secara linear $O(T)$** seiring bertambahnya token konteks, yang membebani memori RAM dan GPU.

Alih-alih melatih model bahasa dari awal dengan biaya komputasi yang masif, proyek WRAI-X mengeksplorasi teknik **Architectural Transmutation (Pencangkokan Arsitektur)**:
1. **Meminjam Otak Pengetahuan**: Mengambil bobot FFN SwiGLU, RMSNorm, dan Embedding dari **Qwen2.5-0.5B** dalam keadaan dibekukan (*100% frozen*).
2. **Mengganti Self-Attention dengan Dual-State Retention ($M_t / R_t$)**: Berlandaskan konsep *RetNet (Microsoft Research)*, memori konteks disimpan dalam matriks berukuran tetap ($128 \times 128$) yang di-update secara rekursif (*in-place*).
3. **Filter Spektral Haar DWT 4-Level**: Memisahkan representasi laten menjadi komponen frekuensi rendah (konteks semantik global) dan komponen frekuensi tinggi (sintaksis lokal).
4. **Engine C Murni (Zero-Heap mmap)**: Menjalankan inferensi mandiri di CPU x86_64 dengan SIMD AVX tanpa ketergantungan PyTorch/Python.

---

## 🔬 Keterbukaan & Batasan Saat Ini (Stay Humble & No Hype)

Kami percaya pada transparansi ilmiah penuh tanpa klaim yang dilebih-lebihkan:

### ✅ Apa yang Sudah Terbukti & Berjalan Baik:
* **Stabilitas Numerik Rekursif**: Mekanisme decay retention $\gamma$ dan normalisasi RetNet GroupNorm per-head terbukti stabil dan tidak meledak (*NaN/Inf free*) pada inferensi sekuensial panjang.
* **Murni 0% KV-Cache ($O(1)$ RAM)**: Konsumsi RAM fisik proses terbukti datar (**+0.00 MB delta**) dari $T=1$ hingga ribuan token.
* **Efisiensi Native C**: Mampu berjalan di CPU komputer lama/hemat daya (teruji di prosesor AMD A8 Puma+ 2014) dengan kecepatan 2 – 3 token/detik murni CPU.
* **Struktur Output**: Berhasil mengadopsi format Chain-of-Thought (`<think> ... </think>`) dan merespons sapaan.

### ⚠️ Batasan & Hal yang Masih Perlu Dikembangkan:
* **Koherensi Tata Bahasa**: Checkpoint rilis ini adalah **Stage-1 Checkpoint** (tahap awal adaptasi router/gating). Tata bahasa dan penalaran kompleks masih belum sebaik Transformer penuh dan terkadang menghasilkan pengulangan (*looping/stuttering*).
* **Alpha Gate Masih Rendah**: Parameter gate transplantasi masih dalam rentang konservatif (~0.01) untuk menjaga kestabilan sinyal residual.
* **Tujuan Penggunaan**: Sangat direkomendasikan untuk **peneliti arsitektur AI, mahasiswa, dan engineer sistem** yang tertarik mempelajari bagaimana *memory-efficient recurrent state* bekerja di level C/hardware.

---

## 📊 Hasil Audit Forensik OS & Hardware

Metrik performa diukur langsung pada kernel **Windows NT API (`psapi.h`)** saat memuat `wrai_x_08b_int8.bin`:

| Metrik Hardware / OS | Nilai Riil Terukur | Keterangan Forensik |
| :--- | :--- | :--- |
| **Ukuran File Biner di Disk** | **1.422.927.244 bytes** (~1.35 GB) | File bobot INT8 row-wise |
| **Physical RAM (Working Set)** | **912.90 MB** | Kapasitas chip RAM fisik yang terisi bobot aktif |
| **Hardware Page Faults (MMU)** | **234.259 halaman** | Bukti transfer fisik blok 4 KB dari SSD ke RAM oleh CPU |
| **Beban Komputasi AVX** | **~1.30 GFLOPs / token** | Eksekusi nyata perkalian matriks di register AVX 256-bit |
| **Skalabilitas KV-Cache** | **Delta 0.00 MB ($O(1)$)** | Memori konteks konstan flat (~14.5 MB State Buffer) |

---

## 🚀 Cara Menjalankan

### Opsi A: Menggunakan Pure Native C Engine (Windows / Linux)
1. Unduh kode sumber dari repositori GitHub:
   ```bash
   git clone https://github.com/YourUsername/WRAI.git
   cd WRAI/"final wrai"
   ```
2. Letakkan file `wrai_x_08b_int8.bin` di subfolder `qwen/`.
3. Klik dua kali **`run_wrai_x.bat`** (atau jalankan `.\engine\wrai_x.exe`).

### Opsi B: Menggunakan PyTorch (Google Colab / Python)
Gunakan checkpoint `wrai_x_08b_transplanted.pt` bersama script training & evaluasi yang tersedia di folder `training/` di repositori GitHub kami:
* `training/colab_train_wrai_x_08b_transplant.py`
* `training/WRAI_X_08B_COLAB.ipynb`

---

## 🙏 Landasan Teori & Ucapan Terima Kasih (Acknowledgements)

Proyek eksperimental ini terwujud berkat karya luar biasa dari komunitas peneliti AI dunia:
1. **Qwen Team (Alibaba Cloud)**: Atas model dasar [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) di bawah lisensi Apache 2.0 yang menyediakan fondasi representasi bahasa berkualitas tinggi.
2. **Microsoft Research (RetNet - Sun et al., 2023)**: Atas perumusan arsitektur *Retentive Network* yang membuktikan kemungkinan inferensi recurrent berdimensi tetap $O(1)$.
3. **Alfréd Haar (1909) & Komunitas Pengolahan Sinyal**: Atas formulasi *Discrete Haar Wavelet Transform (DWT)* untuk dekomposisi fitur multiskala.
4. **Pentti Kanerva & Komunitas Hyperdimensional Computing (HDC)**: Atas konsep memori asosiatif vektor berdimensi tinggi.
5. **Georgi Gerganov (*llama.cpp*)**: Atas inspirasi rekayasa C murni dan pemanfaatan *zero-heap memory-mapping*.

---

## 📜 Lisensi
Model dan seluruh kode turunan ini dilisensikan di bawah **Apache License 2.0**.
