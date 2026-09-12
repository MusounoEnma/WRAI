# 🧠 WRAI-X (0.8B) Model Weights

Folder ini menyimpan bobot model **WRAI-X (0.8B)** hasil transplantasi:

| Nama File | Ukuran | Format | Deskripsi |
| :--- | :--- | :--- | :--- |
| `wrai_x_vocab.bin` | **1.52 MB** | Binary BPE Table | Sudah disertakan langsung di repositori Git. |
| `wrai_x_08b_int8.bin` | **1.35 GB** | INT8 Row-wise (mmap) | Biner terkuantisasi untuk eksekusi di Native C Engine. |
| `wrai_x_08b_transplanted.pt` | **1.66 GB** | PyTorch Checkpoint | Bobot asli untuk fine-tuning lanjutan di Google Colab. |

---

## 📥 Cara Mendapatkan Bobot Model (Download)

Karena file `.bin` (1.35 GB) dan `.pt` (1.66 GB) melebihi batas 100 MB GitHub, bobot dihosting di **Hugging Face Model Hub**:  
👉 **[https://huggingface.co/Musouno-Enma99/WRAI-X-0.8B-Qwen3](https://huggingface.co/Musouno-Enma99/WRAI-X-0.8B-Qwen3)**

### Cara 1: Menggunakan Script Python Otomatis
Cukup jalankan script downloader (akan mengunduh otomatis dari repo Hugging Face):
```bash
python download_weights.py
```

### Cara 2: Download Manual via Browser / Wget
Unduh file langsung dari [Hugging Face Repository](https://huggingface.co/Musouno-Enma99/WRAI-X-0.8B-Qwen3), lalu letakkan di dalam folder `qwen/` ini:
* `wrai_x_08b_int8.bin`
* `wrai_x_08b_transplanted.pt`
