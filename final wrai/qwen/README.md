# 🧠 WRAI-X (0.8B-Class / ~0.83B) Model Weights

This directory stores the **WRAI-X (0.8B-class / ~0.83B, 831M parameters)** model weight artifacts:

| Filename | Size | Format | Description |
| :--- | :--- | :--- | :--- |
| `wrai_x_vocab.bin` | **1.52 MB** | Binary BPE Table | Included directly in this Git repository. |
| `wrai_x_08b_int8.bin` | **1.35 GB** | INT8 Row-wise (mmap) | Quantized binary for the Native C Inference Engine. |
| `wrai_x_08b_transplanted.pt` | **1.66 GB** | PyTorch Checkpoint | Floating-point checkpoint for research & continued fine-tuning. |

---

## 📥 Downloading Model Weights

Because `.bin` (1.35 GB) and `.pt` (1.66 GB) exceed GitHub's 100 MB file limit, model weights are hosted on the **Hugging Face Model Hub**:  
👉 **[https://huggingface.co/Musouno-Enma99/WRAI-X-0.8B-Qwen3](https://huggingface.co/Musouno-Enma99/WRAI-X-0.8B-Qwen3)**

### Method 1: Automated Python Script (Recommended)
Run the automated downloader helper (requires `huggingface_hub`):
```bash
python download_weights.py
```

### Method 2: Manual Download via Browser / Wget
Download directly from the [Hugging Face Repository](https://huggingface.co/Musouno-Enma99/WRAI-X-0.8B-Qwen3) and place the files inside this `qwen/` directory:
* `wrai_x_08b_int8.bin`
* `wrai_x_08b_transplanted.pt`
