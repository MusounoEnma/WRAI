# 📦 WRAI-X (0.8B Qwen) Production Scripts Archive

This directory stores the official, audited production scripts used to train, verify, and quantize the **WRAI-X (0.8B-class / ~0.83B)** model transplanted from Qwen3-0.6B.

## 📄 Production Script Manifest

| Script | Purpose | Status |
| :--- | :--- | :--- |
| **`colab_train_wrai_x_08b_transplant.py`** | Official end-to-end PyTorch transplant distillation script for Google Colab. | ✅ Verified & Deployed |
| **`quantize_wrai_x_08b_colab.py`** | PyTorch FP32 checkpoint -> INT8 symmetric row-wise quantized binary exporter (`wrai_x_08b_int8.bin`). | ✅ Verified & Deployed |
| **`WRAI_X_08B_COLAB.ipynb`** | Standalone interactive Google Colab notebook for 1-click execution. | ✅ Verified & Deployed |
| **`verify_zero_kv_cache_deep_dive.py`** | Rigorous 5-point audit script verifying 29.42 MB constant state memory and $O(1)$ scaling. | ✅ Verified & Deployed |
| **`test_wrai_x_08b_english.py`** | Automated evaluation suite testing conversational fluency, English reasoning, and `<think>` blocks. | ✅ Verified & Deployed |
| **`poc_wrai_x_08b.py`** | Mathematical proof-of-concept for Haar DWT + Dual Retention integration. | ✅ Verified & Deployed |
| **`inspect_wrai_x_drive_model.py`** | Weight inspection and forensic tensor auditor. | ✅ Verified & Deployed |

## 🗄️ Historical Archive
The `archive/` folder contains earlier prototype experiments (v15 3B, v16 1.7B, and initial verification tests).
