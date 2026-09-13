# WRAI-Micro (1.25M) on Physical ESP32 Microcontroller

**100% Offline Recurrent Neural Language Model with Pure Native C Engine & Zero KV-Cache**

---

## ⚡ Overview

While recent community experiments claim to run ~29M models on microcontrollers by parking a passive lookup table in 16MB Flash while exhausting 8MB of external PSRAM for KV-Cache, **WRAI-Micro** takes a fundamentally different engineering approach:

* **Authentic Base Model**: 100% frozen knowledge from [`SimpleStories/SimpleStories-V2-1.25M`](https://huggingface.co/SimpleStories/SimpleStories-V2-1.25M) (*arXiv:2504.09184*).
* **Pure Linear Recurrence**: Replaces quadratic Softmax Attention with **Dual-State Linear Retention ($S_m / S_r$)**.
* **Zero KV-Cache**: Sequence state is compressed *in-place* into fixed matrices ($4 \times 32 \times 32$). Memory complexity is **strictly $O(1)$** with respect to sequence length.
* **Commodity Hardware**: Runs natively on standard **$3 ESP32 DevKit 38-Pin** boards (`ESP32-D0WD-V3`) with **4 MB SPI Flash** and **0 MB External PSRAM**!
* **Pure Internal SRAM Execution**: Uses only **~68 KB of internal SRAM**—leaving over 200 KB of heap free for Wi-Fi, sensors, and peripherals.

---

## 📊 Physical Silicon & Memory Comparison

| Specification | Viral 28.9M Project (slvDev) | WRAI-Micro (1.25M) |
| :--- | :--- | :--- |
| **Microcontroller** | ESP32-S3 (High-end ~$10) | **ESP32 Classic DevKit (~$3)** |
| **External PSRAM Required** | **8 MB Wajib** (Exhausted by KV-Cache) | **0 MB (NOL! Runs in internal SRAM)** |
| **SPI Flash Required** | **16 MB Flash** (14.9 MB model file) | **4 MB Standard Flash** (~1.2 MB INT8) |
| **KV-Cache Complexity** | $O(T)$ Linear Growth (Hits memory wall) | **Zero KV-Cache ($O(1)$ Constant State)** |
| **Working Dynamic RAM** | ~320 KB SRAM + ~4 MB PSRAM | **~68 KB Internal SRAM** |
| **Interaction Mode** | Unidirectional Story Completion | **Interactive 2-Way Chat over Serial** |

---

## 🛠️ Step-by-Step Reproduction

### 1. Download Official Base Model
```bash
python produce/esp32/download_base_model.py
```

### 2. Prepare Hybrid Chat/Logic Dataset
```bash
python produce/esp32/prepare_dataset.py
```

### 3. Run Architectural Transplantation
```bash
python produce/esp32/train_transplant_esp32.py
```

### 4. Quantize to INT8 and Export C Headers
```bash
python produce/esp32/export_to_c.py
```

### 5. Flash to ESP32
Open `produce/esp32/firmware/wrai_micro_esp32.ino` in Arduino IDE:
* Board: `ESP32 Dev Module`
* Partition Scheme: `Huge APP (3MB No OTA / 1MB SPIFFS)`
* Upload Speed: `921600`
* COM Port: Select your device port (e.g., `COM3`)
* Click **Upload**, then open **Serial Monitor** at **115200 baud**!
