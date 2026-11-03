# BananaMind-2-Micro (2.93M) — WRAI Bare-Metal ESP32 Release Candidate
### Complete On-Device $O(1)$ Constant-Memory Language Model Deployment for ESP32

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23920132.svg)](https://doi.org/10.5281/zenodo.23920132)
[![Hardware](https://img.shields.io/badge/Hardware-ESP32%20%7C%20ESP32--S3-red.svg)](#hardware-specification)
[![Memory](https://img.shields.io/badge/Flash-4MB%20SPI%20Flash-blue.svg)](#flash-budget--partition-table)
[![Dynamic KV Cache](https://img.shields.io/badge/Dynamic%20KV--Cache-0.00%20Bytes-success.svg)](#audit-results--verification-gates)
[![Parameters](https://img.shields.io/badge/Params-2.93M%20(Tied)-purple.svg)](#model-provenance)

---

## 1. Overview & Scientific Significance

This repository contains the frozen release candidate for deploying **BananaMind-2-Micro (2.93M)** on a bare-metal **ESP32 microcontroller (4MB SPI Flash, 512KB SRAM)** using the **WRAI (Wavelet / Recurrent / Attentive Intelligence)** architecture.

### The Problem with Standard Transformers on Microcontrollers:
Standard causal transformers store every Key-Value token in an expanding historical cache. On BananaMind-2-Micro, the standard KV-cache consumes $4,608 \times N$ bytes. By token 128, the cache alone consumes **576 KB of RAM**, instantly crashing any standard ESP32 (which only has ~380 KB usable heap) with an Out-Of-Memory (OOM) kernel panic.

### The WRAI Breakthrough:
Through surgical post-training transmutation, all 9 attention layers are replaced with bounded Tri-Memory states:
* **0.00 Bytes Dynamic KV-Cache**: Zero heap memory allocated during autoregressive token generation.
* **Bounded Internal SRAM (~50 KB for INT8 / ~146 KB working total)**: Fits completely in the internal SRAM of ESP32 without requiring external PSRAM.
* **Direct Flash Memory-Mapping (`esp_partition_mmap`)**: Model weights are executed directly from SPI Flash via the hardware MMU cache without being loaded into RAM.

---

## 2. Package Directory Structure

```text
BananaMind-2-Micro/
├── model/
│   ├── bananamind_2m_int8_mmap.bin    # 2.87 MiB INT8 mmap weights (High semantic fidelity)
│   ├── tokenizer_vocab.h              # Embedded C string array for 2,048 tokens
│   └── manifest.json                  # Model provenance, tensor descriptors, & SHA256 hashes
├── firmware/
│   ├── CMakeLists.txt                 # ESP-IDF Root Build script
│   ├── sdkconfig.defaults             # 240 MHz CPU, -O3 optimization, Radios disabled
│   ├── partition_table.csv            # 4MB SPI Flash partition table layout
│   ├── bin/                           # Precompiled flashable binaries
│   │   ├── bootloader.bin             # ESP32 2nd stage bootloader (0x1000)
│   │   ├── partitions.bin             # Custom partition table (0x8000)
│   │   ├── boot_app0.bin              # App rollback descriptor (0xE000)
│   │   └── bananamind_app.bin         # Compiled C++ WRAI inference engine (0x10000)
│   ├── arduino/                       # Arduino IDE compatible project
│   │   └── BananaMind_ESP32/
│   └── main/
│       ├── CMakeLists.txt             # Component build script
│       ├── main.c                     # Bare-metal on-chip autonomous inference program
│       ├── wrai_bananamind_engine.h   # C99 Engine Header (Zero malloc API)
│       ├── wrai_bananamind_engine.c   # Pure C99 forward pass (RMSNorm, RoPE, GEMV, WRAI)
│       ├── ssd1306_oled.h             # Fast SSD1306 OLED driver
│       ├── ssd1306_oled.c             # Fail-safe I2C OLED display implementation
│       └── tokenizer_vocab.h          # Tokenizer table for streaming UART output
├── audit/
│   ├── hardware_info.txt              # Hardware specifications and power footprint
│   ├── flash_budget.csv               # 4MB SPI Flash partition allocation
│   ├── memory_audit.csv               # Detailed SRAM working memory breakdown
│   ├── zero_kv_audit.csv              # Empirical proof of T1-T7 flatline vs Transformer OOM
│   ├── quality_results.csv            # 7-Category functional correctness results
│   ├── benchmark_results.csv          # Multi-prompt empirical benchmark results
│   └── stability.log                  # Multi-prompt verification on physical ESP32
├── flash_and_monitor.bat              # 1-Click complete flash and live monitor tool
├── monitor_only.bat                   # 1-Click lightweight monitor tool (no re-flash)
├── serial_monitor.py                  # Clean Python serial console script
└── README.md                          # Documentation & Flashing Guide
```

---

## 3. Flash Budget & Partition Table (4MB Flash)

The model partition is allocated at offset `0x90000`:

| Partition Name | Type | Subtype | Offset | Size | Purpose |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `nvs` | data | nvs | `0x9000` | 16 KB | System NVS config |
| `otadata` | data | ota | `0xd000` | 8 KB | OTA selection data |
| `phy_init` | data | phy | `0xf000` | 4 KB | PHY RF calibration |
| `factory` | app | factory | `0x10000` | 512 KB | Compiled C++ Firmware App (`bananamind_app.bin`) |
| `model` | data | `0x40` | `0x90000` | 3,392 KB (3.31 MiB) | **`bananamind_2m_int8_mmap.bin` (2.87 MiB)** |

---

## 4. Complete Flashing Guide for ESP32 (Step-by-Step)

There are two primary methods to deploy and flash this project onto your ESP32 board:

### Method 1: Ultra-Lightweight 1-Click Scripts (No Heavy IDE Required!)

Pre-compiled production binaries are provided in `firmware/bin/` so you can flash and run without compiling anything:

1. **Flashing & Running (Complete Package from 0)**:
   - Double-click **`flash_and_monitor.bat`**, or run:
     ```bash
     python -m esptool -p COM3 -b 921600 write-flash 0x1000 "firmware/bin/bootloader.bin" 0x8000 "firmware/bin/partitions.bin" 0xe000 "firmware/bin/boot_app0.bin" 0x10000 "firmware/bin/bananamind_app.bin" 0x90000 "model/bananamind_2m_int8_mmap.bin"
     ```
2. **Subsequent Monitoring (Monitor Only — No Flashing Needed)**:
   - Double-click **`monitor_only.bat`** (or `python serial_monitor.py COM3`) to connect directly to the serial console anytime without re-flashing!
3. **Manual Boot Trigger (Important Hardware Note)**:
   > **Note for boards requiring manual bootloader trigger:**  
   > Many common ESP32 development boards (e.g. NodeMCU, ESP32-WROOM clone modules) lack auto-reset capacitors. If you see `Connecting........_____....._____.....`:  
   > 1. **Press and hold the `BOOT` (or `IO0`) button** as soon as `Connecting...` appears.  
   > 2. Release it once writing begins (`Writing at 0x00001000...`).  
   > 3. After flashing completes, press the **`EN` / `RST`** button once to restart the chip and execute autonomous inference!

---

### Method 2: Full Source Compilation via ESP-IDF (Build & Flash from C Source)

If you have **ESP-IDF v5.x / v6.x** installed:

1. **Open ESP-IDF Command Prompt**:
   Connect your ESP32 board via USB and identify the COM port (e.g., `COM3`).

2. **Navigate to the firmware directory**:
   ```bash
   cd "BananaMind-2-Micro/firmware"
   ```

3. **Set Target Chip Architecture**:
   ```bash
   # For ESP32-S3:
   idf.py set-target esp32s3

   # For standard ESP32:
   idf.py set-target esp32
   ```

4. **Compile Firmware**:
   ```bash
   idf.py build
   ```

5. **Flash Firmware & Model Partition**:
   ```bash
   # Flash bootloader, partition table, and bare-metal application:
   idf.py -p COM3 flash

   # Flash INT8 quantized weights to the dedicated model partition (offset 0x90000):
   esptool.py -p COM3 -b 921600 write_flash 0x90000 "../model/bananamind_2m_int8_mmap.bin"
   ```

6. **Monitor UART Output**:
   ```bash
   idf.py -p COM3 monitor
   ```

---

## 5. Optional Hardware: 0.96" I2C OLED Display (SSD1306)

The firmware includes built-in, fail-safe support for an optional **0.96-inch monochrome OLED display (SSD1306, 128x64 resolution)** over I2C.

### Pinout Wiring Diagram:

| OLED Pin | ESP32 Pin (Standard WROOM / DevKit) | ESP32-S3 Pin | Description |
| :---: | :---: | :---: | :--- |
| **VCC** | **3.3V** | **3.3V** | Power supply (3.3V logic) |
| **GND** | **GND** | **GND** | System Ground |
| **SCL** | **GPIO 22** | **GPIO 9** | I2C Clock (400 kHz Fast Mode) |
| **SDA** | **GPIO 21** | **GPIO 8** | I2C Data |

### Fail-Safe & Robust Headless Fallback:
* **Active Non-Blocking Probe (<50ms)**: During boot, the firmware checks for an I2C ACK on addresses `0x3C` and `0x3D`.
* **Zero-Overhead Headless Fallback**: If the OLED is not attached, unpowered, or disconnected, the driver cleanly tears down the I2C peripheral and disables itself. **Inference continues at full speed with 0 CPU latency and 0 bytes wasted**.
* **Guaranteed Serial Streaming**: The UART / Serial Monitor (`115200 baud`) remains the primary, unhampered communication channel under all conditions. If an OLED is plugged in, tokens stream simultaneously to both the physical screen (with live line wrapping and automatic scrolling) and the terminal console.

---

## 6. Audit Results & Verification Gates (Summary)

| Audit Stage | Gate ID | Status | Key Verification Result |
| :--- | :--- | :---: | :--- |
| **Stage 0** | Gate 0 · Model Provenance | **PASS** | 2,933,301 Unique Tied Parameters verified, SHA256 frozen |
| **Stage 1** | Gate 1 · Static Memory Budget | **PASS** | INT8 (2.87 MiB) fits 4MB SPI Flash with 440 KB slack |
| **Stage 2** | Gate 2 · On-Device Execution | **PASS** | 100% Standalone C execution via `esp_partition_mmap` |
| **Stage 3** | Gate 3 · Zero-KV Cache Proof | **PASS** | **0.00 Bytes Dynamic KV** ($O(1)$ Flatline across prompts) |
| **Stage 4** | Gate 4 · Functional Correctness| **PASS** | High Semantic Fidelity INT8 vs Native PyTorch baseline |
| **Stage 5** | Gate 5 · Hardware Benchmark | **PASS** | **3.78 TPS**, 0 Byte Dynamic KV, 100% stable heap on ESP32 |
| **Stage 6** | Gate 6 · Release Candidate | **FROZEN** | Production-ready firmware package prepared for GitHub |

---

## 7. Citation & License

```bibtex
@misc{musounoenma_2026_wrai,
  author       = {MusounoEnma},
  title        = {{WRAI: From Selective Routing to Fixed-State Memory for Autoregressive Language Models (Microcontroller Proof-of-Execution on ESP32)}},
  month        = nov,
  year         = 2026,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.23920132},
  url          = {https://doi.org/10.5281/zenodo.23920132}
}
```

Released under the open-source [Apache 2.0 License](https://opensource.org/licenses/Apache-2.0).
