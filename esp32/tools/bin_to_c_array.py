"""
Convert wrai_micro_weights.bin into a PROGMEM C array header
so Arduino IDE can flash the model directly without extra SPIFFS tools!
"""

import os
import sys

BASE_DIR = os.path.dirname(__file__)
BIN_PATH = os.path.join(BASE_DIR, "wrai_micro_weights.bin")
OUT_H = os.path.join(BASE_DIR, "firmware", "wrai_micro_weights.h")

print("[*] Generating C PROGMEM weights array from binary...", flush=True)
with open(BIN_PATH, "rb") as f:
    data = f.read()

with open(OUT_H, "w", encoding="utf-8") as f:
    f.write("// Auto-generated PROGMEM INT8 Model Weights for ESP32\n")
    f.write("#ifndef WRAI_MICRO_WEIGHTS_H\n#define WRAI_MICRO_WEIGHTS_H\n\n")
    f.write("#include <pgmspace.h>\n\n")
    f.write(f"#define WRAI_WEIGHTS_SIZE {len(data)}\n\n")
    f.write("const uint8_t WRAI_MODEL_WEIGHTS[] PROGMEM = {\n")
    
    # Write bytes in chunks of 16
    chunk_size = 16
    for i in range(0, len(data), chunk_size):
        chunk = data[i : i + chunk_size]
        hex_str = ", ".join(f"0x{b:02x}" for b in chunk)
        if i + chunk_size < len(data):
            f.write(f"  {hex_str},\n")
        else:
            f.write(f"  {hex_str}\n")
            
    f.write("};\n\n#endif // WRAI_MICRO_WEIGHTS_H\n")

print(f"[SUCCESS] Saved {len(data):,} bytes to {OUT_H}!")
