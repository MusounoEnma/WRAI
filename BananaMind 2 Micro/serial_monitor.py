#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Super-Lightweight ESP32 Serial Monitor for Budget Laptops.
Consumes < 15 MB RAM, 0% CPU. No heavy IDE required!
"""
import sys
import time
import serial
import serial.tools.list_ports

def auto_detect_port():
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        return None
    for p in ports:
        desc = p.description.lower()
        if any(chip in desc for chip in ["cp210", "ch340", "ch341", "ftdi", "uart", "usb-serial", "esp32"]):
            return p.device
    return ports[0].device

def main():
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print("Usage: python serial_monitor.py [PORT] [BAUD]")
        print("  PORT : Serial port (e.g., COM3, /dev/ttyUSB0). If omitted, auto-detected.")
        print("  BAUD : Baud rate (default: 115200)")
        return

    port = sys.argv[1] if len(sys.argv) > 1 else auto_detect_port()
    baud = int(sys.argv[2]) if len(sys.argv) > 2 else 115200
    
    if not port:
        print("[!] No active COM ports detected! Please connect your ESP32 board and retry.")
        input("Press Enter to exit...")
        return

    print("=" * 65)
    print(f" WRAI ESP32 LIGHTWEIGHT SERIAL MONITOR")
    print(f" Port: {port} | Baud: {baud}")
    print(" Press Ctrl+C to exit | Press EN/RST button on ESP32 to restart")
    print("=" * 65 + "\n")

    try:
        ser = serial.Serial(port, baud, timeout=0.1)
    except Exception as e:
        print(f"[ERROR] Could not open {port}: {e}")
        return

    import codecs
    decoder = codecs.getincrementaldecoder("utf-8")("replace")

    try:
        while True:
            raw = ser.read(ser.in_waiting or 1)
            if raw:
                text = decoder.decode(raw)
                if text:
                    sys.stdout.write(text)
                    sys.stdout.flush()
    except KeyboardInterrupt:
        print("\n\n[INFO] Serial monitor closed.")
    finally:
        ser.close()

if __name__ == "__main__":
    main()
