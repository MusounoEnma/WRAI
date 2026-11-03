@echo off
chcp 65001 >nul
title WRAI ESP32 Flasher - Ultra-Lightweight Tool
echo ================================================================================
echo   WRAI BANANAMIND-2-MICRO: ULTRA-LIGHTWEIGHT ESP32 FLASH AND MONITOR TOOL
echo   Complete Image Flash (Bootloader + Partitions + App + INT8 Model Weights)
echo   Zero-Compiler ^| Zero-Java ^| Zero-Lag ^| Minimal RAM Footprint (^<15 MB)
echo ================================================================================
echo.

python -c "import serial.tools.list_ports; ports = list(serial.tools.list_ports.comports()); print('Detected COM Ports:'); [print('  - ', p.device, ':', p.description) for p in ports] if ports else print('  [!] No active serial devices detected. Please plug in your ESP32.')"
echo.

set /p COMPORT="Enter your ESP32 COM port (e.g. COM3) [default COM3]: "
if "%COMPORT%"=="" set COMPORT=COM3

echo.
echo ================================================================================
echo  [IMPORTANT HARDWARE TIP - MANUAL BOOTLOADER TRIGGER]
echo  If your ESP32 board outputs 'Connecting........_____....._____.....':
echo  --^> PRESS AND HOLD the 'BOOT' (or IO0) button on your board NOW!
echo  --^> Release the button once you see 'Writing at 0x00001000...'
echo ================================================================================
echo.
echo [*] Flashing Complete Firmware Image and INT8 Model Weights to %COMPORT%...
echo     - Bootloader:      0x1000   (24.4 KB)
echo     - Partition Table: 0x8000   (3.0 KB)
echo     - Boot App0:       0xE000   (8.0 KB)
echo     - C++ Firmware:    0x10000  (305.5 KB)
echo     - INT8 Weights:    0x90000  (2.87 MiB)
echo.

python -m esptool -p %COMPORT% -b 921600 write-flash 0x1000 "firmware\bin\bootloader.bin" 0x8000 "firmware\bin\partitions.bin" 0xe000 "firmware\bin\boot_app0.bin" 0x10000 "firmware\bin\bananamind_app.bin" 0x90000 "model\bananamind_2m_int8_mmap.bin"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Flashing failed! 
    echo Ensure the correct COM port is specified and hold the BOOT button when connecting.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ================================================================================
echo [SUCCESS] Complete Firmware and Model Flashed Successfully!
echo [TIP] Press the hardware 'EN' / 'RST' button on your ESP32 to restart and run!
echo [NOTE] In the future, you can just run 'monitor_only.bat' to view inference.
echo ================================================================================
echo.

python serial_monitor.py %COMPORT%
pause
