@echo off
chcp 65001 >nul
title WRAI ESP32 Serial Monitor (Monitor Only)
echo ================================================================================
echo   WRAI BANANAMIND-2-MICRO: LIGHTWEIGHT SERIAL MONITOR
echo   Direct Console Stream ^| Minimal RAM Footprint (^<15 MB)
echo ================================================================================
echo.

python -c "import serial.tools.list_ports; ports = list(serial.tools.list_ports.comports()); print('Detected COM Ports:'); [print('  - ', p.device, ':', p.description) for p in ports] if ports else print('  [!] No active serial devices detected. Please plug in your ESP32.')"
echo.

set /p COMPORT="Enter your ESP32 COM port (e.g. COM3) [default COM3]: "
if "%COMPORT%"=="" set COMPORT=COM3

echo.
echo [*] Connecting to %COMPORT% at 115200 baud...
echo [TIP] Press the hardware 'EN' / 'RST' button on your ESP32 to restart inference!
echo.

python serial_monitor.py %COMPORT%
pause
