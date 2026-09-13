@echo off
title WRAI-X (0.8B) Native C Inference Studio
cd /d "%~dp0"

echo =================================================================
echo    LAUNCHING WRAI-X (0.8B) PURE NATIVE C INFERENCE ENGINE
echo =================================================================
echo.

if exist wrai_x.exe (
    wrai_x.exe "..\qwen\wrai_x_08b_int8.bin" "..\qwen\wrai_x_vocab.bin"
) else (
    echo [ERROR] wrai_x.exe not found! Please run build_wrai_x.bat first.
)

echo.
echo =================================================================
echo WRAI-X execution completed.
echo =================================================================
pause
