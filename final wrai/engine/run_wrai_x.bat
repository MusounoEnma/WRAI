@echo off
title WRAI-X (0.6B) Native C Inference Studio
cd /d "%~dp0"

echo =================================================================
echo    LAUNCHING WRAI-X (0.6B) PURE NATIVE C INFERENCE ENGINE
echo =================================================================
echo.

if exist wrai_x.exe (
    wrai_x.exe "..\qwen\wrai_x_06b_int8.bin" "..\qwen\wrai_x_vocab.bin"
) else (
    echo [ERROR] wrai_x.exe tidak ditemukan! Silakan jalankan build_wrai_x.bat terlebih dahulu.
)

echo.
echo =================================================================
echo Program WRAI-X selesai.
echo =================================================================
pause
