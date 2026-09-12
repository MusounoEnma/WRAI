@echo off
title WRAI-X (0.6B) Native C Inference Studio
cd /d "%~dp0"

echo =================================================================
echo    LAUNCHING WRAI-X (0.6B) PURE NATIVE C INFERENCE ENGINE
echo =================================================================
echo.

engine\wrai_x.exe "qwen\wrai_x_06b_int8.bin" "qwen\wrai_x_vocab.bin"

echo.
echo =================================================================
echo Program WRAI-X selesai.
echo =================================================================
pause
