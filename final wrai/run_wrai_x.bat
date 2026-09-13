@echo off
title WRAI-X (0.8B) Native C Inference Studio
cd /d "%~dp0"

echo =================================================================
echo    LAUNCHING WRAI-X (0.8B) PURE NATIVE C INFERENCE ENGINE
echo =================================================================
echo.

engine\wrai_x.exe "qwen\wrai_x_08b_int8.bin" "qwen\wrai_x_vocab.bin"

echo.
echo =================================================================
echo WRAI-X execution completed.
echo =================================================================
pause
