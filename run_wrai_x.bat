@echo off
title WRAI-X (0.6B) Native C Inference Studio
cd /d "%~dp0"

set BINARY=wrai_x.exe
if exist wrai_x_v3.exe (
    set BINARY=wrai_x_v3.exe
) else if exist wrai_x_v2.exe (
    set BINARY=wrai_x_v2.exe
)

echo =================================================================
echo    LAUNCHING WRAI-X (0.6B) PURE NATIVE C INFERENCE ENGINE
echo =================================================================
echo.

%BINARY% "models x\wrai_x_06b_int8.bin" "models x\wrai_x_vocab.bin"

echo.
echo =================================================================
echo Program WRAI-X selesai.
echo =================================================================
pause
