@echo off
set PATH=%~dp0..\..\tools\w64devkit\bin;%PATH%
cd /d "%~dp0"

echo =================================================================
echo    COMPILING WRAI-X (0.8B) NATIVE C ENGINE
echo =================================================================
echo [*] Compiler: GCC MinGW-w64 (AVX 1.0 SIMD + OpenMP)
echo [*] Optimization: -O3 -mavx -msse4.2 -fopenmp
echo.

gcc -O3 -mavx -msse4.2 -fopenmp -Iinclude -Isrc src\wrai_x_engine.c src\wrai_x_cli.c -o wrai_x.exe -lm
if %ERRORLEVEL% EQU 0 (
    echo.
    echo =================================================================
    echo    [SUCCESS] BUILD COMPLETED SUCCESSFULLY! EXECUTABLE: wrai_x.exe      
    echo =================================================================
) else (
    echo.
    echo [ERROR] Compilation failed!
    exit /b 1
)
