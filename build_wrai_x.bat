@echo off
set PATH=%~dp0tools\w64devkit\bin;%PATH%

echo =================================================================
echo    COMPILING WRAI-X (0.6B) NATIVE C ENGINE FOR AMD A8 PUMA+    
echo =================================================================
echo [*] Compiler: GCC MinGW-w64 (AVX 1.0 SIMD + OpenMP)
echo [*] Optimization: -O3 -mavx -msse4.2 -fopenmp
echo.

gcc -O3 -mavx -msse4.2 -fopenmp -Iinclude src\wrai_x_engine.c src\wrai_x_cli.c -o wrai_x.exe -lm

if %ERRORLEVEL% EQU 0 (
    echo.
    echo =================================================================
    echo    [SUCCESS] BUILD BERHASIL 100%%! EXECUTABLE: wrai_x.exe      
    echo =================================================================
    echo.
    echo Jalankan dengan mengetik:
    echo wrai_x.exe
    echo.
) else (
    echo.
    echo [ERROR] Kompilasi gagal!
    exit /b 1
)
