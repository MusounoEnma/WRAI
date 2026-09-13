@echo off
set PATH=%~dp0tools\w64devkit\bin;%PATH%

echo =================================================================
echo    COMPILING WRAI v16 (1.7B) NATIVE C ENGINE FOR AMD A8 PUMA+    
echo =================================================================
echo [*] Compiler: GCC MinGW-w64 (AVX 1.0 SIMD + OpenMP)
echo [*] Optimization: -O3 -mavx -msse4.2 -fopenmp
echo.

gcc -O3 -mavx -msse4.2 -fopenmp -Iinclude src\wrai_v16_engine.c src\wrai_v16_cli.c -o wrai_v16.exe -lm

if %ERRORLEVEL% EQU 0 (
    echo.
    echo =================================================================
    echo    [SUCCESS] BUILD SUCCEEDED 100%%! EXECUTABLE: wrai_v16.exe      
    echo =================================================================
    echo.
    echo Run the engine by typing:
    echo wrai_v16.exe
    echo.
) else (
    echo.
    echo [ERROR] Compilation failed!
    exit /b 1
)
