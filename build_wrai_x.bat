@echo off
set PATH=%~dp0tools\w64devkit\bin;%PATH%

echo =================================================================
echo    COMPILING WRAI-X (0.8B) NATIVE C ENGINE FOR AMD A8 PUMA+    
echo =================================================================
echo [*] Compiler: GCC MinGW-w64 (AVX 1.0 SIMD + OpenMP)
echo [*] Optimization: -O3 -mavx -msse4.2 -fopenmp
echo.

gcc -O3 -mavx -msse4.2 -fopenmp -Iinclude -Isrc src\wrai_x_engine.c src\wrai_x_cli.c -o wrai_x.exe -lm
if %ERRORLEVEL% EQU 0 (
    copy /y wrai_x.exe wrai_x_v3.exe >nul
    copy /y wrai_x.exe "final wrai\engine\wrai_x.exe" >nul
)

if %ERRORLEVEL% EQU 0 (
    echo.
    echo =================================================================
    echo    [SUCCESS] BUILD COMPLETED SUCCESSFULLY! EXECUTABLE: wrai_x.exe      
    echo =================================================================
    echo.
    echo Run via command line:
    echo wrai_x.exe
    echo.
) else (
    echo.
    echo [ERROR] Compilation failed!
    exit /b 1
)
