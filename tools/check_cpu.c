#include <stdio.h>
#include <stdbool.h>

int main() {
    __builtin_cpu_init();
    printf("======================================\n");
    printf("       HOST CPU SIMD CAPABILITY       \n");
    printf("======================================\n");
    printf("MMX:      %s\n", __builtin_cpu_supports("mmx") ? "YES" : "NO");
    printf("SSE:      %s\n", __builtin_cpu_supports("sse") ? "YES" : "NO");
    printf("SSE2:     %s\n", __builtin_cpu_supports("sse2") ? "YES" : "NO");
    printf("SSE3:     %s\n", __builtin_cpu_supports("sse3") ? "YES" : "NO");
    printf("SSSE3:    %s\n", __builtin_cpu_supports("ssse3") ? "YES" : "NO");
    printf("SSE4.1:   %s\n", __builtin_cpu_supports("sse4.1") ? "YES" : "NO");
    printf("SSE4.2:   %s\n", __builtin_cpu_supports("sse4.2") ? "YES" : "NO");
    printf("AVX:      %s\n", __builtin_cpu_supports("avx") ? "YES" : "NO");
    printf("AVX2:     %s\n", __builtin_cpu_supports("avx2") ? "YES" : "NO");
    printf("AVX512F:  %s\n", __builtin_cpu_supports("avx512f") ? "YES" : "NO");
    printf("FMA:      %s\n", __builtin_cpu_supports("fma") ? "YES" : "NO");
    printf("POPCNT:   %s\n", __builtin_cpu_supports("popcnt") ? "YES" : "NO");
    printf("======================================\n");
    return 0;
}
