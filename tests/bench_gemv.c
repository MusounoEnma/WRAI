#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <immintrin.h>
#ifdef _OPENMP
#include <omp.h>
#endif

void gemv_unroll32(const float* scales, const int8_t* W, const float* x, float* y, int M, int K) {
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < M; i++) {
        const int8_t* w_row = W + ((size_t)i * K);
        float scale = scales[i];

        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();

        int k = 0;
        for (; k <= K - 32; k += 32) {
            __m128i r0 = _mm_loadl_epi64((const __m128i*)(w_row + k));
            __m128 w0_lo = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(r0));
            __m128 w0_hi = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(r0, 4)));
            __m256 w0 = _mm256_set_m128(w0_hi, w0_lo);

            __m128i r1 = _mm_loadl_epi64((const __m128i*)(w_row + k + 8));
            __m128 w1_lo = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(r1));
            __m128 w1_hi = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(r1, 4)));
            __m256 w1 = _mm256_set_m128(w1_hi, w1_lo);

            __m128i r2 = _mm_loadl_epi64((const __m128i*)(w_row + k + 16));
            __m128 w2_lo = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(r2));
            __m128 w2_hi = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(r2, 4)));
            __m256 w2 = _mm256_set_m128(w2_hi, w2_lo);

            __m128i r3 = _mm_loadl_epi64((const __m128i*)(w_row + k + 24));
            __m128 w3_lo = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(r3));
            __m128 w3_hi = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(r3, 4)));
            __m256 w3 = _mm256_set_m128(w3_hi, w3_lo);

            __m256 x0 = _mm256_loadu_ps(x + k);
            __m256 x1 = _mm256_loadu_ps(x + k + 8);
            __m256 x2 = _mm256_loadu_ps(x + k + 16);
            __m256 x3 = _mm256_loadu_ps(x + k + 24);

            acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(w0, x0));
            acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(w1, x1));
            acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(w2, x2));
            acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(w3, x3));
        }

        __m256 total = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
        __m128 lo = _mm256_castps256_ps128(total);
        __m128 hi = _mm256_extractf128_ps(total, 1);
        __m128 sum128 = _mm_add_ps(lo, hi);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        float dot = _mm_cvtss_f32(sum128);

        for (; k < K; k++) {
            dot += (float)w_row[k] * x[k];
        }
        y[i] = dot * scale;
    }
}

int main() {
    int M = 2048;
    int K = 2048;
    printf("[*] Benchmarking GEMV 2048x2048 (1 layer retention matrix)...\n");
    float* scales = (float*)calloc(M, sizeof(float));
    int8_t* W = (int8_t*)calloc((size_t)M * K, sizeof(int8_t));
    float* x = (float*)calloc(K, sizeof(float));
    float* y = (float*)calloc(M, sizeof(float));

    clock_t t0 = clock();
    int iters = 20;
    for (int i = 0; i < iters; i++) {
        gemv_unroll32(scales, W, x, y, M, K);
    }
    clock_t t1 = clock();
    double sec = (double)(t1 - t0) / CLOCKS_PER_SEC / iters;
    printf("[OK] GEMV 2048x2048 took: %.3f ms per call\n", sec * 1000.0);

    // Benchmarking 6144x2048 (FFN matrix)
    int M_ffn = 6144;
    float* scales_ffn = (float*)calloc(M_ffn, sizeof(float));
    int8_t* W_ffn = (int8_t*)calloc((size_t)M_ffn * K, sizeof(int8_t));
    float* y_ffn = (float*)calloc(M_ffn, sizeof(float));

    t0 = clock();
    for (int i = 0; i < iters; i++) {
        gemv_unroll32(scales_ffn, W_ffn, x, y_ffn, M_ffn, K);
    }
    t1 = clock();
    sec = (double)(t1 - t0) / CLOCKS_PER_SEC / iters;
    printf("[OK] GEMV 6144x2048 took: %.3f ms per call\n", sec * 1000.0);

    return 0;
}
