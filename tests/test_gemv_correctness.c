#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <immintrin.h>

void gemv_scalar(const float* scales, const int8_t* W, const float* x, float* y, int M, int K) {
    for (int i = 0; i < M; i++) {
        const int8_t* w_row = W + (i * K);
        float dot = 0.0f;
        for (int k = 0; k < K; k++) {
            dot += (float)w_row[k] * x[k];
        }
        y[i] = dot * scales[i];
    }
}

void gemv_avx(const float* scales, const int8_t* W, const float* x, float* y, int M, int K) {
    for (int i = 0; i < M; i++) {
        const int8_t* w_row = W + (i * K);
        float scale = scales[i];
        float dot = 0.0f;

        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        int k = 0;
        for (; k <= K - 16; k += 16) {
            __m128i r0 = _mm_loadl_epi64((const __m128i*)(w_row + k));
            __m128 w0_lo = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(r0));
            __m128 w0_hi = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(r0, 4)));
            __m256 w0 = _mm256_set_m128(w0_hi, w0_lo);

            __m128i r1 = _mm_loadl_epi64((const __m128i*)(w_row + k + 8));
            __m128 w1_lo = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(r1));
            __m128 w1_hi = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(r1, 4)));
            __m256 w1 = _mm256_set_m128(w1_hi, w1_lo);

            __m256 x0 = _mm256_loadu_ps(x + k);
            __m256 x1 = _mm256_loadu_ps(x + k + 8);

            acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(w0, x0));
            acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(w1, x1));
        }
        __m256 total = _mm256_add_ps(acc0, acc1);
        __m128 lo = _mm256_castps256_ps128(total);
        __m128 hi = _mm256_extractf128_ps(total, 1);
        __m128 sum128 = _mm_add_ps(lo, hi);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        dot = _mm_cvtss_f32(sum128);

        for (; k < K; k++) {
            dot += (float)w_row[k] * x[k];
        }
        y[i] = dot * scale;
    }
}

int main() {
    int M = 64;
    int K = 2048;
    float* scales = (float*)malloc(M * sizeof(float));
    int8_t* W = (int8_t*)malloc(M * K * sizeof(int8_t));
    float* x = (float*)malloc(K * sizeof(float));
    float* y_ref = (float*)malloc(M * sizeof(float));
    float* y_avx = (float*)malloc(M * sizeof(float));

    for (int i = 0; i < M; i++) scales[i] = 0.001f * (i + 1);
    for (int i = 0; i < M * K; i++) W[i] = (int8_t)((i % 255) - 128);
    for (int i = 0; i < K; i++) x[i] = 0.01f * ((i % 100) - 50);

    gemv_scalar(scales, W, x, y_ref, M, K);
    gemv_avx(scales, W, x, y_avx, M, K);

    float max_diff = 0.0f;
    for (int i = 0; i < M; i++) {
        float diff = fabsf(y_ref[i] - y_avx[i]);
        if (diff > max_diff) max_diff = diff;
    }
    printf("[GEMV TEST] Max diff between Scalar and AVX: %e\n", max_diff);
    if (max_diff < 1e-3) {
        printf("[PASS] GEMV is numerically identical!\n");
    } else {
        printf("[FAIL] GEMV divergence detected!\n");
    }

    free(scales); free(W); free(x); free(y_ref); free(y_avx);
    return 0;
}
