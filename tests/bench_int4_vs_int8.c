#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <immintrin.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* INT8 GEMV (Current Engine) */
void gemv_int8(const float* scales, const int8_t* W, const float* x, float* y, int M, int K) {
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
        y[i] = _mm_cvtss_f32(sum128) * scale;
    }
}

/* INT4 GEMV (Packed 2 nibbles per byte, sign-extended [-8, 7]) */
void gemv_int4(const float* scales, const uint8_t* W_packed, const float* x, float* y, int M, int K) {
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < M; i++) {
        const uint8_t* w_row = W_packed + ((size_t)i * (K / 2));
        float scale = scales[i];

        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();

        __m128i mask_lo = _mm_set1_epi8(0x0F);
        __m128i offset8 = _mm_set1_epi8(0x08);

        int k = 0;
        for (; k <= K - 32; k += 32) {
            /* Load 16 bytes = 32 nibbles */
            __m128i packed = _mm_loadu_si128((const __m128i*)(w_row + (k / 2)));

            /* Unpack low nibbles: (val & 0x0F) - 8 */
            __m128i lo_bytes = _mm_sub_epi8(_mm_and_si128(packed, mask_lo), offset8);
            /* Unpack high nibbles: (val >> 4) - 8 */
            __m128i hi_bytes = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(packed, 4), mask_lo), offset8);

            /* Interleave low and high nibbles to restore original sequence */
            __m128i seq0 = _mm_unpacklo_epi8(lo_bytes, hi_bytes); /* 16 signed bytes */
            __m128i seq1 = _mm_unpackhi_epi8(lo_bytes, hi_bytes); /* 16 signed bytes */

            /* Convert to float for first 16 elements */
            __m128 f0_lo = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(seq0));
            __m128 f0_hi = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(seq0, 4)));
            __m256 w0 = _mm256_set_m128(f0_hi, f0_lo);

            __m128i seq0_hi8 = _mm_srli_si128(seq0, 8);
            __m128 f1_lo = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(seq0_hi8));
            __m128 f1_hi = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(seq0_hi8, 4)));
            __m256 w1 = _mm256_set_m128(f1_hi, f1_lo);

            __m256 x0 = _mm256_loadu_ps(x + k);
            __m256 x1 = _mm256_loadu_ps(x + k + 8);
            acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(w0, x0));
            acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(w1, x1));

            /* Convert to float for next 16 elements */
            __m128 f2_lo = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(seq1));
            __m128 f2_hi = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(seq1, 4)));
            __m256 w2 = _mm256_set_m128(f2_hi, f2_lo);

            __m128i seq1_hi8 = _mm_srli_si128(seq1, 8);
            __m128 f3_lo = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(seq1_hi8));
            __m128 f3_hi = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(seq1_hi8, 4)));
            __m256 w3 = _mm256_set_m128(f3_hi, f3_lo);

            __m256 x2 = _mm256_loadu_ps(x + k + 16);
            __m256 x3 = _mm256_loadu_ps(x + k + 24);
            acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(w2, x2));
            acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(w3, x3));
        }

        __m256 total = _mm256_add_ps(acc0, acc1);
        __m128 lo = _mm256_castps256_ps128(total);
        __m128 hi = _mm256_extractf128_ps(total, 1);
        __m128 sum128 = _mm_add_ps(lo, hi);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        y[i] = _mm_cvtss_f32(sum128) * scale;
    }
}

int main() {
    int M = 2048;
    int K = 2048;
    int iters = 50;

    printf("=================================================================\n");
    printf("   MICROBENCHMARK: INT8 vs INT4 SIMD (AVX 1.0 + SSE4.2)         \n");
    printf("=================================================================\n");
    printf("[*] Matrix Size: %d x %d\n", M, K);
    printf("[*] Iterations : %d\n\n", iters);

    float* scales = (float*)calloc(M, sizeof(float));
    float* x = (float*)calloc(K, sizeof(float));
    float* y = (float*)calloc(M, sizeof(float));

    /* Allocate INT8 (4 MB) */
    int8_t* W_int8 = (int8_t*)calloc((size_t)M * K, sizeof(int8_t));

    /* Allocate INT4 (2 MB) */
    uint8_t* W_int4 = (uint8_t*)calloc((size_t)M * (K / 2), sizeof(uint8_t));

    /* Warmup */
    gemv_int8(scales, W_int8, x, y, M, K);
    gemv_int4(scales, W_int4, x, y, M, K);

    /* Benchmark INT8 */
    clock_t t0 = clock();
    for (int i = 0; i < iters; i++) {
        gemv_int8(scales, W_int8, x, y, M, K);
    }
    clock_t t1 = clock();
    double ms_int8 = (double)(t1 - t0) / CLOCKS_PER_SEC / iters * 1000.0;
    printf("[OK] INT8 GEMV (4 MB memory per matrix): %.3f ms\n", ms_int8);

    /* Benchmark INT4 */
    t0 = clock();
    for (int i = 0; i < iters; i++) {
        gemv_int4(scales, W_int4, x, y, M, K);
    }
    t1 = clock();
    double ms_int4 = (double)(t1 - t0) / CLOCKS_PER_SEC / iters * 1000.0;
    printf("[OK] INT4 GEMV (2 MB memory per matrix): %.3f ms\n\n", ms_int4);

    printf("Perbandingan:\n");
    if (ms_int4 < ms_int8) {
        printf("  --> INT4 LEBIH CEPAT %.2fx!\n", ms_int8 / ms_int4);
    } else {
        printf("  --> INT8 LEBIH CEPAT %.2fx (karena overhead unpacking nibble di CPU ALU)!\n", ms_int4 / ms_int8);
    }
    printf("=================================================================\n");

    free(scales);
    free(x);
    free(y);
    free(W_int8);
    free(W_int4);
    return 0;
}
