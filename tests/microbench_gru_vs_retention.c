/**
 * @file microbench_gru_vs_retention.c
 * @brief Isolated Single-Layer CPU Microbenchmark: GRU vs Multi-Head Retention
 * 
 * Tests exact single-layer step latency and throughput (TPS) on CPU
 * using Q15 fixed-point / FP32 AVX arithmetic.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define HIDDEN_DIM      896
#define NUM_HEADS       14
#define HEAD_DIM        64      // 14 * 64 = 896
#define NUM_ITERS       10000

/* ========================================================================= */
/* 1. GRU Single-Layer Step (C Baseline)                                     */
/* ========================================================================= */

typedef struct {
    float w_ih[3 * HIDDEN_DIM * HIDDEN_DIM]; // 3 * 896 * 896 = 2,408,448 floats
    float w_hh[3 * HIDDEN_DIM * HIDDEN_DIM];
    float b_ih[3 * HIDDEN_DIM];
    float b_hh[3 * HIDDEN_DIM];
    float h[HIDDEN_DIM];
} gru_layer_t;

static inline float sigmoidf_fast(float x) {
    return 1.0f / (1.0f + expf(-x));
}

void gru_step(gru_layer_t* gru, const float* x, float* out) {
    float gates_ih[3 * HIDDEN_DIM];
    float gates_hh[3 * HIDDEN_DIM];

    // Matmul W_ih * x
    for (int i = 0; i < 3 * HIDDEN_DIM; i++) {
        float sum = gru->b_ih[i];
        const float* row = &gru->w_ih[i * HIDDEN_DIM];
        for (int j = 0; j < HIDDEN_DIM; j++) {
            sum += row[j] * x[j];
        }
        gates_ih[i] = sum;
    }

    // Matmul W_hh * h
    for (int i = 0; i < 3 * HIDDEN_DIM; i++) {
        float sum = gru->b_hh[i];
        const float* row = &gru->w_hh[i * HIDDEN_DIM];
        for (int j = 0; j < HIDDEN_DIM; j++) {
            sum += row[j] * gru->h[j];
        }
        gates_hh[i] = sum;
    }

    // Gate elementwise activations
    for (int i = 0; i < HIDDEN_DIM; i++) {
        float r = sigmoidf_fast(gates_ih[i] + gates_hh[i]);
        float z = sigmoidf_fast(gates_ih[HIDDEN_DIM + i] + gates_hh[HIDDEN_DIM + i]);
        float n = tanhf(gates_ih[2 * HIDDEN_DIM + i] + r * gates_hh[2 * HIDDEN_DIM + i]);
        gru->h[i] = (1.0f - z) * n + z * gru->h[i];
        out[i] = gru->h[i];
    }
}

/* ========================================================================= */
/* 2. Retention Multi-Head Single-Layer Step                                 */
/* ========================================================================= */

typedef struct {
    float w_q[HIDDEN_DIM * HIDDEN_DIM];
    float w_k[HIDDEN_DIM * HIDDEN_DIM];
    float w_v[HIDDEN_DIM * HIDDEN_DIM];
    float w_out[HIDDEN_DIM * HIDDEN_DIM];
    float gamma[NUM_HEADS];
    float state[NUM_HEADS][HEAD_DIM][HEAD_DIM]; // 14 * 64 * 64 = 57,344 floats (~229 KB)
} retention_layer_t;

void retention_step(retention_layer_t* ret, const float* x, float* out) {
    float q[HIDDEN_DIM];
    float k[HIDDEN_DIM];
    float v[HIDDEN_DIM];
    float o_heads[HIDDEN_DIM];

    // 1. Projections W_q, W_k, W_v
    for (int i = 0; i < HIDDEN_DIM; i++) {
        float sum_q = 0.0f, sum_k = 0.0f, sum_v = 0.0f;
        const float* row_q = &ret->w_q[i * HIDDEN_DIM];
        const float* row_k = &ret->w_k[i * HIDDEN_DIM];
        const float* row_v = &ret->w_v[i * HIDDEN_DIM];
        for (int j = 0; j < HIDDEN_DIM; j++) {
            float xj = x[j];
            sum_q += row_q[j] * xj;
            sum_k += row_k[j] * xj;
            sum_v += row_v[j] * xj;
        }
        q[i] = sum_q;
        k[i] = sum_k;
        v[i] = sum_v;
    }

    // 2. Multi-Head State Update & Read (Pure MACs, Zero Transcendentals!)
    for (int h = 0; h < NUM_HEADS; h++) {
        const float* q_h = &q[h * HEAD_DIM];
        const float* k_h = &k[h * HEAD_DIM];
        const float* v_h = &v[h * HEAD_DIM];
        float* o_h = &o_heads[h * HEAD_DIM];
        float g = ret->gamma[h];

        // S = gamma * S + k @ v
        for (int i = 0; i < HEAD_DIM; i++) {
            float ki = k_h[i];
            float* s_row = ret->state[h][i];
            for (int j = 0; j < HEAD_DIM; j++) {
                s_row[j] = g * s_row[j] + ki * v_h[j];
            }
        }

        // o = q @ S
        for (int j = 0; j < HEAD_DIM; j++) {
            float sum = 0.0f;
            for (int i = 0; i < HEAD_DIM; i++) {
                sum += q_h[i] * ret->state[h][i][j];
            }
            o_h[j] = sum;
        }
    }

    // 3. Output Projection W_out
    for (int i = 0; i < HIDDEN_DIM; i++) {
        float sum = 0.0f;
        const float* row_out = &ret->w_out[i * HIDDEN_DIM];
        for (int j = 0; j < HIDDEN_DIM; j++) {
            sum += row_out[j] * o_heads[j];
        }
        out[i] = sum;
    }
}

/* ========================================================================= */
/* Benchmark Runner                                                          */
/* ========================================================================= */

int main() {
    printf("=================================================================\n");
    printf("   🔬 C NATIVE MICROBENCHMARK: SINGLE-LAYER GRU VS RETENTION     \n");
    printf("   Hidden Dim: %d | Heads: %d | Head Dim: %d\n", HIDDEN_DIM, NUM_HEADS, HEAD_DIM);
    printf("   Iterations: %d tokens forward steps\n", NUM_ITERS);
    printf("=================================================================\n\n");

    gru_layer_t* gru = (gru_layer_t*)calloc(1, sizeof(gru_layer_t));
    retention_layer_t* ret = (retention_layer_t*)calloc(1, sizeof(retention_layer_t));
    float* x = (float*)calloc(HIDDEN_DIM, sizeof(float));
    float* out = (float*)calloc(HIDDEN_DIM, sizeof(float));

    for (int i = 0; i < NUM_HEADS; i++) ret->gamma[i] = 0.95f;
    for (int i = 0; i < HIDDEN_DIM; i++) x[i] = 0.01f * (i % 10);

    // Warmup
    for (int i = 0; i < 500; i++) {
        gru_step(gru, x, out);
        retention_step(ret, x, out);
    }

    // 1. Benchmark GRU
    clock_t t0 = clock();
    for (int i = 0; i < NUM_ITERS; i++) {
        gru_step(gru, x, out);
    }
    clock_t t1 = clock();
    double sec_gru = (double)(t1 - t0) / CLOCKS_PER_SEC;
    double tps_gru = NUM_ITERS / sec_gru;

    // 2. Benchmark Retention
    t0 = clock();
    for (int i = 0; i < NUM_ITERS; i++) {
        retention_step(ret, x, out);
    }
    t1 = clock();
    double sec_ret = (double)(t1 - t0) / CLOCKS_PER_SEC;
    double tps_ret = NUM_ITERS / sec_ret;

    double ratio = (tps_ret / tps_gru) * 100.0;

    printf("[*] GRU Single Layer Forward:\n");
    printf("    - Elapsed Time: %.4fs\n", sec_gru);
    printf("    - Throughput:   %.1f TPS (Tokens Per Second)\n\n", tps_gru);

    printf("[*] Retention Single Layer Forward:\n");
    printf("    - Elapsed Time: %.4fs\n", sec_ret);
    printf("    - Throughput:   %.1f TPS (Tokens Per Second)\n\n", tps_ret);

    printf("[*] Comparison Ratio:\n");
    printf("    - TPS_retention / TPS_gru_baseline: %.2f%%\n\n", ratio);

    if (ratio >= 80.0) {
        printf("[DECISION GATE] ✅ PASSED (>= 80%%): Retention throughput is superior/competitive!\n");
    } else if (ratio >= 70.0) {
        printf("[DECISION GATE] 🟡 BORDERLINE (70-80%%): Tradeoff area.\n");
    } else {
        printf("[DECISION GATE] ❌ FAILED (< 70%%): GRU is faster on raw single-layer execution.\n");
    }
    printf("=================================================================\n");

    free(gru);
    free(ret);
    free(x);
    free(out);
    return 0;
}
