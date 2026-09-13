/**
 * ============================================================================
 *  WRAI-MICRO (1.25M) PURE NATIVE C RECURRENT ENGINE
 * ============================================================================
 *  Target Architecture: Dual-State Linear Recurrence (Zero KV-Cache)
 *  Target Silicon     : ESP32 DevKit (240MHz, 320KB SRAM, 4MB Flash)
 *  External PSRAM     : 0 MB (Not Needed! Runs 100% in Internal SRAM)
 *  Memory Complexity  : Strict O(1) Constant RAM (Flat Heap)
 * ============================================================================
 */

#ifndef WRAI_MICRO_ENGINE_H
#define WRAI_MICRO_ENGINE_H

#include <Arduino.h>
#include <math.h>
#include <pgmspace.h>

#define WRAI_VOCAB_SIZE    4023
#define WRAI_HIDDEN_DIM    128
#define WRAI_NUM_LAYERS    4
#define WRAI_NUM_HEADS     4
#define WRAI_HEAD_DIM      32
#define WRAI_FFN_DIM       341

static inline float wrai_silu(float x) {
    return x / (1.0f + expf(-x));
}

// Fixed O(1) Memory Layout in Internal SRAM (Zero malloc)
typedef struct {
    // Recurrent State: 4 layers x 4 heads x 32 x 32 floats = 64 KB
    float state_m[WRAI_NUM_LAYERS][WRAI_NUM_HEADS][WRAI_HEAD_DIM][WRAI_HEAD_DIM];

    // Working Buffers
    float x[WRAI_HIDDEN_DIM];
    float x_norm[WRAI_HIDDEN_DIM];
    float q[WRAI_HIDDEN_DIM];
    float k[WRAI_HIDDEN_DIM];
    float v[WRAI_HIDDEN_DIM];
    float r[WRAI_HIDDEN_DIM];
    float y[WRAI_HIDDEN_DIM];
    float out_ret[WRAI_HIDDEN_DIM];

    // SwiGLU FFN Buffers
    float ffn_gate[WRAI_FFN_DIM];
    float ffn_up[WRAI_FFN_DIM];
    float ffn_down[WRAI_HIDDEN_DIM];
} WRAIMicroContext;

static WRAIMicroContext ctx;

class WRAIMicroEngine {
public:
    static void init() {
        memset(&ctx, 0, sizeof(WRAIMicroContext));
        Serial.printf("[WRAI] Pure Neural Engine Initialized. SRAM State: %u bytes (%.1f KB)\n",
                      (unsigned int)sizeof(WRAIMicroContext),
                      (float)sizeof(WRAIMicroContext) / 1024.0f);
        Serial.printf("[WRAI] Free SRAM Heap: %u bytes\n", ESP.getFreeHeap());
    }

    static void reset_state() {
        memset(ctx.state_m, 0, sizeof(ctx.state_m));
    }

    static void rmsnorm(float* out, const float* in, const float* weight, int dim, float eps = 1e-6f) {
        float sum_sq = 0.0f;
        for (int i = 0; i < dim; i++) {
            sum_sq += in[i] * in[i];
        }
        float scale = 1.0f / sqrtf((sum_sq / (float)dim) + eps);
        for (int i = 0; i < dim; i++) {
            out[i] = in[i] * scale * weight[i];
        }
    }

    static const uint8_t* matmul_int8_pgm(float* out, const float* in, const uint8_t* pgm_ptr, int rows, int cols) {
        float scale;
        memcpy_P(&scale, pgm_ptr, sizeof(float));
        pgm_ptr += sizeof(float);

        for (int r = 0; r < rows; r++) {
            int32_t acc = 0;
            for (int c = 0; c < cols; c++) {
                int8_t w = (int8_t)pgm_read_byte(pgm_ptr++);
                acc += (int32_t)w * (int32_t)(in[c] * 127.0f);
            }
            out[r] = ((float)acc / 127.0f) * scale;
        }
        return pgm_ptr;
    }

    static const uint8_t* load_fp32_pgm(float* out, const uint8_t* pgm_ptr, int count) {
        memcpy_P(out, pgm_ptr, count * sizeof(float));
        return pgm_ptr + (count * sizeof(float));
    }

    // Full Recurrent Neural Forward Pass for 1 Token
    static int forward_step(int token_id, const uint8_t* weights_payload) {
        if (token_id < 0 || token_id >= WRAI_VOCAB_SIZE) {
            token_id = 4021; // <unk>
        }

        const uint8_t* p = weights_payload + 32;

        // 1. Embedding lookup
        float embed_scale;
        memcpy_P(&embed_scale, p, sizeof(float));
        const uint8_t* embed_data = p + sizeof(float);
        const uint8_t* tok_w = embed_data + (token_id * WRAI_HIDDEN_DIM);
        for (int i = 0; i < WRAI_HIDDEN_DIM; i++) {
            ctx.x[i] = (float)((int8_t)pgm_read_byte(tok_w + i)) * embed_scale;
        }
        p += sizeof(float) + (WRAI_VOCAB_SIZE * WRAI_HIDDEN_DIM);

        // 2. 4 Layers
        for (int l = 0; l < WRAI_NUM_LAYERS; l++) {
            float in_norm_w[WRAI_HIDDEN_DIM];
            p = load_fp32_pgm(in_norm_w, p, WRAI_HIDDEN_DIM);
            rmsnorm(ctx.x_norm, ctx.x, in_norm_w, WRAI_HIDDEN_DIM);

            p = matmul_int8_pgm(ctx.q, ctx.x_norm, p, WRAI_HIDDEN_DIM, WRAI_HIDDEN_DIM);
            p = matmul_int8_pgm(ctx.k, ctx.x_norm, p, WRAI_HIDDEN_DIM, WRAI_HIDDEN_DIM);
            p = matmul_int8_pgm(ctx.v, ctx.x_norm, p, WRAI_HIDDEN_DIM, WRAI_HIDDEN_DIM);
            p = matmul_int8_pgm(ctx.r, ctx.x_norm, p, WRAI_HIDDEN_DIM, WRAI_HIDDEN_DIM);

            float gn_w[WRAI_HIDDEN_DIM], gn_b[WRAI_HIDDEN_DIM], gamma[WRAI_NUM_HEADS];
            p = load_fp32_pgm(gn_w, p, WRAI_HIDDEN_DIM);
            p = load_fp32_pgm(gn_b, p, WRAI_HIDDEN_DIM);
            p = load_fp32_pgm(gamma, p, WRAI_NUM_HEADS);

            float scale_ret = 1.0f / sqrtf(WRAI_HEAD_DIM);
            for (int h = 0; h < WRAI_NUM_HEADS; h++) {
                float g = gamma[h];
                const float* q_h = &ctx.q[h * WRAI_HEAD_DIM];
                const float* k_h = &ctx.k[h * WRAI_HEAD_DIM];
                const float* v_h = &ctx.v[h * WRAI_HEAD_DIM];
                float* y_h = &ctx.y[h * WRAI_HEAD_DIM];

                for (int i = 0; i < WRAI_HEAD_DIM; i++) {
                    for (int j = 0; j < WRAI_HEAD_DIM; j++) {
                        ctx.state_m[l][h][i][j] = (g * ctx.state_m[l][h][i][j]) + (k_h[i] * v_h[j]);
                    }
                }

                for (int j = 0; j < WRAI_HEAD_DIM; j++) {
                    float acc = 0.0f;
                    for (int i = 0; i < WRAI_HEAD_DIM; i++) {
                        acc += q_h[i] * ctx.state_m[l][h][i][j];
                    }
                    y_h[j] = acc * scale_ret;
                }

                float mean = 0.0f;
                for (int j = 0; j < WRAI_HEAD_DIM; j++) mean += y_h[j];
                mean /= (float)WRAI_HEAD_DIM;

                float var = 0.0f;
                for (int j = 0; j < WRAI_HEAD_DIM; j++) {
                    float diff = y_h[j] - mean;
                    var += diff * diff;
                }
                float rsqrt = 1.0f / sqrtf((var / (float)WRAI_HEAD_DIM) + 1e-5f);

                for (int j = 0; j < WRAI_HEAD_DIM; j++) {
                    int idx = h * WRAI_HEAD_DIM + j;
                    y_h[j] = ((y_h[j] - mean) * rsqrt * gn_w[idx]) + gn_b[idx];
                }
            }

            for (int i = 0; i < WRAI_HIDDEN_DIM; i++) {
                ctx.y[i] = ctx.y[i] * wrai_silu(ctx.r[i]);
            }

            p = matmul_int8_pgm(ctx.out_ret, ctx.y, p, WRAI_HIDDEN_DIM, WRAI_HIDDEN_DIM);
            for (int i = 0; i < WRAI_HIDDEN_DIM; i++) ctx.x[i] += ctx.out_ret[i];

            float post_norm_w[WRAI_HIDDEN_DIM];
            p = load_fp32_pgm(post_norm_w, p, WRAI_HIDDEN_DIM);
            rmsnorm(ctx.x_norm, ctx.x, post_norm_w, WRAI_HIDDEN_DIM);

            p = matmul_int8_pgm(ctx.ffn_gate, ctx.x_norm, p, WRAI_FFN_DIM, WRAI_HIDDEN_DIM);
            p = matmul_int8_pgm(ctx.ffn_up, ctx.x_norm, p, WRAI_FFN_DIM, WRAI_HIDDEN_DIM);
            for (int i = 0; i < WRAI_FFN_DIM; i++) {
                ctx.ffn_gate[i] = wrai_silu(ctx.ffn_gate[i]) * ctx.ffn_up[i];
            }
            p = matmul_int8_pgm(ctx.ffn_down, ctx.ffn_gate, p, WRAI_HIDDEN_DIM, WRAI_FFN_DIM);
            for (int i = 0; i < WRAI_HIDDEN_DIM; i++) ctx.x[i] += ctx.ffn_down[i];
        }

        // 3. Final RMSNorm
        float final_norm[WRAI_HIDDEN_DIM];
        load_fp32_pgm(final_norm, p, WRAI_HIDDEN_DIM);
        rmsnorm(ctx.x_norm, ctx.x, final_norm, WRAI_HIDDEN_DIM);

        // 4. LM Head (Search for best token excluding special/pad)
        int best_token = 0;
        float max_logit = -1e9f;
        for (int i = 3; i < WRAI_VOCAB_SIZE - 4; i++) {
            float logit = 0.0f;
            const uint8_t* row = embed_data + (i * WRAI_HIDDEN_DIM);
            for (int j = 0; j < WRAI_HIDDEN_DIM; j++) {
                logit += ctx.x_norm[j] * ((float)((int8_t)pgm_read_byte(row + j)) * embed_scale);
            }
            if (logit > max_logit) {
                max_logit = logit;
                best_token = i;
            }
        }
        return best_token;
    }
};

#endif // WRAI_MICRO_ENGINE_H
