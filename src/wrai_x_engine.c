/**
 * @file wrai_x_engine.c
 * @brief Ultra-Fast AVX 1.0 Native Inference Engine for WRAI-X (0.6B)
 *        Dual-State (Mt/Rt), Haar 4-Level Multiresolution, HDC Scratchpad.
 */

#include "wrai_x_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

static inline float fast_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static inline float fast_silu(float x) {
    return x / (1.0f + expf(-x));
}

/* AVX 1.0 Vectorized RMSNorm */
static void rms_norm(const float* x, const float* weight, float* y, int dim) {
    float sum_sq = 0.0f;
#if defined(__AVX__)
    __m256 v_sum = _mm256_setzero_ps();
    int i = 0;
    for (; i <= dim - 8; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        v_sum = _mm256_add_ps(v_sum, _mm256_mul_ps(vx, vx));
    }
    __m128 lo = _mm256_castps256_ps128(v_sum);
    __m128 hi = _mm256_extractf128_ps(v_sum, 1);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum_sq = _mm_cvtss_f32(sum128);
    for (; i < dim; i++) sum_sq += x[i] * x[i];
#else
    for (int i = 0; i < dim; i++) sum_sq += x[i] * x[i];
#endif
    float scale = 1.0f / sqrtf((sum_sq / (float)dim) + 1e-6f);
    for (int i = 0; i < dim; i++) {
        y[i] = x[i] * scale * weight[i];
    }
}

/* AVX 1.0 INT8 GEMV: y = A @ x (A: M x K in int8, scales: M in float) */
static void gemv_int8(
    const int8_t* A,
    const float* A_scales,
    const float* x,
    float* y,
    int M,
    int K
) {
#pragma omp parallel for schedule(static) if(M >= 256)
    for (int r = 0; r < M; r++) {
        const int8_t* a_row = A + (size_t)r * K;
        float row_scale = A_scales[r];
        float sum = 0.0f;

#if defined(__AVX__)
        __m256 v_acc = _mm256_setzero_ps();
        int c = 0;
        for (; c <= K - 8; c += 8) {
            __m128i a_bytes = _mm_loadu_si128((const __m128i*)(a_row + c));
            __m128i a_epi16 = _mm_cvtepi8_epi16(a_bytes);
            __m128i a_lo = _mm_cvtepi16_epi32(a_epi16);
            __m128i a_hi = _mm_cvtepi16_epi32(_mm_srli_si128(a_epi16, 8));

            __m128 a_f_lo = _mm_cvtepi32_ps(a_lo);
            __m128 a_f_hi = _mm_cvtepi32_ps(a_hi);

            __m256 a_f = _mm256_insertf128_ps(_mm256_castps128_ps256(a_f_lo), a_f_hi, 1);
            __m256 x_f = _mm256_loadu_ps(x + c);
            v_acc = _mm256_add_ps(v_acc, _mm256_mul_ps(a_f, x_f));
        }

        __m128 lo = _mm256_castps256_ps128(v_acc);
        __m128 hi = _mm256_extractf128_ps(v_acc, 1);
        __m128 sum128 = _mm_add_ps(lo, hi);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum = _mm_cvtss_f32(sum128);
        for (; c < K; c++) sum += (float)a_row[c] * x[c];
#else
        for (int c = 0; c < K; c++) sum += (float)a_row[c] * x[c];
#endif
        y[r] = sum * row_scale;
    }
}

/* Haar 4-Level Multiresolution Hierarchy (D=1024 -> 64) */
static void haar_multiresolution_1d(
    const float* in,
    float* out_filtered,
    float low_gain,
    float mid_gain,
    float high_gain,
    const float* gate_w,
    const float* gate_b
) {
    const float sqrt2 = 1.41421356237f;
    const float inv_sqrt2 = 0.70710678118f;

    float approx_buf[1024];
    float d0[512], d1[256], d2[128], d3[64];
    memcpy(approx_buf, in, 1024 * sizeof(float));

    /* Level 0: 1024 -> approx 512, d0 512 */
    for (int i = 0; i < 512; i++) {
        float even = approx_buf[2 * i];
        float odd  = approx_buf[2 * i + 1];
        approx_buf[i] = (even + odd) * inv_sqrt2;
        d0[i] = (even - odd) * inv_sqrt2 * high_gain;
    }
    /* Level 1: 512 -> approx 256, d1 256 */
    for (int i = 0; i < 256; i++) {
        float even = approx_buf[2 * i];
        float odd  = approx_buf[2 * i + 1];
        approx_buf[i] = (even + odd) * inv_sqrt2;
        d1[i] = (even - odd) * inv_sqrt2 * mid_gain;
    }
    /* Level 2: 256 -> approx 128, d2 128 */
    for (int i = 0; i < 128; i++) {
        float even = approx_buf[2 * i];
        float odd  = approx_buf[2 * i + 1];
        approx_buf[i] = (even + odd) * inv_sqrt2;
        d2[i] = (even - odd) * inv_sqrt2 * mid_gain;
    }
    /* Level 3: 128 -> approx 64, d3 64 */
    for (int i = 0; i < 64; i++) {
        float even = approx_buf[2 * i];
        float odd  = approx_buf[2 * i + 1];
        approx_buf[i] = (even + odd) * inv_sqrt2 * low_gain;
        d3[i] = (even - odd) * inv_sqrt2 * mid_gain;
    }

    /* Inverse Reconstruction */
    /* Inv Level 3: 64 -> 128 */
    float rec128[128];
    for (int i = 0; i < 64; i++) {
        rec128[2 * i]     = (approx_buf[i] + d3[i]) * inv_sqrt2;
        rec128[2 * i + 1] = (approx_buf[i] - d3[i]) * inv_sqrt2;
    }
    /* Inv Level 2: 128 -> 256 */
    float rec256[256];
    for (int i = 0; i < 128; i++) {
        rec256[2 * i]     = (rec128[i] + d2[i]) * inv_sqrt2;
        rec256[2 * i + 1] = (rec128[i] - d2[i]) * inv_sqrt2;
    }
    /* Inv Level 1: 256 -> 512 */
    float rec512[512];
    for (int i = 0; i < 256; i++) {
        rec512[2 * i]     = (rec256[i] + d1[i]) * inv_sqrt2;
        rec512[2 * i + 1] = (rec256[i] - d1[i]) * inv_sqrt2;
    }
    /* Inv Level 0: 512 -> 1024 */
    float rec1024[1024];
    for (int i = 0; i < 512; i++) {
        rec1024[2 * i]     = (rec512[i] + d0[i]) * inv_sqrt2;
        rec1024[2 * i + 1] = (rec512[i] - d0[i]) * inv_sqrt2;
    }

    for (int i = 0; i < 1024; i++) {
        float g = fast_sigmoid(in[i] * gate_w[i] + gate_b[i]);
        out_filtered[i] = in[i] + (rec1024[i] * g);
    }
}

/* ============================================================================
 * MODEL LOADER (Memory-Mapped Binary Reader)
 * ============================================================================ */
bool wrai_x_load_model(const char* bin_path, wrai_x_model_t* model) {
    if (!bin_path || !model) return false;
    memset(model, 0, sizeof(wrai_x_model_t));

#ifdef _WIN32
    model->hFile = CreateFileA(bin_path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (model->hFile == INVALID_HANDLE_VALUE) return false;

    LARGE_INTEGER li;
    GetFileSizeEx(model->hFile, &li);
    model->file_size = (size_t)li.QuadPart;

    model->hMapping = CreateFileMappingA(model->hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!model->hMapping) { CloseHandle(model->hFile); return false; }

    model->mmap_base = MapViewOfFile(model->hMapping, FILE_MAP_READ, 0, 0, 0);
    if (!model->mmap_base) { CloseHandle(model->hMapping); CloseHandle(model->hFile); return false; }
#else
    FILE* f = fopen(bin_path, "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    model->file_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    model->mmap_base = malloc(model->file_size);
    fread(model->mmap_base, 1, model->file_size, f);
    fclose(f);
#endif

    const uint8_t* ptr = (const uint8_t*)model->mmap_base;
    memcpy(&model->header, ptr, sizeof(wrai_x_header_t));
    ptr += sizeof(wrai_x_header_t);

    if (model->header.magic != WRAI_X_MAGIC || model->header.version != WRAI_X_VERSION) {
        wrai_x_free_model(model);
        return false;
    }

    /* 1. Embeddings */
    model->embed_scales = (const float*)ptr; ptr += (size_t)WRAI_X_VOCAB_SIZE * sizeof(float);
    model->embed_data   = (const int8_t*)ptr; ptr += (size_t)WRAI_X_VOCAB_SIZE * WRAI_X_HIDDEN_DIM;

    /* 2. 28 Layers */
    for (int l = 0; l < WRAI_X_NUM_LAYERS; l++) {
        wrai_x_layer_weights_t* lw = &model->layers[l];
        lw->rms_ret = (const float*)ptr; ptr += WRAI_X_HIDDEN_DIM * sizeof(float);
        lw->rms_ffn = (const float*)ptr; ptr += WRAI_X_HIDDEN_DIM * sizeof(float);

        /* Memory RetNet */
        lw->w_q_scales = (const float*)ptr; ptr += 2048 * sizeof(float);
        lw->w_q_data   = (const int8_t*)ptr; ptr += 2048 * 1024;
        lw->w_k_scales = (const float*)ptr; ptr += 2048 * sizeof(float);
        lw->w_k_data   = (const int8_t*)ptr; ptr += 2048 * 1024;
        lw->w_v_scales = (const float*)ptr; ptr += 2048 * sizeof(float);
        lw->w_v_data   = (const int8_t*)ptr; ptr += 2048 * 1024;
        lw->w_out_scales = (const float*)ptr; ptr += 1024 * sizeof(float);
        lw->w_out_data   = (const int8_t*)ptr; ptr += 1024 * 2048;

        /* Decays */
        lw->decay_m = (const float*)ptr; ptr += WRAI_X_NUM_HEADS * sizeof(float);
        lw->decay_r = (const float*)ptr; ptr += WRAI_X_NUM_HEADS * sizeof(float);

        /* Haar Bridge */
        lw->low_gain  = *(const float*)ptr; ptr += sizeof(float);
        lw->mid_gain  = *(const float*)ptr; ptr += sizeof(float);
        lw->high_gain = *(const float*)ptr; ptr += sizeof(float);
        lw->haar_gate_w = (const float*)ptr; ptr += 1024 * sizeof(float);
        lw->haar_gate_b = (const float*)ptr; ptr += 1024 * sizeof(float);

        /* Reasoning RetNet */
        lw->w_qr_scales = (const float*)ptr; ptr += 2048 * sizeof(float);
        lw->w_qr_data   = (const int8_t*)ptr; ptr += 2048 * 1024;
        lw->w_kr_scales = (const float*)ptr; ptr += 2048 * sizeof(float);
        lw->w_kr_data   = (const int8_t*)ptr; ptr += 2048 * 1024;
        lw->w_vr_scales = (const float*)ptr; ptr += 2048 * sizeof(float);
        lw->w_vr_data   = (const int8_t*)ptr; ptr += 2048 * 1024;
        lw->w_out_r_scales = (const float*)ptr; ptr += 1024 * sizeof(float);
        lw->w_out_r_data   = (const int8_t*)ptr; ptr += 1024 * 2048;

        /* Thinking Gate */
        lw->think_gate_w = (const float*)ptr; ptr += (1024 * 2048) * sizeof(float);
        lw->think_gate_b = (const float*)ptr; ptr += 1024 * sizeof(float);

        /* HDC Scratchpad */
        lw->hdc_k_scales = (const float*)ptr; ptr += 1024 * sizeof(float);
        lw->hdc_k_data   = (const int8_t*)ptr; ptr += 1024 * 1024;
        lw->hdc_v_scales = (const float*)ptr; ptr += 1024 * sizeof(float);
        lw->hdc_v_data   = (const int8_t*)ptr; ptr += 1024 * 1024;
        lw->hdc_gate_w   = (const float*)ptr; ptr += (1024 * 2048) * sizeof(float);
        lw->hdc_gate_b   = (const float*)ptr; ptr += 1024 * sizeof(float);

        /* SwiGLU FFN */
        lw->w_gate_scales = (const float*)ptr; ptr += 3072 * sizeof(float);
        lw->w_gate_data   = (const int8_t*)ptr; ptr += 3072 * 1024;
        lw->w_up_scales   = (const float*)ptr; ptr += 3072 * sizeof(float);
        lw->w_up_data     = (const int8_t*)ptr; ptr += 3072 * 1024;
        lw->w_down_scales = (const float*)ptr; ptr += 1024 * sizeof(float);
        lw->w_down_data   = (const int8_t*)ptr; ptr += 1024 * 3072;
    }

    model->ln_final = (const float*)ptr;
    return true;
}

void wrai_x_free_model(wrai_x_model_t* model) {
    if (!model) return;
#ifdef _WIN32
    if (model->mmap_base) UnmapViewOfFile(model->mmap_base);
    if (model->hMapping) CloseHandle(model->hMapping);
    if (model->hFile != INVALID_HANDLE_VALUE) CloseHandle(model->hFile);
#else
    if (model->mmap_base) free(model->mmap_base);
#endif
    memset(model, 0, sizeof(wrai_x_model_t));
}

/* ============================================================================
 * STATE MANAGEMENT
 * ============================================================================ */
bool wrai_x_state_init(wrai_x_state_t* state) {
    if (!state) return false;
    size_t floats_ret = (size_t)WRAI_X_NUM_LAYERS * WRAI_X_NUM_HEADS * WRAI_X_HEAD_DIM * WRAI_X_HEAD_DIM;
    size_t floats_z   = (size_t)WRAI_X_NUM_LAYERS * WRAI_X_NUM_HEADS * WRAI_X_HEAD_DIM;
    state->state_m = (float*)calloc(floats_ret, sizeof(float));
    state->state_zm = (float*)calloc(floats_z, sizeof(float));
    state->state_r = (float*)calloc(floats_ret, sizeof(float));
    state->state_zr = (float*)calloc(floats_z, sizeof(float));
    state->state_hdc = (float*)calloc((size_t)WRAI_X_NUM_LAYERS * WRAI_X_HIDDEN_DIM, sizeof(float));
    state->current_pos = 0;
    return (state->state_m && state->state_zm && state->state_r && state->state_zr && state->state_hdc);
}

void wrai_x_state_reset(wrai_x_state_t* state) {
    if (!state) return;
    size_t floats_ret = (size_t)WRAI_X_NUM_LAYERS * WRAI_X_NUM_HEADS * WRAI_X_HEAD_DIM * WRAI_X_HEAD_DIM;
    size_t floats_z   = (size_t)WRAI_X_NUM_LAYERS * WRAI_X_NUM_HEADS * WRAI_X_HEAD_DIM;
    if (state->state_m) memset(state->state_m, 0, floats_ret * sizeof(float));
    if (state->state_zm) memset(state->state_zm, 0, floats_z * sizeof(float));
    if (state->state_r) memset(state->state_r, 0, floats_ret * sizeof(float));
    if (state->state_zr) memset(state->state_zr, 0, floats_z * sizeof(float));
    if (state->state_hdc) memset(state->state_hdc, 0, WRAI_X_NUM_LAYERS * WRAI_X_HIDDEN_DIM * sizeof(float));
    state->current_pos = 0;
}

void wrai_x_state_free(wrai_x_state_t* state) {
    if (!state) return;
    if (state->state_m) free(state->state_m);
    if (state->state_zm) free(state->state_zm);
    if (state->state_r) free(state->state_r);
    if (state->state_zr) free(state->state_zr);
    if (state->state_hdc) free(state->state_hdc);
    memset(state, 0, sizeof(wrai_x_state_t));
}

/* ============================================================================
 * FORWARD STEP EXECUTION
 * ============================================================================ */
void wrai_x_forward_step(
    const wrai_x_model_t* model,
    wrai_x_state_t* state,
    int32_t token_id,
    float* out_logits
) {
    float x[WRAI_X_HIDDEN_DIM];
    float x_norm[WRAI_X_HIDDEN_DIM];

    /* 1. Embedding lookup */
    const int8_t* emb_row = model->embed_data + (size_t)token_id * WRAI_X_HIDDEN_DIM;
    float emb_scale = model->embed_scales[token_id];
    for (int i = 0; i < WRAI_X_HIDDEN_DIM; i++) {
        x[i] = (float)emb_row[i] * emb_scale;
    }

    float q[2048], k[2048], v[2048], o_m[1024];
    float o_m_filtered[1024];
    float qr[2048], kr[2048], vr[2048], o_r[1024];
    float gate_ffn[3072], up_ffn[3072], down_ffn[1024];

    /* 2. 28 Layers Execution */
    for (int l = 0; l < WRAI_X_NUM_LAYERS; l++) {
        const wrai_x_layer_weights_t* lw = &model->layers[l];
        rms_norm(x, lw->rms_ret, x_norm, WRAI_X_HIDDEN_DIM);

        /* Memory State RetNet Forward Step */
        gemv_int8(lw->w_q_data, lw->w_q_scales, x_norm, q, 2048, 1024);
        gemv_int8(lw->w_k_data, lw->w_k_scales, x_norm, k, 2048, 1024);
        gemv_int8(lw->w_v_data, lw->w_v_scales, x_norm, v, 2048, 1024);

        float* layer_sm = state->state_m + (size_t)l * (16 * 128 * 128);
        float* layer_zm = state->state_zm + (size_t)l * (16 * 128);
        float ret_out[2048];
        const float head_scale = 0.08838834764831845f; /* 1.0f / sqrtf(128.0f) */

        for (int h = 0; h < 16; h++) {
            float gamma = fast_sigmoid(lw->decay_m[h]);
            float* s_h = layer_sm + (size_t)h * (128 * 128);
            float* z_h = layer_zm + (size_t)h * 128;
            const float* q_h = q + h * 128;
            const float* k_h = k + h * 128;
            const float* v_h = v + h * 128;
            float* o_h = ret_out + h * 128;

            float phi_q[128], phi_k[128];
            for (int r = 0; r < 128; r++) {
                float qs = q_h[r] * head_scale;
                phi_q[r] = (qs > 0.0f) ? (qs + 1.0f) : expf(qs);
                float ks = k_h[r] * head_scale;
                phi_k[r] = (ks > 0.0f) ? (ks + 1.0f) : expf(ks);
            }

            for (int r = 0; r < 128; r++) {
                float kr_val = phi_k[r];
                z_h[r] = z_h[r] * gamma + kr_val;
                for (int c = 0; c < 128; c++) {
                    s_h[r * 128 + c] = s_h[r * 128 + c] * gamma + kr_val * v_h[c];
                }
            }

            float denom = 0.0f;
            for (int r = 0; r < 128; r++) {
                denom += phi_q[r] * z_h[r];
            }
            if (denom < 1e-5f) denom = 1e-5f;
            float inv_denom = 1.0f / denom;

            for (int c = 0; c < 128; c++) {
                float sum = 0.0f;
                for (int r = 0; r < 128; r++) {
                    sum += phi_q[r] * s_h[r * 128 + c];
                }
                o_h[c] = sum * inv_denom;
            }
        }
        gemv_int8(lw->w_out_data, lw->w_out_scales, ret_out, o_m, 1024, 2048);

        /* Haar Multiresolution Bridge */
        haar_multiresolution_1d(o_m, o_m_filtered, lw->low_gain, lw->mid_gain, lw->high_gain, lw->haar_gate_w, lw->haar_gate_b);

        /* Reasoning State RetNet */
        gemv_int8(lw->w_qr_data, lw->w_qr_scales, o_m_filtered, qr, 2048, 1024);
        gemv_int8(lw->w_kr_data, lw->w_kr_scales, o_m_filtered, kr, 2048, 1024);
        gemv_int8(lw->w_vr_data, lw->w_vr_scales, o_m_filtered, vr, 2048, 1024);

        float* layer_sr = state->state_r + (size_t)l * (16 * 128 * 128);
        float* layer_zr = state->state_zr + (size_t)l * (16 * 128);
        float ret_r_out[2048];
        for (int h = 0; h < 16; h++) {
            float gamma_r = fast_sigmoid(lw->decay_r[h]);
            float* sr_h = layer_sr + (size_t)h * (128 * 128);
            float* zr_h = layer_zr + (size_t)h * 128;
            const float* qr_h = qr + h * 128;
            const float* kr_h = kr + h * 128;
            const float* vr_h = vr + h * 128;
            float* or_h = ret_r_out + h * 128;

            float phi_qr[128], phi_kr[128];
            for (int r = 0; r < 128; r++) {
                float qrs = qr_h[r] * head_scale;
                phi_qr[r] = (qrs > 0.0f) ? (qrs + 1.0f) : expf(qrs);
                float krs = kr_h[r] * head_scale;
                phi_kr[r] = (krs > 0.0f) ? (krs + 1.0f) : expf(krs);
            }

            for (int r = 0; r < 128; r++) {
                float kr_val = phi_kr[r];
                zr_h[r] = zr_h[r] * gamma_r + kr_val;
                for (int c = 0; c < 128; c++) {
                    sr_h[r * 128 + c] = sr_h[r * 128 + c] * gamma_r + kr_val * vr_h[c];
                }
            }

            float denom_r = 0.0f;
            for (int r = 0; r < 128; r++) {
                denom_r += phi_qr[r] * zr_h[r];
            }
            if (denom_r < 1e-5f) denom_r = 1e-5f;
            float inv_denom_r = 1.0f / denom_r;

            for (int c = 0; c < 128; c++) {
                float sum = 0.0f;
                for (int r = 0; r < 128; r++) {
                    sum += phi_qr[r] * sr_h[r * 128 + c];
                }
                or_h[c] = sum * inv_denom_r;
            }
        }
        gemv_int8(lw->w_out_r_data, lw->w_out_r_scales, ret_r_out, o_r, 1024, 2048);

        /* Adaptive Thinking Fusion */
        for (int i = 0; i < 1024; i++) {
            x[i] += o_m_filtered[i] + o_r[i] * 0.1f;
        }

        /* SwiGLU FFN */
        rms_norm(x, lw->rms_ffn, x_norm, WRAI_X_HIDDEN_DIM);
        gemv_int8(lw->w_gate_data, lw->w_gate_scales, x_norm, gate_ffn, 3072, 1024);
        gemv_int8(lw->w_up_data, lw->w_up_scales, x_norm, up_ffn, 3072, 1024);
        for (int i = 0; i < 3072; i++) {
            gate_ffn[i] = fast_silu(gate_ffn[i]) * up_ffn[i];
        }
        gemv_int8(lw->w_down_data, lw->w_down_scales, gate_ffn, down_ffn, 1024, 3072);
        for (int i = 0; i < 1024; i++) {
            x[i] += down_ffn[i];
        }
    }

    /* 3. Final Norm & Output Logits */
    rms_norm(x, model->ln_final, x_norm, WRAI_X_HIDDEN_DIM);
    gemv_int8(model->embed_data, model->embed_scales, x_norm, out_logits, WRAI_X_VOCAB_SIZE, WRAI_X_HIDDEN_DIM);
    state->current_pos++;
}

/* ============================================================================
 * TOKENIZER LOADER & SAMPLER
 * ============================================================================ */
bool wrai_x_load_tokenizer(const char* vocab_bin_path, wrai_x_tokenizer_t* tok) {
    if (!vocab_bin_path || !tok) return false;
    memset(tok, 0, sizeof(wrai_x_tokenizer_t));

    FILE* f = fopen(vocab_bin_path, "rb");
    if (!f) return false;

    uint32_t num_tokens, max_len;
    if (fread(&num_tokens, sizeof(uint32_t), 1, f) != 1 ||
        fread(&max_len, sizeof(uint32_t), 1, f) != 1) {
        fclose(f);
        return false;
    }

    tok->num_tokens = num_tokens;
    tok->token_strings = (char**)calloc(num_tokens, sizeof(char*));
    tok->token_lens = (uint16_t*)calloc(num_tokens, sizeof(uint16_t));
    tok->im_start_id = 151644;
    tok->im_end_id   = 151645;
    tok->eos_id      = 151643;

    for (uint32_t i = 0; i < num_tokens; i++) {
        uint8_t len = 0;
        if (fread(&len, 1, 1, f) != 1) break;
        tok->token_lens[i] = len;
        tok->token_strings[i] = (char*)malloc(len + 1);
        if (fread(tok->token_strings[i], 1, len, f) != len) break;
        tok->token_strings[i][len] = '\0';
    }
    fclose(f);
    return true;
}

void wrai_x_free_tokenizer(wrai_x_tokenizer_t* tok) {
    if (!tok) return;
    if (tok->token_strings) {
        for (uint32_t i = 0; i < tok->num_tokens; i++) {
            if (tok->token_strings[i]) free(tok->token_strings[i]);
        }
        free(tok->token_strings);
    }
    if (tok->token_lens) free(tok->token_lens);
    memset(tok, 0, sizeof(wrai_x_tokenizer_t));
}

int wrai_x_sample(const float* logits, int vocab_size, float temperature, float top_p) {
    if (temperature <= 0.01f) {
        int best_id = 0;
        float best_val = logits[0];
        for (int i = 1; i < vocab_size; i++) {
            if (logits[i] > best_val) {
                best_val = logits[i];
                best_id = i;
            }
        }
        return best_id;
    }
    int best_id = 0;
    float best_val = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best_id = i;
        }
    }
    return best_id;
}
