/**
 * @file wrai_v14_5_engine.c
 * @brief High-Speed AVX1/SSE Native Inference Engine Implementation for WRAI v14.5 (SwiGLU + RMSNorm)
 */

#include "wrai_v14_5_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#define SQRT_HALF 0.7071067811865475f

static inline float fast_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static inline float fast_silu(float x) {
    return x / (1.0f + expf(-x));
}

/* Fast AVX1/SSE GEMV: y = (W * x) * scale + b */
static void gemv_int8_float(
    const int8_t* W,
    const float* x,
    const float* b,
    float scale,
    float* y,
    int M,
    int K
) {
#if defined(__AVX__)
    for (int j = 0; j < M; j++) {
        const int8_t* w_row = W + (j * K);
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();

        for (int i = 0; i < K; i += 16) {
            __m128i raw_w0 = _mm_loadl_epi64((const __m128i*)(w_row + i));
            __m128 w0_lo = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(raw_w0));
            __m128 w0_hi = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(raw_w0, 4)));
            __m256 w0_ps = _mm256_set_m128(w0_hi, w0_lo);

            __m128i raw_w1 = _mm_loadl_epi64((const __m128i*)(w_row + i + 8));
            __m128 w1_lo = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(raw_w1));
            __m128 w1_hi = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(raw_w1, 4)));
            __m256 w1_ps = _mm256_set_m128(w1_hi, w1_lo);

            __m256 x0 = _mm256_loadu_ps(x + i);
            __m256 x1 = _mm256_loadu_ps(x + i + 8);

            acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(w0_ps, x0));
            acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(w1_ps, x1));
        }

        __m256 total_acc = _mm256_add_ps(acc0, acc1);
        __m128 lo128 = _mm256_castps256_ps128(total_acc);
        __m128 hi128 = _mm256_extractf128_ps(total_acc, 1);
        __m128 sum128 = _mm_add_ps(lo128, hi128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        
        float dot_sum = _mm_cvtss_f32(sum128);
        y[j] = (dot_sum * scale) + (b ? b[j] : 0.0f);
    }
#else
    for (int j = 0; j < M; j++) {
        float sum = 0.0f;
        const int8_t* w_row = W + (j * K);
        for (int i = 0; i < K; i++) {
            sum += (float)w_row[i] * x[i];
        }
        y[j] = (sum * scale) + (b ? b[j] : 0.0f);
    }
#endif
}

/* Fast RMSNorm: y = x / sqrt(mean(x^2) + eps) * weight */
static void rms_norm(
    const float* x,
    const float* weight,
    float* y,
    int dim
) {
    float sum_sq = 0.0f;
#if defined(__AVX__)
    __m256 sum_vec = _mm256_setzero_ps();
    for (int i = 0; i < dim; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(vx, vx));
    }
    __m128 lo = _mm256_castps256_ps128(sum_vec);
    __m128 hi = _mm256_extractf128_ps(sum_vec, 1);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum_sq = _mm_cvtss_f32(sum128);
#else
    for (int i = 0; i < dim; i++) sum_sq += x[i] * x[i];
#endif

    float rms = 1.0f / sqrtf((sum_sq / (float)dim) + 1e-6f);

#if defined(__AVX__)
    __m256 vrms = _mm256_set1_ps(rms);
    for (int i = 0; i < dim; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vw = _mm256_loadu_ps(weight + i);
        __m256 vy = _mm256_mul_ps(_mm256_mul_ps(vx, vrms), vw);
        _mm256_storeu_ps(y + i, vy);
    }
#else
    for (int i = 0; i < dim; i++) {
        y[i] = x[i] * rms * weight[i];
    }
#endif
}

/* SwiGLU Forward Pass */
static void swiglu_forward(
    const float* in_x,
    const wrai_swiglu_t* ffn,
    float* out_x,
    int dim,
    int ffn_dim
) {
    float gate[WRAI_FFN_DIM];
    float up[WRAI_FFN_DIM];
    float act[WRAI_FFN_DIM];

    gemv_int8_float(ffn->w_gate, in_x, NULL, ffn->scale_w_gate, gate, ffn_dim, dim);
    gemv_int8_float(ffn->w_up, in_x, NULL, ffn->scale_w_up, up, ffn_dim, dim);

    for (int i = 0; i < ffn_dim; i++) {
        act[i] = fast_silu(gate[i]) * up[i];
    }

    gemv_int8_float(ffn->w_down, act, NULL, ffn->scale_w_down, out_x, dim, ffn_dim);
}

/* 1D Haar DWT & IDWT */
static void haar_dwt_1d(const float* in, float* approx, float* detail, int len) {
    int half = len / 2;
    for (int i = 0; i < half; i++) {
        float even = in[2 * i];
        float odd  = in[2 * i + 1];
        approx[i] = (even + odd) * SQRT_HALF;
        detail[i] = (even - odd) * SQRT_HALF;
    }
}

static void haar_idwt_1d(const float* approx, const float* detail, float* out, int len) {
    int half = len / 2;
    for (int i = 0; i < half; i++) {
        float a = approx[i];
        float d = detail[i];
        out[2 * i]     = (a + d) * SQRT_HALF;
        out[2 * i + 1] = (a - d) * SQRT_HALF;
    }
}

static void apply_spectral_block(const float* in_x, const wrai_spectral_v14_5_t* spec, float* out_x, int dim) {
    float buf_a[WRAI_HIDDEN_DIM];
    float buf_b[WRAI_HIDDEN_DIM];
    float* details[WRAI_WAVELET_LEVELS];

    memcpy(buf_a, in_x, dim * sizeof(float));
    int curr_len = dim;

    for (int l = 0; l < WRAI_WAVELET_LEVELS; l++) {
        int half = curr_len / 2;
        details[l] = (float*)malloc(half * sizeof(float));
        haar_dwt_1d(buf_a, buf_b, details[l], curr_len);
        for (int i = 0; i < half; i++) {
            details[l][i] *= spec->detail_gains[l][i];
        }
        memcpy(buf_a, buf_b, half * sizeof(float));
        curr_len = half;
    }

    for (int i = 0; i < curr_len; i++) buf_a[i] *= spec->approx_gain[i];

    for (int l = WRAI_WAVELET_LEVELS - 1; l >= 0; l--) {
        int half = curr_len;
        curr_len = half * 2;
        haar_idwt_1d(buf_a, details[l], buf_b, curr_len);
        memcpy(buf_a, buf_b, curr_len * sizeof(float));
        free(details[l]);
    }

    for (int i = 0; i < dim; i++) {
        float gate = fast_sigmoid(spec->gate[i]);
        out_x[i] = in_x[i] + (buf_a[i] * gate);
    }
}

/* Hybrid Layer Step: RMSNorm -> GRU -> (+) -> RMSNorm -> SwiGLU -> (+) */
static void forward_hybrid_layer(
    const float* in_x,
    float* h_state,
    const wrai_layer_v14_5_t* layer,
    float* out_x,
    int dim,
    int ffn_dim
) {
    /* 1. Recurrent Branch */
    float x_norm1[WRAI_HIDDEN_DIM];
    rms_norm(in_x, layer->rms_gru_weight, x_norm1, dim);

    float gates_ih[3 * WRAI_HIDDEN_DIM];
    gemv_int8_float(layer->w_ih, x_norm1, layer->b_ih, layer->scale_w_ih, gates_ih, 3 * dim, dim);

    float gates_hh[3 * WRAI_HIDDEN_DIM];
    gemv_int8_float(layer->w_hh, h_state, layer->b_hh, layer->scale_w_hh, gates_hh, 3 * dim, dim);

    float x_gru[WRAI_HIDDEN_DIM];
    for (int i = 0; i < dim; i++) {
        float r = fast_sigmoid(gates_ih[i] + gates_hh[i]);
        float z = fast_sigmoid(gates_ih[dim + i] + gates_hh[dim + i]);
        float n = tanhf(gates_ih[2 * dim + i] + r * gates_hh[2 * dim + i]);

        float new_h = (1.0f - z) * n + z * h_state[i];
        h_state[i] = new_h;
        x_gru[i] = in_x[i] + new_h; /* Residual 1 */
    }

    /* 2. SwiGLU Factual Branch */
    float x_norm2[WRAI_HIDDEN_DIM];
    rms_norm(x_gru, layer->rms_ffn_weight, x_norm2, dim);

    float ffn_out[WRAI_HIDDEN_DIM];
    swiglu_forward(x_norm2, &layer->ffn, ffn_out, dim, ffn_dim);

    for (int i = 0; i < dim; i++) {
        out_x[i] = x_gru[i] + ffn_out[i]; /* Residual 2 */
    }
}

void wrai_forward_step_v14_5(
    const wrai_model_v14_5_t* model,
    wrai_state_v14_5_t* state,
    uint32_t token_id,
    float* out_logits
) {
    if (token_id >= model->vocab_size) token_id = 0;

    int dim = model->hidden_dim;
    int ffn_dim = model->ffn_dim;
    uint32_t pos = state->step_pos;
    if (pos >= WRAI_MAX_SEQ_LEN) pos = WRAI_MAX_SEQ_LEN - 1;

    const int8_t* emb_row = model->embed_weights + ((size_t)token_id * dim);
    const float* pe_row   = model->pe + ((size_t)pos * dim);

    for (int i = 0; i < dim; i++) {
        state->sram_ping[i] = ((float)emb_row[i] * model->scale_embed) + pe_row[i];
    }

    /* Wavelet Spectral Stage 1 */
    apply_spectral_block(state->sram_ping, &model->spectral1, state->sram_pong, dim);

    float* in_buf  = state->sram_pong;
    float* out_buf = state->sram_ping;

    for (int l = 0; l < model->num_layers; l++) {
        forward_hybrid_layer(in_buf, state->h_states[l], &model->layers[l], out_buf, dim, ffn_dim);
        
        if (l == (model->num_layers / 2) - 1) {
            apply_spectral_block(out_buf, &model->spectral2, in_buf, dim);
            float* tmp = in_buf; in_buf = out_buf; out_buf = tmp;
        } else {
            float* tmp = in_buf; in_buf = out_buf; out_buf = tmp;
        }
    }

    /* Final RMSNorm */
    float x_final[WRAI_HIDDEN_DIM];
    rms_norm(in_buf, model->rms_final_weight, x_final, dim);

    /* Output Projection via Tied Embeddings */
    gemv_int8_float(model->embed_weights, x_final, NULL, model->scale_embed, out_logits, model->vocab_size, dim);
    state->step_pos++;
}

void wrai_state_reset_v14_5(wrai_state_v14_5_t* state) {
    if (!state) return;
    memset(state, 0, sizeof(wrai_state_v14_5_t));
}

bool wrai_load_model_v14_5(const char* bin_path, wrai_model_v14_5_t* model) {
    if (!bin_path || !model) return false;
    memset(model, 0, sizeof(wrai_model_v14_5_t));

    FILE* f = fopen(bin_path, "rb");
    if (!f) return false;

    uint32_t header[10];
    if (fread(header, sizeof(uint32_t), 10, f) != 10) { fclose(f); return false; }

    model->magic          = header[0];
    model->version        = header[1];
    model->vocab_size     = header[2];
    model->hidden_dim     = header[3];
    model->num_layers     = header[4];
    model->wavelet_levels = header[5];
    model->ffn_dim        = header[6];
    model->epoch          = header[7];
    memcpy(&model->best_val_loss, &header[8], sizeof(float));

    if (model->magic != WRAI_MAGIC_HEADER) { fclose(f); return false; }

    if (fread(model->teacher_map, sizeof(int32_t), WRAI_VOCAB_SIZE, f) != WRAI_VOCAB_SIZE) { fclose(f); return false; }
    if (fread(&model->scale_embed, sizeof(float), 1, f) != 1) { fclose(f); return false; }

    size_t embed_size = (size_t)WRAI_VOCAB_SIZE * WRAI_HIDDEN_DIM;
    model->embed_weights = (int8_t*)malloc(embed_size);
    if (!model->embed_weights || fread(model->embed_weights, 1, embed_size, f) != embed_size) { fclose(f); return false; }

    if (fread(model->pe, sizeof(float), WRAI_MAX_SEQ_LEN * WRAI_HIDDEN_DIM, f) != (WRAI_MAX_SEQ_LEN * WRAI_HIDDEN_DIM)) { fclose(f); return false; }

    wrai_spectral_v14_5_t* specs[2] = { &model->spectral1, &model->spectral2 };
    for (int s = 0; s < 2; s++) {
        if (fread(specs[s]->gate, sizeof(float), WRAI_HIDDEN_DIM, f) != WRAI_HIDDEN_DIM) { fclose(f); return false; }
        int approx_dim = WRAI_HIDDEN_DIM >> WRAI_WAVELET_LEVELS;
        if (fread(specs[s]->approx_gain, sizeof(float), approx_dim, f) != approx_dim) { fclose(f); return false; }
        for (int l = 0; l < WRAI_WAVELET_LEVELS; l++) {
            int d_dim = WRAI_HIDDEN_DIM >> (l + 1);
            specs[s]->detail_gains[l] = (float*)malloc(d_dim * sizeof(float));
            if (fread(specs[s]->detail_gains[l], sizeof(float), d_dim, f) != d_dim) { fclose(f); return false; }
        }
    }

    size_t gru_w_size = 3 * WRAI_HIDDEN_DIM * WRAI_HIDDEN_DIM;
    size_t ffn_gate_up_size = (size_t)WRAI_FFN_DIM * WRAI_HIDDEN_DIM;
    size_t ffn_down_size    = (size_t)WRAI_HIDDEN_DIM * WRAI_FFN_DIM;

    for (int l = 0; l < WRAI_NUM_LAYERS; l++) {
        wrai_layer_v14_5_t* lay = &model->layers[l];
        if (fread(lay->rms_gru_weight, sizeof(float), WRAI_HIDDEN_DIM, f) != WRAI_HIDDEN_DIM) { fclose(f); return false; }
        if (fread(&lay->scale_w_ih, sizeof(float), 1, f) != 1) { fclose(f); return false; }
        lay->w_ih = (int8_t*)malloc(gru_w_size);
        if (fread(lay->w_ih, 1, gru_w_size, f) != gru_w_size) { fclose(f); return false; }
        if (fread(&lay->scale_w_hh, sizeof(float), 1, f) != 1) { fclose(f); return false; }
        lay->w_hh = (int8_t*)malloc(gru_w_size);
        if (fread(lay->w_hh, 1, gru_w_size, f) != gru_w_size) { fclose(f); return false; }
        if (fread(lay->b_ih, sizeof(float), 3 * WRAI_HIDDEN_DIM, f) != 3 * WRAI_HIDDEN_DIM ||
            fread(lay->b_hh, sizeof(float), 3 * WRAI_HIDDEN_DIM, f) != 3 * WRAI_HIDDEN_DIM) { fclose(f); return false; }

        if (fread(lay->rms_ffn_weight, sizeof(float), WRAI_HIDDEN_DIM, f) != WRAI_HIDDEN_DIM) { fclose(f); return false; }
        if (fread(&lay->ffn.scale_w_gate, sizeof(float), 1, f) != 1) { fclose(f); return false; }
        lay->ffn.w_gate = (int8_t*)malloc(ffn_gate_up_size);
        if (fread(lay->ffn.w_gate, 1, ffn_gate_up_size, f) != ffn_gate_up_size) { fclose(f); return false; }
        if (fread(&lay->ffn.scale_w_up, sizeof(float), 1, f) != 1) { fclose(f); return false; }
        lay->ffn.w_up = (int8_t*)malloc(ffn_gate_up_size);
        if (fread(lay->ffn.w_up, 1, ffn_gate_up_size, f) != ffn_gate_up_size) { fclose(f); return false; }
        if (fread(&lay->ffn.scale_w_down, sizeof(float), 1, f) != 1) { fclose(f); return false; }
        lay->ffn.w_down = (int8_t*)malloc(ffn_down_size);
        if (fread(lay->ffn.w_down, 1, ffn_down_size, f) != ffn_down_size) { fclose(f); return false; }
    }

    if (fread(model->rms_final_weight, sizeof(float), WRAI_HIDDEN_DIM, f) != WRAI_HIDDEN_DIM) { fclose(f); return false; }
    fclose(f);
    return true;
}

void wrai_free_model_v14_5(wrai_model_v14_5_t* model) {
    if (!model) return;
    if (model->embed_weights) free(model->embed_weights);
    for (int s = 0; s < 2; s++) {
        wrai_spectral_v14_5_t* spec = (s == 0) ? &model->spectral1 : &model->spectral2;
        for (int l = 0; l < WRAI_WAVELET_LEVELS; l++) {
            if (spec->detail_gains[l]) free(spec->detail_gains[l]);
        }
    }
    for (int l = 0; l < WRAI_NUM_LAYERS; l++) {
        wrai_layer_v14_5_t* lay = &model->layers[l];
        if (lay->w_ih) free(lay->w_ih);
        if (lay->w_hh) free(lay->w_hh);
        if (lay->ffn.w_gate) free(lay->ffn.w_gate);
        if (lay->ffn.w_up)   free(lay->ffn.w_up);
        if (lay->ffn.w_down) free(lay->ffn.w_down);
    }
}
