/**
 * @file wrai_v14_engine.c
 * @brief High-Performance Native C Inference Engine for WRAI v14.4
 *
 * Implements AVX/SSE-accelerated Int8 GEMV, 1D Haar DWT Spectral Layer,
 * 12 ResGRU layers with 8 KB L1 Ping-Pong Buffer, 0% KV-Cache.
 */

#include "wrai_v14_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>

#define INV_SQRT2 0.7071067811865475f

static inline float fast_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

/* AVX / SSE Accelerated Int8 GEMV: y[j] = scale * (sum_{i} W[j, i] * x[i]) + b[j] */
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
            /* Load 8 int8 weights -> 8 floats in AVX 256-bit register */
            __m128i raw_w0 = _mm_loadl_epi64((const __m128i*)(w_row + i));
            __m128 w0_lo = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(raw_w0));
            __m128 w0_hi = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(raw_w0, 4)));
            __m256 w0_ps = _mm256_set_m128(w0_hi, w0_lo);

            /* Load next 8 int8 weights */
            __m128i raw_w1 = _mm_loadl_epi64((const __m128i*)(w_row + i + 8));
            __m128 w1_lo = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(raw_w1));
            __m128 w1_hi = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(raw_w1, 4)));
            __m256 w1_ps = _mm256_set_m128(w1_hi, w1_lo);

            /* Load 16 floats from x */
            __m256 x0 = _mm256_loadu_ps(x + i);
            __m256 x1 = _mm256_loadu_ps(x + i + 8);

            acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(w0_ps, x0));
            acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(w1_ps, x1));
        }

        __m256 total_acc = _mm256_add_ps(acc0, acc1);
        
        /* Horizontal sum of 8 floats in total_acc */
        __m128 lo128 = _mm256_castps256_ps128(total_acc);
        __m128 hi128 = _mm256_extractf128_ps(total_acc, 1);
        __m128 sum128 = _mm_add_ps(lo128, hi128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        
        float dot_sum = _mm_cvtss_f32(sum128);
        y[j] = (dot_sum * scale) + (b ? b[j] : 0.0f);
    }
#elif defined(__SSE2__)
    for (int j = 0; j < M; j++) {
        const int8_t* w_row = W + (j * K);
        __m128 acc0 = _mm_setzero_ps();
        __m128 acc1 = _mm_setzero_ps();

        for (int i = 0; i < K; i += 8) {
            float w_f[8];
            for (int k = 0; k < 8; k++) w_f[k] = (float)w_row[i + k];
            __m128 w0 = _mm_loadu_ps(w_f);
            __m128 w1 = _mm_loadu_ps(w_f + 4);
            __m128 x0 = _mm_loadu_ps(x + i);
            __m128 x1 = _mm_loadu_ps(x + i + 4);
            acc0 = _mm_add_ps(acc0, _mm_mul_ps(w0, x0));
            acc1 = _mm_add_ps(acc1, _mm_mul_ps(w1, x1));
        }
        __m128 sum128 = _mm_add_ps(acc0, acc1);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        float dot_sum = _mm_cvtss_f32(sum128);
        y[j] = (dot_sum * scale) + (b ? b[j] : 0.0f);
    }
#else
    for (int j = 0; j < M; j++) {
        const int8_t* w_row = W + (j * K);
        float sum = 0.0f;
        for (int i = 0; i < K; i++) {
            sum += (float)w_row[i] * x[i];
        }
        y[j] = (sum * scale) + (b ? b[j] : 0.0f);
    }
#endif
}

/* LayerNorm: out = (x - mean) / sqrt(var + eps) * weight + bias */
static void layer_norm(const float* x, const float* weight, const float* bias, float* out, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) sum += x[i];
    float mean = sum / (float)dim;

    float sq_diff_sum = 0.0f;
    for (int i = 0; i < dim; i++) {
        float diff = x[i] - mean;
        sq_diff_sum += diff * diff;
    }
    float inv_std = 1.0f / sqrtf((sq_diff_sum / (float)dim) + 1e-5f);

    for (int i = 0; i < dim; i++) {
        out[i] = ((x[i] - mean) * inv_std) * weight[i] + bias[i];
    }
}

/* Haar DWT Forward & Inverse Spectral Pass */
static void spectral_forward(const float* in, const wrai_spectral_t* spec, float* out, int dim, int levels) {
    float cur[WRAI_HIDDEN_DIM];
    memcpy(cur, in, dim * sizeof(float));

    float details_buf[WRAI_WAVELET_LEVELS][WRAI_HIDDEN_DIM];
    int current_len = dim;

    /* 1. Forward Haar DWT Decomposition */
    for (int l = 0; l < levels; l++) {
        int half = current_len >> 1;
        float approx[WRAI_HIDDEN_DIM >> 1];
        float detail[WRAI_HIDDEN_DIM >> 1];

        for (int i = 0; i < half; i++) {
            float even = cur[2 * i];
            float odd  = cur[2 * i + 1];
            approx[i] = (even + odd) * INV_SQRT2;
            detail[i] = (even - odd) * INV_SQRT2;
        }

        /* Modulate detail with learned gain */
        const float* d_gain = spec->detail_gains[l];
        for (int i = 0; i < half; i++) {
            details_buf[l][i] = detail[i] * d_gain[i];
        }

        memcpy(cur, approx, half * sizeof(float));
        current_len = half;
    }

    /* 2. Modulate coarsest approx with learned gain */
    int approx_len = current_len;
    for (int i = 0; i < approx_len; i++) {
        cur[i] = cur[i] * spec->approx_gain[i];
    }

    /* 3. Inverse Haar DWT Reconstruction */
    for (int l = levels - 1; l >= 0; l--) {
        int half = current_len;
        float reconstructed[WRAI_HIDDEN_DIM];
        const float* detail = details_buf[l];

        for (int i = 0; i < half; i++) {
            float approx_val = cur[i];
            float detail_val = detail[i];
            reconstructed[2 * i]     = (approx_val + detail_val) * INV_SQRT2;
            reconstructed[2 * i + 1] = (approx_val - detail_val) * INV_SQRT2;
        }

        current_len <<= 1;
        memcpy(cur, reconstructed, current_len * sizeof(float));
    }

    /* 4. Adaptive Highway Gating: out = cur * sigmoid(gate) + in * (1 - sigmoid(gate)) */
    for (int i = 0; i < dim; i++) {
        float g = fast_sigmoid(spec->gate[i]);
        out[i] = cur[i] * g + in[i] * (1.0f - g);
    }
}

/* GRU Layer Forward Step (Zero KV-Cache) */
static void gru_layer_step(
    const wrai_gru_layer_t* layer,
    const float* in_x,
    float* h_state,
    float* out_x,
    int dim
) {
    float x_norm[WRAI_HIDDEN_DIM];
    layer_norm(in_x, layer->ln_weight, layer->ln_bias, x_norm, dim);

    /* Compute gates_ih = W_ih * x_norm + b_ih (3072 floats) */
    float gates_ih[3 * WRAI_HIDDEN_DIM];
    gemv_int8_float(layer->w_ih, x_norm, layer->b_ih, layer->scale_w_ih, gates_ih, 3 * dim, dim);

    /* Compute gates_hh = W_hh * h_prev + b_hh (3072 floats) */
    float gates_hh[3 * WRAI_HIDDEN_DIM];
    gemv_int8_float(layer->w_hh, h_state, layer->b_hh, layer->scale_w_hh, gates_hh, 3 * dim, dim);

    /* Reset gate r, Update gate z, Candidate gate n */
    for (int i = 0; i < dim; i++) {
        float r_ih = gates_ih[i];
        float z_ih = gates_ih[dim + i];
        float n_ih = gates_ih[2 * dim + i];

        float r_hh = gates_hh[i];
        float z_hh = gates_hh[dim + i];
        float n_hh = gates_hh[2 * dim + i];

        float r = fast_sigmoid(r_ih + r_hh);
        float z = fast_sigmoid(z_ih + z_hh);
        float n = tanhf(n_ih + r * n_hh);

        float prev_h = h_state[i];
        float new_h = (1.0f - z) * n + z * prev_h;
        
        h_state[i] = new_h;
        out_x[i] = in_x[i] + new_h; /* Residual Connection */
    }
}

bool wrai_load_model_binary(const char* bin_path, wrai_model_t* model) {
    if (!bin_path || !model) return false;
    memset(model, 0, sizeof(wrai_model_t));

    FILE* f = fopen(bin_path, "rb");
    if (!f) {
        printf("[ERROR] Failed to open binary model: %s\n", bin_path);
        return false;
    }

    /* Read Header (32 bytes) */
    uint32_t header[8];
    if (fread(header, sizeof(uint32_t), 8, f) != 8) {
        printf("[ERROR] Failed to read header\n");
        fclose(f);
        return false;
    }

    model->magic          = header[0];
    model->version        = header[1];
    model->vocab_size     = header[2];
    model->hidden_dim     = header[3];
    model->num_layers     = header[4];
    model->wavelet_levels = header[5];
    model->epoch          = header[6];
    memcpy(&model->best_val_loss, &header[7], sizeof(float));

    if (model->magic != WRAI_MAGIC_HEADER) {
        printf("[ERROR] Invalid Magic Header: 0x%08X (Expected 0x%08X)\n", model->magic, WRAI_MAGIC_HEADER);
        fclose(f);
        return false;
    }

    /* Read Vocab Mapping Table */
    if (fread(model->teacher_map, sizeof(int32_t), WRAI_VOCAB_SIZE, f) != WRAI_VOCAB_SIZE) {
        printf("[ERROR] Failed to read teacher_map\n");
        fclose(f);
        return false;
    }

    /* Read Embedding Scale & Weights */
    if (fread(&model->scale_embed, sizeof(float), 1, f) != 1) {
        printf("[ERROR] Failed to read scale_embed\n");
        fclose(f);
        return false;
    }

    size_t embed_size = (size_t)WRAI_VOCAB_SIZE * WRAI_HIDDEN_DIM;
    model->embed_weights = (int8_t*)malloc(embed_size);
    if (!model->embed_weights || fread(model->embed_weights, 1, embed_size, f) != embed_size) {
        printf("[ERROR] Failed to read embed_weights\n");
        fclose(f);
        return false;
    }

    /* Read Positional Encoding */
    if (fread(model->pe, sizeof(float), WRAI_MAX_SEQ_LEN * WRAI_HIDDEN_DIM, f) != (WRAI_MAX_SEQ_LEN * WRAI_HIDDEN_DIM)) {
        printf("[ERROR] Failed to read PE\n");
        fclose(f);
        return false;
    }

    /* Read Spectral Layers (1 and 2) */
    wrai_spectral_t* specs[2] = { &model->spectral1, &model->spectral2 };
    for (int s = 0; s < 2; s++) {
        if (fread(specs[s]->gate, sizeof(float), WRAI_HIDDEN_DIM, f) != WRAI_HIDDEN_DIM) {
            printf("[ERROR] Failed to read spec[%d] gate\n", s);
            fclose(f);
            return false;
        }
        int approx_dim = WRAI_HIDDEN_DIM >> WRAI_WAVELET_LEVELS;
        if (fread(specs[s]->approx_gain, sizeof(float), approx_dim, f) != approx_dim) {
            printf("[ERROR] Failed to read spec[%d] approx_gain\n", s);
            fclose(f);
            return false;
        }
        for (int l = 0; l < WRAI_WAVELET_LEVELS; l++) {
            int d_dim = WRAI_HIDDEN_DIM >> (l + 1);
            specs[s]->detail_gains[l] = (float*)malloc(d_dim * sizeof(float));
            if (fread(specs[s]->detail_gains[l], sizeof(float), d_dim, f) != d_dim) {
                printf("[ERROR] Failed to read spec[%d] detail_gains[%d]\n", s, l);
                fclose(f);
                return false;
            }
        }
    }

    /* Read 12 ResGRU Layers */
    for (int l = 0; l < WRAI_NUM_LAYERS; l++) {
        wrai_gru_layer_t* lay = &model->layers[l];
        if (fread(lay->ln_weight, sizeof(float), WRAI_HIDDEN_DIM, f) != WRAI_HIDDEN_DIM ||
            fread(lay->ln_bias, sizeof(float), WRAI_HIDDEN_DIM, f) != WRAI_HIDDEN_DIM) {
            printf("[ERROR] Failed to read layer[%d] LN\n", l);
            fclose(f);
            return false;
        }

        size_t gru_w_size = 3 * WRAI_HIDDEN_DIM * WRAI_HIDDEN_DIM;
        
        /* W_ih */
        if (fread(&lay->scale_w_ih, sizeof(float), 1, f) != 1) { printf("[ERROR] Failed to read layer[%d] scale_w_ih\n", l); fclose(f); return false; }
        lay->w_ih = (int8_t*)malloc(gru_w_size);
        if (fread(lay->w_ih, 1, gru_w_size, f) != gru_w_size) { printf("[ERROR] Failed to read layer[%d] w_ih\n", l); fclose(f); return false; }

        /* W_hh */
        if (fread(&lay->scale_w_hh, sizeof(float), 1, f) != 1) { printf("[ERROR] Failed to read layer[%d] scale_w_hh\n", l); fclose(f); return false; }
        lay->w_hh = (int8_t*)malloc(gru_w_size);
        if (fread(lay->w_hh, 1, gru_w_size, f) != gru_w_size) { printf("[ERROR] Failed to read layer[%d] w_hh\n", l); fclose(f); return false; }

        /* Biases */
        if (fread(lay->b_ih, sizeof(float), 3 * WRAI_HIDDEN_DIM, f) != 3 * WRAI_HIDDEN_DIM ||
            fread(lay->b_hh, sizeof(float), 3 * WRAI_HIDDEN_DIM, f) != 3 * WRAI_HIDDEN_DIM) {
            printf("[ERROR] Failed to read layer[%d] biases\n", l);
            fclose(f);
            return false;
        }
    }

    /* Final LayerNorm */
    if (fread(model->ln_final_weight, sizeof(float), WRAI_HIDDEN_DIM, f) != WRAI_HIDDEN_DIM ||
        fread(model->ln_final_bias, sizeof(float), WRAI_HIDDEN_DIM, f) != WRAI_HIDDEN_DIM) {
        printf("[ERROR] Failed to read ln_final\n");
        fclose(f);
        return false;
    }

    fclose(f);
    return true;
}

void wrai_free_model(wrai_model_t* model) {
    if (!model) return;
    if (model->embed_weights) free(model->embed_weights);
    for (int s = 0; s < 2; s++) {
        wrai_spectral_t* spec = (s == 0) ? &model->spectral1 : &model->spectral2;
        for (int l = 0; l < WRAI_WAVELET_LEVELS; l++) {
            if (spec->detail_gains[l]) free(spec->detail_gains[l]);
        }
    }
    for (int l = 0; l < WRAI_NUM_LAYERS; l++) {
        if (model->layers[l].w_ih) free(model->layers[l].w_ih);
        if (model->layers[l].w_hh) free(model->layers[l].w_hh);
    }
}

void wrai_state_reset(wrai_inference_state_t* state) {
    if (!state) return;
    memset(state, 0, sizeof(wrai_inference_state_t));
}

void wrai_forward_step(
    const wrai_model_t* model,
    wrai_inference_state_t* state,
    uint32_t token_id,
    float* out_logits
) {
    if (token_id >= WRAI_VOCAB_SIZE) token_id = 0;
    uint32_t pos = state->current_seq_len;
    if (pos >= WRAI_MAX_SEQ_LEN) pos = WRAI_MAX_SEQ_LEN - 1;

    /* 1. Embedding + Scaling (sqrt(1024) = 32.0) + Positional Encoding */
    const int8_t* emb_row = model->embed_weights + (token_id * WRAI_HIDDEN_DIM);
    const float* pe_row = model->pe + (pos * WRAI_HIDDEN_DIM);
    float embed_scale_factor = model->scale_embed * 32.0f;

    for (int i = 0; i < WRAI_HIDDEN_DIM; i++) {
        state->ping_buffer[i] = ((float)emb_row[i] * embed_scale_factor) + pe_row[i];
    }

    /* 2. Spectral Layer 1 */
    spectral_forward(state->ping_buffer, &model->spectral1, state->pong_buffer, WRAI_HIDDEN_DIM, WRAI_WAVELET_LEVELS);

    /* 3. 12 ResGRU Layers (8 KB SRAM Ping-Pong Buffer) */
    float* in_buf = state->pong_buffer;
    float* out_buf = state->ping_buffer;

    for (int l = 0; l < WRAI_NUM_LAYERS; l++) {
        gru_layer_step(&model->layers[l], in_buf, state->h_states[l], out_buf, WRAI_HIDDEN_DIM);
        
        /* Swap ping and pong */
        float* tmp = in_buf;
        in_buf = out_buf;
        out_buf = tmp;
    }

    /* 4. Spectral Layer 2 */
    spectral_forward(in_buf, &model->spectral2, state->spectral_buf, WRAI_HIDDEN_DIM, WRAI_WAVELET_LEVELS);

    /* 5. Final LayerNorm: ln_final(spectral2_out + gru_out) */
    float final_hidden[WRAI_HIDDEN_DIM];
    for (int i = 0; i < WRAI_HIDDEN_DIM; i++) {
        final_hidden[i] = state->spectral_buf[i] + in_buf[i];
    }
    float norm_hidden[WRAI_HIDDEN_DIM];
    layer_norm(final_hidden, model->ln_final_weight, model->ln_final_bias, norm_hidden, WRAI_HIDDEN_DIM);

    /* 6. Output Logit Projection (Tied Embedding Matrix: 32000 x 1024 Int8 GEMV) */
    if (out_logits) {
        gemv_int8_float(
            model->embed_weights,
            norm_hidden,
            NULL,
            model->scale_embed,
            out_logits,
            WRAI_VOCAB_SIZE,
            WRAI_HIDDEN_DIM
        );
    }

    state->current_seq_len++;
}

uint32_t wrai_sample_top_k(
    float* logits,
    uint32_t vocab_size,
    float temperature,
    uint32_t top_k,
    const uint32_t* history_tokens,
    uint32_t history_len,
    float repetition_penalty
) {
    if (temperature <= 0.001f) temperature = 0.001f;

    /* 1. Repetition Penalty */
    if (repetition_penalty > 1.001f && history_tokens && history_len > 0) {
        for (uint32_t i = 0; i < history_len; i++) {
            uint32_t tok = history_tokens[i];
            if (tok < vocab_size) {
                if (logits[tok] < 0.0f) {
                    logits[tok] *= repetition_penalty;
                } else {
                    logits[tok] /= repetition_penalty;
                }
            }
        }
    }

    /* 2. Temperature scaling */
    for (uint32_t i = 0; i < vocab_size; i++) {
        logits[i] /= temperature;
    }

    /* 3. Top-K Selection */
    if (top_k == 0 || top_k > vocab_size) top_k = vocab_size;
    if (top_k > 100) top_k = 100; /* Fast practical top-K */

    typedef struct { float logit; uint32_t id; } candidate_t;
    candidate_t best_cands[100];
    for (uint32_t k = 0; k < top_k; k++) {
        best_cands[k].logit = -1e30f;
        best_cands[k].id = 0;
    }

    for (uint32_t v = 0; v < vocab_size; v++) {
        float val = logits[v];
        if (val > best_cands[top_k - 1].logit) {
            /* Insertion sort into best_cands */
            int pos = top_k - 1;
            while (pos > 0 && val > best_cands[pos - 1].logit) {
                best_cands[pos] = best_cands[pos - 1];
                pos--;
            }
            best_cands[pos].logit = val;
            best_cands[pos].id = v;
        }
    }

    /* 4. Softmax over Top-K */
    float max_l = best_cands[0].logit;
    float sum_exp = 0.0f;
    float probs[100];
    for (uint32_t k = 0; k < top_k; k++) {
        probs[k] = expf(best_cands[k].logit - max_l);
        sum_exp += probs[k];
    }

    /* 5. Random Sampling */
    float r = ((float)rand() / (float)RAND_MAX) * sum_exp;
    float cum = 0.0f;
    for (uint32_t k = 0; k < top_k; k++) {
        cum += probs[k];
        if (r <= cum) {
            return best_cands[k].id;
        }
    }

    return best_cands[0].id;
}
