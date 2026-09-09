/**
 * @file wrai_v16_engine.c
 * @brief Ultra-Fast AVX 1.0 Native Inference Engine for WRAI v16 (1.7B)
 *        Memory-Mapped, O(1) Constant Recurrent State, Zero KV-Cache.
 */

#include "wrai_v16_engine.h"
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

/* Math utilities */
static inline float fast_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static inline float fast_silu(float x) {
    return x / (1.0f + expf(-x));
}

/* AVX 1.0 Vectorized RMSNorm: y = (x / sqrt(mean(x^2) + eps)) * weight */
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

    float rsqrt_val = 1.0f / sqrtf((sum_sq / (float)dim) + 1e-6f);

#if defined(__AVX__)
    __m256 v_scale = _mm256_set1_ps(rsqrt_val);
    i = 0;
    for (; i <= dim - 8; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vw = _mm256_loadu_ps(weight + i);
        __m256 vy = _mm256_mul_ps(_mm256_mul_ps(vx, v_scale), vw);
        _mm256_storeu_ps(y + i, vy);
    }
    for (; i < dim; i++) y[i] = x[i] * rsqrt_val * weight[i];
#else
    for (int i = 0; i < dim; i++) y[i] = x[i] * rsqrt_val * weight[i];
#endif
}

/* AVX 1.0 / SSE4.1 Multi-threaded INT8 Matrix-Vector GEMV: y = (W * x) .* scales (32-unrolled) */
static void gemv_int8_rowwise(
    const float* scales,
    const int8_t* W,
    const float* x,
    float* y,
    int M,
    int K
) {
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(M >= 2048)
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

/* ============================================================================
 * MODEL LOADER (Memory-Mapped, 0 Heap Copy)
 * ============================================================================ */
bool wrai_v16_load_model(const char* bin_path, wrai_v16_model_t* model) {
    if (!bin_path || !model) return false;
    memset(model, 0, sizeof(wrai_v16_model_t));

#ifdef _WIN32
    model->hFile = CreateFileA(bin_path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (model->hFile == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "[ERROR] Cannot open model binary: %s\n", bin_path);
        return false;
    }

    LARGE_INTEGER fsize;
    if (!GetFileSizeEx(model->hFile, &fsize)) {
        CloseHandle(model->hFile);
        return false;
    }
    model->file_size = (size_t)fsize.QuadPart;

    model->hMapping = CreateFileMappingA(model->hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!model->hMapping) {
        CloseHandle(model->hFile);
        return false;
    }

    model->mmap_base = (const uint8_t*)MapViewOfFile(model->hMapping, FILE_MAP_READ, 0, 0, 0);
    if (!model->mmap_base) {
        CloseHandle(model->hMapping);
        CloseHandle(model->hFile);
        return false;
    }
#else
    FILE* fp = fopen(bin_path, "rb");
    if (!fp) return false;
    fseek(fp, 0, SEEK_END);
    model->file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    model->mmap_base = (const uint8_t*)malloc(model->file_size);
    if (fread((void*)model->mmap_base, 1, model->file_size, fp) != model->file_size) {
        fclose(fp);
        return false;
    }
    fclose(fp);
#endif

    /* Parse 64-byte Header */
    memcpy(&model->header, model->mmap_base, sizeof(wrai_v16_header_t));
    if (model->header.magic != WRAI_V16_MAGIC) {
        fprintf(stderr, "[ERROR] Invalid Magic Header (0x%08X vs 0x%08X)\n", model->header.magic, WRAI_V16_MAGIC);
        wrai_v16_free_model(model);
        return false;
    }

    const uint8_t* ptr = model->mmap_base + sizeof(wrai_v16_header_t);

    /* 1. Positional Encoding (512 x 2048 floats) */
    model->pe = (const float*)ptr;
    ptr += WRAI_V16_MAX_SEQ_LEN * WRAI_V16_HIDDEN_DIM * sizeof(float);

    /* 2. Dual Wavelet Spectral Stabilizers */
    model->spec1_gw = (const float*)ptr; ptr += WRAI_V16_HIDDEN_DIM * sizeof(float);
    model->spec1_gb = (const float*)ptr; ptr += WRAI_V16_HIDDEN_DIM * sizeof(float);
    model->spec1_dg = (const float*)ptr; ptr += WRAI_V16_WAVELET_LEVELS * sizeof(float);
    model->spec1_ag = *(const float*)ptr; ptr += sizeof(float);

    model->spec2_gw = (const float*)ptr; ptr += WRAI_V16_HIDDEN_DIM * sizeof(float);
    model->spec2_gb = (const float*)ptr; ptr += WRAI_V16_HIDDEN_DIM * sizeof(float);
    model->spec2_dg = (const float*)ptr; ptr += WRAI_V16_WAVELET_LEVELS * sizeof(float);
    model->spec2_ag = *(const float*)ptr; ptr += sizeof(float);

    /* 3. Final RMSNorm */
    model->ln_final = (const float*)ptr; ptr += WRAI_V16_HIDDEN_DIM * sizeof(float);

    /* 4. Token Embeddings (151936 x 2048, INT8 row-wise) */
    model->embed_scales = (const float*)ptr; ptr += (size_t)WRAI_V16_VOCAB_SIZE * sizeof(float);
    model->embed_data = (const int8_t*)ptr; ptr += (size_t)WRAI_V16_VOCAB_SIZE * WRAI_V16_HIDDEN_DIM * sizeof(int8_t);

    /* 5. 28 Layers */
    for (int l = 0; l < WRAI_V16_NUM_LAYERS; l++) {
        wrai_v16_layer_weights_t* lw = &model->layers[l];

        lw->rms_ret = (const float*)ptr; ptr += WRAI_V16_HIDDEN_DIM * sizeof(float);

        /* Retention: w_q, w_k, w_v, w_out */
        lw->w_q_scales = (const float*)ptr; ptr += WRAI_V16_HIDDEN_DIM * sizeof(float);
        lw->w_q_data   = (const int8_t*)ptr; ptr += (size_t)WRAI_V16_HIDDEN_DIM * WRAI_V16_HIDDEN_DIM;

        lw->w_k_scales = (const float*)ptr; ptr += WRAI_V16_HIDDEN_DIM * sizeof(float);
        lw->w_k_data   = (const int8_t*)ptr; ptr += (size_t)WRAI_V16_HIDDEN_DIM * WRAI_V16_HIDDEN_DIM;

        lw->w_v_scales = (const float*)ptr; ptr += WRAI_V16_HIDDEN_DIM * sizeof(float);
        lw->w_v_data   = (const int8_t*)ptr; ptr += (size_t)WRAI_V16_HIDDEN_DIM * WRAI_V16_HIDDEN_DIM;

        lw->w_out_scales = (const float*)ptr; ptr += WRAI_V16_HIDDEN_DIM * sizeof(float);
        lw->w_out_data   = (const int8_t*)ptr; ptr += (size_t)WRAI_V16_HIDDEN_DIM * WRAI_V16_HIDDEN_DIM;

        /* Retention decay logits + group norm */
        lw->decay_logit = (const float*)ptr; ptr += WRAI_V16_NUM_HEADS * sizeof(float);
        lw->group_norm_w = (const float*)ptr; ptr += WRAI_V16_HIDDEN_DIM * sizeof(float);
        lw->group_norm_b = (const float*)ptr; ptr += WRAI_V16_HIDDEN_DIM * sizeof(float);

        /* FFN */
        lw->rms_ffn = (const float*)ptr; ptr += WRAI_V16_HIDDEN_DIM * sizeof(float);

        lw->w_gate_scales = (const float*)ptr; ptr += WRAI_V16_FFN_DIM * sizeof(float);
        lw->w_gate_data   = (const int8_t*)ptr; ptr += (size_t)WRAI_V16_FFN_DIM * WRAI_V16_HIDDEN_DIM;

        lw->w_up_scales = (const float*)ptr; ptr += WRAI_V16_FFN_DIM * sizeof(float);
        lw->w_up_data   = (const int8_t*)ptr; ptr += (size_t)WRAI_V16_FFN_DIM * WRAI_V16_HIDDEN_DIM;

        lw->w_down_scales = (const float*)ptr; ptr += WRAI_V16_HIDDEN_DIM * sizeof(float);
        lw->w_down_data   = (const int8_t*)ptr; ptr += (size_t)WRAI_V16_HIDDEN_DIM * WRAI_V16_FFN_DIM;
    }

    return true;
}

void wrai_v16_free_model(wrai_v16_model_t* model) {
    if (!model) return;
#ifdef _WIN32
    if (model->mmap_base) UnmapViewOfFile(model->mmap_base);
    if (model->hMapping) CloseHandle(model->hMapping);
    if (model->hFile != INVALID_HANDLE_VALUE) CloseHandle(model->hFile);
#else
    if (model->mmap_base) free((void*)model->mmap_base);
#endif
    memset(model, 0, sizeof(wrai_v16_model_t));
}

/* ============================================================================
 * RECURRENT STATE MANAGEMENT (O(1) Memory ~28 MB)
 * ============================================================================ */
bool wrai_v16_state_init(wrai_v16_state_t* state) {
    if (!state) return false;
    size_t total_floats = (size_t)WRAI_V16_NUM_LAYERS * WRAI_V16_NUM_HEADS * WRAI_V16_HEAD_DIM * WRAI_V16_HEAD_DIM;
    state->state_buffer = (float*)calloc(total_floats, sizeof(float));
    state->current_pos = 0;
    return (state->state_buffer != NULL);
}

void wrai_v16_state_reset(wrai_v16_state_t* state) {
    if (!state || !state->state_buffer) return;
    size_t total_floats = (size_t)WRAI_V16_NUM_LAYERS * WRAI_V16_NUM_HEADS * WRAI_V16_HEAD_DIM * WRAI_V16_HEAD_DIM;
    memset(state->state_buffer, 0, total_floats * sizeof(float));
    state->current_pos = 0;
}

void wrai_v16_state_free(wrai_v16_state_t* state) {
    if (!state) return;
    if (state->state_buffer) free(state->state_buffer);
    state->state_buffer = NULL;
    state->current_pos = 0;
}

/* ============================================================================
 * TOKENIZER (Vocab Binary Loader + Greedy Longest Match)
 * ============================================================================ */
typedef struct {
    uint32_t count;
    uint32_t* indices;
} byte_bucket_t;

bool wrai_v16_load_tokenizer(const char* vocab_bin_path, wrai_v16_tokenizer_t* tok) {
    if (!vocab_bin_path || !tok) return false;
    memset(tok, 0, sizeof(wrai_v16_tokenizer_t));

    FILE* f = fopen(vocab_bin_path, "rb");
    if (!f) return false;

    if (fread(&tok->num_tokens, sizeof(uint32_t), 1, f) != 1) {
        fclose(f);
        return false;
    }

    tok->token_strings = (char**)calloc(tok->num_tokens, sizeof(char*));
    tok->token_lens = (uint16_t*)calloc(tok->num_tokens, sizeof(uint16_t));
    tok->im_start_id = 151644;
    tok->im_end_id   = 151645;
    tok->eos_id      = 151643;

    byte_bucket_t* buckets = (byte_bucket_t*)calloc(256, sizeof(byte_bucket_t));

    for (uint32_t i = 0; i < tok->num_tokens; i++) {
        uint16_t len = 0;
        if (fread(&len, sizeof(uint16_t), 1, f) != 1) break;
        tok->token_lens[i] = len;
        tok->token_strings[i] = (char*)malloc(len + 1);
        if (fread(tok->token_strings[i], 1, len, f) != len) break;
        tok->token_strings[i][len] = '\0';

        if (strcmp(tok->token_strings[i], "<|im_start|>") == 0) tok->im_start_id = i;
        if (strcmp(tok->token_strings[i], "<|im_end|>") == 0)   tok->im_end_id = i;
        if (strcmp(tok->token_strings[i], "<|endoftext|>") == 0) tok->eos_id = i;

        if (len > 0) {
            uint8_t b = (uint8_t)tok->token_strings[i][0];
            buckets[b].count++;
        }
    }
    fclose(f);

    /* Allocate and sort buckets by token length descending */
    for (int b = 0; b < 256; b++) {
        if (buckets[b].count > 0) {
            buckets[b].indices = (uint32_t*)malloc(buckets[b].count * sizeof(uint32_t));
            uint32_t idx = 0;
            for (uint32_t i = 0; i < tok->num_tokens; i++) {
                if (tok->token_lens[i] > 0 && (uint8_t)tok->token_strings[i][0] == (uint8_t)b) {
                    buckets[b].indices[idx++] = i;
                }
            }
            /* Sort descending */
            for (uint32_t x = 0; x < buckets[b].count; x++) {
                for (uint32_t y = x + 1; y < buckets[b].count; y++) {
                    uint32_t idx_x = buckets[b].indices[x];
                    uint32_t idx_y = buckets[b].indices[y];
                    if (tok->token_lens[idx_y] > tok->token_lens[idx_x]) {
                        buckets[b].indices[x] = idx_y;
                        buckets[b].indices[y] = idx_x;
                    }
                }
            }
        }
    }
    tok->lookup_index = (void*)buckets;
    return true;
}

void wrai_v16_free_tokenizer(wrai_v16_tokenizer_t* tok) {
    if (!tok) return;
    if (tok->token_strings) {
        for (uint32_t i = 0; i < tok->num_tokens; i++) {
            if (tok->token_strings[i]) free(tok->token_strings[i]);
        }
        free(tok->token_strings);
    }
    if (tok->token_lens) free(tok->token_lens);
    if (tok->lookup_index) {
        byte_bucket_t* buckets = (byte_bucket_t*)tok->lookup_index;
        for (int b = 0; b < 256; b++) {
            if (buckets[b].indices) free(buckets[b].indices);
        }
        free(buckets);
    }
    memset(tok, 0, sizeof(wrai_v16_tokenizer_t));
}

int wrai_v16_tokenize(const wrai_v16_tokenizer_t* tok, const char* text, uint32_t* tokens_out, int max_tokens) {
    if (!tok || !text || !tokens_out || max_tokens <= 0) return 0;
    int num_out = 0;
    size_t text_len = strlen(text);
    byte_bucket_t* buckets = (byte_bucket_t*)tok->lookup_index;

    /* Pre-convert text into BPE byte representation (space -> \xC4\xA0, \n -> \xC4\x8A) */
    size_t bpe_cap = text_len * 2 + 256;
    char* bpe_text = (char*)malloc(bpe_cap);
    size_t bpe_len = 0;

    for (size_t i = 0; i < text_len; i++) {
        /* Preserve special tags exactly */
        if (strncmp(text + i, "<|im_start|>", 12) == 0) {
            memcpy(bpe_text + bpe_len, "<|im_start|>", 12);
            bpe_len += 12;
            i += 11;
            continue;
        }
        if (strncmp(text + i, "<|im_end|>", 10) == 0) {
            memcpy(bpe_text + bpe_len, "<|im_end|>", 10);
            bpe_len += 10;
            i += 9;
            continue;
        }

        uint8_t c = (uint8_t)text[i];
        if (c == ' ') {
            bpe_text[bpe_len++] = (char)0xC4;
            bpe_text[bpe_len++] = (char)0xA0;
        } else if (c == '\n') {
            bpe_text[bpe_len++] = (char)0xC4;
            bpe_text[bpe_len++] = (char)0x8A;
        } else if (c == '\r') {
            bpe_text[bpe_len++] = (char)0xC4;
            bpe_text[bpe_len++] = (char)0x8D;
        } else if (c == '\t') {
            bpe_text[bpe_len++] = (char)0xC4;
            bpe_text[bpe_len++] = (char)0x89;
        } else {
            bpe_text[bpe_len++] = (char)c;
        }
    }
    bpe_text[bpe_len] = '\0';

    size_t pos = 0;
    while (pos < bpe_len && num_out < max_tokens) {
        if (strncmp(bpe_text + pos, "<|im_start|>", 12) == 0) {
            tokens_out[num_out++] = tok->im_start_id;
            pos += 12;
            continue;
        }
        if (strncmp(bpe_text + pos, "<|im_end|>", 10) == 0) {
            tokens_out[num_out++] = tok->im_end_id;
            pos += 10;
            continue;
        }

        uint8_t first_byte = (uint8_t)bpe_text[pos];
        bool matched = false;
        if (buckets && buckets[first_byte].count > 0) {
            for (uint32_t k = 0; k < buckets[first_byte].count; k++) {
                uint32_t tid = buckets[first_byte].indices[k];
                uint16_t tlen = tok->token_lens[tid];
                if (pos + tlen <= bpe_len && memcmp(bpe_text + pos, tok->token_strings[tid], tlen) == 0) {
                    tokens_out[num_out++] = tid;
                    pos += tlen;
                    matched = true;
                    break;
                }
            }
        }

        if (!matched) {
            tokens_out[num_out++] = (uint32_t)first_byte;
            pos++;
        }
    }

    free(bpe_text);
    return num_out;
}

const char* wrai_v16_decode_token(const wrai_v16_tokenizer_t* tok, uint32_t token_id) {
    if (!tok || token_id >= tok->num_tokens || !tok->token_strings) return "";
    const char* raw = tok->token_strings[token_id];
    static char decoded_buf[512];

    size_t in_len = strlen(raw);
    size_t out_len = 0;

    for (size_t i = 0; i < in_len && out_len < 510; i++) {
        if ((uint8_t)raw[i] == 0xC4 && i + 1 < in_len) {
            uint8_t next_b = (uint8_t)raw[i + 1];
            if (next_b == 0xA0) {
                decoded_buf[out_len++] = ' ';
                i++;
                continue;
            } else if (next_b == 0x8A) {
                decoded_buf[out_len++] = '\n';
                i++;
                continue;
            } else if (next_b == 0x8D) {
                decoded_buf[out_len++] = '\r';
                i++;
                continue;
            } else if (next_b == 0x89) {
                decoded_buf[out_len++] = '\t';
                i++;
                continue;
            }
        }
        decoded_buf[out_len++] = raw[i];
    }
    decoded_buf[out_len] = '\0';
    return decoded_buf;
}

/* ============================================================================
 * SINGLE TOKEN FORWARD STEP (O(1) RECURRENT INFERENCE)
 * ============================================================================ */
void wrai_v16_forward_step(
    const wrai_v16_model_t* model,
    wrai_v16_state_t* state,
    uint32_t token_id,
    float* logits_out,
    bool compute_logits
) {
    const int D = WRAI_V16_HIDDEN_DIM;      /* 2048 */
    const int F = WRAI_V16_FFN_DIM;         /* 6144 */
    const int H = WRAI_V16_NUM_HEADS;        /* 16 */
    const int HD = WRAI_V16_HEAD_DIM;        /* 128 */

    /* Buffers for layer activations */
    float x[WRAI_V16_HIDDEN_DIM];
    float x_norm[WRAI_V16_HIDDEN_DIM];
    float q[WRAI_V16_HIDDEN_DIM];
    float k[WRAI_V16_HIDDEN_DIM];
    float v[WRAI_V16_HIDDEN_DIM];
    float ret_out[WRAI_V16_HIDDEN_DIM];
    float proj_out[WRAI_V16_HIDDEN_DIM];

    float ffn_gate[WRAI_V16_FFN_DIM];
    float ffn_up[WRAI_V16_FFN_DIM];
    float ffn_act[WRAI_V16_FFN_DIM];

    /* 1. Extract Token Embedding (INT8 dequantize) */
    if (token_id >= (uint32_t)WRAI_V16_VOCAB_SIZE) token_id = 0;
    float emb_scale = model->embed_scales[token_id];
    const int8_t* emb_row = model->embed_data + ((size_t)token_id * D);

#if defined(__AVX__)
    __m256 v_scale = _mm256_set1_ps(emb_scale);
    for (int i = 0; i < D; i += 16) {
        __m128i r0 = _mm_loadl_epi64((const __m128i*)(emb_row + i));
        __m128 lo0 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(r0));
        __m128 hi0 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(r0, 4)));
        _mm256_storeu_ps(x + i, _mm256_mul_ps(_mm256_set_m128(hi0, lo0), v_scale));

        __m128i r1 = _mm_loadl_epi64((const __m128i*)(emb_row + i + 8));
        __m128 lo1 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(r1));
        __m128 hi1 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(r1, 4)));
        _mm256_storeu_ps(x + i + 8, _mm256_mul_ps(_mm256_set_m128(hi1, lo1), v_scale));
    }
#else
    for (int i = 0; i < D; i++) x[i] = (float)emb_row[i] * emb_scale;
#endif

    /* 2. Wavelet Spectral Stabilizer 1 (Layer 0) */
    for (int i = 0; i < D; i++) {
        float gate = fast_sigmoid(x[i] * model->spec1_gw[i] + model->spec1_gb[i]);
        x[i] = x[i] + (x[i] * gate);
    }

    /* 4. 28 Layers Loop */
    for (int l = 0; l < WRAI_V16_NUM_LAYERS; l++) {
        const wrai_v16_layer_weights_t* lw = &model->layers[l];
        float* layer_state = state->state_buffer + ((size_t)l * H * HD * HD);

        /* --- A. Retention Sub-layer --- */
        rms_norm(x, lw->rms_ret, x_norm, D);

        gemv_int8_rowwise(lw->w_q_scales, lw->w_q_data, x_norm, q, D, D);
        gemv_int8_rowwise(lw->w_k_scales, lw->w_k_data, x_norm, k, D, D);
        gemv_int8_rowwise(lw->w_v_scales, lw->w_v_data, x_norm, v, D, D);

        /* Recurrent update per head: S = gamma * S + k * v^T, o = q * S */
        for (int h = 0; h < H; h++) {
            float* head_s = layer_state + (h * HD * HD);
            float gamma = fast_sigmoid(lw->decay_logit[h]);
            const float* q_h = q + (h * HD);
            const float* k_h = k + (h * HD);
            const float* v_h = v + (h * HD);
            float* o_h = ret_out + (h * HD);

            memset(o_h, 0, HD * sizeof(float));

            /* Update state matrix (128 x 128) and compute output vector */
            for (int r = 0; r < HD; r++) {
                float kr = k_h[r];
                float qr = q_h[r];
                float* s_row = head_s + (r * HD);

#if defined(__AVX__)
                __m256 v_gamma = _mm256_set1_ps(gamma);
                __m256 v_kr = _mm256_set1_ps(kr);
                __m256 v_qr = _mm256_set1_ps(qr);

                for (int c = 0; c < HD; c += 8) {
                    __m256 vs = _mm256_loadu_ps(s_row + c);
                    __m256 vv = _mm256_loadu_ps(v_h + c);
                    vs = _mm256_add_ps(_mm256_mul_ps(vs, v_gamma), _mm256_mul_ps(v_kr, vv));
                    _mm256_storeu_ps(s_row + c, vs);

                    __m256 vo = _mm256_loadu_ps(o_h + c);
                    vo = _mm256_add_ps(vo, _mm256_mul_ps(v_qr, vs));
                    _mm256_storeu_ps(o_h + c, vo);
                }
#else
                for (int c = 0; c < HD; c++) {
                    s_row[c] = s_row[c] * gamma + kr * v_h[c];
                    o_h[c] += qr * s_row[c];
                }
#endif
            }

            /* Per-Token GroupNorm on o_h (128 elements) */
            float sum = 0.0f;
            for (int c = 0; c < HD; c++) sum += o_h[c];
            float mean = sum / (float)HD;

            float sum_sq = 0.0f;
            for (int c = 0; c < HD; c++) {
                float diff = o_h[c] - mean;
                sum_sq += diff * diff;
            }
            float inv_std = 1.0f / sqrtf((sum_sq / (float)HD) + 1e-5f);

            const float* gn_w = lw->group_norm_w + (h * HD);
            const float* gn_b = lw->group_norm_b + (h * HD);
            for (int c = 0; c < HD; c++) {
                o_h[c] = (o_h[c] - mean) * inv_std * gn_w[c] + gn_b[c];
            }
        }

        /* Project Retention output */
        gemv_int8_rowwise(lw->w_out_scales, lw->w_out_data, ret_out, proj_out, D, D);
        for (int i = 0; i < D; i++) x[i] += proj_out[i];

        /* --- B. SwiGLU FFN Sub-layer --- */
        rms_norm(x, lw->rms_ffn, x_norm, D);

        gemv_int8_rowwise(lw->w_gate_scales, lw->w_gate_data, x_norm, ffn_gate, F, D);
        gemv_int8_rowwise(lw->w_up_scales,   lw->w_up_data,   x_norm, ffn_up,   F, D);

        /* Activation: SiLU(gate) * up */
        for (int i = 0; i < F; i++) {
            ffn_act[i] = fast_silu(ffn_gate[i]) * ffn_up[i];
        }

        /* Project FFN output back to D */
        gemv_int8_rowwise(lw->w_down_scales, lw->w_down_data, ffn_act, proj_out, D, F);
        for (int i = 0; i < D; i++) x[i] += proj_out[i];

        /* Wavelet Spectral Stabilizer 2 at Layer 13 (Middle) */
        if (l == (WRAI_V16_NUM_LAYERS / 2) - 1) {
            for (int i = 0; i < D; i++) {
                float gate = fast_sigmoid(x[i] * model->spec2_gw[i] + model->spec2_gb[i]);
                x[i] = x[i] + (x[i] * gate);
            }
        }
    }

    /* 5. Final RMSNorm & Logits Projection (only if requested) */
    if (compute_logits && logits_out) {
        rms_norm(x, model->ln_final, x_norm, D);
        gemv_int8_rowwise(model->embed_scales, model->embed_data, x_norm, logits_out, WRAI_V16_VOCAB_SIZE, D);
    }

    state->current_pos++;
}

/* ============================================================================
 * SAMPLING ENGINE (Repetition Penalty, Temperature, Top-P, Greedy)
 * ============================================================================ */
typedef struct {
    uint32_t id;
    float val;
} token_score_t;

static int compare_token_scores(const void* a, const void* b) {
    float diff = ((const token_score_t*)b)->val - ((const token_score_t*)a)->val;
    return (diff > 0.0f) ? 1 : ((diff < 0.0f) ? -1 : 0);
}

uint32_t wrai_v16_sample_token(
    float* logits,
    int vocab_size,
    const uint32_t* history_tokens,
    int history_len,
    const wrai_v16_sample_params_t* params
) {
    /* 1. Repetition Penalty */
    if (params && params->repetition_penalty > 1.0f && history_tokens && history_len > 0) {
        float rep_pen = params->repetition_penalty;
        int start_idx = (history_len > 32) ? (history_len - 32) : 0;
        for (int i = start_idx; i < history_len; i++) {
            uint32_t tid = history_tokens[i];
            if (tid < (uint32_t)vocab_size) {
                if (logits[tid] > 0.0f) logits[tid] /= rep_pen;
                else logits[tid] *= rep_pen;
            }
        }
    }

    /* 2. Greedy if Temperature <= 1e-4 */
    if (!params || params->temperature <= 1e-4f) {
        uint32_t best_id = 0;
        float max_val = logits[0];
        for (int i = 1; i < vocab_size; i++) {
            if (logits[i] > max_val) {
                max_val = logits[i];
                best_id = i;
            }
        }
        return best_id;
    }

    /* 3. Scaled Logits with Temperature */
    float inv_temp = 1.0f / params->temperature;
    for (int i = 0; i < vocab_size; i++) logits[i] *= inv_temp;

    /* 4. Top-K Selection */
    int top_k = (params->top_k > 0 && params->top_k < 100) ? params->top_k : 40;
    token_score_t* candidates = (token_score_t*)malloc(top_k * sizeof(token_score_t));
    for (int i = 0; i < top_k; i++) {
        candidates[i].id = i;
        candidates[i].val = logits[i];
    }
    qsort(candidates, top_k, sizeof(token_score_t), compare_token_scores);

    for (int i = top_k; i < vocab_size; i++) {
        if (logits[i] > candidates[top_k - 1].val) {
            candidates[top_k - 1].id = i;
            candidates[top_k - 1].val = logits[i];
            /* Bubble up */
            for (int j = top_k - 1; j > 0 && candidates[j].val > candidates[j - 1].val; j--) {
                token_score_t tmp = candidates[j];
                candidates[j] = candidates[j - 1];
                candidates[j - 1] = tmp;
            }
        }
    }

    /* Softmax on top-k candidates */
    float max_log = candidates[0].val;
    float sum_exp = 0.0f;
    for (int i = 0; i < top_k; i++) {
        candidates[i].val = expf(candidates[i].val - max_log);
        sum_exp += candidates[i].val;
    }
    for (int i = 0; i < top_k; i++) candidates[i].val /= sum_exp;

    /* Top-P Truncation */
    float top_p = (params->top_p > 0.0f && params->top_p < 1.0f) ? params->top_p : 0.9f;
    float cumsum = 0.0f;
    int cutoff = top_k;
    for (int i = 0; i < top_k; i++) {
        cumsum += candidates[i].val;
        if (cumsum >= top_p) {
            cutoff = i + 1;
            break;
        }
    }

    /* Random Sample */
    float r = ((float)rand() / (float)RAND_MAX) * cumsum;
    float acc = 0.0f;
    uint32_t selected = candidates[0].id;
    for (int i = 0; i < cutoff; i++) {
        acc += candidates[i].val;
        if (r <= acc) {
            selected = candidates[i].id;
            break;
        }
    }

    free(candidates);
    return selected;
}
