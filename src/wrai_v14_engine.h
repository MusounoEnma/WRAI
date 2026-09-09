/**
 * @file wrai_v14_engine.h
 * @brief High-Performance Native C Inference Engine for WRAI v14.4 (108M Int8, 0% KV-Cache, AVX/SSE SIMD)
 */

#ifndef WRAI_V14_ENGINE_H
#define WRAI_V14_ENGINE_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WRAI_MAGIC_HEADER      0x57524149  /* ASCII 'WRAI' */
#define WRAI_VERSION           144         /* v14.4 */
#define WRAI_VOCAB_SIZE        32000
#define WRAI_HIDDEN_DIM        1024
#define WRAI_NUM_LAYERS        12
#define WRAI_WAVELET_LEVELS    4
#define WRAI_MAX_SEQ_LEN       512

/* GRU weights container for 1 layer */
typedef struct {
    float ln_weight[WRAI_HIDDEN_DIM];
    float ln_bias[WRAI_HIDDEN_DIM];
    
    float scale_w_ih;
    int8_t* w_ih; /* 3072 x 1024 (3 * 1024 * 1024 bytes) */
    
    float scale_w_hh;
    int8_t* w_hh; /* 3072 x 1024 (3 * 1024 * 1024 bytes) */
    
    float b_ih[3 * WRAI_HIDDEN_DIM]; /* 3072 floats */
    float b_hh[3 * WRAI_HIDDEN_DIM]; /* 3072 floats */
} wrai_gru_layer_t;

/* Spectral DWT container */
typedef struct {
    float gate[WRAI_HIDDEN_DIM];
    float approx_gain[WRAI_HIDDEN_DIM >> WRAI_WAVELET_LEVELS]; /* 1024 / 16 = 64 floats */
    float* detail_gains[WRAI_WAVELET_LEVELS]; /* levels 0: 512, 1: 256, 2: 128, 3: 64 floats */
} wrai_spectral_t;

/* Full Model Weights Container */
typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t vocab_size;
    uint32_t hidden_dim;
    uint32_t num_layers;
    uint32_t wavelet_levels;
    uint32_t epoch;
    float best_val_loss;

    int32_t teacher_map[WRAI_VOCAB_SIZE];
    
    float scale_embed;
    int8_t* embed_weights; /* 32000 x 1024 bytes */
    
    float pe[WRAI_MAX_SEQ_LEN * WRAI_HIDDEN_DIM];
    
    wrai_spectral_t spectral1;
    wrai_spectral_t spectral2;
    
    wrai_gru_layer_t layers[WRAI_NUM_LAYERS];
    
    float ln_final_weight[WRAI_HIDDEN_DIM];
    float ln_final_bias[WRAI_HIDDEN_DIM];
    
    /* Memory block for all dynamic weight buffers */
    uint8_t* raw_weight_memory;
    size_t raw_weight_size;
} wrai_model_t;

/* Generation State Container (Zero-Allocation Ping-Pong State) */
typedef struct {
    float h_states[WRAI_NUM_LAYERS][WRAI_HIDDEN_DIM]; /* 12 x 1024 floats = 48 KB */
    float ping_buffer[WRAI_HIDDEN_DIM];               /* 4 KB (L1 Cache) */
    float pong_buffer[WRAI_HIDDEN_DIM];               /* 4 KB (L1 Cache) */
    float spectral_buf[WRAI_HIDDEN_DIM];
    float logits[WRAI_VOCAB_SIZE];
    uint32_t current_seq_len;
} wrai_inference_state_t;

/* Engine API */
bool wrai_load_model_binary(const char* bin_path, wrai_model_t* model);
void wrai_free_model(wrai_model_t* model);
void wrai_state_reset(wrai_inference_state_t* state);

/* Step-by-Step Forward Pass */
void wrai_forward_step(
    const wrai_model_t* model,
    wrai_inference_state_t* state,
    uint32_t token_id,
    float* out_logits
);

/* Sampling & Generation */
uint32_t wrai_sample_top_k(
    float* logits,
    uint32_t vocab_size,
    float temperature,
    uint32_t top_k,
    const uint32_t* history_tokens,
    uint32_t history_len,
    float repetition_penalty
);

#ifdef __cplusplus
}
#endif

#endif /* WRAI_V14_ENGINE_H */
