/**
 * @file wrai_v14_5_engine.h
 * @brief High-Speed Native C Inference Engine Header for WRAI v14.5 (SwiGLU + RMSNorm + Haar DWT)
 */

#ifndef WRAI_V14_5_ENGINE_H
#define WRAI_V14_5_ENGINE_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WRAI_MAGIC_HEADER      0x57524149  /* "WRAI" */
#define WRAI_VERSION_14_5      145
#define WRAI_VOCAB_SIZE        32000
#define WRAI_HIDDEN_DIM        1024
#define WRAI_FFN_DIM           1536
#define WRAI_NUM_LAYERS        12
#define WRAI_WAVELET_LEVELS    4
#define WRAI_MAX_SEQ_LEN       512
#define WRAI_SRAM_BUFFER_SIZE  4096

/* SwiGLU Gated Feed-Forward Network */
typedef struct {
    float scale_w_gate;
    int8_t* w_gate;     /* [1536 x 1024] */
    float scale_w_up;
    int8_t* w_up;       /* [1536 x 1024] */
    float scale_w_down;
    int8_t* w_down;     /* [1024 x 1536] */
} wrai_swiglu_t;

/* Hybrid Layer (RMSNorm + ResGRU + RMSNorm + SwiGLU) */
typedef struct {
    float rms_gru_weight[WRAI_HIDDEN_DIM];
    
    /* GRU Weights */
    float scale_w_ih;
    int8_t* w_ih;       /* [3072 x 1024] */
    float scale_w_hh;
    int8_t* w_hh;       /* [3072 x 1024] */
    float b_ih[3 * WRAI_HIDDEN_DIM];
    float b_hh[3 * WRAI_HIDDEN_DIM];

    float rms_ffn_weight[WRAI_HIDDEN_DIM];
    wrai_swiglu_t ffn;
} wrai_layer_v14_5_t;

/* Wavelet Spectral Mixer */
typedef struct {
    float gate[WRAI_HIDDEN_DIM];
    float approx_gain[WRAI_HIDDEN_DIM >> WRAI_WAVELET_LEVELS];
    float* detail_gains[WRAI_WAVELET_LEVELS];
} wrai_spectral_v14_5_t;

/* Model Container */
typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t vocab_size;
    uint32_t hidden_dim;
    uint32_t num_layers;
    uint32_t wavelet_levels;
    uint32_t ffn_dim;
    uint32_t epoch;
    float best_val_loss;

    int32_t teacher_map[WRAI_VOCAB_SIZE];
    float scale_embed;
    int8_t* embed_weights;  /* [32000 x 1024] */
    float pe[WRAI_MAX_SEQ_LEN * WRAI_HIDDEN_DIM];

    wrai_spectral_v14_5_t spectral1;
    wrai_spectral_v14_5_t spectral2;

    wrai_layer_v14_5_t layers[WRAI_NUM_LAYERS];
    float rms_final_weight[WRAI_HIDDEN_DIM];
} wrai_model_v14_5_t;

/* Persistent Zero KV-Cache Inference State */
typedef struct {
    float h_states[WRAI_NUM_LAYERS][WRAI_HIDDEN_DIM];
    uint32_t step_pos;
    float sram_ping[WRAI_HIDDEN_DIM];
    float sram_pong[WRAI_HIDDEN_DIM];
    float logits[WRAI_VOCAB_SIZE];
} wrai_state_v14_5_t;

/* Engine API */
bool wrai_load_model_v14_5(const char* bin_path, wrai_model_v14_5_t* model);
void wrai_free_model_v14_5(wrai_model_v14_5_t* model);
void wrai_state_reset_v14_5(wrai_state_v14_5_t* state);

void wrai_forward_step_v14_5(
    const wrai_model_v14_5_t* model,
    wrai_state_v14_5_t* state,
    uint32_t token_id,
    float* out_logits
);

#ifdef __cplusplus
}
#endif

#endif /* WRAI_V14_5_ENGINE_H */
