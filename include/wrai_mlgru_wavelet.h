/**
 * @file wrai_mlgru_wavelet.h
 * @brief MatMul-free Linear GRU (MLGRU) & Haar-DWT Wavelet Spectral Engine C Native Core.
 *
 * Implements MatMul-free state recurrence (0% spectral divergence risk under Int8 fixed-point quantization)
 * paired with 1D Fast Haar Discrete Wavelet Transform (DWT) channel-mixing.
 *
 * Designed for 0% KV-Cache, 2 x 8 KB Ping-Pong SRAM Execution on bare-metal MCU & CPU.
 */

#ifndef WRAI_MLGRU_WAVELET_H
#define WRAI_MLGRU_WAVELET_H

#include "wrai_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================= */
/* CONSTANTS & DIMENSIONS                                                    */
/* ========================================================================= */

#define WRAI_V14_HIDDEN_DIM      1024  /**< Student Hidden Dimension (1024) */
#define WRAI_V14_VOCAB_SIZE       32000 /**< Pruned Active Vocabulary Size */
#define WRAI_V14_WAVELET_LEVELS   4     /**< Haar DWT Decomposition Levels */

/* ========================================================================= */
/* STRUCT DEFINITIONS                                                         */
/* ========================================================================= */

/**
 * @brief MLGRU State Container (MatMul-free Recurrent Hidden State)
 */
typedef struct {
    int8_t  h_state[WRAI_V14_HIDDEN_DIM]; /**< Recurrent Hidden State Vector (Q7 int8) */
    uint32_t step_count;                  /**< Sequence Step Count */
} wrai_mlgru_state_t;

/**
 * @brief Model Weights Container (Int8 Fixed-Point Quantized)
 */
typedef struct {
    const int8_t*  embed_weights;      /**< Vocabulary Embedding Matrix (32000 x 1024 int8) */
    const int8_t*  forget_proj_weight; /**< Forget Gate Linear Proj (1024 x 1024 int8) */
    const int8_t*  cand_proj_weight;   /**< Candidate Gate Linear Proj (1024 x 1024 int8) */
    const q15_t*   wavelet_detail_gains[WRAI_V14_WAVELET_LEVELS]; /**< Haar Detail Gains Q15 */
    const q15_t*   wavelet_approx_gain;                           /**< Haar Approx Gain Q15 */
} wrai_mlgru_weights_t;

/* ========================================================================= */
/* PUBLIC C ENGINE FUNCTION PROTOTYPES                                       */
/* ========================================================================= */

/**
 * @brief Initialize MLGRU Hidden State to Zero.
 */
void wrai_mlgru_state_init(wrai_mlgru_state_t* state);

/**
 * @brief Fast In-Place 1D Haar Discrete Wavelet Transform (DWT) Forward Pass.
 * @param vec Array of Q7/Q15 hidden values of size WRAI_V14_HIDDEN_DIM
 * @param num_levels Decomposition levels (default 4)
 */
void wrai_haar_dwt_forward_q15(q15_t* vec, int num_levels);

/**
 * @brief Fast In-Place 1D Haar Inverse Discrete Wavelet Transform (IDWT).
 * @param vec Array of Q7/Q15 hidden values of size WRAI_V14_HIDDEN_DIM
 * @param num_levels Decomposition levels (default 4)
 */
void wrai_haar_dwt_inverse_q15(q15_t* vec, int num_levels);

/**
 * @brief Execute One Autoregressive MLGRU + Wavelet Step (MatMul-Free Recurrence).
 *
 * @param state Pointer to persistent hidden state (2 x 8 KB SRAM Ping-Pong Buffer)
 * @param token_id Input token ID (0 .. 31999)
 * @param weights Pointer to quantized Int8 model weights
 * @param output_logits Pointer to array of size WRAI_V14_VOCAB_SIZE (or top-K buffer)
 * @return Best predicted next token ID
 */
uint16_t wrai_mlgru_step(
    wrai_mlgru_state_t* state,
    uint16_t token_id,
    const wrai_mlgru_weights_t* weights,
    int32_t* output_logits
);

#ifdef __cplusplus
}
#endif

#endif /* WRAI_MLGRU_WAVELET_H */
