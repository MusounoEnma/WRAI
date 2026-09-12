/**
 * @file wrai_smollm2_config.h
 * @brief WRAI Architecture Constants & Memory Dimensions for SmolLM2 Family
 * 
 * Supports:
 *   - SmolLM2-135M (Default): 30 layers, 576 hidden, 9 heads, 64 head_dim
 *   - SmolLM2-360M:          32 layers, 960 hidden, 15 heads, 64 head_dim
 */

#ifndef WRAI_SMOLLM2_CONFIG_H
#define WRAI_SMOLLM2_CONFIG_H

#include <stddef.h>
#include <stdint.h>

#ifdef WRAI_USE_SMOLLM2_360M
/* =========================================================================
 * SmolLM2-360M Dimensions
 * ========================================================================= */
#define WRAI_SMOL_NUM_LAYERS      32
#define WRAI_SMOL_HIDDEN_DIM      960
#define WRAI_SMOL_NUM_HEADS       15
#define WRAI_SMOL_HEAD_DIM        64
#define WRAI_SMOL_FFN_DIM         2560
#define WRAI_SMOL_VOCAB_SIZE      49152
#define WRAI_SMOL_MODEL_NAME      "WRAI-Smol-360M"

#else
/* =========================================================================
 * SmolLM2-135M Dimensions (Default: Ultra-Lightweight Edge Profile)
 * ========================================================================= */
#define WRAI_SMOL_NUM_LAYERS      30
#define WRAI_SMOL_HIDDEN_DIM      576
#define WRAI_SMOL_NUM_HEADS       9
#define WRAI_SMOL_HEAD_DIM        64
#define WRAI_SMOL_FFN_DIM         1536
#define WRAI_SMOL_VOCAB_SIZE      49152
#define WRAI_SMOL_MODEL_NAME      "WRAI-Smol-135M"

#endif

#define WRAI_SMOL_RMS_EPS         1e-5f
#define WRAI_SMOL_WAVELET_LEVELS  4

/* Head state matrix: 64 x 64 = 4096 elements per head */
#define WRAI_SMOL_HEAD_STATE_SIZE (WRAI_SMOL_HEAD_DIM * WRAI_SMOL_HEAD_DIM)

/* Total recurrent state floats per layer (Mt + Rt): 2 * H * 64 * 64 */
#define WRAI_SMOL_LAYER_STATE_FLOATS (2 * WRAI_SMOL_NUM_HEADS * WRAI_SMOL_HEAD_STATE_SIZE)

/* Total recurrent state buffer bytes across all layers in FP16 */
#define WRAI_SMOL_TOTAL_STATE_BYTES_FP16 (WRAI_SMOL_NUM_LAYERS * WRAI_SMOL_LAYER_STATE_FLOATS * 2)

#endif /* WRAI_SMOLLM2_CONFIG_H */
