/**
 * @file wrai_v8_core.h
 * @brief WRAI Next-Gen v8 High-Dimensional Spectral Engine Core.
 *
 * Implements 4,096-Bin Complex Q31 Spectral FFT, Mamba-2 Selective Wave State,
 * and Fourier-KAN Edge Activation Layers.
 * 100% Zero-GEMM, Fixed-Point Q31/Q15 ANSI C99 bare-metal kernel.
 */

#ifndef WRAI_V8_CORE_H
#define WRAI_V8_CORE_H

#include "wrai_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define WRAI_V8_FFT_SIZE            4096        /**< 4096-Point High-Dimensional FFT */
#define WRAI_V8_SPECTRAL_BINS       2048        /**< 2048 Positive Frequency Bins */
#define WRAI_V8_KAN_HARMONICS       8           /**< 8 Fourier-KAN Edge Harmonics */

/**
 * @brief Mamba-2 Selective Wave Context State Structure
 */
typedef struct {
    wrai_complex_q15_t wave_state[WRAI_V8_FFT_SIZE];  /**< Selective Spectral Wave State */
    q15_t              selective_decay[WRAI_V8_SPECTRAL_BINS]; /**< Selective Decay Masks */
    uint32_t           total_tokens_processed;
} wrai_mamba_wave_state_t;

/**
 * @brief Initializes WRAI v8 Mamba-2 Selective Wave State
 */
void wrai_v8_mamba_state_init(wrai_mamba_wave_state_t* state);

/**
 * @brief Fourier-KAN Edge Activation Function Q31
 * Calculates sum of 8 Fourier Spline Edge Harmonics for zero-GEMM precision logic.
 */
q31_t wrai_v8_kan_fourier_edge_q31(q31_t input_val, const q15_t* kan_weights);

/**
 * @brief High-Dimensional 4096-Point Radix-2 FFT (Q15 Fixed-Point)
 */
void wrai_v8_fft_4096_q15(q15_t* real, q15_t* imag);

#ifdef __cplusplus
}
#endif

#endif /* WRAI_V8_CORE_H */
