/**
 * @file wrai_math.h
 * @brief Fixed-Point Cooley-Tukey Radix-2 FFT Engine & Spectral Signal Processing.
 *
 * Bare-metal C implementation using Q15 arithmetic and Trigonometric LUTs.
 */

#ifndef WRAI_MATH_H
#define WRAI_MATH_H

#include "wrai_types.h"
#include "wrai_lut.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Performs Bit-Reversal Permutation on complex Q15 array in-place
 * @param buffer Complex signal array of length N (WRAI_FFT_SIZE)
 * @param n Length of FFT (must be power of 2)
 */
void wrai_bit_reversal_q15(wrai_complex_q15_t* buffer, uint16_t n);

/**
 * @brief Computes In-Place Radix-2 Cooley-Tukey FFT (Fixed-Point Q15)
 * @param buffer Array of complex samples (Input time domain, Output frequency domain)
 * @param n FFT size (WRAI_FFT_SIZE = 512)
 */
void wrai_fft_q15_radix2(wrai_complex_q15_t* buffer, uint16_t n);

/**
 * @brief Computes fast spectral magnitudes using Alpha Max + Beta Min algorithm
 * @param frame Pointer to spectral frame containing complex FFT bins
 */
void wrai_compute_magnitudes_q15(wrai_spectral_frame_t* frame);

/**
 * @brief Computes Resonance Match Score between input spectrum and target pattern
 * @param input_mags Array of positive bin magnitudes (256 Q15 entries)
 * @param target_coeffs Array of pattern spectral coefficients (256 Q15 entries)
 * @return 32-bit integer Resonance Match Score (Q31 range)
 */
q31_t wrai_compute_resonance_score(const q15_t* input_mags, const q15_t* target_coeffs);

#ifdef __cplusplus
}
#endif

#endif /* WRAI_MATH_H */
