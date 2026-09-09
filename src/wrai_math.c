/**
 * @file wrai_math.c
 * @brief Implementasi Fixed-Point Cooley-Tukey Radix-2 FFT Engine & Spectral Math.
 */

#include "wrai_math.h"

void wrai_bit_reversal_q15(wrai_complex_q15_t* buffer, uint16_t n) {
    uint16_t j = 0;
    for (uint16_t i = 0; i < n - 1; i++) {
        if (i < j) {
            wrai_complex_q15_t temp = buffer[i];
            buffer[i] = buffer[j];
            buffer[j] = temp;
        }
        uint16_t k = n >> 1;
        while (k <= j) {
            j -= k;
            k >>= 1;
        }
        j += k;
    }
}

void wrai_fft_q15_radix2(wrai_complex_q15_t* buffer, uint16_t n) {
    if (!buffer || n < 2) return;

    /* 1. Perform Bit-Reversal Permutation */
    wrai_bit_reversal_q15(buffer, n);

    /* 2. Cooley-Tukey Butterfly Stages */
    for (uint16_t len = 2; len <= n; len <<= 1) {
        uint16_t half = len >> 1;
        uint32_t step = WRAI_LUT_SIZE / len;

        for (uint16_t k = 0; k < n; k += len) {
            for (uint16_t j = 0; j < half; j++) {
                uint32_t lut_idx = j * step;

                q15_t cos_val = wrai_lut_cos(lut_idx);
                q15_t sin_val = wrai_lut_sin(lut_idx);

                q15_t u_r = buffer[k + j].real >> 1;
                q15_t u_i = buffer[k + j].imag >> 1;

                q15_t v_r = buffer[k + j + half].real >> 1;
                q15_t v_i = buffer[k + j + half].imag >> 1;

                /* Complex multiplication with twiddle factor e^(-j*2pi*k/N) */
                q15_t tw_r = WRAI_Q15_MUL(v_r, cos_val) + WRAI_Q15_MUL(v_i, sin_val);
                q15_t tw_i = WRAI_Q15_MUL(v_i, cos_val) - WRAI_Q15_MUL(v_r, sin_val);

                buffer[k + j].real        = u_r + tw_r;
                buffer[k + j].imag        = u_i + tw_i;
                buffer[k + j + half].real = u_r - tw_r;
                buffer[k + j + half].imag = u_i - tw_i;
            }
        }
    }
}

void wrai_compute_magnitudes_q15(wrai_spectral_frame_t* frame) {
    if (!frame) return;

    for (uint16_t m = 0; m < WRAI_SPECTRAL_BINS; m++) {
        q15_t r_abs = WRAI_ABS(frame->bins[m].real);
        q15_t i_abs = WRAI_ABS(frame->bins[m].imag);

        q15_t max_val = (r_abs > i_abs) ? r_abs : i_abs;
        q15_t min_val = (r_abs > i_abs) ? i_abs : r_abs;

        /* Alpha Max + Beta Min Algorithm (Alpha = 1.0, Beta = 3/8) */
        frame->magnitude[m] = max_val + ((3 * min_val) >> 3);
    }
}

q31_t wrai_compute_resonance_score(const q15_t* input_mags, const q15_t* target_coeffs) {
    if (!input_mags || !target_coeffs) return 0;

    q31_t score = 0;
    for (uint16_t m = 0; m < WRAI_SPECTRAL_BINS; m++) {
        score += WRAI_Q15_MUL(input_mags[m], target_coeffs[m]);
    }
    return score;
}
