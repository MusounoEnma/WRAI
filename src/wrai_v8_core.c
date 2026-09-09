/**
 * @file wrai_v8_core.c
 * @brief Implementasi WRAI Next-Gen v8 High-Dimensional Spectral Engine Core.
 */

#include "wrai_v8_core.h"
#include <string.h>

void wrai_v8_mamba_state_init(wrai_mamba_wave_state_t* state) {
    if (!state) return;
    memset(state, 0, sizeof(wrai_mamba_wave_state_t));
    for (uint16_t m = 0; m < WRAI_V8_SPECTRAL_BINS; m++) {
        state->selective_decay[m] = WRAI_Q15(0.88f); // 88% retention default
    }
}

q31_t wrai_v8_kan_fourier_edge_q31(q31_t input_val, const q15_t* kan_weights) {
    if (!kan_weights) return input_val;

    q31_t acc = 0;
    q15_t in_q15 = (q15_t)(input_val >> 16);

    for (uint8_t k = 0; k < WRAI_V8_KAN_HARMONICS; k++) {
        /* Compute Fourier-KAN Spline Edge Component */
        q15_t weight = kan_weights[k];
        q15_t harmonic_val = WRAI_Q15_MUL(in_q15, (q15_t)((k + 1) * 4096));
        acc += ((q31_t)weight * (q31_t)harmonic_val) >> 15;
    }

    return acc;
}

void wrai_v8_fft_4096_q15(q15_t* real, q15_t* imag) {
    if (!real || !imag) return;

    /* Bit-Reversal Permutation for 4096 points */
    uint16_t j = 0;
    for (uint16_t i = 0; i < WRAI_V8_FFT_SIZE - 1; i++) {
        if (i < j) {
            q15_t tr = real[i]; real[i] = real[j]; real[j] = tr;
            q15_t ti = imag[i]; imag[i] = imag[j]; imag[j] = ti;
        }
        uint16_t k = WRAI_V8_FFT_SIZE >> 1;
        while (k <= j) {
            j -= k;
            k >>= 1;
        }
        j += k;
    }

    /* Radix-2 Butterfly Computation */
    uint16_t len = 2;
    while (len <= WRAI_V8_FFT_SIZE) {
        uint16_t half = len >> 1;
        for (uint16_t i = 0; i < WRAI_V8_FFT_SIZE; i += len) {
            for (uint16_t step = 0; step < half; step++) {
                uint16_t idx1 = i + step;
                uint16_t idx2 = idx1 + half;

                q15_t u_r = real[idx1] >> 1;
                q15_t u_i = imag[idx1] >> 1;
                q15_t v_r = real[idx2] >> 1;
                q15_t v_i = imag[idx2] >> 1;

                real[idx1] = wrai_q15_add_sat(u_r, v_r);
                imag[idx1] = wrai_q15_add_sat(u_i, v_i);
                real[idx2] = wrai_q15_add_sat(u_r, -v_r);
                imag[idx2] = wrai_q15_add_sat(u_i, -v_i);
            }
        }
        len <<= 1;
    }
}
