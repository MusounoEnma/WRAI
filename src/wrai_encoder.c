/**
 * @file wrai_encoder.c
 * @brief Implementasi Token-to-Wave Encoder & Resonance Peak Detection.
 */

#include "wrai_encoder.h"
#include <string.h>

void wrai_encode_tokens_to_wave(const wrai_token_wave_t* tokens, 
                                uint8_t num_tokens, 
                                wrai_complex_q15_t* out_buffer) {
    if (!out_buffer) return;

    /* Reset buffer */
    memset(out_buffer, 0, sizeof(wrai_complex_q15_t) * WRAI_FFT_SIZE);

    if (!tokens || num_tokens == 0) return;
    if (num_tokens > WRAI_MAX_SUPERPOSITION) {
        num_tokens = WRAI_MAX_SUPERPOSITION;
    }

    /* Compute right-shift scale factor based on number of active superposition waves */
    uint8_t shift_scale = 0;
    uint8_t temp_k = num_tokens;
    while (temp_k > 1) {
        shift_scale++;
        temp_k >>= 1;
    }

    for (uint16_t n = 0; n < WRAI_FFT_SIZE; n++) {
        q31_t sample_acc = 0;

        for (uint8_t k = 0; k < num_tokens; k++) {
            uint32_t phase_idx = (tokens[k].freq_index * n) + tokens[k].phase_shift;
            q15_t sin_val = wrai_lut_sin(phase_idx);
            
            sample_acc += WRAI_Q15_MUL(tokens[k].amplitude, sin_val);
        }

        /* Scale down to prevent overflow during FFT superposition */
        out_buffer[n].real = (q15_t)(sample_acc >> shift_scale);
        out_buffer[n].imag = 0;
    }
}

uint16_t wrai_detect_peak_resonance(const wrai_spectral_frame_t* input_frame,
                                    const wrai_pattern_packet_t* patterns,
                                    uint32_t num_patterns,
                                    q31_t* out_best_score) {
    if (!input_frame || !patterns || num_patterns == 0) {
        if (out_best_score) *out_best_score = 0;
        return 0;
    }

    q31_t max_score = -2147483647;
    uint16_t best_id = 0;

    for (uint32_t p = 0; p < num_patterns; p++) {
        q31_t score = wrai_compute_resonance_score(input_frame->magnitude, patterns[p].coeffs);
        if (score > max_score) {
            max_score = score;
            best_id = patterns[p].pattern_id;
        }
    }

    if (out_best_score) {
        *out_best_score = max_score;
    }

    return best_id;
}
