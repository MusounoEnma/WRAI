/**
 * @file wrai_encoder.h
 * @brief Token-to-Wave Encoding & Resonance Peak Detection Engine.
 */

#ifndef WRAI_ENCODER_H
#define WRAI_ENCODER_H

#include "wrai_types.h"
#include "wrai_math.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Generates Time-Domain Superposition Signal from Active Token Waves
 * @param tokens Array of active token waves
 * @param num_tokens Number of tokens in sequence (max WRAI_MAX_SUPERPOSITION)
 * @param out_buffer Output complex buffer for FFT processing (Length WRAI_FFT_SIZE)
 */
void wrai_encode_tokens_to_wave(const wrai_token_wave_t* tokens, 
                                uint8_t num_tokens, 
                                wrai_complex_q15_t* out_buffer);

/**
 * @brief Scans spectral patterns to detect highest harmonic resonance
 * @param input_frame Input spectral frame after FFT computation
 * @param patterns Array of stored quantized spectral patterns
 * @param num_patterns Total patterns to compare
 * @param out_best_score Pointer to store winning resonance score
 * @return Winning pattern_id with maximum resonance score
 */
uint16_t wrai_detect_peak_resonance(const wrai_spectral_frame_t* input_frame,
                                    const wrai_pattern_packet_t* patterns,
                                    uint32_t num_patterns,
                                    q31_t* out_best_score);

#ifdef __cplusplus
}
#endif

#endif /* WRAI_ENCODER_H */
