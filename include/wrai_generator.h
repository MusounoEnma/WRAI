/**
 * @file wrai_generator.h
 * @brief Auto-Regressive Spectral Wavelet Token Generator Engine.
 *
 * Implements 100% Zero-GEMM Auto-Regressive Token Synthesis via Spectral Probability Sampling
 * and Inverse Fast Fourier Wavelet Reconstruction.
 */

#ifndef WRAI_GENERATOR_H
#define WRAI_GENERATOR_H

#include "wrai_types.h"
#include "wrai_math.h"
#include "wrai_vocab.h"
#include "wrai_context_memory.h"

#ifdef __cplusplus
extern "C" {
#endif

#define WRAI_MAX_GEN_TOKENS     64          /**< Max Auto-Regressive Generated Tokens per Sequence */

typedef struct {
    uint16_t generated_token_ids[WRAI_MAX_GEN_TOKENS];
    uint8_t  generated_count;
    q31_t    sequence_resonance_score;
} wrai_generation_result_t;

/**
 * @brief Performs one auto-regressive step: predicts next token ID from spectral state
 * @param current_mags Current 256-bin Q15 spectral magnitude state
 * @param vocab Pointer to spectral token vocabulary
 * @param temperature_q15 Temperature parameter in Q15 (default 0.8 Q15)
 * @return Winning next token ID
 */
uint16_t wrai_generator_step_token(const q15_t* current_mags,
                                  const wrai_spectral_vocab_t* vocab,
                                  q15_t temperature_q15);

/**
 * @brief Auto-regressively generates token sequence from context wave state
 * @param ctx Pointer to accumulated context memory
 * @param vocab Pointer to spectral token vocabulary
 * @param max_tokens Max tokens to generate
 * @param out_result Pointer to store generation result
 */
void wrai_generator_run_sequence(wrai_context_memory_t* ctx,
                                const wrai_spectral_vocab_t* vocab,
                                uint8_t max_tokens,
                                wrai_generation_result_t* out_result);

#ifdef __cplusplus
}
#endif

#endif /* WRAI_GENERATOR_H */
