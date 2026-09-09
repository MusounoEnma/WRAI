/**
 * @file wrai_context_memory.h
 * @brief Accumulated Wave Context Memory (Short-Term Conversation State) for WRAI.
 *
 * Implements KV-Cache-Free Multi-Turn Context Memory using 512-byte Q15 Wave Context Accumulation
 * and Harmonic Phase Decay without memory leakage.
 */

#ifndef WRAI_CONTEXT_MEMORY_H
#define WRAI_CONTEXT_MEMORY_H

#include "wrai_types.h"
#include "wrai_math.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    wrai_complex_q15_t accumulated_wave[WRAI_FFT_SIZE]; /**< 512-byte Complex Wave Context State */
    q15_t              context_decay_q15;                /**< Phase attenuation factor (default 0.85 Q15) */
    uint16_t           turns_count;
} wrai_context_memory_t;

/**
 * @brief Initializes Wave Context Memory
 */
void wrai_context_memory_init(wrai_context_memory_t* ctx);

/**
 * @brief Accumulates new turn wave signal into context state with harmonic decay
 * @param ctx Pointer to context memory
 * @param new_turn_wave New turn wave signal buffer
 */
void wrai_context_memory_push_turn(wrai_context_memory_t* ctx, const wrai_complex_q15_t* new_turn_wave);

/**
 * @brief Blends current context wave state with incoming query for multi-turn inference
 * @param ctx Pointer to context memory
 * @param input_query_wave Raw query wave signal (Input & Output blended in-place)
 */
void wrai_context_memory_blend_query(const wrai_context_memory_t* ctx, wrai_complex_q15_t* input_query_wave);

/**
 * @brief Clears / Resets context state
 */
void wrai_context_memory_reset(wrai_context_memory_t* ctx);

#ifdef __cplusplus
}
#endif

#endif /* WRAI_CONTEXT_MEMORY_H */
