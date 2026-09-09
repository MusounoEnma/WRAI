/**
 * @file wrai_context_memory.c
 * @brief Implementasi Accumulated Wave Context Memory (Short-Term Conversation State).
 */

#include "wrai_context_memory.h"
#include <string.h>

void wrai_context_memory_init(wrai_context_memory_t* ctx) {
    if (!ctx) return;
    memset(ctx, 0, sizeof(wrai_context_memory_t));
    ctx->context_decay_q15 = WRAI_Q15(0.85f); // 85% retention, 15% decay per turn
    ctx->turns_count = 0;
}

void wrai_context_memory_push_turn(wrai_context_memory_t* ctx, const wrai_complex_q15_t* new_turn_wave) {
    if (!ctx || !new_turn_wave) return;

    for (uint16_t n = 0; n < WRAI_FFT_SIZE; n++) {
        /* Apply Harmonic Phase Decay to existing context */
        q15_t decayed_r = WRAI_Q15_MUL(ctx->accumulated_wave[n].real, ctx->context_decay_q15);
        q15_t decayed_i = WRAI_Q15_MUL(ctx->accumulated_wave[n].imag, ctx->context_decay_q15);

        /* Blend new turn wave signal */
        ctx->accumulated_wave[n].real = wrai_q15_add_sat(decayed_r, new_turn_wave[n].real >> 1);
        ctx->accumulated_wave[n].imag = wrai_q15_add_sat(decayed_i, new_turn_wave[n].imag >> 1);
    }

    ctx->turns_count++;
}

void wrai_context_memory_blend_query(const wrai_context_memory_t* ctx, wrai_complex_q15_t* input_query_wave) {
    if (!ctx || !input_query_wave || ctx->turns_count == 0) return;

    for (uint16_t n = 0; n < WRAI_FFT_SIZE; n++) {
        /* Superimpose accumulated context wave onto input query wave */
        q15_t ctx_r = ctx->accumulated_wave[n].real >> 1;
        q15_t ctx_i = ctx->accumulated_wave[n].imag >> 1;

        input_query_wave[n].real = wrai_q15_add_sat(input_query_wave[n].real >> 1, ctx_r);
        input_query_wave[n].imag = wrai_q15_add_sat(input_query_wave[n].imag >> 1, ctx_i);
    }
}

void wrai_context_memory_reset(wrai_context_memory_t* ctx) {
    if (!ctx) return;
    memset(ctx, 0, sizeof(wrai_context_memory_t));
    ctx->context_decay_q15 = WRAI_Q15(0.85f);
}
