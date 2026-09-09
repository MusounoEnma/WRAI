/**
 * @file wrai_generator.c
 * @brief Implementasi Auto-Regressive Spectral Wavelet Token Generator Engine.
 */

#include "wrai_generator.h"
#include <string.h>

uint16_t wrai_generator_step_token(const q15_t* current_mags,
                                  const wrai_spectral_vocab_t* vocab,
                                  q15_t temperature_q15) {
    if (!current_mags || !vocab || vocab->vocab_size == 0) return 0;

    q31_t best_score = -2147483647;
    uint16_t winning_token_id = vocab->entries[0].token_id;

    for (uint16_t v = 0; v < vocab->vocab_size; v++) {
        q31_t dot_product = 0;
        for (uint16_t m = 0; m < WRAI_SPECTRAL_BINS; m++) {
            dot_product += WRAI_Q15_MUL(current_mags[m], vocab->entries[v].coeffs[m]);
        }

        /* Apply Q15 Temperature scaling */
        q31_t scaled_score = WRAI_Q15_MUL(dot_product, temperature_q15);

        if (scaled_score > best_score) {
            best_score = scaled_score;
            winning_token_id = vocab->entries[v].token_id;
        }
    }

    return winning_token_id;
}

void wrai_generator_run_sequence(wrai_context_memory_t* ctx,
                                const wrai_spectral_vocab_t* vocab,
                                uint8_t max_tokens,
                                wrai_generation_result_t* out_result) {
    if (!ctx || !vocab || !out_result) return;
    memset(out_result, 0, sizeof(wrai_generation_result_t));

    uint8_t count = 0;
    q15_t temp_q15 = WRAI_Q15(0.85f);

    while (count < max_tokens && count < WRAI_MAX_GEN_TOKENS) {
        /* Extract current spectral magnitudes from accumulated context wave */
        q15_t current_mags[WRAI_SPECTRAL_BINS];
        for (uint16_t m = 0; m < WRAI_SPECTRAL_BINS; m++) {
            q15_t r_abs = WRAI_ABS(ctx->accumulated_wave[m].real);
            q15_t i_abs = WRAI_ABS(ctx->accumulated_wave[m].imag);
            current_mags[m] = (r_abs > i_abs ? r_abs : i_abs) + ((3 * (r_abs < i_abs ? r_abs : i_abs)) >> 3);
        }

        /* Auto-regressive step: predict next token */
        uint16_t next_token_id = wrai_generator_step_token(current_mags, vocab, temp_q15);
        out_result->generated_token_ids[count] = next_token_id;
        count++;

        /* Superimpose new token wave into context memory for next token prediction */
        const wrai_vocab_entry_t* entry = wrai_vocab_get_by_id(vocab, next_token_id);
        if (entry) {
            wrai_complex_q15_t token_wave[WRAI_FFT_SIZE];
            for (uint16_t n = 0; n < WRAI_FFT_SIZE; n++) {
                token_wave[n].real = entry->coeffs[n % WRAI_SPECTRAL_BINS] >> 2;
                token_wave[n].imag = entry->coeffs[(n + 64) % WRAI_SPECTRAL_BINS] >> 2;
            }
            wrai_context_memory_push_turn(ctx, token_wave);
        }
    }

    out_result->generated_count = count;
}
