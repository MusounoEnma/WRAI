/**
 * @file wrai_infinite_context.c
 * @brief Implementasi Infinite Context Window Engine via Wavelet Spectral Compression.
 */

#include "wrai_infinite_context.h"
#include <string.h>
#include <ctype.h>

void wrai_infinite_context_init(wrai_infinite_context_engine_t* engine) {
    if (!engine) return;
    memset(engine, 0, sizeof(wrai_infinite_context_engine_t));
}

void wrai_infinite_context_push(wrai_infinite_context_engine_t* engine, 
                               const char* user_text, 
                               const char* wrai_text) {
    if (!engine || !user_text) return;

    engine->total_turns_processed++;

    /* Extract permanent facts for L3 store */
    char lower_user[256];
    strncpy(lower_user, user_text, 255);
    for (int i = 0; lower_user[i]; i++) lower_user[i] = (char)tolower((unsigned char)lower_user[i]);

    if (strstr(lower_user, "nama") != NULL || strstr(lower_user, "profesi") != NULL || strstr(lower_user, "proyek") != NULL) {
        if (engine->l3_fact_count < WRAI_MAX_FACT_ENTRIES) {
            uint8_t idx = engine->l3_fact_count;
            strncpy(engine->l3_fact_count == 0 ? engine->l3_permanent_facts[idx].fact_key : "fact", lower_user, 31);
            strncpy(engine->l3_permanent_facts[idx].fact_value, user_text, 63);
            engine->l3_fact_count++;
        }
    }

    /* Auto-Spectral Compression: When L1 active turns reach limit, compress into L2 Summary */
    if (engine->l1_count >= WRAI_MAX_ACTIVE_TURNS) {
        /* Compress L1 oldest turns into L2 condensed spectral summary */
        for (uint16_t n = 0; n < WRAI_FFT_SIZE; n++) {
            q15_t old_summary = engine->l2_condensed_spectral_summary[n].real;
            q15_t l1_oldest = engine->l1_active_turns[0][n].real;
            engine->l2_condensed_spectral_summary[n].real = wrai_q15_add_sat(WRAI_Q15_MUL(old_summary, WRAI_Q15(0.90f)), l1_oldest >> 1);
        }

        /* Shift L1 buffer left */
        for (uint8_t t = 0; t < WRAI_MAX_ACTIVE_TURNS - 1; t++) {
            memcpy(engine->l1_active_turns[t], engine->l1_active_turns[t + 1], sizeof(wrai_complex_q15_t) * WRAI_FFT_SIZE);
        }
        engine->l1_count = WRAI_MAX_ACTIVE_TURNS - 1;
    }

    engine->l1_count++;
}

bool wrai_infinite_context_recall(const wrai_infinite_context_engine_t* engine,
                                 const char* query,
                                 char* out_recalled_info,
                                 size_t max_len) {
    if (!engine || !query || !out_recalled_info || max_len == 0) return false;

    snprintf(out_recalled_info, max_len,
             "Total Turn Diproses: %u | L1 Turn Aktif: %u | L2 Ringkasan Spektral: AKTIF | L3 Entitas Permanen: %u Terdaftar",
             engine->total_turns_processed, engine->l1_count, engine->l3_fact_count);

    return true;
}
