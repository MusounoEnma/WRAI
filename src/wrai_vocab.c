/**
 * @file wrai_vocab.c
 * @brief Implementasi Wavelet Spectral Token Vocabulary Engine.
 */

#include "wrai_vocab.h"
#include "wrai_math.h"
#include <string.h>

static uint32_t vocab_fnv1a(const char* s) {
    uint32_t h = 2166136261u;
    while (*s) {
        h ^= (uint8_t)(*s);
        h *= 16777619u;
        s++;
    }
    return h;
}

void wrai_vocab_init(wrai_spectral_vocab_t* vocab) {
    if (!vocab) return;
    memset(vocab, 0, sizeof(wrai_spectral_vocab_t));
}

bool wrai_vocab_add_token(wrai_spectral_vocab_t* vocab, uint16_t id, const char* token_str) {
    if (!vocab || !token_str || vocab->vocab_size >= WRAI_MAX_VOCAB_SIZE) return false;

    uint16_t idx = vocab->vocab_size;
    vocab->entries[idx].token_id = id;
    strncpy(vocab->entries[idx].token_str, token_str, WRAI_TOKEN_MAX_LEN - 1);
    vocab->entries[idx].token_str[WRAI_TOKEN_MAX_LEN - 1] = '\0';

    /* Compute 256-bin Q15 Spectral Signature for Token */
    uint32_t h = vocab_fnv1a(token_str);
    uint16_t primary_bin = (h % (WRAI_SPECTRAL_BINS - 2)) + 1;
    uint16_t harmonic_bin = ((h >> 8) % (WRAI_SPECTRAL_BINS - 2)) + 1;

    memset(vocab->entries[idx].coeffs, 0, sizeof(q15_t) * WRAI_SPECTRAL_BINS);
    vocab->entries[idx].coeffs[primary_bin] = WRAI_Q15(0.95f);
    vocab->entries[idx].coeffs[harmonic_bin] = WRAI_Q15(0.45f);

    vocab->vocab_size++;
    return true;
}

const wrai_vocab_entry_t* wrai_vocab_get_by_id(const wrai_spectral_vocab_t* vocab, uint16_t id) {
    if (!vocab) return NULL;
    for (uint16_t i = 0; i < vocab->vocab_size; i++) {
        if (vocab->entries[i].token_id == id) {
            return &vocab->entries[i];
        }
    }
    return NULL;
}

const wrai_vocab_entry_t* wrai_vocab_get_by_str(const wrai_spectral_vocab_t* vocab, const char* token_str) {
    if (!vocab || !token_str) return NULL;
    for (uint16_t i = 0; i < vocab->vocab_size; i++) {
        if (strcmp(vocab->entries[i].token_str, token_str) == 0) {
            return &vocab->entries[i];
        }
    }
    return NULL;
}
