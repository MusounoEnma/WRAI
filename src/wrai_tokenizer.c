/**
 * @file wrai_tokenizer.c
 * @brief Implementasi Embedded Bare-Metal Tokenizer WRAI.
 */

#include "wrai_tokenizer.h"
#include <string.h>
#include <ctype.h>

/** Fast FNV-1a Hash for String Token deterministic wave parameter generation */
static uint32_t fnv1a_hash(const char* str) {
    uint32_t hash = 2166136261u;
    while (*str) {
        hash ^= (uint8_t)(*str++);
        hash *= 16777619u;
    }
    return hash;
}

void wrai_tokenizer_init(wrai_tokenizer_t* tok) {
    if (!tok) return;
    memset(tok, 0, sizeof(wrai_tokenizer_t));
}

bool wrai_tokenizer_add_word(wrai_tokenizer_t* tok, const char* word, uint16_t token_id) {
    if (!tok || !word || tok->vocab_count >= WRAI_MAX_VOCAB_SIZE) {
        return false;
    }

    uint16_t idx = tok->vocab_count;
    strncpy(tok->entries[idx].word, word, WRAI_MAX_WORD_LEN - 1);
    tok->entries[idx].word[WRAI_MAX_WORD_LEN - 1] = '\0';

    uint32_t h = fnv1a_hash(word);
    
    /* Frequency Bin index (0 to 255 for WRAI_SPECTRAL_BINS) */
    uint16_t freq = (h % (WRAI_SPECTRAL_BINS - 2)) + 1; // avoid DC 0
    uint16_t phase = (h >> 8) % WRAI_LUT_SIZE;

    tok->entries[idx].wave_params.token_id   = token_id;
    tok->entries[idx].wave_params.freq_index = freq;
    tok->entries[idx].wave_params.phase_shift= phase;
    tok->entries[idx].wave_params.amplitude  = WRAI_Q15(0.9f);

    tok->vocab_count++;
    return true;
}

uint8_t wrai_tokenize_text(const wrai_tokenizer_t* tok, 
                          const char* text, 
                          wrai_token_wave_t* out_tokens, 
                          uint8_t max_tokens) {
    if (!tok || !text || !out_tokens || max_tokens == 0) return 0;

    char buffer[256];
    strncpy(buffer, text, sizeof(buffer) - 1);
    buffer[sizeof(buffer) - 1] = '\0';

    /* Convert to lowercase */
    for (int i = 0; buffer[i]; i++) {
        buffer[i] = (char)tolower((unsigned char)buffer[i]);
        if (ispunct((unsigned char)buffer[i])) {
            buffer[i] = ' ';
        }
    }

    uint8_t count = 0;
    char* token_str = strtok(buffer, " \t\r\n");

    while (token_str != NULL && count < max_tokens) {
        /* Lookup in vocab */
        bool found = false;
        for (uint16_t v = 0; v < tok->vocab_count; v++) {
            if (strcmp(tok->entries[v].word, token_str) == 0) {
                out_tokens[count] = tok->entries[v].wave_params;
                count++;
                found = true;
                break;
            }
        }

        /* If word not in registered vocab, generate dynamic token wave on-the-fly */
        if (!found) {
            uint32_t h = fnv1a_hash(token_str);
            out_tokens[count].token_id    = (uint16_t)(h & 0xFFFF);
            out_tokens[count].freq_index  = (h % (WRAI_SPECTRAL_BINS - 2)) + 1;
            out_tokens[count].phase_shift = (h >> 8) % WRAI_LUT_SIZE;
            out_tokens[count].amplitude   = WRAI_Q15(0.8f);
            count++;
        }

        token_str = strtok(NULL, " \t\r\n");
    }

    return count;
}
