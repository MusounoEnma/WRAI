/**
 * @file wrai_tokenizer.h
 * @brief Embedded Bare-Metal Tokenizer for WRAI.
 *
 * Maps text words/tokens to deterministic Waveform parameters (freq_index, phase_shift, amplitude).
 * Static memory layout without dynamic allocation.
 */

#ifndef WRAI_TOKENIZER_H
#define WRAI_TOKENIZER_H

#include "wrai_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define WRAI_MAX_VOCAB_SIZE     1024    /**< Static Vocabulary Size */
#define WRAI_MAX_WORD_LEN       32      /**< Max length of single word token */

typedef struct {
    char             word[WRAI_MAX_WORD_LEN];
    wrai_token_wave_t wave_params;
} wrai_vocab_entry_t;

typedef struct {
    wrai_vocab_entry_t entries[WRAI_MAX_VOCAB_SIZE];
    uint16_t           vocab_count;
} wrai_tokenizer_t;

/**
 * @brief Initializes Tokenizer state
 */
void wrai_tokenizer_init(wrai_tokenizer_t* tok);

/**
 * @brief Registers word token into vocabulary with deterministic frequency mapping
 */
bool wrai_tokenizer_add_word(wrai_tokenizer_t* tok, const char* word, uint16_t token_id);

/**
 * @brief Tokenizes input text sentence into array of token wave parameters
 * @param tok Pointer to tokenizer instance
 * @param text Input text string
 * @param out_tokens Output array of wrai_token_wave_t
 * @param max_tokens Max capacity of out_tokens array
 * @return Number of tokens successfully extracted
 */
uint8_t wrai_tokenize_text(const wrai_tokenizer_t* tok, 
                          const char* text, 
                          wrai_token_wave_t* out_tokens, 
                          uint8_t max_tokens);

#ifdef __cplusplus
}
#endif

#endif /* WRAI_TOKENIZER_H */
