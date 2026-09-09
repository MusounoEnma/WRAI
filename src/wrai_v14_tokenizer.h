/**
 * @file wrai_v14_tokenizer.h
 * @brief Fast C Native Subword Tokenizer & Vocab Lookup for WRAI v14.4
 */

#ifndef WRAI_V14_TOKENIZER_H
#define WRAI_V14_TOKENIZER_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t num_tokens;
    char** token_strings;
    uint16_t* token_lens;
    uint32_t eos_token_id;
    uint32_t pad_token_id;
} wrai_tokenizer_t;

/* Tokenizer API */
bool wrai_tokenizer_load(const char* vocab_bin_path, wrai_tokenizer_t* tok);
void wrai_tokenizer_free(wrai_tokenizer_t* tok);

/* Encode UTF-8 Text to Pruned Token IDs using greedy longest-match */
uint32_t wrai_tokenizer_encode(
    const wrai_tokenizer_t* tok,
    const char* text,
    uint32_t* out_token_ids,
    uint32_t max_tokens
);

/* Decode single token ID to string */
const char* wrai_tokenizer_decode_token(
    const wrai_tokenizer_t* tok,
    uint32_t token_id
);

#ifdef __cplusplus
}
#endif

#endif /* WRAI_V14_TOKENIZER_H */
