/**
 * @file wrai_vocab.h
 * @brief Wavelet Spectral Token Vocabulary Engine for True Generative WRAI.
 *
 * Maps vocabulary tokens to 256-bin Q15 fixed-point spectral signatures.
 * Zero-GEMM, Heap-Free Static Memory Structure for bare-metal execution.
 */

#ifndef WRAI_VOCAB_H
#define WRAI_VOCAB_H

#include "wrai_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define WRAI_MAX_VOCAB_SIZE     1024        /**< Maximum Vocabulary Size */
#define WRAI_TOKEN_MAX_LEN      32          /**< Maximum Length of Single Token */

typedef struct {
    uint16_t token_id;                      /**< Unique Token ID */
    char     token_str[WRAI_TOKEN_MAX_LEN]; /**< Human Readable Token String */
    q15_t    coeffs[WRAI_SPECTRAL_BINS];    /**< 256-bin Q15 Spectral Signature */
} wrai_vocab_entry_t;

typedef struct {
    wrai_vocab_entry_t entries[WRAI_MAX_VOCAB_SIZE];
    uint16_t           vocab_size;
} wrai_spectral_vocab_t;

/**
 * @brief Initializes Wavelet Spectral Vocabulary with core tokens
 */
void wrai_vocab_init(wrai_spectral_vocab_t* vocab);

/**
 * @brief Adds token entry to vocabulary and computes Q15 spectral signature
 */
bool wrai_vocab_add_token(wrai_spectral_vocab_t* vocab, uint16_t id, const char* token_str);

/**
 * @brief Finds token entry by ID
 */
const wrai_vocab_entry_t* wrai_vocab_get_by_id(const wrai_spectral_vocab_t* vocab, uint16_t id);

/**
 * @brief Finds token entry by string
 */
const wrai_vocab_entry_t* wrai_vocab_get_by_str(const wrai_spectral_vocab_t* vocab, const char* token_str);

#ifdef __cplusplus
}
#endif

#endif /* WRAI_VOCAB_H */
