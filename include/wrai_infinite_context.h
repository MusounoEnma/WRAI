/**
 * @file wrai_infinite_context.h
 * @brief Infinite Context Window Engine via Wavelet Spectral Compression & Cascade Store.
 *
 * Implements Gemini/Claude-style context management without amnesia or RAM explosion.
 * Uses Hierarchical Spectral Ring Buffers (L1 Active, L2 Condensed Spectral, L3 Permanent Fact Store).
 */

#ifndef WRAI_INFINITE_CONTEXT_H
#define WRAI_INFINITE_CONTEXT_H

#include "wrai_types.h"
#include "wrai_math.h"

#ifdef __cplusplus
extern "C" {
#endif

#define WRAI_MAX_ACTIVE_TURNS       10      /**< L1 Active Turns */
#define WRAI_MAX_FACT_ENTRIES       32      /**< L3 Permanent Spectral Fact Store */

typedef struct {
    char     fact_key[32];
    char     fact_value[64];
    q15_t    spectral_signature[WRAI_SPECTRAL_BINS];
} wrai_spectral_fact_t;

typedef struct {
    wrai_complex_q15_t  l1_active_turns[WRAI_MAX_ACTIVE_TURNS][WRAI_FFT_SIZE];
    uint8_t             l1_count;
    
    wrai_complex_q15_t  l2_condensed_spectral_summary[WRAI_FFT_SIZE]; /**< Condensed Spectral Abstract */
    
    wrai_spectral_fact_t l3_permanent_facts[WRAI_MAX_FACT_ENTRIES];
    uint8_t             l3_fact_count;
    
    uint32_t            total_turns_processed;
} wrai_infinite_context_engine_t;

/**
 * @brief Initializes Infinite Context Engine
 */
void wrai_infinite_context_init(wrai_infinite_context_engine_t* engine);

/**
 * @brief Ingests turn and handles auto-spectral compression when L1 buffer is full
 */
void wrai_infinite_context_push(wrai_infinite_context_engine_t* engine, 
                               const char* user_text, 
                               const char* wrai_text);

/**
 * @brief Recalls context across all 3 tiers (L1, L2 Condensed, L3 Permanent)
 */
bool wrai_infinite_context_recall(const wrai_infinite_context_engine_t* engine,
                                 const char* query,
                                 char* out_recalled_info,
                                 size_t max_len);

#ifdef __cplusplus
}
#endif

#endif /* WRAI_INFINITE_CONTEXT_H */
