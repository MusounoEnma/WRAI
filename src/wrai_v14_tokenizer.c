/**
 * @file wrai_v14_tokenizer.c
 * @brief Fast Greedy Longest-Match Subword Tokenizer in C Native
 */

#include "wrai_v14_tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* First-byte index table for ultra-fast lookup */
typedef struct {
    uint32_t count;
    uint32_t* token_indices; /* sorted by descending token_len */
} byte_bucket_t;

static byte_bucket_t g_byte_buckets[256];
static bool g_buckets_initialized = false;

static int compare_tokens_by_len(const void* a, const void* b, void* context) {
    const wrai_tokenizer_t* tok = (const wrai_tokenizer_t*)context;
    uint32_t id_a = *(const uint32_t*)a;
    uint32_t id_b = *(const uint32_t*)b;
    /* Descending order: longest token first */
    return (int)tok->token_lens[id_b] - (int)tok->token_lens[id_a];
}

static void build_lookup_index(wrai_tokenizer_t* tok) {
    for (int b = 0; b < 256; b++) {
        g_byte_buckets[b].count = 0;
        if (g_byte_buckets[b].token_indices) {
            free(g_byte_buckets[b].token_indices);
            g_byte_buckets[b].token_indices = NULL;
        }
    }

    /* Count per first-byte */
    for (uint32_t i = 0; i < tok->num_tokens; i++) {
        if (tok->token_lens[i] > 0) {
            uint8_t first = (uint8_t)tok->token_strings[i][0];
            g_byte_buckets[first].count++;
        }
    }

    /* Allocate and fill */
    for (int b = 0; b < 256; b++) {
        if (g_byte_buckets[b].count > 0) {
            g_byte_buckets[b].token_indices = (uint32_t*)malloc(g_byte_buckets[b].count * sizeof(uint32_t));
            uint32_t idx = 0;
            for (uint32_t i = 0; i < tok->num_tokens; i++) {
                if (tok->token_lens[i] > 0 && (uint8_t)tok->token_strings[i][0] == (uint8_t)b) {
                    g_byte_buckets[b].token_indices[idx++] = i;
                }
            }
            /* Sort by length descending using bubble/insertion sort */
            for (uint32_t x = 0; x < g_byte_buckets[b].count; x++) {
                for (uint32_t y = x + 1; y < g_byte_buckets[b].count; y++) {
                    uint32_t t_x = g_byte_buckets[b].token_indices[x];
                    uint32_t t_y = g_byte_buckets[b].token_indices[y];
                    if (tok->token_lens[t_y] > tok->token_lens[t_x]) {
                        g_byte_buckets[b].token_indices[x] = t_y;
                        g_byte_buckets[b].token_indices[y] = t_x;
                    }
                }
            }
        }
    }
    g_buckets_initialized = true;
}

bool wrai_tokenizer_load(const char* vocab_bin_path, wrai_tokenizer_t* tok) {
    if (!vocab_bin_path || !tok) return false;
    memset(tok, 0, sizeof(wrai_tokenizer_t));

    FILE* f = fopen(vocab_bin_path, "rb");
    if (!f) return false;

    if (fread(&tok->num_tokens, sizeof(uint32_t), 1, f) != 1) {
        fclose(f);
        return false;
    }

    tok->token_strings = (char**)calloc(tok->num_tokens, sizeof(char*));
    tok->token_lens = (uint16_t*)calloc(tok->num_tokens, sizeof(uint16_t));
    tok->eos_token_id = 0;
    tok->pad_token_id = 0;

    for (uint32_t i = 0; i < tok->num_tokens; i++) {
        uint16_t len = 0;
        if (fread(&len, sizeof(uint16_t), 1, f) != 1) break;
        tok->token_lens[i] = len;
        tok->token_strings[i] = (char*)malloc(len + 1);
        if (fread(tok->token_strings[i], 1, len, f) != len) break;
        tok->token_strings[i][len] = '\0';

        /* Detect EOS token */
        if (strcmp(tok->token_strings[i], "<|im_end|>") == 0 ||
            strcmp(tok->token_strings[i], "<|endoftext|>") == 0) {
            tok->eos_token_id = i;
        }
    }

    fclose(f);
    build_lookup_index(tok);
    return true;
}

void wrai_tokenizer_free(wrai_tokenizer_t* tok) {
    if (!tok) return;
    if (tok->token_strings) {
        for (uint32_t i = 0; i < tok->num_tokens; i++) {
            if (tok->token_strings[i]) free(tok->token_strings[i]);
        }
        free(tok->token_strings);
    }
    if (tok->token_lens) free(tok->token_lens);
    for (int b = 0; b < 256; b++) {
        if (g_byte_buckets[b].token_indices) {
            free(g_byte_buckets[b].token_indices);
            g_byte_buckets[b].token_indices = NULL;
        }
        g_byte_buckets[b].count = 0;
    }
}

uint32_t wrai_tokenizer_encode(
    const wrai_tokenizer_t* tok,
    const char* text,
    uint32_t* out_token_ids,
    uint32_t max_tokens
) {
    if (!tok || !text || !out_token_ids || max_tokens == 0) return 0;

    size_t text_len = strlen(text);
    size_t p = 0;
    uint32_t num_encoded = 0;

    while (p < text_len && num_encoded < max_tokens) {
        uint8_t first = (uint8_t)text[p];
        bool matched = false;

        if (g_byte_buckets[first].count > 0) {
            for (uint32_t k = 0; k < g_byte_buckets[first].count; k++) {
                uint32_t cand_id = g_byte_buckets[first].token_indices[k];
                uint16_t cand_len = tok->token_lens[cand_id];

                if (p + cand_len <= text_len) {
                    if (memcmp(text + p, tok->token_strings[cand_id], cand_len) == 0) {
                        out_token_ids[num_encoded++] = cand_id;
                        p += cand_len;
                        matched = true;
                        break;
                    }
                }
            }
        }

        if (!matched) {
            /* Single unmapped byte advance */
            p++;
        }
    }

    return num_encoded;
}

const char* wrai_tokenizer_decode_token(
    const wrai_tokenizer_t* tok,
    uint32_t token_id
) {
    if (!tok || !tok->token_strings || token_id >= tok->num_tokens) return "";
    return tok->token_strings[token_id] ? tok->token_strings[token_id] : "";
}
