#include "wrai_v16_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main() {
    wrai_v16_model_t model;
    if (!wrai_v16_load_model("1.7b models/wrai_v16_1.7b_int8.bin", &model)) return 1;

    wrai_v16_tokenizer_t tok;
    if (!wrai_v16_load_tokenizer("1.7b models/wrai_v16_vocab.bin", &tok)) return 1;

    wrai_v16_state_t state;
    wrai_v16_state_init(&state);

    float* logits = (float*)malloc(WRAI_V16_VOCAB_SIZE * sizeof(float));

    const char* user_query = "gimana kabarmu hari ini ?";
    uint32_t prompt_tokens[256];
    int n = 0;
    prompt_tokens[n++] = tok.im_start_id;
    n += wrai_v16_tokenize(&tok, "user\n", prompt_tokens + n, 256 - n);
    n += wrai_v16_tokenize(&tok, user_query, prompt_tokens + n, 256 - n);
    prompt_tokens[n++] = tok.im_end_id;
    n += wrai_v16_tokenize(&tok, "\n", prompt_tokens + n, 256 - n);
    prompt_tokens[n++] = tok.im_start_id;
    n += wrai_v16_tokenize(&tok, "assistant\n", prompt_tokens + n, 256 - n);

    printf("Prompt tokens (%d):\n", n);
    for (int i = 0; i < n; i++) {
        printf("  [%d] %d -> '%s'\n", i, prompt_tokens[i], wrai_v16_decode_token(&tok, prompt_tokens[i]));
    }

    for (int i = 0; i < n; i++) {
        bool is_last = (i == n - 1);
        wrai_v16_forward_step(&model, &state, prompt_tokens[i], logits, is_last);
    }

    /* Print Top 10 logits */
    typedef struct { uint32_t id; float val; } ts_t;
    ts_t top[10];
    for (int j = 0; j < 10; j++) { top[j].id = 0; top[j].val = -1e9f; }
    for (uint32_t v = 0; v < WRAI_V16_VOCAB_SIZE; v++) {
        float val = logits[v];
        if (val > top[9].val) {
            top[9].id = v; top[9].val = val;
            for (int k = 8; k >= 0; k--) {
                if (top[k+1].val > top[k].val) {
                    ts_t tmp = top[k]; top[k] = top[k+1]; top[k+1] = tmp;
                }
            }
        }
    }

    printf("\nTop 10 predicted tokens after prompt:\n");
    for (int j = 0; j < 10; j++) {
        printf("  #%d: ID %u ('%s') -> logit %.3f\n", j+1, top[j].id, wrai_v16_decode_token(&tok, top[j].id), top[j].val);
    }

    /* Generate 30 tokens */
    wrai_v16_sample_params_t sp;
    sp.temperature = 0.35f;
    sp.top_k = 30;
    sp.top_p = 0.90f;
    sp.repetition_penalty = 1.15f;
    sp.max_tokens = 30;

    uint32_t history[256];
    int hlen = 0;

    printf("\nGenerated text: ");
    for (int step = 0; step < sp.max_tokens; step++) {
        uint32_t next_token = wrai_v16_sample_token(logits, WRAI_V16_VOCAB_SIZE, history, hlen, &sp);
        if (step >= 5 && (next_token == tok.im_end_id || next_token == tok.eos_id)) break;
        printf("%s", wrai_v16_decode_token(&tok, next_token));
        fflush(stdout);
        if (hlen < 256) history[hlen++] = next_token;
        wrai_v16_forward_step(&model, &state, next_token, logits, true);
    }
    printf("\n");

    free(logits);
    wrai_v16_state_free(&state);
    wrai_v16_free_tokenizer(&tok);
    wrai_v16_free_model(&model);
    return 0;
}
