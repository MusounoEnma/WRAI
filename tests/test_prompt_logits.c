#include "wrai_v16_engine.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    wrai_v16_model_t model;
    if (!wrai_v16_load_model("1.7b models/wrai_v16_1.7b_int8.bin", &model)) return 1;

    wrai_v16_tokenizer_t tok;
    if (!wrai_v16_load_tokenizer("1.7b models/wrai_v16_vocab.bin", &tok)) return 1;

    wrai_v16_state_t state;
    wrai_v16_state_init(&state);

    float* logits = (float*)malloc(WRAI_V16_VOCAB_SIZE * sizeof(float));

    uint32_t prompt_tokens[64];
    int n = 0;
    prompt_tokens[n++] = tok.im_start_id;
    n += wrai_v16_tokenize(&tok, "user\n", prompt_tokens + n, 64 - n);
    n += wrai_v16_tokenize(&tok, "Halo", prompt_tokens + n, 64 - n);
    prompt_tokens[n++] = tok.im_end_id;
    n += wrai_v16_tokenize(&tok, "\n", prompt_tokens + n, 64 - n);
    prompt_tokens[n++] = tok.im_start_id;
    n += wrai_v16_tokenize(&tok, "assistant\n", prompt_tokens + n, 64 - n);

    printf("Prompt tokens (%d): ", n);
    for (int i = 0; i < n; i++) printf("%d ('%s') ", prompt_tokens[i], wrai_v16_decode_token(&tok, prompt_tokens[i]));
    printf("\n");

    for (int i = 0; i < n; i++) {
        bool is_last = (i == n - 1);
        wrai_v16_forward_step(&model, &state, prompt_tokens[i], logits, is_last);
    }

    printf("\nTop 10 next tokens predicted by WRAI v16:\n");
    for (int top = 0; top < 10; top++) {
        int best = 0;
        float max_val = -1e9f;
        for (int i = 0; i < WRAI_V16_VOCAB_SIZE; i++) {
            if (logits[i] > max_val) {
                max_val = logits[i];
                best = i;
            }
        }
        printf("  Top %d: token %d (logit: %.3f) -> '%s'\n", top + 1, best, max_val, wrai_v16_decode_token(&tok, best));
        logits[best] = -1e9f;
    }

    free(logits);
    wrai_v16_state_free(&state);
    wrai_v16_free_tokenizer(&tok);
    wrai_v16_free_model(&model);
    return 0;
}
