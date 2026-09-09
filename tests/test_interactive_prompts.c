#include "wrai_v16_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void run_prompt(
    wrai_v16_model_t* model,
    wrai_v16_tokenizer_t* tok,
    wrai_v16_state_t* state,
    float* logits,
    const char* user_query
) {
    wrai_v16_state_reset(state);

    uint32_t prompt_tokens[256];
    int n = 0;
    prompt_tokens[n++] = tok->im_start_id;
    n += wrai_v16_tokenize(tok, "user\n", prompt_tokens + n, 256 - n);
    n += wrai_v16_tokenize(tok, user_query, prompt_tokens + n, 256 - n);
    prompt_tokens[n++] = tok->im_end_id;
    n += wrai_v16_tokenize(tok, "\n", prompt_tokens + n, 256 - n);
    prompt_tokens[n++] = tok->im_start_id;
    n += wrai_v16_tokenize(tok, "assistant\n", prompt_tokens + n, 256 - n);

    clock_t t0 = clock();
    for (int i = 0; i < n; i++) {
        bool is_last = (i == n - 1);
        wrai_v16_forward_step(model, state, prompt_tokens[i], logits, is_last);
    }
    clock_t t1 = clock();
    double prefill_sec = (double)(t1 - t0) / CLOCKS_PER_SEC;

    wrai_v16_sample_params_t sp;
    sp.temperature = 0.35f;
    sp.top_k = 30;
    sp.top_p = 0.90f;
    sp.repetition_penalty = 1.15f;
    sp.max_tokens = 70;

    uint32_t history[256];
    int hlen = 0;

    printf("=================================================================\n");
    printf("User > %s\n", user_query);
    printf("WRAI > ");
    fflush(stdout);

    clock_t tg0 = clock();
    int gen_count = 0;
    for (int step = 0; step < sp.max_tokens; step++) {
        uint32_t next_token = wrai_v16_sample_token(logits, WRAI_V16_VOCAB_SIZE, history, hlen, &sp);
        if (step >= 5 && (next_token == tok->im_end_id || next_token == tok->eos_id)) break;

        printf("%s", wrai_v16_decode_token(tok, next_token));
        fflush(stdout);

        gen_count++;
        if (hlen < 256) history[hlen++] = next_token;
        wrai_v16_forward_step(model, state, next_token, logits, true);
    }
    clock_t tg1 = clock();
    double gen_sec = (double)(tg1 - tg0) / CLOCKS_PER_SEC;
    double speed = (gen_sec > 0.0) ? (gen_count / gen_sec) : 0.0;

    printf("\n[STATS] Prefill %d tokens (%.2fs) | Gen %d tokens (%.2fs = %.2f tok/s)\n\n",
           n, prefill_sec, gen_count, gen_sec, speed);
}

int main() {
    wrai_v16_model_t model;
    if (!wrai_v16_load_model("1.7b models/wrai_v16_1.7b_int8.bin", &model)) return 1;

    wrai_v16_tokenizer_t tok;
    if (!wrai_v16_load_tokenizer("1.7b models/wrai_v16_vocab.bin", &tok)) return 1;

    wrai_v16_state_t state;
    wrai_v16_state_init(&state);

    float* logits = (float*)malloc(WRAI_V16_VOCAB_SIZE * sizeof(float));

    run_prompt(&model, &tok, &state, logits, "Siapa kamu?");
    run_prompt(&model, &tok, &state, logits, "Buatkan fungsi Python untuk membalikkan string.");

    free(logits);
    wrai_v16_state_free(&state);
    wrai_v16_free_tokenizer(&tok);
    wrai_v16_free_model(&model);
    return 0;
}
