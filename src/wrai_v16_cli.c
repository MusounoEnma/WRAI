/**
 * @file wrai_v16_cli.c
 * @brief Interactive Native C CLI Terminal for WRAI v16 (1.7B)
 */

#include "wrai_v16_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#endif

#define MAX_PROMPT_LEN 4096
#define MAX_HISTORY_TOKENS 2048

static void print_banner(const wrai_v16_model_t* model) {
    printf("\n=================================================================\n");
    printf("   🌟 WRAI v16 (1.7B) PURE NATIVE C INFERENCE ENGINE            \n");
    printf("=================================================================\n");
    printf(" [*] Architecture   : RetNet Multi-Head + Haar DWT 1D + SwiGLU\n");
    printf(" [*] Parameters     : ~1.84 Billion (1,838,131,658)\n");
    printf(" [*] Quantization   : INT8 Symmetric Row-wise (Loss: %.4f)\n", model->header.loss);
    printf(" [*] Target Hardware: AMD A8 Puma+ (AVX 1.0 SIMD, 8GB RAM)\n");
    printf(" [*] Memory Budget  : O(1) Constant (~28 MB Recurrent State, 0 KV-Cache)\n");
    printf(" [*] Mapping Mode   : Zero-Heap Virtual Memory-Mapped (mmap)\n");
    printf("=================================================================\n\n");
}

int main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif

    const char* model_path = "1.7b models/wrai_v16_1.7b_int8.bin";
    const char* vocab_path = "1.7b models/wrai_v16_vocab.bin";

    if (argc >= 2) model_path = argv[1];
    if (argc >= 3) vocab_path = argv[2];

    printf("[*] Loading WRAI v16 Model from: %s...\n", model_path);
    wrai_v16_model_t model;
    if (!wrai_v16_load_model(model_path, &model)) {
        fprintf(stderr, "[ERROR] Failed to load binary model file.\n");
        return 1;
    }
    printf("[OK] Model mapped successfully! File size: %.2f GB\n", (double)model.file_size / (1024.0 * 1024.0 * 1024.0));

    printf("[*] Loading Tokenizer Vocabulary from: %s...\n", vocab_path);
    wrai_v16_tokenizer_t tok;
    if (!wrai_v16_load_tokenizer(vocab_path, &tok)) {
        fprintf(stderr, "[ERROR] Failed to load tokenizer binary vocabulary.\n");
        wrai_v16_free_model(&model);
        return 1;
    }
    printf("[OK] Tokenizer Ready! Total Vocabulary: %u tokens.\n", tok.num_tokens);

    wrai_v16_state_t state;
    if (!wrai_v16_state_init(&state)) {
        fprintf(stderr, "[ERROR] Failed to allocate recurrent state buffer.\n");
        wrai_v16_free_tokenizer(&tok);
        wrai_v16_free_model(&model);
        return 1;
    }
    printf("[OK] Recurrent State Buffer Ready: ~28 MB RAM (O(1) Constant).\n");

    print_banner(&model);

    /* Allocate logits buffer */
    float* logits = (float*)malloc(WRAI_V16_VOCAB_SIZE * sizeof(float));
    if (!logits) {
        fprintf(stderr, "[ERROR] Failed to allocate logits buffer.\n");
        return 1;
    }

    wrai_v16_sample_params_t sample_params;
    sample_params.temperature = 0.35f;
    sample_params.top_p = 0.90f;
    sample_params.top_k = 30;
    sample_params.repetition_penalty = 1.15f;
    sample_params.max_tokens = 256;

    uint32_t prompt_tokens[1024];
    uint32_t history[MAX_HISTORY_TOKENS];
    int history_len = 0;

    char user_input[MAX_PROMPT_LEN];
    printf("Type your question (or 'exit' to quit, 'reset' to reset context).\n");
    printf("[*] Recommended Prompts:\n");
    printf("    1. Who are you?\n");
    printf("    2. Write a Python function to reverse a string.\n");
    printf("    3. Explain who you are and how the WRAI ecosystem operates.\n");
    printf("    4. What is SQL Injection and how can developers prevent it?\n\n");

    while (1) {
        printf("User > ");
        fflush(stdout);

        if (!fgets(user_input, sizeof(user_input), stdin)) break;

        /* Trim newline */
        size_t ulen = strlen(user_input);
        while (ulen > 0 && (user_input[ulen - 1] == '\n' || user_input[ulen - 1] == '\r')) {
            user_input[--ulen] = '\0';
        }

        if (ulen == 0) continue;
        if (strcmp(user_input, "exit") == 0 || strcmp(user_input, "quit") == 0) break;
        if (strcmp(user_input, "reset") == 0) {
            wrai_v16_state_reset(&state);
            history_len = 0;
            printf("[*] Recurrent memory context successfully reset to zero.\n\n");
            continue;
        }

        /* Intelligent greeting expansion for ultra-short single-word greetings */
        const char* prompt_to_send = user_input;
        if (_stricmp(user_input, "halo") == 0 || _stricmp(user_input, "hai") == 0 ||
            _stricmp(user_input, "hi") == 0 || _stricmp(user_input, "hello") == 0) {
            prompt_to_send = "Hello, who are you?";
        }

        /* Format Qwen ChatML Prompt cleanly */
        int num_p_tokens = 0;
        prompt_tokens[num_p_tokens++] = tok.im_start_id;
        num_p_tokens += wrai_v16_tokenize(&tok, "user\n", prompt_tokens + num_p_tokens, 1024 - num_p_tokens);
        num_p_tokens += wrai_v16_tokenize(&tok, prompt_to_send, prompt_tokens + num_p_tokens, 1024 - num_p_tokens);
        prompt_tokens[num_p_tokens++] = tok.im_end_id;
        num_p_tokens += wrai_v16_tokenize(&tok, "\n", prompt_tokens + num_p_tokens, 1024 - num_p_tokens);
        prompt_tokens[num_p_tokens++] = tok.im_start_id;
        num_p_tokens += wrai_v16_tokenize(&tok, "assistant\n", prompt_tokens + num_p_tokens, 1024 - num_p_tokens);

        if (num_p_tokens <= 0) {
            printf("[WARN] Failed to tokenize input.\n");
            continue;
        }

        /* Prefill prompt tokens into recurrent state (only compute logits on last token!) */
        clock_t t_prefill_start = clock();
        for (int i = 0; i < num_p_tokens; i++) {
            bool is_last_token = (i == num_p_tokens - 1);
            wrai_v16_forward_step(&model, &state, prompt_tokens[i], logits, is_last_token);
            if (history_len < MAX_HISTORY_TOKENS) {
                history[history_len++] = prompt_tokens[i];
            }
        }
        clock_t t_prefill_end = clock();
        double prefill_sec = (double)(t_prefill_end - t_prefill_start) / CLOCKS_PER_SEC;

        printf("WRAI > ");
        fflush(stdout);

        clock_t t_gen_start = clock();
        int gen_tokens_count = 0;

        /* Generation Loop */
        for (int step = 0; step < sample_params.max_tokens; step++) {
            uint32_t next_token = wrai_v16_sample_token(
                logits, WRAI_V16_VOCAB_SIZE, history, history_len, &sample_params
            );

            /* Stop on EOS or <|im_end|> (allow at least 5 tokens) */
            if (step >= 5 && (next_token == tok.im_end_id || next_token == tok.eos_id)) {
                break;
            }

            const char* token_str = wrai_v16_decode_token(&tok, next_token);
            printf("%s", token_str);
            fflush(stdout);

            gen_tokens_count++;
            if (history_len < MAX_HISTORY_TOKENS) {
                history[history_len++] = next_token;
            }

            /* Step forward with generated token (compute logits for next token) */
            wrai_v16_forward_step(&model, &state, next_token, logits, true);
        }

        clock_t t_gen_end = clock();
        double gen_sec = (double)(t_gen_end - t_gen_start) / CLOCKS_PER_SEC;
        double speed = (gen_sec > 0.0) ? (gen_tokens_count / gen_sec) : 0.0;

        printf("\n\n[INFO] Prefill: %d tokens (%.2fs) | Generated: %d tokens in %.2fs (%.2f tokens/s)\n\n",
               num_p_tokens, prefill_sec, gen_tokens_count, gen_sec, speed);
    }

    /* Cleanup */
    free(logits);
    wrai_v16_state_free(&state);
    wrai_v16_free_tokenizer(&tok);
    wrai_v16_free_model(&model);

    printf("\n[*] WRAI v16 engine cleanly terminated. Farewell!\n");
    return 0;
}
