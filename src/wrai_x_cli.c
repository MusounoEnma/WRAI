/**
 * @file wrai_x_cli.c
 * @brief Interactive Native C CLI Terminal for WRAI-X (0.6B)
 */

#include "wrai_x_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#endif

#define MAX_PROMPT_LEN 4096

static void print_banner(const wrai_x_model_t* model) {
    printf("\n=================================================================\n");
    printf("   🌊 WRAI-X (0.6B) PURE NATIVE C INFERENCE ENGINE               \n");
    printf("=================================================================\n");
    printf(" [*] Arsitektur     : Dual-State Retention (Mt/Rt) + 4-Level Haar DWT\n");
    printf(" [*] Dimensi Laten  : D = 1024 (2^10), FFN = 3072, 28 Layers\n");
    printf(" [*] Parameter      : ~650 Juta (Qwen 0.6B Frozen Brain + WRAI-X Adapters)\n");
    printf(" [*] Kuantisasi     : INT8 Symmetric Row-wise (~540 MB Binary)\n");
    printf(" [*] Hardware Target: AMD A8 Puma+ (AVX 1.0 SIMD, RAM ~600 MB)\n");
    printf(" [*] Memory Buffer  : O(1) Constant (~14.5 MB State Buffer, 0%% KV-Cache)\n");
    printf(" [*] Mapping Mode   : Zero-Heap Virtual Memory-Mapped (mmap)\n");
    printf("=================================================================\n\n");
}

int main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif

    const char* model_path = "wrai_x_06b_int8.bin";
    const char* vocab_path = "wrai_x_vocab.bin";

    if (argc >= 2) model_path = argv[1];
    if (argc >= 3) vocab_path = argv[2];

    printf("[*] Memuat Model WRAI-X dari: %s...\n", model_path);
    wrai_x_model_t model;
    if (!wrai_x_load_model(model_path, &model)) {
        fprintf(stderr, "[ERROR] Gagal memuat file binary model WRAI-X (%s).\n", model_path);
        fprintf(stderr, "[INFO] Pastikan file hasil kuantisasi Colab sudah di-download ke folder ini.\n");
        return 1;
    }
    printf("[OK] Model WRAI-X berhasil dimap! Ukuran file: %.2f MB\n", (double)model.file_size / (1024.0 * 1024.0));

    printf("[*] Memuat Vocabulary Tokenizer dari: %s...\n", vocab_path);
    wrai_x_tokenizer_t tok;
    if (!wrai_x_load_tokenizer(vocab_path, &tok)) {
        fprintf(stderr, "[ERROR] Gagal memuat file vocabulary binary (%s).\n", vocab_path);
        wrai_x_free_model(&model);
        return 1;
    }
    printf("[OK] Tokenizer Siap! Total Vocabulary: %u tokens.\n", tok.num_tokens);

    wrai_x_state_t state;
    if (!wrai_x_state_init(&state)) {
        fprintf(stderr, "[ERROR] Gagal menginisialisasi buffer state WRAI-X.\n");
        wrai_x_free_tokenizer(&tok);
        wrai_x_free_model(&model);
        return 1;
    }
    printf("[OK] Dual-State Memory Buffer Siap! (~14.5 MB Allocated)\n");

    print_banner(&model);

    float* logits = (float*)malloc((size_t)WRAI_X_VOCAB_SIZE * sizeof(float));
    char user_input[MAX_PROMPT_LEN];

    printf("Ketik pertanyaanmu (atau 'exit' untuk keluar, 'reset' untuk reset konteks).\n");
    printf("[*] Rekomendasi Pertanyaan:\n");
    printf("    1. halo apa kabar?\n");
    printf("    2. himpunan\n");
    printf("    3. Siapa kamu?\n");
    printf("    4. Buatkan fungsi Python untuk membalikkan string.\n\n");

    while (1) {
        printf("User > ");
        fflush(stdout);

        if (!fgets(user_input, sizeof(user_input), stdin)) break;
        user_input[strcspn(user_input, "\r\n")] = '\0';

        if (strlen(user_input) == 0) continue;
        if (strcmp(user_input, "exit") == 0) break;
        if (strcmp(user_input, "reset") == 0) {
            wrai_x_state_reset(&state);
            printf("[*] Konteks memori percakapan telah direset!\n\n");
            continue;
        }

        /* Format ChatML */
        char prompt[MAX_PROMPT_LEN + 128];
        snprintf(prompt, sizeof(prompt), "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", user_input);

        int32_t prompt_tokens[1024];
        int num_prompt_tokens = 0;

        /* Simple greedy BPE tokenizer lookup */
        size_t p_len = strlen(prompt);
        size_t cursor = 0;
        while (cursor < p_len && num_prompt_tokens < 1024) {
            int best_id = -1;
            size_t best_len = 0;
            for (uint32_t t = 0; t < tok.num_tokens; t++) {
                size_t t_len = tok.token_lens[t];
                if (t_len > 0 && t_len <= (p_len - cursor)) {
                    if (strncmp(prompt + cursor, tok.token_strings[t], t_len) == 0) {
                        if (t_len > best_len) {
                            best_len = t_len;
                            best_id = (int)t;
                        }
                    }
                }
            }
            if (best_id != -1) {
                prompt_tokens[num_prompt_tokens++] = best_id;
                cursor += best_len;
            } else {
                cursor++;
            }
        }

        /* Prefill Phase */
        clock_t t_prefill_start = clock();
        for (int i = 0; i < num_prompt_tokens; i++) {
            wrai_x_forward_step(&model, &state, prompt_tokens[i], logits);
        }
        clock_t t_prefill_end = clock();
        double prefill_sec = (double)(t_prefill_end - t_prefill_start) / CLOCKS_PER_SEC;

        printf("WRAI-X > ");
        fflush(stdout);

        /* Generation Phase */
        clock_t t_gen_start = clock();
        int gen_tokens_count = 0;
        int max_gen = 256;

        for (int step = 0; step < max_gen; step++) {
            int next_token = wrai_x_sample(logits, WRAI_X_VOCAB_SIZE, 0.0f, 0.9f);
            if (next_token == tok.im_end_id || next_token == tok.eos_id) break;

            if (next_token >= 0 && (uint32_t)next_token < tok.num_tokens) {
                printf("%s", tok.token_strings[next_token]);
                fflush(stdout);
            }
            gen_tokens_count++;
            wrai_x_forward_step(&model, &state, next_token, logits);
        }
        clock_t t_gen_end = clock();
        double gen_sec = (double)(t_gen_end - t_gen_start) / CLOCKS_PER_SEC;
        double tok_per_sec = (gen_sec > 0.0) ? ((double)gen_tokens_count / gen_sec) : 0.0;

        printf("\n\n[INFO] Prefill: %d tokens (%.2fs) | Generated: %d tokens in %.2fs (%.2f tokens/s)\n\n",
               num_prompt_tokens, prefill_sec, gen_tokens_count, gen_sec, tok_per_sec);
    }

    free(logits);
    wrai_x_state_free(&state);
    wrai_x_free_tokenizer(&tok);
    wrai_x_free_model(&model);
    return 0;
}
