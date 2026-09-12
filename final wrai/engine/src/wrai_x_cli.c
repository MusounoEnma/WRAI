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
    printf(" [*] Kuantisasi     : INT8 Symmetric Row-wise\n");
    printf(" [*] Hardware Target: CPU (AVX 1.0 SIMD + OpenMP)\n");
    printf(" [*] Memory Buffer  : O(1) Constant (~14.5 MB State Buffer, 0%% KV-Cache)\n");
    printf(" [*] Mapping Mode   : Zero-Heap Virtual Memory-Mapped (mmap)\n");
    printf("=================================================================\n\n");
}

static const char* find_existing_file(const char* candidates[], int count) {
    for (int i = 0; i < count; i++) {
        if (candidates[i]) {
            FILE* f = fopen(candidates[i], "rb");
            if (f) {
                fclose(f);
                return candidates[i];
            }
        }
    }
    return NULL;
}

int main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif
    srand((unsigned int)time(NULL));

    const char* model_candidates[] = {
        argc >= 2 ? argv[1] : NULL,
        "qwen\\wrai_x_06b_int8.bin",
        "qwen/wrai_x_06b_int8.bin",
        "..\\qwen\\wrai_x_06b_int8.bin",
        "../qwen/wrai_x_06b_int8.bin",
        "final wrai\\qwen\\wrai_x_06b_int8.bin",
        "final wrai/qwen/wrai_x_06b_int8.bin",
        "models x\\wrai_x_06b_int8.bin",
        "models x/wrai_x_06b_int8.bin",
        "wrai_x_06b_int8.bin",
        "models\\wrai_x_06b_int8.bin",
        "models/wrai_x_06b_int8.bin"
    };
    const char* vocab_candidates[] = {
        argc >= 3 ? argv[2] : NULL,
        "qwen\\wrai_x_vocab.bin",
        "qwen/wrai_x_vocab.bin",
        "..\\qwen\\wrai_x_vocab.bin",
        "../qwen/wrai_x_vocab.bin",
        "final wrai\\qwen\\wrai_x_vocab.bin",
        "final wrai/qwen/wrai_x_vocab.bin",
        "models x\\wrai_x_vocab.bin",
        "models x/wrai_x_vocab.bin",
        "wrai_x_vocab.bin",
        "models\\wrai_x_vocab.bin",
        "models/wrai_x_vocab.bin"
    };

    const char* model_path = find_existing_file(model_candidates, sizeof(model_candidates) / sizeof(model_candidates[0]));
    const char* vocab_path = find_existing_file(vocab_candidates, sizeof(vocab_candidates) / sizeof(vocab_candidates[0]));

    if (!model_path || !vocab_path) {
        fprintf(stderr, "\n=================================================================\n");
        fprintf(stderr, " [ERROR] File model atau vocabulary WRAI-X tidak ditemukan!\n");
        fprintf(stderr, "=================================================================\n");
        if (!model_path) {
            fprintf(stderr, " Model dicari di jalur berikut:\n");
            for (int i = 0; i < (int)(sizeof(model_candidates) / sizeof(model_candidates[0])); i++) {
                if (model_candidates[i]) fprintf(stderr, "   - %s\n", model_candidates[i]);
            }
        }
        if (!vocab_path) {
            fprintf(stderr, " Vocab dicari di jalur berikut:\n");
            for (int i = 0; i < (int)(sizeof(vocab_candidates) / sizeof(vocab_candidates[0])); i++) {
                if (vocab_candidates[i]) fprintf(stderr, "   - %s\n", vocab_candidates[i]);
            }
        }
        fprintf(stderr, "\nPastikan folder 'models x' berisi wrai_x_06b_int8.bin dan wrai_x_vocab.bin.\n");
        fprintf(stderr, "\nTekan Enter untuk keluar...");
        getchar();
        return 1;
    }

    printf("[*] Memuat Model WRAI-X dari: %s...\n", model_path);
    wrai_x_model_t model;
    if (!wrai_x_load_model(model_path, &model)) {
        fprintf(stderr, "[ERROR] Gagal memuat file binary model WRAI-X (%s).\n", model_path);
        fprintf(stderr, "\nTekan Enter untuk keluar...");
        getchar();
        return 1;
    }
    printf("[OK] Model WRAI-X berhasil dimap! Ukuran file: %.2f MB\n", (double)model.file_size / (1024.0 * 1024.0));

    printf("[*] Memuat Vocabulary Tokenizer dari: %s...\n", vocab_path);
    wrai_x_tokenizer_t tok;
    if (!wrai_x_load_tokenizer(vocab_path, &tok)) {
        fprintf(stderr, "[ERROR] Gagal memuat file vocabulary binary (%s).\n", vocab_path);
        wrai_x_free_model(&model);
        fprintf(stderr, "\nTekan Enter untuk keluar...");
        getchar();
        return 1;
    }
    printf("[OK] Tokenizer Siap! Total Vocabulary: %u tokens (Index 256 Buckets aktif).\n", tok.num_tokens);

    wrai_x_state_t state;
    if (!wrai_x_state_init(&state)) {
        fprintf(stderr, "[ERROR] Gagal menginisialisasi buffer state WRAI-X.\n");
        wrai_x_free_tokenizer(&tok);
        wrai_x_free_model(&model);
        fprintf(stderr, "\nTekan Enter untuk keluar...");
        getchar();
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

        /* Format ChatML matching training distribution */
        char prompt[MAX_PROMPT_LEN + 256];
        snprintf(prompt, sizeof(prompt),
                 "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n",
                 user_input);

        int32_t prompt_tokens[1024];
        clock_t t_tok_start = clock();
        int num_prompt_tokens = wrai_x_tokenize(&tok, prompt, prompt_tokens, 1024);
        clock_t t_tok_end = clock();
        double tok_ms = ((double)(t_tok_end - t_tok_start) / CLOCKS_PER_SEC) * 1000.0;

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
        int32_t history[256];
        int history_len = 0;

        for (int step = 0; step < max_gen; step++) {
            int next_token = wrai_x_sample_advanced(
                logits,
                WRAI_X_VOCAB_SIZE,
                0.10f,   /* temperature (fokus tinggi) */
                0.90f,   /* top_p */
                40,      /* top_k */
                history,
                history_len,
                1.05f    /* rep_penalty */
            );

            if (step >= 2 && (next_token == tok.im_end_id || next_token == tok.eos_id)) break;

            if (next_token >= 0 && (uint32_t)next_token < tok.num_tokens) {
                wrai_x_detokenize_print(tok.token_strings[next_token]);
                fflush(stdout);
            }

            if (history_len < 256) {
                history[history_len++] = next_token;
            }
            gen_tokens_count++;

            wrai_x_forward_step(&model, &state, next_token, logits);
        }
        clock_t t_gen_end = clock();
        double gen_sec = (double)(t_gen_end - t_gen_start) / CLOCKS_PER_SEC;
        double tok_per_sec = (gen_sec > 0.0) ? ((double)gen_tokens_count / gen_sec) : 0.0;

        printf("\n\n[INFO] Tokenized in %.1fms | Prefill: %d tokens (%.2fs) | Gen: %d tokens in %.2fs (%.2f tok/s)\n\n",
               tok_ms, num_prompt_tokens, prefill_sec, gen_tokens_count, gen_sec, tok_per_sec);
    }

    free(logits);
    wrai_x_state_free(&state);
    wrai_x_free_tokenizer(&tok);
    wrai_x_free_model(&model);
    return 0;
}
