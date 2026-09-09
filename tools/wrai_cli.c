/**
 * @file wrai_cli.c
 * @brief High-Speed Standalone WRAI v14.4 Native Inference CLI & Interactive Chat Shell (AVX/SSE)
 */

#include "../src/wrai_v14_engine.h"
#include "../src/wrai_v14_tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <windows.h>

static double get_time_sec(void) {
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)freq.QuadPart;
}

void stream_response(
    const wrai_model_t* model,
    const wrai_tokenizer_t* tok,
    wrai_inference_state_t* state,
    const uint32_t* prompt_tokens,
    uint32_t prompt_len,
    uint32_t max_gen_tokens,
    float temperature,
    uint32_t top_k,
    float repetition_penalty
) {
    /* 1. Ingest prompt tokens */
    for (uint32_t i = 0; i < prompt_len; i++) {
        wrai_forward_step(model, state, prompt_tokens[i], state->logits);
    }

    uint32_t gen_history[512];
    uint32_t history_len = 0;
    for (uint32_t i = 0; i < prompt_len && history_len < 512; i++) {
        gen_history[history_len++] = prompt_tokens[i];
    }

    printf("\033[1;36mWRAI:\033[0m ");
    fflush(stdout);

    double t_start = get_time_sec();
    uint32_t tokens_generated = 0;

    for (uint32_t step = 0; step < max_gen_tokens; step++) {
        uint32_t next_token = wrai_sample_top_k(
            state->logits,
            model->vocab_size,
            temperature,
            top_k,
            gen_history,
            history_len,
            repetition_penalty
        );

        /* Stop condition on EOS */
        const char* str = wrai_tokenizer_decode_token(tok, next_token);
        if (next_token == tok->eos_token_id || 
            strcmp(str, "<|im_end|>") == 0 || 
            strcmp(str, "<|endoftext|>") == 0) {
            tokens_generated++;
            break;
        }

        printf("%s", str);
        fflush(stdout);
        tokens_generated++;

        if (history_len < 512) gen_history[history_len++] = next_token;

        wrai_forward_step(model, state, next_token, state->logits);
    }

    double elapsed = get_time_sec() - t_start;
    double speed = (double)tokens_generated / (elapsed > 0.001 ? elapsed : 0.001);

    printf("\n\033[90m[Stats: %u tokens | %.2f s | %.1f tok/s | Latency: %.1f ms/tok]\033[0m\n\n",
           tokens_generated, elapsed, speed, (elapsed / (tokens_generated > 0 ? tokens_generated : 1)) * 1000.0);
}

void run_benchmark_suite(const wrai_model_t* model, const wrai_tokenizer_t* tok) {
    printf("=================================================================\n");
    printf("   RUNNING WRAI v14.4 MULTI-LINGUAL BENCHMARK SUITE             \n");
    printf("=================================================================\n");

    const char* test_prompts[] = {
        "<ID> Halo, jelaskan apa fungsi dari AI",
        "<EN> Artificial Intelligence is defined as",
        "<PY> def calculate_area(radius):"
    };

    for (int p = 0; p < 3; p++) {
        printf("\n-----------------------------------------------------------------\n");
        printf("\033[1;33mPROMPT:\033[0m %s\n", test_prompts[p]);
        
        uint32_t pids[128];
        uint32_t plen = wrai_tokenizer_encode(tok, test_prompts[p], pids, 128);

        wrai_inference_state_t* bench_state = (wrai_inference_state_t*)calloc(1, sizeof(wrai_inference_state_t));
        wrai_state_reset(bench_state);

        stream_response(model, tok, bench_state, pids, plen, 50, 0.65f, 35, 1.2f);
        free(bench_state);
    }
}

/* Intelligent Automatic Language/Domain Tag Detector */
static const char* detect_auto_tag(const char* prompt) {
    if (!prompt || prompt[0] == '<') return ""; /* Explicit user tag like <ID>, <EN>, <PY> */

    /* 1. Python / Code pattern detection */
    const char* py_keywords[] = {
        "def ", "import ", "from ", "class ", "return ", "lambda ", "print(", 
        "for i in ", "while ", "elif ", "try:", "except:", "math.", "np.", "()", "[]", "->", "{}"
    };
    for (size_t i = 0; i < sizeof(py_keywords)/sizeof(py_keywords[0]); i++) {
        if (strstr(prompt, py_keywords[i])) return "<PY> ";
    }

    /* 2. Lowercase copy for NLP stopword scoring */
    char lower[1024];
    size_t len = strlen(prompt);
    if (len >= 1023) len = 1023;
    for (size_t i = 0; i < len; i++) {
        char c = prompt[i];
        lower[i] = (c >= 'A' && c <= 'Z') ? (c + 32) : c;
    }
    lower[len] = '\0';

    const char* id_words[] = {
        "yang", "dan", "di", "ini", "itu", "dengan", "untuk", "apa", "ada", 
        "dari", "sebuah", "bisa", "kamu", "saya", "mereka", "kita", "adalah", 
        "kenapa", "bagaimana", "siapa", "jelaskan", "tolong", "halo", "hai", 
        "selamat", "mengapa", "jika", "tapi", "seperti", "tentang", "membuat", 
        "apakah", "tidak", "bukan", "karena", "sudah", "akan", "tahu", "punya", "pagi", "siang", "malam"
    };

    const char* en_words[] = {
        "the", "is", "are", "of", "and", "in", "to", "that", "for", "with", 
        "what", "how", "why", "who", "where", "when", "can", "you", "we", 
        "they", "this", "these", "please", "explain", "write", "tell", 
        "about", "create", "have", "will", "would", "should", "not", "hello", "hi", "good"
    };

    int id_score = 0;
    for (size_t i = 0; i < sizeof(id_words)/sizeof(id_words[0]); i++) {
        if (strstr(lower, id_words[i])) id_score += 2;
    }

    int en_score = 0;
    for (size_t i = 0; i < sizeof(en_words)/sizeof(en_words[0]); i++) {
        if (strstr(lower, en_words[i])) en_score += 2;
    }

    if (en_score > id_score) {
        return "<EN> ";
    } else {
        return "<ID> "; /* Default Indonesian */
    }
}

void run_interactive_chat(const wrai_model_t* model, const wrai_tokenizer_t* tok) {
    printf("=================================================================\n");
    printf("   🌟 WRAI v14.4 INTERACTIVE CHAT SHELL (0%% KV-CACHE)          \n");
    printf("=================================================================\n");
    printf("  Features:\n");
    printf("    🤖 Auto-Language Tagging : Otomatis deteksi ID, EN, atau PY!\n");
    printf("    ⚡ 0%% KV-Cache Memory   : Konsumsi RAM tetap 8 KB SRAM konstan\n");
    printf("  Commands:\n");
    printf("    /reset       : Reset conversational hidden state\n");
    printf("    /temp <val>  : Set temperature (current: 0.65)\n");
    printf("    /topk <val>  : Set Top-K (current: 35)\n");
    printf("    /bench       : Run standard benchmark suite\n");
    printf("    /exit        : Exit chat shell\n");
    printf("-----------------------------------------------------------------\n\n");

    wrai_inference_state_t* chat_state = (wrai_inference_state_t*)calloc(1, sizeof(wrai_inference_state_t));
    wrai_state_reset(chat_state);

    float temperature = 0.65f;
    uint32_t top_k = 35;
    char input_line[1024];

    while (1) {
        printf("\033[1;32mUser:\033[0m ");
        fflush(stdout);

        if (!fgets(input_line, sizeof(input_line), stdin)) break;

        /* Remove trailing newline */
        size_t len = strlen(input_line);
        while (len > 0 && (input_line[len - 1] == '\n' || input_line[len - 1] == '\r')) {
            input_line[--len] = '\0';
        }

        if (len == 0) continue;

        /* Handle Commands */
        if (strcmp(input_line, "/exit") == 0 || strcmp(input_line, "/quit") == 0) {
            printf("[*] Exiting WRAI Chat Shell. Goodbye!\n");
            break;
        } else if (strcmp(input_line, "/reset") == 0) {
            wrai_state_reset(chat_state);
            printf("[OK] Conversation state reset to zero.\n\n");
            continue;
        } else if (strncmp(input_line, "/temp ", 6) == 0) {
            temperature = (float)atof(input_line + 6);
            printf("[OK] Temperature set to: %.2f\n\n", temperature);
            continue;
        } else if (strncmp(input_line, "/topk ", 6) == 0) {
            top_k = (uint32_t)atoi(input_line + 6);
            printf("[OK] Top-K set to: %u\n\n", top_k);
            continue;
        } else if (strcmp(input_line, "/bench") == 0) {
            run_benchmark_suite(model, tok);
            continue;
        }

        /* Auto-Detect Language/Domain Tag */
        const char* auto_tag = detect_auto_tag(input_line);
        char full_prompt[1200];
        if (strlen(auto_tag) > 0) {
            snprintf(full_prompt, sizeof(full_prompt), "%s%s", auto_tag, input_line);
            printf("\033[90m[Auto-Router -> %.*s]\033[0m\n", (int)(strlen(auto_tag) - 1), auto_tag);
        } else {
            snprintf(full_prompt, sizeof(full_prompt), "%s", input_line);
        }

        /* Tokenize User Prompt */
        uint32_t pids[256];
        uint32_t plen = wrai_tokenizer_encode(tok, full_prompt, pids, 256);
        if (plen == 0) {
            printf("[WARN] Could not tokenize prompt!\n\n");
            continue;
        }

        /* Stream Model Response */
        stream_response(model, tok, chat_state, pids, plen, 80, temperature, top_k, 1.2f);
    }

    free(chat_state);
}

int main(int argc, char* argv[]) {
    /* Set Windows Console to UTF-8 output */
    SetConsoleOutputCP(CP_UTF8);

    const char* model_path = "models/wrai_v14_4.bin";
    const char* vocab_path = "models/wrai_v14_4_vocab.bin";

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) model_path = argv[++i];
        if (strcmp(argv[i], "--vocab") == 0 && i + 1 < argc) vocab_path = argv[++i];
    }

    printf("[*] Loading WRAI Model Binary: %s...\n", model_path);
    double t_start = get_time_sec();
    
    wrai_model_t* model = (wrai_model_t*)calloc(1, sizeof(wrai_model_t));
    if (!model || !wrai_load_model_binary(model_path, model)) {
        printf("[FATAL] Could not load model binary: %s\n", model_path);
        if (model) free(model);
        return 1;
    }
    double load_time = (get_time_sec() - t_start) * 1000.0;
    printf("[OK 100%% SUCCESS] Model Loaded in %.2f ms (Epoch %u, Loss %.4f)!\n", load_time, model->epoch, model->best_val_loss);

    /* Load Tokenizer */
    wrai_tokenizer_t tok;
    if (!wrai_tokenizer_load(vocab_path, &tok)) {
        printf("[FATAL] Could not load vocab table: %s\n", vocab_path);
        wrai_free_model(model);
        free(model);
        return 1;
    }
    printf("[OK 100%% SUCCESS] Tokenizer Loaded (%u vocabulary tokens)!\n\n", tok.num_tokens);

    bool is_bench_mode = false;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--bench") == 0) is_bench_mode = true;
    }

    if (is_bench_mode) {
        run_benchmark_suite(model, &tok);
    } else {
        run_interactive_chat(model, &tok);
    }

    wrai_tokenizer_free(&tok);
    wrai_free_model(model);
    free(model);
    return 0;
}
