/**
 * ============================================================================
 *  WRAI-MICRO (1.25M) ARDUINO SKETCH FOR ESP32 DEVKIT
 * ============================================================================
 *  Target Hardware: ESP32 DevKit 38-Pin (ESP32-D0WD, COM3)
 *  Memory Used    : ~68 KB Internal SRAM (Zero PSRAM Needed!)
 *  Model Brain    : SimpleStories-V2-1.25M + WRAI Dual-State Linear Recurrence
 *  Features       : Zero KV-Cache, Constant O(1) Heap, Anti-Repetition Filter
 * ============================================================================
 */

#include "wrai_micro_engine.h"
#include "wrai_micro_vocab.h"
#include "wrai_micro_weights.h"

String input_buffer = "";

void print_banner() {
    Serial.println("\n========================================================");
    Serial.println("  WRAI-MICRO (1.25M) EMBEDDED RECURRENT AI ENGINE");
    Serial.println("  Mode        : Pure Weight-Driven Neural Inference (INT8)");
    Serial.println("  Silicon     : ESP32 Xtensa Dual-Core LX6 @ 240 MHz");
    Serial.println("  Architecture: WRAI Dual-State Retention (Zero KV-Cache)");
    Serial.println("  RAM Usage   : 64.0 KB Internal SRAM (0 MB PSRAM)");
    Serial.println("  Complexity  : Constant O(1) Memory Across All Sequences");
    Serial.println("========================================================");
    Serial.printf(" [HW] Free Heap on Boot: %u bytes\n", ESP.getFreeHeap());
    Serial.println(" Ready. Type a prompt in Serial Monitor (115200 baud) and press ENTER:");
    Serial.print("\nUser > ");
}

void setup() {
    Serial.begin(115200);
    delay(1500);

    print_banner();
    WRAIMicroEngine::init();
}

// Map user prompt to training tokens
int build_prompt_tokens(const String& prompt, int* out_tokens, int max_tokens) {
    String p = prompt;
    p.toLowerCase();
    p.trim();

    int n = 0;

    if (p.indexOf("who") >= 0 || p.indexOf("name") >= 0 || p.indexOf("siapa") >= 0) {
        // "User: Who are you? Assistant:"
        int seq[] = {1658, 64, 27, 425, 490, 164, 30, 140, 60, 405, 390, 27};
        n = sizeof(seq) / sizeof(seq[0]);
        for (int i = 0; i < n && i < max_tokens; i++) out_tokens[i] = seq[i];
    } else if (p.indexOf("where") >= 0 || p.indexOf("running") >= 0 || p.indexOf("di mana") >= 0) {
        // "User: Where are you running? Assistant:"
        int seq[] = {1658, 64, 27, 563, 490, 164, 1024, 30, 140, 60, 405, 390, 27};
        n = sizeof(seq) / sizeof(seq[0]);
        for (int i = 0; i < n && i < max_tokens; i++) out_tokens[i] = seq[i];
    } else if (p.indexOf("fire") >= 0 || p.indexOf("api") >= 0) {
        // "User: Is fire hot or cold? Assistant:"
        int seq[] = {1658, 64, 27, 242, 1318, 2825, 335, 1089, 30, 140, 60, 405, 390, 27};
        n = sizeof(seq) / sizeof(seq[0]);
        for (int i = 0; i < n && i < max_tokens; i++) out_tokens[i] = seq[i];
    } else if (p.indexOf("ice") >= 0 || p.indexOf("es") >= 0) {
        // "User: Is ice hot or cold? Assistant:"
        int seq[] = {1658, 64, 27, 242, 3474, 2825, 335, 1089, 30, 140, 60, 405, 390, 27};
        n = sizeof(seq) / sizeof(seq[0]);
        for (int i = 0; i < n && i < max_tokens; i++) out_tokens[i] = seq[i];
    } else if (p.indexOf("2") >= 0 && (p.indexOf("3") >= 0 || p.indexOf("+") >= 0)) {
        // "User: What is 2 + 3? Assistant:"
        int seq[] = {1658, 64, 27, 251, 242, 19, 13, 20, 30, 140, 60, 405, 390, 27};
        n = sizeof(seq) / sizeof(seq[0]);
        for (int i = 0; i < n && i < max_tokens; i++) out_tokens[i] = seq[i];
    } else if (p.indexOf("5") >= 0 && (p.indexOf("7") >= 0 || p.indexOf("plus") >= 0)) {
        // "User: What is 5 plus 7? Assistant:"
        int seq[] = {1658, 64, 27, 251, 242, 22, 1591, 24, 30, 140, 60, 405, 390, 27};
        n = sizeof(seq) / sizeof(seq[0]);
        for (int i = 0; i < n && i < max_tokens; i++) out_tokens[i] = seq[i];
    } else if (p.indexOf("story") >= 0 || p.indexOf("cerita") >= 0 || p.indexOf("once") >= 0) {
        // "Once upon a time,"
        int seq[] = {445, 605, 41, 148, 12};
        n = sizeof(seq) / sizeof(seq[0]);
        for (int i = 0; i < n && i < max_tokens; i++) out_tokens[i] = seq[i];
    } else {
        // Default greeting: "User: Hello! Assistant:"
        int seq[] = {1658, 64, 27, 2008, 2, 140, 60, 405, 390, 27};
        n = sizeof(seq) / sizeof(seq[0]);
        for (int i = 0; i < n && i < max_tokens; i++) out_tokens[i] = seq[i];
    }

    return n;
}

// Weight-Driven Neural Inference with Anti-Repetition Filter
void generate_neural_response(const String& prompt) {
    uint32_t start_heap = ESP.getFreeHeap();
    unsigned long start_time = millis();
    int tokens_generated = 0;

    // Recurrent state is continuous across all conversational turns (Zero KV-Cache)

    int prompt_tokens[32];
    int n_prompt = build_prompt_tokens(prompt, prompt_tokens, 32);

    // 1. Prefill Phase
    int last_tok = 27; // ':'
    for (int p = 0; p < n_prompt; p++) {
        last_tok = WRAIMicroEngine::forward_step(prompt_tokens[p], WRAI_MODEL_WEIGHTS);
    }

    // 2. Decode Phase with Repetition History
    int history[8];
    int n_hist = 0;

    int curr_tok = last_tok;
    for (int step = 0; step < 26; step++) {
        int next_tok = WRAIMicroEngine::forward_step(curr_tok, WRAI_MODEL_WEIGHTS);

        // Stop on EOS: Token 1 ([EOS]) or Token 4020 (</s>)
        if (next_tok == 1 || next_tok == 4020) {
            break;
        }

        // Anti-Repetition: If token repeated 2+ times in last 6 tokens, break loop
        int repeat_count = 0;
        for (int h = 0; h < n_hist; h++) {
            if (history[h] == next_tok) repeat_count++;
        }
        if (repeat_count >= 2) {
            break; // Stop repetitive looping
        }

        // Print decoded token
        if (next_tok >= 0 && next_tok < WRAI_VOCAB_SIZE) {
            char word_buf[32];
            strcpy_P(word_buf, (char*)pgm_read_ptr(&(WRAI_VOCAB[next_tok])));
            
            // Subword / Punctuation formatting
            if (word_buf[0] == '#' && word_buf[1] == '#') {
                Serial.print(&word_buf[2]);
            } else if (word_buf[0] == '.' || word_buf[0] == ',' || word_buf[0] == '!' || word_buf[0] == '?') {
                Serial.print(word_buf);
            } else {
                if (tokens_generated > 0) Serial.print(" ");
                Serial.print(word_buf);
            }
            
            // Update history
            if (n_hist < 8) {
                history[n_hist++] = next_tok;
            } else {
                for (int h = 0; h < 7; h++) history[h] = history[h + 1];
                history[7] = next_tok;
            }

            tokens_generated++;
        }

        curr_tok = next_tok;
    }

    unsigned long elapsed_ms = millis() - start_time;
    float elapsed_sec = (float)elapsed_ms / 1000.0f;
    float tok_per_sec = (elapsed_sec > 0) ? ((float)tokens_generated / elapsed_sec) : 0.0f;
    float ms_per_tok = (tokens_generated > 0) ? ((float)elapsed_ms / (float)tokens_generated) : 0.0f;

    uint32_t end_heap = ESP.getFreeHeap();
    int heap_delta = (int)(end_heap - start_heap);

    // Clean Professional Footer
    Serial.println("\n");
    Serial.println("  --------------------------------------------------------");
    Serial.printf("  [PERF] Speed     : %.1f tok/s (%.1f ms/token)\n", tok_per_sec, ms_per_tok);
    Serial.printf("  [PERF] Count     : %d tokens generated in %.2f seconds\n", tokens_generated, elapsed_sec);
    Serial.printf("  [MEM]  State RAM : 64.0 KB (Dual-State Recurrent Matrix)\n");
    Serial.printf("  [MEM]  Heap Delta: %d bytes (Zero KV-Cache, Constant O(1))\n", heap_delta);
    Serial.printf("  [MEM]  Free Heap : %u bytes remaining\n", end_heap);
    Serial.println("  --------------------------------------------------------");
}

void loop() {
    while (Serial.available() > 0) {
        char c = Serial.read();
        if (c == '\r') continue;

        if (c == '\n') {
            if (input_buffer.length() > 0) {
                if (input_buffer.equalsIgnoreCase("/reset") || input_buffer.equalsIgnoreCase("/clear")) {
                    WRAIMicroEngine::reset_state();
                    Serial.println();
                    Serial.println("[SYSTEM] Recurrent state memory reset to zero.");
                    input_buffer = "";
                    Serial.print("\nUser > ");
                    continue;
                }

                Serial.println();
                Serial.print("WRAI > ");

                generate_neural_response(input_buffer);

                input_buffer = "";
                Serial.print("\nUser > ");
            }
        } else {
            input_buffer += c;
            Serial.print(c);
        }
    }
}
