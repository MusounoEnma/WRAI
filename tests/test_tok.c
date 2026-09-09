#include "wrai_v16_engine.h"
#include <stdio.h>

int main() {
    wrai_v16_tokenizer_t tok;
    if (!wrai_v16_load_tokenizer("1.7b models/wrai_v16_vocab.bin", &tok)) {
        printf("Failed to load tokenizer\n");
        return 1;
    }
    const char* text = "<|im_start|>user\nHalo, siapa kamu?<|im_end|>\n<|im_start|>assistant\n";
    uint32_t tokens[256];
    int n = wrai_v16_tokenize(&tok, text, tokens, 256);
    printf("Token count: %d\n", n);
    for (int i = 0; i < n; i++) {
        printf("  Token %d: %d -> '%s'\n", i, tokens[i], wrai_v16_decode_token(&tok, tokens[i]));
    }
    wrai_v16_free_tokenizer(&tok);
    return 0;
}
