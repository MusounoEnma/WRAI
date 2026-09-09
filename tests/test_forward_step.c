#include "wrai_v16_engine.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    printf("[*] Starting test_forward_step...\n");
    wrai_v16_model_t model;
    if (!wrai_v16_load_model("1.7b models/wrai_v16_1.7b_int8.bin", &model)) {
        printf("[FAIL] Load model failed\n");
        return 1;
    }
    printf("[OK] Model loaded\n");

    wrai_v16_state_t state;
    if (!wrai_v16_state_init(&state)) {
        printf("[FAIL] State init failed\n");
        return 1;
    }
    printf("[OK] State inited\n");

    float* logits = (float*)malloc(WRAI_V16_VOCAB_SIZE * sizeof(float));
    if (!logits) {
        printf("[FAIL] Logits alloc failed\n");
        return 1;
    }

    printf("[*] Running 1 forward step on token 151644...\n");
    fflush(stdout);
    wrai_v16_forward_step(&model, &state, 151644, logits);
    printf("[OK] Forward step completed! Logits[0]=%f, Logits[151644]=%f\n", logits[0], logits[151644]);

    // Find top 5 tokens
    for (int top = 0; top < 5; top++) {
        int best = 0;
        float max_val = -1e9;
        for (int i = 0; i < WRAI_V16_VOCAB_SIZE; i++) {
            if (logits[i] > max_val) {
                max_val = logits[i];
                best = i;
            }
        }
        printf("  Top %d: token %d (logit: %f)\n", top+1, best, max_val);
        logits[best] = -1e9;
    }

    free(logits);
    wrai_v16_state_free(&state);
    wrai_v16_free_model(&model);
    printf("[ALL SUCCESS] Test completed cleanly.\n");
    return 0;
}
