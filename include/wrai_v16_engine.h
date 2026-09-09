#ifndef WRAI_V16_ENGINE_H
#define WRAI_V16_ENGINE_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef _WIN32
#include <windows.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define WRAI_V16_MAGIC           0x57524149  /* "WRAI" */
#define WRAI_V16_VERSION         160         /* v16.0 */
#define WRAI_V16_VOCAB_SIZE      151936
#define WRAI_V16_HIDDEN_DIM      2048
#define WRAI_V16_FFN_DIM         6144
#define WRAI_V16_NUM_LAYERS      28
#define WRAI_V16_NUM_HEADS       16
#define WRAI_V16_HEAD_DIM        128
#define WRAI_V16_WAVELET_LEVELS  4
#define WRAI_V16_MAX_SEQ_LEN     256

/* Model Header Configuration (64 bytes aligned) */
#pragma pack(push, 1)
typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t quant_type;      /* 8 = INT8 row-wise, 4 = INT4 block-wise */
    uint32_t vocab_size;
    uint32_t hidden_dim;
    uint32_t ffn_dim;
    uint32_t num_layers;
    uint32_t num_heads;
    uint32_t head_dim;
    uint32_t wavelet_levels;
    uint32_t max_seq_len;
    float    loss;
    uint8_t  reserved[12];
} wrai_v16_header_t;
#pragma pack(pop)

/* Layer Weights referencing memory-mapped tensors (Zero Heap Copy!) */
typedef struct {
    const float*  rms_ret;
    const float*  w_q_scales;
    const int8_t* w_q_data;
    const float*  w_k_scales;
    const int8_t* w_k_data;
    const float*  w_v_scales;
    const int8_t* w_v_data;
    const float*  w_out_scales;
    const int8_t* w_out_data;

    const float*  decay_logit;
    const float*  group_norm_w;
    const float*  group_norm_b;

    const float*  rms_ffn;
    const float*  w_gate_scales;
    const int8_t* w_gate_data;
    const float*  w_up_scales;
    const int8_t* w_up_data;
    const float*  w_down_scales;
    const int8_t* w_down_data;
} wrai_v16_layer_weights_t;

/* Model Context */
typedef struct {
    wrai_v16_header_t header;

#ifdef _WIN32
    HANDLE hFile;
    HANDLE hMapping;
#endif
    const uint8_t* mmap_base;
    size_t file_size;

    const float*  pe;
    const float*  spec1_gw;
    const float*  spec1_gb;
    const float*  spec1_dg;
    float         spec1_ag;

    const float*  spec2_gw;
    const float*  spec2_gb;
    const float*  spec2_dg;
    float         spec2_ag;

    const float*  ln_final;

    const float*  embed_scales;
    const int8_t* embed_data;

    wrai_v16_layer_weights_t layers[WRAI_V16_NUM_LAYERS];
} wrai_v16_model_t;

/* O(1) Recurrent State Context (~28 MB Total for 1.7B Model) */
typedef struct {
    /* 28 layers * 16 heads * 128 * 128 floats */
    float* state_buffer;
    int current_pos;
} wrai_v16_state_t;

/* Tokenizer Structure */
typedef struct {
    uint32_t num_tokens;
    char** token_strings;
    uint16_t* token_lens;
    uint32_t im_start_id;
    uint32_t im_end_id;
    uint32_t eos_id;
    void* lookup_index;
} wrai_v16_tokenizer_t;

/* Sampling parameters */
typedef struct {
    float temperature;
    float top_p;
    int top_k;
    float repetition_penalty;
    int max_tokens;
} wrai_v16_sample_params_t;

/* Engine API Functions */
bool wrai_v16_load_model(const char* bin_path, wrai_v16_model_t* model);
void wrai_v16_free_model(wrai_v16_model_t* model);

bool wrai_v16_state_init(wrai_v16_state_t* state);
void wrai_v16_state_reset(wrai_v16_state_t* state);
void wrai_v16_state_free(wrai_v16_state_t* state);

bool wrai_v16_load_tokenizer(const char* vocab_bin_path, wrai_v16_tokenizer_t* tok);
void wrai_v16_free_tokenizer(wrai_v16_tokenizer_t* tok);

int wrai_v16_tokenize(const wrai_v16_tokenizer_t* tok, const char* text, uint32_t* tokens_out, int max_tokens);
const char* wrai_v16_decode_token(const wrai_v16_tokenizer_t* tok, uint32_t token_id);

/* Forward single token step: outputs logits for vocab when compute_logits is true */
void wrai_v16_forward_step(
    const wrai_v16_model_t* model,
    wrai_v16_state_t* state,
    uint32_t token_id,
    float* logits_out,
    bool compute_logits
);

/* Sample next token from logits */
uint32_t wrai_v16_sample_token(
    float* logits,
    int vocab_size,
    const uint32_t* history_tokens,
    int history_len,
    const wrai_v16_sample_params_t* params
);

#ifdef __cplusplus
}
#endif

#endif /* WRAI_V16_ENGINE_H */
