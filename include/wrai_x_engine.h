#ifndef WRAI_X_ENGINE_H
#define WRAI_X_ENGINE_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef _WIN32
#include <windows.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define WRAI_X_MAGIC           0x57524149  /* "WRAI" */
#define WRAI_X_VERSION         170         /* v17.0 (WRAI-X) */
#define WRAI_X_VOCAB_SIZE      151936
#define WRAI_X_HIDDEN_DIM      1024        /* D = 1024 (2^10 murni) */
#define WRAI_X_FFN_DIM         3072        /* F = 3072 */
#define WRAI_X_NUM_LAYERS      28
#define WRAI_X_NUM_HEADS       16
#define WRAI_X_HEAD_DIM        128
#define WRAI_X_WAVELET_LEVELS  4
#define WRAI_X_MAX_SEQ_LEN     512

/* Model Header (64 bytes) */
#pragma pack(push, 1)
typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t quant_type;      /* 1 = INT8 row-wise */
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
} wrai_x_header_t;
#pragma pack(pop)

/* Layer Weights referencing memory-mapped tensors */
typedef struct {
    const float*  rms_ret;
    const float*  rms_ffn;

    /* Memory Retention (Mt) */
    const float*  w_q_scales;    const int8_t* w_q_data;
    const float*  w_k_scales;    const int8_t* w_k_data;
    const float*  w_v_scales;    const int8_t* w_v_data;
    const float*  w_out_scales;  const int8_t* w_out_data;
    const float*  decay_m;
    const float*  decay_r;

    /* Haar Spectral Bridge */
    float low_gain, mid_gain, high_gain;
    const float*  haar_gate_w;
    const float*  haar_gate_b;

    /* Reasoning Retention (Rt) */
    const float*  w_qr_scales;   const int8_t* w_qr_data;
    const float*  w_kr_scales;   const int8_t* w_kr_data;
    const float*  w_vr_scales;   const int8_t* w_vr_data;
    const float*  w_out_r_scales;const int8_t* w_out_r_data;

    /* Thinking Gate */
    const float*  think_gate_w;
    const float*  think_gate_b;

    /* HDC Scratchpad */
    const float*  hdc_k_scales;  const int8_t* hdc_k_data;
    const float*  hdc_v_scales;  const int8_t* hdc_v_data;
    const float*  hdc_gate_w;
    const float*  hdc_gate_b;

    /* SwiGLU FFN */
    const float*  w_gate_scales; const int8_t* w_gate_data;
    const float*  w_up_scales;   const int8_t* w_up_data;
    const float*  w_down_scales; const int8_t* w_down_data;
} wrai_x_layer_weights_t;

typedef struct {
    wrai_x_header_t header;
    const float*    embed_scales;
    const int8_t*   embed_data;
    wrai_x_layer_weights_t layers[WRAI_X_NUM_LAYERS];
    const float*    ln_final;

#ifdef _WIN32
    HANDLE hFile;
    HANDLE hMapping;
    void*  mmap_base;
#else
    void*  mmap_base;
#endif
    size_t file_size;
} wrai_x_model_t;

/* Dual Recurrent State (~14.5 MB RAM total for 28 layers) */
typedef struct {
    float* state_m;     /* [28, 16, 128, 128] Memory Buffer */
    float* state_r;     /* [28, 16, 128, 128] Reasoning Buffer */
    float* state_hdc;   /* [28, 1024] HDC Scratchpad */
    size_t current_pos;
} wrai_x_state_t;

typedef struct {
    uint32_t num_tokens;
    char**   token_strings;
    uint16_t* token_lens;
    void*    fast_trie;
    int      im_start_id;
    int      im_end_id;
    int      eos_id;
} wrai_x_tokenizer_t;

/* API */
bool wrai_x_load_model(const char* bin_path, wrai_x_model_t* model);
void wrai_x_free_model(wrai_x_model_t* model);

bool wrai_x_state_init(wrai_x_state_t* state);
void wrai_x_state_reset(wrai_x_state_t* state);
void wrai_x_state_free(wrai_x_state_t* state);

bool wrai_x_load_tokenizer(const char* vocab_bin_path, wrai_x_tokenizer_t* tok);
void wrai_x_free_tokenizer(wrai_x_tokenizer_t* tok);
int  wrai_x_tokenize(const wrai_x_tokenizer_t* tok, const char* text, int32_t* out_ids, int max_tokens);

void wrai_x_forward_step(
    const wrai_x_model_t* model,
    wrai_x_state_t* state,
    int32_t token_id,
    float* out_logits
);

int wrai_x_sample(const float* logits, int vocab_size, float temperature, float top_p);

#ifdef __cplusplus
}
#endif

#endif /* WRAI_X_ENGINE_H */
