/**
 * @file wrai_bananamind_engine.h
 * @brief High-Efficiency Standalone C Engine for BananaMind-2-Micro (2.93M) WRAI on ESP32
 * 
 * Features:
 *  - 0.00 Bytes dynamic heap allocation during decode.
 *  - Supports flash-memory-mapped weights via esp_partition_mmap().
 *  - Supports INT8 and INT4 packed quantized linear projections.
 *  - Bounded working state memory (~51 KB for INT8, ~100 KB for FP16, ~200 KB for FP32).
 */
#ifndef WRAI_BANANAMIND_ENGINE_H
#define WRAI_BANANAMIND_ENGINE_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BM2M_NUM_LAYERS         9
#define BM2M_HIDDEN_SIZE        128
#define BM2M_INTERMEDIATE_SIZE  512
#define BM2M_NUM_HEADS          4
#define BM2M_NUM_KV_HEADS       2
#define BM2M_HEAD_DIM           32
#define BM2M_VOCAB_SIZE         2048
#define BM2M_REFRESH_KERNEL     9

#define BM2M_SINK_SLOTS         4
#define BM2M_N_FINE             16
#define BM2M_HDC_SLOTS          8
#define BM2M_TOTAL_SLOTS        (BM2M_SINK_SLOTS + BM2M_N_FINE + BM2M_HDC_SLOTS) // 28

/* Tensor Data Types */
#define BM2M_DTYPE_FP32         0
#define BM2M_DTYPE_FP16         1
#define BM2M_DTYPE_INT8         2
#define BM2M_DTYPE_INT4_PACKED  3

#pragma pack(push, 1)
typedef struct {
    char     magic[16];          // "BM2M_MMAP_V1\0\0\0\0"
    uint32_t num_layers;
    uint32_t hidden_size;
    uint32_t intermediate_size;
    uint32_t num_heads;
    uint32_t num_kv_heads;
    uint32_t head_dim;
    uint32_t vocab_size;
    uint32_t num_tensors;
    uint32_t data_offset;
    uint32_t total_bytes;
    uint32_t format_type;
    uint8_t  reserved[4];
} bm2m_header_t;

typedef struct {
    char     name[64];
    uint32_t offset;
    uint32_t size_bytes;
    uint16_t shape[4];
    uint8_t  dtype;
    float    scale;
    uint8_t  reserved[43];
} bm2m_tensor_desc_t;
#pragma pack(pop)

/* Bounded Layer Recurrent State (Allocated per-layer in SRAM: ~16.8 KB each) */
typedef struct {
    float state_z[BM2M_NUM_KV_HEADS][BM2M_HEAD_DIM][BM2M_HEAD_DIM]; // 8192 bytes (FP32 invariant)
    
    // INT8 Quantized Slot Banks (Matches 146.25 KB working state budget)
    int8_t sink_k[BM2M_NUM_KV_HEADS][BM2M_SINK_SLOTS][BM2M_HEAD_DIM];
    int8_t sink_v[BM2M_NUM_KV_HEADS][BM2M_SINK_SLOTS][BM2M_HEAD_DIM];
    float  sink_k_scale[BM2M_NUM_KV_HEADS][BM2M_SINK_SLOTS];
    float  sink_v_scale[BM2M_NUM_KV_HEADS][BM2M_SINK_SLOTS];
    
    int8_t ring_k[BM2M_NUM_KV_HEADS][BM2M_N_FINE][BM2M_HEAD_DIM];
    int8_t ring_v[BM2M_NUM_KV_HEADS][BM2M_N_FINE][BM2M_HEAD_DIM];
    float  ring_k_scale[BM2M_NUM_KV_HEADS][BM2M_N_FINE];
    float  ring_v_scale[BM2M_NUM_KV_HEADS][BM2M_N_FINE];
    
    int8_t hdc_k[BM2M_NUM_KV_HEADS][BM2M_HDC_SLOTS][BM2M_HEAD_DIM];
    int8_t hdc_v[BM2M_NUM_KV_HEADS][BM2M_HDC_SLOTS][BM2M_HEAD_DIM];
    float  hdc_k_scale[BM2M_NUM_KV_HEADS][BM2M_HDC_SLOTS];
    float  hdc_v_scale[BM2M_NUM_KV_HEADS][BM2M_HDC_SLOTS];

    float conv_buf[BM2M_REFRESH_KERNEL][BM2M_HIDDEN_SIZE]; // 4608 bytes (FP32)
    uint16_t sink_cnt;
    uint16_t ring_cnt;
    uint16_t hdc_cnt;
    uint16_t ring_ptr;
    uint16_t hdc_ptr;
    uint32_t total_seen;
} bm2m_layer_state_t;

/* Entire Model Context (~19.5 KB scratch buffers in SRAM, zero stack footprint!) */
typedef struct {
    const uint8_t* mmap_base;           // Pointer to memory mapped flash
    const bm2m_header_t* header;
    const bm2m_tensor_desc_t* descs;
    bm2m_layer_state_t* layers[BM2M_NUM_LAYERS]; // Pointers to 9 individual layer states
    
    // Scratch working buffers for single token decode (Allocated in SRAM heap, eliminates stack overflow!)
    float hidden_states[BM2M_HIDDEN_SIZE];
    float orig_embedding[BM2M_HIDDEN_SIZE];
    float norm_buf[BM2M_HIDDEN_SIZE];
    float q_buf[BM2M_NUM_HEADS * BM2M_HEAD_DIM];
    float k_buf[BM2M_NUM_KV_HEADS * BM2M_HEAD_DIM];
    float v_buf[BM2M_NUM_KV_HEADS * BM2M_HEAD_DIM];
    float attn_out[BM2M_HIDDEN_SIZE];
    float refresh_out[BM2M_HIDDEN_SIZE];
    float y_rec[BM2M_HIDDEN_SIZE];
    float y_local[BM2M_HIDDEN_SIZE];
    float combined_y[BM2M_HIDDEN_SIZE];
    float ref_attn_sig[BM2M_HIDDEN_SIZE];
    float ref_emb_val[BM2M_HIDDEN_SIZE];
    float conv_out[BM2M_HIDDEN_SIZE];
    float gate_proj_out[BM2M_HIDDEN_SIZE];
    float val_proj_out[BM2M_HIDDEN_SIZE];
    float ref_gated[BM2M_HIDDEN_SIZE];
    float mlp_gate[BM2M_INTERMEDIATE_SIZE];
    float mlp_up[BM2M_INTERMEDIATE_SIZE];
    float mlp_down[BM2M_HIDDEN_SIZE];
    float logits[BM2M_VOCAB_SIZE];
    
    uint32_t current_step;
} bm2m_model_t;

/* Public API */
bool bm2m_init(bm2m_model_t* model, const uint8_t* mmap_base);
void bm2m_reset_states(bm2m_model_t* model);
int32_t bm2m_forward_step(bm2m_model_t* model, int32_t input_token_id);
int32_t bm2m_forward_step_with_penalty(
    bm2m_model_t* model,
    int32_t input_token_id,
    const int32_t* recent_tokens,
    int recent_count,
    float rep_penalty
);

#ifdef __cplusplus
}
#endif

#endif // WRAI_BANANAMIND_ENGINE_H
