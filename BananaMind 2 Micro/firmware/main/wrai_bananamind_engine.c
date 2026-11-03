#include "wrai_bananamind_engine.h"
#include <math.h>
#include <string.h>
#include <stdio.h>

#define EPS_RMS 1e-6f

/* Fast Math Helpers */
static inline float fast_silu(float x) {
    return x / (1.0f + expf(-x));
}

static inline float fast_elu(float x) {
    return (x > 0.0f) ? x : (expf(x) - 1.0f);
}

/* Linear Search for Tensor Descriptor by Name */
static const bm2m_tensor_desc_t* find_tensor(const bm2m_model_t* m, const char* name) {
    for (uint32_t i = 0; i < m->header->num_tensors; i++) {
        if (strcmp(m->descs[i].name, name) == 0) {
            return &m->descs[i];
        }
    }
    return NULL;
}

/* RMSNorm: out = (x / rms(x)) * w */
static void apply_rmsnorm(float* out, const float* in, const float* weight, int dim) {
    float sum_sq = 0.0f;
    for (int i = 0; i < dim; i++) {
        sum_sq += in[i] * in[i];
    }
    float inv_rms = 1.0f / sqrtf(sum_sq / (float)dim + EPS_RMS);
    for (int i = 0; i < dim; i++) {
        out[i] = (in[i] * inv_rms) * weight[i];
    }
}

/* Matrix-Vector Multiply: out = W * in */
static void gemv_forward(
    float* out,
    const float* in,
    const bm2m_tensor_desc_t* desc,
    const uint8_t* mmap_base,
    int rows,
    int cols
) {
    if (!desc) {
        memset(out, 0, rows * sizeof(float));
        return;
    }
    const uint8_t* data = mmap_base + desc->offset;
    float scale = desc->scale;

    if (desc->dtype == BM2M_DTYPE_FP32) {
        const float* W = (const float*)data;
        for (int r = 0; r < rows; r++) {
            float sum = 0.0f;
            const float* row = &W[r * cols];
            for (int c = 0; c < cols; c++) {
                sum += row[c] * in[c];
            }
            out[r] = sum;
        }
    } else if (desc->dtype == BM2M_DTYPE_INT8) {
        const int8_t* W = (const int8_t*)data;
        for (int r = 0; r < rows; r++) {
            float sum = 0.0f;
            const int8_t* row = &W[r * cols];
            for (int c = 0; c < cols; c++) {
                sum += (float)row[c] * in[c];
            }
            out[r] = sum * scale;
        }
    } else if (desc->dtype == BM2M_DTYPE_INT4_PACKED) {
        // INT4 packed: 2 values per byte (low nibble = val0, high nibble = val1)
        const uint8_t* W = (const uint8_t*)data;
        for (int r = 0; r < rows; r++) {
            float sum = 0.0f;
            const uint8_t* row = &W[(r * cols) / 2];
            for (int c = 0; c < cols; c += 2) {
                uint8_t byte_val = row[c / 2];
                int8_t v0 = (int8_t)(byte_val & 0x0F);
                if (v0 >= 8) v0 -= 16; // sign extend 4-bit
                int8_t v1 = (int8_t)((byte_val >> 4) & 0x0F);
                if (v1 >= 8) v1 -= 16;
                sum += (float)v0 * in[c];
                if (c + 1 < cols) {
                    sum += (float)v1 * in[c + 1];
                }
            }
            out[r] = sum * scale;
        }
    }
}

/* Helper: Symmetric INT8 Quantization with float scale */
static inline void quantize_vec_int8(int8_t* out, float* scale_out, const float* in, int len) {
    float max_abs = 0.0f;
    for (int i = 0; i < len; i++) {
        float a = fabsf(in[i]);
        if (a > max_abs) max_abs = a;
    }
    float scale = (max_abs > 1e-8f) ? (max_abs / 127.0f) : 1.0f;
    float inv_scale = 1.0f / scale;
    for (int i = 0; i < len; i++) {
        float v = in[i] * inv_scale;
        int q = (int)roundf(v);
        if (q > 127) q = 127;
        if (q < -127) q = -127;
        out[i] = (int8_t)q;
    }
    *scale_out = scale;
}

/* RoPE Embedding */
static void apply_rope(float* vec, int num_heads, int head_dim, uint32_t pos) {
    float theta_base = 100000.0f;
    for (int h = 0; h < num_heads; h++) {
        float* head_vec = &vec[h * head_dim];
        for (int i = 0; i < head_dim / 2; i++) {
            float freq = 1.0f / powf(theta_base, (float)(2 * i) / (float)head_dim);
            float angle = (float)pos * freq;
            float c = cosf(angle);
            float s = sinf(angle);
            float v_even = head_vec[2 * i];
            float v_odd = head_vec[2 * i + 1];
            head_vec[2 * i]     = v_even * c - v_odd * s;
            head_vec[2 * i + 1] = v_even * s + v_odd * c;
        }
    }
}

/* Initialize Model Engine */
bool bm2m_init(bm2m_model_t* model, const uint8_t* mmap_base) {
    if (!model || !mmap_base) return false;
    model->mmap_base = mmap_base;
    model->header = (const bm2m_header_t*)mmap_base;

    // Verify Magic
    if (memcmp(model->header->magic, "BM2M_MMAP_V1", 12) != 0) {
        return false;
    }
    model->descs = (const bm2m_tensor_desc_t*)(mmap_base + 64);
    bm2m_reset_states(model);
    return true;
}

void bm2m_reset_states(bm2m_model_t* model) {
    if (!model) return;
    for (int i = 0; i < BM2M_NUM_LAYERS; i++) {
        if (model->layers[i]) {
            memset(model->layers[i], 0, sizeof(bm2m_layer_state_t));
        }
    }
    model->current_step = 0;
}

/* Autoregressive Decode Step (Zero Dynamic Memory!) */
int32_t bm2m_forward_step_with_penalty(
    bm2m_model_t* model,
    int32_t input_token_id,
    const int32_t* recent_tokens,
    int recent_count,
    float rep_penalty
) {
    if (input_token_id < 0 || input_token_id >= BM2M_VOCAB_SIZE) {
        return 2; // EOS
    }
    const uint8_t* mmap = model->mmap_base;
    uint32_t pos = model->current_step;

    // 1. Embedding Lookup (transformer.wte.weight)
    const bm2m_tensor_desc_t* wte_desc = find_tensor(model, "transformer.wte.weight");
    if (!wte_desc) return 2;
    
    if (wte_desc->dtype == BM2M_DTYPE_INT8) {
        const int8_t* wte_data = (const int8_t*)(mmap + wte_desc->offset);
        const int8_t* emb_row = &wte_data[input_token_id * BM2M_HIDDEN_SIZE];
        for (int i = 0; i < BM2M_HIDDEN_SIZE; i++) {
            float val = (float)emb_row[i] * wte_desc->scale;
            model->hidden_states[i] = val;
            model->orig_embedding[i] = val;
        }
    } else {
        const float* wte_data = (const float*)(mmap + wte_desc->offset);
        const float* emb_row = &wte_data[input_token_id * BM2M_HIDDEN_SIZE];
        memcpy(model->hidden_states, emb_row, BM2M_HIDDEN_SIZE * sizeof(float));
        memcpy(model->orig_embedding, emb_row, BM2M_HIDDEN_SIZE * sizeof(float));
    }

    // 2. Loop Through All 9 Homogeneous Transformer Layers
    for (int l = 0; l < BM2M_NUM_LAYERS; l++) {
        bm2m_layer_state_t* st = model->layers[l];
        if (!st) return 2;
        char name_buf[64];

        // A. Input LayerNorm
        snprintf(name_buf, sizeof(name_buf), "transformer.h.%d.input_norm.weight", l);
        const bm2m_tensor_desc_t* in_norm_d = find_tensor(model, name_buf);
        const float* in_norm_w = (const float*)(mmap + in_norm_d->offset);
        apply_rmsnorm(model->norm_buf, model->hidden_states, in_norm_w, BM2M_HIDDEN_SIZE);

        // B. WRAI Attention (Q, K, V projections)
        snprintf(name_buf, sizeof(name_buf), "transformer.h.%d.attention.q_proj.weight", l);
        gemv_forward(model->q_buf, model->norm_buf, find_tensor(model, name_buf), mmap, 128, 128);
        snprintf(name_buf, sizeof(name_buf), "transformer.h.%d.attention.k_proj.weight", l);
        gemv_forward(model->k_buf, model->norm_buf, find_tensor(model, name_buf), mmap, 64, 128);
        snprintf(name_buf, sizeof(name_buf), "transformer.h.%d.attention.v_proj.weight", l);
        gemv_forward(model->v_buf, model->norm_buf, find_tensor(model, name_buf), mmap, 64, 128);

        // Q & K RMSNorm
        snprintf(name_buf, sizeof(name_buf), "transformer.h.%d.attention.q_norm.weight", l);
        const float* q_norm_w = (const float*)(mmap + find_tensor(model, name_buf)->offset);
        for (int h = 0; h < BM2M_NUM_HEADS; h++) {
            apply_rmsnorm(&model->q_buf[h * 32], &model->q_buf[h * 32], q_norm_w, 32);
        }
        snprintf(name_buf, sizeof(name_buf), "transformer.h.%d.attention.k_norm.weight", l);
        const float* k_norm_w = (const float*)(mmap + find_tensor(model, name_buf)->offset);
        for (int kv = 0; kv < BM2M_NUM_KV_HEADS; kv++) {
            apply_rmsnorm(&model->k_buf[kv * 32], &model->k_buf[kv * 32], k_norm_w, 32);
        }

        // Apply RoPE
        apply_rope(model->q_buf, BM2M_NUM_HEADS, BM2M_HEAD_DIM, pos);
        apply_rope(model->k_buf, BM2M_NUM_KV_HEADS, BM2M_HEAD_DIM, pos);

        // Update Recurrent Outer Product Z_t = gamma * Z_{t-1} + phi(k)^T * v
        float gamma = 0.95f;
        float alpha = 0.5f;
        for (int kv = 0; kv < BM2M_NUM_KV_HEADS; kv++) {
            float phi_k[32];
            for (int i = 0; i < 32; i++) {
                phi_k[i] = fast_elu(alpha * model->k_buf[kv * 32 + i]) + 1.0f;
            }
            const float* v_cur = &model->v_buf[kv * 32];
            for (int r = 0; r < 32; r++) {
                for (int c = 0; c < 32; c++) {
                    st->state_z[kv][r][c] = gamma * st->state_z[kv][r][c] + phi_k[r] * v_cur[c];
                }
            }
        }

        // Recurrent Output Y_rec = phi(q) * Z
        for (int h = 0; h < BM2M_NUM_HEADS; h++) {
            int kv = h / 2; // GQA 2:1
            float phi_q[32];
            for (int i = 0; i < 32; i++) {
                phi_q[i] = fast_elu(alpha * model->q_buf[h * 32 + i]) + 1.0f;
            }
            for (int c = 0; c < 32; c++) {
                float sum = 0.0f;
                for (int r = 0; r < 32; r++) {
                    sum += phi_q[r] * st->state_z[kv][r][c];
                }
                model->y_rec[h * 32 + c] = sum;
            }
        }

        // Local Attention over Bounded Ring Buffer + Current Token (Causal Self-Attention)
        int past_slots = st->sink_cnt + st->hdc_cnt + st->ring_cnt;
        int total_slots = past_slots + 1;

        for (int h = 0; h < BM2M_NUM_HEADS; h++) {
            int kv = h / 2;
            const float* q_head = &model->q_buf[h * 32];
            float scores[BM2M_TOTAL_SLOTS + 1];
            int s_idx = 0;
            
            // Score sinks (INT8)
            for (int s = 0; s < st->sink_cnt; s++) {
                float dot = 0.0f;
                for (int d = 0; d < 32; d++) dot += q_head[d] * (float)st->sink_k[kv][s][d];
                scores[s_idx++] = (dot * st->sink_k_scale[kv][s]) * 0.176776695f; // 1 / sqrt(32)
            }
            // Score hdc (INT8)
            for (int s = 0; s < st->hdc_cnt; s++) {
                float dot = 0.0f;
                for (int d = 0; d < 32; d++) dot += q_head[d] * (float)st->hdc_k[kv][s][d];
                scores[s_idx++] = (dot * st->hdc_k_scale[kv][s]) * 0.176776695f;
            }
            // Score ring (INT8)
            for (int s = 0; s < st->ring_cnt; s++) {
                float dot = 0.0f;
                for (int d = 0; d < 32; d++) dot += q_head[d] * (float)st->ring_k[kv][s][d];
                scores[s_idx++] = (dot * st->ring_k_scale[kv][s]) * 0.176776695f;
            }
            // Score Current Token (Self-Attention, FP32)
            const float* k_cur = &model->k_buf[kv * 32];
            float cur_dot = 0.0f;
            for (int d = 0; d < 32; d++) cur_dot += q_head[d] * k_cur[d];
            scores[s_idx++] = cur_dot * 0.176776695f;

            // Softmax
            float max_s = -1e9f;
            for (int s = 0; s < total_slots; s++) {
                if (scores[s] > max_s) max_s = scores[s];
            }
            float sum_exp = 0.0f;
            for (int s = 0; s < total_slots; s++) {
                scores[s] = expf(scores[s] - max_s);
                sum_exp += scores[s];
            }
            float inv_sum = (sum_exp > 1e-8f) ? (1.0f / sum_exp) : 0.0f;

            // Weighted sum over V (INT8 past slots + FP32 current token)
            const float* v_cur = &model->v_buf[kv * 32];
            for (int d = 0; d < 32; d++) {
                float val = 0.0f;
                int v_idx = 0;
                for (int s = 0; s < st->sink_cnt; s++) {
                    val += (scores[v_idx++] * inv_sum) * ((float)st->sink_v[kv][s][d] * st->sink_v_scale[kv][s]);
                }
                for (int s = 0; s < st->hdc_cnt; s++) {
                    val += (scores[v_idx++] * inv_sum) * ((float)st->hdc_v[kv][s][d] * st->hdc_v_scale[kv][s]);
                }
                for (int s = 0; s < st->ring_cnt; s++) {
                    val += (scores[v_idx++] * inv_sum) * ((float)st->ring_v[kv][s][d] * st->ring_v_scale[kv][s]);
                }
                // Add current token V
                val += (scores[v_idx++] * inv_sum) * v_cur[d];
                model->y_local[h * 32 + d] = val;
            }
        }

        // Add current token to slot banks for subsequent steps (quantized to INT8)
        if (st->sink_cnt < BM2M_SINK_SLOTS) {
            for (int kv = 0; kv < BM2M_NUM_KV_HEADS; kv++) {
                quantize_vec_int8(&st->sink_k[kv][st->sink_cnt][0], &st->sink_k_scale[kv][st->sink_cnt], &model->k_buf[kv * 32], 32);
                quantize_vec_int8(&st->sink_v[kv][st->sink_cnt][0], &st->sink_v_scale[kv][st->sink_cnt], &model->v_buf[kv * 32], 32);
            }
            st->sink_cnt++;
        } else if (st->ring_cnt < BM2M_N_FINE) {
            for (int kv = 0; kv < BM2M_NUM_KV_HEADS; kv++) {
                quantize_vec_int8(&st->ring_k[kv][st->ring_cnt][0], &st->ring_k_scale[kv][st->ring_cnt], &model->k_buf[kv * 32], 32);
                quantize_vec_int8(&st->ring_v[kv][st->ring_cnt][0], &st->ring_v_scale[kv][st->ring_cnt], &model->v_buf[kv * 32], 32);
            }
            st->ring_cnt++;
            st->ring_ptr = st->ring_cnt % BM2M_N_FINE;
        } else {
            // Evict ring buffer oldest token to HDC bank
            int ptr = st->ring_ptr;
            if (st->hdc_cnt < BM2M_HDC_SLOTS) {
                for (int kv = 0; kv < BM2M_NUM_KV_HEADS; kv++) {
                    memcpy(&st->hdc_k[kv][st->hdc_cnt][0], &st->ring_k[kv][ptr][0], 32);
                    memcpy(&st->hdc_v[kv][st->hdc_cnt][0], &st->ring_v[kv][ptr][0], 32);
                    st->hdc_k_scale[kv][st->hdc_cnt] = st->ring_k_scale[kv][ptr];
                    st->hdc_v_scale[kv][st->hdc_cnt] = st->ring_v_scale[kv][ptr];
                }
                st->hdc_cnt++;
            } else {
                for (int kv = 0; kv < BM2M_NUM_KV_HEADS; kv++) {
                    memcpy(&st->hdc_k[kv][st->hdc_ptr][0], &st->ring_k[kv][ptr][0], 32);
                    memcpy(&st->hdc_v[kv][st->hdc_ptr][0], &st->ring_v[kv][ptr][0], 32);
                    st->hdc_k_scale[kv][st->hdc_ptr] = st->ring_k_scale[kv][ptr];
                    st->hdc_v_scale[kv][st->hdc_ptr] = st->ring_v_scale[kv][ptr];
                }
                st->hdc_ptr = (st->hdc_ptr + 1) % BM2M_HDC_SLOTS;
            }
            // Put current token into ring buffer slot
            for (int kv = 0; kv < BM2M_NUM_KV_HEADS; kv++) {
                quantize_vec_int8(&st->ring_k[kv][ptr][0], &st->ring_k_scale[kv][ptr], &model->k_buf[kv * 32], 32);
                quantize_vec_int8(&st->ring_v[kv][ptr][0], &st->ring_v_scale[kv][ptr], &model->v_buf[kv * 32], 32);
            }
            st->ring_ptr = (st->ring_ptr + 1) % BM2M_N_FINE;
        }

        // Harmonic Output Blending (Reads wav_logit from model partition)
        snprintf(name_buf, sizeof(name_buf), "transformer.h.%d.attention.wav_logit", l);
        const bm2m_tensor_desc_t* wav_d = find_tensor(model, name_buf);
        const float* wav_l = wav_d ? (const float*)(mmap + wav_d->offset) : NULL;
        for (int h = 0; h < BM2M_NUM_HEADS; h++) {
            float w_mix = wav_l ? (1.0f / (1.0f + expf(-wav_l[h]))) : 1.0f;
            for (int d = 0; d < 32; d++) {
                int idx = h * 32 + d;
                model->combined_y[idx] = (1.0f - w_mix) * model->y_rec[idx] + w_mix * model->y_local[idx];
            }
        }

        // Output Projection: attn_out = combined_y * o_proj
        snprintf(name_buf, sizeof(name_buf), "transformer.h.%d.attention.o_proj.weight", l);
        gemv_forward(model->attn_out, model->combined_y, find_tensor(model, name_buf), mmap, 128, 128);

        // Residual Connection 1: hidden = hidden + attn_out
        for (int i = 0; i < 128; i++) {
            model->hidden_states[i] += model->attn_out[i];
        }

        // C. Refresh Gate Block
        snprintf(name_buf, sizeof(name_buf), "transformer.h.%d.refresh.attention_norm.weight", l);
        const float* ref_attn_norm_w = (const float*)(mmap + find_tensor(model, name_buf)->offset);
        apply_rmsnorm(model->ref_attn_sig, model->attn_out, ref_attn_norm_w, 128);

        snprintf(name_buf, sizeof(name_buf), "transformer.h.%d.refresh.embedding_norm.weight", l);
        const float* ref_emb_norm_w = (const float*)(mmap + find_tensor(model, name_buf)->offset);
        apply_rmsnorm(model->ref_emb_val, model->orig_embedding, ref_emb_norm_w, 128);

        // Update conv buffer (sliding window of 9)
        memmove(&st->conv_buf[0][0], &st->conv_buf[1][0], 8 * 128 * sizeof(float));
        memcpy(&st->conv_buf[8][0], model->ref_attn_sig, 128 * sizeof(float));

        // Depthwise conv1d
        snprintf(name_buf, sizeof(name_buf), "transformer.h.%d.refresh.depthwise_kernel", l);
        const float* dw_kernel = (const float*)(mmap + find_tensor(model, name_buf)->offset);
        for (int i = 0; i < 128; i++) {
            float sum = 0.0f;
            for (int k = 0; k < 9; k++) {
                sum += st->conv_buf[k][i] * dw_kernel[i * 9 + k];
            }
            model->conv_out[i] = sum;
        }

        // Gate projection + conv_out
        snprintf(name_buf, sizeof(name_buf), "transformer.h.%d.refresh.gate_proj.weight", l);
        gemv_forward(model->gate_proj_out, model->ref_attn_sig, find_tensor(model, name_buf), mmap, 128, 128);

        snprintf(name_buf, sizeof(name_buf), "transformer.h.%d.refresh.value_proj.weight", l);
        gemv_forward(model->val_proj_out, model->ref_emb_val, find_tensor(model, name_buf), mmap, 128, 128);

        for (int i = 0; i < 128; i++) {
            model->ref_gated[i] = fast_silu(model->gate_proj_out[i] + model->conv_out[i]) * model->val_proj_out[i];
        }

        snprintf(name_buf, sizeof(name_buf), "transformer.h.%d.refresh.out_proj.weight", l);
        gemv_forward(model->refresh_out, model->ref_gated, find_tensor(model, name_buf), mmap, 128, 128);

        snprintf(name_buf, sizeof(name_buf), "transformer.h.%d.refresh.output_norm.weight", l);
        const float* ref_out_norm_w = (const float*)(mmap + find_tensor(model, name_buf)->offset);
        apply_rmsnorm(model->refresh_out, model->refresh_out, ref_out_norm_w, 128);

        snprintf(name_buf, sizeof(name_buf), "transformer.h.%d.refresh.alpha", l);
        float ref_alpha = *(const float*)(mmap + find_tensor(model, name_buf)->offset);

        // Residual Connection 2: hidden = hidden + alpha * refresh_out
        for (int i = 0; i < 128; i++) {
            model->hidden_states[i] += ref_alpha * model->refresh_out[i];
        }

        // D. SwiGLU MLP
        snprintf(name_buf, sizeof(name_buf), "transformer.h.%d.post_attention_norm.weight", l);
        const float* post_norm_w = (const float*)(mmap + find_tensor(model, name_buf)->offset);
        apply_rmsnorm(model->norm_buf, model->hidden_states, post_norm_w, 128);

        snprintf(name_buf, sizeof(name_buf), "transformer.h.%d.mlp.gate_proj.weight", l);
        gemv_forward(model->mlp_gate, model->norm_buf, find_tensor(model, name_buf), mmap, 512, 128);

        snprintf(name_buf, sizeof(name_buf), "transformer.h.%d.mlp.up_proj.weight", l);
        gemv_forward(model->mlp_up, model->norm_buf, find_tensor(model, name_buf), mmap, 512, 128);

        // Compute intermediate SwiGLU in-place in mlp_gate to eliminate 2048 bytes of stack
        for (int i = 0; i < 512; i++) {
            model->mlp_gate[i] = fast_silu(model->mlp_gate[i]) * model->mlp_up[i];
        }

        snprintf(name_buf, sizeof(name_buf), "transformer.h.%d.mlp.down_proj.weight", l);
        gemv_forward(model->mlp_down, model->mlp_gate, find_tensor(model, name_buf), mmap, 128, 512);

        // Residual Connection 3: hidden = hidden + mlp_down
        for (int i = 0; i < 128; i++) {
            model->hidden_states[i] += model->mlp_down[i];
        }
    }

    // 3. Final LayerNorm (transformer.ln_f.weight)
    const bm2m_tensor_desc_t* ln_f_d = find_tensor(model, "transformer.ln_f.weight");
    const float* ln_f_w = (const float*)(mmap + ln_f_d->offset);
    apply_rmsnorm(model->norm_buf, model->hidden_states, ln_f_w, BM2M_HIDDEN_SIZE);

    // 4. LM Head (Tied to transformer.wte.weight)
    gemv_forward(model->logits, model->norm_buf, wte_desc, mmap, BM2M_VOCAB_SIZE, BM2M_HIDDEN_SIZE);

    // 5. Repetition Penalty Suppression (Prevents Attractor Loops)
    if (recent_tokens && recent_count > 0 && rep_penalty > 1.0f) {
        for (int r = 0; r < recent_count; r++) {
            int32_t tok = recent_tokens[r];
            if (tok >= 0 && tok < BM2M_VOCAB_SIZE) {
                if (model->logits[tok] > 0.0f) {
                    model->logits[tok] /= rep_penalty;
                } else {
                    model->logits[tok] *= rep_penalty;
                }
            }
        }
    }

    // 6. Greedy ArgMax Token Selection
    int32_t best_tok = 2; // EOS default
    float max_logit = -1e9f;
    for (int i = 0; i < BM2M_VOCAB_SIZE; i++) {
        if (model->logits[i] > max_logit) {
            max_logit = model->logits[i];
            best_tok = i;
        }
    }

    model->current_step++;
    return best_tok;
}

int32_t bm2m_forward_step(bm2m_model_t* model, int32_t input_token_id) {
    return bm2m_forward_step_with_penalty(model, input_token_id, NULL, 0, 1.0f);
}
