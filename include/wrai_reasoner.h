/**
 * @file wrai_reasoner.h
 * @brief Multi-Step Wavelet Harmonic Intermodulation & Reasoning Engine for WRAI.
 *
 * Implements Non-GEMM Chain-of-Thought Reasoning using Wavelet Harmonic Intermodulation
 * f_reasoning = |f_A +/- f_B| and Feedback Resonance Cascades.
 */

#ifndef WRAI_REASONER_H
#define WRAI_REASONER_H

#include "wrai_types.h"
#include "wrai_math.h"

#ifdef __cplusplus
extern "C" {
#endif

#define WRAI_MAX_REASONING_STEPS    5   /**< Maximum feedback cascade steps */

typedef struct {
    uint16_t step_id;
    uint16_t premise_a_id;
    uint16_t premise_b_id;
    uint16_t conclusion_id;
    q15_t    intermod_freq;
} wrai_reasoning_node_t;

typedef struct {
    wrai_reasoning_node_t nodes[32];
    uint16_t node_count;
} wrai_reasoning_graph_t;

/**
 * @brief Initializes Reasoning Graph
 */
void wrai_reasoner_init(wrai_reasoning_graph_t* graph);

/**
 * @brief Adds a Logical Premise Coupling rule: (Premise A + Premise B => Conclusion)
 */
bool wrai_reasoner_add_rule(wrai_reasoning_graph_t* graph, 
                           uint16_t premise_a, 
                           uint16_t premise_b, 
                           uint16_t conclusion);

/**
 * @brief Executes Multi-Step Harmonic Intermodulation Reasoning Cascade
 * @param input_frame Primary input spectral frame
 * @param graph Pointer to reasoning rules graph
 * @param out_chain Pointer to store extracted reasoning conclusion IDs
 * @param max_steps Max steps to cascade
 * @return Number of reasoning steps successfully derived
 */
uint8_t wrai_reasoner_execute_cascade(const wrai_spectral_frame_t* input_frame,
                                     const wrai_reasoning_graph_t* graph,
                                     uint16_t* out_chain,
                                     uint8_t max_steps);

#ifdef __cplusplus
}
#endif

#endif /* WRAI_REASONER_H */
