/**
 * @file wrai_cot.h
 * @brief Enhanced Chain-of-Thought (CoT) Multi-Step Wavelet Reasoning Engine for WRAI.
 *
 * Provides deep step-by-step harmonic intermodulation deduction traces (Step 1 -> Step 2 -> Step 3 -> Conclusion).
 * Zero-GEMM, Fixed-Point Q15 arithmetic, Static memory bounds.
 */

#ifndef WRAI_COT_H
#define WRAI_COT_H

#include "wrai_types.h"
#include "wrai_math.h"

#ifdef __cplusplus
extern "C" {
#endif

#define WRAI_COT_MAX_STEPS      8       /**< Max reasoning steps per chain */
#define WRAI_COT_MAX_RULES      64      /**< Max reasoning deduction rules */
#define WRAI_COT_TEXT_LEN       128     /**< Max text length for trace steps */

typedef struct {
    uint16_t rule_id;
    char     premise_a[WRAI_COT_TEXT_LEN];
    char     premise_b[WRAI_COT_TEXT_LEN];
    char     intermediate_deduction[WRAI_COT_TEXT_LEN];
    uint16_t freq_a;
    uint16_t freq_b;
    uint16_t intermod_freq;
} wrai_cot_rule_t;

typedef struct {
    uint8_t  step_number;
    char     description[WRAI_COT_TEXT_LEN];
    q15_t    confidence_q15;
    uint16_t harmonic_bin;
} wrai_cot_step_trace_t;

typedef struct {
    wrai_cot_step_trace_t steps[WRAI_COT_MAX_STEPS];
    uint8_t               total_steps;
    char                  final_conclusion[WRAI_COT_TEXT_LEN];
} wrai_cot_execution_trace_t;

typedef struct {
    wrai_cot_rule_t rules[WRAI_COT_MAX_RULES];
    uint16_t        rule_count;
} wrai_cot_knowledge_graph_t;

/**
 * @brief Initializes CoT Knowledge Graph
 */
void wrai_cot_graph_init(wrai_cot_knowledge_graph_t* graph);

/**
 * @brief Adds a multi-step deduction rule to graph
 */
bool wrai_cot_add_deduction(wrai_cot_knowledge_graph_t* graph,
                            const char* premise_a, uint16_t freq_a,
                            const char* premise_b, uint16_t freq_b,
                            const char* intermediate_deduction, uint16_t intermod_freq);

/**
 * @brief Performs Multi-Step Chain-of-Thought Reasoning Execution Trace
 */
bool wrai_cot_execute_reasoning(const wrai_cot_knowledge_graph_t* graph,
                                const char* input_query,
                                wrai_cot_execution_trace_t* out_trace);

#ifdef __cplusplus
}
#endif

#endif /* WRAI_COT_H */
