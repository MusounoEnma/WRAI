/**
 * @file wrai_cot.c
 * @brief Implementasi Enhanced Chain-of-Thought (CoT) Wavelet Reasoning Engine.
 */

#include "wrai_cot.h"
#include <string.h>
#include <ctype.h>

void wrai_cot_graph_init(wrai_cot_knowledge_graph_t* graph) {
    if (!graph) return;
    memset(graph, 0, sizeof(wrai_cot_knowledge_graph_t));
}

bool wrai_cot_add_deduction(wrai_cot_knowledge_graph_t* graph,
                            const char* premise_a, uint16_t freq_a,
                            const char* premise_b, uint16_t freq_b,
                            const char* intermediate_deduction, uint16_t intermod_freq) {
    if (!graph || graph->rule_count >= WRAI_COT_MAX_RULES) return false;

    uint16_t idx = graph->rule_count;
    graph->rules[idx].rule_id = idx + 1;

    strncpy(graph->rules[idx].premise_a, premise_a, WRAI_COT_TEXT_LEN - 1);
    strncpy(graph->rules[idx].premise_b, premise_b, WRAI_COT_TEXT_LEN - 1);
    strncpy(graph->rules[idx].intermediate_deduction, intermediate_deduction, WRAI_COT_TEXT_LEN - 1);

    graph->rules[idx].freq_a = freq_a;
    graph->rules[idx].freq_b = freq_b;
    graph->rules[idx].intermod_freq = intermod_freq;

    graph->rule_count++;
    return true;
}

bool wrai_cot_execute_reasoning(const wrai_cot_knowledge_graph_t* graph,
                                const char* input_query,
                                wrai_cot_execution_trace_t* out_trace) {
    if (!graph || !input_query || !out_trace) return false;

    memset(out_trace, 0, sizeof(wrai_cot_execution_trace_t));

    /* Convert query to lowercase for keyword matching */
    char query_buf[256];
    strncpy(query_buf, input_query, sizeof(query_buf) - 1);
    for (int i = 0; query_buf[i]; i++) query_buf[i] = (char)tolower((unsigned char)query_buf[i]);

    uint8_t step_idx = 0;

    for (uint16_t r = 0; r < graph->rule_count; r++) {
        const wrai_cot_rule_t* rule = &graph->rules[r];

        /* Check if query keywords trigger premise A or premise B */
        char lower_p_a[128];
        strncpy(lower_p_a, rule->premise_a, 127);
        for (int i = 0; lower_p_a[i]; i++) lower_p_a[i] = (char)tolower((unsigned char)lower_p_a[i]);

        if (strstr(query_buf, lower_p_a) != NULL || strstr(query_buf, "mengapa") != NULL || strstr(query_buf, "bagaimana") != NULL || strstr(query_buf, "apakah") != NULL || strstr(query_buf, "hitung") != NULL || strstr(query_buf, "logic") != NULL) {
            if (step_idx < WRAI_COT_MAX_STEPS) {
                out_trace->steps[step_idx].step_number = step_idx + 1;
                snprintf(out_trace->steps[step_idx].description, WRAI_COT_TEXT_LEN,
                         "Mengevaluasi Premis: '%s' (f=%u Hz) x '%s' (f=%u Hz) -> Intermodulasi: f_reasoning=%u Hz",
                         rule->premise_a, rule->freq_a, rule->premise_b, rule->freq_b, rule->intermod_freq);
                out_trace->steps[step_idx].confidence_q15 = WRAI_Q15(0.92f);
                out_trace->steps[step_idx].harmonic_bin = rule->intermod_freq;
                step_idx++;

                /* Add intermediate deduction step */
                out_trace->steps[step_idx].step_number = step_idx + 1;
                snprintf(out_trace->steps[step_idx].description, WRAI_COT_TEXT_LEN,
                         "Deduksi Langkah #%u: %s", step_idx + 1, rule->intermediate_deduction);
                out_trace->steps[step_idx].confidence_q15 = WRAI_Q15(0.95f);
                out_trace->steps[step_idx].harmonic_bin = rule->intermod_freq + 5;
                step_idx++;

                strncpy(out_trace->final_conclusion, rule->intermediate_deduction, WRAI_COT_TEXT_LEN - 1);

                if (step_idx >= 4) break; // Limit steps per query
            }
        }
    }

    out_trace->total_steps = step_idx;
    return (step_idx > 0);
}
