/**
 * @file wrai_reasoner.c
 * @brief Implementasi Multi-Step Wavelet Harmonic Intermodulation Reasoning Engine.
 */

#include "wrai_reasoner.h"
#include <string.h>

void wrai_reasoner_init(wrai_reasoning_graph_t* graph) {
    if (!graph) return;
    memset(graph, 0, sizeof(wrai_reasoning_graph_t));
}

bool wrai_reasoner_add_rule(wrai_reasoning_graph_t* graph, 
                           uint16_t premise_a, 
                           uint16_t premise_b, 
                           uint16_t conclusion) {
    if (!graph || graph->node_count >= 32) return false;

    uint16_t idx = graph->node_count;
    graph->nodes[idx].step_id       = idx + 1;
    graph->nodes[idx].premise_a_id  = premise_a;
    graph->nodes[idx].premise_b_id  = premise_b;
    graph->nodes[idx].conclusion_id = conclusion;

    /* Calculate intermodulation harmonic bin */
    int diff = (int)premise_a - (int)premise_b;
    if (diff < 0) diff = -diff;
    graph->nodes[idx].intermod_freq = (q15_t)(diff % WRAI_SPECTRAL_BINS);

    graph->node_count++;
    return true;
}

uint8_t wrai_reasoner_execute_cascade(const wrai_spectral_frame_t* input_frame,
                                     const wrai_reasoning_graph_t* graph,
                                     uint16_t* out_chain,
                                     uint8_t max_steps) {
    if (!input_frame || !graph || !out_chain || max_steps == 0) return 0;

    uint8_t derived_steps = 0;
    q15_t active_spectrum[WRAI_SPECTRAL_BINS];
    memcpy(active_spectrum, input_frame->magnitude, sizeof(active_spectrum));

    for (uint8_t step = 0; step < max_steps; step++) {
        bool rule_fired = false;

        for (uint16_t r = 0; r < graph->node_count; r++) {
            uint16_t bin_a = graph->nodes[r].premise_a_id % WRAI_SPECTRAL_BINS;
            uint16_t bin_b = graph->nodes[r].premise_b_id % WRAI_SPECTRAL_BINS;

            /* Check if energy exists at both premise frequency bins */
            if (active_spectrum[bin_a] > 1000 && active_spectrum[bin_b] > 1000) {
                uint16_t conc_bin = graph->nodes[r].conclusion_id % WRAI_SPECTRAL_BINS;

                /* Intermodulation coupling injects new harmonic energy into conclusion bin */
                active_spectrum[conc_bin] = wrai_q15_add_sat(active_spectrum[conc_bin], WRAI_Q15(0.85f));

                out_chain[derived_steps++] = graph->nodes[r].conclusion_id;
                rule_fired = true;

                if (derived_steps >= max_steps) break;
            }
        }

        if (!rule_fired) break;
    }

    return derived_steps;
}
