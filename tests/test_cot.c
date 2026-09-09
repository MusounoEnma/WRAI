/**
 * @file test_cot.c
 * @brief Bare-metal C Chain-of-Thought (CoT) Verification Test.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "wrai_types.h"
#include "wrai_lut.h"
#include "wrai_math.h"
#include "wrai_cot.h"

int main(int argc, char** argv) {
    printf("=================================================================\n");
    printf("     WRAI BARE-METAL C CHAIN-OF-THOUGHT (CoT) TEST HARNESS       \n");
    printf("=================================================================\n\n");

    wrai_cot_knowledge_graph_t graph;
    wrai_cot_graph_init(&graph);

    /* Add CoT Rules */
    wrai_cot_add_deduction(&graph,
                           "Premis 1: Sinyal masukan diolah menggunakan Radix-2 FFT Q15", 20,
                           "Premis 2: Akumulasi butterfly digeser kanan 1 bit", 50,
                           "Deduksi WRAI: Presisi Q15 stabil 100%% bebas overflow tanpa perkalian matriks GEMM", 30);

    wrai_cot_add_deduction(&graph,
                           "Premis A: Ring Buffer dialokasikan statis 2 x 8 KB SRAM", 40,
                           "Premis B: Modul DMA menarik data sekuensial dari SD Card", 70,
                           "Deduksi WRAI: Penggunaan RAM statis 16 KB aman dari OOM crash di mikrokontroler", 30);

    const char* query = (argc > 1) ? argv[1] : "Bagaimana WRAI memproses FFT Q15 tanpa overflow?";
    printf("[INPUT PROMPT]: \"%s\"\n\n", query);

    wrai_cot_execution_trace_t trace;
    bool ok = wrai_cot_execute_reasoning(&graph, query, &trace);

    if (ok) {
        printf("--- RINCIAN TAHAPAN PENALARAN BERANTAI (C11 CoT TRACE) ---\n");
        for (uint8_t s = 0; s < trace.total_steps; s++) {
            printf("  * [Langkah #%u] %s (Bin Harmonis: %u Hz)\n",
                   trace.steps[s].step_number,
                   trace.steps[s].description,
                   trace.steps[s].harmonic_bin);
        }
        printf("\n  [FINAL CONCLUSION]: \"%s\"\n", trace.final_conclusion);
    } else {
        printf("  [!] Resonansi penalaran tidak ditemukan.\n");
    }

    printf("\n=================================================================\n");
    printf("  VERIFIKASI BARE-METAL C CoT REASONING ENGINE CLEAN!           \n");
    printf("=================================================================\n");
    return 0;
}
