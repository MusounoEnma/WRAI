/**
 * @file test_fase1_math.c
 * @brief Comprehensive Integration & Verification Suite for WRAI Architecture (Fases 1-5).
 *
 * Verifies Q15 LUT precision, Fixed-Point Radix-2 FFT, Token Superposition,
 * Binary Model Parsing, Peak Resonance Matching, and Double Ring-Buffer Swap.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <string.h>

#include "wrai_types.h"
#include "wrai_lut.h"
#include "wrai_math.h"
#include "wrai_encoder.h"
#include "wrai_file_format.h"
#include "wrai_ring_buffer.h"

#define PI_DOUBLE 3.14159265358979323846

static void test_q15_lut_precision(void) {
    printf("[TEST 1/5] Testing Q15 Trigonometric LUT Precision vs stdlib math...\n");
    int max_err_sin = 0;
    int max_err_cos = 0;

    for (uint32_t i = 0; i < WRAI_LUT_SIZE; i++) {
        double angle = (2.0 * PI_DOUBLE * i) / WRAI_LUT_SIZE;
        q15_t expected_sin = (q15_t)round(sin(angle) * 32768.0);
        q15_t expected_cos = (q15_t)round(cos(angle) * 32768.0);

        if (expected_sin > 32767) expected_sin = 32767;
        if (expected_sin < -32768) expected_sin = -32768;
        if (expected_cos > 32767) expected_cos = 32767;
        if (expected_cos < -32768) expected_cos = -32768;

        q15_t actual_sin = wrai_lut_sin(i);
        q15_t actual_cos = wrai_lut_cos(i);

        int err_s = abs(actual_sin - expected_sin);
        int err_c = abs(actual_cos - expected_cos);

        if (err_s > max_err_sin) max_err_sin = err_s;
        if (err_c > max_err_cos) max_err_cos = err_c;
    }

    printf("  -> Max Sin LUT Error: %d / 32768 (%.4f%%)\n", max_err_sin, (max_err_sin / 32768.0) * 100.0);
    printf("  -> Max Cos LUT Error: %d / 32768 (%.4f%%)\n", max_err_cos, (max_err_cos / 32768.0) * 100.0);
    assert(max_err_sin <= 2);
    assert(max_err_cos <= 2);
    printf("  [PASS] Q15 LUT Precision Verification Clean!\n\n");
}

static void test_token_superposition_and_fft(void) {
    printf("[TEST 2/5] Testing Token-to-Wave Encoding & Radix-2 Fixed-Point FFT...\n");

    wrai_token_wave_t tokens[2] = {
        { .token_id = 101, .freq_index = 20, .phase_shift = 0, .amplitude = WRAI_Q15(0.8f) },
        { .token_id = 102, .freq_index = 60, .phase_shift = 0, .amplitude = WRAI_Q15(0.5f) }
    };

    static wrai_spectral_frame_t frame;
    memset(&frame, 0, sizeof(frame));

    /* Generate Superposition Waveform */
    wrai_encode_tokens_to_wave(tokens, 2, frame.bins);

    /* Run In-Place Radix-2 Cooley-Tukey FFT */
    wrai_fft_q15_radix2(frame.bins, WRAI_FFT_SIZE);

    /* Compute Magnitudes */
    wrai_compute_magnitudes_q15(&frame);

    /* Verify Spectral Peak Bins (Bin 20 and Bin 60 should have max energy) */
    q15_t peak_bin_20 = frame.magnitude[20];
    q15_t peak_bin_60 = frame.magnitude[60];
    q15_t quiet_bin_50 = frame.magnitude[50];

    printf("  -> Magnitude Bin 20 (Target Peak 1): %d\n", peak_bin_20);
    printf("  -> Magnitude Bin 60 (Target Peak 2): %d\n", peak_bin_60);
    printf("  -> Magnitude Bin 50 (Quiet Noise):  %d\n", quiet_bin_50);

    assert(peak_bin_20 > quiet_bin_50 * 3);
    assert(peak_bin_60 > quiet_bin_50 * 2);
    printf("  [PASS] Fixed-Point Radix-2 FFT & Superposition Peak Detection Clean!\n\n");
}

static void test_binary_model_and_resonance(void) {
    printf("[TEST 3/5 & 4/5] Testing WRAI .bin Parser & Peak Resonance Detection...\n");

    const char* model_path = "tests/model_sample.bin";
    FILE* f = fopen(model_path, "rb");
    if (!f) {
        fprintf(stderr, "Error: Unable to open sample binary model: %s\n", model_path);
        exit(1);
    }

    wrai_file_header_t header;
    size_t r = fread(&header, sizeof(header), 1, f);
    assert(r == 1);

    bool valid = wrai_validate_header(&header);
    assert(valid == true);
    printf("  -> Valid WRAI Header Detected! Patterns: %u, Bins: %u\n", header.num_patterns, header.spectral_bins);

    wrai_pattern_packet_t patterns[10];
    for (uint32_t p = 0; p < header.num_patterns; p++) {
        bool ok = wrai_read_pattern_packet(f, &patterns[p]);
        assert(ok == true);
    }
    fclose(f);

    /* Create input spectrum with dominant peak at Bin 40 (Matching pattern_id = 2) */
    wrai_spectral_frame_t synthetic_frame;
    memset(&synthetic_frame, 0, sizeof(synthetic_frame));
    synthetic_frame.magnitude[40] = 30000;

    q31_t best_score = 0;
    uint16_t winning_id = wrai_detect_peak_resonance(&synthetic_frame, patterns, header.num_patterns, &best_score);

    printf("  -> Resonance Winner Pattern ID: %u (Expected: 2), Score: %d\n", winning_id, best_score);
    assert(winning_id == 2);
    printf("  [PASS] Binary Parsing & Spectral Resonance Search Clean!\n\n");
}

static void test_double_ring_buffer(void) {
    printf("[TEST 5/5] Testing Static Double Ring Buffer (2 x 8 KB SRAM)...\n");

    static wrai_double_ring_buffer_t ring_buf;
    wrai_ring_buffer_init(&ring_buf);

    uint8_t* cpu_ptr_1 = wrai_ring_buffer_get_cpu_ptr(&ring_buf);
    uint8_t* dma_ptr_1 = wrai_ring_buffer_get_dma_ptr(&ring_buf);

    assert(cpu_ptr_1 != dma_ptr_1);

    /* Write marker byte */
    cpu_ptr_1[0] = 0xAA;
    dma_ptr_1[0] = 0xBB;

    /* Ping-Pong Swap */
    wrai_ring_buffer_swap(&ring_buf);

    uint8_t* cpu_ptr_2 = wrai_ring_buffer_get_cpu_ptr(&ring_buf);
    uint8_t* dma_ptr_2 = wrai_ring_buffer_get_dma_ptr(&ring_buf);

    assert(cpu_ptr_2 == dma_ptr_1);
    assert(dma_ptr_2 == cpu_ptr_1);
    assert(cpu_ptr_2[0] == 0xBB);
    assert(dma_ptr_2[0] == 0xAA);

    printf("  -> Static Ring Buffer Memory Footprint: %zu Bytes (2 x %u KB)\n", sizeof(ring_buf), WRAI_RING_BUFFER_SIZE / 1024);
    printf("  [PASS] Double Ring-Buffer Ping-Pong Swap Clean!\n\n");
}

int main(void) {
    printf("=================================================================\n");
    printf("        WRAI ARCHITECTURE Bare-Metal C Verification Suite        \n");
    printf("=================================================================\n\n");

    test_q15_lut_precision();
    test_token_superposition_and_fft();
    test_binary_model_and_resonance();
    test_double_ring_buffer();

    printf("=================================================================\n");
    printf("  ALL 5 PHASES VERIFIED SUCCESSFULLY WITH ZERO COMPROMISES!     \n");
    printf("=================================================================\n");
    return 0;
}
