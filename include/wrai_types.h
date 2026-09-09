/**
 * @file wrai_types.h
 * @brief Core Fixed-Point Types, Sizing Constants, and Spectral Structures for WRAI.
 *
 * Designed for bare-metal execution on Cortex-M, ESP32, and AVX1 x86 CPUs.
 * Zero-GEMM architecture using Q15 (16-bit integer) and Q31 (32-bit integer).
 */

#ifndef WRAI_TYPES_H
#define WRAI_TYPES_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================= */
/* ARCHITECTURAL CONFIGURATION & MEMORY BOUNDS                               */
/* ========================================================================= */

#define WRAI_FFT_SIZE           512     /**< Default FFT Frame Size (Power of 2) */
#define WRAI_SPECTRAL_BINS      (WRAI_FFT_SIZE / 2) /**< Positive Frequency Bins (256) */
#define WRAI_LUT_SIZE           512     /**< Trigonometric Sine/Cosine LUT Size */
#define WRAI_LUT_MASK           (WRAI_LUT_SIZE - 1)

#define WRAI_MAX_SUPERPOSITION  16      /**< Max active superposition waves per frame */
#define WRAI_RING_BUFFER_SIZE   8192    /**< Size in bytes of single ring-buffer frame (8 KB) */

#define WRAI_MAGIC_HEADER       0x57524149 /**< Binary file magic byte "WRAI" */

/* ========================================================================= */
/* FIXED-POINT TYPEDEFS & ARITHMETIC MACROS                                  */
/* ========================================================================= */

typedef int16_t q15_t;  /**< Q1.15 Fixed-Point Format: range [-1.0, 0.999969] */
typedef int32_t q31_t;  /**< Q1.31 Fixed-Point Format: range [-1.0, 0.9999999995] */

/** Conversion from Float literal to Q15 (Compile-time / Generator use only) */
#define WRAI_Q15(x)             ((q15_t)((x) >= 0.9999 ? 32767 : ((x) <= -1.0 ? -32768 : (int32_t)((x) * 32768.0f))))

/** Fixed-Point Q15 Multiplication with 15-bit Right Shift */
#define WRAI_Q15_MUL(a, b)      ((q15_t)((((q31_t)(a)) * ((q31_t)(b))) >> 15))

/** Fixed-Point Q15 Addition with Saturation */
static inline q15_t wrai_q15_add_sat(q15_t a, q15_t b) {
    q31_t sum = (q31_t)a + (q31_t)b;
    if (sum > 32767) return 32767;
    if (sum < -32768) return -32768;
    return (q15_t)sum;
}

/** Fast Integer Absolute Value */
#define WRAI_ABS(x)             ((x) < 0 ? -(x) : (x))

/* ========================================================================= */
/* STRUCT DEFINITIONS                                                         */
/* ========================================================================= */

/**
 * @brief Representation of a Single Token mapped to Waveform Parameters
 */
typedef struct {
    uint16_t token_id;      /**< Unique Token Identifier */
    uint16_t freq_index;    /**< Normalized Frequency Index in LUT (0 .. WRAI_LUT_SIZE-1) */
    uint16_t phase_shift;   /**< Initial Phase Offset in LUT steps */
    q15_t    amplitude;     /**< Wave Amplitude in Q15 */
} wrai_token_wave_t;

/**
 * @brief Complex Spectral Point in Fixed-Point Format
 */
typedef struct {
    q15_t real;
    q15_t imag;
} wrai_complex_q15_t;

/**
 * @brief FFT Spectral Frame containing Frequency Bins and Estimated Magnitudes
 */
typedef struct {
    wrai_complex_q15_t bins[WRAI_FFT_SIZE];         /**< Full Complex FFT Output */
    q15_t              magnitude[WRAI_SPECTRAL_BINS];/**< Positive Bin Magnitudes */
} wrai_spectral_frame_t;

/**
 * @brief Quantized Wavelet Pattern for Resonance Matching
 */
typedef struct {
    uint16_t pattern_id;                            /**< ID of Target Response/Word */
    q15_t    coeffs[WRAI_SPECTRAL_BINS];            /**< Pre-calculated Quantized Spectral Pattern */
} wrai_spectrum_pattern_t;

#ifdef __cplusplus
}
#endif

#endif /* WRAI_TYPES_H */
