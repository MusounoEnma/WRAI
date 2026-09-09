/**
 * @file wrai_file_format.h
 * @brief Binary File Format Specification (.bin) for Wavelet Spectral Coefficients.
 *
 * Quantized model storage format designed for low-latency sequential SD Card streaming.
 * Strict byte alignment for bare-metal MCU memory mapping.
 */

#ifndef WRAI_FILE_FORMAT_H
#define WRAI_FILE_FORMAT_H

#include "wrai_types.h"
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#pragma pack(push, 1)

/**
 * @brief 64-Byte Binary Header at the start of WRAI .bin files
 */
typedef struct {
    uint32_t magic;                 /**< Magic Bytes: 0x57524149 ("WRAI") */
    uint16_t version;               /**< Specification Version (e.g. 0x0100 = v1.0) */
    uint16_t fft_size;              /**< FFT Window Size (e.g., 512) */
    uint16_t spectral_bins;         /**< Number of Spectral Bins per pattern (e.g., 256) */
    uint16_t quant_bits;            /**< Quantization Resolution (16 for Q15) */
    uint32_t num_patterns;          /**< Total number of pattern packets in file */
    uint32_t pattern_entry_bytes;   /**< Size in bytes of each pattern entry packet */
    uint8_t  reserved[44];          /**< Reserved bytes for 64-byte alignment */
} wrai_file_header_t;

/**
 * @brief Individual Quantized Pattern Packet (516 Bytes total for 256 bins)
 */
typedef struct {
    uint16_t pattern_id;                        /**< Target Response / Concept Token ID */
    uint16_t reserved;                          /**< Alignment Padding */
    q15_t    coeffs[WRAI_SPECTRAL_BINS];        /**< 256 Q15 Spectral Magnitude Coefficients */
} wrai_pattern_packet_t;

#pragma pack(pop)

/* ========================================================================= */
/* FILE FORMAT PARSER API                                                    */
/* ========================================================================= */

/**
 * @brief Validates WRAI Binary Header
 * @param header Pointer to header struct
 * @return true if valid WRAI binary header, false otherwise
 */
bool wrai_validate_header(const wrai_file_header_t* header);

/**
 * @brief Reads next pattern packet from open binary stream
 * @param file_handle Open file pointer
 * @param out_packet Destination pattern packet struct
 * @return true if successfully read, false on EOF or error
 */
bool wrai_read_pattern_packet(FILE* file_handle, wrai_pattern_packet_t* out_packet);

#ifdef __cplusplus
}
#endif

#endif /* WRAI_FILE_FORMAT_H */
