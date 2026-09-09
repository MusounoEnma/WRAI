/**
 * @file wrai_file_format.c
 * @brief Binary File Format Parser Implementation.
 */

#include "wrai_file_format.h"

bool wrai_validate_header(const wrai_file_header_t* header) {
    if (!header) return false;
    
    if (header->magic != WRAI_MAGIC_HEADER) {
        return false;
    }
    
    if (header->fft_size != WRAI_FFT_SIZE) {
        return false;
    }
    
    if (header->spectral_bins != WRAI_SPECTRAL_BINS) {
        return false;
    }
    
    if (header->quant_bits != 16) {
        return false;
    }

    return true;
}

bool wrai_read_pattern_packet(FILE* file_handle, wrai_pattern_packet_t* out_packet) {
    if (!file_handle || !out_packet) return false;

    size_t items_read = fread(out_packet, sizeof(wrai_pattern_packet_t), 1, file_handle);
    return (items_read == 1);
}
