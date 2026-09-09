/**
 * @file wrai_ring_buffer.h
 * @brief Double-Buffering (Ping-Pong) Ring Buffer & DMA Sequential Streaming Interface.
 *
 * Designed for static 2 x 8 KB SRAM memory allocation to enable zero-wait CPU/DSP execution.
 */

#ifndef WRAI_RING_BUFFER_H
#define WRAI_RING_BUFFER_H

#include "wrai_types.h"
#include "wrai_file_format.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    WRAI_BUF_0 = 0,
    WRAI_BUF_1 = 1
} wrai_buffer_index_t;

/**
 * @brief 8 KB Single Ring Buffer Block
 */
typedef struct {
    uint8_t raw_data[WRAI_RING_BUFFER_SIZE];
    uint16_t bytes_valid;
    bool is_busy_dma;
} wrai_ring_block_t;

/**
 * @brief Double Ring Buffer Control Structure (Static 16 KB Total SRAM)
 */
typedef struct {
    wrai_ring_block_t   blocks[2];          /**< Ping-Pong Buffers (2 x 8 KB) */
    wrai_buffer_index_t active_cpu_buf;     /**< Currently being processed by CPU/DSP */
    wrai_buffer_index_t active_dma_buf;     /**< Currently being fetched by DMA */
    uint32_t            total_bytes_streamed;
} wrai_double_ring_buffer_t;

/**
 * @brief Initializes Double Ring Buffer state
 * @param ring_buf Pointer to static double ring buffer
 */
void wrai_ring_buffer_init(wrai_double_ring_buffer_t* ring_buf);

/**
 * @brief Swaps Ping-Pong Buffers (Called after CPU completes processing active frame)
 * @param ring_buf Pointer to double ring buffer
 */
void wrai_ring_buffer_swap(wrai_double_ring_buffer_t* ring_buf);

/**
 * @brief Returns pointer to current CPU processing buffer
 */
uint8_t* wrai_ring_buffer_get_cpu_ptr(wrai_double_ring_buffer_t* ring_buf);

/**
 * @brief Returns pointer to current DMA fetching buffer
 */
uint8_t* wrai_ring_buffer_get_dma_ptr(wrai_double_ring_buffer_t* ring_buf);

#ifdef __cplusplus
}
#endif

#endif /* WRAI_RING_BUFFER_H */
