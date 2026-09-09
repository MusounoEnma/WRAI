/**
 * @file wrai_ring_buffer.c
 * @brief Implementasi Double-Buffering Ring Buffer & Logika Ping-Pong.
 */

#include "wrai_ring_buffer.h"
#include <string.h>

void wrai_ring_buffer_init(wrai_double_ring_buffer_t* ring_buf) {
    if (!ring_buf) return;

    memset(ring_buf, 0, sizeof(wrai_double_ring_buffer_t));
    ring_buf->active_cpu_buf = WRAI_BUF_0;
    ring_buf->active_dma_buf = WRAI_BUF_1;
    ring_buf->blocks[WRAI_BUF_0].is_busy_dma = false;
    ring_buf->blocks[WRAI_BUF_1].is_busy_dma = true;
}

void wrai_ring_buffer_swap(wrai_double_ring_buffer_t* ring_buf) {
    if (!ring_buf) return;

    wrai_buffer_index_t old_cpu = ring_buf->active_cpu_buf;
    wrai_buffer_index_t old_dma = ring_buf->active_dma_buf;

    /* Ping-Pong Swap */
    ring_buf->active_cpu_buf = old_dma;
    ring_buf->active_dma_buf = old_cpu;

    ring_buf->blocks[ring_buf->active_cpu_buf].is_busy_dma = false;
    ring_buf->blocks[ring_buf->active_dma_buf].is_busy_dma = true;
}

uint8_t* wrai_ring_buffer_get_cpu_ptr(wrai_double_ring_buffer_t* ring_buf) {
    if (!ring_buf) return NULL;
    return ring_buf->blocks[ring_buf->active_cpu_buf].raw_data;
}

uint8_t* wrai_ring_buffer_get_dma_ptr(wrai_double_ring_buffer_t* ring_buf) {
    if (!ring_buf) return NULL;
    return ring_buf->blocks[ring_buf->active_dma_buf].raw_data;
}
