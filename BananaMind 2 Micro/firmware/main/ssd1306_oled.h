/**
 * @file ssd1306_oled.h
 * @brief Ultra-Lightweight, Fail-Safe SSD1306 I2C OLED Driver (128x64) for ESP32
 *
 * Robust Design:
 *  - Non-blocking I2C probing: If no OLED is attached, the driver gracefully
 *    disables itself in <50ms without hanging the CPU.
 *  - Serial / UART always remains the primary output.
 *  - Framebuffer memory overhead: strictly 1,024 bytes (1 KB).
 */

#ifndef SSD1306_OLED_H
#define SSD1306_OLED_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Default I2C Pin Assignments (Standard ESP32: SDA=21, SCL=22; ESP32-S3: SDA=8, SCL=9)
#if CONFIG_IDF_TARGET_ESP32S3
#define OLED_DEFAULT_SDA_PIN    8
#define OLED_DEFAULT_SCL_PIN    9
#else
#define OLED_DEFAULT_SDA_PIN    21
#define OLED_DEFAULT_SCL_PIN    22
#endif

#define OLED_I2C_ADDR_PRIMARY   0x3C
#define OLED_I2C_ADDR_SECONDARY 0x3D

#define OLED_WIDTH              128
#define OLED_HEIGHT             64

/**
 * @brief Probes I2C bus and initializes SSD1306 if present.
 * @param sda_pin GPIO pin for SDA
 * @param scl_pin GPIO pin for SCL
 * @return true if OLED was detected and initialized, false if no OLED attached.
 */
bool oled_init(int sda_pin, int scl_pin);

/**
 * @brief Checks if OLED display is currently attached and active.
 */
bool oled_is_available(void);

/**
 * @brief Clears the internal 1KB framebuffer.
 */
void oled_clear(void);

/**
 * @brief Flushes framebuffer to physical OLED display via I2C.
 */
void oled_update(void);

/**
 * @brief Draws a string at specified pixel coordinates (6x8 font).
 */
void oled_draw_string(int x, int y, const char* str);

/**
 * @brief Streams a single character with automatic cursor tracking & line wrapping.
 */
void oled_put_char(char c);

/**
 * @brief Streams a null-terminated string with automatic cursor tracking.
 */
void oled_put_string(const char* str);

/**
 * @brief Scrolls display content up by one text line (8 pixels).
 */
void oled_scroll_up(void);

/**
 * @brief Sets cursor position for streaming text (col: 0..20, line: 0..7).
 */
void oled_set_cursor(int col, int line);

#ifdef __cplusplus
}
#endif

#endif // SSD1306_OLED_H
