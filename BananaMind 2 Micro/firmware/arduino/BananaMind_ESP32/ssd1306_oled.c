/**
 * @file ssd1306_oled.c
 * @brief Fail-Safe SSD1306 128x64 OLED Driver Implementation for ESP32
 *
 * Implements non-blocking I2C detection (<50ms) with zero-cost headless fallback.
 * If no OLED is attached, all functions return immediately without holding the CPU
 * or interfering with the primary UART / Serial Monitor stream.
 */

#include "ssd1306_oled.h"
#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/i2c.h"
#include "esp_log.h"

static const char* TAG = "OLED";

#define I2C_MASTER_NUM              I2C_NUM_0
#define I2C_MASTER_FREQ_HZ          400000
#define I2C_MASTER_TX_BUF_DISABLE   0
#define I2C_MASTER_RX_BUF_DISABLE   0
#define I2C_TIMEOUT_MS              50

static bool s_oled_present = false;
static uint8_t s_oled_addr = OLED_I2C_ADDR_PRIMARY;
static uint8_t s_framebuffer[OLED_WIDTH * (OLED_HEIGHT / 8)]; // Strictly 1,024 bytes

static int s_cursor_col = 0;   // 0 .. 20 (each char is 6 pixels wide)
static int s_cursor_line = 0;  // 0 .. 7  (each line is 8 pixels high)

// Standard 5x7 ASCII font table (ASCII 32 to 126)
// Each character is 5 bytes wide, LSB at the top. The 6th column is implicit spacing (0x00).
static const uint8_t s_font5x7[][5] = {
    {0x00, 0x00, 0x00, 0x00, 0x00}, // 32 ' '
    {0x00, 0x00, 0x5F, 0x00, 0x00}, // 33 '!'
    {0x00, 0x07, 0x00, 0x07, 0x00}, // 34 '"'
    {0x14, 0x7F, 0x14, 0x7F, 0x14}, // 35 '#'
    {0x24, 0x2A, 0x7F, 0x2A, 0x12}, // 36 '$'
    {0x23, 0x13, 0x08, 0x64, 0x62}, // 37 '%'
    {0x36, 0x49, 0x55, 0x22, 0x50}, // 38 '&'
    {0x00, 0x05, 0x03, 0x00, 0x00}, // 39 '''
    {0x00, 0x1C, 0x22, 0x41, 0x00}, // 40 '('
    {0x00, 0x41, 0x22, 0x1C, 0x00}, // 41 ')'
    {0x14, 0x08, 0x3E, 0x08, 0x14}, // 42 '*'
    {0x08, 0x08, 0x3E, 0x08, 0x08}, // 43 '+'
    {0x00, 0x50, 0x30, 0x00, 0x00}, // 44 ','
    {0x08, 0x08, 0x08, 0x08, 0x08}, // 45 '-'
    {0x00, 0x60, 0x60, 0x00, 0x00}, // 46 '.'
    {0x20, 0x10, 0x08, 0x04, 0x02}, // 47 '/'
    {0x3E, 0x51, 0x49, 0x45, 0x3E}, // 48 '0'
    {0x00, 0x42, 0x7F, 0x40, 0x00}, // 49 '1'
    {0x42, 0x61, 0x51, 0x49, 0x46}, // 50 '2'
    {0x21, 0x41, 0x45, 0x4B, 0x31}, // 51 '3'
    {0x18, 0x14, 0x12, 0x7F, 0x10}, // 52 '4'
    {0x27, 0x45, 0x45, 0x45, 0x39}, // 53 '5'
    {0x3C, 0x4A, 0x49, 0x49, 0x30}, // 54 '6'
    {0x01, 0x71, 0x09, 0x05, 0x03}, // 55 '7'
    {0x36, 0x49, 0x49, 0x49, 0x36}, // 56 '8'
    {0x06, 0x49, 0x49, 0x29, 0x1E}, // 57 '9'
    {0x00, 0x36, 0x36, 0x00, 0x00}, // 58 ':'
    {0x00, 0x56, 0x36, 0x00, 0x00}, // 59 ';'
    {0x08, 0x14, 0x22, 0x41, 0x00}, // 60 '<'
    {0x14, 0x14, 0x14, 0x14, 0x14}, // 61 '='
    {0x00, 0x41, 0x22, 0x14, 0x08}, // 62 '>'
    {0x02, 0x01, 0x51, 0x09, 0x06}, // 63 '?'
    {0x32, 0x49, 0x79, 0x41, 0x3E}, // 64 '@'
    {0x7E, 0x11, 0x11, 0x11, 0x7E}, // 65 'A'
    {0x7F, 0x49, 0x49, 0x49, 0x36}, // 66 'B'
    {0x3E, 0x41, 0x41, 0x41, 0x22}, // 67 'C'
    {0x7F, 0x41, 0x41, 0x22, 0x1C}, // 68 'D'
    {0x7F, 0x49, 0x49, 0x49, 0x41}, // 69 'E'
    {0x7F, 0x09, 0x09, 0x09, 0x01}, // 70 'F'
    {0x3E, 0x41, 0x49, 0x49, 0x7A}, // 71 'G'
    {0x7F, 0x08, 0x08, 0x08, 0x7F}, // 72 'H'
    {0x00, 0x41, 0x7F, 0x41, 0x00}, // 73 'I'
    {0x20, 0x40, 0x41, 0x3F, 0x01}, // 74 'J'
    {0x7F, 0x08, 0x14, 0x22, 0x41}, // 75 'K'
    {0x7F, 0x40, 0x40, 0x40, 0x40}, // 76 'L'
    {0x7F, 0x02, 0x04, 0x02, 0x7F}, // 77 'M'
    {0x7F, 0x04, 0x08, 0x10, 0x7F}, // 78 'N'
    {0x3E, 0x41, 0x41, 0x41, 0x3E}, // 79 'O'
    {0x7F, 0x09, 0x09, 0x09, 0x06}, // 80 'P'
    {0x3E, 0x41, 0x51, 0x21, 0x5E}, // 81 'Q'
    {0x7F, 0x09, 0x19, 0x29, 0x46}, // 82 'R'
    {0x46, 0x49, 0x49, 0x49, 0x31}, // 83 'S'
    {0x01, 0x01, 0x7F, 0x01, 0x01}, // 84 'T'
    {0x3F, 0x40, 0x40, 0x40, 0x3F}, // 85 'U'
    {0x1F, 0x20, 0x40, 0x20, 0x1F}, // 86 'V'
    {0x3F, 0x40, 0x38, 0x40, 0x3F}, // 87 'W'
    {0x63, 0x14, 0x08, 0x14, 0x63}, // 88 'X'
    {0x07, 0x08, 0x70, 0x08, 0x07}, // 89 'Y'
    {0x61, 0x51, 0x49, 0x45, 0x43}, // 90 'Z'
    {0x00, 0x7F, 0x41, 0x41, 0x00}, // 91 '['
    {0x02, 0x04, 0x08, 0x10, 0x20}, // 92 '\'
    {0x00, 0x41, 0x41, 0x7F, 0x00}, // 93 ']'
    {0x04, 0x02, 0x01, 0x02, 0x04}, // 94 '^'
    {0x40, 0x40, 0x40, 0x40, 0x40}, // 95 '_'
    {0x00, 0x01, 0x02, 0x04, 0x00}, // 96 '`'
    {0x20, 0x54, 0x54, 0x54, 0x78}, // 97 'a'
    {0x7F, 0x48, 0x44, 0x44, 0x38}, // 98 'b'
    {0x38, 0x44, 0x44, 0x44, 0x20}, // 99 'c'
    {0x38, 0x44, 0x44, 0x48, 0x7F}, // 100 'd'
    {0x38, 0x54, 0x54, 0x54, 0x18}, // 101 'e'
    {0x08, 0x7E, 0x09, 0x01, 0x02}, // 102 'f'
    {0x0C, 0x52, 0x52, 0x52, 0x3E}, // 103 'g'
    {0x7F, 0x08, 0x04, 0x04, 0x78}, // 104 'h'
    {0x00, 0x44, 0x7D, 0x40, 0x00}, // 105 'i'
    {0x20, 0x40, 0x44, 0x3D, 0x00}, // 106 'j'
    {0x7F, 0x10, 0x28, 0x44, 0x00}, // 107 'k'
    {0x00, 0x41, 0x7F, 0x40, 0x00}, // 108 'l'
    {0x7C, 0x04, 0x18, 0x04, 0x78}, // 109 'm'
    {0x7C, 0x08, 0x04, 0x04, 0x78}, // 110 'n'
    {0x38, 0x44, 0x44, 0x44, 0x38}, // 111 'o'
    {0x7C, 0x14, 0x14, 0x14, 0x08}, // 112 'p'
    {0x08, 0x14, 0x14, 0x18, 0x7C}, // 113 'q'
    {0x7C, 0x08, 0x04, 0x04, 0x08}, // 114 'r'
    {0x48, 0x54, 0x54, 0x54, 0x20}, // 115 's'
    {0x04, 0x3F, 0x44, 0x40, 0x20}, // 116 't'
    {0x3C, 0x40, 0x40, 0x20, 0x7C}, // 117 'u'
    {0x1C, 0x20, 0x40, 0x20, 0x1C}, // 118 'v'
    {0x3C, 0x40, 0x30, 0x40, 0x3C}, // 119 'w'
    {0x44, 0x28, 0x10, 0x28, 0x44}, // 120 'x'
    {0x0C, 0x50, 0x50, 0x50, 0x3C}, // 121 'y'
    {0x44, 0x64, 0x54, 0x4C, 0x44}, // 122 'z'
    {0x00, 0x08, 0x36, 0x41, 0x00}, // 123 '{'
    {0x00, 0x00, 0x7F, 0x00, 0x00}, // 124 '|'
    {0x00, 0x41, 0x36, 0x08, 0x00}, // 125 '}'
    {0x08, 0x08, 0x2A, 0x1C, 0x08}  // 126 '~'
};

// Internal I2C Helper to send a single command byte
static esp_err_t oled_write_cmd(uint8_t cmd) {
    if (!s_oled_present) return ESP_ERR_NOT_FOUND;
    i2c_cmd_handle_t link = i2c_cmd_link_create();
    i2c_master_start(link);
    i2c_master_write_byte(link, (s_oled_addr << 1) | I2C_MASTER_WRITE, true);
    i2c_master_write_byte(link, 0x00, true); // Co = 0, D/C# = 0 (Command)
    i2c_master_write_byte(link, cmd, true);
    i2c_master_stop(link);
    esp_err_t ret = i2c_master_cmd_begin(I2C_MASTER_NUM, link, pdMS_TO_TICKS(I2C_TIMEOUT_MS));
    i2c_cmd_link_delete(link);
    return ret;
}

// Internal I2C Helper to send a stream of commands
static esp_err_t oled_write_cmd_list(const uint8_t* cmds, size_t count) {
    if (!s_oled_present) return ESP_ERR_NOT_FOUND;
    for (size_t i = 0; i < count; i++) {
        esp_err_t err = oled_write_cmd(cmds[i]);
        if (err != ESP_OK) return err;
    }
    return ESP_OK;
}

// Probe an I2C address with a quick 50ms ACK test
static bool oled_probe_address(uint8_t addr) {
    i2c_cmd_handle_t link = i2c_cmd_link_create();
    i2c_master_start(link);
    i2c_master_write_byte(link, (addr << 1) | I2C_MASTER_WRITE, true);
    i2c_master_stop(link);
    esp_err_t ret = i2c_master_cmd_begin(I2C_MASTER_NUM, link, pdMS_TO_TICKS(I2C_TIMEOUT_MS));
    i2c_cmd_link_delete(link);
    return (ret == ESP_OK);
}

bool oled_init(int sda_pin, int scl_pin) {
    s_oled_present = false;
    memset(s_framebuffer, 0x00, sizeof(s_framebuffer));
    s_cursor_col = 0;
    s_cursor_line = 0;

    // 1. Configure I2C Master bus
    i2c_config_t conf = {
        .mode = I2C_MODE_MASTER,
        .sda_io_num = sda_pin,
        .sda_pullup_en = GPIO_PULLUP_ENABLE,
        .scl_io_num = scl_pin,
        .scl_pullup_en = GPIO_PULLUP_ENABLE,
        .master.clk_speed = I2C_MASTER_FREQ_HZ,
    };
    esp_err_t err = i2c_param_config(I2C_MASTER_NUM, &conf);
    if (err != ESP_OK) {
        printf("[OLED] Failed to configure I2C parameters (0x%x). Operating in serial-only mode.\n", err);
        return false;
    }

    err = i2c_driver_install(I2C_MASTER_NUM, conf.mode, I2C_MASTER_RX_BUF_DISABLE, I2C_MASTER_TX_BUF_DISABLE, 0);
    if (err != ESP_OK) {
        printf("[OLED] Failed to install I2C driver (0x%x). Operating in serial-only mode.\n", err);
        return false;
    }

    // 2. Active non-blocking probe on primary (0x3C) and secondary (0x3D) addresses
    if (oled_probe_address(OLED_I2C_ADDR_PRIMARY)) {
        s_oled_addr = OLED_I2C_ADDR_PRIMARY;
        s_oled_present = true;
    } else if (oled_probe_address(OLED_I2C_ADDR_SECONDARY)) {
        s_oled_addr = OLED_I2C_ADDR_SECONDARY;
        s_oled_present = true;
    } else {
        // OLED is NOT connected. Gracefully tear down driver so pins are freed and no CPU time is wasted.
        i2c_driver_delete(I2C_MASTER_NUM);
        s_oled_present = false;
        printf("[OLED] SSD1306 not detected on I2C bus (SDA: GPIO %d, SCL: GPIO %d).\n", sda_pin, scl_pin);
        printf("[OLED] Graceful fallback enabled: All output streaming directly to Serial Monitor.\n");
        return false;
    }

    printf("[OLED] SSD1306 detected at 0x%02X! Initializing 128x64 display panel...\n", s_oled_addr);

    // 3. SSD1306 Standard Initialization Command Sequence
    const uint8_t init_cmds[] = {
        0xAE,        // Display OFF
        0xD5, 0x80,  // Set Display Clock Divide Ratio / Oscillator Frequency
        0xA8, 0x3F,  // Set Multiplex Ratio (64 lines)
        0xD3, 0x00,  // Set Display Offset (0)
        0x40,        // Set Start Line (line 0)
        0x8D, 0x14,  // Enable Internal Charge Pump (VCC generated by internal DC/DC)
        0x20, 0x00,  // Set Memory Addressing Mode to Horizontal Addressing Mode
        0xA1,        // Set Segment Re-map (0xA1 = column 127 is mapped to SEG0)
        0xC8,        // Set COM Output Scan Direction (0xC8 = remapped mode)
        0xDA, 0x12,  // Set COM Pins Hardware Configuration
        0x81, 0xCF,  // Set Contrast Control (0xCF)
        0xD9, 0xF1,  // Set Pre-charge Period
        0xDB, 0x40,  // Set VCOMH Deselect Level
        0xA4,        // Entire Display ON (resume to RAM content)
        0xA6,        // Set Normal Display (0 = off, 1 = on)
        0xAF         // Display ON
    };

    if (oled_write_cmd_list(init_cmds, sizeof(init_cmds)) != ESP_OK) {
        printf("[OLED] Display command setup failed. Falling back to serial-only mode.\n");
        s_oled_present = false;
        i2c_driver_delete(I2C_MASTER_NUM);
        return false;
    }

    oled_clear();
    oled_update();
    printf("[OLED] SSD1306 initialization complete (128x64 mono, 1KB framebuffer).\n");
    return true;
}

bool oled_is_available(void) {
    return s_oled_present;
}

void oled_clear(void) {
    if (!s_oled_present) return;
    memset(s_framebuffer, 0x00, sizeof(s_framebuffer));
    s_cursor_col = 0;
    s_cursor_line = 0;
}

void oled_update(void) {
    if (!s_oled_present) return;

    // Reset column and page bounds to full 128x64 display area
    uint8_t setup_cmds[] = {
        0x21, 0, 127, // Column address: 0 to 127
        0x22, 0, 7    // Page address: 0 to 7 (8 pages * 8 bits = 64 lines)
    };
    oled_write_cmd_list(setup_cmds, sizeof(setup_cmds));

    // Send the 1,024 byte buffer in 8 page chunks (128 bytes per page)
    for (int p = 0; p < 8; p++) {
        i2c_cmd_handle_t link = i2c_cmd_link_create();
        i2c_master_start(link);
        i2c_master_write_byte(link, (s_oled_addr << 1) | I2C_MASTER_WRITE, true);
        i2c_master_write_byte(link, 0x40, true); // Co = 0, D/C# = 1 (Data stream)
        i2c_master_write(link, &s_framebuffer[p * 128], 128, true);
        i2c_master_stop(link);
        i2c_master_cmd_begin(I2C_MASTER_NUM, link, pdMS_TO_TICKS(I2C_TIMEOUT_MS));
        i2c_cmd_link_delete(link);
    }
}

void oled_scroll_up(void) {
    if (!s_oled_present) return;

    // Shift text lines 1..7 up to 0..6 (each line is 128 bytes in horizontal page mode)
    memmove(&s_framebuffer[0], &s_framebuffer[128], 128 * 7);
    // Clear bottom line (line 7)
    memset(&s_framebuffer[128 * 7], 0x00, 128);
}

void oled_set_cursor(int col, int line) {
    if (col < 0) col = 0;
    if (col > 20) col = 20;
    if (line < 0) line = 0;
    if (line > 7) line = 7;
    s_cursor_col = col;
    s_cursor_line = line;
}

void oled_put_char(char c) {
    if (!s_oled_present) return;

    // Handle newline / carriage return
    if (c == '\n') {
        s_cursor_col = 0;
        s_cursor_line++;
        if (s_cursor_line >= 8) {
            oled_scroll_up();
            s_cursor_line = 7;
        }
        return;
    }
    if (c == '\r') {
        s_cursor_col = 0;
        return;
    }

    // Auto-wrap if reaching line end (21 characters max per 128px line)
    if (s_cursor_col >= 21) {
        s_cursor_col = 0;
        s_cursor_line++;
        if (s_cursor_line >= 8) {
            oled_scroll_up();
            s_cursor_line = 7;
        }
    }

    // Unprintable characters replaced with space
    if (c < 32 || c > 126) {
        c = ' ';
    }

    const uint8_t* glyph = s_font5x7[(uint8_t)c - 32];
    int start_byte = (s_cursor_line * 128) + (s_cursor_col * 6);

    for (int i = 0; i < 5; i++) {
        if (start_byte + i < (int)sizeof(s_framebuffer)) {
            s_framebuffer[start_byte + i] = glyph[i];
        }
    }
    // 6th column is blank space between characters
    if (start_byte + 5 < (int)sizeof(s_framebuffer)) {
        s_framebuffer[start_byte + 5] = 0x00;
    }

    s_cursor_col++;
}

void oled_put_string(const char* str) {
    if (!s_oled_present || !str) return;
    while (*str) {
        oled_put_char(*str++);
    }
}

void oled_draw_string(int x, int y_line, const char* str) {
    if (!s_oled_present || !str) return;
    oled_set_cursor(x / 6, y_line);
    oled_put_string(str);
}
