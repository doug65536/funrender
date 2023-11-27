#pragma once
#include <SDL.h>
#include <memory>

struct glyph_info {
    int codepoint{};
    int advance{};
    int dx{};
    int dw{};
    int sx{};
    int ex{};
    int sy{};
    int ey{};
    SDL_Surface *surface{};
};

struct font_data {
    int w{};
    int h{};
    int n{};
    std::unique_ptr<glyph_info[]> info;
    std::unique_ptr<uint8_t[]> bits;
    uint8_t ascii[128];
};

extern font_data glyphs;

struct fill_job;

void text_init(int size, int first_cp = 1, int last_cp = 0xFFFF);
void glyph_worker(fill_job &job);

#include "funsdl.h"

std::vector<char32_t> ucs32(char const *start, char const *end,
    bool *ret_failed = nullptr);

struct frame_param;
#include "funsdl.h"

__attribute__((__format__(printf, 5, 6)))
bool format_text(frame_param const& frame,
    int x, int y, uint32_t color,
    char const *format, ...);

__attribute__((__format__(printf, 5, 0)))
bool format_text_v(frame_param const& frame,
    int x, int y, uint32_t color,
    char const *format, va_list ap);
