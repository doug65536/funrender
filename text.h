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
void glyph_worker(size_t worker_nr, fill_job &job);

#include "funsdl.h"

std::vector<char32_t> ucs32(char const *start, char const *end,
    bool *ret_failed = nullptr);

bool is_combining_diacritic(uint32_t codepoint);

struct render_target;
#include "funsdl.h"

__attribute__((__format__(printf, 7, 8)))
void format_text(render_target& __restrict frame,
    render_ctx * __restrict ctx,
    int x, int y, float z, uint32_t color,
    char const *format, ...);

__attribute__((__format__(printf, 7, 0)))
void format_text_v(render_target& __restrict frame,
    render_ctx * __restrict ctx,
    int x, int y, float z, uint32_t color,
    char const *format, va_list ap);
