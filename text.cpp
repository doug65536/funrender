#include "text.h"
#include <SDL_ttf.h>
#include <cstdlib>
#include <cstdint>
#include <unordered_map>
#include "fastminmax.h"
#include "funsdl.h"
#include "likely.h"

// Cubemap:
// - Normalize the reflection vector
// - Find largest magnitude component (x, y, or z)
//   this tells you which face to select (of the 6).
// - Divide the other two components by the largest
//   and multiply them by 0.5.
// - This gives normalized -0.5 to +0.5.
// - Add 0.5, shifting the range to +0.0 to +1.0.
// - Face order is usually +x -x +y -y +z -z.
// - If absmax == abs(x): u = y/x, v = z/x, face = x > 0 ? 0 : 1
// - If absmax == abs(y): u = x/y, v = z/y, face = y > 0 ? 2 : 3
// - If absmax == abs(z): u = x/z, v = y/z, face = z > 0 ? 4 : 5

font_data glyphs;

static std::string stringify_glyph(SDL_Surface *surface);
static std::ostream &dump_debug_glyph(std::ostream &out, int ch, SDL_Surface *surface);

void text_init(int size, int first_cp, int last_cp)
{
    font_data data;

    if (TTF_Init() != 0)
        return;

    //char const *font_name = "RobotoMono-VariableFont_wght.ttf";

    // No combining
    //char const *font_name = "/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf";
    // char const *font_name = "/usr/share/fonts/truetype/tlwg/Purisa-BoldOblique.ttf";

    // crap combining
    // char const *font_name = "/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf";

    // perfect!
    char const *font_name = "/usr/share/fonts/truetype/freefont/FreeMono.ttf";

    // char const *font_name = "/usr/share/fonts/truetype/liberation2/LiberationMono-Regular.ttf";
    // char const *font_name = "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf";
    // char const *font_name = "/usr/share/fonts/truetype/tlwg/TlwgMono.ttf";

    // Key is what they look like, value is {sx, width}
    using appearance_map = std::unordered_map<
        std::string, std::pair<int, int>>;
    appearance_map pos_by_appearance;

    std::vector<int> deduped_codepoints;

    TTF_Font *font = TTF_OpenFont(font_name, size);

    if (!font) {
        std::cerr << font_name << " failed to open\n";
        return;
    }

    using glyph_map = std::vector<std::pair<int, glyph_info>>;
    glyph_map char_lookup;

    int height = TTF_FontHeight(font);

    // Measure and render every glyph in the range,
    // and tolerate ones that fail
    int x = 0;
    for (int ch = first_cp; ch <= last_cp; ++ch) {
        int minx{}, miny{}, maxx{}, maxy{}, advance{};
        if (TTF_GlyphMetrics(font, ch, &minx, &maxx,
                &miny, &maxy, &advance))
            continue;

        // note that sx,ex and sy,ey are half open from here on
        SDL_Surface *glyph = TTF_RenderGlyph_Solid(
            font, ch, {0xFF, 0xFF, 0xFF, 0xFF});
        if (!glyph)
            continue;

        glyph_info info;
        // int glyph_w = (maxx - minx) + 1;
        info.codepoint = ch;
        info.advance = advance;
        info.dx = x;
        info.dw = glyph->w;
        info.sx = minx;
        info.ex = maxx + 1;
        info.sy = miny;
        info.ey = maxy + 1;
        std::pair<appearance_map::iterator, bool> ins =
            pos_by_appearance.emplace(
                stringify_glyph(glyph),
                std::make_pair(x, glyph->w));

        if (ins.second) {
            x += glyph->w;
            deduped_codepoints.push_back(ch);
            dump_debug_glyph(std::cerr, ch, glyph);
        } else {
            // This glyph looks identical to another glyph
            // Point it at the identical-looking one
            info.dx = ins.first->second.first;
            info.dw = ins.first->second.second;
            SDL_FreeSurface(glyph);
            glyph = nullptr;
        }

        info.surface = glyph;
        char_lookup.emplace_back(ch, info);
    }
    int total_width = x;

    TTF_CloseFont(font);
    font = nullptr;

    if (char_lookup.empty())
        return;

    // Main metadata
    data.w = total_width;
    data.h = height;
    data.n = char_lookup.size();

    // Compute multiplier for indexing into scanlines of the bitmap
    int bitmap_pitch = (total_width + 7) >> 3;

    // Allocate array for corresponding glyphs codepoints
    data.info = std::make_unique<glyph_info[]>(data.n);
    size_t i = 0;
    for (glyph_map::value_type const &item : char_lookup)
        data.info[i++] = item.second;
    assert(i == (size_t)data.n);

    // Precompute ASCII indices
    std::fill(std::begin(data.ascii), std::end(data.ascii), 0);
    for (size_t i = 0; i < sizeof(data.ascii); ++i) {
        int codepoint = data.info[i].codepoint;
        if ((uint32_t)codepoint < sizeof(data.ascii))
            data.ascii[codepoint] = i;
    }

    // Size of bitmap is bitmap pitch times height
    size_t bitmap_bits_size = bitmap_pitch * data.h;
    // Allocate and clear bitmap
    data.bits = std::make_unique<uint8_t[]>(bitmap_bits_size);
    std::fill_n(data.bits.get(), bitmap_bits_size, 0);
    for (int codepoint : deduped_codepoints) {
        glyph_map::value_type example{codepoint, {}};
        glyph_map::value_type &item = *std::lower_bound(
            char_lookup.begin(), char_lookup.end(), example,
            [](glyph_map::value_type const& lhs,
                    glyph_map::value_type const& rhs) {
                return lhs.first < rhs.first;
            });
        glyph_info& info = item.second;
        SDL_Surface *surface = info.surface;
        assert(surface != nullptr);
        int dx = info.dx;

        uint8_t const *p = static_cast<uint8_t const *>(surface->pixels);
        for (int y = 0; y < surface->h && y < height; ++y) {
            for (int x = 0; x < surface->w; ++x) {
                uint8_t input = p[surface->pitch * y + x];
                int bx = dx + x;
                uint8_t *b = data.bits.get() + y * bitmap_pitch + (bx >> 3);
                *b |= uint8_t(input != 0) << (7 - (bx & 7));
            }
        }

        SDL_FreeSurface(surface);
        info.surface = nullptr;
    }

    glyphs = std::move(data);
}

// Returns ones complement of insertion point if not found
ssize_t find_glyph(int glyph)
{
    // Fastpath ASCII
    if ((uint32_t)glyph < sizeof(glyphs.ascii) &&
            glyphs.ascii[(uint32_t)glyph])
        return glyphs.ascii[glyph];

    size_t st = 0;
    size_t en = glyphs.n;
    size_t mx = en;
    while (st < en) {
        size_t md = ((en - st) >> 1) + st;
        int candidate = glyphs.info[md].codepoint;
        if (candidate < glyph)
            st = md + 1;
        else
            en = md;
    }
    st ^= -(st == mx || glyphs.info[st].codepoint != glyph);
    return ssize_t(st);
}

uint64_t glyph_bits(int index, int row)
{
    glyph_info &info = glyphs.info[index];
    int pitch = (glyphs.w + 7) >> 3;
    int x = info.dx;
    int w = info.dw;
    uint64_t result{};
    for (int i = 0; i < w; ++i) {
        int b = x + i;
        bool bit = glyphs.bits[row * pitch + (b >> 3)] &
            (1U << (7 - (b & 7)));
        result |= (uint64_t(bit) << (w - i - 1));
    }
    return result;
}

void glyph_worker(size_t worker_nr, fill_job &job)
{
    glyph_info const &info = glyphs.info[job.glyph_index];
    uint64_t data = glyph_bits(job.glyph_index, job.row - job.box[1]);
    size_t pixel_index = job.fp.pitch *
            (job.fp.top + job.row) +
            job.box[0] + job.fp.left;
    uint32_t *pixels = &job.fp.pixels[pixel_index];
    float *depths = &job.fp.z_buffer[pixel_index];
    for (size_t bit = info.dw, i = 0;
            data && bit > 0; ++i, --bit) {
        bool set = data & (uint64_t(1) << (bit - 1));
        uint32_t &pixel = pixels[i];
        float &depth = depths[i];
        set &= -(depth > job.z);
        pixel = (job.color & -set) | (pixel & ~-set);
        depth = set ? job.z : depth;
    }
}

__attribute__((__format__(printf, 1, 0)))
std::vector<char> printf_to_vector_v(char const *format, va_list ap)
{
    va_list ap2;
    va_copy(ap2, ap);
    int sz = vsnprintf(nullptr, 0, format, ap);
    if (sz < 0)
        throw std::invalid_argument("format");
    std::vector<char> buffer(sz + 1);
    int sz2 = vsnprintf(buffer.data(), buffer.size(), format, ap2);
    if (sz2 < 0 || sz != sz2)
        throw std::invalid_argument("format");
    va_end(ap2);
    return buffer;
}

__attribute__((__format__(printf, 1, 0)))
int measure_text_v(char const *format, va_list ap)
{
    int total_advance = 0;
    std::vector<char> buffer = printf_to_vector_v(format, ap);
    for (char32_t ch : ucs32(buffer.data(), buffer.data() + buffer.size())) {
        ssize_t glyph_index = find_glyph(ch);
        if (glyph_index >= 0)
            total_advance += glyphs.info[glyph_index].advance;
    }
    return total_advance;
}

__attribute__((__format__(printf, 1, 2)))
int measure_text(char const *format, ...)
{
    va_list ap;
    va_start(ap, format);
    int result = measure_text_v(format, ap);
    va_end(ap);
    return result;
}

__attribute__((__format__(printf, 7, 0)))
void format_text_v(render_target& __restrict frame,
    render_ctx * __restrict ctx,
    int x, int y, float z, uint32_t color,
    char const *format, va_list ap)
{
    std::vector<char> buffer = printf_to_vector_v(format, ap);
    if (!buffer.empty()) {
        // Negative x means right margin, right justify
        if (x < 0) {
            int measurement = measure_text("%s", buffer.data());
            x = frame.width + x - measurement;
        }

        draw_text(frame, ctx, x, y, z, buffer.data(),
            buffer.data() + buffer.size(), 0, color);
    }
}

__attribute__((__format__(printf, 7, 8)))
void format_text(render_target& __restrict frame,
    render_ctx * __restrict ctx,
    int x, int y, float z, uint32_t color,
    char const *format, ...)
{
    va_list ap;
    va_start(ap, format);
    format_text_v(frame, ctx, x, y, z, color, format, ap);
    va_end(ap);
}

void draw_text(render_target& __restrict frame,
    render_ctx * __restrict ctx,
    int x, int y, float z,
    char32_t const *text_st, char32_t const *text_en,
    int wrap_col, uint32_t color)
{
    ensure_scratch(ctx, frame.height);

    // Decompose into lines and recurse
    if (wrap_col > 0) {
        while (text_st < text_en) {
            char32_t const *first_newline = std::find(
                text_st, sane_min(text_st + wrap_col, text_en), L'\n');
            if (first_newline == text_en)
                first_newline = sane_min(text_en, text_st + wrap_col);
            size_t line_length = first_newline - text_st;
            draw_text(frame, ctx, x, y, z,
                text_st, text_st + line_length, -1, color);
            y += glyphs.h;
            text_st += line_length + (text_st + line_length < text_en &&
                text_st[line_length] == '\n');
        }
        return;
    }

    // ... render one line

    // Bail out if it is obviously offscreen
    if (y + glyphs.h < 0 || y >= frame.height || x >= frame.width)
    {
        return;
    }

    int orig_x = x;

    glyph_info *info{};

    for ( ; text_st < text_en;
            ++text_st, orig_x += info ? info->advance : 0) {
        x = orig_x;
        char32_t character = *text_st;
        size_t glyph_index = find_glyph(character);
        if ((ssize_t)glyph_index < 0)
            continue;
        info = &glyphs.info[glyph_index];

        if (likely(info && !is_combining_diacritic(character))) {
        } else if (info) {
            // combining diacritic width
            // int cdw = info->ex - info->sx;
            // // combining diacritic center
            // int cdc = info->sx + cdw / 2;

            //x = last_c - cdc;
            x += info->sx;
            //orig_x += info->sx;
        }

        for (int dy = y,
                ey = dy + glyphs.h;
                dy < ey && dy < frame.height; ++dy) {
            if (x + info->dw > 0 &&
                    x < frame.width &&
                    y + glyphs.h >= 0) {
                size_t slot = dy % task_workers.size();
                fill_job_batch(ctx, slot).emplace_back(
                    frame, ctx, dy, x, y, z,
                    glyph_index, color);
            }
        }
    }

    commit_batches(ctx);
}

void draw_text(render_target& __restrict frame,
    render_ctx * __restrict ctx,
    int x, int y, float z,
    char const *utf8_st, char const *utf8_en,
    int wrap_col, uint32_t color)
{
    if (!utf8_en)
        utf8_en = strchr(utf8_st, 0);

    std::vector<char32_t> codepoints = ucs32(utf8_st, utf8_en);
    char32_t const *text_st = codepoints.data();
    char32_t const *text_en = text_st + codepoints.size();

    draw_text(frame, ctx,
        x, y, z, text_st, text_en, wrap_col, color);
}

bool is_combining_diacritic(uint32_t codepoint)
{
    return (codepoint >= 0x0300 && codepoint <= 0x036F) ||
           (codepoint >= 0x20D0 && codepoint <= 0x20FF) ||
           (codepoint >= 0xFE20 && codepoint <= 0xFE2F);
}

// Convert the given range of UTF-8 to UCS32
std::vector<char32_t> ucs32(char const *start, char const *end,
    bool *ret_failed)
{
    if (ret_failed)
        *ret_failed = false;
    uint8_t const *st = (uint8_t const *)start;
    uint8_t const *en = (uint8_t const *)end;
    std::vector<char32_t> result;
    char32_t unicode_replacement = (char32_t)0xfffd;

    while (st < en) {
        if (*st < 0x80) {
            result.push_back(*st++);
        } else if ((*st & 0xe0) == 0xc0 &&
                st + 1 < en &&
                (st[0] & 0x1f) != 0 &&
                ((st[1] & 0xc0) == 0x80)) {
            // 2-byte
            result.push_back(((st[0] & 0x1f) << 6) |
                (st[1] & 0x3f));
            st += 2;
        } else if ((st[0] & 0xf0) == 0xe0 &&
                st + 2 < en &&
                (st[0] & 0x0F) != 0 &&
                (st[1] & 0xc0) == 0x80 &&
                (st[2] & 0xc0) == 0x80) {
            // 3-byte
            char32_t codepoint = ((st[0] & 0xf) << 12) |
                ((st[1] & 0x3F) << 6) |
                ((st[2] & 0x3F));
            // check for sneaky surrogate pair
            codepoint = (codepoint >= 0xD800 && codepoint <= 0xDFFF)
                ? unicode_replacement
                : codepoint;
            result.push_back(codepoint);
            st += 3;
        } else if ((st[0] & 0xf8) == 0xf0 &&
                st + 3 < en &&
                (st[0] & 0x07) != 0 &&
                (st[1] & 0xc0) == 0x80 &&
                (st[2] & 0xc0) == 0x80 &&
                (st[3] & 0xc0) == 0x80) {
            // 4-byte
            result.push_back(((*st & 0x7) << 18) |
                ((st[1] & 0x3F) << 12) |
                ((st[2] & 0x3F) << 6) |
                ((st[3] & 0x3F)));
            st += 4;
        } else {
            result.push_back(unicode_replacement);
            while (st < en && st[0] & 0x80)
                ++st;
            if (ret_failed)
                *ret_failed = true;
        }
    }

    return result;
}

std::string stringify_glyph(SDL_Surface *surface)
{
    std::string s;
    s = "[\n";
    for (int y = 0; y < surface->h; ++y) {
        for (int x = 0; x < surface->w; ++x) {
            uint8_t *pixels = reinterpret_cast<uint8_t*>(surface->pixels);
            uint8_t pixel = pixels[surface->pitch * y + x];
            s.push_back(pixel ? '*' : ':');
        }
        s += L'\n';
    }
    s += "]\n";
    return s;
}


std::ostream &dump_debug_glyph(std::ostream &out, int ch, SDL_Surface *surface)
{
    return out << stringify_glyph(surface) << '\n';
}

