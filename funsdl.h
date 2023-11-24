#pragma once
#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <cmath>
#include <memory>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <SDL.h>

extern uint64_t last_fps, smooth_fps;

struct frame_param {
    uint16_t width{};
    uint16_t height{};
    uint16_t left{};
    uint16_t top{};
    uint16_t pitch{};
    // alignment gap here
    unsigned *pixels{};
    float *z_buffer{};
};

extern std::vector<glm::mat4> view_mtx_stk;
extern std::vector<glm::mat4> proj_mtx_stk;


using clk = std::chrono::high_resolution_clock;
using time_point = typename clk::time_point;
using duration = typename clk::duration;

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

struct scaninfo {
    glm::vec2 t;
    glm::vec4 p;
    glm::vec3 n;

    scaninfo() = default;

    scaninfo(scaninfo const& rhs) = default;
    scaninfo &operator=(scaninfo const& rhs) = default;

    scaninfo(glm::vec4 const& p)
        : t{}
        , p{p}
        , n{}
    {
    }

    scaninfo(glm::vec2 const& t, glm::vec4 const& p, glm::vec3 const& n)
        : t{t}
        , p{p}
        , n{n}
    {
    }

    scaninfo operator-(scaninfo const& rhs) const
    {
        return {
            t - rhs.t,
            p - rhs.p,
            n - rhs.n
        };
    }

    scaninfo operator+(scaninfo const& rhs) const
    {
        return {
            t + rhs.t,
            p + rhs.p,
            n + rhs.n
        };
    }

    scaninfo operator*(float rhs) const
    {
        return {
            t * rhs,
            p * rhs,
            n * rhs
        };
    }

    bool operator==(scaninfo const& rhs) const
    {
        return t == rhs.t && p == rhs.p && n == rhs.n;
    }

    bool operator!=(scaninfo const& rhs) const
    {
        return !(*this == rhs);
    }

    friend scaninfo operator *(glm::mat4 const& lhs, scaninfo const& rhs)
    {
        scaninfo result{rhs};
        result.p = lhs * rhs.p;
        return result;
    }
};

template <class T>
inline void hash_combine(std::size_t& seed, const T& value) {
    seed ^= std::hash<T>()(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

size_t hash_bytes(void const *data, size_t size);

template<>
struct std::hash<scaninfo> {
    std::size_t operator()(scaninfo const& rhs) const
    {
        #if 0
        // Horrible!
        return hash_bytes(this, sizeof(this));
        #else
        // Wow
        std::size_t seed = 0;
        hash_combine(seed, std::hash<float>()(rhs.t.x));
        hash_combine(seed, std::hash<float>()(rhs.t.y));
        hash_combine(seed, std::hash<float>()(rhs.p.x));
        hash_combine(seed, std::hash<float>()(rhs.p.y));
        hash_combine(seed, std::hash<float>()(rhs.p.z));
        hash_combine(seed, std::hash<float>()(rhs.n.x));
        hash_combine(seed, std::hash<float>()(rhs.n.y));
        hash_combine(seed, std::hash<float>()(rhs.n.z));
        return seed;
        #endif
    }
};

struct edgeinfo {
    unsigned edge_idx;
    unsigned diff_idx;
    float n;
};

struct scanconv_ent {
    uint32_t used = 0;
    edgeinfo range[2];
};

extern std::vector<scanconv_ent> scanconv_scratch;

void fill_triangle(frame_param const& fp,
    scaninfo const& v0, scaninfo const& v1,
    scaninfo v2, unsigned color);
void texture_triangle(frame_param const& fp,
    scaninfo v0, scaninfo v1,
    scaninfo v2);
void fill_box(frame_param const& fp,
    int sx, int sy,
    int ex, int ey, uint32_t color, int border_radius = 0);

void set_texture(uint32_t const *incoming_pixels,
    int incoming_w, int incoming_h, int incoming_pitch,
    void (*free_fn)(void*));

void flip_colors(uint32_t *pixels, int imgw, int imgh);
void bind_texture(size_t binding);
bool delete_texture(size_t binding);
void parallel_clear(frame_param const &frame, uint32_t color);
void draw_text(frame_param const &frame, int x, int y,
               char const *text_st, char const *text_en = nullptr,
               int wrap_col = -1, uint32_t color = 0xFFFFFFFF);

void print_matrix(char const *title, std::ostream &out, glm::mat4 const& pm);

template <typename T, glm::length_t L, glm::qualifier Q>
std::ostream &print_vector(std::ostream &out, glm::vec<L, T, Q> const &v)
{
    out.put('[');
    for (glm::length_t i = 0; i < L; ++i) {
        if (i)
            out.put(',');
        out << v[i];
    }
    return out.put(']');
}

void texture_polygon(frame_param const& frame,
    scaninfo const *vinp, size_t count);

void texture_polygon(frame_param const& frame,
    std::vector<scaninfo> vinp);

template<size_t N>
void texture_polygon(frame_param const& frame,
    std::array<scaninfo, N> const& vinp)
{
    texture_polygon(frame, vinp.data(), N);
}

void set_transform(glm::mat4 const& mat);
