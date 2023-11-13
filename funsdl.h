#pragma once
#include <chrono>
#include <vector>
#include <cmath>
#include <memory>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

struct frame_param {
    uint16_t width{};
    uint16_t height{};
    uint16_t pitch{};
    unsigned *pixels{};
};

static std::vector<glm::mat4> view_mtx{1};
static std::vector<glm::mat4> proj_mtx{1};

using clk = std::chrono::steady_clock;
using time_point = typename clk::time_point;
using duration = typename clk::duration;

struct font_data {
    int w{};
    int h{};
    int n{};
    int lo{};
    std::unique_ptr<int[]> glyphs;
    std::unique_ptr<uint8_t[]> bits;
};

struct scaninfo {
    glm::vec2 t;
    glm::vec3 p;
    glm::vec3 n;

    scaninfo() = default;

    scaninfo(scaninfo const& rhs) = default;
    scaninfo &operator=(scaninfo const& rhs) = default;

    scaninfo(glm::vec3 const& p)
        : t{}
        , p{p}
        , n{}
    {
    }

    scaninfo(glm::vec2 const& t, glm::vec3 const& p, glm::vec3 const& n)
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
};

template <class T>
inline void hash_combine(std::size_t& seed, const T& value) {
    seed ^= std::hash<T>()(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

template<>
struct std::hash<scaninfo> {
    std::size_t operator()(scaninfo const& rhs) const
    {
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
    scaninfo const& v0, scaninfo const& v1,
    scaninfo const& v2);
void fill_box(frame_param const& fp,
    int sx, int sy,
    int ex, int ey, uint32_t color);

void set_texture(uint32_t const *incoming_pixels,
    int incoming_w, int incoming_h, int incoming_pitch,
    void (*free_fn)(void*));

void flip_colors(uint32_t *pixels, int imgw, int imgh);
void bind_texture(size_t binding);
bool delete_texture(size_t binding);
void parallel_clear(frame_param const &frame, uint32_t color);
void draw_text(frame_param const& frame, int x, int y,
    char const *text_st, char const *text_en = nullptr,
    int wrap_col = -1, uint32_t color = 0xFFFFFFFF);
