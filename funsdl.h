#pragma once
#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <cmath>
#include <vector>
#include <memory>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <SDL.h>
struct fill_job;

void ensure_scratch(size_t height);
extern std::vector<std::vector<fill_job>> fill_job_batches;


#include "task_worker.h"
#include "text.h"

extern uint64_t last_fps, smooth_fps;

struct render_target {
    int width{};
    int height{};
    int pitch{};
    int left{};
    int top{};
    // alignment gap here
    unsigned *pixels{};
    float *z_buffer{};

    render_target() = default;

    render_target(int width_, int height_, int pitch_,
            uint32_t *pixels_, float *z_buffer_,
            int left_ = 0, int top_ = 0)
        : width(width_)
        , height(height_)
        , pitch(pitch_)
        , left(left_)
        , top(top_)
        , pixels(pixels_)
        , z_buffer(z_buffer_)
    {
    }

    render_target(render_target const&) = default;
    render_target(render_target&&) = default;

    render_target subset(int left_, int top_, int width_, int height_) const
    {
        render_target proposed{
            width_,
            height_,
            pitch,
            pixels,
            z_buffer,
            left + left_,
            top + top_
        };

        proposed.left = std::min(proposed.left, width);
        proposed.top = std::min(proposed.top, height);
        proposed.width = std::min(width, proposed.width - proposed.left);
        proposed.height = std::min(height, proposed.height - proposed.top);

        return proposed;
    }
};

extern std::vector<glm::mat4> view_mtx_stk;
extern std::vector<glm::mat4> proj_mtx_stk;


struct scaninfo {
    glm::vec4 p;
    glm::vec2 t;
    glm::vec3 n;
    glm::vec3 c;

    scaninfo() = default;

    scaninfo(scaninfo const& rhs) = default;
    scaninfo &operator=(scaninfo const& rhs) = default;

    scaninfo(glm::vec4 const& p)
        : p{p}
        , t{}
        , n{}
        , c{}
    {
    }

    scaninfo(glm::vec2 const& t, glm::vec4 const& p,
            glm::vec3 const& n, glm::vec3 const& c)
        : p{p}
        , t{t}
        , n{n}
        , c{c}
    {
    }

    scaninfo operator-(scaninfo const& rhs) const
    {
        return {
            t - rhs.t,
            p - rhs.p,
            n - rhs.n,
            c - rhs.c
        };
    }

    scaninfo operator+(scaninfo const& rhs) const
    {
        return {
            t + rhs.t,
            p + rhs.p,
            n + rhs.n,
            c + rhs.c
        };
    }

    scaninfo operator*(float rhs) const
    {
        return {
            t * rhs,
            p * rhs,
            n * rhs,
            c * rhs
        };
    }

    bool operator==(scaninfo const& rhs) const
    {
        return t == rhs.t && p == rhs.p && n == rhs.n && c == rhs.c;
    }

    bool operator!=(scaninfo const& rhs) const
    {
        return !(*this == rhs);
    }

    friend scaninfo operator *(glm::mat4 const& lhs, scaninfo const& rhs)
    {
        scaninfo result{rhs};

        // Transform the position
        result.p = lhs * rhs.p;

        // Transform the normal
        result.n = glm::mat3(lhs) * rhs.n;

        return result;
    }

    bool operator<(scaninfo const &rhs) const
    {
        glm::bvec2 tlt = glm::lessThan(t, rhs.t);
        glm::bvec2 tgt = glm::lessThan(rhs.t, t);
        glm::bvec4 plt = glm::lessThan(p, rhs.p);
        glm::bvec4 pgt = glm::lessThan(rhs.p, p);
        glm::bvec3 nlt = glm::lessThan(n, rhs.n);
        glm::bvec3 ngt = glm::lessThan(rhs.n, n);
        glm::bvec3 clt = glm::lessThan(c, rhs.c);
        glm::bvec3 cgt = glm::lessThan(rhs.c, c);

        int return_true = -tlt.s;
        int return_false = -tgt.s;

        return_true |= ~return_false & -tlt.t;
        return_false |= ~return_true & -tgt.t;

        return_true |= ~return_false & -plt.s;
        return_false |= ~return_true & -pgt.s;

        return_true |= ~return_false & -plt.t;
        return_false |= ~return_true & -pgt.t;

        return_true |= ~return_false & -plt.z;
        return_false |= ~return_true & -pgt.z;

        return_true |= ~return_false & -plt.w;
        return_false |= ~return_true & -pgt.w;

        return_true |= ~return_false & -nlt.x;
        return_false |= ~return_true & -ngt.x;

        return_true |= ~return_false & -nlt.y;
        return_false |= ~return_true & -ngt.y;

        return_true |= ~return_false & -nlt.z;
        return_false |= ~return_true & -ngt.z;

        return_true |= ~return_false & -clt.x;
        return_false |= ~return_true & -cgt.x;

        return_true |= ~return_false & -clt.y;
        return_false |= ~return_true & -cgt.y;

        return_true |= ~return_false & -clt.z;
        return_false |= ~return_true & -cgt.z;

        return return_true;
    }
};

template <class T>
inline void hash_combine(std::size_t& seed, const T& value)
{
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
        hash_combine(seed, std::hash<float>()(rhs.t.s));
        hash_combine(seed, std::hash<float>()(rhs.t.t));
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

void fill_triangle(render_target const& fp,
    scaninfo const& v0, scaninfo const& v1,
    scaninfo v2, unsigned color);

void texture_triangle(render_target const& fp,
    scaninfo v0, scaninfo v1,
    scaninfo v2);

void fill_box(render_target const& fp,
    int sx, int sy,
    int ex, int ey, float z,
    uint32_t color, int border_radius = 0);

void set_texture(uint32_t const *incoming_pixels,
    int incoming_w, int incoming_h, int incoming_pitch,
    int incoming_levels, void (*free_fn)(void*));

void flip_colors(uint32_t *pixels, int imgw, int imgh);
void bind_texture(size_t binding);
bool delete_texture(size_t binding);
void parallel_clear(render_target const &frame, uint32_t color);
void draw_text(render_target const &frame, int x, int y, float z,
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

void texture_polygon(render_target const& frame,
    scaninfo const *vinp, size_t count);

void texture_polygon(render_target const& frame,
    std::vector<scaninfo> vinp);

template<size_t N>
void texture_polygon(render_target const& frame,
    std::array<scaninfo, N> const& vinp)
{
    texture_polygon(frame, vinp.data(), N);
}

void texture_elements(render_target const &fp,
    scaninfo const *vertices, size_t vertex_count,
    uint32_t const *elements, size_t element_count);

void set_transform(glm::mat4 const& pm,
    glm::mat4 const& vm);

extern bool mouselook_pressed[]; // WASDRF
extern float mouselook_yaw;
extern float mouselook_pitch;
extern float mouselook_yaw_scale;
extern float mouselook_pitch_scale;
extern glm::vec3 mouselook_pos;
extern glm::vec3 mouselook_vel;
extern glm::vec3 mouselook_acc;
extern glm::vec3 mouselook_px;
extern glm::vec3 mouselook_py;
extern glm::vec3 mouselook_pz;

extern std::vector<std::string> command_line_files;

#pragma once

#include "barrier.h"
#include "text.h"

struct fill_job {
    // barrier
    fill_job(render_target const &fp, barrier *frame_barrier)
        : fp(fp)
        , frame_barrier(frame_barrier)
    {}

    // clean
    fill_job(render_target const &fp, int cleared_row, uint32_t color)
        : fp(fp)
        , clear_row(cleared_row)
        , clear_color(color)
    {}

    // span
    fill_job(render_target const &fp, int y,
        edgeinfo const& lhs, edgeinfo const& rhs, unsigned back_phase)
        : fp(fp)
        , edge_refs(lhs, rhs)
        , back_phase(back_phase)
    {
        box[1] = std::max(int16_t(0),
            std::min((int16_t)y, (int16_t)INT16_MAX));
    }

    // box
    fill_job(render_target const& fp, int y,
            int sx, int sy, int ex, int ey, float z, uint32_t color,
            std::vector<float> const *border_table)
        : fp(fp)
        , clear_color(color)
        , border_table(border_table)
        , z(z)
        , box_y(y)
        , box{(int16_t)sx, (int16_t)sy, (int16_t)ex, (int16_t)ey}
    {}

    // glyph (text)
    fill_job(render_target const& fp, int y, int sx, int sy, float z,
            size_t glyph_index, uint32_t color)
        : fp(fp)
        , clear_color(color)
        , z(z)
        , glyph_index(glyph_index)
        , box_y(y)
        , box{}
    {
        glyph_info const& info = glyphs.info[glyph_index];
        box[0] = (int16_t)sx;
        box[1] = (int16_t)sy;
        box[2] = (int16_t)(sx + (info.ex - info.sx));
        box[3] = (int16_t)(sy + (info.ey - info.sy));
    }

    render_target fp;
    std::pair<edgeinfo, edgeinfo> edge_refs;
    unsigned back_phase;
    int clear_row = -1;
    uint32_t clear_color;
    barrier *frame_barrier = nullptr;
    std::vector<float> const *border_table;
    float z;
    uint16_t glyph_index = -1;
    int16_t box_y = -1;
    int16_t box[4];
};

using fill_task_worker = task_worker<fill_job>;
extern std::vector<fill_task_worker> task_workers;

//  7: 0 blue
// 15: 8 green
// 23:16 blue
// 31:24 alpha

// packed RGBA conversion readability utilities
#define rgba_shift_b        0
#define rgba_shift_g        8
#define rgba_shift_r        16
#define rgba_shift_a        24
#define rgba(r,g,b,a) ( \
    ((b) << rgba_shift_b) | \
    ((g) << rgba_shift_g) | \
    ((r) << rgba_shift_r) | \
    ((a) << rgba_shift_a) \
    )
#define rgb0(r,g,b)         (rgba((r), (g), (b), 0))
#define rgb(r,g,b)          (rgba((r), (g), (b), 0xff))
#define rgb_b(pixel)        (((pixel) >> rgba_shift_b) & 0xff)
#define rgb_g(pixel)        (((pixel) >> rgba_shift_g) & 0xff)
#define rgb_r(pixel)        (((pixel) >> rgba_shift_r) & 0xff)
#define rgb_a(pixel)        (((pixel) >> rgba_shift_a) & 0xff)

int select_mipmap(glm::vec2 const& diff_of_t, float invWidth);
uint32_t indexof_mipmap(int level);

void set_light_enable(size_t light_nr, bool enable);
void set_light_pos(size_t light_nr, glm::vec4 const &pos);
void set_light_spot(size_t light_nr,
    glm::vec3 const& dir, float cutoff, float exponent);
void set_light_diffuse(size_t light_nr, glm::vec3 color);
void set_light_specular(size_t light_nr, glm::vec3 color, float shininess);
