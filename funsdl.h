#pragma once
#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <cmath>
#include <vector>
#include <memory>
#include <unordered_map>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <SDL.h>
#include "barrier.h"

struct render_ctx;

struct fill_job;

std::vector<fill_job> &fill_job_batch(
    render_ctx * __restrict ctx, size_t slot);

void ensure_scratch(render_ctx * __restrict ctx, size_t height);

#include "task_worker.h"
#include "text.h"

extern uint64_t last_fps, smooth_fps;

struct scaninfo {
    glm::vec4 p;
    glm::vec3 t;
    glm::vec3 n;
    glm::vec3 c;
    glm::vec3 u;

    scaninfo() = default;

    scaninfo(scaninfo const& rhs) = default;
    scaninfo &operator=(scaninfo const& rhs) = default;

    scaninfo(glm::vec4 const& p)
        : p{p}
    {
    }

    scaninfo(glm::vec2 const& t, glm::vec4 const& p,
            glm::vec3 const& n, glm::vec3 const& c,
            glm::vec3 const& u = {})
        : p{p}
        , t{t, 0.0f}
        , n{n}
        , c{c}
        , u{u}
    {
    }

    scaninfo operator-(scaninfo const& rhs) const
    {
        return {
            t - rhs.t,
            p - rhs.p,
            n - rhs.n,
            c - rhs.c,
            u - rhs.u
        };
    }

    friend scaninfo operator-(float lhs,
        scaninfo const& rhs)
    {
        return {
            lhs - rhs.t,
            lhs - rhs.p,
            lhs - rhs.n,
            lhs - rhs.c,
            lhs - rhs.u
        };
    }

    scaninfo operator+(scaninfo const& rhs) const
    {
        return {
            t + rhs.t,
            p + rhs.p,
            n + rhs.n,
            c + rhs.c,
            u + rhs.u
        };
    }

    scaninfo operator*(float rhs) const
    {
        return {
            t * rhs,
            p * rhs,
            n * rhs,
            c * rhs,
            u * rhs
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

template<>
struct std::hash<scaninfo> {
    std::size_t operator()(scaninfo const& rhs) const
    {
        std::size_t seed = 0;
        hash_combine(seed, std::hash<float>()(rhs.t.s));
        hash_combine(seed, std::hash<float>()(rhs.t.t));
        // don't need t.p, it's (partly) implied by s and t if used
        hash_combine(seed, std::hash<float>()(rhs.p.x));
        hash_combine(seed, std::hash<float>()(rhs.p.y));
        hash_combine(seed, std::hash<float>()(rhs.p.z));
        // don't need p.w in hash, the z there covers it well enough
        hash_combine(seed, std::hash<float>()(rhs.n.x));
        hash_combine(seed, std::hash<float>()(rhs.n.y));
        // don't need n.z, it's largely implied by n.x and n.y
        // don't need u.xyz in hash - it's usually not used
        return seed;
    }
};

struct edgeinfo {
    unsigned edge_idx;
    unsigned diff_idx;
    float n;
};

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

    render_target(render_target const&) = delete;
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

    barrier present_barrier;

    uint32_t *back_buffer{};

    SDL_Surface * back_surface{};
    bool back_surface_locked{};

    std::vector<scaninfo> edges;
    std::unordered_map<scaninfo, unsigned> edges_lookup;
    float edges_lookup_load_factor{};
};

size_t hash_bytes(void const *data, size_t size);

void texture_raw_triangle(render_target& __restrict fp,
    render_ctx * __restrict ctx,
    scaninfo v0, scaninfo v1, scaninfo v2);

void fill_box(render_target& __restrict fp,
    render_ctx * __restrict ctx,
    int sx, int sy, int ex, int ey, float z,
    uint32_t color, int border_radius = 0);

void set_texture(render_ctx * __restrict ctx,
    uint32_t const * __restrict incoming_pixels,
    int incoming_w, int incoming_h, int incoming_pitch,
    int incoming_levels, void (*free_fn)(void*));

void flip_colors(uint32_t *pixels, int imgw, int imgh);
void bind_texture(render_ctx * __restrict ctx,
    size_t binding);
bool delete_texture(render_ctx * __restrict ctx,
    size_t binding);
void parallel_clear(render_target & __restrict frame,
    render_ctx * __restrict ctx,
    uint32_t color);
void draw_text(render_target & __restrict frame,
    render_ctx * __restrict ctx,
    int x, int y, float z,
    char const *text_st, char const *text_en = nullptr,
    int wrap_col = -1, uint32_t color = 0xFFFFFFFF);

void print_matrix(char const *title,
    std::ostream &out, glm::mat4 const& pm);

template <typename T, glm::length_t L, glm::qualifier Q>
std::ostream &print_vector(std::ostream &out,
    glm::vec<L, T, Q> const &v)
{
    out.put('[');
    for (glm::length_t i = 0; i < L; ++i) {
        if (i)
            out.put(',');
        out << v[i];
    }
    return out.put(']');
}

void draw_polygon(render_target& __restrict frame,
    render_ctx * __restrict ctx,
    scaninfo const *vinp, size_t count,
    bool skip_xform = false);

void draw_polygon(render_target &__restrict frame,
                     render_ctx *__restrict ctx,
                     std::vector<scaninfo> const& vinp);

template <size_t N>
void draw_polygon(render_target &__restrict frame,
                     render_ctx *__restrict ctx,
                     std::array<scaninfo, N> const &vinp)
{
    draw_polygon(frame, ctx, vinp.data(), N);
}

void draw_elements(render_target & __restrict fp,
    render_ctx * __restrict ctx,
    scaninfo const * __restrict vertices, size_t vertex_count,
    uint32_t const * __restrict elements, size_t element_count);

void set_transform(render_ctx * __restrict ctx,
    glm::mat4 const * __restrict vm = nullptr,
    glm::mat4 const * __restrict pm = nullptr);

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
extern bool text_test;

extern std::vector<std::string> command_line_files;

#pragma once

#include "barrier.h"
#include "text.h"

struct fill_job {
    // barrier
    fill_job(render_target & __restrict fp,
            render_ctx * __restrict ctx,
            barrier *frame_barrier);

    // clean
    fill_job(render_target & __restrict fp,
            render_ctx * __restrict ctx,
            int cleared_row, uint32_t color);

    // span
    fill_job(render_target & __restrict fp,
            render_ctx * __restrict ctx,
            int y, edgeinfo const& lhs,
            edgeinfo const& rhs);

    // box
    fill_job(render_target& __restrict fp,
            render_ctx * __restrict ctx,
            int y, int sx, int sy, int ex, int ey,
            float z, uint32_t color,
            std::vector<float> const *border_table);

    // glyph (text)
    fill_job(render_target& __restrict fp,
            render_ctx * __restrict ctx,
            int y, int sx, int sy, float z,
            size_t glyph_index, uint32_t color);

    void (*handler)(size_t worker_nr, fill_job &job){};

    render_target& __restrict fp;
    render_ctx * __restrict ctx;

    // 24 bytes
    std::pair<edgeinfo, edgeinfo> edge_refs;

    //unsigned back_phase;
    int row;
    uint32_t color;
    barrier *frame_barrier;
    std::vector<float> const *border_table;
    float z;
    uint16_t glyph_index;
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

int select_mipmap(render_ctx * __restrict ctx,
    glm::vec2 const& diff_of_t, float invWidth);
uint32_t indexof_mipmap(render_ctx * __restrict ctx, int level);

void set_light_enable(render_ctx * __restrict ctx,
    size_t light_nr, bool enable);
void set_light_pos(render_ctx * __restrict ctx,
    size_t light_nr, glm::vec4 const &pos);
void set_light_spot(render_ctx * __restrict ctx,
    size_t light_nr,
    glm::vec3 const& dir, float cutoff, float exponent);
void set_light_diffuse(render_ctx * __restrict ctx,
    size_t light_nr, glm::vec3 color);
void set_light_specular(render_ctx * __restrict ctx,
    size_t light_nr, glm::vec3 color, float shininess);

void commit_batches(render_ctx * __restrict ctx,
    bool allow_notify = false);

void set_proj_matrix(render_ctx * __restrict ctx,
    glm::mat4 const& mtx);
void set_view_matrix(render_ctx * __restrict ctx,
    glm::mat4 const& mtx);

glm::mat4 &push_proj_matrix(render_ctx * __restrict ctx);
glm::mat4 &push_view_matrix(render_ctx * __restrict ctx);

void pop_proj_matrix(render_ctx * __restrict ctx);
void pop_view_matrix(render_ctx * __restrict ctx);

glm::mat4 &get_proj_matrix(render_ctx * __restrict ctx);
glm::mat4 &get_view_matrix(render_ctx * __restrict ctx);

render_ctx *new_ctx();

render_target create_render_target(int width, int height,
    bool has_color, bool has_z);
render_target create_cubemap_target(int width, int height);

struct draw_stats {
    // Number of polygons partially outside any clip plane
    size_t clipped;

    // Number of polygons entirely inside all clip planes
    size_t unclipped;

    // Number of polygons entirely outside any clip plane
    size_t culled;
};

draw_stats const* get_draw_stats(render_ctx * __restrict ctx);

void set_color_mask(render_ctx * __restrict ctx, bool color_mask);
void set_depth_mask(render_ctx * __restrict ctx, bool depth_mask);
void set_depth_test(render_ctx * __restrict ctx, bool depth_test);
void push_masks(render_ctx * __restrict ctx);
void pop_masks(render_ctx * __restrict ctx);

std::array<float, 2> get_edge_stats();

extern bool measure;
extern bool raytrace;
