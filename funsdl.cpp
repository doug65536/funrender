#include "funsdl.h"
#include "assume.h"
#include <SDL.h>
#include <SDL_ttf.h>
#include <iostream>
#include <iomanip>
#include <iterator>
#include <algorithm>
#include <memory>
#include <atomic>
#include <map>
#include <thread>
#include <unordered_map>
#include <mutex>
#include <filesystem>
#include <vector>
#include <deque>
#include <list>
#include <condition_variable>
#include <functional>
#include <libpng16/png.h>
#include <glm/glm.hpp>

#include "affinity.h"
#include "abstract_vector.h"
#include "barrier.h"
#include "task_worker.h"
#include "fastminmax.h"
#include "text.h"

struct render_ctx {
    static constexpr size_t light_count = 8;

    glm::vec3 light_ambient;

    bool light_enable[light_count];

    // If w=0, light is directional at infinity
    // If w=1, light is at the specific position
    glm::vec4 light_position[light_count];

    // Diffuse color
    glm::vec3 light_diffuse[light_count];

    // Specular color
    glm::vec3 light_specular[light_count];

    // Spot direction in viewspace
    glm::vec3 light_spot_dir[light_count];

    // x is spot cutoff, y is spot exponent, z is shininess
    glm::vec3 light_info[light_count];
};

static int blit_to_screen(SDL_Surface *surface, SDL_Rect &src,
    SDL_Surface *window_surface, SDL_Rect &dest);

uint64_t last_fps;
uint64_t smooth_fps;

constexpr size_t phase_count = 2;

int window_w, window_h;

// [1029.33,0,388.895] -3.85 -2.22
bool mouselook_enabled;
bool mouselook_pressed[6]; // WASDRF
// float mouselook_yaw = 0.0f;
// float mouselook_pitch = 0.0f;
float mouselook_pitch = 1.23f;
float mouselook_yaw = -2.92f;
float mouselook_yaw_scale = -0.01f;
float mouselook_pitch_scale = -0.01f;
glm::vec3 mouselook_pos{1029.33,0,388.895};
glm::vec3 mouselook_vel;
glm::vec3 mouselook_acc;
glm::vec3 mouselook_px;
glm::vec3 mouselook_py;
glm::vec3 mouselook_pz;

#if PIXELS_HISTOGRAM
size_t const pixels_histogram_size = 1024;
std::atomic_ulong pixels_histogram[pixels_histogram_size];
#endif

#if PIXELS_HISTOGRAM
void dump_histogram(int height)
{
    unsigned long peak = 0;
    for (std::atomic_ulong const &ent : pixels_histogram)
        peak = sane_max(peak, ent.load(std::memory_order_acquire));

    // std::vector<char> text_canvas(pixels_histogram_size * height, ' ');
    // for (size_t x = 0; x < pixels_histogram_size; ++x) {
    //     unsigned long value =
    //         pixels_histogram[x].load(std::memory_order_acquire);
    //     int scaled = (long long)sane_min(peak, value) * height / peak;
    //     for (int y = height - 1; y > height - scaled; --y)
    //         text_canvas[y * pixels_histogram_size + x] = '*';
    // }
    // for (int y = 0; y < height; ++y) {
    //     std::cerr << "|";
    //     std::cerr.write(text_canvas.data() + pixels_histogram_size * y,
    //         pixels_histogram_size);
    //     std::cerr << "|\n";
    // }

    for (size_t x = 0; x < pixels_histogram_size; ++x) {
        if (pixels_histogram[x]) {
            std::cerr << std::setw(4) << std::setfill(' ') << x << ':' <<
                std::setw(0) << pixels_histogram[x] << "\n";
        }
    }
}
#endif

ssize_t find_glyph(int glyph);


std::vector<fill_task_worker> task_workers;

std::vector<scanconv_ent> scanconv_scratch;

void user_frame(render_target& frame, render_ctx *ctx);

uint32_t draw_nr;

// Constants for initial window dimensions
// const int SCREEN_WIDTH = 2560;
// const int SCREEN_HEIGHT = 1440;
// const int SCREEN_WIDTH = 1920;
// const int SCREEN_HEIGHT = 1080;
const int SCREEN_WIDTH = 960;
const int SCREEN_HEIGHT = 540;
// const int SCREEN_WIDTH = 640;
// const int SCREEN_HEIGHT = 480;

// Global SDL variables
SDL_Window* render_window = nullptr;
struct free_deleter {
    void operator()(void*p) const noexcept
    {
        free(p);
    }
};
struct huge_free_deleter {
    huge_free_deleter() = default;
    huge_free_deleter(size_t size)
        : size(size) {}
    void operator()(void*p) const noexcept
    {
        huge_free(p, size);
    }
    size_t size = 0;
};
std::unique_ptr<uint32_t, huge_free_deleter> back_buffer_memory;
render_target render_targets[phase_count];
// SDL_Surface * window_surface = nullptr;

// 1 means drawing to back_surfaces[1]
size_t back_phase;

float *z_buffer;

SDL_Rect fb_viewport;
SDL_Rect fb_area;

struct texture_info {
    uint32_t const *pixels{};
    glm::vec2 fsz;
    unsigned iw{}, ih{}, lvl0sz{}, pitch{};
    void (*free_fn)(void*) = nullptr;
    int mipmap_levels;
};

using scoped_lock = std::unique_lock<std::mutex>;
using textures_map = std::map<size_t, texture_info>;
std::mutex textures_lock;
textures_map textures;
texture_info *texture;

__attribute__((__noinline__))
void clear_worker(fill_job &job)
{
    for (int y = job.clear_row;
            y < job.fp.height; y += task_workers.size()) {
        float *z_output = job.fp.z_buffer +
            job.fp.pitch * (y + job.fp.top) + job.fp.left;
        //std::numeric_limits<float>::max());
        std::fill_n(z_output, job.fp.width, 1.0f);
    }

    for (int y = job.clear_row;
            y < job.fp.height; y += task_workers.size()) {
        uint32_t *output = job.fp.pixels +
            job.fp.pitch * (y + job.fp.top) + job.fp.left;
        std::fill_n(output, job.fp.width, job.clear_color);
    }
}

template<typename T>
void box_worker(fill_job &job)
{
    using D = typename vecinfo_t<T>::as_float;
    size_t pixel_index = (job.box_y + job.fp.top) * job.fp.pitch;
    // Set up to process this scanline
    uint32_t *pixels = job.fp.pixels + pixel_index;
    float *depths = job.fp.z_buffer + pixel_index;

    // Set up to mask off an 8-bit red green blue or alpha
    T constexpr bytemask = vecinfo_t<T>::vec_broadcast(0xFFU);

    // Set up the new color vectors
    T ca = (job.clear_color >> 24) & bytemask;
    T cr = (job.clear_color >> 16) & bytemask;
    T cg = (job.clear_color >> 8) & bytemask;
    T cb = (job.clear_color) & bytemask;
    // Set up the multiplier for the other side of the alpha blend
    T const na = 255 - ca;

    //  8.24 fixedpoint gives correct digits up to 0.003921568 for 1.0/255.0
    //       so it actually divides by 255.0000408000065
    // 16.16 fixedpoint gives correct digits up to 0.0039215 for 1.0/255.0
    //       so it actually divides by 255.004462578...
    // Multiply instructions that give you the high half are likely to exist
    static constexpr unsigned fixedpoint_shift = 16;
    //static constexpr unsigned fixedpoint_shift = 24;  // negligibly better
    // Fixedpoint multiplier ((1U<<fixedpoint_shift)/255U)

    // Multiply by fp_div then shift right fixedpoint_shift
    // when you want fast divide by 255
    uint32_t constexpr fp_div = (1U << fixedpoint_shift) / UINT8_MAX;

    // Premultiply source alpha into 16.16 fixedpoint
    cr = cr * ca;
    cg = cg * ca;
    cb = cb * ca;

    int x = job.box[0] + job.fp.left;
    int ex = job.box[2] + job.fp.left;

    int from_top_edge = job.box_y - job.box[1];
    int from_bot_edge = job.box[3] - job.box_y;

    if (job.border_table &&
            from_top_edge < (int)job.border_table->size()) {
        int adj = (int)(*job.border_table)[from_top_edge];
        // adj = (job.border_table->size() - 1) -  adj;
        x += adj;
        ex -= adj;
    } else if (job.border_table &&
            from_bot_edge < (int)job.border_table->size()) {
        int adj = (int)(*job.border_table)[from_bot_edge];
        // adj = (job.border_table->size() - 1) -  adj;
        x += adj;
        ex -= adj;
    }

    T mask = ~vecinfo_t<T>::lanemask[x & (vecinfo_t<T>::sz - 1)];

    // Align
    x &= -vecinfo_t<T>::sz;

    for ( ; x < ex; x += vecinfo_t<T>::sz, mask |= -1U) {
        mask &= vecinfo_t<T>::lanemask[sane_min(
            int(vecinfo_t<T>::sz), ex - x)];

        // Save a backup in case we do the mask operation in the last loop
        T const backup = *(T*)(pixels + x);

        D depth_data = *(D*)(depths + x);
        D z_vec = vecinfo_t<D>::vec_broadcast(job.z);
        mask &= z_vec < depth_data;

        // no writeback though, because it is transparent
        // depth_data = vec_blend(depth_data, z_vec, mask);
        // *(D*)(depths + x) = depth_data;

        // Extract the red green and blue so we can blend them separately
        T dr = (backup >> 16) & bytemask;
        T dg = (backup >> 8) & bytemask;
        T db = backup & bytemask;

        // Blend the destination red green blues with source red green blues

        // Scale destination contribution
        dr = dr * na;
        dg = dg * na;
        db = db * na;
        dr *= fp_div;
        dg *= fp_div;
        db *= fp_div;

        // Add the destination and source contributions
        dr += cr;
        dg += cg;
        db += cb;

        // Convert from 16.16 fixedpoint to 32 bit unsigned integer
        dr >>= fixedpoint_shift;
        dg >>= fixedpoint_shift;
        db >>= fixedpoint_shift;

        // Reassemble blended additive primaries into packed 32 bit RGBA
        dr <<= 16;
        dg <<= 8;
        // Force FF result alpha
        db |= 0xFF000000;
        // Merge green into red
        dr |= dg;
        // Merge red and green into blue and alpha
        db |= dr;
        // At boundary conditions with 0's in mask, restore pixels from backup
        db = vec_blend(backup, db, mask); // (db & mask) | (backup & ~mask);
        // Write back
        *(T*)(pixels + x) = db;
    }
}

template<typename T, bool write_color = true,
    bool test_z = true, bool write_z = true>
void fill_mainloop(
    fill_job const& __restrict job,
    std::pair<scaninfo, scaninfo> const& __restrict work,
    scaninfo const& __restrict diff,
    glm::vec2 const& tdiff, float invWidth,
    int i, float n, int x, int y, int pixels)
{
    using F = typename vecinfo_t<T>::as_float;
    //using I = typename vecinfo_t<T>::as_int;

    constexpr size_t vec_sz = vecinfo_t<T>::sz;
    constexpr size_t vec_mask = vecinfo_t<T>::sz - 1;

#if ENABLE_MIPMAP
    // the texture coordinates are actually u/w and v/w and w is 1/w,
    // we have to divide them by 1/w to get back to screenspace
    int mipmap_level = select_mipmap((work.second.t / work.second.p.w) -
        (work.first.t / work.first.p.w), std::abs(invWidth));
    size_t mipmap_offset = indexof_mipmap(mipmap_level);
    uint32_t const * __restrict texture_pixels =
        texture->pixels + mipmap_offset;
    unsigned texture_w = texture->iw >> mipmap_level;
    unsigned texture_h = texture->ih >> mipmap_level;
    float texture_fw = (float)texture_w;
    float texture_fh = (float)texture_h;
#else
    uint32_t const * __restrict texture_pixels = texture->pixels;
    unsigned &texture_w = texture->iw;
    unsigned &texture_h = texture->ih;
    float &texture_fw = texture->fsz.s;
    float &texture_fh = texture->fsz.t;
#endif
    // Adjust by viewport before alignment
    x += job.fp.left;

    // Initialize left mask to fix up round down that is about to happen
    T mask = ~vecinfo_t<T>::lanemask[x & vec_mask];

    // Round down to {vec_sz} pixel aligned boundary (vector size alignment)
    i -= x & vec_mask;
    n -= (x & vec_mask) * invWidth;
    x &= -vec_sz;

    F n_step = invWidth * vecinfo_t<F>::vec_broadcast((float)vec_sz);

    size_t pixel_index = job.fp.pitch * (y + job.fp.top) + x;
    uint32_t * __restrict pixel_io = job.fp.pixels + pixel_index;
    float * __restrict depth_io = job.fp.z_buffer + pixel_index;

    assume((uintptr_t)pixel_io == ((uintptr_t)pixel_io & -vec_sz));
    assume((uintptr_t)depth_io == ((uintptr_t)depth_io & -vec_sz));

    F n_vec = invWidth * vecinfo_t<F>::laneoffs + n;
    for ( ; i < pixels; (i += vec_sz), (x += vec_sz),
            (mask = vecinfo_t<T>::lanemask[vec_sz]),
            (n_vec += n_step)) {
        mask &= vecinfo_t<T>::lanemask[sane_min(int(vec_sz), pixels - i)];

        // Linear interpolate z for depth buffering
        F z_vec = (n_vec * diff.p.z) + work.first.p.z;

        // Fetch z-buffer values (to see if new pixels are closer)
        F depths = *(F*)depth_io; // _mm256_load_ps(depth_io);

        if constexpr (test_z) {
            // Only write pixels that are closer
            mask &= (T)(z_vec < depths);
        }

        if (vec_movemask(mask)) {
            if constexpr (write_z) {
                // Update depth (get it over with so we can do the real work)
                F upd_depth = vec_blend(depths, z_vec, mask);
                *(F*)depth_io = upd_depth;
            }

            if constexpr (write_color) {
                // Fetch existing pixels for masked merge with new pixels
                T upd_pixels = *(T*)pixel_io;

                // Linear interpolate the u/w and v/w
                F v_vec = (n_vec * diff.t.t) + work.first.t.t;
                F u_vec = (n_vec * diff.t.s) + work.first.t.s;

                // Linear interpolate 1/w for perspective correction
                F w_vec = (n_vec * diff.p.w) + work.first.p.w;

                // Perspective correction
                v_vec /= w_vec;
                u_vec /= w_vec;

                F b_vec = (n_vec * diff.c.b) + work.first.c.b;
                F g_vec = (n_vec * diff.c.g) + work.first.c.g;
                F r_vec = (n_vec * diff.c.r) + work.first.c.r;

                // Scale by texture width and height
                v_vec *= texture_fh;//texture->fh;
                u_vec *= texture_fw;//texture->fw;

                // Convert coordinates to integers
                T ty_vec = convert_to<T>(v_vec);
                T tx_vec = convert_to<T>(u_vec);

                // Wrap coordinates to within texture size
                ty_vec &= (texture_h-1);
                tx_vec &= (texture_w-1);

                // Compute pixel offset within texture
                T tex_off = (ty_vec * texture_w) + tx_vec;

    #if DEBUG_MIPMAPS
                T texels = vecinfo_t<T>::vec_broadcast(mipmap_level == 0
                    ? 0xFF0000FF
                    : mipmap_level == 1
                    ? 0xFF00FF00
                    : mipmap_level == 2
                    ? 0xFFFF0000
                    : 0xFFFFFFFF);
                texels = vec_blend(upd_pixels, texels, mask);
    #else
                // Fetch {vec_sz} texels with those {vec_sz} array indices
                T texels = vec_gather(texture_pixels,
                    tex_off, upd_pixels, mask);
    #endif

                // Translate texel to rgb floating
                // point for multiplication
                // with interpolated color
                T texels_ib = rgb_b(texels);
                T texels_ig = rgb_g(texels);
                T texels_ir = rgb_r(texels);

                F texels_fb = convert_to<F>(texels_ib);
                F texels_fg = convert_to<F>(texels_ig);
                F texels_fr = convert_to<F>(texels_ir);

                b_vec *= texels_fb;
                g_vec *= texels_fg;
                r_vec *= texels_fr;

                // Clamp
                constexpr auto bytemax = vecinfo_t<F>::vec_broadcast(255.0f);
                constexpr auto bytemin = vecinfo_t<F>::vec_broadcast(0.0f);

                b_vec = min(b_vec, bytemax);
                g_vec = min(g_vec, bytemax);
                r_vec = min(r_vec, bytemax);

                b_vec = max(b_vec, bytemin);
                g_vec = max(g_vec, bytemin);
                r_vec = max(r_vec, bytemin);

                texels_ib = convert_to<T>(b_vec);
                texels_ig = convert_to<T>(g_vec);
                texels_ir = convert_to<T>(r_vec);

                texels_ib <<= rgba_shift_b;
                texels_ig <<= rgba_shift_g;
                texels_ir <<= rgba_shift_r;

                texels_ib |= texels_ig;
                texels_ib |= texels_ir;

                texels = vec_blend(texels, texels_ib, mask);

                *(T*)pixel_io = texels;
            }
        }

        pixel_io += vec_sz;
        depth_io += vec_sz;
    }
}

using mainloop_pfn = void(*)(
    fill_job const& __restrict job,
    std::pair<scaninfo, scaninfo> const& __restrict work,
    scaninfo const& __restrict diff,
    glm::vec2 const& tdiff, float invWidth,
    int i, float n, int x, int y, int pixels);

template<typename T, mainloop_pfn mainloop = fill_mainloop<T>>
static void fill_worker(size_t worker_nr, fill_job &job)
{
    if (job.frame_barrier)
        return job.frame_barrier->arrive_and_expect(task_workers.size());

    if (job.clear_row >= 0) {
        assume(job.clear_row % task_workers.size() == worker_nr);
        return clear_worker(job);
    }

    if (job.glyph_index != (uint16_t)-1U) {
        assume(job.box_y % task_workers.size() == worker_nr);
        return glyph_worker(job);
    }

    if (job.box_y >= 0) {
        assume(job.box_y % task_workers.size() == worker_nr);
        return box_worker<T>(job);
    }

    std::pair<scaninfo, scaninfo> work = {
        job.fp.edges[job.edge_refs.first.edge_idx] +
        job.fp.edges[job.edge_refs.first.diff_idx] *
        job.edge_refs.first.n,
        job.fp.edges[job.edge_refs.second.edge_idx] +
        job.fp.edges[job.edge_refs.second.diff_idx] *
        job.edge_refs.second.n
    };
    // Original t diff for mipmap selection
    glm::vec2 tdiff = work.second.t - work.first.t;
    // Must floor the x coordinates so polygons fit together perfectly
    work.first.p.x = floorf(work.first.p.x);
    work.second.p.x = floorf(work.second.p.x);
    int y = job.box[1];
    assume(y % task_workers.size() == worker_nr);
    if (y >= job.fp.height) {
        assume(!"Clipping messed up");
        return;
    }
    scaninfo diff = work.second - work.first;
    int pixels = std::abs((int)diff.p.x);
    if (!pixels)
        return;
    float invWidth = 1.0f / diff.p.x;
#if PIXELS_HISTOGRAM
    ++pixels_histogram[sane_max(0,
        sane_min(pixels, (int)pixels_histogram_size-1))];
#endif
    int x = (int)work.first.p.x;
    float n = 0.0f;
    int i = 0;
    // Clamp onto the left edge of the screen,
    // if it started off the left side

    //if (x < 0) {
        // Skip forward to first visible pixel
        n -= sane_min(0.0f, work.first.p.x) * invWidth;
        // All ones if x is negative, else all zeros (sign extend)
        int is_neg = (int)x >> 31;
        // Advance i by the number of skipped pixels
        i -= x & is_neg;
        // Make x 0 if it was negative
        x &= ~is_neg;
    //}
    // Clamp the pixel count to end at the right edge of the screen
    if (x + pixels > job.fp.width)
        pixels = job.fp.width - x;
    if (pixels > 0)
        mainloop(job, work, diff, tdiff, invWidth, i, n, x, y, pixels);
}

// Function to initialize SDL and create the window
bool initSDL(int width, int height) {
    size_t cpu_count = 0;
    char const *ev = getenv("CPUS");
    if (!ev) {
        cpu_count = 4;
    } else if (!strcmp(ev, "max")) {
        cpu_count = std::thread::hardware_concurrency();
        cpu_count = sane_max(size_t(1), cpu_count >> 1);
    } else if (!strcmp(ev, "all")) {
        cpu_count = std::thread::hardware_concurrency();
    } else {
        while (*ev >= '0' && *ev <= '9') {
            cpu_count *= 10;
            cpu_count += *ev - '0';
            ++ev;
        }
    }
    std::cout << "Using " << cpu_count << " CPUs\n";

    cpu_count = std::max(2UL, cpu_count);

    task_workers.reserve(cpu_count);
    for (size_t i = 1; i < cpu_count; ++i) {
#if HAVE_VEC512
        task_workers.emplace_back(fill_worker<vecu32x16>, i - 1, i);
#elif HAVE_VEC256
        task_workers.emplace_back(fill_worker<vecu32x8>, i - 1, i);
#elif HAVE_VEC128
        task_workers.emplace_back(fill_worker<vecu32x4>, i - 1, i);
#endif
    }

    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "SDL could not initialize!"
            " SDL_Error: " << SDL_GetError() << std::endl;
        return false;
    }

    window_w = width;
    window_h = height;

    // Create a resizable window
    render_window = SDL_CreateWindow("SDL Double Buffered Framebuffers",
        SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
        width, height, SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);

    if (render_window == nullptr) {
        std::cerr << "Window could not be created!"
            " SDL_Error: " << SDL_GetError() << std::endl;
        return false;
    }

    fb_area.x = 0;
    fb_area.y = 0;
    fb_area.w = (SCREEN_WIDTH + 63) & -64;
    fb_area.h = SCREEN_HEIGHT;

    fb_viewport.x = 0;
    fb_viewport.y = 0;
    fb_viewport.w = SCREEN_WIDTH;
    fb_viewport.h = SCREEN_HEIGHT;

    // z_buffer.reset((float*)aligned_alloc(sizeof(vecf32x8),
    //      (sizeof(float) * (fb_area.w * fb_area.h + fb_area.w)) &
    //     -sizeof(vecf32x8)));


    size_t frame_pixels = fb_area.w * fb_area.h + 64;
    frame_pixels &= -64;
    size_t alloc_size = sizeof(uint32_t) * (phase_count + 1) * frame_pixels;
    back_buffer_memory.reset((uint32_t*)huge_alloc(alloc_size));
    back_buffer_memory.get_deleter().size = alloc_size;
    //z_buffer = (float*)(back_buffer_memory.get() + 3 * frame_pixels);
    z_buffer = (float*)back_buffer_memory.get();
    for (size_t i = 0; i < phase_count; ++i) {
        render_targets[i].edges.reserve(262144);
        render_targets[i].back_buffer =
            back_buffer_memory.get() + (1 + i) * frame_pixels;
        render_targets[i].back_surface =
            SDL_CreateRGBSurfaceFrom(
                render_targets[i].back_buffer,
                fb_area.w, fb_area.h, 32, sizeof(uint32_t) * fb_area.w,
                rgba(0,0,0xFF,0), rgba(0,0xFF,0,0),
                rgba(0xFF,0,0,0), rgba(0,0,0,0xFF));
        if (render_targets[i].back_surface == nullptr) {
            std::cerr << "Back surface could not be created!"
                " SDL_Error: " << SDL_GetError() << std::endl;
            return false;
        }

        SDL_SetSurfaceBlendMode(
            render_targets[i].back_surface, SDL_BLENDMODE_NONE);
    }

    return true;
}

bool handle_key_down_event(SDL_KeyboardEvent const& e)
{
    bool is_keydown = (e.type == SDL_KEYDOWN);

    switch(e.keysym.sym) {
    case SDLK_ESCAPE:
        if (e.type == SDL_KEYUP)
            return true;
        if (SDL_ShowCursor(SDL_QUERY) != SDL_ENABLE) {
            mouselook_enabled = false;
            SDL_SetRelativeMouseMode(SDL_FALSE);
            SDL_ShowCursor(SDL_ENABLE);
        } else {
            mouselook_enabled = true;
            SDL_ShowCursor(SDL_DISABLE);
            SDL_SetRelativeMouseMode(SDL_TRUE);
        }

        break;

    case SDLK_w:
        // W key
        mouselook_pressed[0] = is_keydown;
        break;

    case SDLK_a:
        // A key
        mouselook_pressed[1] = is_keydown;
        break;

    case SDLK_s:
        // S key
        mouselook_pressed[2] = is_keydown;
        break;

    case SDLK_d:
        // D key
        mouselook_pressed[3] = is_keydown;
        break;

    case SDLK_r:
        // R key
        mouselook_pressed[4] = is_keydown;
        break;

    case SDLK_f:
        // F key
        mouselook_pressed[5] = is_keydown;
        break;

    case SDLK_z:
        if (is_keydown) {
            mouselook_pitch = 0.0f;
            mouselook_yaw = 0.0f;
        }

        break;

    case SDLK_HOME:
        if (is_keydown)
            mouselook_pos = { 0, 0, 0 };
        break;

    }
    return true;
}

bool handle_mouse_wheel_event(SDL_MouseWheelEvent const& e)
{
    // handle the event here

    return true;
}

bool handle_mouse_motion_event(SDL_MouseMotionEvent const& e)
{
    if (mouselook_enabled) {
        mouselook_yaw += e.xrel * mouselook_yaw_scale;
        mouselook_pitch += e.yrel * mouselook_pitch_scale;
        while (mouselook_yaw > M_PIf)
            mouselook_yaw -= M_PIf * 2.0f;
        while (mouselook_yaw < -M_PIf)
            mouselook_yaw += M_PIf * 2.0f;

        if (mouselook_pitch > M_PI_2f)
            mouselook_pitch = M_PI_2f;
        if (mouselook_pitch < -M_PI_2)
            mouselook_pitch = -M_PI_2f;
    }
    return true;
}

bool handle_window_event(SDL_WindowEvent const& e)
{
    switch (e.event) {
    case SDL_WINDOWEVENT_RESIZED:
        window_w = e.data1;
        window_h = e.data2;
        return true;

    case SDL_WINDOWEVENT_FOCUS_GAINED:
        // std::cerr << "Focus gained\n";
        // mouselook_enabled = true;
        // SDL_SetRelativeMouseMode(SDL_TRUE);
        // SDL_ShowCursor(SDL_FALSE);
        return true;

    case SDL_WINDOWEVENT_FOCUS_LOST:
        std::cerr << "Focus lost\n";
        mouselook_enabled = false;
        SDL_SetRelativeMouseMode(SDL_FALSE);
        SDL_ShowCursor(SDL_ENABLE);
        // Mark everything as not pressed
        std::fill(std::begin(mouselook_pressed),
            std::end(mouselook_pressed), false);
        return true;
    }
    return true;
}

// Function to handle window events
bool handleEvents() {
    SDL_Event e;
    while (SDL_PollEvent(&e)) {
        switch (e.type) {
        case SDL_QUIT:
            return false;

        case SDL_WINDOWEVENT:
            handle_window_event(e.window);
            break;

        case SDL_MOUSEMOTION:
            handle_mouse_motion_event(e.motion);
            break;

        case SDL_MOUSEWHEEL:
            handle_mouse_wheel_event(e.wheel);
            break;

        case SDL_KEYDOWN:   // fall through
        case SDL_KEYUP:
            handle_key_down_event(e.key);
            break;

        }
    }

    return true;
}

// Function to render to the current framebuffer
void render(render_ctx *ctx)
{
    size_t phase = back_phase;
    size_t prev_phase = (phase == 0)
        ? (phase_count - 1)
        : (phase - 1);
    render_target& fp = render_targets[phase];
    if (SDL_MUSTLOCK(fp.back_surface)) {
        fp.back_surface_locked = true;
        SDL_LockSurface(fp.back_surface);
    }

    // std::cout << "Rendering phase " << phase << '\n';
    int pitch = fb_area.w;
    uint32_t *pixels = fp.back_buffer;
    //time_point render_st = clk::now();
    assume(pitch < std::numeric_limits<uint16_t>::max());
    //std::cout << "Drawing to " << pixels << "\n";
    // fp.width = SCREEN_WIDTH - 150;
    // fp.height = SCREEN_HEIGHT - 150;
    fp.width = SCREEN_WIDTH;
    fp.height = SCREEN_HEIGHT;
    fp.pitch = (uint16_t)pitch;
    fp.pixels = pixels;
    fp.top = 0;
    fp.left = 0;
    // fp.top = 75;
    // fp.left = 75;
    fp.z_buffer = z_buffer;

    user_frame(fp, ctx);//.subset(42, 33, 420, 420));

    // std::cout << "Enqueue barrier at end of phase " << phase << "\n";
    //time_point qbarrier_st = clk::now();
    fp.present_barrier.reset();
    for (fill_task_worker &worker : task_workers)
        worker.emplace(true, fp, &fp.present_barrier);
    //time_point qbarrier_en = clk::now();


    if (fp.back_surface_locked) {
        fp.back_surface_locked = false;
        SDL_UnlockSurface(fp.back_surface);
    }

    // Wait for the previous completion barrier to finish

    //time_point wait_st = clk::now();
    render_targets[prev_phase].present_barrier.wait();
    //time_point wait_en = clk::now();

    if (render_targets[prev_phase].back_surface_locked) {
        render_targets[prev_phase].back_surface_locked = false;
        SDL_UnlockSurface(render_targets[prev_phase].back_surface);
    }

    SDL_Surface *window_surface =
        SDL_GetWindowSurface(render_window);
    assume(window_surface != nullptr);

    SDL_Rect src = fb_viewport;
    SDL_Rect dest = fb_viewport;

#if RENDER_ZBUFFER
    std::vector<uint32_t> debug = [&] {
        std::vector<uint32_t> debug;
        size_t fb_sz = src.w * src.h;
        debug.reserve(fb_sz);
        std::transform(z_buffer, z_buffer + fb_sz,
            std::back_inserter(debug), [&](float fz) {
                if (fz == INFINITY)
                    fz = 255.0f;
                return sane_min(255U, sane_max(0U, unsigned(fz * 16.0))) *
                    0x10101U | 0xFF000000U;
            });
        return debug;
    }();

    SDL_Surface *z_buffer_surface = SDL_CreateRGBSurfaceFrom(debug.data(),
        src.w, src.h, 32, sizeof(uint32_t) * fb_area.w,
        rgba(0,0,0xFF,0),
        rgba(0,0xFF,0,0),
        rgba(0xFF,0,0,0),
        rgba(0,0,0,0xFF));

    int status = blit_to_screen(z_buffer_surface, src,
        window_surface, dest);
    assume(status == 0);
    SDL_FreeSurface(z_buffer_surface);
#else
    // std::cout << "Presenting " << phase << "\n";
    int status = blit_to_screen(
        render_targets[prev_phase].back_surface, src,
        window_surface, dest);
    //std::this_thread::sleep_for(std::chrono::milliseconds(200));

#endif

    status = SDL_UpdateWindowSurface(render_window);
    assume(status == 0);

    static time_point last_frame;
    time_point fps_time = clk::now();
    duration fps_since = fps_time - last_frame;
    last_frame = fps_time;
    uint64_t fps_ns = std::chrono::duration_cast<
        std::chrono::nanoseconds>(fps_since).count();
    last_fps = UINT64_C(1000000000) / fps_ns;
    smooth_fps = (smooth_fps * 8 + last_fps * 3) / 11;
    // std::cout << last_fps << " fps\n";

    // Click over to the next phase
    back_phase = phase = ((phase + 1) < phase_count) ? (phase + 1) : 0;

#if 0
    static int div;
    if (++div == 120) {
        uint64_t render_us = std::chrono::duration_cast<
            std::chrono::microseconds>(wait_st - render_st).count();
        uint64_t wait_us = std::chrono::duration_cast<
            std::chrono::microseconds>(wait_en - wait_st).count();
        uint64_t qbarrier_us = std::chrono::duration_cast<
            std::chrono::microseconds>(qbarrier_en - qbarrier_st).count();
        std::cout << render_us << " µs render"
            ", " << qbarrier_us << " µs qbarrier"
            ", " << wait_us << " µs wait\n";
        int worker_nr = 0;
        for (fill_task_worker &worker : task_workers) {
            uint64_t t = worker.wait_us + worker.work_us;
            std::cout << "#" << worker_nr <<
                " peak " << worker.get_reset_peak() <<
                " waits " << worker.waits <<
                " wait_µs " << worker.wait_us <<
                " work_µs " << worker.work_us <<
                " (" << (t ? 100 * worker.work_us / t : -1) << "%)\n";
            worker.waits = 0;
            worker.wait_us = 0;
            worker.work_us = 0;

            ++worker_nr;
        }
        std::cout << "----\n";
        div = 0;
    }
#endif

    fp.edges.clear();
    fp.edges_lookup.clear();
}

static int blit_to_screen(SDL_Surface *surface, SDL_Rect &src,
    SDL_Surface *window_surface, SDL_Rect &dest)
{
    int status;

    if (window_w != dest.w || window_h != dest.h)
    {
        float xscale = (float)window_w / dest.w;
        float yscale = (float)window_h / dest.h;
        float scale = std::min(xscale, yscale);
        int new_w = dest.w * scale;
        int new_h = dest.h * scale;
        int xunused = window_w - new_w;
        int yunused = window_h - new_h;
        int xofs = xunused >> 1;
        int yofs = yunused >> 1;
        SDL_Rect scaled_rect{
            xofs,
            yofs,
            new_w,
            new_h};
        status = SDL_BlitScaled(
            surface,
            &src, window_surface, &scaled_rect);
    }
    else
    {
        status = SDL_BlitSurface(
            surface,
            &src, window_surface, &dest);
    }
    assume(status == 0);
    return status;
}

void bind_texture(size_t binding)
{
    std::unique_lock<std::mutex> lock(textures_lock);
    texture = &textures[binding];
}

bool delete_texture(size_t binding)
{
    std::unique_lock<std::mutex> lock(textures_lock);
    textures_map::iterator it = textures.find(binding);
    if (it == textures.end())
        return false;
    if (texture == &it->second)
        texture = nullptr;
    textures.erase(it);
    return true;
}

int select_mipmap(glm::vec2 const& diff_of_t, float invScreenspaceWidth)
{
    // How many texels are covered in the horizontal
    float s_texels = std::abs(diff_of_t.s * texture->fsz.s);
    // How many texels are covered in the vertical
    float t_texels = std::abs(diff_of_t.t * texture->fsz.t);
    // Divide by number of screenspace texels on both axes
    float s_texels_per_pixel = s_texels * invScreenspaceWidth;
    float t_texels_per_pixel = t_texels * invScreenspaceWidth;
    // std::cerr << diff_of_t.s <<
    //     ' ' << diff_of_t.t <<
    //     ' ' << invScreenspaceWidth <<
    //     ' ' << s_texels <<
    //     ' ' << t_texels <<
    //     ' ' << s_texels_per_pixel <<
    //     ' ' << t_texels_per_pixel <<
    //     '\n';
    // d is the approximate number of texels per screen pixel in the span
    float d = std::max(s_texels_per_pixel, t_texels_per_pixel);
    // Level is max(0, -floor(log2(d)))
    int level = std::max(0, std::ilogbf(d));
    // Clamp to number of mipmap levels that exist
    level = std::min(texture->mipmap_levels - 1, level);
    // if (level)
    //     std::cerr << "Using level " << level << "\n";
    return level;
}

uint32_t indexof_mipmap(int level)
{
    assume(level < 16);
    // Fill the value with the sign bit of (level - 1) using
    // sign extension, then invert it, to get all ones
    // when level is > 0
    int mask = ~((level - 1) >> 31);
    // How many bits of the 1010101... pattern are kept for the
    // fixedpoint multiply by 1.010101... later
    // If it is going to go negative, wipe it to zero
    // to avoid negative shift later
    int kept = 2 * (level - 1) & mask;
    // Shift fixedpoint 1.01010101... pattern
    // so the correct number of digits are shifted off the right
    // to leave the correct number of powers of 4^-x after the 1.
    // and wipe out the multiplier to zero if level 0
    uint32_t mul = (0x55555555 >> ((32 - kept) - 2)) & mask;
    // Do the fixedpoint multiply and do the right shift
    // to get it back to the whole part, given how much
    // we scaled it up, when we put those 01 binary digit
    // pairs after the decimal
    return (uint32_t)(((uint64_t)texture->lvl0sz * mul) >> kept);
}

void set_light_enable(render_ctx *ctx,
    size_t light_nr, bool enable)
{
    if (light_nr < ctx->light_count)
        ctx->light_enable[light_nr] = enable;
}

void set_light_pos(render_ctx *ctx,
    size_t light_nr, glm::vec4 const &pos)
{
    if (light_nr < ctx->light_count) {
        if (pos.w) {
            // It is a position, do the transform,
            // but preserve their w (positional flag)
            ctx->light_position[light_nr] = glm::vec4(
                glm::vec3(view_mtx_stk.back() * pos), pos.w);
        } else {
            // It is a direction
            ctx->light_position[light_nr] = glm::vec4(
                glm::mat3(view_mtx_stk.back()) * glm::vec3(pos), pos.w);
        }
    }
}

void set_light_spot(render_ctx *ctx,
    size_t light_nr,
    glm::vec3 const& dir, float cutoff, float exponent)
{
    if (light_nr < ctx->light_count) {
        ctx->light_spot_dir[light_nr] =
            glm::mat3(view_mtx_stk.back()) * dir;
        ctx->light_info[light_nr].x = cutoff;
        ctx->light_info[light_nr].y = exponent;
    }
}

void set_light_diffuse(render_ctx *ctx,
    size_t light_nr, glm::vec3 color)
{
    if (light_nr < ctx->light_count)
        ctx->light_diffuse[light_nr] = color;
}

void set_light_specular(render_ctx *ctx,
    size_t light_nr, glm::vec3 color, float shininess)
{
    if (light_nr < ctx->light_count) {
        ctx->light_specular[light_nr] = color;
        ctx->light_info[light_nr].z = shininess;
    }
}

void set_texture(render_ctx *ctx,
    uint32_t const *incoming_pixels,
    int incoming_w, int incoming_h, int incoming_pitch,
    int incoming_levels, void (*free_fn)(void *p))
{
    if (texture->pixels && texture->free_fn)
        texture->free_fn((void*)texture->pixels);
    texture->pixels = incoming_pixels;
    texture->iw = incoming_w;
    texture->ih = incoming_h;
    texture->lvl0sz = incoming_w * incoming_h;
    texture->fsz.s = incoming_w;
    texture->fsz.t = incoming_h;
    texture->pitch = incoming_pitch;
    texture->free_fn = free_fn;

    texture->mipmap_levels = incoming_levels;
}

int setup(render_ctx *ctx, int width, int height);
void cleanup(render_ctx *ctx, int width, int height);

bool measure;
std::vector<std::string> command_line_files;

int main(int argc, char const *const *argv)
{
    std::unique_ptr<render_ctx> ctx =
        std::make_unique<render_ctx>();

    for (int i = 1; i < argc; ++i)
    {
        if (!strcmp(argv[i], "--measure")) {
            measure = true;
            continue;
        }

        if (!std::filesystem::exists(argv[i])) {
            std::cerr << "File not found: " << argv[i] << "\n";
            return EXIT_FAILURE;
        }

        command_line_files.emplace_back(argv[i]);
        continue;
    }

    if (!initSDL(SCREEN_WIDTH, SCREEN_HEIGHT))
        return 1;

    text_init(24);

    if (setup(ctx.get(), SCREEN_WIDTH, SCREEN_HEIGHT) == EXIT_FAILURE)
        return EXIT_FAILURE;

    atexit(SDL_Quit);

    // Main loop
    bool quit = false;
    uint64_t count = 0;
    while (!quit) {
        if (!handleEvents())
            break;

        // Render to the current framebuffer
        render(ctx.get());

        ++count;
        if (measure && count >= 400)
            break;
    }

    cleanup(ctx.get(), SCREEN_WIDTH, SCREEN_HEIGHT);

    for (textures_map::value_type const &tex : textures) {
        if (tex.second.free_fn)
            tex.second.free_fn((void*)tex.second.pixels);
    }

    task_workers.clear();

    // Cleanup and exit
    for (size_t i = 0; i < phase_count; ++i) {
        SDL_FreeSurface(render_targets[i].back_surface);
        render_targets[i].back_surface = nullptr;
    }
    //SDL_DestroyRenderer(gRenderer);
    SDL_DestroyWindow(render_window);
    SDL_Quit();

    return 0;
}

template<typename T>
void simple_edge(int x1, int y1, int x2, int y2, T callback)
{
    // Make sure it always starts at the top
    if (y2 < y1) {
        std::swap(x1, x2);
        std::swap(y1, y2);
    }
    int xdiff = x2 - x1;
    int ydiff = y2 - y1;

    if (ydiff != 0) {
        for (int y = y1; y < y2; ++y) {
            int partial = xdiff * (y - y1) / ydiff;
            int x = x1 + partial;
            callback(x, y);
        }
    }
}

unsigned dedup_edge(render_target &fp, scaninfo const& v0)
{
    std::vector<scaninfo> &edge_table = fp.edges;

    std::pair<std::unordered_map<scaninfo, unsigned>::iterator, bool> ins =
        fp.edges_lookup.emplace(v0, (unsigned)edge_table.size());
    if (ins.second)
        edge_table.emplace_back(v0);
    return ins.first->second;
}

template<typename C>
void interp_edge(render_target& fp,
    scaninfo const& v0, scaninfo const& v1, C&& callback)
{
    float fheight = v1.p.y - v0.p.y;
    scaninfo const *p0;
    scaninfo const *p1;
    if (fheight >= 0) {
        p0 = &v0;
        p1 = &v1;
    } else {
        p0 = &v1;
        p1 = &v0;
        fheight = -fheight;
    }
    assume(fheight >= 0.0f);
    if (fheight < 1.0f)
        return;
    scaninfo diff = *p1 - *p0;
    float n = 0;
    float invHeight = 1.0f / fheight;
    // Adjust n by how many scanlines are off the top
    float nadj = sane_min(p0->p.y * invHeight, 0.0f);
    n -= nadj;
    int y = (int)(p0->p.y);
    int ey = (int)(p0->p.y + fheight);
    // Wipe y to zero if it is negative
    y &= -(y >= 0);

    unsigned edge_idx = dedup_edge(fp, *p0);
    unsigned diff_idx = dedup_edge(fp, diff);
    for (ey = sane_min(ey, fp.height-1);
        y <= ey; ++y, n += invHeight)
    {
        callback(y, edge_idx, diff_idx, n);
    }
}

std::vector<std::vector<fill_job>> fill_job_batches;

void ensure_scratch(size_t height)
{
    if (scanconv_scratch.empty()) {
        scanconv_scratch.resize(height);
        fill_job_batches.resize(task_workers.size());
    }

    // Low 2 bits as span flags
    draw_nr += 4;
    // At the middle, clear with start, at the start, clear with middle
    if ((draw_nr & 0x7fffffff) == 0) {
        // once every 524284 draws
        uint32_t fill = draw_nr ^ 0x80000000U;
        for (size_t i = 0; i < height; ++i)
            scanconv_scratch[i].used = fill;
        // Skip over draw_nr 0
        draw_nr += (!draw_nr) << 2;
    }
}

std::mutex border_lookup_lock;
std::map<int, std::vector<float>> border_lookup;

void fill_box(render_target& fp, int sx, int sy,
    int ex, int ey, float z,
    uint32_t color, int border_radius)
{
    std::vector<float> *border_table = nullptr;
    if (border_radius) {
        std::unique_lock<std::mutex> lock(border_lookup_lock);
        border_table = &border_lookup[border_radius];
        if (border_table->empty()) {
            border_table->resize(border_radius);
            float r = (float)border_radius;
            float s = 0;
            for (int i = 0; i < border_radius; ++i, ++s) {
                border_table->at(i) = (border_radius - 1) -
                    std::sqrt((r * r) - (s - r) * (s - r));
            }
        }
    }

    size_t slot = sy % task_workers.size();
    for (int y = sy; y <= ey; ++y) {
        assume(y % task_workers.size() == slot);
        task_workers[slot].emplace(false,
            fp, y, sx, sy, ex, ey, z, color, border_table);

        ++slot;
        slot &= -(slot < task_workers.size());
    }
}

void texture_triangle(render_target& fp,
    scaninfo v0, scaninfo v1, scaninfo v2)
{
    ensure_scratch(fp.height);
    size_t dir;

    glm::vec3 p0{v0.p};
    glm::vec3 p1{v1.p};
    glm::vec3 p2{v2.p};
    bool backfacing = (glm::cross(p1 - p0, p2 - p0).z < 0);

    int maxy = INT_MIN;
    int miny = INT_MAX;

    auto draw = [&](int y, unsigned edge_idx, unsigned diff_idx, float n) {
        maxy = std::max(maxy, y);
        miny = std::min(miny, y);
        auto &row = scanconv_scratch[y];
        row.range[dir ^ backfacing] = {
            edge_idx,
            diff_idx,
            n
        };
        // if draw_nr is wrong,
        // then
        //   wipe whole thing to zero
        // otherwise
        //   keep the whole thing
        row.used = (row.used &
            -((row.used & -4) == draw_nr)) |
            draw_nr |
            (1U << dir);
    };

    dir = v0.p.y < v1.p.y ? 1 : 0;
    interp_edge(fp, v0, v1, draw);
    dir = v1.p.y < v2.p.y ? 1 : 0;
    interp_edge(fp, v1, v2, draw);
    dir = v2.p.y < v0.p.y ? 1 : 0;
    interp_edge(fp, v2, v0, draw);

    if (maxy == miny)
        return;

    int y = sane_max(0, miny);
    //size_t slot = y % task_workers.size();
    for (int e = sane_min((int)fp.height, maxy); y < e; ++y) {
        auto &row = scanconv_scratch[y];

        uint32_t used_value = row.used;
        row.used = 0;
        if (used_value != (draw_nr | 3))
            continue;

        size_t slot = y % task_workers.size();
        //fill_job &job =
        fill_job_batches[slot].emplace_back(
            fp, y, row.range[0], row.range[1], back_phase);
    }

    for (size_t i = 0; i < fill_job_batches.size(); ++i) {
        if (!fill_job_batches[i].empty()) {
            task_workers[i].add(fill_job_batches[i].data(),
                fill_job_batches[i].size(), false);
            fill_job_batches[i].clear();
        }
    }
}


static char const *plane_names[] = {
    "-X",
    "-Y",
    "-Z",
    "+X",
    "+Y",
    "+Z"
};

int clipping_needed(glm::vec4 const& v)
{
    bool clipNX = (v.x + v.w < 0.0f);  // left plane (-X)
    bool clipNY = (v.y + v.w < 0.0f);  // top plane ()
    bool clipNZ = (v.z + v.w < 0.0f);  // near plane
    bool clipPX = (v.x - v.w > 0.0f);  // right plane
    bool clipPY = (v.y - v.w > 0.0f);  // bottom plane
    bool clipPZ = (v.z - v.w > 0.0f);  // far plane
    int clipMask = clipNX | (clipNY << 1) | (clipNZ << 2) |
        (clipPX << 3) | (clipPY << 4) | (clipPZ << 5);
    return clipMask;
}

// Find intercept on positive plane
float intercept(float x1, float w1, float x2, float w2)
{
    float t = (w1 - x1) / ((w1 - x1) - (w2 - x2));
    assume(t >= 0.0f && t <= 1.0f);
    return t;
}

// Find intercept on negative plane
float intercept_neg(float x1, float w1, float x2, float w2)
{
    float t = (w1 + x1) / ((w1 + x1) - (w2 + x2));
    assume(t >= 0.0f && t <= 1.0f);
    return t;
}

#define DEBUG_CLIPPING 0

static std::vector<int> clip_masks;
static glm::mat4 combined_xform;

std::vector<glm::mat4> view_mtx_stk{1};
std::vector<glm::mat4> proj_mtx_stk{1};

void set_transform(glm::mat4 const& proj_mtx,
    glm::mat4 const& view_mtx)
{
    glm::mat4 vmt = view_mtx;
    vmt = glm::transpose(vmt);
    mouselook_px = glm::vec3{vmt[0]};
    mouselook_py = glm::vec3{vmt[1]};
    mouselook_pz = glm::vec3{vmt[2]};
    combined_xform = proj_mtx * view_mtx;
}


static glm::vec3 reflect(glm::vec3 const& incident, glm::vec3 const& normal)
{
    return incident - 2.0f * glm::dot(incident, normal) * normal;
}

static void light_vertex(render_ctx *ctx, scaninfo &v)
{
    glm::vec3 color = ctx->light_ambient;

    for (size_t light = 0; light < ctx->light_count; ++light) {
        if (!ctx->light_enable[light])
            continue;
        if (ctx->light_position[light].w) {
            // Light has a position
            glm::vec3 vec_to_light = glm::normalize(
                glm::vec3(ctx->light_position[light]) - glm::vec3(v.p));

            // Diffuse
            float diffuse_cos = std::max(0.0f, glm::dot(vec_to_light, v.n));

            float &spotlight_cutoff = ctx->light_info[light].x;
            float &spotlight_exponent = ctx->light_info[light].y;
            float &light_shininess = ctx->light_info[light].z;

            if (spotlight_cutoff != 1.0f) {
                // Spot light cutoff
                if (diffuse_cos > spotlight_cutoff) {
                    // Outside umbra (lit volume)
                    diffuse_cos = 0;
                } else {
                    // Inside penumbra (fading volume)
                    float spotlight_effect = std::pow(
                        (diffuse_cos / spotlight_cutoff),
                        spotlight_exponent);
                    color += ctx->light_diffuse[light] *
                        spotlight_effect;
                }
            } else {
                // Omnidirectional light
                color += ctx->light_diffuse[light] * diffuse_cos;
            }

            // Compute the specular reflection
            glm::vec3 reflection = reflect(-vec_to_light, v.n);

            float specular_cos = std::pow(std::max(reflection.z, 0.0f),
                light_shininess);

            color += ctx->light_specular[light] * specular_cos;
        } else {
            // Light is at infinity and only has a direction
        }
    }

    v.c *= color;
}

static scaninfo *scaninfo_transform(
    render_target& fp, render_ctx *ctx,
    scaninfo const *vinp, size_t count)
{
    fp.reset_clip_scratch(count);
    size_t const parallel_threshold = 4096;
    if (count < parallel_threshold) {
        for (size_t i = 0; i < count; ++i) {
            scaninfo s = view_mtx_stk.back() * vinp[i];
            light_vertex(ctx, s);
            s = proj_mtx_stk.back() * s;
            fp.vinp_scratch.emplace_back(s);
        }
    } else {
        std::atomic_size_t next_chunk{0};
        size_t constexpr chunk_sz = parallel_threshold;
        fp.vinp_scratch.resize(count);
        auto worker = [&] {
            // Keep going if it looks like there is another
            // chunk available for processing
            while (next_chunk < count) {
                // Take chunk and advance it by a chunk size
                size_t chunk_start = next_chunk.fetch_add(
                    chunk_sz, std::memory_order_acq_rel);
                // If it went way over, undo what we did
                if (chunk_start >= count + chunk_sz) {
                    next_chunk.fetch_add(-chunk_sz,
                        std::memory_order_acq_rel);
                }
                if (chunk_start < count) {
                    for (size_t i = chunk_start;
                            i < count && i < chunk_start + chunk_sz; ++i) {
                        fp.vinp_scratch[i] = view_mtx_stk.back() * vinp[i];
                        light_vertex(ctx, fp.vinp_scratch[i]);
                        fp.vinp_scratch[i] = proj_mtx_stk.back() * fp.vinp_scratch[i];
                    }
                }
            }
        };
        auto cpu_count = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        threads.reserve(cpu_count - 1);
        for (size_t i = 1; i < cpu_count; ++i)
            threads.emplace_back(worker);
        worker();
        for (auto &worker : threads)
            worker.join();
    }
    return fp.vinp_scratch.data();
}

// Simpler to just say them all twice than to wrap around the index
static float constexpr glm::vec4::*plane_lookup[] = {
    &glm::vec4::x,
    &glm::vec4::y,
    &glm::vec4::z,
    &glm::vec4::x,
    &glm::vec4::y,
    &glm::vec4::z
};

static float constexpr plane_sign[] = {
    1.0f,
    1.0f,
    1.0f,
    -1.0f,
    -1.0f,
    -1.0f
};

void texture_polygon(render_target& fp, render_ctx *ctx,
    scaninfo const *user_verts, size_t count)
{
    scaninfo *verts = scaninfo_transform(
        fp, ctx, user_verts, count);

    // Do 0-to-1, 1-to-2, 2-to-0
    for (size_t plane = 0; plane < 6 && fp.vinp_scratch.size() >= 3; ++plane,
            // Swap in/out, clear out
            fp.vinp_scratch.swap(fp.vout_scratch), fp.vout_scratch.clear(),
            // Switch to the new scratch input
            (verts = fp.vinp_scratch.data()), (count = fp.vinp_scratch.size())) {
        float glm::vec4::*field;
        clip_masks.clear();
        clip_masks.reserve(count);
        std::transform(verts, verts + count,
            std::back_inserter(clip_masks), [](auto const& v) {
                return clipping_needed(v.p);
            });

        int need_union = 0;
        int need_intersection = 0b111111;
        for (int mask : clip_masks) {
            need_union |= mask;
            need_intersection &= mask;
        }

        // If every vertex is clipped by one of the planes, it's culled
        if (need_intersection) {
            // TODO: might want to increment a counter
            return;
        }

        // If nothing even touched a clip plane, skip all that
        if (!need_union) {
            // TODO: might want to increment a counter
            break;
        }

        for (int edge = 0; edge < (int)count; ++edge) {
            int next = (edge + 1) & -(edge < (int)count - 1);
            scaninfo const &vst = verts[edge];
            scaninfo const &ven = verts[next];
            int edge_mask = clip_masks[edge];
            int next_mask = clip_masks[next];

#if DEBUG_CLIPPING
            auto dump = [&](float t, char const *comment) {
                glm::vec4 intercept = (ven.p - vst.p) * t + vst.p;
                std::cout << "plane=" << plane << ' ' <<
                    plane_names[plane] << ' ';
                print_vector(std::cout, vst.p) << " to ";
                print_vector(std::cout, ven.p) << " intercepts "
                    " t=" << t << " at ";
                print_vector(std::cout, intercept) << " " <<
                    comment << "\n";
            };
#endif

            // Do the XYZ planes twice, once for - and once for +
            field = plane_lookup[plane];
            // Adjust the intercept function for -w or +w, appropriately
            float clip_sign = plane_sign[plane];

            // If it crossed from clipped to not clipped or vice versa
            int diff_mask = edge_mask ^ next_mask;

            if (diff_mask & (1 << plane)) {
                // The edge crossed the plane
                // Starts outside clip plane
                float t = intercept_neg(
                    vst.p.*field * clip_sign, vst.p.w,
                    ven.p.*field * clip_sign, ven.p.w
                    );
#if DEBUG_CLIPPING
                dump(t, "(starts outside)");
#endif

                if (!(edge_mask & (1 << plane)))
                    fp.vout_scratch.emplace_back(vst);
                scaninfo new_vertex;
                new_vertex = ((ven - vst) * t) + vst;
                fp.vout_scratch.emplace_back(new_vertex);
            } else if (!(edge_mask & (1 << plane))) {
                fp.vout_scratch.emplace_back(vst);
            }
        }
    }

    // Project to screenspace
    float frame_width = fp.width;
    float frame_height = fp.height;
    for (size_t i = 0; i < count; ++i) {
        scaninfo &vertex = verts[i];
        // Divide x, y, and z by w, and store inverse w in w
        float oow = 1.0f / vertex.p.w;
        vertex.p.x *= oow;
        vertex.p.y *= oow;
        vertex.p.z *= oow;
        // +y is up, so flip it vertically
        vertex.p.y *= -1.0f;
        vertex.p.w = oow;
        vertex.t.s *= oow;
        vertex.t.t *= oow;
        // Scale -1.0-to-1.0 range down to -0.5-to-0.5
        vertex.p.x *= 0.5f;
        vertex.p.y *= 0.5f;
        //vertex.p.z *= 0.5f;
        // Shift -0.5-to-0.5 over to 0-to-1.0
        vertex.p.x += 0.5f;
        vertex.p.y += 0.5f;

        // Leave z as -1 to +1 to get an extra bit of precision
        //vertex.p.z += 0.5f;
        // Scale to screen coordinate
        vertex.p.x *= frame_width;
        vertex.p.y *= frame_height;
    }

    // Make a fan from the polygon vertices
    for (size_t i = 1; i + 1 < count; ++i)
        texture_triangle(fp, verts[0], verts[i], verts[i+1]);
}

void texture_polygon(render_target& frame, render_ctx *ctx,
    std::vector<scaninfo> vinp)
{
    texture_polygon(frame, ctx, vinp.data(), vinp.size());
}

void flip_colors(uint32_t *pixels, int imgw, int imgh)
{
    for (size_t i = 0, e = imgw * imgh; i < e; ++i) {
        uint32_t texel = pixels[i];
        uint8_t r = rgb_r(texel);
        uint8_t g = rgb_g(texel);
        uint8_t b = rgb_b(texel);
        uint8_t a = rgb_a(texel);
        texel = rgba(b, g, r, a);
        pixels[i] = texel;
    }
}

void parallel_clear(render_target &frame, uint32_t color)
{
    // One huge work item does all of the scanlines
    // that alias onto that worker, in one go
    for (size_t slot = 0; slot < task_workers.size(); ++slot)
        task_workers[slot].emplace(false,
             frame, slot, color);
}

void print_matrix(char const *title, std::ostream &out, glm::mat4 const& pm)
{
    out << title << "=[" <<
        '['<<pm[0][0]<<','<<pm[1][0]<<','<<pm[2][0]<<','<<pm[3][0]<<"],\n" <<
        '['<<pm[0][1]<<','<<pm[1][1]<<','<<pm[2][1]<<','<<pm[3][1]<<"],\n" <<
        '['<<pm[0][2]<<','<<pm[1][2]<<','<<pm[2][2]<<','<<pm[3][2]<<"],\n" <<
        '['<<pm[0][3]<<','<<pm[1][3]<<','<<pm[2][3]<<','<<pm[3][3]<<"]\n"
        "]\n";
}

size_t hash_bytes(void const *data, size_t size)
{
    return std::hash<std::string_view>()(
        std::string_view((char const *)data, size));
}

void texture_elements(render_target &fp, render_ctx *ctx,
    scaninfo const *user_vertices, size_t vertex_count,
    uint32_t const *elements, size_t element_count)
{
    // Transform into clipping scratch
    scaninfo *vertices = scaninfo_transform(fp, ctx, user_vertices, vertex_count);
    // Take it and give clipping scratch the empty vector
    std::vector<scaninfo> xfv;
    // todo: 64 is a bit arbitrary, consider updating after analysis
    xfv.reserve(64);
    xfv.swap(fp.vinp_scratch);

    size_t constexpr vertices_per_triangle = 3;

    for (size_t i = 0; i + vertices_per_triangle <= element_count;
            i += vertices_per_triangle) {
        texture_triangle(fp, vertices[elements[i]],
            vertices[elements[i+1]], vertices[elements[i+2]]);
    }
}
