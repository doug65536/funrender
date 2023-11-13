#include "funsdl.h"
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
#include <mutex>
#include <vector>
#include <deque>
#include <condition_variable>
#include <functional>
#include <libpng16/png.h>
#include "affinity.h"

//  7: 0 blue
// 15: 8 green
// 23:16 blue
// 31:24 alpha

// packed RGBA conversion readability utilities
#define rgba(r,g,b,a)       ((b) | ((g) << 8) | ((r) << 16) | ((a) << 24))
#define rgb0(r,g,b)         (rgba((r), (g), (b), 0))
#define rgb(r,g,b)          (rgba((r), (g), (b), 0xff))
#define rgb_a(pixel)        (((pixel)>>24)&0xFF)
#define rgb_r(pixel)        (((pixel)>>16)&0xFF)
#define rgb_g(pixel)        (((pixel)>>8)&0xFF)
#define rgb_b(pixel)        (((pixel))&0xFF)

font_data text_init(int size, int first_cp = 1, int last_cp = 0xFFFF);

std::vector<scaninfo> edges[2];
std::unordered_map<scaninfo, unsigned> edges_lookup[2];

#if PIXELS_HISTOGRAM
// size_t const pixels_histogram_size = 1024;
// std::atomic_ulong pixels_histogram[pixels_histogram_size];
#endif

static font_data glyphs;

#if PIXELS_HISTOGRAM
void dump_histogram(int height)
{
    unsigned long peak = 0;
    for (std::atomic_ulong const &ent : pixels_histogram)
        peak = std::max(peak, ent.load(std::memory_order_acquire));

    // std::vector<char> text_canvas(pixels_histogram_size * height, ' ');
    // for (size_t x = 0; x < pixels_histogram_size; ++x) {
    //     unsigned long value =
    //         pixels_histogram[x].load(std::memory_order_acquire);
    //     int scaled = (long long)std::min(peak, value) * height / peak;
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

template<typename T, typename F = std::function<void(T&)>,
    typename = typename std::is_invocable<F, T&>::type>
class task_worker {
    using scoped_lock = std::unique_lock<std::mutex>;
public:
    task_worker() = default;

    template<typename U,
        typename = typename std::enable_if_t<std::is_convertible_v<U,F>>>
    task_worker(U&& fn, size_t cpu_nr)
        : task_handler{std::forward<U>(fn)}
    {
        start(cpu_nr);
    }

    task_worker& operator=(task_worker const&) = delete;
    task_worker(task_worker const&) = delete;

    task_worker& operator=(task_worker&&) = default;
    task_worker(task_worker&& rhs) = default;

    ~task_worker()
    {
        if (worker_thread.joinable())
            stop();
    }

    template<typename U,
        typename = typename std::enable_if_t<std::is_convertible_v<U,F>>>
    void set_task_handler(U&& fn, size_t cpu_nr)
    {
        scoped_lock lock(unm->worker_lock);
        task_handler = std::forward<U>(fn);
        if (!worker_thread.joinable()) {
            lock.unlock();
            start(cpu_nr);
        }
    }

    void start(size_t cpu_nr)
    {
        scoped_lock lock(unm->worker_lock);
        if (!worker_thread.joinable())
            worker_thread = std::thread(&task_worker::worker, this, cpu_nr);
    }

    void stop()
    {
        scoped_lock lock(unm->worker_lock);
        done = true;
        idle = false;
        unm->not_empty.notify_all();
        dump_statistics(std::cerr);
#if PIXELS_HISTOGRAM
        dump_histogram(12);
#endif
        lock.unlock();
        worker_thread.join();
        worker_thread = std::thread();
    }

    void dump_statistics(std::ostream &out)
    {
        out << "#" << cpu_nr <<
            ": executed " << executed <<
            ", drained " << drained << " times (" <<
            (100ULL * drained / executed) << '.' <<
            std::setw(3) << std::setfill('0') <<
            ((100000ULL * drained / executed) % 1000) << "%)\n";
    }

    size_t get_reset_peak()
    {
        scoped_lock lock(unm->worker_lock);
        size_t peak = peak_depth;
        peak_depth = 0;
        return peak;
    }

    void add(T const *item, size_t count, bool allow_notify = true)
    {
        scoped_lock lock(unm->worker_lock);
        //bool notify = queue.empty();
        for (size_t i = 0; i < count; ++i)
            queue.emplace_back(item[i]);
        if (count && allow_notify)
            after_add(true);
    }

    void add(T const& item, bool allow_notify)
    {
        scoped_lock lock(unm->worker_lock);
        //bool notify = queue.empty();
        queue.emplace_back(item);
        if (allow_notify || queue.size() == high_water)
            after_add(true);
    }

    template<typename ...Args>
    void emplace(bool allow_notify, Args&& ...args)
    {
        scoped_lock lock(unm->worker_lock);
        //bool notify = queue.empty();
        T &job = queue.emplace_back(std::forward<Args>(args)...);
        if (allow_notify || queue.size() == high_water)
            after_add(true);
    }

    void add(T&& item, bool allow_notify = true)
    {
        scoped_lock lock(unm->worker_lock);
        //bool notify = queue.empty();
        queue.emplace_back(std::move(item));
        if (allow_notify || queue.size() == high_water)
            after_add(true);
    }

    void wait_for_idle() const
    {
        scoped_lock lock(unm->worker_lock);
        while (!idle)
            unm->is_empty->wait(lock);
    }

    bool is_idle() const
    {
        scoped_lock lock(unm->worker_lock);
        return idle;
    }

private:
    void after_add(bool notify)
    {
        peak_depth = std::max(peak_depth, queue.size());
        idle = false;
        if (notify)
            unm->not_empty.notify_one();
    }

    void worker(size_t cpu_nr)
    {
        this->cpu_nr = cpu_nr;

        fix_thread_affinity(cpu_nr);

        scoped_lock lock(unm->worker_lock);
        for (;;) {
            if (queue.empty())
                ++drained;

            while (!done && queue.empty())
                unm->not_empty.wait(lock);

            if (done)
                break;

            T &item = queue.front();
            lock.unlock();

            task_handler(item);

            lock.lock();
            queue.pop_front();
            ++executed;

            if (queue.empty()) {
                idle = true;
                unm->is_empty.notify_all();
            }
        }
    }

    size_t cpu_nr = -size_t(1);
    F task_handler;
    std::thread worker_thread;

    // Host the unmovable members
    struct unmovable {
        std::mutex worker_lock;
        std::condition_variable not_empty;
        std::condition_variable is_empty;
    };
    std::unique_ptr<unmovable> unm = std::make_unique<unmovable>();

    std::deque<T> queue;

    // The highest count of items in the queue
    size_t peak_depth = 0;

    // The number of times it ran out of work and had to wait
    size_t drained = 0;

    // The number of items executed
    size_t executed = 0;

    // The queue level where it will notify the queue,
    // even if the caller said not to notify the queue
    size_t high_water = 16;

    // This becomes true when it has just
    // finished an item when the queue is empty
    bool idle = true;

    // Set to true to make readers give up
    // instead of waiting when the queue is empty
    bool done = false;
};

class barrier {
public:
    barrier() = default;
    barrier(barrier const&) = delete;
    barrier(barrier&&) = delete;

    void arrive_and_expect(int incoming_expect)
    {
        assert(incoming_expect > 0);
        std::unique_lock<std::mutex> lock(barrier_lock);
        if (count < 0) {
            count = 0;
            expect = incoming_expect;
        } else {
            assert(expect == incoming_expect);
        }
        assert(count < expect);
        if (++count == expect)
            all_reached_cond.notify_all();
    }

    void reset()
    {
        count = -1;
        expect = 0;
    }

    bool wait_until(std::chrono::steady_clock::time_point const& timeout) const
    {
        std::unique_lock<std::mutex> lock(barrier_lock);

        while (count != expect) {
            std::cv_status wait_status =
                all_reached_cond.wait_until(lock, timeout);

            if (wait_status == std::cv_status::timeout)
                return false;
        }
        return true;
    }

    bool wait() const
    {
        return wait_until(std::chrono::steady_clock::time_point::max());
    }

private:
    mutable std::mutex barrier_lock;
    mutable std::condition_variable all_reached_cond;
    // Initial wait is instantly done
    int count = -1;
    int expect = -1;
};

static barrier present_barriers[2];

ssize_t find_glyph(int glyph);

struct fill_job {
    fill_job(frame_param const &fp, barrier *frame_barrier)
        : fp(fp)
        , frame_barrier(frame_barrier)
    {}

    fill_job(frame_param const &fp, int cleared_row, uint32_t color)
        : fp(fp)
        , clear_row(cleared_row)
        , clear_color(color)
    {}

    fill_job(frame_param const &fp, edgeinfo const& lhs, edgeinfo const& rhs,
            unsigned back_phase)
        : fp(fp)
        , edge_refs(lhs, rhs)
        , back_phase(back_phase)
    {}

    fill_job(frame_param const& fp, int y,
            int sx, int sy, int ex, int ey, uint32_t color)
        : fp(fp)
        , clear_color(color)
        , box_y(y)
        , box{(int16_t)sx, (int16_t)sy, (int16_t)ex, (int16_t)ey}
    {}

    fill_job(frame_param const& fp, int y, int sx, int sy,
            char16_t character, uint32_t color)
        : fp(fp)
        , glyph(character)
        , box_y(y)
        , box{
            (int16_t)sx,
            (int16_t)sy,
            (int16_t)(sx + glyphs.w),
            (int16_t)(sy + glyphs.h)
        }
        , clear_color(color)
    {
        glyph = find_glyph(glyph);
    }

    frame_param fp;
    std::pair<edgeinfo, edgeinfo> edge_refs;
    int back_phase;
    int clear_row = -1;
    uint32_t clear_color;
    barrier *frame_barrier = nullptr;
    uint16_t glyph = -1;
    int16_t box_y = -1;
    int16_t box[4];
};

using fill_task_worker = task_worker<fill_job>;
std::vector<fill_task_worker> task_workers;

std::vector<scanconv_ent> scanconv_scratch;

void render_frame(frame_param const& frame);

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
SDL_Surface * back_surfaces[2];
struct free_deleter { void operator()(void*p) const { free(p); } };
std::unique_ptr<uint32_t, free_deleter> back_buffers[2];
frame_param frame_params[2];
SDL_Surface * window_surface = nullptr;

// 1 means drawing to back_surfaces[1]
size_t back_phase;

std::unique_ptr<float, free_deleter> z_buffer;

SDL_Rect fb_viewport;
SDL_Rect fb_area;

struct texture_info {
    uint32_t const *pixels{};
    float fw{}, fh{};
    unsigned iw{}, ih{}, pitch{};
    void (*free_fn)(void*) = nullptr;
};

using scoped_lock = std::unique_lock<std::mutex>;
static std::mutex textures_lock;
using textures_map = std::map<size_t, texture_info>;
static textures_map textures;
static texture_info *texture;

void render_box(frame_param const& frame, int sx, int sy,
    int ex, int ey, uint32_t color = 0xFFFFFFFFU)
{
    size_t slot = sy % task_workers.size();
    for (int y = sy; y <= ey; (y += task_workers.size()), ++slot) {
        task_workers[slot].emplace(false, frame, y, sy, ey, sx, ex, color);
        ++slot;
        slot &= -(slot < task_workers.size());
    }
}

static uint32_t sample_bilinear(scaninfo const& frag, int x, int y)
{
    // Bilinear
    float ftx = frag.t.x * texture->fw;
    float fty = frag.t.y * texture->fh;
    float rtx = floor(ftx);
    float rty = floor(fty);
    int tx = (int)ftx & (texture->iw - 1);
    int ty = (int)fty & (texture->ih - 1);
    int tx1 = (tx + 1) & (texture->iw - 1);
    int ty1 = (ty + 1) & (texture->ih - 1);
    uint32_t texel0 = texture->pixels[ty * texture->iw + tx];
    uint32_t texel1 = texture->pixels[ty * texture->iw + tx1];
    uint32_t texel2 = texture->pixels[ty1 * texture->iw + tx];
    uint32_t texel3 = texture->pixels[ty1 * texture->iw + tx1];
    glm::vec4 vt0(float((texel0) & 0xFF), float((texel0 >> 8) & 0xFF),
        float((texel0 >> 16) & 0xFF), float((texel0 >> 24) & 0xFF));
    glm::vec4 vt1(float((texel0) & 0xFF), float((texel1 >> 8) & 0xFF),
        float((texel1 >> 16) & 0xFF), float((texel1 >> 24) & 0xFF));
    glm::vec4 vt2(float((texel2) & 0xFF), float((texel2 >> 8) & 0xFF),
        float((texel2 >> 16) & 0xFF), float((texel2 >> 24) & 0xFF));
    glm::vec4 vt3(float((texel3) & 0xFF), float((texel3 >> 8) & 0xFF),
        float((texel3 >> 16) & 0xFF), float((texel3 >> 24) & 0xFF));
    float fx = ftx - rtx;
    float oofx = 1.0f - fx;
    glm::vec4 vt01 = vt0 * oofx + vt1 * fx;
    float fy = fty - rty;
    float oofy = 1.0f - fy;
    glm::vec4 vt23 = vt2 * oofx + vt3 * fx;
    glm::vec4 vt = vt01 * oofy + vt23 * fy;
    uint32_t texel = int(vt.x) + (int(vt.y) << 8) +
        (int(vt.z) << 16) + (int(vt.w) << 24);
    return texel;
}

static uint32_t sample_nearest(scaninfo const& frag)
{
    // Nearest
    unsigned tx = (unsigned)(frag.t.x * texture->fw /* * frag.p.z*/) & (texture->iw-1);
    unsigned ty = (unsigned)(frag.t.y * texture->fh /* * frag.p.z*/) & (texture->ih-1);
    uint32_t texel = texture->pixels[ty * texture->iw + tx];
    return texel;
}

__attribute__((__noinline__, __optimize__("O3")))
void clear_worker(fill_job &job)
{
    for (int y = job.clear_row; y < job.fp.height; y += task_workers.size()) {
        uint32_t *output = job.fp.pixels + job.fp.pitch * y;
        std::fill_n(output, job.fp.width, job.clear_color);
    }
    for (int y = job.clear_row; y < job.fp.height; y += task_workers.size()) {
        float *z_output = z_buffer.get() + job.fp.pitch * y;
        std::fill_n(z_output, job.fp.width, INFINITY);
    }
}

using veci32x8 = int32_t __attribute__((__vector_size__(32)));
using vecu32x8 = uint32_t __attribute__((__vector_size__(32)));
using vecf32x8 = float __attribute__((__vector_size__(32)));

static vecf32x8 constexpr laneoffs = {
    0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f
};

static vecu32x8 constexpr lanebits = {
    0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01
    //0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80
};

// The subscript is expected to be min(8, end - i)
// If the vector is aligned, subscript is ex & 7
static vecu32x8 constexpr lanemask[] = {
    {   0,   0,   0,   0,   0,   0,   0,   0 },
    { -1U,   0,   0,   0,   0,   0,   0,   0 },
    { -1U, -1U,   0,   0,   0,   0,   0,   0 },
    { -1U, -1U, -1U,   0,   0,   0,   0,   0 },
    { -1U, -1U, -1U, -1U,   0,   0,   0,   0 },
    { -1U, -1U, -1U, -1U, -1U,   0,   0,   0 },
    { -1U, -1U, -1U, -1U, -1U, -1U,   0,   0 },
    { -1U, -1U, -1U, -1U, -1U, -1U, -1U,   0 },
    { -1U, -1U, -1U, -1U, -1U, -1U, -1U, -1U }
};

// The subscript is expected to be (x & 7),
// if the vector were aligned to a 256 bit boundary
static vecu32x8 leftmask[] = {
    { -1U, -1U, -1U, -1U, -1U, -1U, -1U, -1U },
    {   0, -1U, -1U, -1U, -1U, -1U, -1U, -1U },
    {   0,   0, -1U, -1U, -1U, -1U, -1U, -1U },
    {   0,   0,   0, -1U, -1U, -1U, -1U, -1U },
    {   0,   0,   0,   0, -1U, -1U, -1U, -1U },
    {   0,   0,   0,   0,   0, -1U, -1U, -1U },
    {   0,   0,   0,   0,   0,   0, -1U, -1U },
    {   0,   0,   0,   0,   0,   0,   0, -1U }
};

// static uint32_t mix_pixel(uint32_t existing, uint32_t replacement)
// {
//     unsigned dr = rgb_r(existing);
//     unsigned dg = rgb_g(existing);
//     unsigned db = rgb_b(existing);
//     unsigned sr = rgb_r(replacement);
//     unsigned sg = rgb_g(replacement);
//     unsigned sb = rgb_b(replacement);
//     unsigned sa = rgb_a(replacement);
//     unsigned na = 255U - sa;
//     unsigned r = dr * na / 255U + sr * sa / 255U;
//     unsigned g = dg * na / 255U + sg * sa / 255U;
//     unsigned b = db * na / 255U + sb * sa / 255U;
//     return rgba(r, g, b, 0xFF);
// }

void box_worker(fill_job &job)
{
    // Set up to process this scanline
    uint32_t *pixel = job.fp.pixels + job.box_y * job.fp.pitch;

    // Set up to mask off an 8-bit red green blue or alpha
    vecu32x8 const bytemask{0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

    // Set up the new color vectors
    vecu32x8 sa = (job.clear_color >> 24) & bytemask;
    vecu32x8 sr = (job.clear_color >> 16) & bytemask;
    vecu32x8 sg = (job.clear_color >> 8) & bytemask;
    vecu32x8 sb = (job.clear_color) & bytemask;
    // Set up the multiplier for the other side of the alpha blend
    vecu32x8 na = 255 - sa;

    // Fixedpoint multiplier
    uint32_t fpm = 65536 / 255;

    int x = job.box[0];
    int ex = job.box[2];

    vecu32x8 mask = leftmask[x & 7];

    // Align
    x &= -8;

    for ( ; x < ex; x += 8, mask |= -1U) {
        mask &= lanemask[std::min(8, ex - x)];

        // Read existing pixels to blend with
        vecu32x8 work = (vecu32x8)_mm256_load_si256((__m256i*)(pixel + x));
        // Save a backup in case we do the mask operation in the last loop
        vecu32x8 backup = work;
        // Extract the red green and blue so we can blend them separately
        vecu32x8 dr = (work >> 16) & bytemask;
        vecu32x8 dg = (work >> 8) & bytemask;
        vecu32x8 db = work & bytemask;
        // Blend the destination red green blues with source red green blues
        dr = ((dr * na * fpm) + (sr * sa * fpm)) >> 16;
        dg = ((dg * na * fpm) + (sg * sa * fpm)) >> 16;
        db = ((db * na * fpm) + (sb * sa * fpm)) >> 16;
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
        db = (db & mask) | (backup & ~mask);
        // Write back
        _mm256_store_si256((__m256i*)(pixel + x), (__m256i)db);
    }
}

// Returns ones complement of insertion point if not found
ssize_t find_glyph(int glyph)
{
    size_t st = 0;
    size_t en = glyph < 0x80 ? glyphs.lo : glyphs.n;
    size_t mx = en;
    while (st < en) {
        size_t md = ((en - st) >> 1) + st;
        int candidate = glyphs.glyphs[md];
        if (candidate < glyph)
            st = md + 1;
        else
            en = md;
    }
    st ^= -(st == mx || glyphs.glyphs[st] != glyph);
    return ssize_t(st);
}

uint64_t glyph_bits(int index, int row)
{
    int pitch = (glyphs.w * glyphs.n + 7) >> 3;
    int x = glyphs.w * index;
    int ex = x + glyphs.w;
    int stbyte = pitch * row + (x >> 3);
    int enbyte = pitch * row + ((ex + 7) >> 3);
    int stbit = x & 7;
    uint64_t data{};
    memcpy((uint8_t*)&data, glyphs.bits.get() + stbyte, enbyte - stbyte);
    return (data >> stbit) & ~-(1U << glyphs.w);
}

void glyph_worker(fill_job &job)
{
    uint64_t data = glyph_bits(job.glyph, job.box_y - job.box[1]);
    for (size_t bit = glyphs.w, i = 0; data && bit > 0; ++i, --bit) {
        bool set = data & (1U << (bit - 1));
        if (set) {
            job.fp.pixels[job.fp.pitch * job.box_y +
                job.box[0] + i] = job.clear_color;
        }
    }
}

__attribute__((__hot__))
static void fill_worker_256(fill_job &job)
{
    if (job.frame_barrier)
        return job.frame_barrier->arrive_and_expect(task_workers.size());

    if (job.clear_row >= 0)
        return clear_worker(job);

    if (job.glyph != (uint16_t)-1U)
        return glyph_worker(job);

    if (job.box_y >= 0)
        return box_worker(job);

    std::pair<scaninfo, scaninfo> work = {
        edges[job.back_phase][job.edge_refs.first.edge_idx] +
        edges[job.back_phase][job.edge_refs.first.diff_idx] *
        job.edge_refs.first.n,
        edges[job.back_phase][job.edge_refs.second.edge_idx] +
        edges[job.back_phase][job.edge_refs.second.diff_idx] *
        job.edge_refs.second.n
    };
    int y = work.first.p.y;
    if (y >= job.fp.height)
        return;
    uint32_t *output = job.fp.pixels + job.fp.pitch * y;
    scaninfo diff = work.second - work.first;
    float invWidth = 1.0f / diff.p.x;
    int pixels = std::abs((int)diff.p.x);
#if PIXELS_HISTOGRAM
    ++pixels_histogram[std::max(0,
        std::min(pixels, (int)pixels_histogram_size-1))];
#endif
    if (!pixels)
        return;
    int x = (int)work.first.p.x;
    float n = 0.0f;
    int i = 0;
    if (x < 0) {
        // Skip forward to first visible pixel
        n -= work.first.p.x * invWidth;
        // Advance i by the number of skipped pixels
        i -= x;
        x = 0;
    }

    // Initialize left mask to fix up round down that is about to happen
    vecu32x8 mask = leftmask[x & 7];

    // Round down to 8 pixel aligned boundary (256-bit alignment)
    i -= x & 7;
    n -= (x & 7) * invWidth;
    x &= -8;

    vecf32x8 n_step = invWidth *
        vecf32x8{8.0f, 8.0f, 8.0f, 8.0f, 8.0f, 8.0f, 8.0f, 8.0f};

    uint32_t *pixel_io = output + x;
    float *depth_io = z_buffer.get() + job.fp.pitch * y + x;

    //_mm_prefetch(pixel_io, _MM_HINT_NTA);

    vecf32x8 n_vec = invWidth * laneoffs + n;
    for ( ; i < pixels; i += 8, x += 8, mask = lanemask[8]) {
        mask &= lanemask[std::min(8, pixels - i)];

        // Linear interpolate u and v and z
        vecf32x8 v_vec = (n_vec * diff.t.y) + work.first.t.y;
        vecf32x8 u_vec = (n_vec * diff.t.x) + work.first.t.x;
        vecf32x8 z_vec = (n_vec * diff.p.z) + work.first.p.z;
        // Scale by texture width and height
        v_vec *= texture->fh;
        u_vec *= texture->fw;

        // Convert coordinates to integers
        vecu32x8 ty_vec = __builtin_convertvector(v_vec, vecu32x8);
        vecu32x8 tx_vec = __builtin_convertvector(u_vec, vecu32x8);
        // Wrap coordinates to within texture size
        ty_vec &= texture->ih-1;
        tx_vec &= texture->iw-1;
        // Compute pixel offset within texture
        vecu32x8 tex_off = (ty_vec * texture->iw) + tx_vec;

        // Fetch 8 texels with those 8 array indexes
#if 1
        vecu32x8 texels = (vecu32x8)_mm256_i32gather_epi32(
            (int const*)texture->pixels, (__m256i)tex_off, sizeof(uint32_t));
#else
        vecu32x8 texels{
            texture->pixels[tex_off[0]],
            texture->pixels[tex_off[1]],
            texture->pixels[tex_off[2]],
            texture->pixels[tex_off[3]],
            texture->pixels[tex_off[4]],
            texture->pixels[tex_off[5]],
            texture->pixels[tex_off[6]],
            texture->pixels[tex_off[7]]
        };
#endif
        // Fetch z-buffer values (to see if new pixels are closer)
        vecf32x8 depths = *(vecf32x8*)depth_io; // _mm256_load_ps(depth_io);

        veci32x8 closer = z_vec < depths;
        mask &= closer;

        // int anycloser = _mm256_movemask_epi8((__m256i)mask);
        // if (anycloser) {
            // if (i + 8 <= pixels && anycloser == -1) {
            //     // It is within the end, and they are all closer
            //     // Just overwrite all the color and zbuffer pixels
            //     _mm256_store_si256((__m256i*)(pixel_io), (__m256i)texels);
            //     _mm256_store_ps(z_buffer.get() + job.fp.pitch * y + x, z_vec);
            // } else {
                //mask &= lanemask[std::min(8, pixels - i)];

#if 0
                _mm256_maskstore_epi32((int*)pixel_io,
                    (__m256i)mask, (__m256i)texels);
#else
                vecu32x8 upd_pixels = *(vecu32x8*)pixel_io;
                // (vecu32x8)_mm256_load_si256(
                //     (__m256i*)pixel_io);

                upd_pixels = (texels & mask) | (upd_pixels & ~mask);

                // _mm256_store_si256((__m256i*)pixel_io,
                //     (__m256i)upd_pixels);
                *(vecu32x8*)pixel_io = upd_pixels;
                pixel_io += 8;
#endif

#if 0
                _mm256_maskstore_ps(z_buffer.get() + job.fp.pitch * y + x,
                    (__m256i)mask, (__m256)z_vec);
#else
                vecf32x8 upd_depth = (vecf32x8)(
                    ((veci32x8)z_vec & mask) |
                    ((veci32x8)depths & ~mask));

                // _mm256_store_ps(depth_io, (__m256)upd_depth);
                *(vecf32x8*)depth_io = upd_depth;
                depth_io += 8;
#endif
            // }
        //}
        // Step to next 8 pixels
        n_vec += n_step;
    }
}

// Function to initialize SDL and create the window
bool initSDL(int width, int height) {
    size_t cpu_count = 0;
    char const *ev = getenv("CPUS");
    if (!ev) {
        cpu_count = 4;
    } else if (!strcmp(ev, "max")) {
        cpu_count = std::thread::hardware_concurrency();
        cpu_count = std::max(size_t(1), cpu_count >> 1);
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

    task_workers.reserve(cpu_count);
    for (size_t i = 0; i < cpu_count; ++i)
        task_workers.emplace_back(fill_worker_256, i);

    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "SDL could not initialize!"
            " SDL_Error: " << SDL_GetError() << std::endl;
        return false;
    }

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

    z_buffer.reset((float*)aligned_alloc(sizeof(vecf32x8),
         (sizeof(float) * (fb_area.w * fb_area.h + fb_area.w)) &
        -sizeof(vecf32x8)));

    for (size_t i = 0; i < 2; ++i) {
        edges[i].reserve(262144);
        back_buffers[i].reset((uint32_t*)aligned_alloc(sizeof(vecu32x8),
            sizeof(uint32_t) * (fb_area.w * fb_area.h + fb_area.w)));
        back_surfaces[i] = SDL_CreateRGBSurfaceFrom(back_buffers[i].get(),
            fb_area.w, fb_area.h, 32,
            sizeof(uint32_t) * fb_area.w,
            rgba(0,0,0xFF,0),
            rgba(0,0xFF,0,0),
            rgba(0xFF,0,0,0),
            rgba(0,0,0,0xFF));
        if (back_surfaces[i] == nullptr) {
            std::cerr << "Back surface could not be created!"
                " SDL_Error: " << SDL_GetError() << std::endl;
            return false;
        }
    }

    return true;
}

// Function to handle window events
bool handleEvents() {
    SDL_Event e;
    while (SDL_PollEvent(&e)) {
        if (e.type == SDL_QUIT) {
            return false;
        }
    }
    return true;
}

// Function to render to the current framebuffer
void render()
{
    int pitch = fb_area.w;
    uint32_t *pixels = back_buffers[back_phase].get();
    time_point render_st = clk::now();
    assert(pitch < std::numeric_limits<uint16_t>::max());
    frame_param fp = {SCREEN_WIDTH, SCREEN_HEIGHT, (uint16_t)pitch, pixels};
    render_frame(fp);

    time_point qbarrier_st = clk::now();
    present_barriers[back_phase].reset();
    for (fill_task_worker &worker : task_workers)
        worker.emplace(true, fp, &present_barriers[back_phase]);
    time_point qbarrier_en = clk::now();

    // Wait for the previous completion barrier to finish
    back_phase ^= 1;
    time_point wait_st = clk::now();
    present_barriers[back_phase].wait();
    time_point wait_en = clk::now();
    edges[back_phase].clear();
    edges_lookup[back_phase].clear();
    SDL_Rect src = fb_viewport;
    SDL_Rect dest = fb_viewport;

    SDL_Surface *window_surface =
        SDL_GetWindowSurface(render_window);
    assert(window_surface != nullptr);

    // std::vector<uint32_t> debug = [&] {
    //     std::vector<uint32_t> debug;
    //     debug.reserve(z_buffer.size());
    //     std::transform(z_buffer.begin(), z_buffer.end(),
    //         std::back_inserter(debug), [&](float fz) {
    //             return std::min(255U, std::max(0U, unsigned(fz * 10.0f))) *
    //                 0x10101U | 0xFF000000U;
    //         });
    //     return debug;
    // }();

    // SDL_Surface *z_buffer_surface = SDL_CreateRGBSurfaceFrom(debug.data(),
    //     src.w, src.h, 32, sizeof(uint32_t) * fb_area.w,
    //     rgba(0,0,0xFF,0),
    //     rgba(0,0xFF,0,0),
    //     rgba(0xFF,0,0,0),
    //     rgba(0,0,0,0xFF));

    int status;
    status = SDL_BlitSurface(
        back_surfaces[back_phase],
        // z_buffer_surface,
        &src, window_surface, &dest);
    assert(status == 0);

    //SDL_FreeSurface(z_buffer_surface);

    status = SDL_UpdateWindowSurface(render_window);
    assert(status == 0);

#if 1
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
            std::cout << "#" << worker_nr << " peak " <<
                worker.get_reset_peak() << "\n";
            ++worker_nr;
        }
        std::cout << "----\n";
        div = 0;
    }
#endif
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

void set_texture(uint32_t const *incoming_pixels,
    int incoming_w, int incoming_h, int incoming_pitch,
    void (*free_fn)(void *p))
{
    if (texture->pixels && texture->free_fn)
        texture->free_fn((void*)texture->pixels);
    texture->pixels = incoming_pixels;
    texture->iw = incoming_w;
    texture->ih = incoming_h;
    texture->fw = incoming_w;
    texture->fh = incoming_h;
    texture->pitch = incoming_pitch;
    texture->free_fn = free_fn;
}

int setup(int width, int height);
void cleanup(int width, int height);

int main(int argc, char* args[])
{
    if (!initSDL(SCREEN_WIDTH, SCREEN_HEIGHT))
        return 1;

    glyphs = text_init(48);

    if (setup(SCREEN_WIDTH, SCREEN_HEIGHT))
        return EXIT_FAILURE;

    atexit(SDL_Quit);

    // Main loop
    bool quit = false;
    while (!quit) {
        if (!handleEvents())
            break;

        // Render to the current framebuffer
        render();
    }

    cleanup(SCREEN_WIDTH, SCREEN_HEIGHT);

    for (textures_map::value_type const &tex : textures) {
        if (tex.second.free_fn)
            tex.second.free_fn((void*)tex.second.pixels);
    }

    task_workers.clear();


    // Cleanup and exit
    for (size_t i = 0; i < 2; ++i)
        SDL_FreeSurface(back_surfaces[i]);
    //SDL_DestroyRenderer(gRenderer);
    SDL_DestroyWindow(render_window);
    SDL_Quit();

    return 0;
}

// template<typename T>
// void breesenham_line(int x1, int y1, int x2, int y2, T callback)
// {
//     int dx = std::abs(x2 - x1);
//     int dy = std::abs(y2 - y1);
//     int sx = x1 < x2 ? 1 : -1;
//     int sy = y1 < y2 ? 1 : -1;
//     int err = dx - dy;

//     while (x1 != x2 || y1 != y2) {
//         callback(x1, y1);

//         int e2 = err + err;
//         if (e2 > -dy) {
//             err -= dy;
//             x1 += sx;
//         }
//         if (e2 < dx) {
//             err += dx;
//             y1 += sy;
//         }
//     }
// }

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

// void draw_line(frame_param const& fp, glm::vec3 v0, glm::vec3 v1,
//     unsigned color)
// {
//     breesenham_line(v0.x, v0.y, v1.x, v1.y, [&](int x, int y) {
//         if (x >= 0 && y >= 0 && x < fp.width && y < fp.height)
//             fp.pixels[y * fp.pitch + x] = color;
//     });
// }

unsigned dedup_edge(scaninfo const& v0)
{
    unsigned result;

    std::vector<scaninfo> &edge_table = edges[back_phase];

    std::pair<std::unordered_map<scaninfo, unsigned>::iterator, bool> ins =
        edges_lookup[back_phase].emplace(v0, (unsigned)edge_table.size());
    if (ins.second)
        edge_table.emplace_back(v0);
    return ins.first->second;
}

template<typename C>
void interp_edge(frame_param const& fp,
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
    if (fheight < 1.0f)
        return;
    scaninfo diff = *p1 - *p0;
    float n = 0;
    float invHeight = 1.0f / fheight;
    // Adjust n by how many scanlines are off the top
    float nadj = std::min(p0->p.y * invHeight, 0.0f);
    n -= nadj;
    int y = (int)(p0->p.y);
    int ey = (int)(p0->p.y + fheight);
    // Wipe y to zero if it is negative
    y &= -(y >= 0);
    unsigned edge_idx = dedup_edge(*p0);
    unsigned diff_idx = dedup_edge(diff);
    for (ey = std::min(ey, fp.height-1);
        y <= ey; ++y, n += invHeight)
    {
        callback(y, edge_idx, diff_idx, n);
    }
}

void ensure_scratch(size_t height)
{
    if (scanconv_scratch.empty())
        scanconv_scratch.resize(height);

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

void fill_box(frame_param const& fp, int sx, int sy,
    int ex, int ey, uint32_t color)
{
    size_t slot = sy % task_workers.size();
    for (int y = sy; y <= ey; ++y) {
        task_workers[slot].emplace(false, fp, y, sx, sy, ex, ey, color);

        ++slot;
        slot &= -(slot < task_workers.size());
    }
}

void texture_triangle(frame_param const& fp,
    scaninfo const& v0, scaninfo const& v1, scaninfo const& v2)
{
    ensure_scratch(fp.height);
    int miny = (int)floor(std::min(std::min(v0.p.y, v1.p.y), v2.p.y));
    int maxy = (int)ceil(std::max(std::max(v0.p.y, v1.p.y), v2.p.y));
    size_t dir;

    if (maxy - miny < 1.0f)
        return;

    bool backfacing = (glm::cross(v1.p - v0.p, v2.p - v0.p).z < 0);

    auto draw = [&](int y, unsigned edge_idx, unsigned diff_idx, float n) {
        assert(y >= miny && y <= maxy);
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

    int y = std::max(0, miny);
    size_t slot = y % task_workers.size();
    for (int e = std::min((int)fp.height, maxy); y < e; y++) {
        auto &row = scanconv_scratch[y];

        uint32_t used_value = row.used;
        row.used = 0;
        if (used_value != (draw_nr | 3))
            continue;

        // Peek at x size to get rid of flood of zero width ones
        std::vector<scaninfo> const &edge_list = edges[back_phase];
        float ex = edge_list[row.range[1].edge_idx].p.x +
            edge_list[row.range[1].diff_idx].p.x *
            row.range[1].n;
        float sx = edge_list[row.range[0].edge_idx].p.x +
            edge_list[row.range[0].diff_idx].p.x *
            row.range[0].n;
        float sz = ex - sx;

        if (sz < 1.0f)
            continue;

        task_workers[slot].emplace(false,
            fp, row.range[0], row.range[1], back_phase);

        ++slot;
        slot &= -(slot < task_workers.size());
    }
}

// void fill_triangle(frame_param const& fp,
//     scaninfo const& v0, scaninfo const& v1, scaninfo v2, unsigned color)
// {
//     // Take the cross product of two vectors on the polygon's plane, to
//     // see if it is facing toward or away from the camera
//     // if (glm::cross(v1.p - v0.p, v2.p - v0.p).z < 0)
//     //     return;

//     ensure_scratch(fp.height);

//     int miny = (int)floor(std::min(std::min(v0.p.y, v1.p.y), v2.p.y));
//     int maxy = (int)ceil(std::max(std::max(v0.p.y, v1.p.y), v2.p.y));
//     size_t dir;

//     if (miny == maxy)
//         return;

//     auto draw = [&](int x, int y) {
//         if (y >= 0 && y < fp.height) {
//             x = std::min(fp.width-1, std::max(0, x));
//             scanconv_ent &row = scanconv_scratch[y];
//             row.range[dir].p.x = x;
//             if (row.used != draw_nr)
//                 row.range[dir ^ 1] = {
//                     {0,0},
//                     {0,0,0},
//                     {0,0,0}
//                 };
//             row.used = draw_nr;
//         }
//     };

//     dir = v0.p.y < v1.p.y ? 1 : 0;
//     simple_edge(v0.p.x, v0.p.y, v1.p.x, v1.p.y, draw);
//     dir = v1.p.y < v2.p.y ? 1 : 0;
//     simple_edge(v1.p.x, v1.p.y, v2.p.x, v2.p.y, draw);
//     dir = v2.p.y < v0.p.y ? 1 : 0;
//     simple_edge(v2.p.x, v2.p.y, v0.p.x, v0.p.y, draw);

//     for (int y = std::max(0, miny), e = std::min(fp.height, maxy); y < e; y++) {
//         auto &row = scanconv_scratch[y];
//         if (row.range[0].p.x > row.range[1].p.x || row.used != draw_nr)
//             continue;
//         std::fill_n(fp.pixels + (y * fp.pitch) + (int)row.range[0].p.x,
//             row.range[1].p.x - row.range[0].p.x, color);
//     }
// }

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

void parallel_clear(frame_param const &frame, uint32_t color)
{
    // One huge work item does all of the scanlines
    // that alias onto that worker, in one go
    for (size_t slot = 0; slot < task_workers.size(); ++slot)
        task_workers[slot].emplace(false, frame, slot, color);
}

void draw_text(frame_param const& frame, int x, int y,
    char const *text_st, char const *text_en,
    int wrap_col, uint32_t color)
{
    if (!text_en)
        text_en = strchr(text_st, 0);

    // Decompose into lines and recurse
    if (wrap_col > 0) {
        while (text_st < text_en) {
            char const *first_newline = std::find(
                text_st, std::min(text_st + wrap_col, text_en), '\n');
            if (first_newline == text_en)
                first_newline = std::min(text_en, text_st + wrap_col);
            size_t line_length = first_newline - text_st;
            draw_text(frame, x, y, text_st, text_st + line_length, -1);
            y += glyphs.h;
            text_st += line_length + (text_st + line_length < text_en &&
                text_st[line_length] == '\n');
        }
        return;
    }

    // ... render one line

    // Bail out if it is obviously offscreen
    if (y + glyphs.h < 0 || y >= frame.height ||
        x >= frame.width || x + glyphs.w < 0)
    {
        return;
    }

    int orig_x = x;

    for (size_t slot_st = (unsigned)y % task_workers.size();
            text_st < text_en; ++text_st, orig_x += glyphs.w) {
        size_t slot = slot_st;
        x = orig_x;
        for (int dy = y, ey = y + glyphs.h;
                dy < ey && dy < frame.height;
                ++dy, ++slot) {
            slot &= -(slot < task_workers.size());
            if (x + glyphs.w > 0 && x < frame.width && y + glyphs.h >= 0) {
                task_workers[slot].emplace(false,
                    frame, dy, x, y, char16_t(*text_st), color);
            }
        }
    }
}

font_data text_init(int size, int first_cp, int last_cp)
{
    font_data data;

    if (TTF_Init() != 0)
        return data;

    char const *font_name = "/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf";

    TTF_Font *font = TTF_OpenFont(font_name, size);

    if (!font)
        return data;

    using glyph_map = std::vector<std::pair<int, SDL_Surface *>>;
    glyph_map char_lookup;

    for (int ch = first_cp; ch <= last_cp; ++ch) {
        SDL_Surface *glyph = TTF_RenderGlyph32_Solid(
            font, ch, {0xFF, 0xFF, 0xFF, 0xFF});
        if (!glyph)
            continue;
        if (!char_lookup.empty()) {
            if (glyph->w != char_lookup.front().second->w)
                continue;
            if (glyph->h != char_lookup.front().second->h)
                continue;
        }
        char_lookup.emplace_back(ch, glyph);
    }

    TTF_CloseFont(font);
    font = nullptr;

    if (char_lookup.empty())
        return data;

    SDL_Surface *first_surface = char_lookup.begin()->second;
    data.w = first_surface->w;
    data.h = first_surface->h;
    data.n = char_lookup.size();
    int bitmap_pitch = (data.w * data.n + 7) >> 3;
    data.glyphs = std::make_unique<int[]>(data.n);
    size_t i = 0;
    for (glyph_map::value_type const &item : char_lookup)
        data.glyphs[i++] = item.first;
    int *end_of_ascii = std::upper_bound(data.glyphs.get(),
        data.glyphs.get() + data.n, 0x80);
    data.lo = end_of_ascii - data.glyphs.get();
    size_t bitmap_bits_size = bitmap_pitch * data.h;
    data.bits = std::make_unique<uint8_t[]>(bitmap_bits_size);
    std::fill_n(data.bits.get(), bitmap_bits_size, 0);
    i = 0;
    for (glyph_map::value_type const &item : char_lookup) {
        int dx = data.w * i;

        uint8_t const *p = static_cast<uint8_t const *>(item.second->pixels);
        for (int y = 0; y < item.second->h; ++y) {
            for (int x = 0; x < item.second->w; ++x) {
                uint8_t input = p[item.second->pitch * y + x];
                int bx = dx + x;
                uint8_t *b = data.bits.get() + y * bitmap_pitch + (bx >> 3);
                *b |= (!!input) << (7 - (bx & 7));
            }
        }

        SDL_FreeSurface(item.second);
        ++i;
    }

    return data;
}
