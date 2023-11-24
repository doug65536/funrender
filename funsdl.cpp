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
#include <unordered_map>
#include <mutex>
#include <vector>
#include <deque>
#include <list>
#include <condition_variable>
#include <functional>
#include <libpng16/png.h>
#include <glm/glm.hpp>

#include "affinity.h"
#include "abstract_vector.h"


static int blit_to_screen(SDL_Surface *surface, SDL_Rect &src,
    SDL_Surface *window_surface, SDL_Rect &dest);

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

uint64_t last_fps;
uint64_t smooth_fps;

font_data text_init(int size, int first_cp = 1, int last_cp = 0xFFFF);
constexpr size_t phase_count = 2;
std::vector<scaninfo> edges[phase_count];
std::unordered_map<scaninfo, unsigned> edges_lookup[phase_count];

int window_w, window_h;

#if PIXELS_HISTOGRAM
size_t const pixels_histogram_size = 1024;
std::atomic_ulong pixels_histogram[pixels_histogram_size];
#endif

static font_data glyphs;

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

// Some idiot changed libstdc++ to use an if instead of conditional operator
// and min generates branches, ffs.
template<typename T>
constexpr T sane_min(T a, T b)
{
    return a <= b ? a : b;
}

template<typename T>
constexpr T sane_max(T a, T b)
{
    return a >= b ? a : b;
}

template<typename T, typename F = std::function<void(size_t, T&)>,
    typename = typename std::is_invocable<F, T&>::type>
class task_worker {
    using scoped_lock = std::unique_lock<std::mutex>;
public:
    task_worker() = default;

    template<typename U,
        typename = typename std::enable_if_t<std::is_convertible_v<U,F>>>
    task_worker(U&& fn, size_t worker_nr, size_t cpu_nr)
        : worker_nr(worker_nr)
        , task_handler{std::forward<U>(fn)}
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
            worker_thread = std::thread(
                &task_worker::worker, this, cpu_nr);
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
        if (count && (allow_notify ||
                (queue.size() - count < high_water &&
                queue.size() >= high_water)))
            after_add(true);
    }

    void add(T const& item, bool allow_notify = true)
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
        T &job = queue.emplace_back(std::move(item));
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

    uint64_t wait_us = 0;
    uint64_t work_us = 0;
    uint64_t waits = 0;

private:
    void after_add(bool notify)
    {
        peak_depth = sane_max(peak_depth, queue.size());
        idle = false;
        if (notify)
            unm->not_empty.notify_one();
    }

    void worker(size_t cpu_nr)
    {
        this->cpu_nr = cpu_nr;

        fix_thread_affinity(cpu_nr);

        //time_point st, en;

        scoped_lock lock(unm->worker_lock);
        for (;;) {
            if (queue.empty())
                ++drained;

            while (!done && queue.empty()) {
                ++waits;

                // st = clk::now();
                unm->not_empty.wait(lock);
                // en = clk::now();

                // wait_us += std::chrono::duration_cast<
                //     std::chrono::microseconds>(en - st).count();
            }

            if (done)
                break;

            T &item = queue.front();
            lock.unlock();

            // st = clk::now();
            task_handler(worker_nr, item);
            // en = clk::now();

            // work_us += std::chrono::duration_cast<
            //     std::chrono::microseconds>(en - st).count();

            lock.lock();
            queue.pop_front();
            ++executed;

            if (queue.empty()) {
                idle = true;
                unm->is_empty.notify_all();
            }
        }
    }

    size_t worker_nr{-1U};
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
    size_t high_water = 128;

    // This becomes true when it has just
    // finished an item when the queue is empty
    bool idle = true;

    // Set to true to make readers give up
    // instead of waiting when the queue is empty
    bool done = false;
};

class barrier {
public:
    void arrive_and_expect(int incoming_expect)
    {
        assert(incoming_expect > 0);
        std::unique_lock<std::mutex> lock(barrier_lock);
        if (count < 0) {
            count = 0;
            assert(expect == 0);
            expect = incoming_expect;
        } else {
            assert(expect == incoming_expect);
        }
        assert(expect > 0);
        assert(count >= 0);
        assert(count < expect);
        if (++count == expect)
            all_reached_cond.notify_all();
    }

    void reset()
    {
        std::unique_lock<std::mutex> lock(barrier_lock);
        count = -1;
        expect = 0;
    }

    bool wait_until(std::chrono::high_resolution_clock::time_point const& timeout) const
    {
        std::unique_lock<std::mutex> lock(barrier_lock);

        while (count != expect) {
            std::cv_status wait_status =
                all_reached_cond.wait_until(lock, timeout);

            if (wait_status == std::cv_status::timeout)
                break;
        }
        return count == expect;
    }

    bool wait() const
    {
        return wait_until(time_point::max());
    }

private:
    mutable std::mutex barrier_lock;
    mutable std::condition_variable all_reached_cond;
    // Initial wait is instantly done
    int count = -1;
    int expect = -1;
};

static barrier present_barriers[phase_count];

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

    fill_job(frame_param const &fp, int y,
        edgeinfo const& lhs, edgeinfo const& rhs, unsigned back_phase)
        : fp(fp)
        , edge_refs(lhs, rhs)
        , back_phase(back_phase)
    {
        box[1] = std::max(int16_t(0),
            std::min((int16_t)y, (int16_t)INT16_MAX));
    }

    fill_job(frame_param const& fp, int y,
            int sx, int sy, int ex, int ey, uint32_t color,
            std::vector<float> const *border_table)
        : fp(fp)
        , clear_color(color)
        , border_table(border_table)
        , box_y(y)
        , box{(int16_t)sx, (int16_t)sy, (int16_t)ex, (int16_t)ey}
    {}

    fill_job(frame_param const& fp, int y, int sx, int sy,
            size_t glyph_index, uint32_t color)
        : fp(fp)
        , clear_color(color)
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

    frame_param fp;
    std::pair<edgeinfo, edgeinfo> edge_refs;
    unsigned back_phase;
    int clear_row = -1;
    uint32_t clear_color;
    barrier *frame_barrier = nullptr;
    std::vector<float> const *border_table;
    uint16_t glyph_index = -1;
    int16_t box_y = -1;
    int16_t box[4];
};

template<typename T, size_t log2_bucket_sz = 7>
class pool
{
public:
    using bucket_shift = std::integral_constant<
        size_t, log2_bucket_sz>;
    using bucket_sz = std::integral_constant<
        size_t, size_t(1) << log2_bucket_sz>;
    using value_type = T;
    using size_type = size_t;
    using difference_type = ptrdiff_t;

    struct pool_item_deleter {
        pool_item_deleter(pool *owner_) : owner(owner_) {}
        void operator()(void *p)
        {
            owner->recycle(p);
        }
        pool *owner{};
    };

    using item_ptr = std::unique_ptr<T, pool_item_deleter>;

    template<typename ...Args>
    item_ptr alloc(Args&& ...args)
    {
        void *item_memory;

        // Fastpath
        if (free_list) {
            item_memory = reinterpret_cast<void*>(free_list);
            free_slot *next_free = free_list->next;
            free_list->~free_slot();
        } else {
            // Create a bucket if needed
            if (buckets.empty() ||
                    buckets.back().bump_alloc == bucket_sz::value) {
                // Create a bucket
                buckets.emplace_back();
                capacity_ += bucket_sz::value;
            }

            // Take item
            size_t index = buckets.back().bump_alloc++;

            item_memory = buckets.back().storage[index].data;

        }
        T *item = new (item_memory) T(std::forward<Args>(args)...);

        ++used_;

        return item_ptr{item, this};
    }

    void recycle(void *p)
    {
        T *item = reinterpret_cast<T*>(p);
        item->~T();
        free_slot *slot = new (item) free_slot{};
        slot->next = free_list;
        free_list = slot;
        --used_;
    }

    size_t capacity() const
    {
        return capacity_;
    }

    size_t avail() const
    {
        return capacity_ - used_;
    }

    size_t used() const
    {
        return used_;
    }

    void reserve(size_t capacity)
    {
        if (capacity_ >= capacity)
            return;

        size_t rounded_capacity = (capacity + (bucket_sz::value - 1)) /
            bucket_sz::value;
        while (capacity_ < rounded_capacity) {
            buckets.emplace_back();
            capacity_ += bucket_sz::value;
        }
    }

    // Try to return some buckets to the OS memory pool
    // if the usage threshold is below the provided percentage
    void trim(int threshold_percent)
    {
        if (used_ == 0) {
            buckets.clear();
            return;
        }

        if (threshold_percent != 0) {
            if (used_ * 100 / capacity_ >= threshold_percent)
                return;
        }

        using bucket_to_bucket_list_it =
            std::map<void *, typename bucket_list::iterator>;
        bucket_to_bucket_list_it bucket_list_its;
        for (typename bucket_list::iterator it = buckets.begin();
                it != buckets.end(); ++it)
            bucket_list_its.emplace(it->storage.data(), it);

        // Walk the free list
        using free_counts_map =
            std::map<typename bucket_list::iterator, size_t>;
        free_counts_map free_counts;
        for (free_slot *item = free_list; item; item = item->next) {
            typename bucket_to_bucket_list_it::iterator it =
                bucket_list_its.lower_bound((void*)item);
            assert(it != bucket_list_its.end());
            ++free_counts[it];
        }

        // Walk the free counts
        for (typename free_counts_map::iterator it = free_counts.begin();
                it != free_counts.end(); ++it) {
            if (it->second == bucket_sz::value)
                buckets.erase(it->first);
        }
    }
private:
    struct free_slot {
        free_slot *next{};
    };

    using slot = std::aligned_storage_t<
        std::max(sizeof(T), sizeof(free_slot)),
        std::max(alignof(T), alignof(free_slot))>;

    using bucket_storage = std::array<slot, bucket_sz::value>;

    struct bucket {
        size_t bump_alloc{};
        bucket_storage storage;
    };

    using bucket_list = std::list<bucket>;
    bucket_list buckets;

    free_slot *free_list{};

    size_t capacity_{};
    size_t used_{};
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
SDL_Surface * back_surfaces[phase_count];
bool back_surface_locked[phase_count];
struct free_deleter { void operator()(void*p) const { free(p); } };
struct huge_free_deleter {
    huge_free_deleter() = default;
    huge_free_deleter(size_t size)
        : size(size) {}
    void operator()(void*p) const
    {
        huge_free(p, size);
    }
    size_t size = 0;
};
std::unique_ptr<uint32_t, huge_free_deleter> back_buffer_memory;
uint32_t *back_buffers[phase_count];
frame_param frame_params[phase_count];
// SDL_Surface * window_surface = nullptr;

// 1 means drawing to back_surfaces[1]
size_t back_phase;

float *z_buffer;

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

__attribute__((__noinline__))
void clear_worker(fill_job &job)
{
    for (int y = job.clear_row; y < job.fp.height; y += task_workers.size()) {
        float *z_output = job.fp.z_buffer +
            job.fp.pitch * (y + job.fp.top) + job.fp.left;
        //std::numeric_limits<float>::max());
        std::fill_n(z_output, job.fp.width, 1.0f);
    }

    for (int y = job.clear_row; y < job.fp.height; y += task_workers.size()) {
        uint32_t *output = job.fp.pixels +
            job.fp.pitch * (y + job.fp.top) + job.fp.left;
        std::fill_n(output, job.fp.width, job.clear_color);
    }
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
    int w = info.ex - info.sx;
    uint64_t result{};
    for (int i = 0; i < w; ++i) {
        int b = x + i;
        bool bit = glyphs.bits[row * pitch + (b >> 3)] &
            (1U << (7 - (b & 7)));
        result |= (bit << (w - i - 1));
    }
    return result;
}

void glyph_worker(fill_job &job)
{
    glyph_info const &info = glyphs.info[job.glyph_index];
    uint64_t data = glyph_bits(job.glyph_index, job.box_y - job.box[1]);
    uint32_t *pixels = &job.fp.pixels[job.fp.pitch *
            (job.fp.top + job.box_y) +
            job.box[0] + job.fp.left];
    for (size_t bit = info.ex - info.sx, i = 0; data && bit > 0; ++i, --bit) {
        bool set = data & (1U << (bit - info.sx - 1));
        uint32_t &pixel = pixels[i];
        pixel = (job.clear_color & -set) | (pixel & ~-set);
    }
}

template<typename T>
void box_worker(fill_job &job)
{
    // Set up to process this scanline
    uint32_t *pixel = job.fp.pixels +
        (job.box_y + job.fp.top) * job.fp.pitch;

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

    if (job.border_table && from_top_edge < job.border_table->size()) {
        int adj = (int)(*job.border_table)[from_top_edge];
        // adj = (job.border_table->size() - 1) -  adj;
        x += adj;
        ex -= adj;
    } else if (job.border_table && from_bot_edge < job.border_table->size()) {
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
        T const backup = *(T*)(pixel + x);
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
        *(T*)(pixel + x) = db;
    }
}

template<typename T>
void fill_mainloop(fill_job &job,
    std::pair<scaninfo, scaninfo> const& work, scaninfo const& diff,
    float invWidth, int i, float n, int x, int y, int pixels)
{
    using F = typename vecinfo_t<T>::as_float;
    //using I = typename vecinfo_t<T>::as_int;

    constexpr size_t vec_sz = vecinfo_t<T>::sz;
    constexpr size_t vec_mask = vecinfo_t<T>::sz - 1;

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
    uint32_t *pixel_io = job.fp.pixels + pixel_index;
    float *depth_io = job.fp.z_buffer + pixel_index;

    F n_vec = invWidth * vecinfo_t<F>::laneoffs + n;
    for ( ; i < pixels; i += vec_sz, x += vec_sz,
            mask = vecinfo_t<T>::lanemask[vec_sz]) {
        mask &= vecinfo_t<T>::lanemask[sane_min(int(vec_sz), pixels - i)];

        // Linear interpolate the u/w and v/w
        F v_vec = (n_vec * diff.t.y) + work.first.t.y;
        F u_vec = (n_vec * diff.t.x) + work.first.t.x;

        // Linear interpolate 1/w for perspective correction
        F w_vec = (n_vec * diff.p.w) + work.first.p.w;

        // Linear interpolate z for depth buffering
        F z_vec = (n_vec * diff.p.z) + work.first.p.z;

        // Step to next {vec_sz} pixels
        n_vec += n_step;

        // Perspective correction
        u_vec /= w_vec;
        v_vec /= w_vec;

        // Scale by texture width and height
        v_vec *= texture->fh;
        u_vec *= texture->fw;

        // Convert coordinates to integers
        T ty_vec = __builtin_convertvector(v_vec, T);
        T tx_vec = __builtin_convertvector(u_vec, T);

        // Wrap coordinates to within texture size
        ty_vec &= texture->ih-1;
        tx_vec &= texture->iw-1;

        // Fetch z-buffer values (to see if new pixels are closer)
        F depths = *(F*)depth_io; // _mm256_load_ps(depth_io);

        // Compute pixel offset within texture
        T tex_off = (ty_vec * texture->iw) + tx_vec;

        // Only write pixels that are closer
        mask &= (T)(z_vec < depths);

        // // Only write pixels that are in front of you
        // clipping works mask &= z_vec >= 0.0f;

        if (vec_movemask(mask)) {
            // Fetch existing pixels for masked merge with new pixels
            T upd_pixels = *(T*)pixel_io;

            // Fetch {vec_sz} texels with those {vec_sz} array indexes
            T texels = vec_gather(texture->pixels, tex_off, upd_pixels, mask);

            *(T*)pixel_io = texels;

            F upd_depth = vec_blend(depths, z_vec, mask);

            *(F*)depth_io = upd_depth;
        }

        pixel_io += vec_sz;
        depth_io += vec_sz;
    }
}

using mainloop_pfn = void(*)(
    fill_job &job, std::pair<scaninfo, scaninfo> const& work,
    scaninfo const& diff, float invWidth,
    int i, float n, int x, int y, int pixels);

template<typename T, mainloop_pfn mainloop = fill_mainloop<T>>
static void fill_worker(size_t worker_nr, fill_job &job)
{
    if (job.frame_barrier)
        return job.frame_barrier->arrive_and_expect(task_workers.size());

    if (job.clear_row >= 0) {
        assert(job.clear_row % task_workers.size() == worker_nr);
        return clear_worker(job);
    }

    if (job.glyph_index != (uint16_t)-1U) {
        assert(job.box_y % task_workers.size() == worker_nr);
        return glyph_worker(job);
    }

    if (job.box_y >= 0) {
        assert(job.box_y % task_workers.size() == worker_nr);
        return box_worker<T>(job);
    }

    std::pair<scaninfo, scaninfo> work = {
        edges[job.back_phase][job.edge_refs.first.edge_idx] +
        edges[job.back_phase][job.edge_refs.first.diff_idx] *
        job.edge_refs.first.n,
        edges[job.back_phase][job.edge_refs.second.edge_idx] +
        edges[job.back_phase][job.edge_refs.second.diff_idx] *
        job.edge_refs.second.n
    };
    // Must floor the x coordinates so polygons fit together perfectly
    work.first.p.x = floorf(work.first.p.x);
    work.second.p.x = floorf(work.second.p.x);
    int y = job.box[1];
    assert(y % task_workers.size() == worker_nr);
    if (y >= job.fp.height) {
        assert(!"Clipping messed up");
        return;
    }
    work.first.p.w = 1.0f / work.first.p.w;
    work.second.p.w = 1.0f / work.second.p.w;
    // std::cerr << "Scanline at " << y << "\n";
    work.first.t.x *= work.first.p.w;
    work.first.t.y *= work.first.p.w;
    work.second.t.x *= work.second.p.w;
    work.second.t.y *= work.second.p.w;
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
        mainloop(job, work, diff, invWidth, i, n, x, y, pixels);
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
        edges[i].reserve(262144);
        back_buffers[i] = back_buffer_memory.get() + (1 + i) * frame_pixels;
        back_surfaces[i] = SDL_CreateRGBSurfaceFrom(back_buffers[i],
            fb_area.w, fb_area.h, 32, sizeof(uint32_t) * fb_area.w,
            rgba(0,0,0xFF,0), rgba(0,0xFF,0,0),
            rgba(0xFF,0,0,0), rgba(0,0,0,0xFF));
        if (back_surfaces[i] == nullptr) {
            std::cerr << "Back surface could not be created!"
                " SDL_Error: " << SDL_GetError() << std::endl;
            return false;
        }

        SDL_SetSurfaceBlendMode(back_surfaces[i], SDL_BLENDMODE_NONE);
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
            return handle_window_event(e.window);
        }
    }
    return true;
}

// Function to render to the current framebuffer
void render()
{
    size_t phase = back_phase;
    size_t prev_phase = (phase == 0)
        ? (phase_count - 1)
        : (phase - 1);
    if (SDL_MUSTLOCK(back_surfaces[phase])) {
        back_surface_locked[phase] = true;
        SDL_LockSurface(back_surfaces[phase]);
    }

    // std::cout << "Rendering phase " << phase << '\n';
    int pitch = fb_area.w;
    uint32_t *pixels = back_buffers[phase];
    //time_point render_st = clk::now();
    assert(pitch < std::numeric_limits<uint16_t>::max());
    //std::cout << "Drawing to " << pixels << "\n";
    frame_param fp{};
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

    render_frame(fp);

    // std::cout << "Enqueue barrier at end of phase " << phase << "\n";
    //time_point qbarrier_st = clk::now();
    present_barriers[phase].reset();
    for (fill_task_worker &worker : task_workers)
        worker.emplace(true, fp, &present_barriers[phase]);
    //time_point qbarrier_en = clk::now();


    if (back_surface_locked[phase]) {
        back_surface_locked[phase] = false;
        SDL_UnlockSurface(back_surfaces[phase]);
    }

    // Wait for the previous completion barrier to finish

    //time_point wait_st = clk::now();
    present_barriers[prev_phase].wait();
    //time_point wait_en = clk::now();

    if (back_surface_locked[prev_phase]) {
        back_surface_locked[prev_phase] = false;
        SDL_UnlockSurface(back_surfaces[prev_phase]);
    }

    SDL_Surface *window_surface =
        SDL_GetWindowSurface(render_window);
    assert(window_surface != nullptr);

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
    assert(status == 0);
    SDL_FreeSurface(z_buffer_surface);
#else
    // std::cout << "Presenting " << phase << "\n";
    int status = blit_to_screen(back_surfaces[prev_phase], src,
        window_surface, dest);
    //std::this_thread::sleep_for(std::chrono::milliseconds(200));

#endif

    status = SDL_UpdateWindowSurface(render_window);
    assert(status == 0);

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

    edges[phase].clear();
    edges_lookup[phase].clear();
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
    assert(status == 0);
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

bool measure;

int main(int argc, char** argv)
{
    measure = (argc > 1 && !strcmp(argv[1], "--measure"));

    if (!initSDL(SCREEN_WIDTH, SCREEN_HEIGHT))
        return 1;

    glyphs = text_init(24);

    if (setup(SCREEN_WIDTH, SCREEN_HEIGHT))
        return EXIT_FAILURE;

    atexit(SDL_Quit);

    // Main loop
    bool quit = false;
    uint64_t count = 0;
    while (!quit) {
        if (!handleEvents())
            break;

        // Render to the current framebuffer
        render();

        ++count;
        if (measure && count >= 400)
            break;
    }

    cleanup(SCREEN_WIDTH, SCREEN_HEIGHT);

    for (textures_map::value_type const &tex : textures) {
        if (tex.second.free_fn)
            tex.second.free_fn((void*)tex.second.pixels);
    }

    task_workers.clear();

    // Cleanup and exit
    for (size_t i = 0; i < phase_count; ++i)
        SDL_FreeSurface(back_surfaces[i]);
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

unsigned dedup_edge(scaninfo const& v0)
{
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
    assert(fheight >= 0.0f);
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

    unsigned edge_idx = dedup_edge(*p0);
    unsigned diff_idx = dedup_edge(diff);
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

void fill_box(frame_param const& fp, int sx, int sy,
    int ex, int ey, uint32_t color, int border_radius)
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
        assert(y % task_workers.size() == slot);
        task_workers[slot].emplace(false,
            fp, y, sx, sy, ex, ey, color, border_table);

        ++slot;
        slot &= -(slot < task_workers.size());
    }
}

void texture_triangle(frame_param const& fp,
    scaninfo v0, scaninfo v1, scaninfo v2)
{
    ensure_scratch(fp.height);
    size_t dir;

    // Project homogenous coordinates to NDC 3D
    // float inv_v0w = 1.0f / v0.p.w;
    // float inv_v1w = 1.0f / v1.p.w;
    // float inv_v2w = 1.0f / v2.p.w;
    // glm::vec3 p0 =
    //     glm::vec3(v0.p.x * inv_v0w, v0.p.y * inv_v0w, v0.p.z * inv_v0w);
    // glm::vec3 p1 =
    //     glm::vec3(v1.p.x * inv_v1w, v1.p.y * inv_v1w, v1.p.z * inv_v1w);
    // glm::vec3 p2 =
    //     glm::vec3(v2.p.x * inv_v2w, v2.p.y * inv_v2w, v2.p.z * inv_v2w);

    // v0.p.x = p0.x; v0.p.y = p0.y; v0.p.z = p0.z; v0.p.w = inv_v0w;
    // v1.p.x = p1.x; v1.p.y = p1.y; v1.p.z = p1.z; v1.p.w = inv_v1w;
    // v2.p.x = p2.x; v2.p.y = p2.y; v2.p.z = p2.z; v2.p.w = inv_v2w;

    // v0.p.x *= fp.width * 0.5f;
    // v1.p.x *= fp.width * 0.5f;
    // v2.p.x *= fp.width * 0.5f;

    // v0.p.y *= fp.height * 0.5f;
    // v1.p.y *= fp.height * 0.5f;
    // v2.p.y *= fp.height * 0.5f;

    // v0.p.z *= 0.5f;
    // v1.p.z *= 0.5f;
    // v2.p.z *= 0.5f;

    // v0.p.x += fp.width * 0.5f;
    // v1.p.x += fp.width * 0.5f;
    // v2.p.x += fp.width * 0.5f;

    // v0.p.y += fp.height * 0.5f;
    // v1.p.y += fp.height * 0.5f;
    // v2.p.y += fp.height * 0.5f;

    // v0.p.z += 0.5f;
    // v1.p.z += 0.5f;
    // v2.p.z += 0.5f;

    glm::vec3 p0{v0.p.x, v0.p.y, v0.p.z};
    glm::vec3 p1{v1.p.x, v1.p.y, v1.p.z};
    glm::vec3 p2{v2.p.x, v2.p.y, v2.p.z};
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
    assert(t >= 0.0f && t <= 1.0f);
    return t;
}

// Find intercept on negative plane
float intercept_neg(float x1, float w1, float x2, float w2)
{
    float t = (w1 + x1) / ((w1 + x1) - (w2 + x2));
    //       *(w1 - x1) / ((w1 - x1) - (w2 - x2));
    //        (-1.0f - x1) / (w2 - w1);
    //        (w1 - x1) / (x2 - x1 + w1 - w2);
    //        (w1 - x1) / (x1 - x2 + w1 - w2);
    //        (x1 - w1) / ((x1 - w1) - (x2 - w2));
    //        (w1 - x1) / (x2 - x1 - (w2 - w1));
    //        (w1 - x1) / (x2 - x1 + w1 - w2);
    //        (x1 - w1) / (x1 - x2 + w1 - w2);
    //        (w1 - x1) / (x2 - x1 + w1 - w2);
    //        (w1 - x1) / ((x2 - x1) + (w1 - w2));
    //        (w1 - x1) / (x2 - x1 + w1 - w2);
    //        (w1 + x1) / (x1 + w1 - x2 - w2);
    assert(t >= 0.0f && t <= 1.0f);
    return t;
}

#define DEBUG_CLIPPING 0

static std::vector<int> clip_masks;
static glm::mat4 combined_xform;

std::vector<glm::mat4> view_mtx_stk{1};
std::vector<glm::mat4> proj_mtx_stk{1};

void set_transform(glm::mat4 const& mat)
{
    combined_xform = mat;
}

static std::vector<scaninfo> vinp_scratch;
static std::vector<scaninfo> vout_scratch;

static void reset_clip_scratch(size_t reserve)
{
    vout_scratch.clear();
    vinp_scratch.clear();
    vinp_scratch.reserve(reserve);
    vout_scratch.reserve(reserve);
}

static scaninfo *scaninfo_transform(frame_param const& fp,
    scaninfo const *vinp, size_t count)
{
    reset_clip_scratch(count);
    for (size_t i = 0; i < count; ++i) {
        scaninfo s = combined_xform * vinp[i];
        vinp_scratch.emplace_back(s);
    }
    return vinp_scratch.data();
}

static float constexpr (glm::vec4::*plane_lookup[]) = {
    &glm::vec4::x,
    &glm::vec4::y,
    &glm::vec4::z
};

void texture_polygon(frame_param const& frame,
    scaninfo const *user_verts, size_t count)
{
    scaninfo *verts = scaninfo_transform(frame, user_verts, count);

    // Do 0-to-1, 1-to-2, 2-to-0
    for (size_t plane = 0; plane < 6 && vinp_scratch.size() >= 3; ++plane,
            // Swap in/out, clear out
            vinp_scratch.swap(vout_scratch), vout_scratch.clear(),
            // Switch to the new scratch input
            (verts = vinp_scratch.data()), (count = vinp_scratch.size())) {
        float (glm::vec4::*field);
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

        for (int edge = 0; edge < count; ++edge) {
            int next = (edge + 1) & -(edge < count - 1);
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
            field = plane_lookup[plane < 3 ? plane : plane - 3];
            // Adjust the intercept function for -w or +w, appropriately
            float clip_sign = plane < 3 ? 1.0f : -1.0f;

            // If it crossed from clipped to not clipped or vice versa
            int diff_mask = edge_mask ^ next_mask;

            if (diff_mask & (1 << plane)) {
                // The edge crossed the plane
                if (edge_mask & (1 << plane)) {
                    // Starts outside clip plane
                    float t = intercept_neg(
                        vst.p.*field * clip_sign, vst.p.w,
                        ven.p.*field * clip_sign, ven.p.w
                        );
#if DEBUG_CLIPPING
                    dump(t, "(starts outside)");
#endif
                    vout_scratch.emplace_back(((ven - vst) * t) + vst);
                } else {
                    // Starts inside clip plane
                    float t = intercept_neg(
                        vst.p.*field * clip_sign, vst.p.w,
                        ven.p.*field * clip_sign, ven.p.w);
#if DEBUG_CLIPPING
                    dump(t, "(starts inside)");
#endif

                    vout_scratch.emplace_back(vst);
                    vout_scratch.emplace_back(((ven - vst) * t) + vst);
                }
            } else if (!(edge_mask & (1 << plane))) {
                vout_scratch.emplace_back(vst);
            }
        }
    }

    // Project to screenspace
    float frame_width = frame.width;
    float frame_height = frame.height;
    for (size_t i = 0; i < count; ++i) {
        scaninfo &vertex = verts[i];
        // Divide x, y, and z by w, and store inverse w in w
        float oow = 1.0f / vertex.p.w;
        vertex.p.x *= oow;
        vertex.p.y *= oow;
        vertex.p.z *= oow;
        vertex.p.w = oow;
        // Scale -1.0-to-1.0 range down to -0.5-to-0.5
        vertex.p.x *= 0.5f;
        vertex.p.y *= 0.5f;
        vertex.p.z *= 0.5f;
        // Shift -0.5-to-0.5 over to 0-to-1.0
        vertex.p.x += 0.5f;
        vertex.p.y += 0.5f;
        vertex.p.z += 0.5f;
        // Scale to screen coordinate
        vertex.p.x *= frame_width;
        vertex.p.y *= frame_height;
    }

    // Make a fan from the polygon vertices
    for (size_t i = 1; i + 1 < count; ++i)
        texture_triangle(frame, verts[0], verts[i], verts[i+1]);
}

void texture_polygon(frame_param const& frame,
    std::vector<scaninfo> vinp)
{
    texture_polygon(frame, vinp.data(), vinp.size());
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

void parallel_clear(frame_param const &frame, uint32_t color)
{
    // One huge work item does all of the scanlines
    // that alias onto that worker, in one go
    for (size_t slot = 0; slot < task_workers.size(); ++slot)
        task_workers[slot].emplace(false,
             frame, slot, color);
}

// Convert the given range of UTF-8 to UCS32
std::vector<char32_t> ucs32(char const *start, char const *end,
    bool *ret_failed = nullptr)
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
                st + 1 < en && ((st[1] & 0xc0) == 0x80)) {
            // 2-byte
            result.push_back(((st[0] & 0x1f) << 6) |
                (st[1] & 0x3F));
            st += 2;
        } else if ((st[0] & 0xf0) == 0xe0 &&
                st + 2 < en &&
                (st[1] & 0xc0) == 0x80 &&
                (st[2] & 0xc0) == 0x80) {
            // 3-byte
            result.push_back(((st[0] & 0xf) << 12) |
                ((st[1] & 0x3F) << 6) |
                ((st[2] & 0x3F)));
            st += 3;
        } else if ((st[0] & 0xf8) == 0xf0 &&
                st + 3 < en &&
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

void draw_text(frame_param const& frame, int x, int y,
    char32_t const *text_st, char32_t const *text_en,
    int wrap_col, uint32_t color)
{
    ensure_scratch(frame.height);

    // Decompose into lines and recurse
    if (wrap_col > 0) {
        while (text_st < text_en) {
            char32_t const *first_newline = std::find(
                text_st, sane_min(text_st + wrap_col, text_en), L'\n');
            if (first_newline == text_en)
                first_newline = sane_min(text_en, text_st + wrap_col);
            size_t line_length = first_newline - text_st;
            draw_text(frame, x, y, text_st, text_st + line_length, -1, color);
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

    for ( ; text_st < text_en; ++text_st, orig_x += info ? info->advance : 0) {
        char32_t character = *text_st;
        size_t glyph_index = find_glyph(character);
        if ((ssize_t)glyph_index < 0)
            continue;
        info = &glyphs.info[glyph_index];
        x = orig_x;

        for (int dy = y, ey = dy + glyphs.h;
                dy < ey && dy < frame.height; ++dy) {
            if (x + (info->ex - info->sx) > 0 &&
                    x < frame.width && y + glyphs.h >= 0) {
                size_t slot = dy % task_workers.size();
                fill_job_batches[slot].emplace_back(
                    frame, dy, x, y, glyph_index, color);
            }
        }
    }

    for (size_t slot = 0; slot < task_workers.size(); ++slot) {
        task_workers[slot].add(fill_job_batches[slot].data(),
            fill_job_batches[slot].size(), false);
        fill_job_batches[slot].clear();
    }
}

void draw_text(frame_param const& frame, int x, int y,
    char const *utf8_st, char const *utf8_en,
    int wrap_col, uint32_t color)
{
    if (!utf8_en)
        utf8_en = strchr(utf8_st, 0);

    std::vector<char32_t> codepoints = ucs32(utf8_st, utf8_en);
    char32_t const *text_st = codepoints.data();
    char32_t const *text_en = text_st + codepoints.size();

    draw_text(frame, x, y, text_st, text_en, wrap_col, color);
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

font_data text_init(int size, int first_cp, int last_cp)
{
    font_data data;

    if (TTF_Init() != 0)
        return data;

    char const *font_name = "/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf";
    //char const *font_name = "/usr/share/fonts/truetype/tlwg/Purisa-BoldOblique.ttf";
    //char const *font_name = "/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf";
    //char const *font_name = "/usr/share/fonts/truetype/freefont/FreeMono.ttf";
    //char const *font_name = "/usr/share/fonts/truetype/liberation2/LiberationMono-Regular.ttf";
    //char const *font_name = "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf";
    //char const *font_name = "/usr/share/fonts/truetype/tlwg/TlwgMono.ttf";

    // Key is what they look like, value is {sx, width}
    using appearance_map = std::unordered_map<
        std::string, std::pair<int, int>>;
    appearance_map pos_by_appearance;

    std::vector<int> deduped_codepoints;

    TTF_Font *font = TTF_OpenFont(font_name, size);

    if (!font)
        return data;

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
            pos_by_appearance.emplace(stringify_glyph(glyph),
                std::make_pair(x, glyph->w));

        if (ins.second) {
            x += glyph->w;
            deduped_codepoints.push_back(ch);
            dump_debug_glyph(std::cerr, ch, glyph);
        } else {
            // This glyph looks identical to another glyph, don't need dup
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
        return data;

    // Use the first surface as a reference

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
    assert(i == data.n);

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

        //std::cerr << "Getting bits for codepoint " << item.first << '\n';

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

    return data;
}

size_t hash_bytes(void const *data, size_t size)
{
    return std::hash<std::string_view>()(
        std::string_view((char const *)data, size));
}
