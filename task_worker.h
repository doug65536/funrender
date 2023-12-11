#pragma once
#include <functional>
#include <mutex>
#include <thread>
#include <deque>
#include <iomanip>
#include <condition_variable>
#include <cassert>
#include "fastminmax.h"
#include "affinity.h"

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
            (executed ? 100ULL * drained / executed : 0) << '.' <<
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

    T &add(T const& item, bool allow_notify = true)
    {
        scoped_lock lock(unm->worker_lock);
        //bool notify = queue.empty();
        T& returned_item = queue.emplace_back(item);
        if (allow_notify || queue.size() == high_water)
            after_add(true);
        return returned_item;
    }

    template<typename ...Args>
    T &emplace(bool allow_notify, Args&& ...args)
    {
        scoped_lock lock(unm->worker_lock);
        //bool notify = queue.empty();
        T &item = queue.emplace_back(std::forward<Args>(args)...);
        if (allow_notify || queue.size() == high_water)
            after_add(true);
        return item;
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
