#pragma once
#include <cassert>
#include <mutex>
#include <chrono>
#include "perf.h"

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
