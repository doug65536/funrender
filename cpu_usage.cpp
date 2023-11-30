#include "cpu_usage.h"

#ifndef _WIN32
#include <cstdlib>
#include <sys/times.h>
cpu_usage_info cpu_usage()
{
    tms t{};
    clock_t timestamp = times(&t);
    bool ok = (timestamp != -1);
    uint64_t total = uint64_t(timestamp);
    uint64_t kernel = uint64_t(t.tms_stime);
    uint64_t user = uint64_t(t.tms_utime);
    if (!ok) {
        total = -1U;
        kernel = -1U;
        user = -1U;
    }
    return {
        total,
        kernel,
        user,
        ok
    };
}
#else
#include <windows.h>
cpu_usage_info cpu_usage()
{
    FILETIME idleTime{}, kernelTime{}, userTime{};
    bool ok = GetSystemTimes(&idleTime, &kernelTime, &userTime);

    uint64_t idle = idleTime.LowPart |
        (uint64_t(idleTime.HighPart) << 32);
    uint64_t kernel = kernelTime.LowPart |
        (uint64_t(kernelTime.HighPart) << 32);
    uint64_t user = userTime.LowPart |
        (uint64_t(userTime.HighPart) << 32);
    uint64_t total = kernel + user;

    if (!ok) {
        total = -1U;
        kernel = -1U;
        user = -1U;
        idle = 0;
    }

    return {
        total,
        kernel - idle,
        user,
        ok
    };
}
#endif
