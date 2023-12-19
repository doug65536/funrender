#include "huge_alloc.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cctype>
#include <cmath>

#ifndef __WIN32__
#include <sys/mman.h>
#include <unistd.h>

static size_t get_huge_page_size_cache;

static size_t get_huge_page_size()
{
    if (get_huge_page_size_cache)
        return get_huge_page_size_cache;

    size_t n;
#if defined(__APPLE__)
    n = getpagesize();
#elif defined(__linux__)
    std::ifstream meminfo("/proc/meminfo");
    if (!meminfo.is_open())
        return getpagesize();
    std::string line;
    static char const field[] = "Hugepagesize:";
    while (std::getline(meminfo, line)) {
        // If line is long enough and
        if (line.size() > sizeof(field) &&
                // the colon matches up and
                line[sizeof(field) - 2] ==
                    field[sizeof(field) - 2] &&
                // the whole prefix matches
                std::equal(line.data(),
                line.data() + sizeof(field) - 1,
                field, field + sizeof(field) - 1))
            // then break out without clearing it
            break;
        // Make it clear after we see it didn't match
        // so end of loop has it clear if never matched
        line.clear();
    }
    meminfo.close();

    std::string::const_iterator it;
    if (line.size() <= sizeof(field)) {
        it = line.end();
        n = getpagesize();
    } else {
        it = line.begin() + sizeof(field);
    }

    for (n = 0 ; it != line.end(); ++it) {
        // This is true many times at first
        if (std::isspace(*it))
            continue;

        // On-the-fly atoi, simple enough to do myself
        if (std::isdigit(*it)) {
            n = n * 10 + (*it - '0');
            continue;
        }

        // SI units (the power of 1024 unit)
        // It should be 'k' for the forseeable future
        // B is the end of the value
        if (*it == 'k')
            n = n << 10;
        else if (*it == 'B')
            break;
        else if (*it == 'M')
            n = n << 20;
        else if (*it == 'G')
            n = n << 30;
    }
#elif defined(__WIN32__)
    n = GetLargePageMinimum();
#else
    // Don't know what to do
    #error You need to return the page size here
    n = 4096;   // wild guess, if you are desperate
#endif
    // Sanity check that it's a power of two
    // nullify the nonsense value if not
    if (n & (n - 1))
        n = 0;

    get_huge_page_size_cache = n;
    return n;
}

void *huge_alloc(size_t size, size_t *ret_size)
{
    size_t round_up_to = get_huge_page_size();
    size_t log2pagesize = (size_t)std::log2(round_up_to);

    // Be all ambitious at first
    int flags = MAP_HUGETLB |
        (log2pagesize << MAP_HUGE_SHIFT) | MAP_ANONYMOUS |
        MAP_POPULATE | MAP_LOCKED | MAP_PRIVATE;

#if HAVE_HUGEPAGES
    constexpr bool have_huge = true;
#else
    constexpr bool have_huge = false;
#endif

    void *p{MAP_FAILED};
    for (int pass = 0; pass < 2; ++pass) {
        size_t large_size = (size + (round_up_to - 1)) & -round_up_to;
        if (have_huge || pass) {
            if (ret_size)
                *ret_size = large_size;
            p = mmap(nullptr, large_size,
                PROT_READ | PROT_WRITE, flags, -1, 0);
        }

        // Guarantee that size is zero on failure
        if (p == MAP_FAILED && ret_size)
            *ret_size = 0;

        // Stop if worked or it's the 2nd pass and didn't work
        if (p != MAP_FAILED || pass)
            break;

        // Be less ambitious, don't ask for huge pages
        flags &= ~(MAP_HUGETLB |
            (log2pagesize << MAP_HUGE_SHIFT));

        // Chill on the rounding up
        round_up_to = getpagesize();
    }
    return (p != MAP_FAILED) ? p : nullptr;
}

void huge_free(void *p, size_t size)
{
    size_t huge_page_size = get_huge_page_size();
    if (p && size) {
        size = (size + (huge_page_size-1)) & -huge_page_size;
        munmap(p, size);
    }
}
#else
#include <windows.h>

bool fix_thread_affinity(size_t cpu_nr)
{
    HANDLE threadHandle = GetCurrentThread();
    DWORD_PTR affinityMask = (1ull << cpu_nr);
    return SetThreadAffinityMask(threadHandle, affinityMask) != 0;
}

void *huge_alloc(size_t size, size_t *ret_size)
{
    // First try huge pages
    size_t round_up_to = get_huge_page_size();
    void *result{};
    DWORD flags = MEM_COMMIT | MEM_RESERVE | MEM_LARGE_PAGES;
    for (int pass = 0; !result; ++pass) {
        size_t large_size = (size + round_up_to - 1) & -round_up_to;
        if (ret_size)
            *ret_size = large_size;

        result = VirtualAlloc(nullptr, large_size, flags, PAGE_READWRITE);

        // If it succeeded, return it
        if (result || pass || GetLastError() != ERROR_INVALID_PARAMETER)
            // That succeeded, or we already retried, or
            // it didn't fail like we expected for large pages
            return result;

        // Give up on large pages
        flags &= ~MEM_LARGE_PAGES;

        // Get regular page size
        SYSTEM_INFO si{};
        GetSystemInfo(&si);
        round_up_to = si.dwPageSize;
    }

    return result;
}

void huge_free(void *p, size_t size)
{
    if (p)
        VirtualFree(p, 0, MEM_RELEASE);
}
#endif
