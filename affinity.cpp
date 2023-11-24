#include "affinity.h"

#include <unistd.h>
#include <cstdio>
#include <unistd.h>

#ifndef __WIN32__
#include <pthread.h>
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>

bool fix_thread_affinity(size_t cpu_nr)
{
    cpu_set_t cpus;
    CPU_ZERO(&cpus);
    CPU_SET(cpu_nr, &cpus);
    pthread_t self = pthread_self();
    return pthread_setaffinity_np(self, sizeof(cpu_set_t), &cpus) == 0;
}

void *huge_alloc(size_t size)
{
    size = (size + ((1U<<21)-1)) & -(1U<<21);
#if HAVE_HUGEPAGES
    void *p = mmap(nullptr, size,
        PROT_READ | PROT_WRITE, MAP_HUGETLB |
        (21 << MAP_HUGE_SHIFT) | MAP_ANONYMOUS |
        MAP_POPULATE | MAP_LOCKED | MAP_PRIVATE, -1, 0);
    if (p == MAP_FAILED)
        perror("Allocating huge pages");
    if (long(p) & ~-(1 << 21))
        printf("They can't be 2MB pages\n");
#else
    void *p = mmap(nullptr, size,
        PROT_READ | PROT_WRITE, MAP_ANONYMOUS |
        MAP_POPULATE | MAP_LOCKED | MAP_PRIVATE, -1, 0);
    if (p == MAP_FAILED)
        perror("Allocating pages");
#endif
    return (p != MAP_FAILED) ? p : nullptr;
}

void huge_free(void *p, size_t size)
{
    if (p && size) {
        size = (size + ((1U<<21)-1)) & -(1U<<21);
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

void *huge_alloc(size_t size)
{
    return VirtualAlloc(nullptr, size,
        MEM_COMMIT | MEM_RESERVE | MEM_LARGE_PAGES, PAGE_READWRITE);
}

void huge_free(void *p, size_t size)
{
    if (p)
        VirtualFree(p, 0, MEM_RELEASE);
}
#endif
