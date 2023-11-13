#include "affinity.h"

#ifndef __WIN32__
#include <pthread.h>
#endif

bool fix_thread_affinity(size_t cpu_nr)
{
    cpu_set_t cpus;
    CPU_ZERO(&cpus);
    CPU_SET(cpu_nr, &cpus);
    pthread_t self = pthread_self();
    return pthread_setaffinity_np(self, sizeof(cpu_set_t), &cpus) == 0;
}
