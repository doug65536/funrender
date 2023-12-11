#include "affinity.h"

#include <unistd.h>
#include <cstdio>
#include <cmath>
#include <unistd.h>

#include <pthread.h>
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>
#include <fstream>

bool fix_thread_affinity(size_t cpu_nr)
{
    cpu_set_t cpus;
    CPU_ZERO(&cpus);
    CPU_SET(cpu_nr, &cpus);
    pthread_t self = pthread_self();
    return pthread_setaffinity_np(self, sizeof(cpu_set_t), &cpus) == 0;
}
