#pragma once
#include <stdlib.h>

bool fix_thread_affinity(size_t cpu_nr);
void *huge_alloc(size_t size);
void huge_free(void *p, size_t size);
