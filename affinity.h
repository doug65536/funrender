#pragma once
#include <stdlib.h>

bool fix_thread_affinity(size_t cpu_nr);
void *huge_alloc(size_t size);
void huge_free(void *p, size_t size);

#define expect_value(value, expected_value) \
    (__builtin_expect((value), (expected_value)))
#define unlikely(k) (expect_value(!!(k), 0))
#define likely(k) (expect_value(!!(k), 1))
