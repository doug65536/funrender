#pragma once
#include <cstdlib>

void *huge_alloc(size_t size, size_t *ret_size = nullptr);
void huge_free(void *p, size_t size);
