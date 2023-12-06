#pragma once
#include <cassert>
#include "affinity.h"

#ifndef NDEBUG
#define assume(expr) do { \
    if (unlikely(!static_cast<bool>(expr))) \
        __assert_fail(#expr, __FILE__, __LINE__, __ASSERT_FUNCTION); \
} while(0)
#else
#define assume(expr) do { if (!static_cast<bool>(expr)) \
    __builtin_unreachable(); \
} while(0)
#endif
