#pragma once

// Some idiot changed libstdc++ to use an if
// instead of conditional operator
// and min generates branches, ffs.
template<typename T>
__attribute__((__always_inline__))
constexpr T sane_min(T a, T b)
{
    return a <= b ? a : b;
}

template<typename T>
__attribute__((__always_inline__))
constexpr T sane_max(T a, T b)
{
    return a >= b ? a : b;
}
