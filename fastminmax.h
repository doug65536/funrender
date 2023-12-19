#pragma once

// Some idiot changed libstdc++ to use an if
// instead of conditional operator
// and min generates branches, ffs.
template<typename T>
constexpr T sane_min(T a, T b)
{
    return a <= b ? a : b;
}

template<typename T>
constexpr T sane_max(T a, T b)
{
    return a >= b ? a : b;
}
