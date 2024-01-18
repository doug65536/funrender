#pragma once
#include <utility>
#include <cstdint>
#include <climits>
#include <limits>
#include <type_traits>

#ifdef __x86_64__
#include <xmmintrin.h>
#endif
#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

template<typename T>
struct next_bigger;

template<> struct next_bigger<uint8_t> { using type = uint16_t; };
template<> struct next_bigger<uint16_t> { using type = uint32_t; };
template<> struct next_bigger<uint32_t> { using type = uint64_t; };

template<> struct next_bigger<int8_t> { using type = int16_t; };
template<> struct next_bigger<int16_t> { using type = int32_t; };
template<> struct next_bigger<int32_t> { using type = int64_t; };

template<typename T>
using next_bigger_t = typename next_bigger<T>::type;

template<typename T>
struct next_smaller;

template<> struct next_smaller<uint16_t> { using type = uint8_t; };
template<> struct next_smaller<uint32_t> { using type = uint16_t; };
template<> struct next_smaller<uint64_t> { using type = uint32_t; };

template<> struct next_smaller<int16_t> { using type = int8_t; };
template<> struct next_smaller<int32_t> { using type = int16_t; };
template<> struct next_smaller<int64_t> { using type = int32_t; };

template<typename T>
using next_smaller_t = typename next_smaller<T>::type;

//
// 128-bit vectors

using vecu8x16 = uint8_t
    __attribute__((__vector_size__(sizeof(uint8_t) * 16)));
using veci8x16 = int8_t
    __attribute__((__vector_size__(sizeof(int8_t) * 16)));

using vecu16x8 = uint16_t
    __attribute__((__vector_size__(sizeof(uint16_t) * 8)));
using veci16x8 = int16_t
    __attribute__((__vector_size__(sizeof(int16_t) * 8)));

using vecu32x4 = uint32_t
    __attribute__((__vector_size__(sizeof(uint32_t) * 4)));
using veci32x4 = int32_t
    __attribute__((__vector_size__(sizeof(int32_t) * 4)));
using vecf32x4 = float
    __attribute__((__vector_size__(sizeof(float) * 4)));

using vecu64x2 = uint64_t
    __attribute__((__vector_size__(sizeof(uint64_t) * 2)));
using veci64x2 = int64_t
    __attribute__((__vector_size__(sizeof(int64_t) * 2)));
using vecf64x2 = double
    __attribute__((__vector_size__(sizeof(double) * 2)));

//
// 256-bit vectors

using vecu8x32 = uint8_t
    __attribute__((__vector_size__(sizeof(uint8_t) * 32)));
using veci8x32 = int8_t
    __attribute__((__vector_size__(sizeof(int8_t) * 32)));

using vecu16x16 = uint16_t
    __attribute__((__vector_size__(sizeof(uint16_t) * 16)));
using veci16x16 = int16_t
    __attribute__((__vector_size__(sizeof(int16_t) * 16)));

using vecu32x8 = uint32_t
    __attribute__((__vector_size__(sizeof(uint32_t) * 8)));
using veci32x8 = int32_t
    __attribute__((__vector_size__(sizeof(int32_t) * 8)));
using vecf32x8 = float
    __attribute__((__vector_size__(sizeof(float) * 8)));

using veci64x4 = int64_t
    __attribute__((__vector_size__(sizeof(int64_t) * 4)));
using vecu64x4 = uint64_t
    __attribute__((__vector_size__(sizeof(uint64_t) * 4)));
using vecf64x4 = double
    __attribute__((__vector_size__(sizeof(double) * 4)));

//
// 512-bit vectors

using vecu8x64 = uint8_t
    __attribute__((__vector_size__(sizeof(uint8_t) * 64)));
using veci8x64 = int8_t
    __attribute__((__vector_size__(sizeof(int8_t) * 64)));

using vecu16x32 = uint16_t
    __attribute__((__vector_size__(sizeof(uint16_t) * 32)));
using veci16x32 = int16_t
    __attribute__((__vector_size__(sizeof(int16_t) * 32)));

using vecu32x16 = uint32_t
    __attribute__((__vector_size__(sizeof(uint32_t) * 16)));
using veci32x16 = int32_t
    __attribute__((__vector_size__(sizeof(int32_t) * 16)));
using vecf32x16 = float
    __attribute__((__vector_size__(sizeof(float) * 16)));

using vecu64x8 = uint64_t
    __attribute__((__vector_size__(sizeof(uint64_t) * 8)));
using veci64x8 = int64_t
    __attribute__((__vector_size__(sizeof(int64_t) * 8)));
using vecf64x8 = double
    __attribute__((__vector_size__(sizeof(double) * 8)));

//
// Forward lookup

template<typename T, size_t N> struct vec;

// Define the 128-, 256-, and 512-bit versions of each component type

template<> struct vec<uint8_t, 16> { using type = vecu8x16; };
template<> struct vec<uint8_t, 32> { using type = vecu8x32; };
template<> struct vec<uint8_t, 64> { using type = vecu8x64; };

template<> struct vec<int8_t, 16> { using type = veci8x16; };
template<> struct vec<int8_t, 32> { using type = veci8x32; };
template<> struct vec<int8_t, 64> { using type = veci8x64; };

template<> struct vec<uint16_t, 8> { using type = vecu16x8; };
template<> struct vec<uint16_t, 16> { using type = vecu16x16; };
template<> struct vec<uint16_t, 32> { using type = vecu16x32; };

template<> struct vec<int16_t, 8> { using type = veci16x8; };
template<> struct vec<int16_t, 16> { using type = veci16x16; };
template<> struct vec<int16_t, 32> { using type = veci16x32; };

template<> struct vec<uint32_t, 4> { using type = vecu32x4; };
template<> struct vec<uint32_t, 8> { using type = vecu32x8; };
template<> struct vec<uint32_t, 16> { using type = vecu32x16; };

template<> struct vec<int32_t, 4> { using type = veci32x4; };
template<> struct vec<int32_t, 8> { using type = veci32x8; };
template<> struct vec<int32_t, 16> { using type = veci32x16; };

template<> struct vec<uint64_t, 2> { using type = vecu64x2; };
template<> struct vec<uint64_t, 4> { using type = vecu64x4; };
template<> struct vec<uint64_t, 8> { using type = vecu64x8; };

template<> struct vec<int64_t, 2> { using type = veci64x2; };
template<> struct vec<int64_t, 4> { using type = veci64x4; };
template<> struct vec<int64_t, 8> { using type = veci64x8; };

template<> struct vec<float, 4> { using type = vecf32x4; };
template<> struct vec<float, 8> { using type = vecf32x8; };
template<> struct vec<float, 16> { using type = vecf32x16; };

template<> struct vec<double, 2> { using type = vecf64x2; };
template<> struct vec<double, 4> { using type = vecf64x4; };
template<> struct vec<double, 8> { using type = vecf64x8; };

template<typename Tcomp, size_t N>
using vec_t = typename vec<Tcomp, N>::type;

//
// Reverse lookup

template<typename T> struct to_vec;

template<> struct to_vec<vecu8x16> { using type = vec<uint8_t, 16>; };
template<> struct to_vec<veci8x16> { using type = vec<int8_t, 16>; };

template<> struct to_vec<vecu16x8> { using type = vec<uint16_t, 8>; };
template<> struct to_vec<veci16x8> { using type = vec<int16_t, 8>; };

template<> struct to_vec<vecu32x4> { using type = vec<uint32_t, 4>; };
template<> struct to_vec<veci32x4> { using type = vec<int32_t, 4>; };
template<> struct to_vec<vecf32x4> { using type = vec<float, 4>; };

template<> struct to_vec<vecu64x2> { using type = vec<uint64_t, 2>; };
template<> struct to_vec<veci64x2> { using type = vec<int64_t, 2>; };
template<> struct to_vec<vecf64x2> { using type = vec<double, 2>; };

template<> struct to_vec<vecu8x32> { using type = vec<uint8_t, 32>; };
template<> struct to_vec<veci8x32> { using type = vec<int8_t, 32>; };

template<> struct to_vec<vecu16x16> { using type = vec<uint16_t, 16>; };
template<> struct to_vec<veci16x16> { using type = vec<int16_t, 16>; };

template<> struct to_vec<vecu32x8> { using type = vec<uint32_t, 8>; };
template<> struct to_vec<veci32x8> { using type = vec<int32_t, 8>; };
template<> struct to_vec<vecf32x8> { using type = vec<float, 8>; };

template<> struct to_vec<vecu64x4> { using type = vec<uint64_t, 4>; };
template<> struct to_vec<veci64x4> { using type = vec<int64_t, 4>; };
template<> struct to_vec<vecf64x4> { using type = vec<double, 4>; };

template<> struct to_vec<vecu8x64> { using type = vec<uint8_t, 64>; };
template<> struct to_vec<veci8x64> { using type = vec<int8_t, 64>; };

template<> struct to_vec<vecu16x32> { using type = vec<uint16_t, 32>; };
template<> struct to_vec<veci16x32> { using type = vec<int16_t, 32>; };

template<> struct to_vec<vecu32x16> { using type = vec<uint32_t, 16>; };
template<> struct to_vec<veci32x16> { using type = vec<int32_t, 16>; };
template<> struct to_vec<vecf32x16> { using type = vec<float, 16>; };

template<> struct to_vec<vecu64x8> { using type = vec<uint64_t, 8>; };
template<> struct to_vec<veci64x8> { using type = vec<int64_t, 8>; };
template<> struct to_vec<vecf64x8> { using type = vec<double, 8>; };

template<typename T>
using to_vec_t = typename to_vec<T>::type;

template<typename T>
struct component_of;

template <> struct component_of<vecu8x16> { using type = uint8_t; };
template <> struct component_of<vecu8x32> { using type = uint8_t; };
template <> struct component_of<vecu8x64> { using type = uint8_t; };

template <> struct component_of<veci8x16> { using type = int8_t; };
template <> struct component_of<veci8x32> { using type = int8_t; };
template <> struct component_of<veci8x64> { using type = int8_t; };

template <> struct component_of<vecu16x8> { using type = uint16_t; };
template <> struct component_of<vecu16x16> { using type = uint16_t; };
template <> struct component_of<vecu16x32> { using type = uint16_t; };

template <> struct component_of<veci16x8> { using type = int16_t; };
template <> struct component_of<veci16x16> { using type = int16_t; };
template <> struct component_of<veci16x32> { using type = int16_t; };

template <> struct component_of<vecu32x4> { using type = uint32_t; };
template <> struct component_of<vecu32x8> { using type = uint32_t; };
template <> struct component_of<vecu32x16> { using type = uint32_t; };

template <> struct component_of<veci32x4> { using type = int32_t; };
template <> struct component_of<veci32x8> { using type = int32_t; };
template <> struct component_of<veci32x16> { using type = int32_t; };

template <> struct component_of<vecf32x4> { using type = float; };
template <> struct component_of<vecf32x8> { using type = float; };
template <> struct component_of<vecf32x16> { using type = float; };

template <> struct component_of<vecu64x2> { using type = uint64_t; };
template <> struct component_of<vecu64x4> { using type = uint64_t; };
template <> struct component_of<vecu64x8> { using type = uint64_t; };

template <> struct component_of<veci64x2> { using type = int64_t; };
template <> struct component_of<veci64x4> { using type = int64_t; };
template <> struct component_of<veci64x8> { using type = int64_t; };

template <> struct component_of<vecf64x2> { using type = double; };
template <> struct component_of<vecf64x4> { using type = double; };
template <> struct component_of<vecf64x8> { using type = double; };

template<typename T>
using component_of_t = typename component_of<T>::type;

template<typename T>
struct comp_count;

template<> struct comp_count<vecu8x16> { static constexpr size_t value = 16; };
template<> struct comp_count<veci8x16> { static constexpr size_t value = 16; };
template<> struct comp_count<vecu16x8> { static constexpr size_t value = 8; };
template<> struct comp_count<veci16x8> { static constexpr size_t value = 8; };
template<> struct comp_count<vecu32x4> { static constexpr size_t value = 4; };
template<> struct comp_count<veci32x4> { static constexpr size_t value = 4; };
template<> struct comp_count<vecu64x2> { static constexpr size_t value = 2; };
template<> struct comp_count<veci64x2> { static constexpr size_t value = 2; };

template<> struct comp_count<vecu8x32> { static constexpr size_t value = 32; };
template<> struct comp_count<veci8x32> { static constexpr size_t value = 32; };
template<> struct comp_count<vecu16x16> { static constexpr size_t value = 16; };
template<> struct comp_count<veci16x16> { static constexpr size_t value = 16; };
template<> struct comp_count<vecu32x8> { static constexpr size_t value = 8; };
template<> struct comp_count<veci32x8> { static constexpr size_t value = 8; };
template<> struct comp_count<vecu64x4> { static constexpr size_t value = 4; };
template<> struct comp_count<veci64x4> { static constexpr size_t value = 4; };

template<> struct comp_count<vecf32x4> { static constexpr size_t value = 4; };
template<> struct comp_count<vecf32x8> { static constexpr size_t value = 8; };
template<> struct comp_count<vecf32x16> { static constexpr size_t value = 16; };

template<> struct comp_count<vecf64x2> { static constexpr size_t value = 2; };
template<> struct comp_count<vecf64x4> { static constexpr size_t value = 4; };
template<> struct comp_count<vecf64x8> { static constexpr size_t value = 8; };

template<> struct comp_count<vecu8x64> { static constexpr size_t value = 64; };
template<> struct comp_count<veci8x64> { static constexpr size_t value = 64; };
template<> struct comp_count<vecu16x32> { static constexpr size_t value = 32; };
template<> struct comp_count<veci16x32> { static constexpr size_t value = 32; };
template<> struct comp_count<vecu32x16> { static constexpr size_t value = 16; };
template<> struct comp_count<veci32x16> { static constexpr size_t value = 16; };
template<> struct comp_count<vecu64x8> { static constexpr size_t value = 8; };
template<> struct comp_count<veci64x8> { static constexpr size_t value = 8; };

template<typename T>
constexpr auto comp_count_v = comp_count<T>::value;

template<size_t N> struct int_by_size;

template<> struct int_by_size<sizeof(int8_t)>
    { using type = int8_t; };

template<> struct int_by_size<sizeof(int16_t)>
    { using type = int16_t; };

template<> struct int_by_size<sizeof(int32_t)>
    { using type = int32_t; };

template<> struct int_by_size<sizeof(int64_t)>
    { using type = int64_t; };

template<> struct int_by_size<sizeof(__int128_t)>
    { using type = __int128_t; };

template<size_t N> struct uint_by_size;
template<> struct uint_by_size<sizeof(uint8_t)>
    { using type = uint8_t; };

template<> struct uint_by_size<sizeof(uint16_t)>
    { using type = uint16_t; };

template<> struct uint_by_size<sizeof(uint32_t)>
    { using type = uint32_t; };

template<> struct uint_by_size<sizeof(uint64_t)>
    { using type = uint64_t; };

template<> struct uint_by_size<sizeof(__uint128_t)>
    { using type = __uint128_t; };

template<size_t N> struct float_by_size;
// template<> struct float_by_size<sizeof(__fp16)>
//     { using type = __fp16; };
template<> struct float_by_size<sizeof(float)>
    { using type = float; };
template<> struct float_by_size<sizeof(double)>
    { using type = double; };
// template<> struct float_by_size<sizeof(long double)>
//     { using type = long double; };
// template<> struct float_by_size<sizeof(__float128)>
//     { using type = __float128; };

template<size_t N>
using float_by_size_t = typename float_by_size<N>::type;

template<size_t N>
using int_by_size_t = typename int_by_size<N>::type;

template<size_t N>
using uint_by_size_t = typename uint_by_size<N>::type;

#ifndef always_inline
#define always_inline __attribute__((__always_inline__)) static inline
#define always_inline_method __attribute__((__always_inline__)) inline
#endif

always_inline vecu32x8 vec_lo(vecu32x16 whole)
{
    return vecu32x8{
        whole[0], whole[1], whole[2], whole[3],
        whole[4], whole[5], whole[6], whole[7]
    };
}

always_inline veci32x8 vec_lo(veci32x16 whole)
{
    return veci32x8{
        whole[0], whole[1], whole[2], whole[3],
        whole[4], whole[5], whole[6], whole[7]
    };
}

always_inline vecf32x8 vec_lo(vecf32x16 whole)
{
    return vecf32x8{
        whole[0], whole[1], whole[2], whole[3],
        whole[4], whole[5], whole[6], whole[7]
    };
}

always_inline vecf32x4 vec_lo(vecf32x8 whole)
{
    return vecf32x4{
        whole[0], whole[1], whole[2], whole[3]
    };
}

always_inline vecu32x4 vec_lo(vecu32x8 whole)
{
    return vecu32x4{
        whole[0], whole[1], whole[2], whole[3]
    };
}

always_inline veci32x4 vec_lo(veci32x8 whole)
{
    return veci32x4{
        whole[0], whole[1], whole[2], whole[3]
    };
}

always_inline vecu32x8 vec_combine(vecu32x4 lo, vecu32x4 hi)
{
    return vecu32x8{
        lo[0], lo[1], lo[2], lo[3],
        hi[0], hi[1], hi[2], hi[3]
    };
}

always_inline veci32x8 vec_combine(veci32x4 lo, veci32x4 hi)
{
    return veci32x8{
        lo[0], lo[1], lo[2], lo[3],
        hi[0], hi[1], hi[2], hi[3]
    };
}

always_inline vecu32x16 vec_combine(vecu32x8 lo, vecu32x8 hi)
{
    return vecu32x16{
        lo[0], lo[1], lo[2], lo[3], lo[4], lo[5], lo[6], lo[7],
        hi[0], hi[1], hi[2], hi[3], hi[4], hi[5], hi[6], hi[7]
    };
}

always_inline vecf32x16 vec_combine(vecf32x8 lo, vecf32x8 hi)
{
    return vecf32x16{
        lo[0], lo[1], lo[2], lo[3], lo[4], lo[5], lo[6], lo[7],
        hi[0], hi[1], hi[2], hi[3], hi[4], hi[5], hi[6], hi[7]
    };
}

always_inline vecf32x8 vec_combine(vecf32x4 lo, vecf32x4 hi)
{
    return vecf32x8{
        lo[0], lo[1], lo[2], lo[3],
        hi[0], hi[1], hi[2], hi[3]
    };
}

always_inline vecu32x8 vec_hi(vecu32x16 whole)
{
    return vecu32x8{
        whole[ 8], whole[ 9], whole[10], whole[11],
        whole[12], whole[13], whole[14], whole[15]
    };
}

always_inline veci32x8 vec_hi(veci32x16 whole)
{
    return veci32x8{
        whole[ 8], whole[ 9], whole[10], whole[11],
        whole[12], whole[13], whole[14], whole[15]
    };
}

always_inline vecf32x8 vec_hi(vecf32x16 whole)
{
    return vecf32x8{
        whole[ 8], whole[ 9], whole[10], whole[11],
        whole[12], whole[13], whole[14], whole[15]
    };
}

always_inline vecu32x4 vec_hi(vecu32x8 whole)
{
    return vecu32x4{
        whole[ 4], whole[ 5], whole[ 6], whole[ 7]
    };
}

always_inline veci32x4 vec_hi(veci32x8 whole)
{
    return veci32x4{
        whole[ 4], whole[ 5], whole[ 6], whole[ 7]
    };
}

always_inline vecf32x4 vec_hi(vecf32x8 whole)
{
    return vecf32x4{
        whole[ 4], whole[ 5], whole[ 6], whole[ 7]
    };
}

template<size_t sz>
struct vec_info_of_sz;

template<>
struct vec_info_of_sz<16> {
    using as_float = vecf32x16;
    using as_unsigned = vecu32x16;
    using as_int = veci32x16;

    using compact_bitmask = uint16_t;
    using bitmask = unsigned;

    static constexpr size_t sz = 16;
    static constexpr vecu32x16 lanebit[] = {
        0x0001, 0x0002, 0x0004, 0x0008,
        0x0010, 0x0020, 0x0040, 0x0080,
        0x0100, 0x0200, 0x0400, 0x0800,
        0x1000, 0x2000, 0x4000, 0x8000
    };

    static constexpr vecu32x16 lanemask[] = {
        {   0,   0,   0,   0,   0,   0,   0,   0,    0,   0,   0,   0,   0,   0,   0,   0 },
        { -1U,   0,   0,   0,   0,   0,   0,   0,    0,   0,   0,   0,   0,   0,   0,   0 },
        { -1U, -1U,   0,   0,   0,   0,   0,   0,    0,   0,   0,   0,   0,   0,   0,   0 },
        { -1U, -1U, -1U,   0,   0,   0,   0,   0,    0,   0,   0,   0,   0,   0,   0,   0 },
        { -1U, -1U, -1U, -1U,   0,   0,   0,   0,    0,   0,   0,   0,   0,   0,   0,   0 },
        { -1U, -1U, -1U, -1U, -1U,   0,   0,   0,    0,   0,   0,   0,   0,   0,   0,   0 },
        { -1U, -1U, -1U, -1U, -1U, -1U,   0,   0,    0,   0,   0,   0,   0,   0,   0,   0 },
        { -1U, -1U, -1U, -1U, -1U, -1U, -1U,   0,    0,   0,   0,   0,   0,   0,   0,   0 },
        { -1U, -1U, -1U, -1U, -1U, -1U, -1U, -1U,    0,   0,   0,   0,   0,   0,   0,   0 },
        { -1U, -1U, -1U, -1U, -1U, -1U, -1U, -1U,  -1U,   0,   0,   0,   0,   0,   0,   0 },
        { -1U, -1U, -1U, -1U, -1U, -1U, -1U, -1U,  -1U, -1U,   0,   0,   0,   0,   0,   0 },
        { -1U, -1U, -1U, -1U, -1U, -1U, -1U, -1U,  -1U, -1U, -1U,   0,   0,   0,   0,   0 },
        { -1U, -1U, -1U, -1U, -1U, -1U, -1U, -1U,  -1U, -1U, -1U, -1U,   0,   0,   0,   0 },
        { -1U, -1U, -1U, -1U, -1U, -1U, -1U, -1U,  -1U, -1U, -1U, -1U, -1U,   0,   0,   0 },
        { -1U, -1U, -1U, -1U, -1U, -1U, -1U, -1U,  -1U, -1U, -1U, -1U, -1U, -1U,   0,   0 },
        { -1U, -1U, -1U, -1U, -1U, -1U, -1U, -1U,  -1U, -1U, -1U, -1U, -1U, -1U, -1U,   0 },
        { -1U, -1U, -1U, -1U, -1U, -1U, -1U, -1U,  -1U, -1U, -1U, -1U, -1U, -1U, -1U, -1U }
    };
    static constexpr vecf32x16 laneoffs = {
        0.0f,  1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,
        8.0f,  9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f
    };
    static constexpr veci32x16 laneoffsi = {
        0,  1,  2,  3,  4,  5,  6,  7,
        8,  9, 10, 11, 12, 13, 14, 15
    };
    static constexpr vecu32x16 laneoffsu = {
        0,  1,  2,  3,  4,  5,  6,  7,
        8,  9, 10, 11, 12, 13, 14, 15
    };
    always_inline constexpr vecf32x16 vec_broadcast(float n)
    {
        return vecf32x16{
            n, n, n, n,
            n, n, n, n,
            n, n, n, n,
            n, n, n, n
        };
    }
    always_inline constexpr vecu32x16 vec_broadcast(unsigned n)
    {
        return vecu32x16{
            n, n, n, n,
            n, n, n, n,
            n, n, n, n,
            n, n, n, n
        };
    }
    always_inline constexpr veci32x16 vec_broadcast(int n)
    {
        return veci32x16{
            n, n, n, n,
            n, n, n, n,
            n, n, n, n,
            n, n, n, n
        };
    }
};

template<>
struct vec_info_of_sz<8> {
    using as_float = vecf32x8;
    using as_unsigned = vecu32x8;
    using as_int = veci32x8;

    using compact_bitmask = uint8_t;
    using bitmask = unsigned;

    static constexpr size_t sz = 8;
    static constexpr vecu32x8 lanebit[] = {
        0x01, 0x02, 0x04, 0x08,
        0x10, 0x20, 0x40, 0x80
    };
    static constexpr vecu32x8 lanemask[] = {
        {   0,   0,   0,   0,   0,   0,   0,   0 },
        { -1U,   0,   0,   0,   0,   0,   0,   0 },
        { -1U, -1U,   0,   0,   0,   0,   0,   0 },
        { -1U, -1U, -1U,   0,   0,   0,   0,   0 },
        { -1U, -1U, -1U, -1U,   0,   0,   0,   0 },
        { -1U, -1U, -1U, -1U, -1U,   0,   0,   0 },
        { -1U, -1U, -1U, -1U, -1U, -1U,   0,   0 },
        { -1U, -1U, -1U, -1U, -1U, -1U, -1U,   0 },
        { -1U, -1U, -1U, -1U, -1U, -1U, -1U, -1U }
    };
    static constexpr vecf32x8 laneoffs = {
        0.0f,  1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f
    };
    static constexpr vecu32x8 laneoffsu = {
        0,  1,  2,  3,  4,  5,  6,  7
    };
    static constexpr veci32x8 laneoffsi = {
        0,  1,  2,  3,  4,  5,  6,  7
    };
    static constexpr vecf32x8 vec_broadcast(float n)
    {
        return vecf32x8{
            n, n, n, n,
            n, n, n, n
        };
    }
    static constexpr vecu32x8 vec_broadcast(unsigned n)
    {
        return vecu32x8{
            n, n, n, n,
            n, n, n, n
        };
    }
    static constexpr veci32x8 vec_broadcast(int n)
    {
        return veci32x8{
            n, n, n, n,
            n, n, n, n
        };
    }
};

template<typename To, typename From>
always_inline To convert_to(From const& orig)
{
    return __builtin_convertvector(orig, To);
}

template<typename As, typename From>
always_inline constexpr As cast_to(From const& rhs)
{
    return __builtin_bit_cast(As, rhs);
}

always_inline constexpr vecu32x4 vec_blend(
    vecu32x4 existing, vecu32x4 updated, veci32x4 mask)
{
    return (updated & mask) | (existing & ~mask);
}

always_inline vecu32x16 vec_blend(
    vecu32x16 existing, vecu32x16 updated, veci32x16 mask)
{
#if defined(__AVX512F__)
    return cast_to<vecu32x16>(
        _mm512_blendv_epi8(
        cast_to<__m512i>(existing),
        cast_to<__m512i>(updated),
        cast_to<__m512i>(mask)));
#elif defined(__AVX2__)
    vecu32x8 lo = cast_to<vecu32x8>(
        _mm256_blendv_epi8(
        cast_to<__m256i>(vec_lo(existing)),
        cast_to<__m256i>(vec_lo(updated)),
        cast_to<__m256i>(vec_lo(mask))));
    vecu32x8 hi = cast_to<vecu32x8>(
        _mm256_blendv_epi8(
        cast_to<__m256i>(vec_hi(existing)),
        cast_to<__m256i>(vec_hi(updated)),
        cast_to<__m256i>(vec_hi(mask))));

    return vec_combine(lo, hi);
#else
    return (existing & ~mask) | (updated & mask);
#endif
}

always_inline vecu32x8 vec_blend(
    vecu32x8 existing, vecu32x8 updated, veci32x8 mask)
{
#if defined(__AVX2__)
    return cast_to<vecu32x8>(
        _mm256_blendv_epi8(
        cast_to<__m256i>(existing),
        cast_to<__m256i>(updated),
        cast_to<__m256i>(mask)));
#else
    return (existing & ~mask) | (updated & mask);
#endif
}

always_inline vecf32x8 vec_blend(
    vecf32x8 existing, vecf32x8 updated, veci32x8 mask)
{
#if defined(__AVX2__)
    return cast_to<vecf32x8>(
        _mm256_blendv_ps(
        cast_to<__m256>(existing),
        cast_to<__m256>(updated),
        cast_to<__m256>(mask)));
#else
    return cast_to<vecf32x8>(
        (cast_to<vecu32x8>(existing) & ~mask) |
        (cast_to<vecu32x8>(updated) & mask));
#endif
}

always_inline vecf32x16 vec_blend(
    vecf32x16 existing, vecf32x16 updated, veci32x16 mask)
{
#if defined(__AVX2__)
    vecf32x8 lo = cast_to<vecf32x8>(
        _mm256_blendv_ps(
        cast_to<__m256>(vec_lo(existing)),
        cast_to<__m256>(vec_lo(updated)),
        cast_to<__m256>(vec_lo(mask))));
    vecf32x8 hi = cast_to<vecf32x8>(
        _mm256_blendv_ps(
        cast_to<__m256>(vec_hi(existing)),
        cast_to<__m256>(vec_hi(updated)),
        cast_to<__m256>(vec_hi(mask))));
    return vec_combine(lo, hi);
#else
    return cast_to<vecf32x16>(
        (cast_to<vecu32x16>(existing) & ~mask) |
        (cast_to<vecu32x16>(updated) & mask));
#endif
}

always_inline vecf32x4 vec_blend(
    vecf32x4 existing, vecf32x4 updated, veci32x4 mask)
{
#if defined(__SSE41__)
    return cast_to<vecf32x4>(
        _mm_blendv_ps(
        cast_to<__m128>(existing),
        cast_to<__m128>(updated),
        cast_to<__m128>(mask)));
#else
    return cast_to<vecf32x4>(
        (cast_to<vecu32x4>(existing) & ~mask) |
        (cast_to<vecu32x4>(updated) & mask));
#endif
}

template<typename V, typename M,
    typename VC = component_of_t<V>,
    typename MC = component_of_t<M>,
    size_t VN = comp_count_v<V>,
    size_t MN = comp_count_v<M>,
    typename = typename std::enable_if_t<VN == MN>>
always_inline V vec_blend(V const& existing,
    V const& updated, M const& mask)
{
    return vec_blend(existing, updated,
        cast_to<vec_t<int, MN>>(mask));
}

always_inline vecf32x4 max(vecf32x4 const& a, vecf32x4 const& b)
{
#if defined(__SSE2__)
    return cast_to<vecf32x4>(
        _mm_max_ps(
        cast_to<__m128>(a),
        cast_to<__m128>(b)));
#elif defined(__ARM_NEON)
    return cast_to<vecf32x4>(vmaxq_f32(
        cast_to<__m128>(a),
        cast_to<__m128>(b)));
#else
    return {
        a[0] >= b[0] ? a[0] : b[0],
        a[1] >= b[1] ? a[1] : b[1],
        a[2] >= b[2] ? a[2] : b[2],
        a[3] >= b[3] ? a[3] : b[3]
    };
#endif
}

always_inline vecu32x8 min(vecu32x8 const &a, vecu32x8 const& b)
{
#if defined(__SSE41__)
    vecu32x4 lo = cast_to<vecu32x4>(
        _mm_min_epu32(
            cast_to<__m128i>(vec_lo(a)),
            cast_to<__m128i>(vec_lo(b))));
    vecu32x4 hi = cast_to<vecu32x4>(
        _mm_min_epu32(
            cast_to<__m128i>(vec_hi(a)),
            cast_to<__m128i>(vec_hi(b))));
    return vec_combine(lo, hi);
#elif defined(__SSE2__)
    vecu32x4 alo = vec_lo(a);
    vecu32x4 blo = vec_lo(b);
    vecu32x4 ahi = vec_hi(a);
    vecu32x4 bhi = vec_hi(b);
    veci32x4 lma = alo <= blo;
    veci32x4 hma = ahi <= bhi;
    vecu32x4 rlo = vec_blend(blo, alo, lma);
    vecu32x4 rhi = vec_blend(bhi, ahi, hma);
    return vec_combine(rlo, rhi);
#else
    return cast_to<vecu32x8>(
        _mm256_min_epu32(
            cast_to<__m256i>(a),
            cast_to<__m256i>(b)));
#endif
}

always_inline veci32x8 min(veci32x8 const &a, veci32x8 const& b)
{
#if defined(__SSE2__)
    veci32x4 lo = cast_to<veci32x4>(
        _mm_min_epi32(
            cast_to<__m128i>(vec_lo(a)),
            cast_to<__m128i>(vec_lo(b))));
    veci32x4 hi = cast_to<veci32x4>(
        _mm_min_epi32(
            cast_to<__m128i>(vec_hi(a)),
            cast_to<__m128i>(vec_hi(b))));
    return vec_combine(lo, hi);
#else
    return cast_to<vecu32x8>(
        _mm256_min_epu32(
            cast_to<__m256i>(a),
            cast_to<__m256i>(b)));
#endif
}

always_inline vecf32x4 min(vecf32x4 const& a, vecf32x4 const& b)
{
#if defined(__SSE2__)
    return cast_to<vecf32x4>(
        _mm_min_ps(
        cast_to<__m128>(a),
        cast_to<__m128>(b)));
#elif defined(__ARM_NEON)
    return cast_to<vecf32x4>(vmaxq_f32(
        cast_to<__m128>(a),
        cast_to<__m128>(b)));
#else
    return {
        a[0] <= b[0] ? a[0] : b[0],
        a[1] <= b[1] ? a[1] : b[1],
        a[2] <= b[2] ? a[2] : b[2],
        a[3] <= b[3] ? a[3] : b[3]
    };
#endif
}

always_inline vecf32x8 min(
    vecf32x8 const& a, vecf32x8 const& b)
{
#if defined(__AVX2__)
    return cast_to<vecf32x8>(
        _mm256_min_ps(
        cast_to<__m256>(a),
        cast_to<__m256>(b)));
#elif defined(__SSE2__)
    auto lo = cast_to<vecf32x4>(
        _mm_min_ps(
            cast_to<__m128>(vec_lo(a)),
            cast_to<__m128>(vec_lo(b))));
    auto hi = cast_to<vecf32x4>(
        _mm_min_ps(
            cast_to<__m128>(vec_hi(a)),
            cast_to<__m128>(vec_hi(b))));
    return vec_combine(lo, hi);
#elif defined(__ARM_NEON)
    auto lo = cast_to<vecf32x4>(
        vminq_f32(
            cast_to<__m128>(vec_lo(a)),
            cast_to<__m128>(vec_lo(b))));
    auto hi = cast_to<vecf32x4>(
        vminq_f32(
            cast_to<__m128>(vec_hi(a)),
            cast_to<__m128>(vec_hi(b))));
    return vec_combine(lo, hi);
#else
    return vec_blend(a, b, a > b);
#endif
}

always_inline vecf32x8 max(vecf32x8 const& a, vecf32x8 const& b)
{
#if defined(__AVX2__)
    return cast_to<vecf32x8>(
        _mm256_max_ps(
            cast_to<__m256>(a),
            cast_to<__m256>(b)));
#elif defined(__SSE2__)
    auto lo = cast_to<vecf32x4>(
        _mm_max_ps(
            cast_to<__m128>(vec_lo(a)),
            cast_to<__m128>(vec_lo(b))));
    auto hi = cast_to<vecf32x4>(
        _mm_max_ps(
            cast_to<__m128>(vec_hi(a)),
            cast_to<__m128>(vec_hi(b))));
    return vec_combine(lo, hi);
#elif defined(__ARM_NEON)
    auto lo = cast_to<vecf32x4>(
        vmaxq_f32(
            cast_to<__m128>(vec_lo(a)),
            cast_to<__m128>(vec_lo(b))));
    auto hi = cast_to<vecf32x4>(
        vmaxq_f32(
            cast_to<__m128>(vec_hi(a)),
            cast_to<__m128>(vec_hi(b))));
    return vec_combine(lo, hi);
#else
    return vec_blend(a, b, b > a);
#endif
}

template<typename T,
    typename = to_vec_t<T>>
always_inline T ntload(T *address)
{
    return __builtin_nontemporal_load(address);
}

template<typename From,
    typename Tcomp = component_of_t<From>,
    size_t N = comp_count_v<From>,
    typename Tbigger = next_bigger_t<Tcomp>,
    typename Vbigger = vec_t<Tbigger, N / 2>>
always_inline void unpack(
    Vbigger *lo_result, Vbigger *hi_result, From const& rhs)
{
    constexpr auto shift = sizeof(Tbigger) * CHAR_BIT / 2;
    constexpr auto mask = ~-(Tcomp(1) << shift);
    *lo_result = cast_to<Vbigger>(rhs) & mask;
    *hi_result = cast_to<Vbigger>(rhs >> shift);
}

template<typename From,
    typename Tcomp = component_of_t<From>,
    size_t N = comp_count_v<From>,
    typename Tsmaller = next_smaller_t<Tcomp>,
    typename Vsmaller = vec_t<Tsmaller, N * 2>>
always_inline Vsmaller pack(From const& lo, From const& hi)
{
    constexpr auto shift = sizeof(Tsmaller) * CHAR_BIT / 2;
    constexpr auto mask = ~-(Tcomp(1) << shift);
    Vsmaller lo_result = cast_to<Vsmaller>(lo & mask);
    Vsmaller hi_result = cast_to<Vsmaller>((hi & mask) << shift);
    lo_result |= hi_result;
    return lo_result;
}

template<>
struct vec_info_of_sz<4> {
    using as_float = vecf32x4;
    using as_unsigned = vecu32x4;
    using as_int = veci32x4;

    using compact_bitmask = uint8_t;
    using bitmask = unsigned;

    static constexpr size_t sz = 4;
    static constexpr vecu32x4 lanebit[] = {
        0x0001, 0x0002, 0x0004, 0x0008
    };

    static constexpr vecu32x4 lanemask[] = {
        {   0,   0,   0,   0 },
        { -1U,   0,   0,   0 },
        { -1U, -1U,   0,   0 },
        { -1U, -1U, -1U,   0 },
        { -1U, -1U, -1U, -1U }
    };
    static constexpr vecf32x4 laneoffs = {
        0.0f,  1.0f,  2.0f,  3.0f
    };
    static constexpr vecu32x4 laneoffsu = {
        0,  1,  2,  3
    };
    static constexpr veci32x4 laneoffsi = {
        0,  1,  2,  3
    };
    static constexpr vecf32x4 vec_broadcast(float n)
    {
        return vecf32x4{
            n, n, n, n
        };
    }
    static constexpr vecu32x4 vec_broadcast(unsigned n)
    {
        return vecu32x4{
            n, n, n, n
        };
    }
    static constexpr veci32x4 vec_broadcast(int n)
    {
        return veci32x4{
            n, n, n, n
        };
    }
};

template<>
struct vec_info_of_sz<2> {
    using as_float = vecf64x2;
    using as_unsigned = vecu64x2;
    using as_int = veci64x2;

    using compact_bitmask = uint8_t;
    using bitmask = unsigned;

    static constexpr size_t sz = 2;
    static constexpr vecu64x2 lanebit[] = {
        0x0001, 0x0002
    };

    static constexpr vecu32x4 lanemask[] = {
        {   0,   0,   0,   0 },
        { -1U,   0,   0,   0 },
        { -1U, -1U,   0,   0 }
    };

    static constexpr vecf64x2 laneoffs = {
        0.0f,  1.0f
    };
    static constexpr vecu64x2 laneoffsu = {
        0,  1
    };
    static constexpr veci64x2 laneoffsi = {
        0,  1
    };

    always_inline constexpr vecf64x2 vec_broadcast(double n)
    {
        return vecf64x2{
            n, n
        };
    }

    always_inline constexpr vecu64x2 vec_broadcast(uint64_t n)
    {
        return vecu64x2{
            n, n
        };
    }

    always_inline constexpr veci64x2 vec_broadcast(int64_t n)
    {
        return veci64x2{
            n, n
        };
    }
};

template<typename T>
struct vecinfo_t;

template<>
struct vecinfo_t<vecf32x4> : public vec_info_of_sz<4> {
};

template<>
struct vecinfo_t<vecu32x4> : public vec_info_of_sz<4> {
};

template<>
struct vecinfo_t<veci32x4> : public vec_info_of_sz<4> {
};

template<>
struct vecinfo_t<vecf32x8> : public vec_info_of_sz<8> {
};

template<>
struct vecinfo_t<vecu32x8> : public vec_info_of_sz<8> {
};

template<>
struct vecinfo_t<veci32x8> : public vec_info_of_sz<8> {
};

template<>
struct vecinfo_t<vecf32x16> : public vec_info_of_sz<16> {
};

template<>
struct vecinfo_t<vecu32x16> : public vec_info_of_sz<16> {
};

template<>
struct vecinfo_t<veci32x16> : public vec_info_of_sz<16> {
};

template<typename F,
    typename C = component_of_t<F>>
always_inline constexpr F vec_broadcast(C&& rhs)
{
    return vecinfo_t<F>::vec_broadcast(
        std::forward<C>(rhs));
}

// It sucks, on AMD anyway.
// A bunch of scalar loads and inserts beat it, lol.
#define GATHER_IS_GOOD 1

// 128-bit gather
always_inline vecu32x4 vec_gather(uint32_t const *buffer,
    vecu32x4 indices, vecu32x4 background, veci32x4 mask)
{
#if GATHER_IS_GOOD && defined(__AVX2__)
    vecu32x4 result = cast_to<vecu32x4>(
        _mm_mask_i32gather_epi32(
        cast_to<__m128i>(background),
        reinterpret_cast<int const*>(buffer),
        cast_to<__m128i>(indices),
        cast_to<__m128i>(mask),
        sizeof(uint32_t)));
#elif 1
    vecu32x4 result{
        buffer[indices[0]],
        buffer[indices[1]],
        buffer[indices[2]],
        buffer[indices[3]]
    };
    result = vec_blend(background, result, mask);
#endif
    return result;
}

//
// 256-bit scatter

#if defined(__AVX512F__)
always_inline void vec_scatter(float *addr,
    vecu32x8 const& indices, vecf32x8 const& values)
{
    _mm256_i32scatter_ps(addr,
        cast_to<__m256i>(indices),
        cast_to<__m256>(values),
        sizeof(float));
}

always_inline void vec_scatter(uint32_t *addr,
    vecu32x8 const& indices, vecu32x8 const& values)
{
    _mm256_i32scatter_epi32(addr,
        cast_to<__m256i>(indices),
        cast_to<__m256i>(values),
        sizeof(uint32_t));
}

always_inline void vec_scatter(int32_t *addr,
    vecu32x8 const& indices, veci32x8 const& values)
{
    _mm256_i32scatter_epi32(addr,
        cast_to<__m256i>(indices),
        cast_to<__m256i>(values),
        sizeof(int32_t));
}
#endif

//
// 128-bit scatter

template<typename V,
    typename U,
    typename C,
    size_t N>
always_inline void vec_scatter(
    C *addr, U const& indices, V const& values)
{
    for (size_t i = 0; i < N; ++i)
        addr[indices[i]] = values[i];
}

template<typename X,
    typename V = to_vec_t<X>,
    typename C = component_of_t<X>,
    size_t N = comp_count_v<X>,
    typename U = typename V::as_unsigned>
always_inline void vec_scatter(
    C *addr, U indices, X values)
{
    vec_scatter(addr, indices, values);
}

#if defined(__AVX512F__)
always_inline void vec_scatter(float *addr,
    vecu32x4 const& indices, vecf32x4 const& values)
{
    _mm_i32scatter_ps(addr,
        cast_to<__m128i>(indices),
        cast_to<__m128>(values),
        sizeof(float));
}

always_inline void vec_scatter(int32_t *addr,
    vecu32x4 const& indices, veci32x4 const& values)
{
    _mm_i32scatter_epi32(addr,
        cast_to<__m128i>(indices),
        cast_to<__m128i>(values),
        sizeof(int32_t));
}
#endif

// 256-bit gather
always_inline vecu32x8 vec_gather(uint32_t const *buffer,
    vecu32x8 indices, vecu32x8 background, veci32x8 mask)
{
#if GATHER_IS_GOOD && defined(__AVX2__)
    vecu32x8 result = cast_to<vecu32x8>(
        _mm256_mask_i32gather_epi32(
        cast_to<__m256i>(background),
        reinterpret_cast<int const*>(buffer),
        cast_to<__m256i>(indices),
        cast_to<__m256i>(mask),
        sizeof(uint32_t)));
#else
    vecu32x8 result{
        buffer[indices[0]],
        buffer[indices[1]],
        buffer[indices[2]],
        buffer[indices[3]],
        buffer[indices[4]],
        buffer[indices[5]],
        buffer[indices[6]],
        buffer[indices[7]]
    };
    result = vec_blend(background, result, mask);
#endif
    return result;
}

// 512-bit gather
always_inline vecu32x16 vec_gather(uint32_t const *buffer,
    vecu32x16 indices, vecu32x16 background, veci32x16 mask)
{
#if GATHER_IS_GOOD && defined(__AVX512F__)
    __mmask8 compact_mask = _mm512_test_epi32_mask(
        _mm512_set1_epi32(0x80000000), mask);
    vecu32x16 result = cast_to<vecu32x16>(
        _mm512_mask_i32gather_epi32(
        background,
        reinterpret_cast<int const *>(texture->pixels),
        cast_to<__m512i>(indices),
        compact_mask,
        sizeof(uint32_t)));
#elif GATHER_IS_GOOD && defined(__AVX2__)
    // Pair of 256 bit operations
    vecu32x8 lo = cast_to<vecu32x8>(
        _mm256_mask_i32gather_epi32(
        cast_to<__m256i>(vec_lo(background)),
        reinterpret_cast<int const*>(buffer),
        cast_to<__m256i>(vec_lo(indices)),
        cast_to<__m256i>(vec_lo(mask)),
        sizeof(uint32_t)));
    vecu32x8 hi = cast_to<vecu32x8>(
        _mm256_mask_i32gather_epi32(
        cast_to<__m256i>(vec_hi(background)),
        reinterpret_cast<int const*>(buffer),
        cast_to<__m256i>(vec_hi(indices)),
        cast_to<__m256i>(vec_hi(mask)),
        sizeof(uint32_t)));
    vecu32x16 result = vec_combine(lo, hi);
#else
    // Generic
    vecu32x16 result{
        buffer[indices[0]],
        buffer[indices[1]],
        buffer[indices[2]],
        buffer[indices[3]],
        buffer[indices[4]],
        buffer[indices[5]],
        buffer[indices[6]],
        buffer[indices[7]],
        buffer[indices[8]],
        buffer[indices[9]],
        buffer[indices[10]],
        buffer[indices[11]],
        buffer[indices[12]],
        buffer[indices[13]],
        buffer[indices[14]],
        buffer[indices[15]]
    };
    result = vec_blend(background, result, mask);
#endif
    return result;
}

template<typename I, typename B, typename M,
    typename IC = component_of_t<I>,
    typename BC = component_of_t<B>,
    typename MC = component_of_t<M>,
    size_t IN = comp_count_v<I>,
    size_t BN = comp_count_v<B>,
    size_t MN = comp_count_v<M>,
    typename = std::enable_if_t<IN == MN && BN == MN>,
    typename IM = vec_t<int, BN>>
B vec_gather(BC const *buffer,
    I indices, B background, M mask)
{
    return vec_gather(buffer, indices,
        background, cast_to<IM>(mask));
}


template<typename T>
struct vec_larger_type;

template<>
struct vec_larger_type<vecu8x16> { using type = vecu16x8; };


always_inline vecf32x4 abs(vecf32x4 f)
{
    return cast_to<vecf32x4>(cast_to<vecu32x4>(f) &
        std::numeric_limits<uint32_t>::max());
}

always_inline vecf32x8 abs(vecf32x8 f)
{
    return cast_to<vecf32x8>(cast_to<vecu32x8>(f) &
        std::numeric_limits<uint32_t>::max());
}

always_inline vecf32x16 abs(vecf32x16 f)
{
    return cast_to<vecf32x16>(cast_to<vecu32x16>(f) &
        std::numeric_limits<uint32_t>::max());
}

template<typename V>
always_inline void cross(
    V& __restrict out_x,
    V&  __restrict out_y,
    V&  __restrict out_z,
    V const& __restrict lhs_x,
    V const& __restrict lhs_y,
    V const& __restrict lhs_z,
    V const& __restrict rhs_x,
    V const& __restrict rhs_y,
    V const& __restrict rhs_z)
{
    out_x = lhs_y * rhs_z - rhs_y * lhs_z;
    out_y = lhs_z * rhs_x - rhs_z * lhs_x;
    out_z = lhs_x * rhs_y - rhs_x * lhs_y;
}

template<typename V>
always_inline void cross_inplace(
    V& __restrict lhs_x,
    V& __restrict lhs_y,
    V& __restrict lhs_z,
    V const& __restrict rhs_x,
    V const& __restrict rhs_y,
    V const& __restrict rhs_z)
{
    V tmp_x = lhs_y * rhs_z - rhs_y * lhs_z;
    V tmp_y = lhs_z * rhs_x - rhs_z * lhs_x;
    V tmp_z = lhs_x * rhs_y - rhs_x * lhs_y;
    lhs_x = tmp_x;
    lhs_y = tmp_y;
    lhs_z = tmp_z;
}

// SIMD dot product
template<typename V>
always_inline constexpr V dot(
    V const& lhs_x, V const& lhs_y, V const& lhs_z,
    V const& rhs_x, V const& rhs_y, V const& rhs_z)
{
    return lhs_x * rhs_x +
        lhs_y * rhs_y +
        lhs_z * rhs_z;
}

// Return lowest set bit, or -1 if no bits set at all
always_inline constexpr int ffs0(uint32_t n)
{
    return __builtin_ffs(n) - 1;
}

// Return lowest set bit, or -1 if no bits set at all
always_inline int ffs0(uint64_t n)
{
    return __builtin_ffsll(n) - 1;
}

#if HAVE_VEC512
using vecf32auto = vecf32x16;
#elif HAVE_VEC256
using vecf32auto = vecf32x8;
#elif HAVE_VEC128
using vecf32auto = vecf32x4;
#else
#error Unknown prerferred vector size
#endif

template<typename V, typename U>
always_inline V vec_shuffle(V const& lhs, U const& sel)
{
    static_assert(sizeof(V) == sizeof(U));
    static_assert(comp_count_v<V> == comp_count_v<U>);
    static_assert(std::is_integral_v<component_of_t<U>>);
    return __builtin_shuffle(lhs, sel);
}

template<typename V,
    typename D = vecinfo_t<V>,
    size_t sz = comp_count_v<V>,
    typename R = typename D::as_unsigned>
always_inline R vec_lanemask(size_t n)
{
    return D::lanemask[std::min(n, sz)];
}

template<typename V,
    typename D = vecinfo_t<V>,
    typename R = typename D::as_unsigned>
always_inline R vec_laneoffsu()
{
    return D::laneoffsu;
}

template<typename V, typename U>
always_inline V vec_shuffle2(V const& lhs,
    V const& rhs, U const& sel)
{
    static_assert(sizeof(V) == sizeof(U));
    static_assert(comp_count_v<V> == comp_count_v<U>);
    static_assert(std::is_integral_v<component_of_t<U>>);
    return __builtin_shuffle(lhs, rhs, sel);
}

template<typename V, typename U>
V vec_permute(V const& lhs, U const& rhs)
{
    static_assert(comp_count_v<V> == comp_count_v<U>);

    V result;

    for (size_t c = 0; c < comp_count_v<V>; ++c)
        result[c] = lhs[rhs[c]];

    return result;
}

#if defined(__SSE2__)
always_inline vecf32x4 vec_permute(
    vecf32x4 const& lhs, vecu32x4 const& rhs)
{
    return cast_to<vecf32x4>(
        _mm_permutevar_ps(
            cast_to<__m128>(lhs),
            cast_to<__m128i>(rhs)));
}
#endif

#if defined(__AVX2__)
always_inline vecf32x8 vec_permute(
    vecf32x8 const& lhs, vecu32x8 const& rhs)
{
    return cast_to<vecf32x8>(
        _mm256_permutevar8x32_ps(
            cast_to<__m256>(lhs),
            cast_to<__m256i>(rhs)));
}
#else
// Synthesize N bit permute with 4 N/2 bit permutes
template<typename X,
    typename U = typename vecinfo_t<X>::as_unsigned>
always_inline X vec_permute_synthetic(
        X const& lhs, U const& sel)
{
    constexpr auto sz = comp_count_v<X>;
    constexpr auto half = sz >> 1;

    using C = component_of_t<X>;
    using Vhalf = vec_t<C, half>;

    Vhalf lhs_lo = cast_to<Vhalf>(vec_lo(lhs));
    Vhalf lhs_hi = cast_to<Vhalf>(vec_hi(lhs));

    Vhalf sel_lo = vec_lo(sel) & ~-sz;
    Vhalf sel_hi = vec_hi(sel) & ~-sz;

    Vhalf lo_sel_lo = (sel_lo < half);
    Vhalf hi_sel_lo = (sel_hi < half);

    // Get the stuff in the low half
    // selecting from the low half
    Vhalf res_lo = vec_permute(
        lhs_lo, sel_lo) & lo_sel_lo;

    // Get the stuff from the high half
    // selecting from the high half
    Vhalf res_hi = vec_permute(
        lhs_hi, sel_hi - half) & ~hi_sel_lo;

    // Get the stuff in the low half
    // selecting from the high half
    res_lo |= vec_permute(
        lhs_hi, sel_lo - half) & ~lo_sel_lo;

    // Get the stuff in the high half
    // selecting from the low half
    res_hi |= vec_permute(
        lhs_lo, sel_hi) & hi_sel_lo;

    // Combine the permutation
    return vec_combine(res_lo, res_hi);
}

// Synthesize 256 bit permute with 4 128 bit permutes
always_inline vecf32x8 vec_permute(
        vecf32x8 const& lhs, vecu32x8 const& sel)
{
    return vec_permute_synthetic(lhs, sel);
}

// Synthesize 512 bit permute with 4 256 bit permutes
always_inline vecf32x16 vec_permute(
        vecf32x16 const& lhs, vecu32x16 const& sel)
{
    return vec_permute_synthetic(lhs, sel);
}
#endif

#if defined(__AVX2__)
always_inline vecu32x8 vec_permute(
    vecu32x8 const& lhs, vecu32x8 const& rhs)
{
    return cast_to<vecu32x8>(
        _mm256_permutevar8x32_epi32(
            cast_to<__m256i>(lhs),
            cast_to<__m256i>(rhs)));
}
#endif

// Expects you to pass a reference to an unsigned,
// which we call `count`, which tracks how many
// components of the SoA have been used.
// This structure sets up a permutation mask
// to rapidly collect unmasked fields into the
// fields of the last struct of an SoA, also
// setting up the continuation to deposit the
// remaining fields onto the next SoA
template<typename V,
    typename U = typename vecinfo_t<V>::as_unsigned,
    size_t vec_sz = comp_count_v<V>>
struct deposit_mask {
    deposit_mask() = default;
    deposit_mask(deposit_mask const&) = delete;
    deposit_mask(deposit_mask&&) = default;
    deposit_mask& operator=(deposit_mask const&) = delete;
    deposit_mask& operator=(deposit_mask&&) = default;
    ~deposit_mask() = default;

    template<typename A, typename ...Args>
    deposit_mask(unsigned count, unsigned bitmask, A& soa_vector,
        Args&& ...args)
    {
        // Catch blundering callers preparing to deposit nothing
        assert(bitmask != 0);
        setup(count, bitmask,
            soa_vector, std::forward<Args>(args)...);
    }

    template<typename A, typename ...Args>
    bool setup(unsigned count, unsigned bitmask, A& soa_vector,
        Args&& ...args)
    {
        // Fill the lower components
        // with an integer sequence
        // to keep the existing components
        // in the source

        // Component number to put next item
        unsigned comp = count & ~-vec_sz;

        // Infer the blend mask from the permutation mask
        perm_selector = vec_broadcast<U>(-1U);

        unsigned ofs = 0;
        for (deposited = 0; bitmask && comp < vec_sz;
                ++deposited, ++comp, ++ofs) {
            int bit = ffs0(bitmask);
            ofs += bit;
            bitmask >>= bit + 1;
            perm_selector[comp] = ofs;
        }

        // Add a structure of arrays if necessary
        // This is a somewhat odd place to do this, but it is here
        // so we don't have to check every time we write a field
        // The penalty of this is that the aos might have one
        // entire unused object at the end.
        if (count == soa_vector.size() * vec_sz)
            soa_vector.emplace_back(std::forward<Args>(args)...);

        remainder = bitmask;

        return true;
    }

    always_inline_method
    bool need_continuation() const
    {
        return remainder != 0;
    }

    // Bump count forward by the amount deposited
    // Make a new deposit mask that continues
    // in the next structure of vectors
    template<typename C>
    bool continue_from(unsigned &count, C &soa_vector)
    {
        // Update count (after caller updates all the fields)
        count += deposited;

        // Clear deposited so we can catch
        // erroneous calls to deposit_into
        deposited = 0;

        // We expect small appends to be
        // more frequent than full ones
        if (remainder == 0)
            return false;

        // This sets up deposited, so it is okay
        // to deposit_into after this returns true
        return setup(count, remainder, soa_vector);
    }

    always_inline_method
    void deposit_into(V& field, V const& data)
    {
        assert(deposited);

        V positioned = vec_permute(
            data, perm_selector);
        field = vec_blend(field, positioned,
            perm_selector != -1U);
    }

    // The mask for the shuffle/permute
    // moving the values into position
    U perm_selector;

    // The number of items that get deposited
    unsigned deposited;

    // The bitmask with the
    // deposited bits cleared
    unsigned remainder;
};

template<typename V,
    typename D = vecinfo_t<V>,
    typename M = typename D::bitmask,
    typename U = typename D::as_unsigned>
always_inline V compress(V const& lhs, U& mask)
{
    M used = vec_movemask(mask);
    U new_mask = {};

    // Copy it so we can just move one
    // component at a time most simply
    V result = lhs;
    int out;

    for (out = 0; used; ++out) {
        int component = ffs0(used);
        used &= ~(1U << component);

        //if (out != component)
            result[out] = result[component];
    }

    mask = D::laneoffsi < out;
    return result;
}

template<typename F>
always_inline F mix(F const& x, F const& y, F const& a)
{
    return (x * (1.0f - a)) + (y * a);
}

template<typename F>
always_inline F newton_raphson_rsqrt(F const& x, F const& y)
{
    return y * (1.5f - 0.5f * x * y * y);
}

template<typename F>
always_inline F newton_raphson_sqrt(F const& x, F const& y)
{
    return 0.5f * (y + x / y);
}

always_inline vecf32x4 sqrt(vecf32x4 const& x)
{
#if defined(__SSE2__)
    vecf32x4 y = cast_to<vecf32x4>(_mm_sqrt_ps(
        cast_to<__m128>(x)));
    return y;
#else
    return {
        std::sqrt(x[0]),
        std::sqrt(x[1]),
        std::sqrt(x[2]),
        std::sqrt(x[3])
    };
#endif
}

always_inline vecf32x8 sqrt(vecf32x8 const& x)
{
#if defined(__AVX2__)
    vecf32x8 y = cast_to<vecf32x8>(
        _mm256_sqrt_ps(
            cast_to<__m256>(x)));
    return y;
#else
    return vec_combine(
        sqrt(vec_lo(x)),
        sqrt(vec_hi(x)));
#endif
}

always_inline vecf32x16 sqrt(vecf32x16 const& x)
{
#if defined(__AVX512F__)
    vecf32x16 y = cast_to<vecf32x16>(
        _mm512_sqrt_ps(
            cast_to<__m512>(x)));
    return y;
#else
    return vec_combine(
        sqrt(vec_lo(x)),
        sqrt(vec_hi(x)));
#endif
}

always_inline vecf32x4 inv_sqrt(vecf32x4 const& x)
{
#if defined(__SSE2__)
    vecf32x4 y = cast_to<vecf32x4>(_mm_rsqrt_ps(
        cast_to<__m128>(x)));
    y = newton_raphson_rsqrt(x, y);
    return y;
#else
    return 1.0f / sqrt(x);
#endif
}

always_inline vecf32x8 inv_sqrt(vecf32x8 const& x)
{
#if defined(__AVX2__)
    vecf32x8 y = cast_to<vecf32x8>(_mm256_rsqrt_ps(
        cast_to<__m256>(x)));
    y = newton_raphson_rsqrt(x, y);
    return y;
#else
    return vec_combine(
        inv_sqrt(vec_lo(x)),
        inv_sqrt(vec_hi(x)));
#endif
}

always_inline vecf32x16 inv_sqrt(vecf32x16 const& x)
{
#if defined(__AVX512F__)
    vecf32x16 y = cast_to<vecf32x16>(
        _mm512_rsqrt28_ps(
            cast_to<__m512>(x)));
    // it already did a Newton-Raphson iteration...
    return y;
#else
    return vec_combine(
        inv_sqrt(vec_lo(x)),
        inv_sqrt(vec_hi(x)));
#endif
}

template<typename F>
always_inline constexpr void vec_normalize(F &x, F &y, F &z)
{
    F sq_len = x * x + y * y + z * z;
    F inv_len = inv_sqrt(sq_len);
    x *= inv_len;
    y *= inv_len;
    z *= inv_len;
}

template<typename X>
always_inline bool vec_all_eq(X const& lhs, X const& rhs)
{
    for (size_t c = 0; c < comp_count_v<X>; ++c)
        if (lhs[c] != rhs[c])
            return false;
    return true;
}

#include "vector_movemask.h"

// Gives a generalized mask
// plus an example of "all true" in `second`
// It does this so you could, for a 128 bit
// example, use movmskps and get a 4 bit mask,
// or it could use movmskb and get a 16 bit mask
// It tells you what kind of mask with `second`
//
// any true: x.first != 0
// all false: x.first == 0
// all true: x.first == x.second
// any false: x.first != x.second
template<typename X>
always_inline std::pair<uint64_t, uint64_t>
vec_mask_pair(X const& i)
{
    using C = component_of_t<X>;
    constexpr auto csz = sizeof(C);
    constexpr auto integral = std::is_integral_v<C>;
    constexpr auto sz = comp_count_v<X>;

#if defined(__AVX512F__)
    if constexpr (integral &&
            csz == sizeof(uint64_t) &&
            sz == 4)
        return {
            _mm512_movepi64_mask(
                cast_to<__m512i>(i)),
            ~-(1U << sz)
        };
    if constexpr (integral &&
            csz == sizeof(uint32_t) &&
            sz == 16)
        return {
            _mm512_movepi32_mask(
                cast_to<__m512i>(i)),
            ~-(1U << sz)
        };
    if constexpr (integral &&
            csz == sizeof(uint16_t) &&
            sz == 32)
        return {
            _mm512_movepi16_mask(
                cast_to<__m512i>(i)),
            ~-(1U << sz)
        };
    if constexpr (integral &&
            csz == sizeof(uint8_t) &&
            sz == 64)
        return {
            _mm512_movepi8_mask(
                cast_to<__m512i>(i)),
            ~-(1U << sz)
        };
    if constexpr (!integral &&
            csz == sizeof(float) &&
            sz == 16)
        return {
            _mm512_movepi32_mask(
                cast_to<__m512i>(i)),
            ~-(1U << sz)
        };
    if constexpr (!integral &&
            csz == sizeof(double) &&
            sz == 8)
        return {
            _mm512_movepi64_mask(
                cast_to<__m512i>(i)),
            ~-(1U << sz)
        };
#else
    if constexpr (sizeof(X) == 64)
        return vec_mask_pair_synthetic(i);
#endif

#ifdef __AVX2__
    if constexpr (sizeof(X) == 32 && integral)
        return {
            _mm256_movemask_epi8(
                cast_to<__m256>(i)),
            ~-(1UL << sizeof(X))
        };
    if constexpr (sizeof(X) == 32 &&
            sizeof(C) == sizeof(float) && !integral)
        return {
            _mm256_movemask_ps(
                cast_to<__m256>(i)),
            ~-(1U << sz)
        };
    if constexpr (sizeof(X) == 32 &&
            sizeof(C) == sizeof(double) && !integral)
        return {
            _mm256_movemask_pd(
                cast_to<__m256>(i)),
            ~-(1U << sz)
        };
#else
    if (sizeof(X) == 32)
        return vec_mask_pair_synthetic(i);
#endif
#if defined(__SSE2__)
    if constexpr (sizeof(X) == 16 && integral)
        return {
            _mm_movemask_epi8(
                cast_to<__m128>(i)),
            ~-(1U << sizeof(X))
        };
    if constexpr (sizeof(X) == 16 &&
            sizeof(C) == sizeof(float) && !integral)
        return {
            _mm_movemask_ps(
                cast_to<__m128>(i)),
            ~-(1U << sz)
        };
    if constexpr (sizeof(X) == 16 &&
            sizeof(C) == sizeof(double) && !integral)
        return {
            _mm_movemask_pd(
                cast_to<__m128>(i)),
            ~-(1U << sz)
        };
#elif defined(__ARM_NEON)
    if constexpr (sizeof(X) > 16)
        return vec_mask_pair_synthetic(i);

    if constexpr (sizeof(X) == 16 &&
            sizeof(C) == sizeof(uint64_t)) {
        return vpaddq_u64(vandq_u64(
            cast_to<uint64>(i),
            int64x2{
                1U << 0,
                1U << 1
            };
        )
    }

    if constexpr (sizeof(X) == 16 &&
            sizeof(C) == sizeof(uint32_t)) {
        return vpaddq_u32(vandq_u32(
            cast_to<uint32>(i),
            int32x4{
                1U << 0,
                1U << 1,
                1U << 2,
                1U << 3
            };
        )
    }

    if constexpr (sizeof(X) == 16 &&
            sizeof(C) == sizeof(uint16_t)) {
        // Need to unpack so component is big
        // enough to contain the whole mask
        uint16x8_t n = vandq_u16(
            cast_to<uint16x8_t>(i),
            int16x8_t{
                1U << 0,
                1U << 1,
                1U << 2,
                1U << 3,
                1U << 4,
                1U << 5,
                1U << 6,
                1U << 7
            });
        return {
            vpaddq_u16(lo),
            ~-(1U << sz)
        };
    }

    if constexpr (sizeof(X) == 16 &&
            sizeof(C) == sizeof(uint8_t)) {
        // Need to unpack so component is big
        // enough to contain the whole mask
        vecu16x8 lo, hi;
        unpack(&lo, &hi, i);

        lo = vandq_u16(
            cast_to<uint16x8_t>(lo),
            int16x8_t{
                1U << 0,
                1U << 1,
                1U << 2,
                1U << 3,
                1U << 4,
                1U << 5,
                1U << 6,
                1U << 7
            });
        hi = vandq_u16(
            cast_to<uint16x8_t>(hi),
            int16x8_t{
                1U << 8,
                1U << 9,
                1U << 10,
                1U << 11,
                1U << 12,
                1U << 13,
                1U << 14,
                1U << 15
            });
        return {
            vpaddq_u16(lo) + vpaddq_u16(hi),
            ~-(1U << sz)
        };
    }
#endif
}

template<typename X>
always_inline bool vec_any_true(X const& i)
{
    auto x = vec_mask_pair(i);
    return x.first;
}

template<typename X>
always_inline bool vec_any_false(X const& i)
{
    auto x = vec_mask_pair(i);
    return x.first != x.second;
}

template<typename X>
always_inline bool vec_all_true(X const& i)
{
    auto x = vec_mask_pair(i);
    return x.first == x.second;
}

template<typename X>
always_inline bool vec_all_false(X const& i)
{
    auto x = vec_mask_pair(i);
    return !x.first;
}
