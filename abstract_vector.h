#pragma once
#include <cstdint>
#include <type_traits>

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

template<> struct next_smaller<uint16_t> { using type = uint16_t; };
template<> struct next_smaller<uint32_t> { using type = uint32_t; };
template<> struct next_smaller<uint64_t> { using type = uint64_t; };

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

template <> struct component_of<vecf64x4> { using type = double; };

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

template<typename T>
constexpr auto comp_count_v = comp_count<T>::value;

template<size_t sz>
struct vec_info_of_sz;

template<>
struct vec_info_of_sz<16> {
    using as_float = vecf32x16;
    using as_unsigned = vecu32x16;
    using as_int = veci32x16;

    static constexpr size_t sz = 16;
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
    static constexpr vecf32x16 vec_broadcast(float n)
    {
        return vecf32x16{
            n, n, n, n,
            n, n, n, n,
            n, n, n, n,
            n, n, n, n
        };
    }
    static constexpr vecu32x16 vec_broadcast(unsigned n)
    {
        return vecu32x16{
            n, n, n, n,
            n, n, n, n,
            n, n, n, n,
            n, n, n, n
        };
    }
    static constexpr veci32x16 vec_broadcast(int n)
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
    static constexpr size_t sz = 8;
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
To convert_to(From const& orig)
{
    return __builtin_convertvector(orig, To);
}

template<typename As, typename From,
    typename = to_vec_t<From>>
As cast_to(From const& rhs)
{
    return __builtin_bit_cast(As, rhs);
}

template<typename T,
    typename = to_vec_t<T>>
T min(T const& a, T const& b)
{
    return __builtin_elementwise_min(a, b);
}

template<typename T,
    typename = to_vec_t<T>>
T max(T const& a, T const& b)
{
    return __builtin_elementwise_max(a, b);
}

template<typename T,
    typename = to_vec_t<T>>
T ntload(T *address)
{
    return __builtin_nontemporal_load(address);
}

template<typename From,
    typename Tcomp = component_of_t<From>,
    size_t N = comp_count_v<From>,
    typename Tbigger = next_bigger_t<Tcomp>,
    typename Vbigger = vec_t<Tbigger, N / 2>>
void unpack(Vbigger *lo_result, Vbigger *hi_result, From const& rhs)
{
    constexpr auto shift = sizeof(Tbigger) * CHAR_BIT / 2;
    constexpr auto mask = ~-(Tcomp(1) << shift);
    *lo_result = (Vbigger)rhs & mask;
    *hi_result = (Vbigger)(rhs >> shift);
}

template<typename From,
    typename Tcomp = component_of_t<From>,
    size_t N = comp_count_v<From>,
    typename Tsmaller = next_smaller_t<Tcomp>,
    typename Vsmaller = vec_t<Tsmaller, N * 2>>
Vsmaller pack(From const& lo, From const& hi)
{
    constexpr auto shift = sizeof(Tsmaller) * CHAR_BIT / 2;
    constexpr auto mask = ~-(Tcomp(1) << shift);
    Vsmaller lo_result = (Vsmaller)(lo & mask);
    Vsmaller hi_result = (Vsmaller)((hi & mask) << shift);
    lo_result |= hi_result;
    return lo_result;
}

template<>
struct vec_info_of_sz<4> {
    using as_float = vecf32x4;
    using as_unsigned = vecu32x4;
    using as_int = veci32x4;
    static constexpr size_t sz = 4;
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

__attribute__((__always_inline__))
static inline vecf32x16 vec_broadcast_16(float n) {
    return vecf32x16{
        n, n, n, n,
        n, n, n, n,
        n, n, n, n,
        n, n, n, n
    };
}

__attribute__((__always_inline__))
static inline vecu32x16 vec_broadcast_16(uint32_t n) {
    return vecu32x16{
        n, n, n, n,
        n, n, n, n,
        n, n, n, n,
        n, n, n, n
    };
}

__attribute__((__always_inline__))
static inline veci32x16 vec_broadcast_16(int32_t n) {
    return veci32x16{
        n, n, n, n,
        n, n, n, n,
        n, n, n, n,
        n, n, n, n
    };
}

__attribute__((__always_inline__))
static inline vecf32x8 vec_broadcast_f8(float n) {
    return vecf32x8{
        n, n, n, n,
        n, n, n, n
    };
}

__attribute__((__always_inline__))
static inline vecu32x8 vec_broadcast_u8(uint32_t n) {
    return vecu32x8{
        n, n, n, n,
        n, n, n, n
    };
}

__attribute__((__always_inline__))
static inline veci32x8 vec_broadcast_i8(int32_t n) {
    return veci32x8{
        n, n, n, n,
        n, n, n, n
    };
}

__attribute__((__always_inline__))
static inline vecf32x4 vec_broadcast_f4(float n) {
    return vecf32x4{
        n, n, n, n
    };
}

__attribute__((__always_inline__))
static inline vecu32x4 vec_broadcast_u4(uint32_t n) {
    return vecu32x4{
        n, n, n, n
    };
}

static inline veci32x4 vec_broadcast_i4(int32_t n) {
    return veci32x4{
        n, n, n, n
    };
}

__attribute__((__always_inline__))
static inline vecu32x8 vec_lo(vecu32x16 whole)
{
    return vecu32x8{
        whole[0], whole[1], whole[2], whole[3],
        whole[4], whole[5], whole[6], whole[7]
    };
}

__attribute__((__always_inline__))
static inline vecf32x8 vec_lo(vecf32x16 whole)
{
    return vecf32x8{
        whole[0], whole[1], whole[2], whole[3],
        whole[4], whole[5], whole[6], whole[7]
    };
}

__attribute__((__always_inline__))
static inline vecf32x4 vec_lo(vecf32x8 whole)
{
    return vecf32x4{
        whole[0], whole[1], whole[2], whole[3]
    };
}

__attribute__((__always_inline__))
static inline vecu32x4 vec_lo(vecu32x8 whole)
{
    return vecu32x4{
        whole[0], whole[1], whole[2], whole[3]
    };
}

__attribute__((__always_inline__))
static inline vecu32x8 vec_combine(vecu32x4 lo, vecu32x4 hi)
{
    return vecu32x8{
        lo[0], lo[1], lo[2], lo[3],
        hi[0], hi[1], hi[2], hi[3]
    };
}

__attribute__((__always_inline__))
static inline vecu32x16 vec_combine(vecu32x8 lo, vecu32x8 hi)
{
    return vecu32x16{
        lo[0], lo[1], lo[2], lo[3], lo[4], lo[5], lo[6], lo[7],
        hi[0], hi[1], hi[2], hi[3], hi[4], hi[5], hi[6], hi[7]
    };
}

__attribute__((__always_inline__))
static inline vecf32x16 vec_combine(vecf32x8 lo, vecf32x8 hi)
{
    return vecf32x16{
        lo[0], lo[1], lo[2], lo[3], lo[4], lo[5], lo[6], lo[7],
        hi[0], hi[1], hi[2], hi[3], hi[4], hi[5], hi[6], hi[7]
    };
}

__attribute__((__always_inline__))
static inline vecu32x8 vec_hi(vecu32x16 whole)
{
    return vecu32x8{
        whole[ 8], whole[ 9], whole[10], whole[11],
        whole[12], whole[13], whole[14], whole[15]
    };
}

__attribute__((__always_inline__))
static inline vecf32x8 vec_hi(vecf32x16 whole)
{
    return vecf32x8{
        whole[ 8], whole[ 9], whole[10], whole[11],
        whole[12], whole[13], whole[14], whole[15]
    };
}

__attribute__((__always_inline__))
static inline vecu32x4 vec_hi(vecu32x8 whole)
{
    return vecu32x4{
        whole[ 4], whole[ 5], whole[ 6], whole[ 7]
    };
}

__attribute__((__always_inline__))
static inline veci32x4 vec_hi(veci32x8 whole)
{
    return veci32x4{
        whole[ 4], whole[ 5], whole[ 6], whole[ 7]
    };
}

__attribute__((__always_inline__))
static inline vecf32x4 vec_hi(vecf32x8 whole)
{
    return vecf32x4{
        whole[ 4], whole[ 5], whole[ 6], whole[ 7]
    };
}

__attribute__((__always_inline__))
static inline vecu32x4 vec_blend(
    vecu32x4 existing, vecu32x4 updated, vecu32x4 mask)
{
    return (updated & mask) | (existing & ~mask);
}

__attribute__((__always_inline__))
static inline vecu32x16 vec_blend(
    vecu32x16 existing, vecu32x16 updated, vecu32x16 mask)
{
#if defined(__AVX512F__)
    return (vecu32x16)_mm512_blendv_epi8(
        (__m512i)existing, (__m512i)updated, (__m512i)mask);
#elif defined(__AVX2__)
    vecu32x8 lo = (vecu32x8)_mm256_blendv_epi8(
        (__m256i)vec_lo(existing),
        (__m256i)vec_lo(updated),
        (__m256i)vec_lo(mask));
    vecu32x8 hi = (vecu32x8)_mm256_blendv_epi8(
        (__m256i)vec_hi(existing),
        (__m256i)vec_hi(updated),
        (__m256i)vec_hi(mask));

    return vec_combine(lo, hi);
#else
    return (existing & ~mask) | (updated & mask);
#endif
}

__attribute__((__always_inline__))
static inline vecu32x8 vec_blend(
    vecu32x8 existing, vecu32x8 updated, vecu32x8 mask)
{
#if defined(__AVX2__)
    return (vecu32x8)_mm256_blendv_epi8(
        (__m256i)existing, (__m256i)updated, (__m256i)mask);
#else
    return (existing & ~mask) | (updated & mask);
#endif
}

__attribute__((__always_inline__))
static inline vecf32x8 vec_blend(
    vecf32x8 existing, vecf32x8 updated, vecu32x8 mask)
{
#if defined(__AVX2__)
    return (vecf32x8)_mm256_blendv_ps(
        (__m256)existing, (__m256)updated, (__m256)mask);
#else
    return (vecf32x8)(((vecu32x8)existing & ~mask) |
        ((vecu32x8)updated & mask));
#endif
}

__attribute__((__always_inline__))
static inline vecf32x16 vec_blend(
    vecf32x16 existing, vecf32x16 updated, vecu32x16 mask)
{
#if defined(__AVX2__)
    vecf32x8 lo = (vecf32x8)_mm256_blendv_ps(
        (__m256)vec_lo(existing),
        (__m256)vec_lo(updated),
        (__m256)vec_lo(mask));
    vecf32x8 hi = (vecf32x8)_mm256_blendv_ps(
        (__m256)vec_hi(existing),
        (__m256)vec_hi(updated),
        (__m256)vec_hi(mask));
    return vec_combine(lo, hi);
#else
    return (vecf32x16)(((vecu32x16)existing & ~mask) |
        ((vecu32x16)updated & mask));
#endif
}

__attribute__((__always_inline__))
static inline vecf32x4 vec_blend(
    vecf32x4 existing, vecf32x4 updated, vecu32x4 mask)
{
#if defined(__AVX2__)
    return (vecf32x4)_mm_blendv_ps(
        (__m128)existing, (__m128)updated, (__m128)mask);
#else
    return (vecf32x4)(((vecu32x4)existing & ~mask) |
        ((vecu32x4)updated & mask));
#endif
}

// It sucks, on AMD anyway. A bunch of scalar loads and inserts beat it, lol.
#define GATHER_IS_GOOD 1

// 128-bit gather
__attribute__((__always_inline__))
static inline vecu32x4 vec_gather(uint32_t const *buffer,
    vecu32x4 indices, vecu32x4 background, vecu32x4 mask)
{
#if GATHER_IS_GOOD && defined(__AVX2__)
    vecu32x4 result = (vecu32x4)_mm_mask_i32gather_epi32(
        (__m128i)background, (int const*)buffer,
        (__m128i)indices, (__m128i)mask, sizeof(uint32_t));
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

// 256-bit gather
__attribute__((__always_inline__))
static inline vecu32x8 vec_gather(uint32_t const *buffer,
    vecu32x8 indices, vecu32x8 background, vecu32x8 mask)
{
#if GATHER_IS_GOOD && defined(__AVX2__)
    vecu32x8 result = (vecu32x8)_mm256_mask_i32gather_epi32(
        (__m256i)background, (int const*)buffer,
        (__m256i)indices, (__m256i)mask, sizeof(uint32_t));
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
__attribute__((__always_inline__))
static inline vecu32x16 vec_gather(uint32_t const *buffer,
    vecu32x16 indices, vecu32x16 background, vecu32x16 mask)
{
#if GATHER_IS_GOOD && defined(__AVX512F__)
    __mmask8 compact_mask = _mm512_test_epi32_mask(
        _mm512_set1_epi32(0x80000000), mask);
    vecu32x16 result = (vecu32x16)_mm512_mask_i32gather_epi32(
        background,
        (int const *)texture->pixels,
        (__m512i)indices,
        compact_mask,
        sizeof(uint32_t));
#elif GATHER_IS_GOOD && defined(__AVX2__)
    // Pair of 256 bit operations
    vecu32x8 lo = (vecu32x8)_mm256_mask_i32gather_epi32(
        (__m256i)vec_lo(background), (int const*)buffer,
        (__m256i)vec_lo(indices), (__m256i)vec_lo(mask), sizeof(uint32_t));
    vecu32x8 hi = (vecu32x8)_mm256_mask_i32gather_epi32(
        (__m256i)vec_hi(background), (int const*)buffer,
        (__m256i)vec_hi(indices), (__m256i)vec_hi(mask), sizeof(uint32_t));
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

__attribute__((__always_inline__))
static inline int vec_movemask(vecu32x8 i)
{
#if defined(__AVX2__)
    return _mm256_movemask_epi8((__m256i)i);
#elif defined(__SSE2__)
    int lo = _mm_movemask_epi8((__m128i)vec_lo(i));
    int hi = _mm_movemask_epi8((__m128i)vec_hi(i));
    return lo | (hi << 16);
#else
    vecu32x8 bits{
        0xF,
        0xF0,
        0xF00,
        0xF000,
        0xF0000,
        0xF00000,
        0xF000000,
        0xF0000000
    };
    vecu32x8 msb = vec_broadcast_u8(0x80000000U);
    bits = bits & msb;

    return (bits[0] | bits[1]) | (bits[2] | bits[3]) |
        (bits[4] | bits[5]) | (bits[6] | bits[7]);
#endif
}

__attribute__((__always_inline__))
static inline int vec_movemask(vecf32x8 i)
{
#if defined(__AVX2__)
    return _mm256_movemask_ps(i);
#elif defined(__SSE2__)
    int lo = _mm_movemask_ps((__m128)vec_lo(i));
    int hi = _mm_movemask_ps((__m128)vec_hi(i));
    return lo | (hi << 16);
#else
    vecu32x8 bits{
        0xF,
        0xF0,
        0xF00,
        0xF000,
        0xF0000,
        0xF00000,
        0xF000000,
        0xF0000000
    };
    vecu32x8 msb = vec_broadcast_u8(0x80000000U);
    bits = bits & msb;

    return (bits[0] | bits[1]) | (bits[2] | bits[3]) |
        (bits[4] | bits[5]) | (bits[6] | bits[7]);
#endif
}

__attribute__((__always_inline__))
static inline int vec_movemask(veci32x8 i)
{
    return vec_movemask((vecu32x8)i);
}

__attribute__((__always_inline__))
static inline int vec_movemask(vecu32x4 i)
{
#if defined(__SSE2__)
    return _mm_movemask_epi8((__m128i)i);
#else
    vecu32x4 bits{
        0xF,
        0xF0,
        0xF00,
        0xF000
    };
    vecu32x4 msb = vec_broadcast_u4(0x80000000U);
    bits = bits & msb;

    return (bits[0] | bits[1]) | (bits[2] | bits[3]);
#endif
}

__attribute__((__always_inline__))
static inline int vec_movemask(veci32x4 i)
{
    return vec_movemask((vecu32x4)i);
}

template<typename T>
struct vec_larger_type;

template<>
struct vec_larger_type<vecu8x16> { using type = vecu16x8; };
