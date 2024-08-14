#pragma once

#include <array>
#include <utility>
#include <cstdint>
#include <climits>
#include <limits>
#include <type_traits>

#ifdef __x86_64__
#include <immintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>
#include <xmmintrin.h>
#endif

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#ifndef always_inline
#define always_inline static inline __attribute__((__always_inline__))
#define always_inline_method inline __attribute__((__always_inline__))
#endif

#include <initializer_list>

template<typename C, size_t N>
struct intrin_type {};

#if defined(__ARM_NEON)
template<> struct intrin_type<float, 4> { using type = floatx4_t; };
template<> struct intrin_type<double, 2> { using type = doublex2_t; };
template<> struct intrin_type<uint8_t, 16> { using type = uint8x16_t; };
template<> struct intrin_type<uint16_t, 8> { using type = uint16x8_t; };
template<> struct intrin_type<uint32_t, 4> { using type = uint32x4_t; };
template<> struct intrin_type<uint64_t, 2> { using type = uint64x2_t; };
template<> struct intrin_type<int8_t, 16> { using type = int8x16_t; };
template<> struct intrin_type<int16_t, 8> { using type = int16x8_t; };
template<> struct intrin_type<int32_t, 4> { using type = int32x4_t; };
template<> struct intrin_type<int64_t, 2> { using type = int64x2_t; };
#elif defined(__SSE2__)
template<> struct intrin_type<float, 4> { using type = __m128; };
template<> struct intrin_type<double, 2> { using type = __m128d; };
template<> struct intrin_type<uint8_t, 16> { using type = __m128i; };
template<> struct intrin_type<uint16_t, 8> { using type = __m128i; };
template<> struct intrin_type<uint32_t, 4> { using type = __m128i; };
template<> struct intrin_type<uint64_t, 2> { using type = __m128i; };
template<> struct intrin_type<int8_t, 16> { using type = __m128i; };
template<> struct intrin_type<int16_t, 8> { using type = __m128i; };
template<> struct intrin_type<int32_t, 4> { using type = __m128i; };
template<> struct intrin_type<int64_t, 2> { using type = __m128i; };
#endif

#if defined(__AVX2__)
template<> struct intrin_type<float, 8> { using type = __m256; };
template<> struct intrin_type<double, 4> { using type = __m256d; };
template<> struct intrin_type<uint8_t, 32> { using type = __m256i; };
template<> struct intrin_type<uint16_t, 16> { using type = __m256i; };
template<> struct intrin_type<uint32_t, 8> { using type = __m256i; };
template<> struct intrin_type<uint64_t, 4> { using type = __m256i; };
template<> struct intrin_type<int8_t, 32> { using type = __m256i; };
template<> struct intrin_type<int16_t, 16> { using type = __m256i; };
template<> struct intrin_type<int32_t, 8> { using type = __m256i; };
template<> struct intrin_type<int64_t, 4> { using type = __m256i; };
#else
template<> struct intrin_type<float, 8> { using type = std::array<intrin_type<float, 4>::type, 2>; };
template<> struct intrin_type<double, 4> { using type = std::array<intrin_type<double, 2>::type, 2>; };
template<> struct intrin_type<uint8_t, 32> { using type = std::array<intrin_type<uint8_t, 16>::type, 2>; };
template<> struct intrin_type<uint16_t, 16> { using type = std::array<intrin_type<uint16_t, 8>::type, 2>; };
template<> struct intrin_type<uint32_t, 8> { using type = std::array<intrin_type<uint32_t, 4>::type, 2>; };
template<> struct intrin_type<uint64_t, 4> { using type = std::array<intrin_type<uint64_t, 2>::type, 2>; };
template<> struct intrin_type<int8_t, 32> { using type = std::array<intrin_type<int8_t, 16>::type, 2>; };
template<> struct intrin_type<int16_t, 16> { using type = std::array<intrin_type<int16_t, 8>::type, 2>; };
template<> struct intrin_type<int32_t, 8> { using type = std::array<intrin_type<int32_t, 4>::type, 2>; };
template<> struct intrin_type<int64_t, 4> { using type = std::array<intrin_type<int64_t, 2>::type, 2>; };
#endif

#if defined(__AVX512F__)
template<> struct intrin_type<float, 16> { using type = __m512; };
template<> struct intrin_type<double, 8> { using type = __m512d; };
template<> struct intrin_type<uint8_t, 64> { using type = __m512i; };
template<> struct intrin_type<uint16_t, 32> { using type = __m512i; };
template<> struct intrin_type<uint32_t, 16> { using type = __m512i; };
template<> struct intrin_type<uint64_t, 8> { using type = __m512i; };
template<> struct intrin_type<int8_t, 64> { using type = __m512i; };
template<> struct intrin_type<int16_t, 32> { using type = __m512i; };
template<> struct intrin_type<int32_t, 16> { using type = __m512i; };
template<> struct intrin_type<int64_t, 8> { using type = __m512i; };
#else
template<> struct intrin_type<float, 16> { using type = std::array<intrin_type<float, 8>::type, 2>; };
template<> struct intrin_type<double, 8> { using type = std::array<intrin_type<double, 4>::type, 2>; };
template<> struct intrin_type<uint8_t, 64> { using type = std::array<intrin_type<uint8_t, 32>::type, 2>; };
template<> struct intrin_type<uint16_t, 32> { using type = std::array<intrin_type<uint16_t, 16>::type, 2>; };
template<> struct intrin_type<uint32_t, 16> { using type = std::array<intrin_type<uint32_t, 8>::type, 2>; };
template<> struct intrin_type<uint64_t, 8> { using type = std::array<intrin_type<uint64_t, 4>::type, 2>; };
template<> struct intrin_type<int8_t, 64> { using type = std::array<intrin_type<int8_t, 32>::type, 2>; };
template<> struct intrin_type<int16_t, 32> { using type = std::array<intrin_type<int16_t, 16>::type, 2>; };
template<> struct intrin_type<int32_t, 16> { using type = std::array<intrin_type<int32_t, 8>::type, 2>; };
template<> struct intrin_type<int64_t, 8> { using type = std::array<intrin_type<int64_t, 4>::type, 2>; };
#endif

template<typename C, size_t N>
using intrin_type_t = typename intrin_type<C, N>::type;

template<typename C, size_t N>
class clsvec;

//
// 8-bit

// 128-bit int8_t init
always_inline void init(clsvec<int8_t, 16> *lhs,
    int8_t v00, int8_t v01, int8_t v02, int8_t v03,
    int8_t v04, int8_t v05, int8_t v06, int8_t v07,
    int8_t v08, int8_t v09, int8_t v10, int8_t v11,
    int8_t v12, int8_t v13, int8_t v14, int8_t v15);

// 128-bit uint8_t
always_inline void init(clsvec<uint8_t, 16> *lhs,
    uint8_t v00, uint8_t v01, uint8_t v02, uint8_t v03,
    uint8_t v04, uint8_t v05, uint8_t v06, uint8_t v07,
    uint8_t v08, uint8_t v09, uint8_t v10, uint8_t v11,
    uint8_t v12, uint8_t v13, uint8_t v14, uint8_t v15);

// 256-bit int8_t
always_inline void init(clsvec<int8_t, 32> *lhs,
    int8_t v00, int8_t v01, int8_t v02, int8_t v03,
    int8_t v04, int8_t v05, int8_t v06, int8_t v07,
    int8_t v08, int8_t v09, int8_t v10, int8_t v11,
    int8_t v12, int8_t v13, int8_t v14, int8_t v15,
    int8_t v16, int8_t v17, int8_t v18, int8_t v19,
    int8_t v20, int8_t v21, int8_t v22, int8_t v23,
    int8_t v24, int8_t v25, int8_t v26, int8_t v27,
    int8_t v28, int8_t v29, int8_t v30, int8_t v31);

// 256-bit uint8_t
always_inline void init(clsvec<uint8_t, 32> *lhs,
   uint8_t v00, uint8_t v01, uint8_t v02, uint8_t v03,
   uint8_t v04, uint8_t v05, uint8_t v06, uint8_t v07,
   uint8_t v08, uint8_t v09, uint8_t v10, uint8_t v11,
   uint8_t v12, uint8_t v13, uint8_t v14, uint8_t v15,
   uint8_t v16, uint8_t v17, uint8_t v18, uint8_t v19,
   uint8_t v20, uint8_t v21, uint8_t v22, uint8_t v23,
   uint8_t v24, uint8_t v25, uint8_t v26, uint8_t v27,
   uint8_t v28, uint8_t v29, uint8_t v30, uint8_t v31);

#if defined(__AVX512F__)
// 512-bit int8_t
always_inline void init(clsvec<int8_t, 64> *lhs,
   int8_t v00, int8_t v01, int8_t v02, int8_t v03,
   int8_t v04, int8_t v05, int8_t v06, int8_t v07,
   int8_t v08, int8_t v09, int8_t v10, int8_t v11,
   int8_t v12, int8_t v13, int8_t v14, int8_t v15,
   int8_t v16, int8_t v17, int8_t v18, int8_t v19,
   int8_t v20, int8_t v21, int8_t v22, int8_t v23,
   int8_t v24, int8_t v25, int8_t v26, int8_t v27,
   int8_t v28, int8_t v29, int8_t v30, int8_t v31,
   int8_t v32, int8_t v33, int8_t v34, int8_t v35,
   int8_t v36, int8_t v37, int8_t v38, int8_t v39,
   int8_t v40, int8_t v41, int8_t v42, int8_t v43,
   int8_t v44, int8_t v45, int8_t v46, int8_t v47,
   int8_t v48, int8_t v49, int8_t v50, int8_t v51,
   int8_t v52, int8_t v53, int8_t v54, int8_t v55,
   int8_t v56, int8_t v57, int8_t v58, int8_t v59,
   int8_t v60, int8_t v61, int8_t v62, int8_t v63);

// 512-bit uint8_t
always_inline void init(clsvec<uint8_t, 64> *lhs,
   uint8_t v00, uint8_t v01, uint8_t v02, uint8_t v03,
   uint8_t v04, uint8_t v05, uint8_t v06, uint8_t v07,
   uint8_t v08, uint8_t v09, uint8_t v10, uint8_t v11,
   uint8_t v12, uint8_t v13, uint8_t v14, uint8_t v15,
   uint8_t v16, uint8_t v17, uint8_t v18, uint8_t v19,
   uint8_t v20, uint8_t v21, uint8_t v22, uint8_t v23,
   uint8_t v24, uint8_t v25, uint8_t v26, uint8_t v27,
   uint8_t v28, uint8_t v29, uint8_t v30, uint8_t v31,
   uint8_t v32, uint8_t v33, uint8_t v34, uint8_t v35,
   uint8_t v36, uint8_t v37, uint8_t v38, uint8_t v39,
   uint8_t v40, uint8_t v41, uint8_t v42, uint8_t v43,
   uint8_t v44, uint8_t v45, uint8_t v46, uint8_t v47,
   uint8_t v48, uint8_t v49, uint8_t v50, uint8_t v51,
   uint8_t v52, uint8_t v53, uint8_t v54, uint8_t v55,
   uint8_t v56, uint8_t v57, uint8_t v58, uint8_t v59,
   uint8_t v60, uint8_t v61, uint8_t v62, uint8_t v63);
#endif

//
// 16-bit

// 128-bit int16_t
always_inline void init(clsvec<int16_t, 8> *lhs,
    int16_t v00, int16_t v01, int16_t v02, int16_t v03,
    int16_t v04, int16_t v05, int16_t v06, int16_t v07);

// 128-bit uint16_t
always_inline void init(clsvec<uint16_t, 8> *lhs,
    uint16_t v00, uint16_t v01, uint16_t v02, uint16_t v03,
    uint16_t v04, uint16_t v05, uint16_t v06, uint16_t v07);

// 256-bit int16_t
always_inline void init(clsvec<int16_t, 16> *lhs,
    int16_t v00, int16_t v01, int16_t v02, int16_t v03,
    int16_t v04, int16_t v05, int16_t v06, int16_t v07,
    int16_t v08, int16_t v09, int16_t v10, int16_t v11,
    int16_t v12, int16_t v13, int16_t v14, int16_t v15);

// 256-bit uint16_t
always_inline void init(clsvec<uint16_t, 16> *lhs,
   uint16_t v00, uint16_t v01, uint16_t v02, uint16_t v03,
   uint16_t v04, uint16_t v05, uint16_t v06, uint16_t v07,
   uint16_t v08, uint16_t v09, uint16_t v10, uint16_t v11,
   uint16_t v12, uint16_t v13, uint16_t v14, uint16_t v15);

// 512-bit int16_t
always_inline void init(clsvec<int16_t, 32> *lhs,
   int16_t v00, int16_t v01, int16_t v02, int16_t v03,
   int16_t v04, int16_t v05, int16_t v06, int16_t v07,
   int16_t v08, int16_t v09, int16_t v10, int16_t v11,
   int16_t v12, int16_t v13, int16_t v14, int16_t v15,
   int16_t v16, int16_t v17, int16_t v18, int16_t v19,
   int16_t v20, int16_t v21, int16_t v22, int16_t v23,
   int16_t v24, int16_t v25, int16_t v26, int16_t v27,
   int16_t v28, int16_t v29, int16_t v30, int16_t v31);

// 512-bit uint16_t
always_inline void init(clsvec<uint16_t, 32> *lhs,
   uint16_t v00, uint16_t v01, uint16_t v02, uint16_t v03,
   uint16_t v04, uint16_t v05, uint16_t v06, uint16_t v07,
   uint16_t v08, uint16_t v09, uint16_t v10, uint16_t v11,
   uint16_t v12, uint16_t v13, uint16_t v14, uint16_t v15,
   uint16_t v16, uint16_t v17, uint16_t v18, uint16_t v19,
   uint16_t v20, uint16_t v21, uint16_t v22, uint16_t v23,
   uint16_t v24, uint16_t v25, uint16_t v26, uint16_t v27,
   uint16_t v28, uint16_t v29, uint16_t v30, uint16_t v31,
   uint16_t v32, uint16_t v33, uint16_t v34, uint16_t v35,
   uint16_t v36, uint16_t v37, uint16_t v38, uint16_t v39,
   uint16_t v40, uint16_t v41, uint16_t v42, uint16_t v43,
   uint16_t v44, uint16_t v45, uint16_t v46, uint16_t v47,
   uint16_t v48, uint16_t v49, uint16_t v50, uint16_t v51,
   uint16_t v52, uint16_t v53, uint16_t v54, uint16_t v55,
   uint16_t v56, uint16_t v57, uint16_t v58, uint16_t v59,
   uint16_t v60, uint16_t v61, uint16_t v62, uint16_t v63);

//
// 32-bit

// 128-bit int32_t
always_inline void init(clsvec<int32_t, 4> *lhs,
    int32_t v00, int32_t v01, int32_t v02, int32_t v03);

// 128-bit uint32_t
always_inline void init(clsvec<uint32_t, 4> *lhs,
    uint32_t v00, uint32_t v01, uint32_t v02, uint32_t v03);

#if defined(__AVX2__)
// 256-bit int32_t
always_inline void init(clsvec<int32_t, 8> *lhs,
    int32_t v00, int32_t v01, int32_t v02, int32_t v03,
    int32_t v04, int32_t v05, int32_t v06, int32_t v07);

// 256-bit uint32_t
always_inline void init(clsvec<uint32_t, 8> *lhs,
   uint32_t v00, uint32_t v01, uint32_t v02, uint32_t v03,
   uint32_t v04, uint32_t v05, uint32_t v06, uint32_t v07);
#endif

#if defined(__AVX512F__)
// 512-bit int32_t
always_inline void init(clsvec<int32_t, 16> *lhs,
   int32_t v00, int32_t v01, int32_t v02, int32_t v03,
   int32_t v04, int32_t v05, int32_t v06, int32_t v07,
   int32_t v08, int32_t v09, int32_t v10, int32_t v11,
   int32_t v12, int32_t v13, int32_t v14, int32_t v15);

// 512-bit uint32_t
always_inline void init(clsvec<uint32_t, 16> *lhs,
   uint32_t v00, uint32_t v01, uint32_t v02, uint32_t v03,
   uint32_t v04, uint32_t v05, uint32_t v06, uint32_t v07,
   uint32_t v08, uint32_t v09, uint32_t v10, uint32_t v11,
   uint32_t v12, uint32_t v13, uint32_t v14, uint32_t v15);
#endif

// 128-bit float
always_inline void init(clsvec<float, 4> *lhs,
   float v00, float v01, float v02, float v03);

#if defined(__AVX2__)
// 256-bit float
always_inline void init(clsvec<float, 8> *lhs,
   float v00, float v01, float v02, float v03,
   float v04, float v05, float v06, float v07);
#endif

#if defined(__AVX512F__)
// 512-bit float
always_inline void init(clsvec<float, 16> *lhs,
   float v00, float v01, float v02, float v03,
   float v04, float v05, float v06, float v07,
   float v08, float v09, float v10, float v11,
   float v12, float v13, float v14, float v15);
#endif

//
// 64-bit

// 128-bit int64_t
always_inline void init(clsvec<int64_t, 2> *lhs,
    int64_t v00, int64_t v01);

// 128-bit uint64_t
always_inline void init(clsvec<uint64_t, 2> *lhs,
    uint64_t v00, uint64_t v01);

// 256-bit int64_t
always_inline void init(clsvec<int64_t, 4> *lhs,
    int64_t v00, int64_t v01, int64_t v02, int64_t v03);

// 256-bit uint64_t
always_inline void init(clsvec<uint64_t, 4> *lhs,
   uint64_t v00, uint64_t v01, uint64_t v02, uint64_t v03);

// 512-bit int64_t
always_inline void init(clsvec<int64_t, 8> *lhs,
   int64_t v00, int64_t v01, int64_t v02, int64_t v03,
   int64_t v04, int64_t v05, int64_t v06, int64_t v07);

// 512-bit uint64_t
always_inline void init(clsvec<uint64_t, 8> *lhs,
   uint64_t v00, uint64_t v01, uint64_t v02, uint64_t v03,
   uint64_t v04, uint64_t v05, uint64_t v06, uint64_t v07);

// 128-bit double
always_inline void init(clsvec<double, 2> *lhs,
   double v00, double v01);

// 256-bit double
always_inline void init(clsvec<double, 4> *lhs,
   double v00, double v01, double v02, double v03);

// 512-bit double
always_inline void init(clsvec<double, 8> *lhs,
   double v00, double v01, double v02, double v03,
   double v04, double v05, double v06, double v07);

//
// Broadcasts

always_inline void init(clsvec<int8_t, 16> *lhs, int8_t scalar);
always_inline void init(clsvec<int8_t, 32> *lhs, int8_t scalar);
always_inline void init(clsvec<int8_t, 64> *lhs, int8_t scalar);

always_inline void init(clsvec<uint8_t, 16> *lhs, uint8_t scalar);
always_inline void init(clsvec<uint8_t, 32> *lhs, uint8_t scalar);
always_inline void init(clsvec<uint8_t, 64> *lhs, uint8_t scalar);

always_inline void init(clsvec<int16_t, 8> *lhs, int16_t scalar);
always_inline void init(clsvec<int16_t, 16> *lhs, int16_t scalar);
always_inline void init(clsvec<int16_t, 32> *lhs, int16_t scalar);

always_inline void init(clsvec<uint16_t, 8> *lhs, uint16_t scalar);
always_inline void init(clsvec<uint16_t, 16> *lhs, uint16_t scalar);
always_inline void init(clsvec<uint16_t, 32> *lhs, uint16_t scalar);

always_inline void init(clsvec<int32_t, 4> *lhs, int32_t scalar);
always_inline void init(clsvec<int32_t, 8> *lhs, int32_t scalar);
always_inline void init(clsvec<int32_t, 16> *lhs, int32_t scalar);

always_inline void init(clsvec<uint32_t, 4> *lhs, uint32_t scalar);
always_inline void init(clsvec<uint32_t, 8> *lhs, uint32_t scalar);
always_inline void init(clsvec<uint32_t, 16> *lhs, uint32_t scalar);

always_inline void init(clsvec<float, 4> *lhs, float scalar);
always_inline void init(clsvec<float, 8> *lhs, float scalar);
always_inline void init(clsvec<float, 16> *lhs, float scalar);

always_inline void init(clsvec<int64_t, 2> *lhs, int64_t scalar);
always_inline void init(clsvec<int64_t, 4> *lhs, int64_t scalar);
always_inline void init(clsvec<int64_t, 8> *lhs, int64_t scalar);

always_inline void init(clsvec<uint64_t, 2> *lhs, uint64_t scalar);
always_inline void init(clsvec<uint64_t, 4> *lhs, uint64_t scalar);
always_inline void init(clsvec<uint64_t, 8> *lhs, uint64_t scalar);

always_inline void init(clsvec<double, 2> *lhs, double scalar);
always_inline void init(clsvec<double, 4> *lhs, double scalar);
always_inline void init(clsvec<double, 8> *lhs, double scalar);

// 128-bit int8_t add
always_inline void add(clsvec<int8_t, 16> const *lhs,
    intrin_type_t<int8_t, 16> const& rhs,
    intrin_type_t<int8_t, 16>& res);

// 256-bit int8_t
always_inline void add(clsvec<int8_t, 32> const *lhs,
    intrin_type_t<int8_t, 32> const& rhs,
    intrin_type_t<int8_t, 32>& res);

// 512-bit int8_t
always_inline void add(clsvec<int8_t, 64> const *lhs,
    intrin_type_t<int8_t, 64> const& rhs,
    intrin_type_t<int8_t, 64>& res);

// 128-bit uint8_t
always_inline void add(clsvec<uint8_t, 16> const *lhs,
    intrin_type_t<uint8_t, 16> const& rhs,
    intrin_type_t<uint8_t, 16>& res);

// 256-bit uint8_t
always_inline void add(clsvec<uint8_t, 32> const *lhs,
    intrin_type_t<uint8_t, 32> const& rhs,
    intrin_type_t<uint8_t, 32>& res);

// 512-bit uint8_t
always_inline void add(clsvec<uint8_t, 64> const *lhs,
    intrin_type_t<uint8_t, 64> const& rhs,
    intrin_type_t<uint8_t, 64>& res);

// 128-bit int16_t
always_inline void add(clsvec<int16_t, 8> const *lhs,
    intrin_type_t<int16_t, 8> const& rhs,
    intrin_type_t<int16_t, 8>& res);

// 256-bit int16_t
always_inline void add(clsvec<int16_t, 16> const *lhs,
    intrin_type_t<int16_t, 16> const& rhs,
    intrin_type_t<int16_t, 16>& res);

// 512-bit int16_t
always_inline void add(clsvec<int16_t, 32> const *lhs,
    intrin_type_t<int16_t, 32> const& rhs,
    intrin_type_t<int16_t, 32>& res);

// 128-bit int16_t
always_inline void add(clsvec<uint16_t, 8> const *lhs,
    intrin_type_t<uint16_t, 8> const& rhs,
    intrin_type_t<uint16_t, 8>& res);

// 256-bit int16_t
always_inline void add(clsvec<uint16_t, 16> const *lhs,
    intrin_type_t<uint16_t, 16> const& rhs,
    intrin_type_t<uint16_t, 16>& res);

// 512-bit uint16_t
always_inline void add(clsvec<uint16_t, 32> const *lhs,
    intrin_type_t<uint16_t, 32> const& rhs,
    intrin_type_t<uint16_t, 32>& res);

// 128-bit int32_t
always_inline void add(clsvec<int32_t, 4> const *lhs,
    intrin_type_t<int32_t, 4> const& rhs,
    intrin_type_t<int32_t, 4>& res);

// 256-bit int32_t
always_inline void add(clsvec<int32_t, 8> const *lhs,
    intrin_type_t<int32_t, 8> const& rhs,
    intrin_type_t<int32_t, 8>& res);

// 512-bit int32_t
always_inline void add(clsvec<int32_t, 16> const *lhs,
    intrin_type_t<int32_t, 16> const& rhs,
    intrin_type_t<int32_t, 16>& res);

// 128-bit uint32_t
always_inline void add(clsvec<uint32_t, 4> const *lhs,
    intrin_type_t<uint32_t, 4> const& rhs,
    intrin_type_t<uint32_t, 4>& res);

// 256-bit uint32_t
always_inline void add(clsvec<uint32_t, 8> const *lhs,
    intrin_type_t<uint32_t, 8> const& rhs,
    intrin_type_t<uint32_t, 8>& res);

// 512-bit uint32_t
always_inline void add(clsvec<uint32_t, 16> const *lhs,
    intrin_type_t<uint32_t, 16> const& rhs,
    intrin_type_t<uint32_t, 16>& res);

// 128-bit float
always_inline void add(clsvec<float, 4> const *lhs,
    intrin_type_t<float, 4> const& rhs,
    intrin_type_t<float, 4>& res);

// 256-bit float
always_inline void add(clsvec<float, 8> const *lhs,
    intrin_type_t<float, 8> const& rhs,
    intrin_type_t<float, 8>& res);

// 512-bit float
always_inline void add(clsvec<float, 16> const *lhs,
    intrin_type_t<float, 16> const& rhs,
    intrin_type_t<float, 16>& res);

// 128-bit int64_t
always_inline void add(clsvec<int64_t, 2> const *lhs,
    intrin_type_t<int64_t, 2> const& rhs,
    intrin_type_t<int64_t, 2>& res);

// 256-bit int64_t
always_inline void add(clsvec<int64_t, 4> const *lhs,
    intrin_type_t<int64_t, 4> const& rhs,
    intrin_type_t<int64_t, 4>& res);

// 512-bit int64_t
always_inline void add(clsvec<int64_t, 8> const *lhs,
    intrin_type_t<int64_t, 8> const& rhs,
    intrin_type_t<int64_t, 8>& res);

// 128-bit uint64_t
always_inline void add(clsvec<uint64_t, 2> const *lhs,
    intrin_type_t<uint64_t, 2> const& rhs,
    intrin_type_t<uint64_t, 2>& res);

// 256-bit uint64_t
always_inline void add(clsvec<uint64_t, 4> const *lhs,
    intrin_type_t<uint64_t, 4> const& rhs,
    intrin_type_t<uint64_t, 4>& res);

// 512-bit uint64_t
always_inline void add(clsvec<uint64_t, 8> const *lhs,
    intrin_type_t<uint64_t, 8> const& rhs,
    intrin_type_t<uint64_t, 8>& res);

// 128-bit double
always_inline void add(clsvec<double, 2> const *lhs,
    intrin_type_t<double, 2> const& rhs,
    intrin_type_t<double, 2>& res);

// 256-bit double
always_inline void add(clsvec<double, 4> const *lhs,
    intrin_type_t<double, 4> const& rhs,
    intrin_type_t<double, 4>& res);

// 512-bit double
always_inline void add(clsvec<double, 8> const *lhs,
    intrin_type_t<double, 8> const& rhs,
    intrin_type_t<double, 8>& res);

// 128-bit int8_t sub
always_inline void sub(clsvec<int8_t, 16> const *lhs,
    intrin_type_t<int8_t, 16> const& rhs,
    intrin_type_t<int8_t, 16>& res);

// 256-bit int8_t
always_inline void sub(clsvec<int8_t, 32> const *lhs,
    intrin_type_t<int8_t, 32> const& rhs,
    intrin_type_t<int8_t, 32>& res);

// 512-bit int8_t
always_inline void sub(clsvec<int8_t, 64> const *lhs,
    intrin_type_t<int8_t, 64> const& rhs,
    intrin_type_t<int8_t, 64>& res);

// 128-bit uint8_t
always_inline void sub(clsvec<uint8_t, 16> const *lhs,
    intrin_type_t<uint8_t, 16> const& rhs,
    intrin_type_t<uint8_t, 16>& res);

// 256-bit uint8_t
always_inline void sub(clsvec<uint8_t, 32> const *lhs,
    intrin_type_t<uint8_t, 32> const& rhs,
    intrin_type_t<uint8_t, 32>& res);

// 512-bit uint8_t
always_inline void sub(clsvec<uint8_t, 64> const *lhs,
    intrin_type_t<uint8_t, 64> const& rhs,
    intrin_type_t<uint8_t, 64>& res);

// 128-bit int16_t
always_inline void sub(clsvec<int16_t, 8> const *lhs,
    intrin_type_t<int16_t, 8> const& rhs,
    intrin_type_t<int16_t, 8>& res);

// 256-bit int16_t
always_inline void sub(clsvec<int16_t, 16> const *lhs,
    intrin_type_t<int16_t, 16> const& rhs,
    intrin_type_t<int16_t, 16>& res);

// 512-bit int16_t
always_inline void sub(clsvec<int16_t, 32> const *lhs,
    intrin_type_t<int16_t, 32> const& rhs,
    intrin_type_t<int16_t, 32>& res);

// 128-bit uint16_t
always_inline void sub(clsvec<uint16_t, 8> const *lhs,
    intrin_type_t<uint16_t, 8> const& rhs,
    intrin_type_t<uint16_t, 8>& res);

// 256-bit uint16_t
always_inline void sub(clsvec<uint16_t, 16> const *lhs,
    intrin_type_t<uint16_t, 16> const& rhs,
    intrin_type_t<uint16_t, 16>& res);

// 512-bit uint16_t
always_inline void sub(clsvec<uint16_t, 32> const *lhs,
    intrin_type_t<uint16_t, 32> const& rhs,
    intrin_type_t<uint16_t, 32>& res);

// 128-bit int32_t
always_inline void sub(clsvec<int32_t, 4> const *lhs,
    intrin_type_t<int32_t, 4> const& rhs,
    intrin_type_t<int32_t, 4>& res);

// 256-bit int32_t
always_inline void sub(clsvec<int32_t, 8> const *lhs,
    intrin_type_t<int32_t, 8> const& rhs,
    intrin_type_t<int32_t, 8>& res);

// 512-bit int32_t
always_inline void sub(clsvec<int32_t, 16> const *lhs,
    intrin_type_t<int32_t, 16> const& rhs,
    intrin_type_t<int32_t, 16>& res);

// 128-bit uint32_t
always_inline void sub(clsvec<uint32_t, 4> const *lhs,
    intrin_type_t<uint32_t, 4> const& rhs,
    intrin_type_t<uint32_t, 4>& res);

// 256-bit uint32_t
always_inline void sub(clsvec<uint32_t, 8> const *lhs,
    intrin_type_t<uint32_t, 8> const& rhs,
    intrin_type_t<uint32_t, 8>& res);

// 512-bit uint32_t
always_inline void sub(clsvec<uint32_t, 16> const *lhs,
    intrin_type_t<uint32_t, 16> const& rhs,
    intrin_type_t<uint32_t, 16>& res);

// 128-bit float
always_inline void sub(clsvec<float, 4> const *lhs,
    intrin_type_t<float, 4> const& rhs,
    intrin_type_t<float, 4>& res);

// 256-bit float
always_inline void sub(clsvec<float, 8> const *lhs,
    intrin_type_t<float, 8> const& rhs,
    intrin_type_t<float, 8>& res);

// 512-bit float
always_inline void sub(clsvec<float, 16> const *lhs,
    intrin_type_t<float, 16> const& rhs,
    intrin_type_t<float, 16>& res);

// 128-bit int64_t
always_inline void sub(clsvec<int64_t, 2> const *lhs,
    intrin_type_t<int64_t, 2> const& rhs,
    intrin_type_t<int64_t, 2>& res);

// 256-bit int64_t
always_inline void sub(clsvec<int64_t, 4> const *lhs,
    intrin_type_t<int64_t, 4> const& rhs,
    intrin_type_t<int64_t, 4>& res);

// 512-bit int64_t
always_inline void sub(clsvec<int64_t, 8> const *lhs,
    intrin_type_t<int64_t, 8> const& rhs,
    intrin_type_t<int64_t, 8>& res);

// 128-bit uint64_t
always_inline void sub(clsvec<uint64_t, 2> const *lhs,
    intrin_type_t<uint64_t, 2> const& rhs,
    intrin_type_t<uint64_t, 2>& res);

// 256-bit uint64_t
always_inline void sub(clsvec<uint64_t, 4> const *lhs,
    intrin_type_t<uint64_t, 4> const& rhs,
    intrin_type_t<uint64_t, 4>& res);

// 512-bit uint64_t
always_inline void sub(clsvec<uint64_t, 8> const *lhs,
    intrin_type_t<uint64_t, 8> const& rhs,
    intrin_type_t<uint64_t, 8>& res);

// 128-bit double
always_inline void sub(clsvec<double, 2> const *lhs,
    intrin_type_t<double, 2> const& rhs,
    intrin_type_t<double, 2>& res);

// 256-bit double
always_inline void sub(clsvec<double, 4> const *lhs,
    intrin_type_t<double, 4> const& rhs,
    intrin_type_t<double, 4>& res);

// 512-bit double
always_inline void sub(clsvec<double, 8> const *lhs,
    intrin_type_t<double, 8> const& rhs,
    intrin_type_t<double, 8>& res);

// 128-bit int8_t mul
always_inline void mul(clsvec<int8_t, 16> const *lhs,
    intrin_type_t<int8_t, 16> const& rhs,
    intrin_type_t<int8_t, 16>& res);

// 256-bit int8_t
always_inline void mul(clsvec<int8_t, 32> const *lhs,
    intrin_type_t<int8_t, 32> const& rhs,
    intrin_type_t<int8_t, 32>& res);

// 512-bit int8_t
always_inline void mul(clsvec<int8_t, 64> const *lhs,
    intrin_type_t<int8_t, 64> const& rhs,
    intrin_type_t<int8_t, 64>& res);

// 128-bit uint8_t
always_inline void mul(clsvec<uint8_t, 16> const *lhs,
    intrin_type_t<uint8_t, 16> const& rhs,
    intrin_type_t<uint8_t, 16>& res);

// 256-bit uint8_t
always_inline void mul(clsvec<uint8_t, 32> const *lhs,
    intrin_type_t<uint8_t, 32> const& rhs,
    intrin_type_t<uint8_t, 32>& res);

// 512-bit uint8_t
always_inline void mul(clsvec<uint8_t, 64> const *lhs,
    intrin_type_t<uint8_t, 64> const& rhs,
    intrin_type_t<uint8_t, 64>& res);

// 128-bit int16_t
always_inline void mul(clsvec<int16_t, 8> const *lhs,
    intrin_type_t<int16_t, 8> const& rhs,
    intrin_type_t<int16_t, 8>& res);

// 256-bit int16_t
always_inline void mul(clsvec<int16_t, 16> const *lhs,
    intrin_type_t<int16_t, 16> const& rhs,
    intrin_type_t<int16_t, 16>& res);

// 512-bit int16_t
always_inline void mul(clsvec<int16_t, 32> const *lhs,
    intrin_type_t<int16_t, 32> const& rhs,
    intrin_type_t<int16_t, 32>& res);

// 128-bit uint16_t
always_inline void mul(clsvec<uint16_t, 8> const *lhs,
    intrin_type_t<uint16_t, 8> const& rhs,
    intrin_type_t<uint16_t, 8>& res);

// 256-bit uint16_t
always_inline void mul(clsvec<uint16_t, 16> const *lhs,
    intrin_type_t<uint16_t, 16> const& rhs,
    intrin_type_t<uint16_t, 16>& res);

// 512-bit uint16_t
always_inline void mul(clsvec<uint16_t, 32> const *lhs,
    intrin_type_t<uint16_t, 32> const& rhs,
    intrin_type_t<uint16_t, 32>& res);

// 128-bit int32_t
always_inline void mul(clsvec<int32_t, 4> const *lhs,
    intrin_type_t<int32_t, 4> const& rhs,
    intrin_type_t<int32_t, 4>& res);

// 256-bit int32_t
always_inline void mul(clsvec<int32_t, 8> const *lhs,
    intrin_type_t<int32_t, 8> const& rhs,
    intrin_type_t<int32_t, 8>& res);

// 512-bit int32_t
always_inline void mul(clsvec<int32_t, 16> const *lhs,
    intrin_type_t<int32_t, 16> const& rhs,
    intrin_type_t<int32_t, 16>& res);

// 128-bit uint32_t
always_inline void mul(clsvec<uint32_t, 4> const *lhs,
    intrin_type_t<uint32_t, 4> const& rhs,
    intrin_type_t<uint32_t, 4>& res);

// 256-bit uint32_t
always_inline void mul(clsvec<uint32_t, 8> const *lhs,
    intrin_type_t<uint32_t, 8> const& rhs,
    intrin_type_t<uint32_t, 8>& res);

// 512-bit uint32_t
always_inline void mul(clsvec<uint32_t, 16> const *lhs,
    intrin_type_t<uint32_t, 16> const& rhs,
    intrin_type_t<uint32_t, 16>& res);

// 128-bit float
always_inline void mul(clsvec<float, 4> const *lhs,
    intrin_type_t<float, 4> const& rhs,
    intrin_type_t<float, 4>& res);

// 256-bit float
always_inline void mul(clsvec<float, 8> const *lhs,
    intrin_type_t<float, 8> const& rhs,
    intrin_type_t<float, 8>& res);

// 512-bit float
always_inline void mul(clsvec<float, 16> const *lhs,
    intrin_type_t<float, 16> const& rhs,
    intrin_type_t<float, 16>& res);

// 128-bit int64_t
always_inline void mul(clsvec<int64_t, 2> const *lhs,
    intrin_type_t<int64_t, 2> const& rhs,
    intrin_type_t<int64_t, 2>& res);

// 256-bit int64_t
always_inline void mul(clsvec<int64_t, 4> const *lhs,
    intrin_type_t<int64_t, 4> const& rhs,
    intrin_type_t<int64_t, 4>& res);

// 512-bit int64_t
always_inline void mul(clsvec<int64_t, 8> const *lhs,
    intrin_type_t<int64_t, 8> const& rhs,
    intrin_type_t<int64_t, 8>& res);

// 128-bit uint64_t
always_inline void mul(clsvec<uint64_t, 2> const *lhs,
    intrin_type_t<uint64_t, 2> const& rhs,
    intrin_type_t<uint64_t, 2>& res);

// 256-bit uint64_t
always_inline void mul(clsvec<uint64_t, 4> const *lhs,
    intrin_type_t<uint64_t, 4> const& rhs,
    intrin_type_t<uint64_t, 4>& res);

// 512-bit uint64_t
always_inline void mul(clsvec<uint64_t, 8> const *lhs,
    intrin_type_t<uint64_t, 8> const& rhs,
    intrin_type_t<uint64_t, 8>& res);

// 128-bit double
always_inline void mul(clsvec<double, 2> const *lhs,
    intrin_type_t<double, 2> const& rhs,
    intrin_type_t<double, 2>& res);

// 256-bit double
always_inline void mul(clsvec<double, 4> const *lhs,
    intrin_type_t<double, 4> const& rhs,
    intrin_type_t<double, 4>& res);

// 512-bit double
always_inline void mul(clsvec<double, 8> const *lhs,
    intrin_type_t<double, 8> const& rhs,
    intrin_type_t<double, 8>& res);

// 128-bit int8_t div
always_inline void div(clsvec<int8_t, 16> const *lhs,
    intrin_type_t<int8_t, 16> const& rhs,
    intrin_type_t<int8_t, 16>& res);

// 256-bit int8_t
always_inline void div(clsvec<int8_t, 32> const *lhs,
    intrin_type_t<int8_t, 32> const& rhs,
    intrin_type_t<int8_t, 32>& res);

// 512-bit int8_t
always_inline void div(clsvec<int8_t, 64> const *lhs,
    intrin_type_t<int8_t, 64> const& rhs,
    intrin_type_t<int8_t, 64>& res);

// 128-bit uint8_t
always_inline void div(clsvec<uint8_t, 16> const *lhs,
    intrin_type_t<uint8_t, 16> const& rhs,
    intrin_type_t<uint8_t, 16>& res);

// 256-bit uint8_t
always_inline void div(clsvec<uint8_t, 32> const *lhs,
    intrin_type_t<uint8_t, 32> const& rhs,
    intrin_type_t<uint8_t, 32>& res);

// 512-bit uint8_t
always_inline void div(clsvec<uint8_t, 64> const *lhs,
    intrin_type_t<uint8_t, 64> const& rhs,
    intrin_type_t<uint8_t, 64>& res);

// 128-bit int16_t
always_inline void div(clsvec<int16_t, 8> const *lhs,
    intrin_type_t<int16_t, 8> const& rhs,
    intrin_type_t<int16_t, 8>& res);

// 256-bit int16_t
always_inline void div(clsvec<int16_t, 16> const *lhs,
    intrin_type_t<int16_t, 16> const& rhs,
    intrin_type_t<int16_t, 16>& res);

// 512-bit int16_t
always_inline void div(clsvec<int16_t, 32> const *lhs,
    intrin_type_t<int16_t, 32> const& rhs,
    intrin_type_t<int16_t, 32>& res);

// 128-bit uint16_t
always_inline void div(clsvec<uint16_t, 8> const *lhs,
    intrin_type_t<uint16_t, 8> const& rhs,
    intrin_type_t<uint16_t, 8>& res);

// 256-bit uint16_t
always_inline void div(clsvec<uint16_t, 16> const *lhs,
    intrin_type_t<uint16_t, 16> const& rhs,
    intrin_type_t<uint16_t, 16>& res);

// 512-bit uint16_t
always_inline void div(clsvec<uint16_t, 32> const *lhs,
    intrin_type_t<uint16_t, 32> const& rhs,
    intrin_type_t<uint16_t, 32>& res);

// 128-bit int32_t
always_inline void div(clsvec<int32_t, 4> const *lhs,
    intrin_type_t<int32_t, 4> const& rhs,
    intrin_type_t<int32_t, 4>& res);

// 256-bit int32_t
always_inline void div(clsvec<int32_t, 8> const *lhs,
    intrin_type_t<int32_t, 8> const& rhs,
    intrin_type_t<int32_t, 8>& res);

// 512-bit int32_t
always_inline void div(clsvec<int32_t, 16> const *lhs,
    intrin_type_t<int32_t, 16> const& rhs,
    intrin_type_t<int32_t, 16>& res);

// 128-bit uint32_t
always_inline void div(clsvec<uint32_t, 4> const *lhs,
    intrin_type_t<uint32_t, 4> const& rhs,
    intrin_type_t<uint32_t, 4>& res);

// 256-bit uint32_t
always_inline void div(clsvec<uint32_t, 8> const *lhs,
    intrin_type_t<uint32_t, 8> const& rhs,
    intrin_type_t<uint32_t, 8>& res);

// 512-bit uint32_t
always_inline void div(clsvec<uint32_t, 16> const *lhs,
    intrin_type_t<uint32_t, 16> const& rhs,
    intrin_type_t<uint32_t, 16>& res);

// 128-bit float
always_inline void div(clsvec<float, 4> const *lhs,
    intrin_type_t<float, 4> const& rhs,
    intrin_type_t<float, 4>& res);

// 256-bit float
always_inline void div(clsvec<float, 8> const *lhs,
    intrin_type_t<float, 8> const& rhs,
    intrin_type_t<float, 8>& res);

// 512-bit float
always_inline void div(clsvec<float, 16> const *lhs,
    intrin_type_t<float, 16> const& rhs,
    intrin_type_t<float, 16>& res);

// 128-bit int64_t
always_inline void div(clsvec<int64_t, 2> const *lhs,
    intrin_type_t<int64_t, 2> const& rhs,
    intrin_type_t<int64_t, 2>& res);

// 256-bit int64_t
always_inline void div(clsvec<int64_t, 4> const *lhs,
    intrin_type_t<int64_t, 4> const& rhs,
    intrin_type_t<int64_t, 4>& res);

// 512-bit int64_t
always_inline void div(clsvec<int64_t, 8> const *lhs,
    intrin_type_t<int64_t, 8> const& rhs,
    intrin_type_t<int64_t, 8>& res);

// 128-bit uint64_t
always_inline void div(clsvec<uint64_t, 2> const *lhs,
    intrin_type_t<uint64_t, 2> const& rhs,
    intrin_type_t<uint64_t, 2>& res);

// 256-bit uint64_t
always_inline void div(clsvec<uint64_t, 4> const *lhs,
    intrin_type_t<uint64_t, 4> const& rhs,
    intrin_type_t<uint64_t, 4>& res);

// 512-bit uint64_t
always_inline void div(clsvec<uint64_t, 8> const *lhs,
    intrin_type_t<uint64_t, 8> const& rhs,
    intrin_type_t<uint64_t, 8>& res);

// 128-bit double
always_inline void div(clsvec<double, 2> const *lhs,
    intrin_type_t<double, 2> const& rhs,
    intrin_type_t<double, 2>& res);

// 256-bit double
always_inline void div(clsvec<double, 4> const *lhs,
    intrin_type_t<double, 4> const& rhs,
    intrin_type_t<double, 4>& res);

// 512-bit double
always_inline void div(clsvec<double, 8> const *lhs,
    intrin_type_t<double, 8> const& rhs,
    intrin_type_t<double, 8>& res);

template<typename C, size_t N>
class clsvec
{
public:
    using component_t = C;
    static size_t constexpr component_count = N;

    clsvec()
    {
    }

    clsvec(C scalar)
    {
        init(this, scalar);
    }

    template<typename ...Args>
    clsvec(Args&& ...args)
    {
        static_assert(sizeof...(args) == N);
        init(this, std::forward<Args>(args)...);
    }

    clsvec operator+(clsvec const& rhs) const
    {
        clsvec res;
        add(this, rhs.data, res.data);
        return res;
    }

    clsvec &operator+=(clsvec const& rhs)
    {
        add(this, rhs.data, data);
        return *this;
    }

    clsvec operator-(clsvec const& rhs) const
    {
        clsvec res;
        sub(this, rhs.data, res.data);
        return res;
    }

    clsvec &operator-=(clsvec const& rhs)
    {
        sub(this, rhs.data, data);
        return *this;
    }

    clsvec operator*(clsvec const& rhs) const
    {
        clsvec res;
        mul(this, rhs.data, res.data);
        return res;
    }

    clsvec &operator*=(clsvec const& rhs)
    {
        mul(this, rhs.data, data);
        return *this;
    }

    clsvec operator/(clsvec const& rhs) const
    {
        clsvec res;
        div(this, rhs.data, res.data);
        return res;
    }

    clsvec &operator/=(clsvec const& rhs)
    {
        div(this, rhs.data, data);
        return *this;
    }

    using underlying_type = intrin_type_t<C, N>;
    underlying_type data;
};

template class clsvec<uint8_t, 16>;
template class clsvec<int8_t, 16>;
template class clsvec<uint16_t, 8>;
template class clsvec<int16_t, 8>;
template class clsvec<uint32_t, 4>;
template class clsvec<int32_t, 4>;
template class clsvec<float, 4>;
template class clsvec<uint64_t, 2>;
template class clsvec<int64_t, 2>;
template class clsvec<double, 2>;

#if defined(__AVX2__)
template class clsvec<uint8_t, 32>;
template class clsvec<int8_t, 32>;
template class clsvec<uint16_t, 16>;
template class clsvec<int16_t, 16>;
template class clsvec<uint32_t, 8>;
template class clsvec<int32_t, 8>;
template class clsvec<float, 8>;
template class clsvec<uint64_t, 4>;
template class clsvec<int64_t, 4>;
template class clsvec<double, 4>;
#endif

#if defined(__AVX512F__)
template class clsvec<uint8_t, 64>;
template class clsvec<int8_t, 64>;
template class clsvec<uint16_t, 32>;
template class clsvec<int16_t, 32>;
template class clsvec<uint32_t, 16>;
template class clsvec<int32_t, 16>;
template class clsvec<float, 16>;
template class clsvec<uint64_t, 8>;
template class clsvec<int64_t, 8>;
template class clsvec<double, 8>;
#endif

//
// 8-bit

// 128-bit int8_t init
always_inline void init(clsvec<int8_t, 16> *lhs,
    int8_t v00, int8_t v01, int8_t v02, int8_t v03,
    int8_t v04, int8_t v05, int8_t v06, int8_t v07,
    int8_t v08, int8_t v09, int8_t v10, int8_t v11,
    int8_t v12, int8_t v13, int8_t v14, int8_t v15);

// 128-bit uint8_t
always_inline void init(clsvec<uint8_t, 16> *lhs,
    uint8_t v00, uint8_t v01, uint8_t v02, uint8_t v03,
    uint8_t v04, uint8_t v05, uint8_t v06, uint8_t v07,
    uint8_t v08, uint8_t v09, uint8_t v10, uint8_t v11,
    uint8_t v12, uint8_t v13, uint8_t v14, uint8_t v15);

// 256-bit int8_t
always_inline void init(clsvec<int8_t, 32> *lhs,
    int8_t v00, int8_t v01, int8_t v02, int8_t v03,
    int8_t v04, int8_t v05, int8_t v06, int8_t v07,
    int8_t v08, int8_t v09, int8_t v10, int8_t v11,
    int8_t v12, int8_t v13, int8_t v14, int8_t v15,
    int8_t v16, int8_t v17, int8_t v18, int8_t v19,
    int8_t v20, int8_t v21, int8_t v22, int8_t v23,
    int8_t v24, int8_t v25, int8_t v26, int8_t v27,
    int8_t v28, int8_t v29, int8_t v30, int8_t v31);

// 256-bit uint8_t
always_inline void init(clsvec<uint8_t, 32> *lhs,
   uint8_t v00, uint8_t v01, uint8_t v02, uint8_t v03,
   uint8_t v04, uint8_t v05, uint8_t v06, uint8_t v07,
   uint8_t v08, uint8_t v09, uint8_t v10, uint8_t v11,
   uint8_t v12, uint8_t v13, uint8_t v14, uint8_t v15,
   uint8_t v16, uint8_t v17, uint8_t v18, uint8_t v19,
   uint8_t v20, uint8_t v21, uint8_t v22, uint8_t v23,
   uint8_t v24, uint8_t v25, uint8_t v26, uint8_t v27,
   uint8_t v28, uint8_t v29, uint8_t v30, uint8_t v31);

// 512-bit int8_t
always_inline void init(clsvec<int8_t, 64> *lhs,
   int8_t v00, int8_t v01, int8_t v02, int8_t v03,
   int8_t v04, int8_t v05, int8_t v06, int8_t v07,
   int8_t v08, int8_t v09, int8_t v10, int8_t v11,
   int8_t v12, int8_t v13, int8_t v14, int8_t v15,
   int8_t v16, int8_t v17, int8_t v18, int8_t v19,
   int8_t v20, int8_t v21, int8_t v22, int8_t v23,
   int8_t v24, int8_t v25, int8_t v26, int8_t v27,
   int8_t v28, int8_t v29, int8_t v30, int8_t v31,
   int8_t v32, int8_t v33, int8_t v34, int8_t v35,
   int8_t v36, int8_t v37, int8_t v38, int8_t v39,
   int8_t v40, int8_t v41, int8_t v42, int8_t v43,
   int8_t v44, int8_t v45, int8_t v46, int8_t v47,
   int8_t v48, int8_t v49, int8_t v50, int8_t v51,
   int8_t v52, int8_t v53, int8_t v54, int8_t v55,
   int8_t v56, int8_t v57, int8_t v58, int8_t v59,
   int8_t v60, int8_t v61, int8_t v62, int8_t v63);

// 512-bit uint8_t
always_inline void init(clsvec<uint8_t, 64> *lhs,
   uint8_t v00, uint8_t v01, uint8_t v02, uint8_t v03,
   uint8_t v04, uint8_t v05, uint8_t v06, uint8_t v07,
   uint8_t v08, uint8_t v09, uint8_t v10, uint8_t v11,
   uint8_t v12, uint8_t v13, uint8_t v14, uint8_t v15,
   uint8_t v16, uint8_t v17, uint8_t v18, uint8_t v19,
   uint8_t v20, uint8_t v21, uint8_t v22, uint8_t v23,
   uint8_t v24, uint8_t v25, uint8_t v26, uint8_t v27,
   uint8_t v28, uint8_t v29, uint8_t v30, uint8_t v31,
   uint8_t v32, uint8_t v33, uint8_t v34, uint8_t v35,
   uint8_t v36, uint8_t v37, uint8_t v38, uint8_t v39,
   uint8_t v40, uint8_t v41, uint8_t v42, uint8_t v43,
   uint8_t v44, uint8_t v45, uint8_t v46, uint8_t v47,
   uint8_t v48, uint8_t v49, uint8_t v50, uint8_t v51,
   uint8_t v52, uint8_t v53, uint8_t v54, uint8_t v55,
   uint8_t v56, uint8_t v57, uint8_t v58, uint8_t v59,
   uint8_t v60, uint8_t v61, uint8_t v62, uint8_t v63);

//
// 16-bit

// 128-bit int16_t
always_inline void init(clsvec<int16_t, 8> *lhs,
    int16_t v00, int16_t v01, int16_t v02, int16_t v03,
    int16_t v04, int16_t v05, int16_t v06, int16_t v07);

// 128-bit uint16_t
always_inline void init(clsvec<uint16_t, 8> *lhs,
    uint16_t v00, uint16_t v01, uint16_t v02, uint16_t v03,
    uint16_t v04, uint16_t v05, uint16_t v06, uint16_t v07);

// 256-bit int16_t
always_inline void init(clsvec<int16_t, 16> *lhs,
    int16_t v00, int16_t v01, int16_t v02, int16_t v03,
    int16_t v04, int16_t v05, int16_t v06, int16_t v07,
    int16_t v08, int16_t v09, int16_t v10, int16_t v11,
    int16_t v12, int16_t v13, int16_t v14, int16_t v15);

// 256-bit uint16_t
always_inline void init(clsvec<uint16_t, 16> *lhs,
   uint16_t v00, uint16_t v01, uint16_t v02, uint16_t v03,
   uint16_t v04, uint16_t v05, uint16_t v06, uint16_t v07,
   uint16_t v08, uint16_t v09, uint16_t v10, uint16_t v11,
   uint16_t v12, uint16_t v13, uint16_t v14, uint16_t v15);

// 512-bit int16_t
always_inline void init(clsvec<int16_t, 32> *lhs,
   int16_t v00, int16_t v01, int16_t v02, int16_t v03,
   int16_t v04, int16_t v05, int16_t v06, int16_t v07,
   int16_t v08, int16_t v09, int16_t v10, int16_t v11,
   int16_t v12, int16_t v13, int16_t v14, int16_t v15,
   int16_t v16, int16_t v17, int16_t v18, int16_t v19,
   int16_t v20, int16_t v21, int16_t v22, int16_t v23,
   int16_t v24, int16_t v25, int16_t v26, int16_t v27,
   int16_t v28, int16_t v29, int16_t v30, int16_t v31);

// 512-bit uint16_t
always_inline void init(clsvec<uint16_t, 32> *lhs,
   uint16_t v00, uint16_t v01, uint16_t v02, uint16_t v03,
   uint16_t v04, uint16_t v05, uint16_t v06, uint16_t v07,
   uint16_t v08, uint16_t v09, uint16_t v10, uint16_t v11,
   uint16_t v12, uint16_t v13, uint16_t v14, uint16_t v15,
   uint16_t v16, uint16_t v17, uint16_t v18, uint16_t v19,
   uint16_t v20, uint16_t v21, uint16_t v22, uint16_t v23,
   uint16_t v24, uint16_t v25, uint16_t v26, uint16_t v27,
   uint16_t v28, uint16_t v29, uint16_t v30, uint16_t v31,
   uint16_t v32, uint16_t v33, uint16_t v34, uint16_t v35,
   uint16_t v36, uint16_t v37, uint16_t v38, uint16_t v39,
   uint16_t v40, uint16_t v41, uint16_t v42, uint16_t v43,
   uint16_t v44, uint16_t v45, uint16_t v46, uint16_t v47,
   uint16_t v48, uint16_t v49, uint16_t v50, uint16_t v51,
   uint16_t v52, uint16_t v53, uint16_t v54, uint16_t v55,
   uint16_t v56, uint16_t v57, uint16_t v58, uint16_t v59,
   uint16_t v60, uint16_t v61, uint16_t v62, uint16_t v63);

//
// 32-bit

// 128-bit int32_t
always_inline void init(clsvec<int32_t, 4> *lhs,
    int32_t v00, int32_t v01, int32_t v02, int32_t v03);

// 128-bit uint32_t
always_inline void init(clsvec<uint32_t, 4> *lhs,
    uint32_t v00, uint32_t v01, uint32_t v02, uint32_t v03);

#if defined(__AVX2__)
// 256-bit int32_t
always_inline void init(clsvec<int32_t, 8> *lhs,
    int32_t v00, int32_t v01, int32_t v02, int32_t v03,
    int32_t v04, int32_t v05, int32_t v06, int32_t v07);

// 256-bit uint32_t
always_inline void init(clsvec<uint32_t, 8> *lhs,
   uint32_t v00, uint32_t v01, uint32_t v02, uint32_t v03,
   uint32_t v04, uint32_t v05, uint32_t v06, uint32_t v07);
#endif

#if defined(__AVX512F__)
// 512-bit int32_t
always_inline void init(clsvec<int32_t, 16> *lhs,
   int32_t v00, int32_t v01, int32_t v02, int32_t v03,
   int32_t v04, int32_t v05, int32_t v06, int32_t v07,
   int32_t v08, int32_t v09, int32_t v10, int32_t v11,
   int32_t v12, int32_t v13, int32_t v14, int32_t v15);

// 512-bit uint32_t
always_inline void init(clsvec<uint32_t, 16> *lhs,
   uint32_t v00, uint32_t v01, uint32_t v02, uint32_t v03,
   uint32_t v04, uint32_t v05, uint32_t v06, uint32_t v07,
   uint32_t v08, uint32_t v09, uint32_t v10, uint32_t v11,
   uint32_t v12, uint32_t v13, uint32_t v14, uint32_t v15);
#endif

// 128-bit float
always_inline void init(clsvec<float, 4> *lhs,
   float v00, float v01, float v02, float v03);

#if defined(__AVX2__)
// 256-bit float
always_inline void init(clsvec<float, 8> *lhs,
   float v00, float v01, float v02, float v03,
   float v04, float v05, float v06, float v07);
#endif

#if defined(__AVX512F__)
// 512-bit float
always_inline void init(clsvec<float, 16> *lhs,
   float v00, float v01, float v02, float v03,
   float v04, float v05, float v06, float v07,
   float v08, float v09, float v10, float v11,
   float v12, float v13, float v14, float v15);
#endif

//
// 64-bit

// 128-bit int64_t
always_inline void init(clsvec<int64_t, 2> *lhs,
    int64_t v00, int64_t v01);

// 128-bit uint64_t
always_inline void init(clsvec<uint64_t, 2> *lhs,
    uint64_t v00, uint64_t v01);

// 256-bit int64_t
always_inline void init(clsvec<int64_t, 4> *lhs,
    int64_t v00, int64_t v01, int64_t v02, int64_t v03);

// 256-bit uint64_t
always_inline void init(clsvec<uint64_t, 4> *lhs,
   uint64_t v00, uint64_t v01, uint64_t v02, uint64_t v03);

// 512-bit int64_t
always_inline void init(clsvec<int64_t, 8> *lhs,
   int64_t v00, int64_t v01, int64_t v02, int64_t v03,
   int64_t v04, int64_t v05, int64_t v06, int64_t v07);

// 512-bit uint64_t
always_inline void init(clsvec<uint64_t, 8> *lhs,
   uint64_t v00, uint64_t v01, uint64_t v02, uint64_t v03,
   uint64_t v04, uint64_t v05, uint64_t v06, uint64_t v07);

// 128-bit double
always_inline void init(clsvec<double, 2> *lhs,
   double v00, double v01);

// 256-bit double
always_inline void init(clsvec<double, 4> *lhs,
   double v00, double v01, double v02, double v03);

// 512-bit double
always_inline void init(clsvec<double, 8> *lhs,
   double v00, double v01, double v02, double v03,
   double v04, double v05, double v06, double v07);

//
// Broadcasts

always_inline void init(clsvec<int8_t, 16> *lhs, int8_t scalar)
{
    lhs->data = _mm_set1_epi8(scalar);
}

always_inline void init(clsvec<int8_t, 32> *lhs, int8_t scalar)
{
    lhs->data = _mm256_set1_epi8(scalar);
}

#if defined(__AVX512F__)
always_inline void init(clsvec<int8_t, 64> *lhs, int8_t scalar)
{
    lhs->data = _mm512_set1_epi8(scalar);
}
#endif

always_inline void init(clsvec<uint8_t, 16> *lhs, uint8_t scalar)
{
    lhs->data = _mm_set1_epi8(scalar);
}

always_inline void init(clsvec<uint8_t, 32> *lhs, uint8_t scalar)
{
    lhs->data = _mm256_set1_epi8(scalar);
}

#if defined(__AVX512F__)
always_inline void init(clsvec<uint8_t, 64> *lhs, uint8_t scalar)
{
    lhs->data = _mm512_set1_epi8(scalar);
}
#endif

always_inline void init(clsvec<int16_t, 8> *lhs, int16_t scalar)
{
    lhs->data = _mm_set1_epi16(scalar);
}

always_inline void init(clsvec<int16_t, 16> *lhs, int16_t scalar)
{
    lhs->data = _mm256_set1_epi16(scalar);
}

#if defined(__AVX512F__)
always_inline void init(clsvec<int16_t, 32> *lhs, int16_t scalar)
{
    lhs->data = _mm512_set1_epi16(scalar);
}
#endif

always_inline void init(clsvec<uint16_t, 8> *lhs, uint16_t scalar)
{
        lhs->data = _mm_set1_epi16(scalar);
}

always_inline void init(clsvec<uint16_t, 16> *lhs, uint16_t scalar)
{
    lhs->data = _mm256_set1_epi16(scalar);
}

#if defined(__AVX512F__)
always_inline void init(clsvec<uint16_t, 32> *lhs, uint16_t scalar)
{
    lhs->data = _mm512_set1_epi16(scalar);
}
#endif

always_inline void init(clsvec<int32_t, 4> *lhs, int32_t scalar)
{
    lhs->data = _mm_set1_epi32(scalar);
}

always_inline void init(clsvec<int32_t, 8> *lhs, int32_t scalar)
{
    lhs->data = _mm256_set1_epi32(scalar);
}

#if defined(__AVX512F__)
always_inline void init(clsvec<int32_t, 16> *lhs, int32_t scalar)
{
    lhs->data = _mm512_set1_epi32(scalar);
}
#endif

always_inline void init(clsvec<uint32_t, 4> *lhs, uint32_t scalar)
{
    lhs->data = _mm_set1_epi32(scalar);
}

always_inline void init(clsvec<uint32_t, 8> *lhs, uint32_t scalar)
{
    lhs->data = _mm256_set1_epi32(scalar);
}

#if defined(__AVX512F__)
always_inline void init(clsvec<uint32_t, 16> *lhs, uint32_t scalar)
{
    lhs->data = _mm512_set1_epi32(scalar);
}
#endif

always_inline void init(clsvec<float, 4> *lhs, float scalar)
{
    lhs->data = _mm_set1_ps(scalar);
}

always_inline void init(clsvec<float, 8> *lhs, float scalar)
{
    lhs->data = _mm256_set1_ps(scalar);
}

#if defined(__AVX512F__)
always_inline void init(clsvec<float, 16> *lhs, float scalar)
{
    lhs->data = _mm512_set1_ps(scalar);
}
#endif

always_inline void init(clsvec<int64_t, 2> *lhs, int64_t scalar)
{
    lhs->data = _mm_set1_epi64x(scalar);
}

always_inline void init(clsvec<int64_t, 4> *lhs, int64_t scalar)
{
    lhs->data = _mm256_set1_epi64x(scalar);
}

#if defined(__AVX512F__)
always_inline void init(clsvec<int64_t, 8> *lhs, int64_t scalar)
{
    lhs->data = _mm512_set1_epi64(scalar);
}
#endif

always_inline void init(clsvec<uint64_t, 2> *lhs, uint64_t scalar)
{
    lhs->data = _mm_set1_epi64x(scalar);
}

always_inline void init(clsvec<uint64_t, 4> *lhs, uint64_t scalar)
{
    lhs->data = _mm256_set1_epi64x(scalar);
}

#if defined(__AVX512F__)
always_inline void init(clsvec<uint64_t, 8> *lhs, uint64_t scalar)
{
    lhs->data = _mm512_set1_epi64(scalar);
}
#endif

always_inline void init(clsvec<double, 2> *lhs, double scalar)
{
    lhs->data = _mm_set1_pd(scalar);
}

always_inline void init(clsvec<double, 4> *lhs, double scalar)
{
    lhs->data = _mm256_set1_pd(scalar);
}

#if defined(__AVX512F__)
always_inline void init(clsvec<double, 8> *lhs, double scalar)
{
    lhs->data = _mm512_set1_pd(scalar);
}
#endif

always_inline void init(clsvec<uint8_t, 16> *lhs,
    uint8_t v00, uint8_t v01, uint8_t v02, uint8_t v03,
    uint8_t v04, uint8_t v05, uint8_t v06, uint8_t v07,
    uint8_t v08, uint8_t v09, uint8_t v10, uint8_t v11,
    uint8_t v12, uint8_t v13, uint8_t v14, uint8_t v15)
{
    lhs->data = _mm_setr_epi8(
        (char)v00, (char)v01, (char)v02, (char)v03,
        (char)v04, (char)v05, (char)v06, (char)v07,
        (char)v08, (char)v09, (char)v10, (char)v11,
        (char)v12, (char)v13, (char)v14, (char)v15);
}

always_inline void init(clsvec<uint16_t, 8> *lhs,
    uint16_t v00, uint16_t v01, uint16_t v02, uint16_t v03,
    uint16_t v04, uint16_t v05, uint16_t v06, uint16_t v07)
{
    lhs->data = _mm_setr_epi16(
        (short)v00, (short)v01, (short)v02, (short)v03,
        (short)v04, (short)v05, (short)v06, (short)v07);
}

always_inline void init(clsvec<uint32_t, 4> *lhs,
    uint32_t v00, uint32_t v01, uint32_t v02, uint32_t v03)
{
    lhs->data = _mm_setr_epi32(
        (int)v00, (int)v01, (int)v02, (int)v03);
}

always_inline void init(clsvec<uint64_t, 2> *lhs,
    uint64_t v00, uint64_t v01)
{
    lhs->data = _mm_setr_epi32(
        (int32_t)v00, (int32_t)(v00 >> 32),
        (int32_t)v01, (int32_t)(v01 >> 32));
}

// 128-bit int8_t add
always_inline void add(clsvec<int8_t, 16> const *lhs,
    intrin_type_t<int8_t, 16> const& rhs,
    intrin_type_t<int8_t, 16>& res)
{
    res = _mm_add_epi8(lhs->data, rhs);
}

#if defined(__AVX2__)
// 256-bit int8_t
always_inline void add(clsvec<int8_t, 32> const *lhs,
    intrin_type_t<int8_t, 32> const& rhs,
    intrin_type_t<int8_t, 32>& res)
{
    intrin_type_t<int8_t, 32> zero = _mm256_setzero_si256();
    intrin_type_t<int16_t, 16> lhs0 = _mm256_unpacklo_epi8(zero, lhs->data);
    intrin_type_t<int16_t, 16> lhs1 = _mm256_unpackhi_epi8(zero, lhs->data);
    intrin_type_t<int16_t, 16> rhs0 = _mm256_unpacklo_epi8(zero, rhs);
    intrin_type_t<int16_t, 16> rhs1 = _mm256_unpackhi_epi8(zero, rhs);
    lhs0 = _mm256_srai_epi16(lhs0, 8);
    lhs1 = _mm256_srai_epi16(lhs1, 8);
    rhs0 = _mm256_srai_epi16(rhs0, 8);
    rhs1 = _mm256_srai_epi16(rhs1, 8);
    intrin_type_t<int8_t, 32> res0 = _mm256_add_epi16(lhs0, rhs0);
    intrin_type_t<int8_t, 32> res1 = _mm256_add_epi16(lhs1, rhs1);
    res = _mm256_packs_epi16(res0, res1);
}
#endif

#if defined(__AVX512F__)
// 512-bit int8_t
always_inline void add(clsvec<int8_t, 64> const *lhs,
    intrin_type_t<int8_t, 64> const& rhs,
    intrin_type_t<int8_t, 64>& res)
{
    res = _mm512_add_epi8(lhs->data, rhs);
}
#endif

// 128-bit uint8_t
always_inline void add(clsvec<uint8_t, 16> const *lhs,
    intrin_type_t<uint8_t, 16> const& rhs,
    intrin_type_t<uint8_t, 16>& res)
{
    res = _mm_add_epi8(lhs->data, rhs);
}

#if defined(__AVX2__)
// 256-bit uint8_t
always_inline void add(clsvec<uint8_t, 32> const *lhs,
    intrin_type_t<uint8_t, 32> const& rhs,
    intrin_type_t<uint8_t, 32>& res)
{
    res = _mm256_add_epi8(lhs->data, rhs);
}
#endif

#if defined(__AVX512F__)
// 512-bit uint8_t
always_inline void add(clsvec<uint8_t, 64> const *lhs,
    intrin_type_t<uint8_t, 64> const& rhs,
    intrin_type_t<uint8_t, 64>& res)
{
    res = _mm512_add_epi8(lhs->data, rhs);
}
#endif

// 128-bit int16_t
always_inline void add(clsvec<int16_t, 8> const *lhs,
    intrin_type_t<int16_t, 8> const& rhs,
    intrin_type_t<int16_t, 8>& res)
{
    res = _mm_add_epi16(lhs->data, rhs);
}

#if defined(__AVX2__)
// 256-bit int16_t
always_inline void add(clsvec<int16_t, 16> const *lhs,
    intrin_type_t<int16_t, 16> const& rhs,
    intrin_type_t<int16_t, 16>& res)
{
    res = _mm256_add_epi16(lhs->data, rhs);
}
#endif

#if defined(__AVX512F__)
// 512-bit int16_t
always_inline void add(clsvec<int16_t, 32> const *lhs,
    intrin_type_t<int16_t, 32> const& rhs,
    intrin_type_t<int16_t, 32>& res)
{
    res = _mm512_add_epi16(lhs->data, rhs);
}
#endif

// 128-bit int16_t
always_inline void add(clsvec<uint16_t, 8> const *lhs,
    intrin_type_t<uint16_t, 8> const& rhs,
    intrin_type_t<uint16_t, 8>& res)
{
    res = _mm_add_epi16(lhs->data, rhs);
}

#if defined(__AVX2__)
// 256-bit int16_t
always_inline void add(clsvec<uint16_t, 16> const *lhs,
    intrin_type_t<uint16_t, 16> const& rhs,
    intrin_type_t<uint16_t, 16>& res)
{
    res = _mm256_add_epi16(lhs->data, rhs);
}
#endif

#if defined(__AVX512F__)
// 512-bit uint16_t
always_inline void add(clsvec<uint16_t, 32> const *lhs,
    intrin_type_t<uint16_t, 32> const& rhs,
    intrin_type_t<uint16_t, 32>& res)
{
    res = _mm512_add_epi16(lhs->data, rhs);
}
#endif

// 128-bit int32_t
always_inline void add(clsvec<int32_t, 4> const *lhs,
    intrin_type_t<int32_t, 4> const& rhs,
    intrin_type_t<int32_t, 4>& res)
{
    res = _mm_add_epi32(lhs->data, rhs);
}

#if defined(__AVX2__)
// 256-bit int32_t
always_inline void add(clsvec<int32_t, 8> const *lhs,
    intrin_type_t<int32_t, 8> const& rhs,
    intrin_type_t<int32_t, 8>& res)
{
    res = _mm256_add_epi32(lhs->data, rhs);
}
#endif

#if defined(__AVX512F__)
// 512-bit int32_t
always_inline void add(clsvec<int32_t, 16> const *lhs,
    intrin_type_t<int32_t, 16> const& rhs,
    intrin_type_t<int32_t, 16>& res)
{
    res = _mm512_add_epi32(lhs->data, rhs);
}
#endif

// 128-bit uint32_t
always_inline void add(clsvec<uint32_t, 4> const *lhs,
    intrin_type_t<uint32_t, 4> const& rhs,
    intrin_type_t<uint32_t, 4>& res)
{
    res = _mm_add_epi32(lhs->data, rhs);
}

#if defined(__AVX2__)
// 256-bit uint32_t
always_inline void add(clsvec<uint32_t, 8> const *lhs,
    intrin_type_t<uint32_t, 8> const& rhs,
    intrin_type_t<uint32_t, 8>& res)
{
    res = _mm256_add_epi32(lhs->data, rhs);
}
#endif

#if defined(__AVX512F__)
// 512-bit uint32_t
always_inline void add(clsvec<uint32_t, 16> const *lhs,
    intrin_type_t<uint32_t, 16> const& rhs,
    intrin_type_t<uint32_t, 16>& res)
{
    res = _mm512_add_epi32(lhs->data, rhs);
}
#endif

// 128-bit float
always_inline void add(clsvec<float, 4> const *lhs,
    intrin_type_t<float, 4> const& rhs,
    intrin_type_t<float, 4>& res)
{
    res = _mm_add_ps(lhs->data, rhs);
}

#if defined(__AVX2__)
// 256-bit float
always_inline void add(clsvec<float, 8> const *lhs,
    intrin_type_t<float, 8> const& rhs,
    intrin_type_t<float, 8>& res)
{
    res = _mm256_add_ps(lhs->data, rhs);
}
#endif

#if defined(__AVX512F__)
// 512-bit float
always_inline void add(clsvec<float, 16> const *lhs,
    intrin_type_t<float, 16> const& rhs,
    intrin_type_t<float, 16>& res)
{
    res = _mm512_add_ps(lhs->data, rhs);
}
#endif

// 128-bit int64_t
always_inline void add(clsvec<int64_t, 2> const *lhs,
    intrin_type_t<int64_t, 2> const& rhs,
    intrin_type_t<int64_t, 2>& res)
{
    res = _mm_add_epi64(lhs->data, rhs);
}

#if defined(__AVX2__)
// 256-bit int64_t
always_inline void add(clsvec<int64_t, 4> const *lhs,
    intrin_type_t<int64_t, 4> const& rhs,
    intrin_type_t<int64_t, 4>& res)
{
    res = _mm256_add_epi64(lhs->data, rhs);
}
#endif

#if defined(__AVX512F__)
// 512-bit int64_t
always_inline void add(clsvec<int64_t, 8> const *lhs,
    intrin_type_t<int64_t, 8> const& rhs,
    intrin_type_t<int64_t, 8>& res)
{
    res = _mm512_add_epi64(lhs->data, rhs);
}
#endif

// 128-bit uint64_t
always_inline void add(clsvec<uint64_t, 2> const *lhs,
    intrin_type_t<uint64_t, 2> const& rhs,
    intrin_type_t<uint64_t, 2>& res)
{
    res = _mm_add_epi64(lhs->data, rhs);
}

#if defined(__AVX2__)
// 256-bit uint64_t
always_inline void add(clsvec<uint64_t, 4> const *lhs,
    intrin_type_t<uint64_t, 4> const& rhs,
    intrin_type_t<uint64_t, 4>& res)
{
    res = _mm256_add_epi64(lhs->data, rhs);
}
#endif

#if defined(__AVX512F__)
// 512-bit uint64_t
always_inline void add(clsvec<uint64_t, 8> const *lhs,
    intrin_type_t<uint64_t, 8> const& rhs,
    intrin_type_t<uint64_t, 8>& res)
{
    res = _mm512_add_epi64(lhs->data, rhs);
}
#endif

// 128-bit double
always_inline void add(clsvec<double, 2> const *lhs,
    intrin_type_t<double, 2> const& rhs,
    intrin_type_t<double, 2>& res)
{
    res = _mm_add_pd(lhs->data, rhs);
}

#if defined(__AVX2__)
// 256-bit double
always_inline void add(clsvec<double, 4> const *lhs,
    intrin_type_t<double, 4> const& rhs,
    intrin_type_t<double, 4>& res)
{
    res = _mm256_add_pd(lhs->data, rhs);
}
#endif

#if defined(__AVX512F__)
// 512-bit double
always_inline void add(clsvec<double, 8> const *lhs,
    intrin_type_t<double, 8> const& rhs,
    intrin_type_t<double, 8>& res)
{
    res = _mm512_add_pd(lhs->data, rhs);
}
#endif

// 128-bit int8_t sub
always_inline void sub(clsvec<int8_t, 16> const *lhs,
    intrin_type_t<int8_t, 16> const& rhs,
    intrin_type_t<int8_t, 16>& res)
{
    res = _mm_sub_epi8(lhs->data, rhs);
}

#if defined(__AVX2__)
// 256-bit int8_t sub
always_inline void sub(clsvec<int8_t, 32> const *lhs,
    intrin_type_t<int8_t, 32> const& rhs,
    intrin_type_t<int8_t, 32>& res)
{
    res = _mm256_sub_epi8(lhs->data, rhs);
}
#endif

#if defined(__AVX512F__)
// 512-bit int8_t sub
always_inline void sub(clsvec<int8_t, 64> const *lhs,
    intrin_type_t<int8_t, 64> const& rhs,
    intrin_type_t<int8_t, 64>& res)
{
    res = _mm512_sub_epi8(lhs->data, rhs);
}
#endif

// 128-bit uint8_t sub
always_inline void sub(clsvec<uint8_t, 16> const *lhs,
    intrin_type_t<uint8_t, 16> const& rhs,
    intrin_type_t<uint8_t, 16>& res)
{
    res = _mm_sub_epi8(lhs->data, rhs);
}

#if defined(__AVX2__)
// 256-bit uint8_t sub
always_inline void sub(clsvec<uint8_t, 32> const *lhs,
    intrin_type_t<uint8_t, 32> const& rhs,
    intrin_type_t<uint8_t, 32>& res)
{
    res = _mm256_sub_epi8(lhs->data, rhs);
}
#endif

#if defined(__AVX512F__)
// 512-bit uint8_t sub
always_inline void sub(clsvec<uint8_t, 64> const *lhs,
    intrin_type_t<uint8_t, 64> const& rhs,
    intrin_type_t<uint8_t, 64>& res)
{
    res = _mm512_sub_epi8(lhs->data, rhs);
}
#endif

// 128-bit int16_t sub
always_inline void sub(clsvec<int16_t, 8> const *lhs,
    intrin_type_t<int16_t, 8> const& rhs,
    intrin_type_t<int16_t, 8>& res)
{
    res = _mm_sub_epi16(lhs->data, rhs);
}

#if defined(__AVX2__)
// 256-bit int16_t sub
always_inline void sub(clsvec<int16_t, 16> const *lhs,
    intrin_type_t<int16_t, 16> const& rhs,
    intrin_type_t<int16_t, 16>& res)
{
    res = _mm256_sub_epi16(lhs->data, rhs);
}
#endif

#if defined(__AVX512F__)
// 512-bit int16_t sub
always_inline void sub(clsvec<int16_t, 32> const *lhs,
    intrin_type_t<int16_t, 32> const& rhs,
    intrin_type_t<int16_t, 32>& res)
{
    res = _mm512_sub_epi16(lhs->data, rhs);
}
#endif

// 128-bit uint16_t sub
always_inline void sub(clsvec<uint16_t, 8> const *lhs,
    intrin_type_t<uint16_t, 8> const& rhs,
    intrin_type_t<uint16_t, 8>& res)
{
    res = _mm_sub_epi16(lhs->data, rhs);
}

#if defined(__AVX2__)
// 256-bit uint16_t sub
always_inline void sub(clsvec<uint16_t, 16> const *lhs,
    intrin_type_t<uint16_t, 16> const& rhs,
    intrin_type_t<uint16_t, 16>& res)
{
    res = _mm256_sub_epi16(lhs->data, rhs);
}
#endif

#if defined(__AVX512F__)
// 512-bit uint16_t sub
always_inline void sub(clsvec<uint16_t, 32> const *lhs,
    intrin_type_t<uint16_t, 32> const& rhs,
    intrin_type_t<uint16_t, 32>& res)
{
    res = _mm512_sub_epi16(lhs->data, rhs);
}
#endif

// 128-bit int32_t sub
always_inline void sub(clsvec<int32_t, 4> const *lhs,
    intrin_type_t<int32_t, 4> const& rhs,
    intrin_type_t<int32_t, 4>& res)
{
    res = _mm_sub_epi32(lhs->data, rhs);
}

#if defined(__AVX2__)
// 256-bit int32_t sub
always_inline void sub(clsvec<int32_t, 8> const *lhs,
    intrin_type_t<int32_t, 8> const& rhs,
    intrin_type_t<int32_t, 8>& res)
{
    res = _mm256_sub_epi32(lhs->data, rhs);
}
#endif

#if defined(__AVX512F__)
// 512-bit int32_t sub
always_inline void sub(clsvec<int32_t, 16> const *lhs,
    intrin_type_t<int32_t, 16> const& rhs,
    intrin_type_t<int32_t, 16>& res)
{
    res = _mm512_sub_epi32(lhs->data, rhs);
}
#endif

// 128-bit uint32_t sub
always_inline void sub(clsvec<uint32_t, 4> const *lhs,
    intrin_type_t<uint32_t, 4> const& rhs,
    intrin_type_t<uint32_t, 4>& res)
{
    res = _mm_sub_epi32(lhs->data, rhs);
}

#if defined(__AVX2__)
// 256-bit uint32_t sub
always_inline void sub(clsvec<uint32_t, 8> const *lhs,
    intrin_type_t<uint32_t, 8> const& rhs,
    intrin_type_t<uint32_t, 8>& res)
{
    res = _mm256_sub_epi32(lhs->data, rhs);
}
#endif

#if defined(__AVX512F__)
// 512-bit uint32_t sub
always_inline void sub(clsvec<uint32_t, 16> const *lhs,
    intrin_type_t<uint32_t, 16> const& rhs,
    intrin_type_t<uint32_t, 16>& res)
{
    res = _mm512_sub_epi32(lhs->data, rhs);
}
#endif

// 128-bit float sub
always_inline void sub(clsvec<float, 4> const *lhs,
    intrin_type_t<float, 4> const& rhs,
    intrin_type_t<float, 4>& res)
{
    res = _mm_sub_ps(lhs->data, rhs);
}

#if defined(__AVX2__)
// 256-bit float sub
always_inline void sub(clsvec<float, 8> const *lhs,
    intrin_type_t<float, 8> const& rhs,
    intrin_type_t<float, 8>& res)
{
    res = _mm256_sub_ps(lhs->data, rhs);
}
#endif

#if defined(__AVX512F__)
// 512-bit float sub
always_inline void sub(clsvec<float, 16> const *lhs,
    intrin_type_t<float, 16> const& rhs,
    intrin_type_t<float, 16>& res)
{
    res = _mm512_sub_ps(lhs->data, rhs);
}
#endif

// 128-bit int64_t sub
always_inline void sub(clsvec<int64_t, 2> const *lhs,
    intrin_type_t<int64_t, 2> const& rhs,
    intrin_type_t<int64_t, 2>& res)
{
    res = _mm_sub_epi64(lhs->data, rhs);
}

#if defined(__AVX2__)
// 256-bit int64_t sub
always_inline void sub(clsvec<int64_t, 4> const *lhs,
    intrin_type_t<int64_t, 4> const& rhs,
    intrin_type_t<int64_t, 4>& res)
{
    res = _mm256_sub_epi64(lhs->data, rhs);
}
#endif

#if defined(__AVX512F__)
// 512-bit int64_t sub
always_inline void sub(clsvec<int64_t, 8> const *lhs,
    intrin_type_t<int64_t, 8> const& rhs,
    intrin_type_t<int64_t, 8>& res)
{
    res = _mm512_sub_epi64(lhs->data, rhs);
}
#endif

// 128-bit uint64_t sub
always_inline void sub(clsvec<uint64_t, 2> const *lhs,
    intrin_type_t<uint64_t, 2> const& rhs,
    intrin_type_t<uint64_t, 2>& res)
{
    res = _mm_sub_epi64(lhs->data, rhs);
}

#if defined(__AVX2__)
// 256-bit uint64_t sub
always_inline void sub(clsvec<uint64_t, 4> const *lhs,
    intrin_type_t<uint64_t, 4> const& rhs,
    intrin_type_t<uint64_t, 4>& res)
{
    res = _mm256_sub_epi64(lhs->data, rhs);
}
#endif

#if defined(__AVX512F__)
// 512-bit uint64_t sub
always_inline void sub(clsvec<uint64_t, 8> const *lhs,
    intrin_type_t<uint64_t, 8> const& rhs,
    intrin_type_t<uint64_t, 8>& res)
{
    res = _mm512_sub_epi64(lhs->data, rhs);
}
#endif

// 128-bit double sub
always_inline void sub(clsvec<double, 2> const *lhs,
    intrin_type_t<double, 2> const& rhs,
    intrin_type_t<double, 2>& res)
{
    res = _mm_sub_pd(lhs->data, rhs);
}

#if defined(__AVX2__)
// 256-bit double sub
always_inline void sub(clsvec<double, 4> const *lhs,
    intrin_type_t<double, 4> const& rhs,
    intrin_type_t<double, 4>& res)
{
    res = _mm256_sub_pd(lhs->data, rhs);
}
#endif

#if defined(__AVX512F__)
// 512-bit double sub
always_inline void sub(clsvec<double, 8> const *lhs,
    intrin_type_t<double, 8> const& rhs,
    intrin_type_t<double, 8>& res)
{
    res = _mm512_sub_pd(lhs->data, rhs);
}
#endif

// 128-bit int8_t mul
always_inline void mul(clsvec<int8_t, 16> const *lhs,
    intrin_type_t<int8_t, 16> const& rhs,
    intrin_type_t<int8_t, 16>& res)
{
    // Unpack backwards, so the value is in the upper 8 bits...
    intrin_type_t<int8_t, 16> const zero =
        _mm_setzero_si128();
    intrin_type_t<int8_t, 16> lhs_hi =
        _mm_unpackhi_epi8(zero, lhs->data);
    intrin_type_t<int8_t, 16> lhs_lo =
        _mm_unpacklo_epi8(zero, lhs->data);
    intrin_type_t<int8_t, 16> rhs_hi =
        _mm_unpackhi_epi8(zero, rhs);
    intrin_type_t<int8_t, 16> rhs_lo =
        _mm_unpacklo_epi8(zero, rhs);
    // ...so I can sign extend them with an arithmetic shift immediate
    lhs_hi = _mm_srai_epi16(lhs_hi, 8);
    rhs_hi = _mm_srai_epi16(rhs_hi, 8);
    lhs_lo = _mm_srai_epi16(lhs_lo, 8);
    rhs_lo = _mm_srai_epi16(rhs_lo, 8);
    // Multiply as 16 bit
    intrin_type_t<int8_t, 16> res_hi =
        _mm_mullo_epi16(lhs_hi, rhs_hi);
    intrin_type_t<int8_t, 16> res_lo =
        _mm_mullo_epi16(lhs_lo, rhs_lo);
    // Signed pack back down to 8 bit
    res = _mm_packs_epi16(res_lo, res_hi);
}

#if defined(__AVX2__)
// 256-bit int8_t
always_inline void mul(clsvec<int8_t, 32> const *lhs,
    intrin_type_t<int8_t, 32> const& rhs,
    intrin_type_t<int8_t, 32>& res)
{
    // Unpack backwards, so the value is in the upper 8 bits...
    intrin_type_t<int8_t, 32> const zero =
        _mm256_setzero_si256();
    intrin_type_t<int8_t, 32> lhs_hi =
        _mm256_unpackhi_epi8(zero, lhs->data);
    intrin_type_t<int8_t, 32> lhs_lo =
        _mm256_unpacklo_epi8(zero, lhs->data);
    intrin_type_t<int8_t, 32> rhs_hi =
        _mm256_unpackhi_epi8(zero, rhs);
    intrin_type_t<int8_t, 32> rhs_lo =
        _mm256_unpacklo_epi8(zero, rhs);
    // ...so I can sign extend them with an arithmetic shift immediate
    lhs_hi = _mm256_srai_epi16(lhs_hi, 8);
    rhs_hi = _mm256_srai_epi16(rhs_hi, 8);
    lhs_lo = _mm256_srai_epi16(lhs_lo, 8);
    rhs_lo = _mm256_srai_epi16(rhs_lo, 8);
    // Multiply as 16 bit
    intrin_type_t<int8_t, 32> res_hi =
        _mm256_mullo_epi16(lhs_hi, rhs_hi);
    intrin_type_t<int8_t, 32> res_lo =
        _mm256_mullo_epi16(lhs_lo, rhs_lo);
    // Signed pack back down to 8 bit
    res = _mm256_packs_epi16(res_lo, res_hi);
}
#endif

#if defined(__AVX512F__)
// 512-bit int8_t
always_inline void mul(clsvec<int8_t, 64> const *lhs,
    intrin_type_t<int8_t, 64> const& rhs,
    intrin_type_t<int8_t, 64>& res)
{
    // Unpack backwards, so the value is in the upper 8 bits...
    intrin_type_t<int8_t, 64> const zero =
        _mm512_setzero_si512();
    intrin_type_t<int8_t, 64> lhs_hi =
        _mm512_unpackhi_epi8(zero, lhs->data);
    intrin_type_t<int8_t, 64> lhs_lo =
        _mm512_unpacklo_epi8(zero, lhs->data);
    intrin_type_t<int8_t, 64> rhs_hi =
        _mm512_unpackhi_epi8(zero, rhs);
    intrin_type_t<int8_t, 64> rhs_lo =
        _mm512_unpacklo_epi8(zero, rhs);
    // ...so I can sign extend them with an arithmetic shift immediate
    lhs_hi = _mm512_srai_epi16(lhs_hi, 8);
    rhs_hi = _mm512_srai_epi16(rhs_hi, 8);
    lhs_lo = _mm512_srai_epi16(lhs_lo, 8);
    rhs_lo = _mm512_srai_epi16(rhs_lo, 8);
    // Multiply as 16 bit
    intrin_type_t<int8_t, 64> res_hi =
        _mm512_mullo_epi16(lhs_hi, rhs_hi);
    intrin_type_t<int8_t, 64> res_lo =
        _mm512_mullo_epi16(lhs_lo, rhs_lo);
    // Signed pack back down to 8 bit
    res = _mm512_packs_epi16(res_lo, res_hi);
}
#endif

// 128-bit uint8_t
always_inline void mul(clsvec<uint8_t, 16> const *lhs,
    intrin_type_t<uint8_t, 16> const& rhs,
    intrin_type_t<uint8_t, 16>& res)
{
    // Zero extend unsigned values
    intrin_type_t<uint8_t, 16> const zero =
        _mm_setzero_si128();
    intrin_type_t<uint8_t, 16> lhs_hi =
        _mm_unpackhi_epi8(lhs->data, zero);
    intrin_type_t<uint8_t, 16> lhs_lo =
        _mm_unpacklo_epi8(lhs->data, zero);
    intrin_type_t<uint8_t, 16> rhs_hi =
        _mm_unpackhi_epi8(rhs, zero);
    intrin_type_t<uint8_t, 16> rhs_lo =
        _mm_unpacklo_epi8(rhs, zero);
    // Multiply as 16 bit
    intrin_type_t<uint8_t, 16> res_hi =
        _mm_mullo_epi16(lhs_hi, rhs_hi);
    intrin_type_t<uint8_t, 16> res_lo =
        _mm_mullo_epi16(lhs_lo, rhs_lo);
    // Unsigned pack back down to 8 bit
    res = _mm_packus_epi16(res_lo, res_hi);
}

#if defined(__AVX2__)
// 256-bit uint8_t
always_inline void mul(clsvec<uint8_t, 32> const *lhs,
    intrin_type_t<uint8_t, 32> const& rhs,
    intrin_type_t<uint8_t, 32>& res)
{
    // Zero extend unsigned values
    intrin_type_t<uint8_t, 32> const zero =
        _mm256_setzero_si256();
    intrin_type_t<uint8_t, 32> lhs_hi =
        _mm256_unpackhi_epi8(lhs->data, zero);
    intrin_type_t<uint8_t, 32> lhs_lo =
        _mm256_unpacklo_epi8(lhs->data, zero);
    intrin_type_t<uint8_t, 32> rhs_hi =
        _mm256_unpackhi_epi8(rhs, zero);
    intrin_type_t<uint8_t, 32> rhs_lo =
        _mm256_unpacklo_epi8(rhs, zero);
    // Multiply as 16 bit
    intrin_type_t<uint8_t, 32> res_hi =
        _mm256_mullo_epi16(lhs_hi, rhs_hi);
    intrin_type_t<uint8_t, 32> res_lo =
        _mm256_mullo_epi16(lhs_lo, rhs_lo);
    // Unsigned pack back down to 8 bit
    res = _mm256_packus_epi16(res_lo, res_hi);
}
#endif

#if defined(__AVX512F__)
// 512-bit uint8_t
always_inline void mul(clsvec<uint8_t, 64> const *lhs,
    intrin_type_t<uint8_t, 64> const& rhs,
    intrin_type_t<uint8_t, 64>& res)
{
    // Zero extend unsigned values
    intrin_type_t<uint8_t, 64> const zero =
        _mm512_setzero_si512();
    intrin_type_t<uint8_t, 64> lhs_hi =
        _mm512_unpackhi_epi8(lhs->data, zero);
    intrin_type_t<uint8_t, 64> lhs_lo =
        _mm512_unpacklo_epi8(lhs->data, zero);
    intrin_type_t<uint8_t, 64> rhs_hi =
        _mm512_unpackhi_epi8(rhs, zero);
    intrin_type_t<uint8_t, 64> rhs_lo =
        _mm512_unpacklo_epi8(rhs, zero);
    // Multiply as 16 bit
    intrin_type_t<uint8_t, 64> res_hi =
        _mm512_mullo_epi16(lhs_hi, rhs_hi);
    intrin_type_t<uint8_t, 64> res_lo =
        _mm512_mullo_epi16(lhs_lo, rhs_lo);
    // Unsigned pack back down to 8 bit
    res = _mm512_packus_epi16(res_lo, res_hi);
}
#endif

// 128-bit int16_t
always_inline void mul(clsvec<int16_t, 8> const *lhs,
    intrin_type_t<int16_t, 8> const& rhs,
    intrin_type_t<int16_t, 8>& res)
{
    res = _mm_mullo_epi16(lhs->data, rhs);
}

#if defined(__AVX2__)
// 256-bit int16_t
always_inline void mul(clsvec<int16_t, 16> const *lhs,
    intrin_type_t<int16_t, 16> const& rhs,
    intrin_type_t<int16_t, 16>& res)
{
    res = _mm256_mullo_epi16(lhs->data, rhs);
}
#endif

#if defined(__AVX512F__)
// 512-bit int16_t
always_inline void mul(clsvec<int16_t, 32> const *lhs,
    intrin_type_t<int16_t, 32> const& rhs,
    intrin_type_t<int16_t, 32>& res)
{
    res = _mm512_mullo_epi16(lhs->data, rhs);
}
#endif

// 128-bit uint16_t
always_inline void mul(clsvec<uint16_t, 8> const *lhs,
    intrin_type_t<uint16_t, 8> const& rhs,
    intrin_type_t<uint16_t, 8>& res)
{
    res = _mm_mullo_epi16(lhs->data, rhs);
}

#if defined(__AVX2__)
// 256-bit uint16_t
always_inline void mul(clsvec<uint16_t, 16> const *lhs,
    intrin_type_t<uint16_t, 16> const& rhs,
    intrin_type_t<uint16_t, 16>& res)
{
    res = _mm256_mullo_epi16(lhs->data, rhs);
}
#endif

#if defined(__AVX512F__)
// 512-bit uint16_t
always_inline void mul(clsvec<uint16_t, 32> const *lhs,
    intrin_type_t<uint16_t, 32> const& rhs,
    intrin_type_t<uint16_t, 32>& res)
{
    res = _mm512_mullo_epi16(lhs->data, rhs);
}
#endif

// 128-bit int32_t
always_inline void mul(clsvec<int32_t, 4> const *lhs,
    intrin_type_t<int32_t, 4> const& rhs,
    intrin_type_t<int32_t, 4>& res)
{
    res = _mm_mullo_epi32(lhs->data, rhs);
}

#if defined(__AVX2__)
// 256-bit int32_t
always_inline void mul(clsvec<int32_t, 8> const *lhs,
    intrin_type_t<int32_t, 8> const& rhs,
    intrin_type_t<int32_t, 8>& res)
{
    res = _mm256_mullo_epi32(lhs->data, rhs);
}
#endif

#if defined(__AVX512F__)
// 512-bit int32_t
always_inline void mul(clsvec<int32_t, 16> const *lhs,
    intrin_type_t<int32_t, 16> const& rhs,
    intrin_type_t<int32_t, 16>& res)
{
    res = _mm512_mullo_epi32(lhs->data, rhs);
}
#endif

// 128-bit uint32_t
always_inline void mul(clsvec<uint32_t, 4> const *lhs,
    intrin_type_t<uint32_t, 4> const& rhs,
    intrin_type_t<uint32_t, 4>& res)
{
    res = _mm_mullo_epi32(lhs->data, rhs);
}

#if defined(__AVX2__)
// 256-bit uint32_t
always_inline void mul(clsvec<uint32_t, 8> const *lhs,
    intrin_type_t<uint32_t, 8> const& rhs,
    intrin_type_t<uint32_t, 8>& res)
{
    res = _mm256_mullo_epi32(lhs->data, rhs);
}
#endif

#if defined(__AVX512F__)
// 512-bit uint32_t
always_inline void mul(clsvec<uint32_t, 16> const *lhs,
    intrin_type_t<uint32_t, 16> const& rhs,
    intrin_type_t<uint32_t, 16>& res)
{
    res = _mm512_mullo_epi32(lhs->data, rhs);
}
#endif

// 128-bit float
always_inline void mul(clsvec<float, 4> const *lhs,
    intrin_type_t<float, 4> const& rhs,
    intrin_type_t<float, 4>& res)
{
    res = _mm_mul_ps(lhs->data, rhs);
}

#if defined(__AVX2__)
// 256-bit float
always_inline void mul(clsvec<float, 8> const *lhs,
    intrin_type_t<float, 8> const& rhs,
    intrin_type_t<float, 8>& res)
{
    res = _mm256_mul_ps(lhs->data, rhs);
}
#endif

#if defined(__AVX512F__)
// 512-bit float
always_inline void mul(clsvec<float, 16> const *lhs,
    intrin_type_t<float, 16> const& rhs,
    intrin_type_t<float, 16>& res)
{
    res = _mm512_mul_ps(lhs->data, rhs);
}
#endif

always_inline intrin_type_t<uint64_t, 2>
_mm_setr_epi64x(long long v0, long long v1)
{
    return _mm_set_epi64x(v1, v0);
}

// 128-bit int64_t
always_inline void mul(clsvec<int64_t, 2> const *lhs,
    intrin_type_t<int64_t, 2> const& rhs,
    intrin_type_t<int64_t, 2>& res)
{
    res = _mm_setr_epi64x(
        _mm_extract_epi64(lhs->data, 0) * _mm_extract_epi64(rhs, 0),
        _mm_extract_epi64(lhs->data, 1) * _mm_extract_epi64(rhs, 1));
}

#if defined(__AVX2__)
// 256-bit int64_t
always_inline void mul(clsvec<int64_t, 4> const *lhs,
    intrin_type_t<int64_t, 4> const& rhs,
    intrin_type_t<int64_t, 4>& res)
{
    intrin_type_t<int64_t, 2> const lhs0 =
        _mm256_extracti128_si256(lhs->data, 0);
    intrin_type_t<int64_t, 2> const lhs1 =
        _mm256_extracti128_si256(lhs->data, 1);
    intrin_type_t<int64_t, 2> const rhs0 =
        _mm256_extracti128_si256(rhs, 0);
    intrin_type_t<int64_t, 2> const rhs1 =
        _mm256_extracti128_si256(rhs, 1);
    res = _mm256_setr_epi64x(
        _mm_extract_epi64(lhs0, 0) * _mm_extract_epi64(rhs0, 0),
        _mm_extract_epi64(lhs0, 1) * _mm_extract_epi64(rhs0, 1),
        _mm_extract_epi64(lhs1, 0) * _mm_extract_epi64(rhs1, 0),
        _mm_extract_epi64(lhs1, 1) * _mm_extract_epi64(rhs1, 1));
}
#endif

#if defined(__AVX512F__)
// 512-bit int64_t
always_inline void mul(clsvec<int64_t, 8> const *lhs,
    intrin_type_t<int64_t, 8> const& rhs,
    intrin_type_t<int64_t, 8>& res)
{
    res = _mm512_mullo_epi64(lhs->data, rhs);
}
#endif

// 128-bit uint64_t
always_inline void mul(clsvec<uint64_t, 2> const *lhs,
    intrin_type_t<uint64_t, 2> const& rhs,
    intrin_type_t<uint64_t, 2>& res)
{
    res = _mm_setr_epi64x(
        (int)((uint32_t)_mm_extract_epi64(lhs->data, 0) *
              (uint32_t)_mm_extract_epi64(rhs, 0)),
        (int)((uint32_t)_mm_extract_epi64(lhs->data, 1) *
              (uint32_t)_mm_extract_epi64(rhs, 1)));
}

#if defined(__AVX2__)
// 256-bit uint64_t
always_inline void mul(clsvec<uint64_t, 4> const *lhs,
    intrin_type_t<uint64_t, 4> const& rhs,
    intrin_type_t<uint64_t, 4>& res)
{
    intrin_type_t<uint64_t, 2> const lhs0 =
        _mm256_extracti128_si256(lhs->data, 0);
    intrin_type_t<uint64_t, 2> const lhs1 =
        _mm256_extracti128_si256(lhs->data, 1);
    intrin_type_t<uint64_t, 2> const rhs0 =
        _mm256_extracti128_si256(rhs, 0);
    intrin_type_t<uint64_t, 2> const rhs1 =
        _mm256_extracti128_si256(rhs, 1);
    res = _mm256_setr_epi64x(
        (long long)((uint64_t)_mm_extract_epi64(lhs0, 0) *
                    (uint64_t)_mm_extract_epi64(rhs0, 0)),
        (long long)((uint64_t)_mm_extract_epi64(lhs0, 1) *
                    (uint64_t)_mm_extract_epi64(rhs0, 1)),
        (long long)((uint64_t)_mm_extract_epi64(lhs1, 0) *
                    (uint64_t)_mm_extract_epi64(rhs1, 0)),
        (long long)((uint64_t)_mm_extract_epi64(lhs1, 1) *
                    (uint64_t)_mm_extract_epi64(rhs1, 1)));
}
#endif

#if defined(__AVX512F__)
// 512-bit uint64_t
always_inline void mul(clsvec<uint64_t, 8> const *lhs,
    intrin_type_t<uint64_t, 8> const& rhs,
    intrin_type_t<uint64_t, 8>& res)
{
    res = _mm512_mullo_epi64(lhs->data, rhs);
}
#endif

// 128-bit double
always_inline void mul(clsvec<double, 2> const *lhs,
    intrin_type_t<double, 2> const& rhs,
    intrin_type_t<double, 2>& res)
{
    res = _mm_mul_pd(lhs->data, rhs);
}

#if defined(__AVX2__)
// 256-bit double
always_inline void mul(clsvec<double, 4> const *lhs,
    intrin_type_t<double, 4> const& rhs,
    intrin_type_t<double, 4>& res)
{
    res = _mm256_mul_pd(lhs->data, rhs);
}
#endif

#if defined(__AVX512F__)
// 512-bit double
always_inline void mul(clsvec<double, 8> const *lhs,
    intrin_type_t<double, 8> const& rhs,
    intrin_type_t<double, 8>& res)
{
    res = _mm512_mul_pd(lhs->data, rhs);
}
#endif

// 128-bit int8_t div
always_inline void div(clsvec<int8_t, 16> const *lhs,
    intrin_type_t<int8_t, 16> const& rhs,
    intrin_type_t<int8_t, 16>& res)
{
    res = _mm_setr_epi8(
        (int8_t)_mm_extract_epi8(lhs->data,  0) / (int8_t)_mm_extract_epi8(rhs,  0),
        (int8_t)_mm_extract_epi8(lhs->data,  1) / (int8_t)_mm_extract_epi8(rhs,  1),
        (int8_t)_mm_extract_epi8(lhs->data,  2) / (int8_t)_mm_extract_epi8(rhs,  2),
        (int8_t)_mm_extract_epi8(lhs->data,  3) / (int8_t)_mm_extract_epi8(rhs,  3),
        (int8_t)_mm_extract_epi8(lhs->data,  4) / (int8_t)_mm_extract_epi8(rhs,  4),
        (int8_t)_mm_extract_epi8(lhs->data,  5) / (int8_t)_mm_extract_epi8(rhs,  5),
        (int8_t)_mm_extract_epi8(lhs->data,  6) / (int8_t)_mm_extract_epi8(rhs,  6),
        (int8_t)_mm_extract_epi8(lhs->data,  7) / (int8_t)_mm_extract_epi8(rhs,  7),
        (int8_t)_mm_extract_epi8(lhs->data,  8) / (int8_t)_mm_extract_epi8(rhs,  8),
        (int8_t)_mm_extract_epi8(lhs->data,  9) / (int8_t)_mm_extract_epi8(rhs,  9),
        (int8_t)_mm_extract_epi8(lhs->data, 10) / (int8_t)_mm_extract_epi8(rhs, 10),
        (int8_t)_mm_extract_epi8(lhs->data, 11) / (int8_t)_mm_extract_epi8(rhs, 11),
        (int8_t)_mm_extract_epi8(lhs->data, 12) / (int8_t)_mm_extract_epi8(rhs, 12),
        (int8_t)_mm_extract_epi8(lhs->data, 13) / (int8_t)_mm_extract_epi8(rhs, 13),
        (int8_t)_mm_extract_epi8(lhs->data, 14) / (int8_t)_mm_extract_epi8(rhs, 14),
        (int8_t)_mm_extract_epi8(lhs->data, 15) / (int8_t)_mm_extract_epi8(rhs, 15));
}

#if defined(__AVX2__)
// 256-bit int8_t
always_inline void div(clsvec<int8_t, 32> const *lhs,
    intrin_type_t<int8_t, 32> const& rhs,
    intrin_type_t<int8_t, 32>& res)
{
    intrin_type_t<int8_t, 16> lhs0 = _mm256_extracti128_si256(lhs->data, 0);
    intrin_type_t<int8_t, 16> lhs1 = _mm256_extracti128_si256(lhs->data, 1);
    intrin_type_t<int8_t, 16> rhs0 = _mm256_extracti128_si256(rhs, 0);
    intrin_type_t<int8_t, 16> rhs1 = _mm256_extracti128_si256(rhs, 1);
    res = _mm256_setr_epi8(
        (int8_t)_mm_extract_epi8(lhs0,  0) / (int8_t)_mm_extract_epi8(rhs0,  0),
        (int8_t)_mm_extract_epi8(lhs0,  1) / (int8_t)_mm_extract_epi8(rhs0,  1),
        (int8_t)_mm_extract_epi8(lhs0,  2) / (int8_t)_mm_extract_epi8(rhs0,  2),
        (int8_t)_mm_extract_epi8(lhs0,  3) / (int8_t)_mm_extract_epi8(rhs0,  3),
        (int8_t)_mm_extract_epi8(lhs0,  4) / (int8_t)_mm_extract_epi8(rhs0,  4),
        (int8_t)_mm_extract_epi8(lhs0,  5) / (int8_t)_mm_extract_epi8(rhs0,  5),
        (int8_t)_mm_extract_epi8(lhs0,  6) / (int8_t)_mm_extract_epi8(rhs0,  6),
        (int8_t)_mm_extract_epi8(lhs0,  7) / (int8_t)_mm_extract_epi8(rhs0,  7),
        (int8_t)_mm_extract_epi8(lhs0,  8) / (int8_t)_mm_extract_epi8(rhs0,  8),
        (int8_t)_mm_extract_epi8(lhs0,  9) / (int8_t)_mm_extract_epi8(rhs0,  9),
        (int8_t)_mm_extract_epi8(lhs0, 10) / (int8_t)_mm_extract_epi8(rhs0, 10),
        (int8_t)_mm_extract_epi8(lhs0, 11) / (int8_t)_mm_extract_epi8(rhs0, 11),
        (int8_t)_mm_extract_epi8(lhs0, 12) / (int8_t)_mm_extract_epi8(rhs0, 12),
        (int8_t)_mm_extract_epi8(lhs0, 13) / (int8_t)_mm_extract_epi8(rhs0, 13),
        (int8_t)_mm_extract_epi8(lhs0, 14) / (int8_t)_mm_extract_epi8(rhs0, 14),
        (int8_t)_mm_extract_epi8(lhs0, 15) / (int8_t)_mm_extract_epi8(rhs0, 15),
        (int8_t)_mm_extract_epi8(lhs1,  0) / (int8_t)_mm_extract_epi8(rhs1,  0),
        (int8_t)_mm_extract_epi8(lhs1,  1) / (int8_t)_mm_extract_epi8(rhs1,  1),
        (int8_t)_mm_extract_epi8(lhs1,  2) / (int8_t)_mm_extract_epi8(rhs1,  2),
        (int8_t)_mm_extract_epi8(lhs1,  3) / (int8_t)_mm_extract_epi8(rhs1,  3),
        (int8_t)_mm_extract_epi8(lhs1,  4) / (int8_t)_mm_extract_epi8(rhs1,  4),
        (int8_t)_mm_extract_epi8(lhs1,  5) / (int8_t)_mm_extract_epi8(rhs1,  5),
        (int8_t)_mm_extract_epi8(lhs1,  6) / (int8_t)_mm_extract_epi8(rhs1,  6),
        (int8_t)_mm_extract_epi8(lhs1,  7) / (int8_t)_mm_extract_epi8(rhs1,  7),
        (int8_t)_mm_extract_epi8(lhs1,  8) / (int8_t)_mm_extract_epi8(rhs1,  8),
        (int8_t)_mm_extract_epi8(lhs1,  9) / (int8_t)_mm_extract_epi8(rhs1,  9),
        (int8_t)_mm_extract_epi8(lhs1, 10) / (int8_t)_mm_extract_epi8(rhs1, 10),
        (int8_t)_mm_extract_epi8(lhs1, 11) / (int8_t)_mm_extract_epi8(rhs1, 11),
        (int8_t)_mm_extract_epi8(lhs1, 12) / (int8_t)_mm_extract_epi8(rhs1, 12),
        (int8_t)_mm_extract_epi8(lhs1, 13) / (int8_t)_mm_extract_epi8(rhs1, 13),
        (int8_t)_mm_extract_epi8(lhs1, 14) / (int8_t)_mm_extract_epi8(rhs1, 14),
        (int8_t)_mm_extract_epi8(lhs1, 15) / (int8_t)_mm_extract_epi8(rhs1, 15));
}
#endif

#if defined(__AVX512F__)
always_inline intrin_type_t<int8_t, 64> _mm512_setr_epi8(
    char e63, char e62, char e61, char e60,
    char e59, char e58, char e57, char e56,
    char e55, char e54, char e53, char e52,
    char e51, char e50, char e49, char e48,
    char e47, char e46, char e45, char e44,
    char e43, char e42, char e41, char e40,
    char e39, char e38, char e37, char e36,
    char e35, char e34, char e33, char e32,
    char e31, char e30, char e29, char e28,
    char e27, char e26, char e25, char e24,
    char e23, char e22, char e21, char e20,
    char e19, char e18, char e17, char e16,
    char e15, char e14, char e13, char e12,
    char e11, char e10, char  e9, char  e8,
    char  e7, char  e6, char  e5, char  e4,
    char  e3, char  e2, char  e1, char  e0)
{
    return _mm512_set_epi8(
         e0,  e1,  e2,  e3,
         e4,  e5,  e6,  e7,
         e8,  e9, e10, e11,
        e12, e13, e14, e15,
        e16, e17, e18, e19,
        e20, e21, e22, e23,
        e24, e25, e26, e27,
        e28, e29, e30, e31,
        e32, e33, e34, e35,
        e36, e37, e38, e39,
        e40, e41, e42, e43,
        e44, e45, e46, e47,
        e48, e49, e50, e51,
        e52, e53, e54, e55,
        e56, e57, e58, e59,
        e60, e61, e62, e63);
}

always_inline __m512i _mm512_setr_epi16(
    char e31, char e30, char e29, char e28,
    char e27, char e26, char e25, char e24,
    char e23, char e22, char e21, char e20,
    char e19, char e18, char e17, char e16,
    char e15, char e14, char e13, char e12,
    char e11, char e10, char  e9, char  e8,
    char  e7, char  e6, char  e5, char  e4,
    char  e3, char  e2, char  e1, char  e0)
{
    return _mm512_set_epi16(
         e0,  e1,  e2,  e3,
         e4,  e5,  e6,  e7,
         e8,  e9, e10, e11,
        e12, e13, e14, e15,
        e16, e17, e18, e19,
        e20, e21, e22, e23,
        e24, e25, e26, e27,
        e28, e29, e30, e31);
}
#endif

#if defined(__AVX512F__)
// 512-bit int8_t
always_inline void div(clsvec<int8_t, 64> const *lhs,
    intrin_type_t<int8_t, 64> const& rhs,
    intrin_type_t<int8_t, 64>& res)
{
    intrin_type_t<int8_t, 16> lhs0 = _mm512_extracti32x4_epi32(lhs->data, 0);
    intrin_type_t<int8_t, 16> lhs1 = _mm512_extracti32x4_epi32(lhs->data, 1);
    intrin_type_t<int8_t, 16> lhs2 = _mm512_extracti32x4_epi32(lhs->data, 2);
    intrin_type_t<int8_t, 16> lhs3 = _mm512_extracti32x4_epi32(lhs->data, 3);
    intrin_type_t<int8_t, 16> rhs0 = _mm512_extracti32x4_epi32(rhs, 0);
    intrin_type_t<int8_t, 16> rhs1 = _mm512_extracti32x4_epi32(rhs, 1);
    intrin_type_t<int8_t, 16> rhs2 = _mm512_extracti32x4_epi32(rhs, 2);
    intrin_type_t<int8_t, 16> rhs3 = _mm512_extracti32x4_epi32(rhs, 3);
    res = _mm512_setr_epi8(
        (int8_t)_mm_extract_epi8(lhs0,  0) /
        (int8_t)_mm_extract_epi8(rhs0,  0),
        (int8_t)_mm_extract_epi8(lhs0,  1) /
        (int8_t)_mm_extract_epi8(rhs0,  1),
        (int8_t)_mm_extract_epi8(lhs0,  2) /
        (int8_t)_mm_extract_epi8(rhs0,  2),
        (int8_t)_mm_extract_epi8(lhs0,  3) /
        (int8_t)_mm_extract_epi8(rhs0,  3),
        (int8_t)_mm_extract_epi8(lhs0,  4) /
        (int8_t)_mm_extract_epi8(rhs0,  4),
        (int8_t)_mm_extract_epi8(lhs0,  5) /
        (int8_t)_mm_extract_epi8(rhs0,  5),
        (int8_t)_mm_extract_epi8(lhs0,  6) /
        (int8_t)_mm_extract_epi8(rhs0,  6),
        (int8_t)_mm_extract_epi8(lhs0,  7) /
        (int8_t)_mm_extract_epi8(rhs0,  7),
        (int8_t)_mm_extract_epi8(lhs0,  8) /
        (int8_t)_mm_extract_epi8(rhs0,  8),
        (int8_t)_mm_extract_epi8(lhs0,  9) /
        (int8_t)_mm_extract_epi8(rhs0,  9),
        (int8_t)_mm_extract_epi8(lhs0, 10) /
        (int8_t)_mm_extract_epi8(rhs0, 10),
        (int8_t)_mm_extract_epi8(lhs0, 11) /
        (int8_t)_mm_extract_epi8(rhs0, 11),
        (int8_t)_mm_extract_epi8(lhs0, 12) /
        (int8_t)_mm_extract_epi8(rhs0, 12),
        (int8_t)_mm_extract_epi8(lhs0, 13) /
        (int8_t)_mm_extract_epi8(rhs0, 13),
        (int8_t)_mm_extract_epi8(lhs0, 14) /
        (int8_t)_mm_extract_epi8(rhs0, 14),
        (int8_t)_mm_extract_epi8(lhs0, 15) /
        (int8_t)_mm_extract_epi8(rhs0, 15),
        (int8_t)_mm_extract_epi8(lhs1,  0) /
        (int8_t)_mm_extract_epi8(rhs1,  0),
        (int8_t)_mm_extract_epi8(lhs1,  1) /
        (int8_t)_mm_extract_epi8(rhs1,  1),
        (int8_t)_mm_extract_epi8(lhs1,  2) /
        (int8_t)_mm_extract_epi8(rhs1,  2),
        (int8_t)_mm_extract_epi8(lhs1,  3) /
        (int8_t)_mm_extract_epi8(rhs1,  3),
        (int8_t)_mm_extract_epi8(lhs1,  4) /
        (int8_t)_mm_extract_epi8(rhs1,  4),
        (int8_t)_mm_extract_epi8(lhs1,  5) /
        (int8_t)_mm_extract_epi8(rhs1,  5),
        (int8_t)_mm_extract_epi8(lhs1,  6) /
        (int8_t)_mm_extract_epi8(rhs1,  6),
        (int8_t)_mm_extract_epi8(lhs1,  7) /
        (int8_t)_mm_extract_epi8(rhs1,  7),
        (int8_t)_mm_extract_epi8(lhs1,  8) /
        (int8_t)_mm_extract_epi8(rhs1,  8),
        (int8_t)_mm_extract_epi8(lhs1,  9) /
        (int8_t)_mm_extract_epi8(rhs1,  9),
        (int8_t)_mm_extract_epi8(lhs1, 10) /
        (int8_t)_mm_extract_epi8(rhs1, 10),
        (int8_t)_mm_extract_epi8(lhs1, 11) /
        (int8_t)_mm_extract_epi8(rhs1, 11),
        (int8_t)_mm_extract_epi8(lhs1, 12) /
        (int8_t)_mm_extract_epi8(rhs1, 12),
        (int8_t)_mm_extract_epi8(lhs1, 13) /
        (int8_t)_mm_extract_epi8(rhs1, 13),
        (int8_t)_mm_extract_epi8(lhs1, 14) /
        (int8_t)_mm_extract_epi8(rhs1, 14),
        (int8_t)_mm_extract_epi8(lhs1, 15) /
        (int8_t)_mm_extract_epi8(rhs1, 15),
        (int8_t)_mm_extract_epi8(lhs2,  0) /
        (int8_t)_mm_extract_epi8(rhs2,  0),
        (int8_t)_mm_extract_epi8(lhs2,  1) /
        (int8_t)_mm_extract_epi8(rhs2,  1),
        (int8_t)_mm_extract_epi8(lhs2,  2) /
        (int8_t)_mm_extract_epi8(rhs2,  2),
        (int8_t)_mm_extract_epi8(lhs2,  3) /
        (int8_t)_mm_extract_epi8(rhs2,  3),
        (int8_t)_mm_extract_epi8(lhs2,  4) /
        (int8_t)_mm_extract_epi8(rhs2,  4),
        (int8_t)_mm_extract_epi8(lhs2,  5) /
        (int8_t)_mm_extract_epi8(rhs2,  5),
        (int8_t)_mm_extract_epi8(lhs2,  6) /
        (int8_t)_mm_extract_epi8(rhs2,  6),
        (int8_t)_mm_extract_epi8(lhs2,  7) /
        (int8_t)_mm_extract_epi8(rhs2,  7),
        (int8_t)_mm_extract_epi8(lhs2,  8) /
        (int8_t)_mm_extract_epi8(rhs2,  8),
        (int8_t)_mm_extract_epi8(lhs2,  9) /
        (int8_t)_mm_extract_epi8(rhs2,  9),
        (int8_t)_mm_extract_epi8(lhs2, 10) /
        (int8_t)_mm_extract_epi8(rhs2, 10),
        (int8_t)_mm_extract_epi8(lhs2, 11) /
        (int8_t)_mm_extract_epi8(rhs2, 11),
        (int8_t)_mm_extract_epi8(lhs2, 12) /
        (int8_t)_mm_extract_epi8(rhs2, 12),
        (int8_t)_mm_extract_epi8(lhs2, 13) /
        (int8_t)_mm_extract_epi8(rhs2, 13),
        (int8_t)_mm_extract_epi8(lhs2, 14) /
        (int8_t)_mm_extract_epi8(rhs2, 14),
        (int8_t)_mm_extract_epi8(lhs2, 15) /
        (int8_t)_mm_extract_epi8(rhs2, 15),
        (int8_t)_mm_extract_epi8(lhs3,  0) /
        (int8_t)_mm_extract_epi8(rhs3,  0),
        (int8_t)_mm_extract_epi8(lhs3,  1) /
        (int8_t)_mm_extract_epi8(rhs3,  1),
        (int8_t)_mm_extract_epi8(lhs3,  2) /
        (int8_t)_mm_extract_epi8(rhs3,  2),
        (int8_t)_mm_extract_epi8(lhs3,  3) /
        (int8_t)_mm_extract_epi8(rhs3,  3),
        (int8_t)_mm_extract_epi8(lhs3,  4) /
        (int8_t)_mm_extract_epi8(rhs3,  4),
        (int8_t)_mm_extract_epi8(lhs3,  5) /
        (int8_t)_mm_extract_epi8(rhs3,  5),
        (int8_t)_mm_extract_epi8(lhs3,  6) /
        (int8_t)_mm_extract_epi8(rhs3,  6),
        (int8_t)_mm_extract_epi8(lhs3,  7) /
        (int8_t)_mm_extract_epi8(rhs3,  7),
        (int8_t)_mm_extract_epi8(lhs3,  8) /
        (int8_t)_mm_extract_epi8(rhs3,  8),
        (int8_t)_mm_extract_epi8(lhs3,  9) /
        (int8_t)_mm_extract_epi8(rhs3,  9),
        (int8_t)_mm_extract_epi8(lhs3, 10) /
        (int8_t)_mm_extract_epi8(rhs3, 10),
        (int8_t)_mm_extract_epi8(lhs3, 11) /
        (int8_t)_mm_extract_epi8(rhs3, 11),
        (int8_t)_mm_extract_epi8(lhs3, 12) /
        (int8_t)_mm_extract_epi8(rhs3, 12),
        (int8_t)_mm_extract_epi8(lhs3, 13) /
        (int8_t)_mm_extract_epi8(rhs3, 13),
        (int8_t)_mm_extract_epi8(lhs3, 14) /
        (int8_t)_mm_extract_epi8(rhs3, 14),
        (int8_t)_mm_extract_epi8(lhs3, 15) /
        (int8_t)_mm_extract_epi8(rhs3, 15));
}
#endif

// 128-bit uint8_t
always_inline void div(clsvec<uint8_t, 16> const *lhs,
    intrin_type_t<uint8_t, 16> const& rhs,
    intrin_type_t<uint8_t, 16>& res)
{
    intrin_type_t<uint8_t, 16> lhs0 = lhs->data;
    intrin_type_t<uint8_t, 16> rhs0 = rhs;
    res = _mm_setr_epi8(
        (char)((uint8_t)_mm_extract_epi8(lhs0,  0) /
               (uint8_t)_mm_extract_epi8(rhs0,  0)),
        (char)((uint8_t)_mm_extract_epi8(lhs0,  1) /
               (uint8_t)_mm_extract_epi8(rhs0,  1)),
        (char)((uint8_t)_mm_extract_epi8(lhs0,  2) /
               (uint8_t)_mm_extract_epi8(rhs0,  2)),
        (char)((uint8_t)_mm_extract_epi8(lhs0,  3) /
               (uint8_t)_mm_extract_epi8(rhs0,  3)),
        (char)((uint8_t)_mm_extract_epi8(lhs0,  4) /
               (uint8_t)_mm_extract_epi8(rhs0,  4)),
        (char)((uint8_t)_mm_extract_epi8(lhs0,  5) /
               (uint8_t)_mm_extract_epi8(rhs0,  5)),
        (char)((uint8_t)_mm_extract_epi8(lhs0,  6) /
               (uint8_t)_mm_extract_epi8(rhs0,  6)),
        (char)((uint8_t)_mm_extract_epi8(lhs0,  7) /
               (uint8_t)_mm_extract_epi8(rhs0,  7)),
        (char)((uint8_t)_mm_extract_epi8(lhs0,  8) /
               (uint8_t)_mm_extract_epi8(rhs0,  8)),
        (char)((uint8_t)_mm_extract_epi8(lhs0,  9) /
               (uint8_t)_mm_extract_epi8(rhs0,  9)),
        (char)((uint8_t)_mm_extract_epi8(lhs0, 10) /
               (uint8_t)_mm_extract_epi8(rhs0, 10)),
        (char)((uint8_t)_mm_extract_epi8(lhs0, 11) /
               (uint8_t)_mm_extract_epi8(rhs0, 11)),
        (char)((uint8_t)_mm_extract_epi8(lhs0, 12) /
               (uint8_t)_mm_extract_epi8(rhs0, 12)),
        (char)((uint8_t)_mm_extract_epi8(lhs0, 13) /
               (uint8_t)_mm_extract_epi8(rhs0, 13)),
        (char)((uint8_t)_mm_extract_epi8(lhs0, 14) /
               (uint8_t)_mm_extract_epi8(rhs0, 14)),
        (char)((uint8_t)_mm_extract_epi8(lhs0, 15) /
               (uint8_t)_mm_extract_epi8(rhs0, 15)));
}

#if defined(__AVX2__)
// 256-bit uint8_t
always_inline void div(clsvec<uint8_t, 32> const *lhs,
    intrin_type_t<uint8_t, 32> const& rhs,
    intrin_type_t<uint8_t, 32>& res)
{
    intrin_type_t<uint8_t, 16> lhs0 =
        _mm256_extracti128_si256(lhs->data, 0);
    intrin_type_t<uint8_t, 16> lhs1 =
        _mm256_extracti128_si256(lhs->data, 1);
    intrin_type_t<uint8_t, 16> rhs0 =
        _mm256_extracti128_si256(rhs, 0);
    intrin_type_t<uint8_t, 16> rhs1 =
        _mm256_extracti128_si256(rhs, 1);
    res = _mm256_setr_epi8(
        (char)((uint8_t)_mm_extract_epi8(lhs0,  0) /
               (uint8_t)_mm_extract_epi8(rhs0,  0)),
        (char)((uint8_t)_mm_extract_epi8(lhs0,  1) /
               (uint8_t)_mm_extract_epi8(rhs0,  1)),
        (char)((uint8_t)_mm_extract_epi8(lhs0,  2) /
               (uint8_t)_mm_extract_epi8(rhs0,  2)),
        (char)((uint8_t)_mm_extract_epi8(lhs0,  3) /
               (uint8_t)_mm_extract_epi8(rhs0,  3)),
        (char)((uint8_t)_mm_extract_epi8(lhs0,  4) /
               (uint8_t)_mm_extract_epi8(rhs0,  4)),
        (char)((uint8_t)_mm_extract_epi8(lhs0,  5) /
               (uint8_t)_mm_extract_epi8(rhs0,  5)),
        (char)((uint8_t)_mm_extract_epi8(lhs0,  6) /
               (uint8_t)_mm_extract_epi8(rhs0,  6)),
        (char)((uint8_t)_mm_extract_epi8(lhs0,  7) /
               (uint8_t)_mm_extract_epi8(rhs0,  7)),
        (char)((uint8_t)_mm_extract_epi8(lhs0,  8) /
               (uint8_t)_mm_extract_epi8(rhs0,  8)),
        (char)((uint8_t)_mm_extract_epi8(lhs0,  9) /
               (uint8_t)_mm_extract_epi8(rhs0,  9)),
        (char)((uint8_t)_mm_extract_epi8(lhs0, 10) /
               (uint8_t)_mm_extract_epi8(rhs0, 10)),
        (char)((uint8_t)_mm_extract_epi8(lhs0, 11) /
               (uint8_t)_mm_extract_epi8(rhs0, 11)),
        (char)((uint8_t)_mm_extract_epi8(lhs0, 12) /
               (uint8_t)_mm_extract_epi8(rhs0, 12)),
        (char)((uint8_t)_mm_extract_epi8(lhs0, 13) /
               (uint8_t)_mm_extract_epi8(rhs0, 13)),
        (char)((uint8_t)_mm_extract_epi8(lhs0, 14) /
               (uint8_t)_mm_extract_epi8(rhs0, 14)),
        (char)((uint8_t)_mm_extract_epi8(lhs0, 15) /
               (uint8_t)_mm_extract_epi8(rhs0, 15)),
        (char)((uint8_t)_mm_extract_epi8(lhs1,  0) /
               (uint8_t)_mm_extract_epi8(rhs1,  0)),
        (char)((uint8_t)_mm_extract_epi8(lhs1,  1) /
               (uint8_t)_mm_extract_epi8(rhs1,  1)),
        (char)((uint8_t)_mm_extract_epi8(lhs1,  2) /
               (uint8_t)_mm_extract_epi8(rhs1,  2)),
        (char)((uint8_t)_mm_extract_epi8(lhs1,  3) /
               (uint8_t)_mm_extract_epi8(rhs1,  3)),
        (char)((uint8_t)_mm_extract_epi8(lhs1,  4) /
               (uint8_t)_mm_extract_epi8(rhs1,  4)),
        (char)((uint8_t)_mm_extract_epi8(lhs1,  5) /
               (uint8_t)_mm_extract_epi8(rhs1,  5)),
        (char)((uint8_t)_mm_extract_epi8(lhs1,  6) /
               (uint8_t)_mm_extract_epi8(rhs1,  6)),
        (char)((uint8_t)_mm_extract_epi8(lhs1,  7) /
               (uint8_t)_mm_extract_epi8(rhs1,  7)),
        (char)((uint8_t)_mm_extract_epi8(lhs1,  8) /
               (uint8_t)_mm_extract_epi8(rhs1,  8)),
        (char)((uint8_t)_mm_extract_epi8(lhs1,  9) /
               (uint8_t)_mm_extract_epi8(rhs1,  9)),
        (char)((uint8_t)_mm_extract_epi8(lhs1, 10) /
               (uint8_t)_mm_extract_epi8(rhs1, 10)),
        (char)((uint8_t)_mm_extract_epi8(lhs1, 11) /
               (uint8_t)_mm_extract_epi8(rhs1, 11)),
        (char)((uint8_t)_mm_extract_epi8(lhs1, 12) /
               (uint8_t)_mm_extract_epi8(rhs1, 12)),
        (char)((uint8_t)_mm_extract_epi8(lhs1, 13) /
               (uint8_t)_mm_extract_epi8(rhs1, 13)),
        (char)((uint8_t)_mm_extract_epi8(lhs1, 14) /
               (uint8_t)_mm_extract_epi8(rhs1, 14)),
        (char)((uint8_t)_mm_extract_epi8(lhs1, 15) /
               (uint8_t)_mm_extract_epi8(rhs1, 15)));
}
#endif

#if defined(__AVX512F__)
// 512-bit uint8_t
always_inline void div(clsvec<uint8_t, 64> const *lhs,
    intrin_type_t<uint8_t, 64> const& rhs,
    intrin_type_t<uint8_t, 64>& res)
{
    intrin_type_t<uint8_t, 16> lhs0 =
        _mm512_extracti32x4_epi32(lhs->data, 0);
    intrin_type_t<uint8_t, 16> lhs1 =
        _mm512_extracti32x4_epi32(lhs->data, 1);
    intrin_type_t<uint8_t, 16> lhs2 =
        _mm512_extracti32x4_epi32(lhs->data, 2);
    intrin_type_t<uint8_t, 16> lhs3 =
        _mm512_extracti32x4_epi32(lhs->data, 3);
    intrin_type_t<int8_t, 16> rhs0 =
        _mm512_extracti32x4_epi32(rhs, 0);
    intrin_type_t<int8_t, 16> rhs1 =
        _mm512_extracti32x4_epi32(rhs, 1);
    intrin_type_t<int8_t, 16> rhs2 =
        _mm512_extracti32x4_epi32(rhs, 2);
    intrin_type_t<int8_t, 16> rhs3 =
        _mm512_extracti32x4_epi32(rhs, 3);
    res = _mm512_setr_epi8(
        (char)((uint8_t)_mm_extract_epi8(lhs0,  0) /
               (uint8_t)_mm_extract_epi8(rhs0,  0)),
        (char)((uint8_t)_mm_extract_epi8(lhs0,  1) /
               (uint8_t)_mm_extract_epi8(rhs0,  1)),
        (char)((uint8_t)_mm_extract_epi8(lhs0,  2) /
               (uint8_t)_mm_extract_epi8(rhs0,  2)),
        (char)((uint8_t)_mm_extract_epi8(lhs0,  3) /
               (uint8_t)_mm_extract_epi8(rhs0,  3)),
        (char)((uint8_t)_mm_extract_epi8(lhs0,  4) /
               (uint8_t)_mm_extract_epi8(rhs0,  4)),
        (char)((uint8_t)_mm_extract_epi8(lhs0,  5) /
               (uint8_t)_mm_extract_epi8(rhs0,  5)),
        (char)((uint8_t)_mm_extract_epi8(lhs0,  6) /
               (uint8_t)_mm_extract_epi8(rhs0,  6)),
        (char)((uint8_t)_mm_extract_epi8(lhs0,  7) /
               (uint8_t)_mm_extract_epi8(rhs0,  7)),
        (char)((uint8_t)_mm_extract_epi8(lhs0,  8) /
               (uint8_t)_mm_extract_epi8(rhs0,  8)),
        (char)((uint8_t)_mm_extract_epi8(lhs0,  9) /
               (uint8_t)_mm_extract_epi8(rhs0,  9)),
        (char)((uint8_t)_mm_extract_epi8(lhs0, 10) /
               (uint8_t)_mm_extract_epi8(rhs0, 10)),
        (char)((uint8_t)_mm_extract_epi8(lhs0, 11) /
               (uint8_t)_mm_extract_epi8(rhs0, 11)),
        (char)((uint8_t)_mm_extract_epi8(lhs0, 12) /
               (uint8_t)_mm_extract_epi8(rhs0, 12)),
        (char)((uint8_t)_mm_extract_epi8(lhs0, 13) /
               (uint8_t)_mm_extract_epi8(rhs0, 13)),
        (char)((uint8_t)_mm_extract_epi8(lhs0, 14) /
               (uint8_t)_mm_extract_epi8(rhs0, 14)),
        (char)((uint8_t)_mm_extract_epi8(lhs0, 15) /
               (uint8_t)_mm_extract_epi8(rhs0, 15)),
        (char)((uint8_t)_mm_extract_epi8(lhs1,  0) /
               (uint8_t)_mm_extract_epi8(rhs1,  0)),
        (char)((uint8_t)_mm_extract_epi8(lhs1,  1) /
               (uint8_t)_mm_extract_epi8(rhs1,  1)),
        (char)((uint8_t)_mm_extract_epi8(lhs1,  2) /
               (uint8_t)_mm_extract_epi8(rhs1,  2)),
        (char)((uint8_t)_mm_extract_epi8(lhs1,  3) /
               (uint8_t)_mm_extract_epi8(rhs1,  3)),
        (char)((uint8_t)_mm_extract_epi8(lhs1,  4) /
               (uint8_t)_mm_extract_epi8(rhs1,  4)),
        (char)((uint8_t)_mm_extract_epi8(lhs1,  5) /
               (uint8_t)_mm_extract_epi8(rhs1,  5)),
        (char)((uint8_t)_mm_extract_epi8(lhs1,  6) /
               (uint8_t)_mm_extract_epi8(rhs1,  6)),
        (char)((uint8_t)_mm_extract_epi8(lhs1,  7) /
               (uint8_t)_mm_extract_epi8(rhs1,  7)),
        (char)((uint8_t)_mm_extract_epi8(lhs1,  8) /
               (uint8_t)_mm_extract_epi8(rhs1,  8)),
        (char)((uint8_t)_mm_extract_epi8(lhs1,  9) /
               (uint8_t)_mm_extract_epi8(rhs1,  9)),
        (char)((uint8_t)_mm_extract_epi8(lhs1, 10) /
               (uint8_t)_mm_extract_epi8(rhs1, 10)),
        (char)((uint8_t)_mm_extract_epi8(lhs1, 11) /
               (uint8_t)_mm_extract_epi8(rhs1, 11)),
        (char)((uint8_t)_mm_extract_epi8(lhs1, 12) /
               (uint8_t)_mm_extract_epi8(rhs1, 12)),
        (char)((uint8_t)_mm_extract_epi8(lhs1, 13) /
               (uint8_t)_mm_extract_epi8(rhs1, 13)),
        (char)((uint8_t)_mm_extract_epi8(lhs1, 14) /
               (uint8_t)_mm_extract_epi8(rhs1, 14)),
        (char)((uint8_t)_mm_extract_epi8(lhs1, 15) /
               (uint8_t)_mm_extract_epi8(rhs1, 15)),
        (char)((uint8_t)_mm_extract_epi8(lhs2,  0) /
               (uint8_t)_mm_extract_epi8(rhs2,  0)),
        (char)((uint8_t)_mm_extract_epi8(lhs2,  1) /
               (uint8_t)_mm_extract_epi8(rhs2,  1)),
        (char)((uint8_t)_mm_extract_epi8(lhs2,  2) /
               (uint8_t)_mm_extract_epi8(rhs2,  2)),
        (char)((uint8_t)_mm_extract_epi8(lhs2,  3) /
               (uint8_t)_mm_extract_epi8(rhs2,  3)),
        (char)((uint8_t)_mm_extract_epi8(lhs2,  4) /
               (uint8_t)_mm_extract_epi8(rhs2,  4)),
        (char)((uint8_t)_mm_extract_epi8(lhs2,  5) /
               (uint8_t)_mm_extract_epi8(rhs2,  5)),
        (char)((uint8_t)_mm_extract_epi8(lhs2,  6) /
               (uint8_t)_mm_extract_epi8(rhs2,  6)),
        (char)((uint8_t)_mm_extract_epi8(lhs2,  7) /
               (uint8_t)_mm_extract_epi8(rhs2,  7)),
        (char)((uint8_t)_mm_extract_epi8(lhs2,  8) /
               (uint8_t)_mm_extract_epi8(rhs2,  8)),
        (char)((uint8_t)_mm_extract_epi8(lhs2,  9) /
               (uint8_t)_mm_extract_epi8(rhs2,  9)),
        (char)((uint8_t)_mm_extract_epi8(lhs2, 10) /
               (uint8_t)_mm_extract_epi8(rhs2, 10)),
        (char)((uint8_t)_mm_extract_epi8(lhs2, 11) /
               (uint8_t)_mm_extract_epi8(rhs2, 11)),
        (char)((uint8_t)_mm_extract_epi8(lhs2, 12) /
               (uint8_t)_mm_extract_epi8(rhs2, 12)),
        (char)((uint8_t)_mm_extract_epi8(lhs2, 13) /
               (uint8_t)_mm_extract_epi8(rhs2, 13)),
        (char)((uint8_t)_mm_extract_epi8(lhs2, 14) /
               (uint8_t)_mm_extract_epi8(rhs2, 14)),
        (char)((uint8_t)_mm_extract_epi8(lhs2, 15) /
               (uint8_t)_mm_extract_epi8(rhs2, 15)),
        (char)((uint8_t)_mm_extract_epi8(lhs3,  0) /
               (uint8_t)_mm_extract_epi8(rhs3,  0)),
        (char)((uint8_t)_mm_extract_epi8(lhs3,  1) /
               (uint8_t)_mm_extract_epi8(rhs3,  1)),
        (char)((uint8_t)_mm_extract_epi8(lhs3,  2) /
               (uint8_t)_mm_extract_epi8(rhs3,  2)),
        (char)((uint8_t)_mm_extract_epi8(lhs3,  3) /
               (uint8_t)_mm_extract_epi8(rhs3,  3)),
        (char)((uint8_t)_mm_extract_epi8(lhs3,  4) /
               (uint8_t)_mm_extract_epi8(rhs3,  4)),
        (char)((uint8_t)_mm_extract_epi8(lhs3,  5) /
               (uint8_t)_mm_extract_epi8(rhs3,  5)),
        (char)((uint8_t)_mm_extract_epi8(lhs3,  6) /
               (uint8_t)_mm_extract_epi8(rhs3,  6)),
        (char)((uint8_t)_mm_extract_epi8(lhs3,  7) /
               (uint8_t)_mm_extract_epi8(rhs3,  7)),
        (char)((uint8_t)_mm_extract_epi8(lhs3,  8) /
               (uint8_t)_mm_extract_epi8(rhs3,  8)),
        (char)((uint8_t)_mm_extract_epi8(lhs3,  9) /
               (uint8_t)_mm_extract_epi8(rhs3,  9)),
        (char)((uint8_t)_mm_extract_epi8(lhs3, 10) /
               (uint8_t)_mm_extract_epi8(rhs3, 10)),
        (char)((uint8_t)_mm_extract_epi8(lhs3, 11) /
               (uint8_t)_mm_extract_epi8(rhs3, 11)),
        (char)((uint8_t)_mm_extract_epi8(lhs3, 12) /
               (uint8_t)_mm_extract_epi8(rhs3, 12)),
        (char)((uint8_t)_mm_extract_epi8(lhs3, 13) /
               (uint8_t)_mm_extract_epi8(rhs3, 13)),
        (char)((uint8_t)_mm_extract_epi8(lhs3, 14) /
               (uint8_t)_mm_extract_epi8(rhs3, 14)),
        (char)((uint8_t)_mm_extract_epi8(lhs3, 15) /
               (uint8_t)_mm_extract_epi8(rhs3, 15)));
}
#endif

// 128-bit int16_t
always_inline void div(clsvec<int16_t, 8> const *lhs,
    intrin_type_t<int16_t, 8> const& rhs,
    intrin_type_t<int16_t, 8>& res)
{
    intrin_type_t<int16_t, 8> lhs0 = lhs->data;
    res = _mm_setr_epi16(
        _mm_extract_epi16(lhs0, 0) / _mm_extract_epi16(rhs, 0),
        _mm_extract_epi16(lhs0, 1) / _mm_extract_epi16(rhs, 1),
        _mm_extract_epi16(lhs0, 2) / _mm_extract_epi16(rhs, 2),
        _mm_extract_epi16(lhs0, 3) / _mm_extract_epi16(rhs, 3),
        _mm_extract_epi16(lhs0, 4) / _mm_extract_epi16(rhs, 4),
        _mm_extract_epi16(lhs0, 5) / _mm_extract_epi16(rhs, 5),
        _mm_extract_epi16(lhs0, 6) / _mm_extract_epi16(rhs, 6),
        _mm_extract_epi16(lhs0, 7) / _mm_extract_epi16(rhs, 7));
}

#if defined(__AVX2__)
// 256-bit int16_t
always_inline void div(clsvec<int16_t, 16> const *lhs,
    intrin_type_t<int16_t, 16> const& rhs,
    intrin_type_t<int16_t, 16>& res)
{
    intrin_type_t<int16_t, 8> const& lhs0 =
        _mm256_extracti128_si256(lhs->data, 0);
    intrin_type_t<int16_t, 8> const& lhs1 =
        _mm256_extracti128_si256(lhs->data, 1);
    intrin_type_t<int16_t, 8> const& rhs0 =
        _mm256_extracti128_si256(rhs, 0);
    intrin_type_t<int16_t, 8> const& rhs1 =
        _mm256_extracti128_si256(rhs, 1);
    res = _mm256_setr_epi16(
        _mm_extract_epi16(lhs0, 0) / _mm_extract_epi16(rhs0, 0),
        _mm_extract_epi16(lhs0, 1) / _mm_extract_epi16(rhs0, 1),
        _mm_extract_epi16(lhs0, 2) / _mm_extract_epi16(rhs0, 2),
        _mm_extract_epi16(lhs0, 3) / _mm_extract_epi16(rhs0, 3),
        _mm_extract_epi16(lhs0, 4) / _mm_extract_epi16(rhs0, 4),
        _mm_extract_epi16(lhs0, 5) / _mm_extract_epi16(rhs0, 5),
        _mm_extract_epi16(lhs0, 6) / _mm_extract_epi16(rhs0, 6),
        _mm_extract_epi16(lhs0, 7) / _mm_extract_epi16(rhs0, 7),
        _mm_extract_epi16(lhs1, 0) / _mm_extract_epi16(rhs1, 0),
        _mm_extract_epi16(lhs1, 1) / _mm_extract_epi16(rhs1, 1),
        _mm_extract_epi16(lhs1, 2) / _mm_extract_epi16(rhs1, 2),
        _mm_extract_epi16(lhs1, 3) / _mm_extract_epi16(rhs1, 3),
        _mm_extract_epi16(lhs1, 4) / _mm_extract_epi16(rhs1, 4),
        _mm_extract_epi16(lhs1, 5) / _mm_extract_epi16(rhs1, 5),
        _mm_extract_epi16(lhs1, 6) / _mm_extract_epi16(rhs1, 6),
        _mm_extract_epi16(lhs1, 7) / _mm_extract_epi16(rhs1, 7));
}
#endif

#if defined(__AVX512F__)
// 512-bit int16_t
always_inline void div(clsvec<int16_t, 32> const *lhs,
    intrin_type_t<int16_t, 32> const& rhs,
    intrin_type_t<int16_t, 32>& res)
{
    intrin_type_t<int16_t, 8> const& lhs0 =
        _mm512_extracti32x4_epi32(lhs->data, 0);
    intrin_type_t<int16_t, 8> const& lhs1 =
        _mm512_extracti32x4_epi32(lhs->data, 1);
    intrin_type_t<int16_t, 8> const& lhs2 =
        _mm512_extracti32x4_epi32(lhs->data, 2);
    intrin_type_t<int16_t, 8> const& lhs3 =
        _mm512_extracti32x4_epi32(lhs->data, 3);
    intrin_type_t<int16_t, 8> const& rhs0 =
        _mm512_extracti32x4_epi32(rhs, 0);
    intrin_type_t<int16_t, 8> const& rhs1 =
        _mm512_extracti32x4_epi32(rhs, 1);
    intrin_type_t<int16_t, 8> const& rhs2 =
        _mm512_extracti32x4_epi32(rhs, 2);
    intrin_type_t<int16_t, 8> const& rhs3 =
        _mm512_extracti32x4_epi32(rhs, 3);
    res = _mm512_setr_epi16(
        _mm_extract_epi16(lhs0, 0) / _mm_extract_epi16(rhs0, 0),
        _mm_extract_epi16(lhs0, 1) / _mm_extract_epi16(rhs0, 1),
        _mm_extract_epi16(lhs0, 2) / _mm_extract_epi16(rhs0, 2),
        _mm_extract_epi16(lhs0, 3) / _mm_extract_epi16(rhs0, 3),
        _mm_extract_epi16(lhs0, 4) / _mm_extract_epi16(rhs0, 4),
        _mm_extract_epi16(lhs0, 5) / _mm_extract_epi16(rhs0, 5),
        _mm_extract_epi16(lhs0, 6) / _mm_extract_epi16(rhs0, 6),
        _mm_extract_epi16(lhs0, 7) / _mm_extract_epi16(rhs0, 7),
        _mm_extract_epi16(lhs1, 0) / _mm_extract_epi16(rhs1, 0),
        _mm_extract_epi16(lhs1, 1) / _mm_extract_epi16(rhs1, 1),
        _mm_extract_epi16(lhs1, 2) / _mm_extract_epi16(rhs1, 2),
        _mm_extract_epi16(lhs1, 3) / _mm_extract_epi16(rhs1, 3),
        _mm_extract_epi16(lhs1, 4) / _mm_extract_epi16(rhs1, 4),
        _mm_extract_epi16(lhs1, 5) / _mm_extract_epi16(rhs1, 5),
        _mm_extract_epi16(lhs1, 6) / _mm_extract_epi16(rhs1, 6),
        _mm_extract_epi16(lhs1, 7) / _mm_extract_epi16(rhs1, 7),
        _mm_extract_epi16(lhs2, 0) / _mm_extract_epi16(rhs2, 0),
        _mm_extract_epi16(lhs2, 1) / _mm_extract_epi16(rhs2, 1),
        _mm_extract_epi16(lhs2, 2) / _mm_extract_epi16(rhs2, 2),
        _mm_extract_epi16(lhs2, 3) / _mm_extract_epi16(rhs2, 3),
        _mm_extract_epi16(lhs2, 4) / _mm_extract_epi16(rhs2, 4),
        _mm_extract_epi16(lhs2, 5) / _mm_extract_epi16(rhs2, 5),
        _mm_extract_epi16(lhs2, 6) / _mm_extract_epi16(rhs2, 6),
        _mm_extract_epi16(lhs2, 7) / _mm_extract_epi16(rhs2, 7),
        _mm_extract_epi16(lhs3, 0) / _mm_extract_epi16(rhs3, 0),
        _mm_extract_epi16(lhs3, 1) / _mm_extract_epi16(rhs3, 1),
        _mm_extract_epi16(lhs3, 2) / _mm_extract_epi16(rhs3, 2),
        _mm_extract_epi16(lhs3, 3) / _mm_extract_epi16(rhs3, 3),
        _mm_extract_epi16(lhs3, 4) / _mm_extract_epi16(rhs3, 4),
        _mm_extract_epi16(lhs3, 5) / _mm_extract_epi16(rhs3, 5),
        _mm_extract_epi16(lhs3, 6) / _mm_extract_epi16(rhs3, 6),
        _mm_extract_epi16(lhs3, 7) / _mm_extract_epi16(rhs3, 7));
}
#endif

// 128-bit uint16_t
always_inline void div(clsvec<uint16_t, 8> const *lhs,
    intrin_type_t<uint16_t, 8> const& rhs,
    intrin_type_t<uint16_t, 8>& res)
{
    intrin_type_t<uint16_t, 8> const& lhs0 = lhs->data;
    res = _mm_setr_epi16(
        _mm_extract_epi16(lhs0, 0) / _mm_extract_epi16(rhs, 0),
        _mm_extract_epi16(lhs0, 1) / _mm_extract_epi16(rhs, 1),
        _mm_extract_epi16(lhs0, 2) / _mm_extract_epi16(rhs, 2),
        _mm_extract_epi16(lhs0, 3) / _mm_extract_epi16(rhs, 3),
        _mm_extract_epi16(lhs0, 4) / _mm_extract_epi16(rhs, 4),
        _mm_extract_epi16(lhs0, 5) / _mm_extract_epi16(rhs, 5),
        _mm_extract_epi16(lhs0, 6) / _mm_extract_epi16(rhs, 6),
        _mm_extract_epi16(lhs0, 7) / _mm_extract_epi16(rhs, 7));
}

#if defined(__AVX2__)
// 256-bit uint16_t
always_inline void div(clsvec<uint16_t, 16> const *lhs,
    intrin_type_t<uint16_t, 16> const& rhs,
    intrin_type_t<uint16_t, 16>& res)
{
    intrin_type_t<uint16_t, 8> lhs0 =
        _mm256_extracti128_si256(lhs->data, 0);
    intrin_type_t<uint16_t, 8> lhs1 =
        _mm256_extracti128_si256(lhs->data, 1);
    intrin_type_t<uint16_t, 8> rhs0 =
        _mm256_extracti128_si256(rhs, 0);
    intrin_type_t<uint16_t, 8> rhs1 =
        _mm256_extracti128_si256(rhs, 1);
    res = _mm256_setr_epi16(
        _mm_extract_epi16(lhs0, 0) / _mm_extract_epi16(rhs0, 0),
        _mm_extract_epi16(lhs0, 1) / _mm_extract_epi16(rhs0, 1),
        _mm_extract_epi16(lhs0, 2) / _mm_extract_epi16(rhs0, 2),
        _mm_extract_epi16(lhs0, 3) / _mm_extract_epi16(rhs0, 3),
        _mm_extract_epi16(lhs0, 4) / _mm_extract_epi16(rhs0, 4),
        _mm_extract_epi16(lhs0, 5) / _mm_extract_epi16(rhs0, 5),
        _mm_extract_epi16(lhs0, 6) / _mm_extract_epi16(rhs0, 6),
        _mm_extract_epi16(lhs0, 7) / _mm_extract_epi16(rhs0, 7),
        _mm_extract_epi16(lhs1, 0) / _mm_extract_epi16(rhs1, 0),
        _mm_extract_epi16(lhs1, 1) / _mm_extract_epi16(rhs1, 1),
        _mm_extract_epi16(lhs1, 2) / _mm_extract_epi16(rhs1, 2),
        _mm_extract_epi16(lhs1, 3) / _mm_extract_epi16(rhs1, 3),
        _mm_extract_epi16(lhs1, 4) / _mm_extract_epi16(rhs1, 4),
        _mm_extract_epi16(lhs1, 5) / _mm_extract_epi16(rhs1, 5),
        _mm_extract_epi16(lhs1, 6) / _mm_extract_epi16(rhs1, 6),
        _mm_extract_epi16(lhs1, 7) / _mm_extract_epi16(rhs1, 7));
}
#endif

#if defined(__AVX512F__)
// 512-bit uint16_t
always_inline void div(clsvec<uint16_t, 32> const *lhs,
    intrin_type_t<uint16_t, 32> const& rhs,
    intrin_type_t<uint16_t, 32>& res)
{
    intrin_type_t<uint16_t, 8> lhs0 =
        _mm512_extracti32x4_epi32(lhs->data, 0);
    intrin_type_t<uint16_t, 8> lhs1 =
        _mm512_extracti32x4_epi32(lhs->data, 1);
    intrin_type_t<uint16_t, 8> lhs2 =
        _mm512_extracti32x4_epi32(lhs->data, 2);
    intrin_type_t<uint16_t, 8> lhs3 =
        _mm512_extracti32x4_epi32(lhs->data, 3);
    intrin_type_t<uint16_t, 8> rhs0 =
        _mm512_extracti32x4_epi32(rhs, 0);
    intrin_type_t<uint16_t, 8> rhs1 =
        _mm512_extracti32x4_epi32(rhs, 1);
    intrin_type_t<uint16_t, 8> rhs2 =
        _mm512_extracti32x4_epi32(rhs, 2);
    intrin_type_t<uint16_t, 8> rhs3 =
        _mm512_extracti32x4_epi32(rhs, 3);
    res = _mm512_setr_epi16(
        (uint16_t)_mm_extract_epi16(lhs0, 0) /
        (uint16_t)_mm_extract_epi16(rhs0, 0),
        (uint16_t)_mm_extract_epi16(lhs0, 1) /
        (uint16_t)_mm_extract_epi16(rhs0, 1),
        (uint16_t)_mm_extract_epi16(lhs0, 2) /
        (uint16_t)_mm_extract_epi16(rhs0, 2),
        (uint16_t)_mm_extract_epi16(lhs0, 3) /
        (uint16_t)_mm_extract_epi16(rhs0, 3),
        (uint16_t)_mm_extract_epi16(lhs0, 4) /
        (uint16_t)_mm_extract_epi16(rhs0, 4),
        (uint16_t)_mm_extract_epi16(lhs0, 5) /
        (uint16_t)_mm_extract_epi16(rhs0, 5),
        (uint16_t)_mm_extract_epi16(lhs0, 6) /
        (uint16_t)_mm_extract_epi16(rhs0, 6),
        (uint16_t)_mm_extract_epi16(lhs0, 7) /
        (uint16_t)_mm_extract_epi16(rhs0, 7),
        (uint16_t)_mm_extract_epi16(lhs1, 0) /
        (uint16_t)_mm_extract_epi16(rhs1, 0),
        (uint16_t)_mm_extract_epi16(lhs1, 1) /
        (uint16_t)_mm_extract_epi16(rhs1, 1),
        (uint16_t)_mm_extract_epi16(lhs1, 2) /
        (uint16_t)_mm_extract_epi16(rhs1, 2),
        (uint16_t)_mm_extract_epi16(lhs1, 3) /
        (uint16_t)_mm_extract_epi16(rhs1, 3),
        (uint16_t)_mm_extract_epi16(lhs1, 4) /
        (uint16_t)_mm_extract_epi16(rhs1, 4),
        (uint16_t)_mm_extract_epi16(lhs1, 5) /
        (uint16_t)_mm_extract_epi16(rhs1, 5),
        (uint16_t)_mm_extract_epi16(lhs1, 6) /
        (uint16_t)_mm_extract_epi16(rhs1, 6),
        (uint16_t)_mm_extract_epi16(lhs1, 7) /
        (uint16_t)_mm_extract_epi16(rhs1, 7),
        (uint16_t)_mm_extract_epi16(lhs2, 0) /
        (uint16_t)_mm_extract_epi16(rhs2, 0),
        (uint16_t)_mm_extract_epi16(lhs2, 1) /
        (uint16_t)_mm_extract_epi16(rhs2, 1),
        (uint16_t)_mm_extract_epi16(lhs2, 2) /
        (uint16_t)_mm_extract_epi16(rhs2, 2),
        (uint16_t)_mm_extract_epi16(lhs2, 3) /
        (uint16_t)_mm_extract_epi16(rhs2, 3),
        (uint16_t)_mm_extract_epi16(lhs2, 4) /
        (uint16_t)_mm_extract_epi16(rhs2, 4),
        (uint16_t)_mm_extract_epi16(lhs2, 5) /
        (uint16_t)_mm_extract_epi16(rhs2, 5),
        (uint16_t)_mm_extract_epi16(lhs2, 6) /
        (uint16_t)_mm_extract_epi16(rhs2, 6),
        (uint16_t)_mm_extract_epi16(lhs2, 7) /
        (uint16_t)_mm_extract_epi16(rhs2, 7),
        (uint16_t)_mm_extract_epi16(lhs3, 0) /
        (uint16_t)_mm_extract_epi16(rhs3, 0),
        (uint16_t)_mm_extract_epi16(lhs3, 1) /
        (uint16_t)_mm_extract_epi16(rhs3, 1),
        (uint16_t)_mm_extract_epi16(lhs3, 2) /
        (uint16_t)_mm_extract_epi16(rhs3, 2),
        (uint16_t)_mm_extract_epi16(lhs3, 3) /
        (uint16_t)_mm_extract_epi16(rhs3, 3),
        (uint16_t)_mm_extract_epi16(lhs3, 4) /
        (uint16_t)_mm_extract_epi16(rhs3, 4),
        (uint16_t)_mm_extract_epi16(lhs3, 5) /
        (uint16_t)_mm_extract_epi16(rhs3, 5),
        (uint16_t)_mm_extract_epi16(lhs3, 6) /
        (uint16_t)_mm_extract_epi16(rhs3, 6),
        (uint16_t)_mm_extract_epi16(lhs3, 7) /
        (uint16_t)_mm_extract_epi16(rhs3, 7));
}
#endif

// 128-bit int32_t
always_inline void div(clsvec<int32_t, 4> const *lhs,
    intrin_type_t<int32_t, 4> const& rhs,
    intrin_type_t<int32_t, 4>& res)
{
    intrin_type_t<int32_t, 4> const& lhs0 = lhs->data;
    intrin_type_t<int32_t, 4> const& rhs0 = rhs;
    res = _mm_setr_epi32(
        (int32_t)_mm_extract_epi32(lhs0, 0) /
        (int32_t)_mm_extract_epi32(rhs0, 0),
        (int32_t)_mm_extract_epi32(lhs0, 1) /
        (int32_t)_mm_extract_epi32(rhs0, 1),
        (int32_t)_mm_extract_epi32(lhs0, 2) /
        (int32_t)_mm_extract_epi32(rhs0, 2),
        (int32_t)_mm_extract_epi32(lhs0, 3) /
        (int32_t)_mm_extract_epi32(rhs0, 3));
}

#if defined(__AVX2__)
// 256-bit int32_t
always_inline void div(clsvec<int32_t, 8> const *lhs,
    intrin_type_t<int32_t, 8> const& rhs,
    intrin_type_t<int32_t, 8>& res)
{
    intrin_type_t<int32_t, 4> lhs0 =
        _mm256_extracti128_si256(lhs->data, 0);
    intrin_type_t<int32_t, 4> lhs1 =
        _mm256_extracti128_si256(lhs->data, 1);
    intrin_type_t<int32_t, 4> rhs0 =
        _mm256_extracti128_si256(rhs, 0);
    intrin_type_t<int32_t, 4> rhs1 =
        _mm256_extracti128_si256(rhs, 1);
    res = _mm256_setr_epi32(
        (int32_t)_mm_extract_epi32(lhs0, 0) /
        (int32_t)_mm_extract_epi32(rhs0, 0),
        (int32_t)_mm_extract_epi32(lhs0, 1) /
        (int32_t)_mm_extract_epi32(rhs0, 1),
        (int32_t)_mm_extract_epi32(lhs0, 2) /
        (int32_t)_mm_extract_epi32(rhs0, 2),
        (int32_t)_mm_extract_epi32(lhs0, 3) /
        (int32_t)_mm_extract_epi32(rhs0, 3),
        (int32_t)_mm_extract_epi32(lhs1, 0) /
        (int32_t)_mm_extract_epi32(rhs1, 0),
        (int32_t)_mm_extract_epi32(lhs1, 1) /
        (int32_t)_mm_extract_epi32(rhs1, 1),
        (int32_t)_mm_extract_epi32(lhs1, 2) /
        (int32_t)_mm_extract_epi32(rhs1, 2),
        (int32_t)_mm_extract_epi32(lhs1, 3) /
        (int32_t)_mm_extract_epi32(rhs1, 3));
}
#endif

#if defined(__AVX512F__)
// 512-bit int32_t
always_inline void div(clsvec<int32_t, 16> const *lhs,
    intrin_type_t<int32_t, 16> const& rhs,
    intrin_type_t<int32_t, 16>& res)
{
    intrin_type_t<int32_t, 4> lhs0 =
        _mm512_extracti32x4_epi32(lhs->data, 0);
    intrin_type_t<int32_t, 4> lhs1 =
        _mm512_extracti32x4_epi32(lhs->data, 1);
    intrin_type_t<int32_t, 4> lhs2 =
        _mm512_extracti32x4_epi32(lhs->data, 2);
    intrin_type_t<int32_t, 4> lhs3 =
        _mm512_extracti32x4_epi32(lhs->data, 3);
    intrin_type_t<int32_t, 4> rhs0 =
        _mm512_extracti32x4_epi32(rhs, 0);
    intrin_type_t<int32_t, 4> rhs1 =
        _mm512_extracti32x4_epi32(rhs, 1);
    intrin_type_t<int32_t, 4> rhs2 =
        _mm512_extracti32x4_epi32(rhs, 2);
    intrin_type_t<int32_t, 4> rhs3 =
        _mm512_extracti32x4_epi32(rhs, 3);
    res = _mm512_setr_epi32(
        (int32_t)_mm_extract_epi32(lhs0, 0) /
        (int32_t)_mm_extract_epi32(rhs0, 0),
        (int32_t)_mm_extract_epi32(lhs0, 1) /
        (int32_t)_mm_extract_epi32(rhs0, 1),
        (int32_t)_mm_extract_epi32(lhs0, 2) /
        (int32_t)_mm_extract_epi32(rhs0, 2),
        (int32_t)_mm_extract_epi32(lhs0, 3) /
        (int32_t)_mm_extract_epi32(rhs0, 3),
        (int32_t)_mm_extract_epi32(lhs1, 0) /
        (int32_t)_mm_extract_epi32(rhs1, 0),
        (int32_t)_mm_extract_epi32(lhs1, 1) /
        (int32_t)_mm_extract_epi32(rhs1, 1),
        (int32_t)_mm_extract_epi32(lhs1, 2) /
        (int32_t)_mm_extract_epi32(rhs1, 2),
        (int32_t)_mm_extract_epi32(lhs1, 3) /
        (int32_t)_mm_extract_epi32(rhs1, 3),
        (int32_t)_mm_extract_epi32(lhs2, 0) /
        (int32_t)_mm_extract_epi32(rhs2, 0),
        (int32_t)_mm_extract_epi32(lhs2, 1) /
        (int32_t)_mm_extract_epi32(rhs2, 1),
        (int32_t)_mm_extract_epi32(lhs2, 2) /
        (int32_t)_mm_extract_epi32(rhs2, 2),
        (int32_t)_mm_extract_epi32(lhs2, 3) /
        (int32_t)_mm_extract_epi32(rhs2, 3),
        (int32_t)_mm_extract_epi32(lhs3, 0) /
        (int32_t)_mm_extract_epi32(rhs3, 0),
        (int32_t)_mm_extract_epi32(lhs3, 1) /
        (int32_t)_mm_extract_epi32(rhs3, 1),
        (int32_t)_mm_extract_epi32(lhs3, 2) /
        (int32_t)_mm_extract_epi32(rhs3, 2),
        (int32_t)_mm_extract_epi32(lhs3, 3) /
        (int32_t)_mm_extract_epi32(rhs3, 3));
}
#endif

// 128-bit uint32_t
always_inline void div(clsvec<uint32_t, 4> const *lhs,
    intrin_type_t<uint32_t, 4> const& rhs,
    intrin_type_t<uint32_t, 4>& res)
{
    intrin_type_t<int32_t, 4> const& lhs0 = lhs->data;
    intrin_type_t<int32_t, 4> const& rhs0 = rhs;
    res = _mm_setr_epi32(
        (int)((uint32_t)_mm_extract_epi32(lhs0, 0) /
              (uint32_t)_mm_extract_epi32(rhs0, 0)),
        (int)((uint32_t)_mm_extract_epi32(lhs0, 1) /
              (uint32_t)_mm_extract_epi32(rhs0, 1)),
        (int)((uint32_t)_mm_extract_epi32(lhs0, 2) /
              (uint32_t)_mm_extract_epi32(rhs0, 2)),
        (int)((uint32_t)_mm_extract_epi32(lhs0, 3) /
              (uint32_t)_mm_extract_epi32(rhs0, 3)));
}

#if defined(__AVX2__)
// 256-bit uint32_t
always_inline void div(clsvec<uint32_t, 8> const *lhs,
    intrin_type_t<uint32_t, 8> const& rhs,
    intrin_type_t<uint32_t, 8>& res)
{
    intrin_type_t<uint32_t, 4> lhs0 = _mm256_extracti128_si256(lhs->data, 0);
    intrin_type_t<uint32_t, 4> lhs1 = _mm256_extracti128_si256(lhs->data, 1);
    intrin_type_t<uint32_t, 4> rhs0 = _mm256_extracti128_si256(rhs, 0);
    intrin_type_t<uint32_t, 4> rhs1 = _mm256_extracti128_si256(rhs, 1);
    res = _mm256_setr_epi32(
        (int)((uint32_t)_mm_extract_epi32(lhs0, 0) /
              (uint32_t)_mm_extract_epi32(rhs0, 0)),
        (int)((uint32_t)_mm_extract_epi32(lhs0, 1) /
              (uint32_t)_mm_extract_epi32(rhs0, 1)),
        (int)((uint32_t)_mm_extract_epi32(lhs0, 2) /
              (uint32_t)_mm_extract_epi32(rhs0, 2)),
        (int)((uint32_t)_mm_extract_epi32(lhs0, 3) /
              (uint32_t)_mm_extract_epi32(rhs0, 3)),
        (int)((uint32_t)_mm_extract_epi32(lhs1, 0) /
              (uint32_t)_mm_extract_epi32(rhs1, 0)),
        (int)((uint32_t)_mm_extract_epi32(lhs1, 1) /
              (uint32_t)_mm_extract_epi32(rhs1, 1)),
        (int)((uint32_t)_mm_extract_epi32(lhs1, 2) /
              (uint32_t)_mm_extract_epi32(rhs1, 2)),
        (int)((uint32_t)_mm_extract_epi32(lhs1, 3) /
              (uint32_t)_mm_extract_epi32(rhs1, 3)));
}
#endif

#if defined(__AVX512F__)
// 512-bit uint32_t
always_inline void div(clsvec<uint32_t, 16> const *lhs,
    intrin_type_t<uint32_t, 16> const& rhs,
    intrin_type_t<uint32_t, 16>& res)
{
    intrin_type_t<uint32_t, 4> lhs0 = _mm512_extracti32x4_epi32(lhs->data, 0);
    intrin_type_t<uint32_t, 4> lhs1 = _mm512_extracti32x4_epi32(lhs->data, 1);
    intrin_type_t<uint32_t, 4> lhs2 = _mm512_extracti32x4_epi32(lhs->data, 2);
    intrin_type_t<uint32_t, 4> lhs3 = _mm512_extracti32x4_epi32(lhs->data, 3);
    intrin_type_t<uint32_t, 4> rhs0 = _mm512_extracti32x4_epi32(rhs, 0);
    intrin_type_t<uint32_t, 4> rhs1 = _mm512_extracti32x4_epi32(rhs, 1);
    intrin_type_t<uint32_t, 4> rhs2 = _mm512_extracti32x4_epi32(rhs, 2);
    intrin_type_t<uint32_t, 4> rhs3 = _mm512_extracti32x4_epi32(rhs, 3);
    res = _mm512_setr_epi32(
        (int)((uint32_t)_mm_extract_epi32(lhs0, 0) /
              (uint32_t)_mm_extract_epi32(rhs0, 0)),
        (int)((uint32_t)_mm_extract_epi32(lhs0, 1) /
              (uint32_t)_mm_extract_epi32(rhs0, 1)),
        (int)((uint32_t)_mm_extract_epi32(lhs0, 2) /
              (uint32_t)_mm_extract_epi32(rhs0, 2)),
        (int)((uint32_t)_mm_extract_epi32(lhs0, 3) /
              (uint32_t)_mm_extract_epi32(rhs0, 3)),
        (int)((uint32_t)_mm_extract_epi32(lhs1, 0) /
              (uint32_t)_mm_extract_epi32(rhs1, 0)),
        (int)((uint32_t)_mm_extract_epi32(lhs1, 1) /
              (uint32_t)_mm_extract_epi32(rhs1, 1)),
        (int)((uint32_t)_mm_extract_epi32(lhs1, 2) /
              (uint32_t)_mm_extract_epi32(rhs1, 2)),
        (int)((uint32_t)_mm_extract_epi32(lhs1, 3) /
              (uint32_t)_mm_extract_epi32(rhs1, 3)),
        (int)((uint32_t)_mm_extract_epi32(lhs2, 0) /
              (uint32_t)_mm_extract_epi32(rhs2, 0)),
        (int)((uint32_t)_mm_extract_epi32(lhs2, 1) /
              (uint32_t)_mm_extract_epi32(rhs2, 1)),
        (int)((uint32_t)_mm_extract_epi32(lhs2, 2) /
              (uint32_t)_mm_extract_epi32(rhs2, 2)),
        (int)((uint32_t)_mm_extract_epi32(lhs2, 3) /
              (uint32_t)_mm_extract_epi32(rhs2, 3)),
        (int)((uint32_t)_mm_extract_epi32(lhs3, 0) /
              (uint32_t)_mm_extract_epi32(rhs3, 0)),
        (int)((uint32_t)_mm_extract_epi32(lhs3, 1) /
              (uint32_t)_mm_extract_epi32(rhs3, 1)),
        (int)((uint32_t)_mm_extract_epi32(lhs3, 2) /
              (uint32_t)_mm_extract_epi32(rhs3, 2)),
        (int)((uint32_t)_mm_extract_epi32(lhs3, 3) /
              (uint32_t)_mm_extract_epi32(rhs3, 3)));
}
#endif

// 128-bit float
always_inline void div(clsvec<float, 4> const *lhs,
    intrin_type_t<float, 4> const& rhs,
    intrin_type_t<float, 4>& res)
{
    res = _mm_div_ps(lhs->data, rhs);
}

#if defined(__AVX2__)
// 256-bit float
always_inline void div(clsvec<float, 8> const *lhs,
    intrin_type_t<float, 8> const& rhs,
    intrin_type_t<float, 8>& res)
{
    res = _mm256_div_ps(lhs->data, rhs);
}
#endif

#if defined(__AVX512F__)
// 512-bit float
always_inline void div(clsvec<float, 16> const *lhs,
    intrin_type_t<float, 16> const& rhs,
    intrin_type_t<float, 16>& res)
{
    res = _mm512_div_ps(lhs->data, rhs);
}
#endif

// always_inline __m256i _mm256_setr_epi64x(
//     long long v0, long long v1,
//     long long v2, long long v3)
// {
//     return _mm256_set_epi64x(v3, v2, v1, v0);
// }

// always_inline __m512i _mm512_setr_epi64(
//     long long v0, long long v1,
//     long long v2, long long v3,
//     long long v4, long long v5,
//     long long v6, long long v7)
// {
//     return _mm512_set_epi64(v7, v6, v5, v4, v3, v2, v1, v0);
// }

// 128-bit int64_t
always_inline void div(clsvec<int64_t, 2> const *lhs,
    intrin_type_t<int64_t, 2> const& rhs,
    intrin_type_t<int64_t, 2>& res)
{
    intrin_type_t<uint64_t, 2> const& lhs0 = lhs->data;
    intrin_type_t<uint64_t, 2> const& rhs0 = rhs;

    res = _mm_setr_epi64x(
        (int64_t)((uint64_t)_mm_extract_epi64(lhs0, 0) /
                  (uint64_t)_mm_extract_epi64(rhs0, 0)),
        (int64_t)((uint64_t)_mm_extract_epi64(lhs0, 1) /
                  (uint64_t)_mm_extract_epi64(rhs0, 1)));
}

#if defined(__AVX2__)
// 256-bit int64_t
always_inline void div(clsvec<int64_t, 4> const *lhs,
    intrin_type_t<int64_t, 4> const& rhs,
    intrin_type_t<int64_t, 4>& res)
{
    intrin_type_t<int64_t, 2> lhs0 = _mm256_extracti128_si256(lhs->data, 0);
    intrin_type_t<int64_t, 2> lhs1 = _mm256_extracti128_si256(lhs->data, 1);
    intrin_type_t<int64_t, 2> rhs0 = _mm256_extracti128_si256(rhs, 0);
    intrin_type_t<int64_t, 2> rhs1 = _mm256_extracti128_si256(rhs, 1);
    res = _mm256_setr_epi64x(
        _mm_extract_epi64(lhs0, 0) /
        _mm_extract_epi64(rhs0, 0),
        _mm_extract_epi64(lhs0, 1) /
        _mm_extract_epi64(rhs0, 1),
        _mm_extract_epi64(lhs1, 0) /
        _mm_extract_epi64(rhs1, 0),
        _mm_extract_epi64(lhs1, 1) /
        _mm_extract_epi64(rhs1, 1));
}
#endif

#if defined(__AVX512F__)
// 512-bit int64_t
always_inline void div(clsvec<int64_t, 8> const *lhs,
    intrin_type_t<int64_t, 8> const& rhs,
    intrin_type_t<int64_t, 8>& res)
{
    intrin_type_t<int64_t, 2> const& lhs0 = lhs->data;
    intrin_type_t<int64_t, 2> const& rhs0 = rhs;

    res = _mm_setr_epi64x(
        _mm_extract_epi64(lhs0, 0) /
        _mm_extract_epi64(rhs0, 0)),
        _mm_extract_epi64(lhs0, 1) /
        _mm_extract_epi64(rhs0, 1));

#endif

// 128-bit uint64_t
always_inline void div(clsvec<uint64_t, 2> const *lhs,
    intrin_type_t<uint64_t, 2> const& rhs,
    intrin_type_t<uint64_t, 2>& res)
{

}

#if defined(__AVX2__)
// 256-bit uint64_t
always_inline void div(clsvec<uint64_t, 4> const *lhs,
    intrin_type_t<uint64_t, 4> const& rhs,
    intrin_type_t<uint64_t, 4>& res)
{

}
#endif

#if defined(__AVX512F__)
// 512-bit uint64_t
always_inline void div(clsvec<uint64_t, 8> const *lhs,
    intrin_type_t<uint64_t, 8> const& rhs,
    intrin_type_t<uint64_t, 8>& res)
{

}
#endif

// 128-bit double
always_inline void div(clsvec<double, 2> const *lhs,
    intrin_type_t<double, 2> const& rhs,
    intrin_type_t<double, 2>& res)
{
    res = _mm_div_pd(lhs->data, rhs);
}

#if defined(__AVX2__)
// 256-bit double
always_inline void div(clsvec<double, 4> const *lhs,
    intrin_type_t<double, 4> const& rhs,
    intrin_type_t<double, 4>& res)
{
    res = _mm256_div_pd(lhs->data, rhs);
}
#endif

#if defined(__AVX512F__)
// 512-bit double
always_inline void div(clsvec<double, 8> const *lhs,
    intrin_type_t<double, 8> const& rhs,
    intrin_type_t<double, 8>& res)
{
    res = _mm512_div_pd(lhs->data, rhs);
}
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

template<> struct component_of<vecu8x16> { using type = uint8_t; };
template<> struct component_of<vecu8x32> { using type = uint8_t; };
template<> struct component_of<vecu8x64> { using type = uint8_t; };

template<> struct component_of<veci8x16> { using type = int8_t; };
template<> struct component_of<veci8x32> { using type = int8_t; };
template<> struct component_of<veci8x64> { using type = int8_t; };

template<> struct component_of<vecu16x8> { using type = uint16_t; };
template<> struct component_of<vecu16x16> { using type = uint16_t; };
template<> struct component_of<vecu16x32> { using type = uint16_t; };

template<> struct component_of<veci16x8> { using type = int16_t; };
template<> struct component_of<veci16x16> { using type = int16_t; };
template<> struct component_of<veci16x32> { using type = int16_t; };

template<> struct component_of<vecu32x4> { using type = uint32_t; };
template<> struct component_of<vecu32x8> { using type = uint32_t; };
template<> struct component_of<vecu32x16> { using type = uint32_t; };

template<> struct component_of<veci32x4> { using type = int32_t; };
template<> struct component_of<veci32x8> { using type = int32_t; };
template<> struct component_of<veci32x16> { using type = int32_t; };

template<> struct component_of<vecf32x4> { using type = float; };
template<> struct component_of<vecf32x8> { using type = float; };
template<> struct component_of<vecf32x16> { using type = float; };

template<> struct component_of<vecu64x2> { using type = uint64_t; };
template<> struct component_of<vecu64x4> { using type = uint64_t; };
template<> struct component_of<vecu64x8> { using type = uint64_t; };

template<> struct component_of<veci64x2> { using type = int64_t; };
template<> struct component_of<veci64x4> { using type = int64_t; };
template<> struct component_of<veci64x8> { using type = int64_t; };

template<> struct component_of<vecf64x2> { using type = double; };
template<> struct component_of<vecf64x4> { using type = double; };
template<> struct component_of<vecf64x8> { using type = double; };

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
#ifdef __clang__
    static_assert(sizeof(As) == sizeof(From),
        "This is too much to handle at once, make them the same size first");
    As result;
    __builtin_memcpy(&result, &rhs, sizeof(result));
    return result;
#else
    return __builtin_bit_cast(As, rhs);
#endif
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
        _mm512_mask_blend_epi32(
            _mm512_movepi32_mask(
                cast_to<__m512i>(mask)),
            cast_to<__m512i>(existing),
            cast_to<__m512i>(updated)));
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
        compact_mask,
        cast_to<__m512i>(indices),
        reinterpret_cast<int const *>(buffer),
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
                cast_to<__m256i>(i)),
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
