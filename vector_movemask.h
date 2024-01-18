#pragma once

// FIXME: these do work to carefully get bit 31, but usually,
// it doesn't even matter which bit it gets

template<typename X>
always_inline unsigned vec_movemask(X const& i)
{
    static_assert(sizeof(component_of_t<X>) == sizeof(float));
    static_assert(sizeof(comp_count_v<X>) <= sizeof(int) * CHAR_BIT);
    unsigned r = 0;
    for (size_t c = 0; c < comp_count_v<X>; ++c)
        r |= ((unsigned)(cast_to<float>(i[r]) < 0) << c);
    return r;
}

#if defined(__ARM_NEON)
#include <arm_neon.h>
always_inline unsigned vec_movemask(vecf32x4 const& i) {
    int32x4_t shifted = vshrq_n_s32(
        vreinterpretq_s32_f32(
            cast_to<float32x4_t>(i)), 31);
    int32x4_t masked = vandq_s32(shifted,
        {1, 2, 4, 8});
    return vaddvq_s32(masked);
}
#endif

// Compose a bigger movemask from halves
template<typename X>
always_inline unsigned vec_movemask_synthetic(X const& i)
{
    constexpr auto sz = comp_count_v<X>;
    constexpr auto half = sz / 2;
    using C = component_of_t<X>;
    using Vhalf = vec_t<C, half>;
    Vhalf lo = vec_lo(i);
    Vhalf hi = vec_hi(i);
    unsigned lomask = vec_movemask(lo);
    unsigned himask = vec_movemask(hi);
    return lomask | (himask << half);
}

#if defined(__SSE2__)
always_inline unsigned vec_movemask(vecf32x4 const& i) {
    return (unsigned)_mm_movemask_ps(cast_to<__m128>(i)); }
#endif
#if defined(__AVX2__)
always_inline unsigned vec_movemask(vecf32x8 const& i) {
    return (unsigned)_mm256_movemask_ps(cast_to<__m256>(i)); }
#elif defined(__SSE2__) || defined(__ARM_NEON)
// Synthesize 256 one from lo and hi 128 ones
always_inline unsigned vec_movemask(vecf32x8 const& i) {
    return vec_movemask_synthetic(i); }
#endif
#if defined(__AVX512F__)
always_inline unsigned vec_movemask(vecf32x16 const& i) {
    return _mm512_movepi32_mask(cast_to<__m512i>(i)); }
#elif defined(__AVX2__) || defined(__SSE2__) || defined(__ARM_NEON)
// Synthesize 512 one from lo and hi 256 ones
always_inline unsigned vec_movemask(vecf32x16 const& i) {
    return vec_movemask_synthetic(i); }
#endif

#if defined(__SSE2__) || defined(__AVX2__) || defined(__ARM_NEON)
// Quickest way is to take a bypass delay and use _ps one
always_inline unsigned vec_movemask(vecu32x4 const& i) {
    return vec_movemask(cast_to<vecf32x4>(i)); }
always_inline unsigned vec_movemask(veci32x4 const& i) {
    return vec_movemask(cast_to<vecf32x4>(i)); }

always_inline unsigned vec_movemask(vecu32x8 const& i) {
    return vec_movemask(cast_to<vecf32x8>(i)); }
always_inline unsigned vec_movemask(veci32x8 const& i) {
    return vec_movemask(cast_to<vecf32x8>(i)); }

always_inline unsigned vec_movemask(vecu32x16 const& i) {
    return vec_movemask(cast_to<vecf32x16>(i)); }
always_inline unsigned vec_movemask(veci32x16 const& i) {
    return vec_movemask(cast_to<vecf32x16>(i)); }
#endif
