#include <gtest/gtest.h>
#include "abstract_vector.h"

TEST(vectors, traits)
{
    static_assert(comp_count_v<vecu64x2> == 2);
    static_assert(comp_count_v<veci64x2> == 2);
    static_assert(comp_count_v<vecf64x2> == 2);

    static_assert(comp_count_v<vecu32x4> == 4);
    static_assert(comp_count_v<veci32x4> == 4);
    static_assert(comp_count_v<vecf32x4> == 4);
    static_assert(comp_count_v<vecu64x4> == 4);
    static_assert(comp_count_v<veci64x4> == 4);
    static_assert(comp_count_v<vecf64x4> == 4);

    static_assert(comp_count_v<vecu16x8> == 8);
    static_assert(comp_count_v<veci16x8> == 8);
    static_assert(comp_count_v<vecu32x8> == 8);
    static_assert(comp_count_v<veci32x8> == 8);
    static_assert(comp_count_v<vecf32x8> == 8);
    static_assert(comp_count_v<vecu64x8> == 8);
    static_assert(comp_count_v<veci64x8> == 8);
    static_assert(comp_count_v<vecf64x8> == 8);

    static_assert(comp_count_v<vecu8x16> == 16);
    static_assert(comp_count_v<veci8x16> == 16);
    static_assert(comp_count_v<vecu16x16> == 16);
    static_assert(comp_count_v<veci16x16> == 16);
    static_assert(comp_count_v<vecu32x16> == 16);
    static_assert(comp_count_v<veci32x16> == 16);
    static_assert(comp_count_v<vecf32x16> == 16);

    static_assert(comp_count_v<vecu8x32> == 32);
    static_assert(comp_count_v<veci8x32> == 32);
    static_assert(comp_count_v<vecu16x32> == 32);
    static_assert(comp_count_v<veci16x32> == 32);

    static_assert(comp_count_v<vecu8x64> == 64);
    static_assert(comp_count_v<veci8x64> == 64);

    static_assert(sizeof(vecu8x16) == 16);
    static_assert(sizeof(veci8x16) == 16);
    static_assert(sizeof(vecu16x8) == 16);
    static_assert(sizeof(vecu16x8) == 16);
    static_assert(sizeof(vecu32x4) == 16);
    static_assert(sizeof(vecu32x4) == 16);
    static_assert(sizeof(vecf32x4) == 16);
    static_assert(sizeof(vecu64x2) == 16);
    static_assert(sizeof(veci64x2) == 16);
    static_assert(sizeof(vecu64x2) == 16);

    static_assert(sizeof(vecu8x32) == 32);
    static_assert(sizeof(veci8x32) == 32);
    static_assert(sizeof(vecu16x16) == 32);
    static_assert(sizeof(vecu16x16) == 32);
    static_assert(sizeof(vecu32x8) == 32);
    static_assert(sizeof(vecu32x8) == 32);
    static_assert(sizeof(vecf32x8) == 32);
    static_assert(sizeof(vecu64x4) == 32);
    static_assert(sizeof(veci64x4) == 32);
    static_assert(sizeof(vecu64x4) == 32);

    static_assert(sizeof(vecu8x64) == 64);
    static_assert(sizeof(veci8x64) == 64);
    static_assert(sizeof(vecu16x32) == 64);
    static_assert(sizeof(vecu16x32) == 64);
    static_assert(sizeof(vecu32x16) == 64);
    static_assert(sizeof(vecu32x16) == 64);
    static_assert(sizeof(vecf32x16) == 64);
    static_assert(sizeof(vecu64x8) == 64);
    static_assert(sizeof(veci64x8) == 64);
    static_assert(sizeof(vecu64x8) == 64);

    static_assert(std::is_same_v<
        component_of_t<vecu8x16>, uint8_t>);
    static_assert(std::is_same_v<
        component_of_t<vecu8x32>, uint8_t>);
    static_assert(std::is_same_v<
        component_of_t<vecu8x64>, uint8_t>);

    static_assert(std::is_same_v<
        component_of_t<veci8x16>, int8_t>);
    static_assert(std::is_same_v<
        component_of_t<veci8x32>, int8_t>);
    static_assert(std::is_same_v<
        component_of_t<veci8x64>, int8_t>);

    static_assert(std::is_same_v<
        component_of_t<vecu16x8>, uint16_t>);
    static_assert(std::is_same_v<
        component_of_t<vecu16x16>, uint16_t>);
    static_assert(std::is_same_v<
        component_of_t<vecu16x32>, uint16_t>);

    static_assert(std::is_same_v<
        component_of_t<veci16x8>, int16_t>);
    static_assert(std::is_same_v<
        component_of_t<veci16x16>, int16_t>);
    static_assert(std::is_same_v<
        component_of_t<veci16x32>, int16_t>);

    static_assert(std::is_same_v<
        component_of_t<vecu32x4>, uint32_t>);
    static_assert(std::is_same_v<
        component_of_t<vecu32x8>, uint32_t>);
    static_assert(std::is_same_v<
        component_of_t<vecu32x16>, uint32_t>);

    static_assert(std::is_same_v<
        component_of_t<veci32x4>, int32_t>);
    static_assert(std::is_same_v<
        component_of_t<veci32x8>, int32_t>);
    static_assert(std::is_same_v<
        component_of_t<veci32x16>, int32_t>);

    static_assert(std::is_same_v<
        component_of_t<vecu64x2>, uint64_t>);
    static_assert(std::is_same_v<
        component_of_t<vecu64x4>, uint64_t>);
    static_assert(std::is_same_v<
        component_of_t<vecu64x8>, uint64_t>);

    static_assert(std::is_same_v<
        component_of_t<veci64x2>, int64_t>);
    static_assert(std::is_same_v<
        component_of_t<veci64x4>, int64_t>);
    static_assert(std::is_same_v<
        component_of_t<veci64x8>, int64_t>);

    static_assert(std::is_same_v<
        component_of_t<vecf32x4>, float>);
    static_assert(std::is_same_v<
        component_of_t<vecf32x8>, float>);
    static_assert(std::is_same_v<
        component_of_t<vecf32x16>, float>);
    static_assert(std::is_same_v<
        component_of_t<vecf64x2>, double>);
    static_assert(std::is_same_v<
        component_of_t<vecf64x4>, double>);
    static_assert(std::is_same_v<
        component_of_t<vecf64x8>, double>);
}

#ifdef GTEST_HAS_DEATH_TEST
TEST(vectors, improper_append_asserts)
{
    struct s {
        vecf32x4 a;
    };
    std::vector<s> soa_vector;
    unsigned item_count = 0;
    vecf32x4 val{1.0f, 2.0f, 3.0f, 4.0f};
    unsigned bitmask = vec_movemask(val < 4.0f);
    deposit_mask<vecf32x4> dep(item_count, bitmask, soa_vector);
    auto &bundle = soa_vector.back();

    do {
        dep.deposit_into(bundle.a, val);
    } while (dep.continue_from(item_count, soa_vector));

    EXPECT_EQ(item_count, 3);

    EXPECT_DEATH({
        // without a call to .setup it should assert
        // because `deposited` is left dangling as 0.
        // the last continue_from returned false, so
        // you must call setup or make a new deposit_mask
        dep.deposit_into(bundle.a, val);
    }, "Assertion");
}
#endif

TEST(deposit_mask, append)
{
    struct f {
        vecu32x8 a{};
        vecu32x8 b{};
        vecu32x8 c{};
    };
    unsigned count = 0;
    std::vector<f> aos;

    vecu32x8 a{
        0xEEEEEEEEU, 0, 0xEEEEEEEEU, 1,
        2, 0xEEEEEEEEU, 3, 102
    };
    vecu32x8 b{
        0xEEEEEEEEU, 4, 0xEEEEEEEEU, 5,
        6, 0xEEEEEEEEU, 7, 102
    };
    vecu32x8 c{
        0xEEEEEEEEU, 8, 0xEEEEEEEEU, 9,
        10, 0xEEEEEEEEU, 11, 102
    };
    unsigned bitmask = vec_movemask(a != 0xEEEEEEEEU);

    deposit_mask<vecu32x8> dep{count, bitmask, aos};
    EXPECT_EQ(aos.size(), 1);
    auto *back = &aos.back();
    dep.deposit_into(back->a, a);
    dep.deposit_into(back->b, b);
    dep.deposit_into(back->c, c);
    EXPECT_EQ(back->a[0], 0);
    EXPECT_EQ(back->a[1], 1);
    EXPECT_EQ(back->a[2], 2);
    EXPECT_EQ(back->a[3], 3);
    EXPECT_EQ(back->a[4], 102);
    EXPECT_EQ(back->b[0], 4);
    EXPECT_EQ(back->b[1], 5);
    EXPECT_EQ(back->b[2], 6);
    EXPECT_EQ(back->b[3], 7);
    EXPECT_EQ(back->b[4], 102);
    EXPECT_EQ(back->c[0], 8);
    EXPECT_EQ(back->c[1], 9);
    EXPECT_EQ(back->c[2], 10);
    EXPECT_EQ(back->c[3], 11);
    EXPECT_EQ(back->c[4], 102);
    EXPECT_EQ(count, 0);
    bool continues = dep.continue_from(count, aos);
    EXPECT_EQ(count, 5);
    EXPECT_EQ(continues, false);

    dep.setup(count, bitmask, aos);
    dep.deposit_into(back->a, a);
    dep.deposit_into(back->b, b);
    dep.deposit_into(back->c, c);

    EXPECT_EQ(back->a[0], 0);
    EXPECT_EQ(back->a[1], 1);
    EXPECT_EQ(back->a[2], 2);
    EXPECT_EQ(back->a[3], 3);
    EXPECT_EQ(back->a[4], 102);
    EXPECT_EQ(back->a[5], 0);
    EXPECT_EQ(back->a[6], 1);
    EXPECT_EQ(back->a[7], 2);

    EXPECT_EQ(back->b[0], 4);
    EXPECT_EQ(back->b[1], 5);
    EXPECT_EQ(back->b[2], 6);
    EXPECT_EQ(back->b[3], 7);
    EXPECT_EQ(back->b[4], 102);
    EXPECT_EQ(back->b[5], 4);
    EXPECT_EQ(back->b[6], 5);
    EXPECT_EQ(back->b[7], 6);

    EXPECT_EQ(back->c[0], 8);
    EXPECT_EQ(back->c[1], 9);
    EXPECT_EQ(back->c[2], 10);
    EXPECT_EQ(back->c[3], 11);
    EXPECT_EQ(back->c[4], 102);
    EXPECT_EQ(back->c[5], 8);
    EXPECT_EQ(back->c[6], 9);
    EXPECT_EQ(back->c[7], 10);
    EXPECT_EQ(count, 5);
    continues = dep.continue_from(count, aos);
    EXPECT_EQ(continues, true);
    EXPECT_EQ(count, 8);
    EXPECT_EQ(aos.size(), 2);

    dep.deposit_into(back->a, a);
    dep.deposit_into(back->b, b);
    dep.deposit_into(back->c, c);

    continues = dep.continue_from(count, aos);
    EXPECT_EQ(continues, false);
    EXPECT_EQ(count, 10);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}