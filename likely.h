#pragma once

#define expect_value(value, expected_value) \
    (__builtin_expect((value), (expected_value)))
#define unlikely(k) (expect_value(!!(k), 0))
#define likely(k) (expect_value(!!(k), 1))
