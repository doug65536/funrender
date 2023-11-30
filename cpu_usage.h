#pragma once
#include <cstdint>

struct cpu_usage_info {
    uint64_t total;
    uint64_t kernel;
    uint64_t user;
    bool ok;
};
cpu_usage_info cpu_usage();
