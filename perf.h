#pragma once

#include <chrono>


using clk = std::chrono::high_resolution_clock;
using time_point = typename clk::time_point;
using duration = typename clk::duration;
