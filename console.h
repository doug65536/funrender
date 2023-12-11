#pragma once
#include <vector>
#include "huge_alloc.h"

class console {
public:

private:
    int x_scroll;
    

    // 0 means all the way at the bottom
    // negative number means scrolled up
    // viewport_height
    int y_scroll;
    int crsr_pos;
    std::vector<std::string> lines;
};
