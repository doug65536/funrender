#pragma once
#include <cstdint>

class threeds_loader {
public:
    bool load(char const *pathanme);
    bool parse(uint16_t block_id, uint32_t block_sz, uint8_t const *block);

private:
};