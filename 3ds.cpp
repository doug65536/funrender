#include "3ds.h"
#include <fstream>
#include <vector>

bool threeds_loader::load(char const *pathname)
{
    std::ifstream file(pathname, std::ios::in | std::ios::binary);
    file.seekg(0, std::ios::end);
    size_t file_sz = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<uint8_t> buffer(file_sz);
    file.read((char*)buffer.data(), buffer.size());
    file.close();

    uint8_t const *data_st = buffer.data();
    uint8_t const *data_en = buffer.data() + buffer.size();

    uint16_t block_id{};
    uint32_t block_sz{};
    memcpy(&block_id, data_st, sizeof(block_id));
    memcpy(&block_sz, data_st +
        sizeof(block_id), sizeof(block_sz));

    return parse(block_id, block_sz, data_st + 6);
}

enum {

};

bool threeds_loader::parse(uint16_t block_id,
    uint32_t block_sz, uint8_t const *block)
{
    switch (block_id) {

    }
    return true;
}
