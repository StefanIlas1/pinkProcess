#ifndef CARTRIDGE_KERNEL_H
#define CARTRIDGE_KERNEL_H

#include <cstdint>

const int FEATURE_COUNT = 4096;
const int CLASS_SLOTS = 16;
const int MAX_ACTIVE = 64;

void cartridge_kernel(
    const int8_t weights[FEATURE_COUNT][CLASS_SLOTS],
    const int32_t bias[CLASS_SLOTS],
    const uint16_t feature_index[MAX_ACTIVE],
    const uint8_t feature_count[MAX_ACTIVE],
    uint8_t active,
    uint8_t refuse_slot,
    int32_t scores[CLASS_SLOTS],
    uint8_t *best_slot,
    int32_t *margin);

#endif
