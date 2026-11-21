#include "cartridge_kernel.h"

void cartridge_kernel(
    const int8_t weights[FEATURE_COUNT][CLASS_SLOTS],
    const int32_t bias[CLASS_SLOTS],
    const uint16_t feature_index[MAX_ACTIVE],
    const uint8_t feature_count[MAX_ACTIVE],
    uint8_t active,
    uint8_t refuse_slot,
    int32_t scores[CLASS_SLOTS],
    uint8_t *best_slot,
    int32_t *margin) {
#pragma HLS INTERFACE s_axilite port = return bundle = control
#pragma HLS INTERFACE s_axilite port = weights bundle = control
#pragma HLS INTERFACE s_axilite port = bias bundle = control
#pragma HLS INTERFACE s_axilite port = feature_index bundle = control
#pragma HLS INTERFACE s_axilite port = feature_count bundle = control
#pragma HLS INTERFACE s_axilite port = active bundle = control
#pragma HLS INTERFACE s_axilite port = refuse_slot bundle = control
#pragma HLS INTERFACE s_axilite port = scores bundle = control
#pragma HLS INTERFACE s_axilite port = best_slot bundle = control
#pragma HLS INTERFACE s_axilite port = margin bundle = control
#pragma HLS BIND_STORAGE variable = weights type = ram_1p impl = bram

    int32_t totals[CLASS_SLOTS];
#pragma HLS ARRAY_PARTITION variable = totals complete

    for (int slot = 0; slot < CLASS_SLOTS; slot++) {
#pragma HLS UNROLL
        totals[slot] = bias[slot];
    }

    for (int position = 0; position < MAX_ACTIVE; position++) {
#pragma HLS PIPELINE II = 1
        if (position < active) {
            const uint16_t row = feature_index[position];
            const int16_t count = feature_count[position];
            for (int slot = 0; slot < CLASS_SLOTS; slot++) {
#pragma HLS UNROLL
                totals[slot] += count * (int16_t)weights[row][slot];
            }
        }
    }

    uint8_t best = 0;
    for (int slot = 0; slot < CLASS_SLOTS; slot++) {
        scores[slot] = totals[slot];
        if (slot < refuse_slot && totals[slot] > totals[best]) {
            best = slot;
        }
    }
    *best_slot = best;
    *margin = totals[best] - totals[refuse_slot];
}
