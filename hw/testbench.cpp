#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>

#include "cartridge_kernel.h"

static int8_t weights[FEATURE_COUNT][CLASS_SLOTS];

int main(int argc, char **argv) {
    std::ifstream input(argc > 1 ? argv[1] : "vectors.txt");
    if (!input) {
        std::cerr << "cannot open vectors file\n";
        return 2;
    }
    std::string keyword, name;
    long checked = 0, mismatches = 0;
    while (input >> keyword >> name) {
        int rows, slots;
        input >> rows >> slots;
        if (keyword != "cartridge" || rows != FEATURE_COUNT || slots != CLASS_SLOTS) {
            std::cerr << "unexpected cartridge header\n";
            return 2;
        }
        int32_t bias[CLASS_SLOTS];
        uint8_t refuse_slot = 0;
        for (int slot = 0; slot < CLASS_SLOTS; slot++) {
            input >> bias[slot];
            if (bias[slot] > -(1 << 29)) refuse_slot = slot;
        }
        for (int row = 0; row < FEATURE_COUNT; row++) {
            for (int slot = 0; slot < CLASS_SLOTS; slot++) {
                int value;
                input >> value;
                weights[row][slot] = (int8_t)value;
            }
        }
        int vectors;
        input >> keyword >> vectors;
        for (int vector = 0; vector < vectors; vector++) {
            int active;
            input >> active;
            uint16_t feature_index[MAX_ACTIVE] = {0};
            uint8_t feature_count[MAX_ACTIVE] = {0};
            for (int position = 0; position < active; position++) {
                int index, count;
                input >> index >> count;
                feature_index[position] = (uint16_t)index;
                feature_count[position] = (uint8_t)count;
            }
            int32_t scores[CLASS_SLOTS], margin;
            uint8_t best;
            cartridge_kernel(weights, bias, feature_index, feature_count, (uint8_t)active, refuse_slot,
                             scores, &best, &margin);
            for (int slot = 0; slot < CLASS_SLOTS; slot++) {
                long expected;
                input >> expected;
                checked++;
                if (expected != scores[slot]) mismatches++;
            }
        }
        std::printf("%-8s %d vectors replayed\n", name.c_str(), vectors);
    }
    std::printf("%ld scores checked, %ld mismatches\n", checked, mismatches);
    return mismatches == 0 ? 0 : 1;
}
