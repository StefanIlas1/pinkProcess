# Cartridge kernel

`cartridge_kernel.cpp` is the Vitis HLS source for the PYNQ-Z2 overlay. One call adds up to 64 weight rows into 16 accumulators and returns every score, the best card and its margin over the refuse class.

Check it against the Python reference on any machine with a C++ compiler:

```sh
python3 -m cartridge.check
clang++ -std=c++17 -O2 -Wno-unknown-pragmas hw/cartridge_kernel.cpp hw/testbench.cpp -o /tmp/kernel_tb
/tmp/kernel_tb hw/vectors.txt
```

Expected last line: `3264 scores checked, 0 mismatches`.

## Build for the board

Needs Vitis HLS and Vivado 2020.2 to 2022.2 on x86 Linux. This has not been run yet.

1. Vitis HLS: top function `cartridge_kernel`, part `xc7z020clg400-1`, clock 10 ns. Use `testbench.cpp` and `vectors.txt` for C simulation, then export RTL as IP.
2. Vivado: Zynq PS plus the IP on `M_AXI_GP0`. Generate the bitstream.
3. Copy `cartridge_kernel.bit` and `.hwh` to `hw/build/` on the board.
4. Copy the register offsets from the generated `xcartridge_kernel_hw.h` into `hw/build/offsets.json` with the keys `weights`, `bias`, `feature_index`, `feature_count`, `active`, `refuse_slot`, `scores`, `best_slot`.
5. On the board: `python3 -m cartridge.server --host 0.0.0.0`. The server uses the overlay when those files exist and refuses to answer if the overlay's scores differ from the integer reference.

## Sizes

| item | size |
|---|---|
| weight memory | 4096 rows × 16 int8 = 64 KB of the 630 KB block RAM |
| accumulators | 16 × 18 bits needed, 32 used |
| multipliers | 16 int8 × 4-bit products per row, within 220 DSP48 |
