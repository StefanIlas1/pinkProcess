"""Verify the experimental scaled argmax RTL against FLAN's NumPy rounding."""

from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import numpy as np


def score(score_i32, activation_scale, row_scale):
    activation = np.uint32(activation_scale).view(np.float32)
    scale = np.uint32(row_scale).view(np.float32)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        return np.float32(score_i32) * activation * scale


def generate_vectors(stream):
    rng = np.random.default_rng(391)

    def write_frame(frame):
        winner = None
        invalid = False
        for index, dot, activation, scale in frame:
            value = score(dot, activation, scale)
            valid_scales = all(0 <= f < 0x7F800000 or f == 0x80000000 for f in (activation, scale))
            if not valid_scales or np.isnan(value):
                invalid = True
            elif winner is None or winner[0] < value:
                winner = (value, index)
        out_score, out_index = winner if winner is not None else (np.float32(0), 0)
        for row, (index, dot, activation, scale) in enumerate(frame):
            last = row == len(frame) - 1
            stream.write(
                f"{index:08x} {dot & 0xFFFFFFFF:08x} {activation:08x} {scale:08x} "
                f"{int(last)} {int(invalid)} {out_index:08x} {out_score.view(np.uint32):08x}\n"
            )

    one = 0x3F800000
    write_frame([(0, -1, one, 0), (1, 1, one, 0)])
    write_frame([(5, 1 << 24, one, one), (6, (1 << 24) + 1, one, one)])
    write_frame([(1, -1, one, one), (2, -1, one, one - 1)])
    write_frame([(1, 2, one, one), (2, 1, 0x7FC00000, one)])
    write_frame([(1, 2, 0x7F7FFFFF, 0)])
    write_frame([(1, 1, 0x00800000, 0x33800000), (2, 1, 0x00800000, 0x33800001)])
    for dot in (-(1 << 31), -(1 << 24) - 1, -1, 0, 1, (1 << 24) + 1, (1 << 31) - 1):
        for activation in (0, 1, 0x00800000, 0x3F800000, 0x3F800001, 0x7F7FFFFF):
            for scale in (0, 1, 0x007FFFFF, 0x3F800000, 0x3F800001, 0x7F7FFFFF):
                write_frame([(0, dot, activation, scale)])
    for length in (1,) * 2048 + (7, 17, 64, 192, 502):
        dots = rng.integers(-(1 << 31), 1 << 31, size=length)
        activation = int(rng.integers(0, 0x7F800000))
        scales = rng.integers(0, 0x7F800000, size=length)
        write_frame([(i, int(dot), activation, int(scale))
                     for i, (dot, scale) in enumerate(zip(dots, scales))])
    activations = np.float32(0.0097).view(np.uint32)
    dots = rng.integers(-200000, 200000, size=32128)
    scales = rng.uniform(0.0001, 0.02, size=32128).astype(np.float32).view(np.uint32)
    write_frame([(i, int(dot), int(activations), int(scale))
                 for i, (dot, scale) in enumerate(zip(dots, scales))])


def main():
    fpga = Path(__file__).resolve().parent
    with TemporaryDirectory() as directory:
        work = Path(directory)
        with (work / "argmax.txt").open("w") as stream:
            generate_vectors(stream)
        subprocess.run(
            ["iverilog", "-g2012", "-s", "projection_scaled_argmax_tb", "-o", "argmax",
             fpga / "projection_scaled_argmax.sv", fpga / "projection_scaled_argmax_tb.sv"],
            cwd=work, check=True,
        )
        subprocess.run(["vvp", "argmax"], cwd=work, check=True)
        subprocess.run(
            ["iverilog", "-g2012", "-s", "projection_argmax_dma_tb", "-o", "argmax-dma",
             fpga / "pinkprocess_argmax.v", fpga / "projection_dma.v",
             fpga / "projection_scaled_argmax.sv", fpga / "projection_argmax_dma_tb.sv"],
            cwd=work, check=True,
        )
        subprocess.run(["vvp", "argmax-dma"], cwd=work, check=True)


if __name__ == "__main__":
    main()
