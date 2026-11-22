"""Compare the projection tile against signed integer matrix multiplication."""

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile
from time import monotonic


FLUSH_STEPS = 14


def drive(activations, weights, *, step=True):
    # Lane zero occupies the least significant byte on the tile bus.
    a_bytes = bytes(value & 255 for value in reversed(activations))
    w_bytes = bytes(value & 255 for value in reversed(weights))
    return f"{2 if step else 3} {a_bytes.hex()} {w_bytes.hex()}\n"


def expect(sums):
    return "5 " + "".join(f"{value & 0xffffffff:08x}" for value in reversed(sums)) + "\n"


def tile_stimulus(activations, weights, expected, decode, *, stalls=False):
    depth = len(activations[0])
    positions, columns = (1, 64) if decode else (8, 8)
    for row in range(0, len(activations), positions):
        inputs = activations[row:row + positions]
        for col in range(0, len(weights), columns):
            coefficients = weights[col:col + columns]
            yield f"1 {int(decode)}\n"
            for index in range(depth):
                if stalls and index in (0, 1, depth - 1):
                    yield drive([127] * 8, [-128] * 64, step=False)
                yield drive([vector[index] for vector in inputs],
                            [vector[index] for vector in coefficients])
            if not decode:
                if stalls:
                    yield drive([-128] * 8, [127] * 64, step=False)
                yield from ["4\n"] * FLUSH_STEPS

            sums = [0] * 64
            for position in range(len(inputs)):
                values = expected[row + position][col:col + len(coefficients)]
                offset = position * columns
                sums[offset:offset + len(values)] = values
            yield expect(sums)


def edge_cases():
    depth = 19
    values = [-128, 127, -1, 0, 1, -127, 64, -3, 19, 82, -92]
    activations = [[values[(position + k) % len(values)] for k in range(depth)]
                   for position in range(9)]
    weights = [[values[(3 * col + 2 * k) % len(values)] for k in range(depth)]
               for col in range(65)]
    weights[0] = [-128] * depth
    weights[1] = [127] * depth
    activations[0] = [-128] * depth
    expected = [[sum(a * w for a, w in zip(vector, coefficients)) for coefficients in weights]
                for vector in activations]

    yield "0\n"
    yield expect([0] * 64)
    for mode in (0, 1):
        for reset in ("0\n", f"1 {mode}\n"):
            # Abort with pending pipeline data and nonzero accumulators.
            yield f"1 {mode}\n"
            yield from [drive([127] * 8, [-128] * 64)] * 15
            yield reset
            yield expect([0] * 64)
            yield from ["4\n"] * FLUSH_STEPS
            yield expect([0] * 64)
    yield from tile_stimulus(activations, weights, expected, False, stalls=True)
    yield from tile_stimulus(activations[:1], weights, expected[:1], True, stalls=True)


def load_capture(path):
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Reading a projection capture requires NumPy.") from exc

    with np.load(path, allow_pickle=False) as capture:
        activations = capture["activations"]
        weights = capture["weights"]
    if any(matrix.ndim != 2 or not all(matrix.shape) or matrix.dtype != np.int8
           for matrix in (activations, weights)) or activations.shape[1] != weights.shape[1]:
        raise ValueError(f"{path}: expected nonempty int8 matrices with matching reduction dimensions")
    return activations, weights, activations.astype(np.int32) @ weights.astype(np.int32).T


def capture_stimulus(path):
    activations, weights, expected = load_capture(path)
    batch, depth = activations.shape
    outputs = weights.shape[0]
    decode = batch == 1
    tiles = (outputs + 63) // 64 if decode else ((batch + 7) // 8) * ((outputs + 7) // 8)
    cycles = tiles * (1 + depth + (0 if decode else FLUSH_STEPS))
    description = (f"{path.name}: {batch} × {depth} by {outputs} × {depth} ({'decode' if decode else 'encode'}), "
                   f"{batch * outputs:,} sums, {batch * outputs * depth:,} MACs, "
                   f"{tiles:,} tiles, {cycles:,} tile cycles including clear"
                   + ("" if decode else " and flush"))
    stimulus = tile_stimulus(activations.tolist(), weights.tolist(), expected.tolist(), decode)
    return description, stimulus


def system_capture(path, work):
    activations, weights, expected = load_capture(path)
    import numpy as np

    if not __package__:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from lm_reference.projection_overlay import pack_projection

    packed_a, packed_w = pack_projection(weights, activations)
    positions, depth, _ = packed_a.shape
    outputs, _, columns = packed_w.shape
    position_lanes = 64 // columns
    padded = np.zeros((positions * position_lanes, outputs * columns), dtype="<i4")
    padded[:len(activations), :len(weights)] = expected
    packed_y = padded.reshape(positions, position_lanes, outputs, columns).transpose(0, 2, 1, 3)
    for name, array in (("activation", packed_a), ("weights", packed_w), ("expected", packed_y)):
        words = np.ascontiguousarray(array).ravel().view("<u8")
        (work / f"{name}.hex").write_text("".join(f"{word:016x}\n" for word in words))
    return ["+capture", f"+mode={int(position_lanes == 1)}", f"+depth={depth}",
            f"+positions={positions}", f"+outputs={outputs}"]


def compile_simulation(here, work, target, sources):
    sim = work / target
    subprocess.run(["iverilog", "-g2012", "-s", target, "-o", str(sim),
                    *(str(here / source) for source in sources)], check=True)
    return sim


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, action="append", default=[],
                        help="Replay an INT8 NPZ capture at the tile pins. Can be repeated.")
    parser.add_argument("--system-capture", type=Path, action="append", default=[],
                        help="Replay an INT8 NPZ capture through AXI and simulated DDR. Can be repeated.")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory() as temporary:
        work = Path(temporary)
        system = compile_simulation(here, work, "projection_system_tb", (
            "pinkprocess_projection.v", "projection_dma.v", "projection_engine.sv",
            "projection_tile.sv", "projection_system_tb.sv",
        ))
        subprocess.run(["vvp", str(system)], cwd=work, check=True)
        for path in args.system_capture:
            parameters = system_capture(path, work)
            print(f"Replaying {path.name} through AXI and simulated DDR.", flush=True)
            start = monotonic()
            subprocess.run(["vvp", str(system), *parameters], cwd=work, check=True)
            print(f"AXI simulation took {monotonic() - start:.2f} s.", flush=True)

        sim = compile_simulation(here, work, "projection_tb", ("projection_tile.sv", "projection_tb.sv"))
        for path in [None, *args.capture]:
            if path is None:
                description = "Signed extremes, partial tiles, stalls, reset and clear"
                stimulus = edge_cases()
            else:
                description, stimulus = capture_stimulus(path)
            with (work / "projection.txt").open("w") as stream:
                stream.writelines(stimulus)
                stream.write("6\n")
            start = monotonic()
            subprocess.run(["vvp", str(sim)], cwd=work, check=True)
            print(f"Matched {description}. Simulation took {monotonic() - start:.2f} s.", flush=True)


if __name__ == "__main__":
    main()
