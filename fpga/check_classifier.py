"""Replay reference vectors on the RTL simulator or the PYNQ classifier."""

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import tempfile
from time import monotonic, sleep

INPUT = 0x8000
SCORES = 0x8040
STATUS = 0x8840
COMMAND = 0x8844
MASK = 0xffffffff


@contextmanager
def board_lock():
    import fcntl

    with open("/run/lock/pinkprocess-classifier.lock", "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another pinkProcess process is using the classifier.") from exc
        yield


def writes(address, values):
    data = bytes(value & 255 for value in values)
    for offset in range(0, len(data), 4):
        yield 0, address + offset, int.from_bytes(data[offset:offset + 4], "little")


def evaluate(activation, expected):
    yield from writes(INPUT, activation)
    yield 0, COMMAND, 1
    yield 1, STATUS, 2
    for row, value in enumerate(expected):
        yield 2, SCORES + row * 4, value & MASK


def operations(reference):
    weights, vectors = reference["weights"], reference["vectors"]
    if len(weights) != 512 or any(len(row) != 64 for row in weights):
        raise ValueError("Expected a 512 x 64 classifier.")
    if not vectors or any(len(v["input"]) != 64 or len(v["scores"]) != 512 for v in vectors):
        raise ValueError("Expected 64 inputs and 512 scores per vector.")
    for row, values in enumerate(weights):
        yield from writes(row * 64, values)
    for vector in vectors:
        yield from evaluate(vector["input"], vector["scores"])

    # Exercise a live row replacement and restoration with near-maximum signed sums.
    activation = [127, -127] * 32
    expected = [sum(w * x for w, x in zip(row, activation)) for row in weights]
    target = 326
    original = expected[target]
    for sign in (1, -1):
        replacement = [sign * value for value in activation]
        expected[target] = sign * 64 * 127 * 127
        yield from writes(target * 64, replacement)
        yield from evaluate(activation, expected)
    expected[target] = original
    yield from writes(target * 64, weights[target])
    yield from evaluate(activation, expected)


def simulate(stimulus):
    here = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory() as temporary:
        work = Path(temporary)
        (work / "stimulus.txt").write_text(
            "".join(f"{op} {address:04x} {value:08x}\n" for op, address, value in stimulus)
        )
        subprocess.run(["iverilog", "-g2012", "-s", "classifier_tb", "-o", str(work / "sim"),
                        str(here / "classifier.v"), str(here / "classifier_tb.sv")], check=True)
        subprocess.run(["vvp", str(work / "sim")], cwd=work, check=True)


def check_board(bitstream, stimulus):
    from pynq import Overlay

    overlay = Overlay(str(bitstream))
    mmio = overlay.classifier.mmio
    for op, address, value in stimulus:
        if op == 0:
            mmio.write(address, value)
        elif op == 1:
            deadline = monotonic() + 1
            while mmio.read(address) != value:
                if monotonic() >= deadline:
                    raise TimeoutError("Classifier did not report done.")
                sleep(0.001)
        else:
            actual = mmio.read(address)
            if actual != value:
                raise RuntimeError(f"Score at {address:#06x}: got {actual:#010x}, expected {value:#010x}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("vectors", type=Path)
    parser.add_argument("--bitstream", type=Path, help="Run on PYNQ using this .bit and matching .hwh")
    args = parser.parse_args()
    reference = json.loads(args.vectors.read_text())
    stimulus = operations(reference)
    if args.bitstream is None:
        simulate(stimulus)
    else:
        with board_lock():
            check_board(args.bitstream, stimulus)
    print(f"Matched {(len(reference['vectors']) + 3) * 512:,} scores, including row replacement and restoration.")


if __name__ == "__main__":
    main()
