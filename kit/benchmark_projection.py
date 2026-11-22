"""Time one W8A8 projection on the current PYNQ overlay and the FLAN host path."""

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import median
from time import perf_counter_ns

import numpy as np

from fpga.check_classifier import board_lock
from lm_reference.fabric import ROWS, WIDTH, TiledMatvec


class Timings:
    def __init__(self):
        self.elapsed = Counter()
        self.calls = Counter()

    def measure(self, phase, function, *args):
        start = perf_counter_ns()
        try:
            return function(*args)
        finally:
            self.elapsed[phase] += perf_counter_ns() - start
            self.calls[phase] += 1


class TimedClassifier:
    def __init__(self, classifier, timings):
        self.classifier = classifier
        self.timings = timings

    @property
    def weight_revision(self):
        return self.classifier.weight_revision

    def write(self, address, values):
        self.timings.measure("weight_write", self.classifier.write, address, values)

    def score_rows_into(self, activation, first_row, output):
        self.timings.measure("score_roundtrip", self.classifier.score_rows_into, activation, first_row, output)


class TimedMatvec(TiledMatvec):
    def _gather_array_tile(self, np, chunk):
        return self.classifier.timings.measure("tile_gather", super()._gather_array_tile, np, chunk)


class SoftwareClassifier:
    """Model the signed INT8 tile contract for offline verification of the runner."""

    def __init__(self):
        self.weights = np.zeros((ROWS, WIDTH), dtype=np.int8)
        self.weight_revision = 0

    def write(self, address, values):
        first_row = address // WIDTH
        rows = np.frombuffer(values, dtype=np.int8).reshape(-1, WIDTH)
        self.weights[first_row:first_row + len(rows)] = rows
        self.weight_revision += 1

    def score_rows_into(self, activation, first_row, output):
        vector = np.frombuffer(activation, dtype=np.int8).astype(np.int32)
        output[:] = self.weights[first_row:first_row + len(output)].astype(np.int32) @ vector


def load_capture(path):
    with np.load(path, allow_pickle=False) as capture:
        weights = np.asarray(capture["weights"], order="C")
        activations = np.asarray(capture["activations"], order="C")
    if (weights.ndim != 2 or activations.ndim != 2 or weights.dtype != np.int8
            or activations.dtype != np.int8 or weights.shape[1] != activations.shape[1]
            or not all((*weights.shape, *activations.shape))):
        raise ValueError("Capture needs nonempty INT8 weights [rows, columns] and activations [batch, columns].")
    return weights, activations


def benchmark(classifier, weights, activations, repeats):
    rows, columns = weights.shape
    # INT32 suffices for usual FLAN shapes. Larger inputs get a separate int64 oracle.
    oracle_dtype = np.int32 if columns * 128**2 < (1 << 31) else np.int64
    expected = activations.astype(oracle_dtype) @ weights.astype(oracle_dtype).T
    samples = []
    host_matches = True
    for _ in range(repeats):
        start = perf_counter_ns()
        host = activations.astype(np.float32) @ weights.astype(np.float32).T
        host_ns = perf_counter_ns() - start

        # Start each repetition with a cold tile cache, while retaining normal intra-call reuse.
        timings = Timings()
        fabric = TimedMatvec(TimedClassifier(classifier, timings))
        start = perf_counter_ns()
        actual = fabric.matmat_array(weights, activations, rows)
        projection_ns = perf_counter_ns() - start
        if not np.array_equal(actual, expected):
            raise RuntimeError("Tiled projection disagrees with the integer reference.")
        host_matches = host_matches and np.array_equal(host, expected)

        samples.append({
            "host_global_ns": host_ns,
            "projection_ns": projection_ns,
            "tile_gather_ns": timings.elapsed["tile_gather"],
            "weight_write_ns": timings.elapsed["weight_write"],
            "score_roundtrip_ns": timings.elapsed["score_roundtrip"],
            "other_projection_ns": projection_ns - sum(timings.elapsed.values()),
            "tiles_gathered": timings.calls["tile_gather"],
            "weight_writes": timings.calls["weight_write"],
            "score_roundtrips": timings.calls["score_roundtrip"],
        })
    return {
        "shape": {"rows": rows, "columns": columns, "batch": len(activations)},
        "reserved_rows": fabric.reserved_rows,
        "integer_reference_matched": True,
        "host_float32_matched_integer_reference": bool(host_matches),
        "median": {key: median(sample[key] for sample in samples) for key in samples[0]},
        "samples": samples,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True,
                        help="NPZ containing signed INT8 weights [rows, columns] and activations [batch, columns].")
    parser.add_argument("--repeats", type=int, default=3)
    backend = parser.add_mutually_exclusive_group(required=True)
    backend.add_argument("--bitstream", type=Path,
                         help="Use the existing overlay and acquire the board lock before programming it.")
    backend.add_argument("--simulate", action="store_true",
                         help="Verify offline with a software tile model. Its timing is not FPGA timing.")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    weights, activations = load_capture(args.capture)
    if args.simulate:
        report = benchmark(SoftwareClassifier(), weights, activations, args.repeats)
    else:
        from lm_reference.live_demo import PynqClassifier

        with board_lock():
            report = benchmark(PynqClassifier(args.bitstream), weights, activations, args.repeats)
    report["backend"] = "software_tile_model" if args.simulate else "pynq_classifier"
    report["capture"] = str(args.capture)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
